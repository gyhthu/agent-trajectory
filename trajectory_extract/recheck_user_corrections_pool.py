#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recheck the 258 user-correction candidate pool against v-final.

The pool mixes several sources: some rows have B/C ids from run3, while
redflag rows may only have a seed corrector id and a short "what".  This script
uses the seed only to select a local transcript window, then asks the model to
identify the real t0/B/C under the signed v-final definition or reject it.
Results are cached as JSONL so the run can be resumed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import build_gold_review_sheet as review  # noqa: E402
import runtime_lane as rl  # noqa: E402
import task_stitch as ts  # noqa: E402
from turn_intent import _client, _LLM_MODEL  # noqa: E402


DATA = "/opt/shared/data/task-trajectory"
POOL = f"{DATA}/user_corrections_pool.jsonl"
OUT = f"{DATA}/user_corrections_pool_vfinal_recheck.jsonl"
REVIEW_OUT = f"{DATA}/gold_review_sheet_vfinal_recheck_20260709.md"


PROMPT = """你是纠错样本 v-final 重核员。下面是一段围绕候选 seed 的真实群聊，对话按时间编号。

请判断这段里是否存在一个合格三消息纠错样本：
- t0 = 真人原始指令/问题 A，必须早于 B；
- B = bot 后来被推翻的一句具体错误陈述/做法，必须是 bot 实质消息，不能是回执/系统提示；
- C = 真人触发纠正的消息，必须晚于 B；只能是用户点破、给相反事实/执行反证，或真人质疑触发同线返工；
- C 必须确实推翻 B；新问题、纯追问、照做、确认、需求变更、方向调整、寒暄都不算；
- 同一 t0 多个 B/C 合并成一条，选最核心的一组。

重要：seed 只是候选中心，可能是错的。你可以从窗口里选择真正的 t0/B/C；若窗口不足以确认，就 reject。

只输出一个 JSON，不要 markdown：
{{"verdict":"ACCEPT|REJECT","t0_msg_id":"om_xxx或空","bot_error_msg_id":"om_xxx或空","corrector_msg_id":"om_xxx或空","bot_error_quote":"逐字摘录B里被推翻的短句，reject为空","what":"一句话说明错在哪，reject也要写原因","kind":"negate|counter|user_trigger|newreq|confirm|unrelated|insufficient|other"}}

【候选 seed】
corrector_msg_id={seed_cid}
known_anchor_msg_id={seed_bid}
repair_msg_id={repair_id}
sources={sources}
whats={whats}

【对话窗口】
{transcript}
输出 JSON："""


def _load_jsonl(path: str) -> list[dict]:
    rows = []
    p = Path(path)
    if not p.exists():
        return rows
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _json_from_text(text: str) -> dict:
    start = text.index("{")
    end = text.rindex("}") + 1
    return json.loads(text[start:end])


def _raw_maps():
    raw_rows = review._load_jsonl(review.GUARDED) + review._load_jsonl(review.RAW)
    raw_by_c, _t0_by_anchor = review._build_raw_and_t0_maps(raw_rows, review._load_jsonl(review.SNAP))
    return raw_by_c


SELF_CORRECTION = re.compile(
    r"(你说得对|你问到点|你戳|你骂得|你提醒|我上一条|我刚才|我之前|"
    r"我修的方向偏|理解偏|说错|搞错|错了|确实|收回|纠正|误标|漏了|"
    r"认账|掀翻|方向正好反|不成立|逻辑上不成立|不是[^，。；\n]{0,18}而是)"
)
SUMMARY_CONFIRM = re.compile(r"(我梳理一下|我理解|可以理解为|我认这个理由|你先做一版|开始做吧)")
STRONG_CORRECTION = re.compile(
    r"(不对|不是这个|不是这样|错了|搞反|理解偏|你说错|你搞错|应该[^，。；\n]{0,24}不是|"
    r"不是[^，。；\n]{0,24}吗|它不是|本来[^，。；\n]{0,24}吧|应该有|实际上不|"
    r"实际不是|实际并非|实际没有|实际显示|实测|反证|推翻|很奇怪|奇怪啊|搞错|没[^，。；\n]{0,18}过)"
)
DIRECT_COUNTER = re.compile(
    r"(不对|不是这个|不是这样|错了|搞反|理解偏|你说错|你搞错|"
    r"实际上不|实际不是|实际并非|实际没有|实测|反证|推翻|很奇怪|奇怪啊|搞错|没[^，。；\n]{0,18}过)"
)
QUESTION_CHALLENGE = re.compile(r"(吗|么|？|\?|为什么|怎么会|确定|不是[^，。；\n]{0,24}吗|应该有)")


def _is_human_event(event: dict | None) -> bool:
    return bool(event) and review._role_of(event) == "user"


def _quote_norm(text: str) -> str:
    text = ts._strip_feishu(text or "")
    text = re.sub(r"[*_`~]+", "", text)
    return re.sub(r"\s+", "", text)


def _parent_chain_ids(msg_id: str, by_id: dict, include_self: bool = False, max_depth: int = 16) -> list[str]:
    out = [msg_id] if include_self and msg_id in by_id else []
    seen = set(out)
    cur = msg_id
    for _ in range(max_depth):
        event = by_id.get(cur)
        if not event:
            break
        parent = event.get("parent_id")
        if not parent or parent in seen or parent not in by_id:
            break
        out.append(parent)
        seen.add(parent)
        cur = parent
    return out


def _same_reply_line(left_msg_id: str, right_msg_id: str, by_id: dict) -> bool:
    if not left_msg_id or not right_msg_id or left_msg_id not in by_id or right_msg_id not in by_id:
        return False
    return rl._root_of(left_msg_id, by_id) == rl._root_of(right_msg_id, by_id)


def _looks_like_correction_trigger(event: dict | None) -> bool:
    if not event:
        return False
    text = event.get("text") or ""
    return bool(STRONG_CORRECTION.search(text) or QUESTION_CHALLENGE.search(text))


def _looks_like_original_instruction(event: dict | None) -> bool:
    if not _is_human_event(event):
        return False
    text = event.get("text") or ""
    if _looks_like_correction_trigger(event):
        return False
    if SUMMARY_CONFIRM.search(text):
        return False
    return bool(ts._strip_feishu(text).strip())


def _resolve_seed_focus(seed_cid: str, seed_bid: str, by_id: dict, idx: dict, evs: list[dict]) -> dict:
    """If seed is a bot repair/self-correction, move C focus to the human trigger.

    Redflag rows sometimes store the bot's "you are right / I was wrong" repair
    as the seed.  Under v-final that repair is evidence, not C.  C must be the
    human message between the original bot error and the repair, preferably the
    one that replied to the erroneous bot message.
    """
    focus = {
        "seed_corrector_msg_id": seed_cid,
        "focus_corrector_msg_id": seed_cid,
        "seed_anchor_msg_id": seed_bid,
        "repair_msg_id": "",
    }
    seed = by_id.get(seed_cid)
    if not seed or review._role_of(seed) != "bot" or not SELF_CORRECTION.search(seed.get("text") or ""):
        return focus

    ci = idx.get(seed_cid)
    if ci is None:
        return focus
    ai = idx.get(seed_bid) if seed_bid else None
    start = (ai + 1) if ai is not None and ai < ci else max(0, ci - 12)
    window = evs[start:ci]
    humans = [event for event in window if _is_human_event(event)]
    if not humans:
        return focus

    strong = [event for event in humans if STRONG_CORRECTION.search(event.get("text") or "")]
    reply_to_anchor = [event for event in humans if seed_bid and event.get("parent_id") == seed_bid]
    pick = (strong[-1] if strong else (reply_to_anchor[-1] if reply_to_anchor else humans[-1]))
    if SUMMARY_CONFIRM.search(pick.get("text") or "") and not STRONG_CORRECTION.search(pick.get("text") or ""):
        return focus

    focus["focus_corrector_msg_id"] = pick["msg_id"]
    focus["repair_msg_id"] = seed_cid
    return focus


def _render_llm_transcript(evs: list[dict], idx: dict, seed_bid: str | None, seed_cid: str, pad_before: int, pad_after: int, cap: int) -> str:
    records, _shown = review._transcript_records(
        evs, idx, None, seed_bid, seed_cid,
        pad_before=pad_before, pad_after=pad_after, cap=cap,
    )
    lines = []
    for rec in records:
        if rec["kind"] == "gap":
            lines.append(f"... 省略 {rec['n_omitted']} 条 ...")
            continue
        if rec["kind"] == "truncated":
            lines.append(f"... 窗口超过 {rec['cap']} 条，已截断 ...")
            continue
        role = rec["role"]
        parent = f" ↩{rec['parent_label']}" if rec["parent_label"] else ""
        marks = " ".join(rec["marks"])
        mark = f" {marks}" if marks else ""
        lines.append(
            f"{rec['label']} [{rec['msg_id']}] {role}/{rec['name']}{mark}{parent}: {rec['text']}"
        )
    return "\n".join(lines)


def _validate_pick(result: dict, by_id: dict, idx: dict) -> tuple[bool, str]:
    if str(result.get("verdict", "")).upper() != "ACCEPT":
        return False, "model_reject"
    t0id = result.get("t0_msg_id") or ""
    bid = result.get("bot_error_msg_id") or ""
    cid = result.get("corrector_msg_id") or ""
    missing = [name for name, mid in (("t0", t0id), ("B", bid), ("C", cid)) if mid not in by_id]
    if missing:
        return False, "missing_" + ",".join(missing)
    roles = {
        "t0": review._role_of(by_id[t0id]),
        "B": review._role_of(by_id[bid]),
        "C": review._role_of(by_id[cid]),
    }
    if roles["t0"] != "user":
        return False, f"t0_is_{roles['t0']}"
    if roles["B"] != "bot":
        return False, f"B_is_{roles['B']}"
    if roles["C"] != "user":
        return False, f"C_is_{roles['C']}"
    if not (idx[t0id] < idx[bid] < idx[cid]):
        return False, "bad_order"
    quote = ts._strip_feishu(result.get("bot_error_quote") or "").strip()
    if len(re.sub(r"\s+", "", quote)) < 4:
        return False, "quote_too_short"
    btxt = by_id[bid].get("text") or ""
    if _quote_norm(quote) not in _quote_norm(btxt):
        return False, "quote_not_in_B"
    c_event = by_id[cid]
    kind = (result.get("kind") or "").strip()
    if c_event.get("parent_id") and kind == "user_trigger" and not DIRECT_COUNTER.search(c_event.get("text") or ""):
        c_chain = set(_parent_chain_ids(cid, by_id))
        if bid not in c_chain and not _same_reply_line(cid, bid, by_id):
            return False, "C_not_same_line_with_B"
    return True, ""


def _repair_t0_pick(result: dict, by_id: dict, idx: dict, evs: list[dict]) -> dict:
    if str(result.get("verdict", "")).upper() != "ACCEPT":
        return result
    t0id = result.get("t0_msg_id") or ""
    bid = result.get("bot_error_msg_id") or ""
    if t0id not in by_id or bid not in idx:
        return result
    t0_event = by_id[t0id]
    t0_role = review._role_of(t0_event)
    suspicious = t0_role != "user" or (
        t0_role == "user"
        and _looks_like_correction_trigger(t0_event)
        and bool(t0_event.get("parent_id"))
    )
    if not suspicious:
        return result
    bi = idx[bid]
    for pos in range(bi - 1, -1, -1):
        event = evs[pos]
        if _looks_like_original_instruction(event):
            repaired = dict(result)
            repaired["t0_msg_id"] = event["msg_id"]
            repaired["t0_repaired_from"] = t0id
            return repaired
    return result


def _repair_sys_t0_pick(result: dict, by_id: dict, idx: dict, evs: list[dict]) -> dict:
    return _repair_t0_pick(result, by_id, idx, evs)


def _repair_parent_bot_error_pick(result: dict, by_id: dict, idx: dict) -> dict:
    """Prefer the bot message that C directly/indirectly replies to when usable."""
    if str(result.get("verdict", "")).upper() != "ACCEPT":
        return result
    cid = result.get("corrector_msg_id") or ""
    bid = result.get("bot_error_msg_id") or ""
    t0id = result.get("t0_msg_id") or ""
    if cid not in by_id or bid not in by_id or t0id not in by_id:
        return result
    quote = ts._strip_feishu(result.get("bot_error_quote") or "").strip()
    if not quote:
        return result
    chain = _parent_chain_ids(cid, by_id)
    if bid in chain:
        return result
    for parent_id in chain:
        parent = by_id[parent_id]
        if review._role_of(parent) != "bot":
            continue
        if parent_id not in idx or not (idx[t0id] < idx[parent_id] < idx[cid]):
            continue
        if _quote_norm(quote) not in _quote_norm(parent.get("text") or ""):
            continue
        repaired = dict(result)
        repaired["bot_error_msg_id"] = parent_id
        repaired["bot_error_repaired_from"] = bid
        return repaired
    return result


def _lock_corrector_pick(result: dict, focus_cid: str) -> dict:
    """Keep C deterministic: the model may judge the candidate, not choose a new C."""
    if str(result.get("verdict", "")).upper() != "ACCEPT" or not focus_cid:
        return result
    picked_cid = result.get("corrector_msg_id") or ""
    if picked_cid == focus_cid:
        return result
    repaired = dict(result)
    repaired["corrector_msg_id"] = focus_cid
    repaired["corrector_repaired_from"] = picked_cid
    return repaired


def _repair_known_bot_error_pick(result: dict, seed_raw: dict, by_id: dict, idx: dict) -> dict:
    """Use the already guarded session/census anchor when the LLM drifts to another B."""
    if str(result.get("verdict", "")).upper() != "ACCEPT":
        return result
    seed_bid = seed_raw.get("anchor_msg_id") or ""
    seed_quote = ts._strip_feishu(seed_raw.get("bot_error_quote") or "").strip()
    cid = result.get("corrector_msg_id") or ""
    if not seed_bid or seed_bid not in by_id or cid not in idx or seed_bid not in idx:
        return result
    if review._role_of(by_id[seed_bid]) != "bot" or not (idx[seed_bid] < idx[cid]):
        return result
    if not seed_quote or _quote_norm(seed_quote) not in _quote_norm(by_id[seed_bid].get("text") or ""):
        return result
    repaired = dict(result)
    if repaired.get("bot_error_msg_id") != seed_bid:
        repaired["bot_error_repaired_from"] = repaired.get("bot_error_msg_id") or ""
    repaired["bot_error_msg_id"] = seed_bid
    repaired["bot_error_quote"] = seed_quote
    return repaired


def _recheck_one(row: dict, evs: list[dict], by_id: dict, idx: dict, raw_by_c: dict, args) -> dict:
    seed_cid = row.get("corrector_msg_id") or ""
    seed_raw = raw_by_c.get(seed_cid) or row
    seed_bid = seed_raw.get("anchor_msg_id") or ""
    if seed_cid not in by_id:
        return {**row, "verdict": "REJECT", "kind": "missing_seed", "what": "seed corrector 不在事件流"}
    focus = _resolve_seed_focus(seed_cid, seed_bid, by_id, idx, evs)
    focus_cid = focus["focus_corrector_msg_id"]
    transcript = _render_llm_transcript(evs, idx, seed_bid, focus_cid, args.pad_before, args.pad_after, args.cap)
    prompt = PROMPT.format(
        seed_cid=focus_cid,
        seed_bid=seed_bid,
        repair_id=focus["repair_msg_id"],
        sources="+".join(row.get("sources") or []),
        whats="；".join(row.get("whats") or []),
        transcript=transcript,
    )
    try:
        resp = _client().chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            timeout=args.timeout,
        )
        parsed = _json_from_text(resp.choices[0].message.content or "")
    except Exception as ex:
        return {**row, "verdict": "ERROR", "kind": "llm_error", "what": str(ex)[:180]}
    parsed = _lock_corrector_pick(parsed, focus_cid)
    parsed = _repair_parent_bot_error_pick(parsed, by_id, idx)
    parsed = _repair_known_bot_error_pick(parsed, seed_raw, by_id, idx)
    parsed = _repair_t0_pick(parsed, by_id, idx, evs)
    ok, reason = _validate_pick(parsed, by_id, idx)
    out = {
        "seed_corrector_msg_id": seed_cid,
        "focus_corrector_msg_id": focus_cid,
        "repair_msg_id": focus["repair_msg_id"],
        "sources": row.get("sources") or [],
        "seed_whats": row.get("whats") or [],
        **parsed,
        "valid": ok,
        "invalid_reason": reason,
    }
    return out


def _append_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _accepted_cards(rows: list[dict]) -> list[dict]:
    cards = []
    for row in rows:
        if not row.get("valid"):
            continue
        cid = row.get("corrector_msg_id")
        cards.append({
            "cid": cid,
            "srcs": ["vfinal_recheck"] + list(row.get("sources") or []),
            "whats": [row.get("what") or ""],
            "corrector": "",
            "raw": {
                "anchor_msg_id": row.get("bot_error_msg_id"),
                "bot_error_quote": row.get("bot_error_quote") or "",
                "what": row.get("what") or "",
            },
            "t0": {"msg_id": row.get("t0_msg_id")},
            "score": 3,
        })
    return cards


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", default=POOL)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--review-out", default=REVIEW_OUT)
    ap.add_argument("--limit", type=int, default=0, help="0 means all pending rows")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--model", default=_LLM_MODEL)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--pad-before", type=int, default=10)
    ap.add_argument("--pad-after", type=int, default=8)
    ap.add_argument("--cap", type=int, default=90)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    evs = rl.load_events()
    by_id = {e.get("msg_id"): e for e in evs}
    idx = {e.get("msg_id"): i for i, e in enumerate(evs)}
    raw_by_c = _raw_maps()

    pool = _load_jsonl(args.pool)
    done = {r.get("seed_corrector_msg_id") for r in _load_jsonl(args.out)}
    pending = [r for r in pool if r.get("corrector_msg_id") not in done]
    if args.limit:
        pending = pending[:args.limit]
    print(f"[pool] total={len(pool)} done={len(done)} pending_run={len(pending)} model={args.model}", flush=True)

    new_rows: list[dict] = []
    if pending:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_recheck_one, row, evs, by_id, idx, raw_by_c, args) for row in pending]
            for n, fut in enumerate(as_completed(futs), 1):
                row = fut.result()
                new_rows.append(row)
                if n % 10 == 0 or n == len(futs):
                    acc = sum(1 for r in new_rows if r.get("valid"))
                    err = sum(1 for r in new_rows if r.get("verdict") == "ERROR")
                    print(f"[recheck] {n}/{len(futs)} accepted={acc} errors={err}", flush=True)
        _append_jsonl(args.out, new_rows)

    rows = _load_jsonl(args.out)
    valid = [r for r in rows if r.get("valid")]
    print(f"[summary] rows={len(rows)} valid={len(valid)} -> {args.out}", flush=True)
    if valid:
        cards = _accepted_cards(valid)
        md = review.render(
            cards, evs, by_id, idx,
            title="金标人工核对台 · v-final 重核候选",
            pad_before=6, pad_after=6, cap=90,
        )
        Path(args.review_out).write_text(md, encoding="utf-8")
        print(f"[review] {len(cards)} cards -> {args.review_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
