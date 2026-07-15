#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""金标人工核对台生成器（张耀明 2026-07-08 定义 v-final）。

把候选纠错的三条原文（t0 用户指令 / B bot错误 / C 用户纠正）+ 前后上下文拉成
逐卡片的 markdown 核对台，供人照 7 条判据逐卡打勾。可复用：--n 控制卡片数、
--sources 选来源分层，试点小批验格式后直接扩到全金标。

数据源（全部现成，DRY）：
  · runtime_lane.load_events()            → 全量事件 msg_id→原文/name/ts/序位
  · session_corrections.jsonl             → 护栏后 run3(anchor=B, corrector=C, quote, what, _session)
  · session_corrections_raw.jsonl         → 旧产物兜底；只在护栏产物缺明细时读取
  · pre_instruction_snapshots.jsonl       → t0(用户指令)，优先键 corrections[].anchor==anchor_msg_id(B)，兼容旧的 C 键
  · user_corrections_pool.jsonl           → 258 候选池(键 corrector_msg_id, sources 分层)
"""
import os, sys, json, argparse, re
from pathlib import Path
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import runtime_lane as rl
import task_stitch as ts
from correction_census import _is_bot_row, PLACEHOLDER

DATA = "/opt/shared/data/task-trajectory"
POOL = f"{DATA}/user_corrections_pool.jsonl"
GUARDED = f"{DATA}/session_corrections.jsonl"
RAW = f"{DATA}/session_corrections_raw.jsonl"
SNAP = f"{DATA}/pre_instruction_snapshots.jsonl"

REVIEW_LINE_RE = re.compile(r"^- \**(L\d+)\** 〔([^·]+)·([^〕]+)〕.*$")
CARD_RE = re.compile(r"^## #(\d+)\b.*$", re.M)


def _load_jsonl(p):
    out = []
    for l in open(p, encoding="utf-8"):
        l = l.strip()
        if not l:
            continue
        try:
            out.append(json.loads(l))
        except Exception:
            continue
    return out


def _build_raw_and_t0_maps(raw_rows, snapshot_rows):
    """Return raw-by-corrector and t0-by-anchor maps used by review cards."""
    raw_by_c = {}
    for r in raw_rows:
        cid = r.get("corrector_msg_id")
        if cid and cid not in raw_by_c:
            raw_by_c[cid] = r

    t0_by_anchor = {}
    for s in snapshot_rows:
        t0 = s.get("t0") or {}
        for corr in s.get("corrections", []):
            anchor = corr.get("anchor")
            if anchor and anchor not in t0_by_anchor:
                t0_by_anchor[anchor] = t0
    return raw_by_c, t0_by_anchor


def _flat(s, n):
    """压平成单行 + 中和内部 markdown，防止 B/C 原文里的 ##/表格/** 冲垮卡片结构。"""
    s = ts._strip_feishu(s or "")
    s = s.replace("\r", "")
    s = re.sub(r"\s*\n\s*", " ⏎ ", s)          # 换行折成 ⏎，卡片不再被多行撑开
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = s.replace("**", "").replace("~~", "")   # 去加粗/删除线
    s = s.replace("#", "＃").replace("|", "｜").replace("`", "'")  # 标题/表格/代码符全角化，不再被渲染
    s = re.sub(r"｜(?:[\s\-｜]*｜)+", " 〔表格略〕 ", s)  # 折叠 ｜--｜--｜ 表格残骸，别喷成一坨
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = s.strip()
    return s if len(s) <= n else s[:n] + f"…〔+{len(s)-n}字〕"


def _text_fingerprint(s, min_chars=20):
    """Stable coarse text fingerprint for old review/context de-duplication."""
    s = ts._strip_feishu(s or "")
    s = re.sub(r"…〔\+\d+字〕$", "", s)
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[`*_#｜|\-—~，。！？；：、,.!?;:()\[\]（）【】《》\"'“”‘’]+", "", s)
    if len(s) < min_chars:
        return ""
    return s[:180]


def _reviewed_fingerprints_empty():
    return {"msg_ids": set(), "roots": set(), "texts": set()}


def _add_event_fingerprint(out, event, by_id):
    if not event:
        return
    mid = event.get("msg_id")
    if mid:
        out["msg_ids"].add(mid)
        if mid in by_id:
            out["roots"].add(rl._root_of(mid, by_id))
    fp = _text_fingerprint(event.get("text"))
    if fp:
        out["texts"].add(fp)


def _line_text_prefix(line):
    body = line.split("　")[-1].strip()
    body = re.sub(r"…〔\+\d+字〕$", "", body).strip()
    return body


def _build_review_event_index(evs):
    index = {}
    for event in evs:
        name = event.get("name") or ""
        flat = _flat(event.get("text"), 10000)
        fp = _text_fingerprint(flat, min_chars=8)
        if not fp:
            continue
        rec = {"event": event, "flat": flat}
        for n in (24, 16, 8):
            index.setdefault((name, fp[:n]), []).append(rec)
    return index


def _resolve_review_line(line, event_index):
    m = REVIEW_LINE_RE.match(line)
    if not m:
        return None
    label, _role_tag, name = m.groups()
    prefix = _line_text_prefix(line)
    fp = _text_fingerprint(prefix, min_chars=8)
    if not fp or prefix.startswith("⋯"):
        return None
    candidates = []
    for n in (24, 16, 8):
        candidates = event_index.get((name, fp[:n]), [])
        if candidates:
            break
    matches = [rec["event"] for rec in candidates if rec["flat"].startswith(prefix) or prefix in rec["flat"]]
    return {"label": label, "prefix": prefix, "matches": matches}


def _review_sheet_fingerprints(path, evs, by_id):
    """Resolve rendered old review-sheet rows back to events and collect fingerprints."""
    text = Path(path).read_text(encoding="utf-8")
    starts = [(int(m.group(1)), m.start(), m.end()) for m in CARD_RE.finditer(text)]
    out = _reviewed_fingerprints_empty()
    idx = {e.get("msg_id"): i for i, e in enumerate(evs)}
    event_by_id = {e.get("msg_id"): e for e in evs}
    event_index = _build_review_event_index(evs)

    for i, (_card_no, _start, body_start) in enumerate(starts):
        body_end = starts[i + 1][1] if i + 1 < len(starts) else len(text)
        body = text[body_start:body_end]
        resolved = []
        unresolved = []
        for line in body.splitlines():
            rec = _resolve_review_line(line, event_index)
            if not rec:
                continue
            if len(rec["matches"]) == 1:
                resolved.append(rec["matches"][0])
            elif rec["matches"]:
                unresolved.append(rec)

        if unresolved and resolved:
            positions = sorted(idx[e.get("msg_id")] for e in resolved if e.get("msg_id") in idx)
            if positions:
                lo, hi = positions[0], positions[-1]
                for rec in unresolved:
                    scored = []
                    for event in rec["matches"]:
                        pos = idx.get(event.get("msg_id"))
                        if pos is None:
                            continue
                        dist = 0 if lo <= pos <= hi else min(abs(pos - lo), abs(pos - hi))
                        scored.append((dist, pos, event.get("msg_id")))
                    scored.sort()
                    if scored and (len(scored) == 1 or scored[0][0] < scored[1][0]):
                        resolved.append(event_by_id[scored[0][2]])

        for event in resolved:
            _add_event_fingerprint(out, event, by_id)
    return out


def _jsonl_gold_fingerprints(path, by_id):
    out = _reviewed_fingerprints_empty()
    for row in _load_jsonl(path):
        for mid in (row.get("gold") or {}).values():
            _add_event_fingerprint(out, by_id.get(mid), by_id)
        for mid in (row.get("machine") or {}).values():
            if isinstance(mid, str) and mid.startswith("om_"):
                _add_event_fingerprint(out, by_id.get(mid), by_id)
    return out


def _merge_fingerprints(dst, src):
    for key in dst:
        dst[key].update(src[key])


def _default_reviewed_paths(exclude_paths=()):
    exclude = {str(Path(p).resolve()) for p in exclude_paths if p}
    names = [
        "gold_review_sheet_pilot.md",
        "gold_review_sheet_new_batch_20260708.md",
        "gold_review_sheet_new_batch_20260708_expanded.md",
        "gold_review_sheet_new_batch_20260708_guarded_t0_probe.md",
        "gold_review_sheet_new_batch_20260708_parent_chain_probe.md",
        "gold_review_sheet_conservative_batch_20260708.md",
        "gold_review_sheet_next8_20260709.md",
        "gold_review_sheet_clean8_20260709.md",
        "gold_review_adjudications_20260708.jsonl",
        "gold_review_adjudications_20260708_expanded.jsonl",
        "gold_review_adjudications_conservative_batch_20260708.jsonl",
        "gold_review_adjudications_20260708_with_conservative_batch.jsonl",
    ]
    paths = []
    for name in names:
        p = Path(DATA) / name
        if p.exists() and str(p.resolve()) not in exclude:
            paths.append(str(p))
    return paths


def _load_reviewed_fingerprints(evs, by_id, reviewed_paths):
    out = _reviewed_fingerprints_empty()
    for path in reviewed_paths:
        if path.endswith(".jsonl"):
            _merge_fingerprints(out, _jsonl_gold_fingerprints(path, by_id))
        elif path.endswith(".md"):
            _merge_fingerprints(out, _review_sheet_fingerprints(path, evs, by_id))
    return out


def _eligible_for_human_review(card, by_id):
    """Return whether the machine guess is worth sending to manual t0/B/C review.

    The review UI can display wrong guesses, but after the pilot confirmed that
    bot/sys C anchors are a pipeline bug, routine sampling should not keep
    asking humans to reject known-invalid C rows.
    """
    raw = card["raw"] or {}
    cid = card["cid"]
    bid = raw.get("anchor_msg_id")
    c = by_id.get(cid)
    b = by_id.get(bid)
    if not c:
        return False, "missing_C"
    if _role_of(c) != "user":
        return False, f"C_is_{_role_of(c)}"
    if not b:
        return False, "missing_B"
    if _role_of(b) != "bot":
        return False, f"B_is_{_role_of(b)}"
    return True, ""


def _review_dedupe_key(card, by_id):
    """Coarse key for review sampling: avoid repeated cards from one reply chain."""
    raw = card["raw"] or {}
    cid = card["cid"]
    bid = raw.get("anchor_msg_id")
    c_root = rl._root_of(cid, by_id) if cid in by_id else cid
    b_root = rl._root_of(bid, by_id) if bid in by_id else bid
    return c_root, b_root


def _reviewed_overlap_reason(card, by_id, reviewed):
    raw = card["raw"] or {}
    t0 = card["t0"] or {}
    ids = [t0.get("msg_id"), raw.get("anchor_msg_id"), card["cid"]]
    for mid in ids:
        if mid and mid in reviewed["msg_ids"]:
            return f"seen_msg:{mid}"
        if mid and mid in by_id and rl._root_of(mid, by_id) in reviewed["roots"]:
            return f"seen_root:{rl._root_of(mid, by_id)}"
        fp = _text_fingerprint((by_id.get(mid) or {}).get("text"))
        if fp and fp in reviewed["texts"]:
            return "seen_text"
    return ""


def build(
    n,
    sources_filter,
    offset=0,
    eligible_only=True,
    dedupe_roots=True,
    exclude_reviewed=True,
    reviewed_paths=None,
):
    evs = rl.load_events()
    by_id = {e.get("msg_id"): e for e in evs}
    idx = {e.get("msg_id"): i for i, e in enumerate(evs)}
    reviewed = _reviewed_fingerprints_empty()
    if exclude_reviewed:
        reviewed = _load_reviewed_fingerprints(evs, by_id, reviewed_paths or _default_reviewed_paths())

    # run3：键 corrector_msg_id → 该条(含 anchor=B, quote, what, _session)。
    # 优先用护栏/核验后的 session_corrections.jsonl；旧产物没有明细字段时再由 raw 兜底。
    raw_rows = _load_jsonl(GUARDED) + _load_jsonl(RAW)
    # 快照主形态：corrections[].anchor 是被纠错 anchor(B)，历史产物也混有按 corrector(C)
    # 落键的行。核对台按 C 枚举候选，所以 t0 先通过 raw 找 B，查不到再按 C 兼容旧行。
    raw_by_c, t0_by_anchor = _build_raw_and_t0_maps(raw_rows, _load_jsonl(SNAP))

    pool = _load_jsonl(POOL)
    # 分层过滤 + 优先三料齐（B、C、t0 都拿得到）的候选，试点最干净
    cards = []
    for c in pool:
        cid = c.get("corrector_msg_id")
        srcs = c.get("sources", [])
        if sources_filter and not (set(srcs) & set(sources_filter)):
            continue
        raw = raw_by_c.get(cid)
        bid = raw.get("anchor_msg_id") if raw else None
        t0 = t0_by_anchor.get(bid) or t0_by_anchor.get(cid)
        has_B = bool(raw and by_id.get(raw.get("anchor_msg_id")))
        has_C = bool(by_id.get(cid))
        has_t0 = bool(t0 and by_id.get(t0.get("msg_id")))
        cards.append({
            "cid": cid, "srcs": srcs, "whats": c.get("whats", []),
            "corrector": c.get("corrector", ""), "raw": raw, "t0": t0,
            "score": has_B + has_C + has_t0,  # 三料齐=3，优先
        })
    cards.sort(key=lambda x: -x["score"])
    if eligible_only:
        for card in cards:
            ok, why = _eligible_for_human_review(card, by_id)
            card["eligible"] = ok
            card["drop_reason"] = why
        cards = [c for c in cards if c["eligible"]]
    if exclude_reviewed:
        for card in cards:
            why = _reviewed_overlap_reason(card, by_id, reviewed)
            card["reviewed_overlap"] = why
        cards = [c for c in cards if not c["reviewed_overlap"]]
    if dedupe_roots:
        seen = set()
        unique = []
        for card in cards:
            key = _review_dedupe_key(card, by_id)
            if key in seen:
                continue
            seen.add(key)
            unique.append(card)
        cards = unique
    return cards[offset:offset + n], by_id, idx


_ROLE_TAG = {"user": "真人", "bot": "bot", "sys": "sys忽略"}


def _role_of(e):
    """真人 / bot / sys(占位·回执·系统卡片)。sys 与 bot 都不能当纠正 C。"""
    txt = e.get("text") or ""
    if PLACEHOLDER.match(txt):
        return "sys"
    t = txt.lstrip()
    # 本地补充：bot 系统通知（入队/压缩/排队），只影响本核对台展示，不碰共享 PLACEHOLDER
    if re.match(r"^\s*(🕒|🧹|⏳|🗜️ ?已|已入队|已压缩)", txt):
        return "sys"
    if t.startswith('{"') and ("请升级至最新版本" in txt or '"image_key"' in txt[:160]):
        return "sys"  # 客户端升级提示 / 图片卡片 JSON，非实质发言
    if _is_bot_row({"role": e.get("role"), "who": e.get("name"), "name": e.get("name")}):
        return "bot"
    return "user"


def _parent_chain_indices(evs, idx, msg_id, max_depth=12):
    """Return parent-chain indices for msg_id, oldest reachable ancestors included.

    Gold review cards are meant to expose the real t0/B/C, not only the model's
    guessed span. A quoted reply often points to the missing earlier instruction,
    so we include reply-chain ancestors without dumping every intervening message.
    """
    out, seen = [], set()
    cur = msg_id
    for _ in range(max_depth):
        j = idx.get(cur)
        if j is None:
            break
        pid = evs[j].get("parent_id")
        if not pid or pid in seen or pid not in idx:
            break
        seen.add(pid)
        out.append(idx[pid])
        cur = pid
    return out


def _transcript_records(evs, idx, t0id, bid, cid, pad_before=3, pad_after=4, cap=56):
    """Return transcript records with stable L labels and msg_id metadata."""
    pts = sorted({idx[a] for a in (t0id, bid, cid) if a and a in idx})
    if not pts:
        return [], set()
    lo = max(0, pts[0] - pad_before)
    hi = min(len(evs) - 1, pts[-1] + pad_after)
    base = list(range(lo, hi + 1))
    include = set(base)
    # Include parent chains for the whole local window. This catches cases where
    # the real t0 is the quoted earlier instruction of a nearby "好了吗/继续" turn.
    for j in base:
        include.update(_parent_chain_indices(evs, idx, evs[j].get("msg_id")))
    ordered = sorted(include)
    truncated = len(ordered) > cap
    if truncated:
        keep = set(ordered[: max(0, cap // 2)])
        keep.update(ordered[-max(1, cap - len(keep)):])
        ordered = sorted(keep)
    label = {evs[j].get("msg_id"): f"L{n}" for n, j in enumerate(ordered, 1)}
    out = []
    prev = None
    for n, j in enumerate(ordered, 1):
        if prev is not None and j > prev + 1:
            out.append({"kind": "gap", "n_omitted": j - prev - 1})
        prev = j
        e = evs[j]
        mid = e.get("msg_id")
        role = _role_of(e)
        marks = []
        if mid and mid == t0id:
            marks.append("«猜t0?»")
        if mid and mid == bid:
            marks.append("«猜B?»")
        if mid and mid == cid:
            marks.append("«猜C?»")
        gm = (" " + " ".join(marks)) if marks else ""
        pid = e.get("parent_id")
        lab = f"L{n}"
        important = bool(marks) or role == "user"   # 真人行 / 三条猜测行给足字数，纯 bot 上下文压短
        out.append({
            "kind": "message",
            "label": lab,
            "msg_id": mid,
            "role": role,
            "name": e.get("name"),
            "marks": marks,
            "parent_label": label.get(pid, "(更早)") if pid else "",
            "text": _flat(e.get("text"), 140 if important else 60),
            "index": j,
            "guess_suffix": gm,
        })
    if truncated:
        out.append({"kind": "truncated", "cap": cap})
    shown = {evs[j].get("msg_id") for j in ordered}
    return out, shown


def _transcript(evs, idx, t0id, bid, cid, pad_before=3, pad_after=4, cap=56):
    """以管线猜测 t0/B/C 的索引跨度为轴取一段连续对话流，逐条编号 L1..Ln。
    每条标 作者类型(真人/bot/sys) + 回复指向(↩L?) + 管线猜测标记(«猜t0?/B?/C?»)。
    真人行加粗突出（只有真人行能被指认为纠正 C）。"""
    records, shown = _transcript_records(evs, idx, t0id, bid, cid, pad_before, pad_after, cap)
    out = []
    for r in records:
        if r["kind"] == "gap":
            out.append(f"- 　⋯（省略中间 {r['n_omitted']} 条非回复链消息）⋯")
            continue
        if r["kind"] == "truncated":
            out.append(f"- 　⋯（窗口超 {r['cap']} 条已截断，越界的猜测见卡尾反查）⋯")
            continue
        lab = r["label"]
        head = f"**{lab}**" if r["role"] == "user" else lab
        rp = ("　↩" + r["parent_label"]) if r["parent_label"] else ""
        out.append(
            f"- {head} 〔{_ROLE_TAG[r['role']]}·{r['name']}〕"
            f"{r['guess_suffix']}{rp}　{r['text']}"
        )
    return out, shown


def render(
    cards,
    evs,
    by_id,
    idx,
    title="金标人工核对台 · 试点批（对话流指认版）",
    pad_before=3,
    pad_after=4,
    cap=56,
):
    lines = [
        f"# {title}", "",
        "**怎么标**：每卡是一段编号对话流。只有〔真人〕行可能是纠正 C，〔bot〕〔sys忽略〕都不能当 C。",
        "`«猜t0?/猜B?/猜C?»` 是管线的猜测，**大概率锚错、别被带跑**——你按对话自己指认 L 行号。",
        "`↩L?` 是这条引用回复了哪一行（判据3 按回复链找纠正就看它）。", "",
        "每卡填卡尾：判定 + 真t0/真B/真C 各是第几行(L?)；真 C 只能选〔真人〕行，无合格真 C → 不合格。", "",
        "判据速记：①B是bot一句具体错话 ②B非回执/系统提示 ③纠正须真人触发(直接点破/贴反证/质疑触发返工且返工改的错与质疑同线) "
        "④确推翻B非新问题/追问 ⑤需求变更不算 ⑥单侧回溯补齐 ⑦同t0多纠合并。", "",
    ]
    for k, card in enumerate(cards, 1):
        cid = card["cid"]
        raw = card["raw"] or {}
        t0 = card["t0"] or {}
        bid = raw.get("anchor_msg_id")
        t0id = t0.get("msg_id")
        q = raw.get("bot_error_quote")
        what = _flat("; ".join(card["whats"]), 160)
        lines.append(f"## #{k}　来源:{'+'.join(card['srcs'])}")
        lines.append(f"- 机器猜(可能错)：" + (f"B?错句「{_flat(q, 90)}」；" if q else "") + f"what={what}")
        lines.append("")
        trans, shown = _transcript(
            evs, idx, t0id, bid, cid,
            pad_before=pad_before, pad_after=pad_after, cap=cap,
        )
        if trans:
            lines.extend(trans)
        else:
            lines.append("- ❓三条猜测都没落进事件流，需 msg_id 反查")
        lines.append("")
        # 只为「没出现在上面对话流里」的越界猜测留一行纯文本反查（无反引号，不再渲染成灰块）
        off = [(nm, mid) for nm, mid in (("t0", t0id), ("B", bid), ("C", cid)) if mid and mid not in shown]
        if off:
            lines.append("越界猜测反查：" + "　".join(f"{nm}猜 {mid}" for nm, mid in off))
            lines.append("")
        lines.append("**你填**：判定=☐合格 ☐不合格(判据#__)　｜　真t0=L__　真B=L__　真C=L__")
        lines.append("\n---\n")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--offset", type=int, default=0,
                    help="跳过排序后的前 N 个候选，用于生成不重复的新批次")
    ap.add_argument("--sources", nargs="*", default=None,
                    help="按来源分层过滤，如 run3_survive verify_drop guard_drop redflag_137")
    ap.add_argument("--include-ineligible", action="store_true",
                    help="兼容旧诊断模式：允许 C=bot/sys、B缺失等已知脏候选进入核对表")
    ap.add_argument("--allow-duplicate-roots", action="store_true",
                    help="兼容旧诊断模式：允许同一回复链根反复抽样")
    ap.add_argument("--include-reviewed", action="store_true",
                    help="兼容旧诊断模式：允许旧人工表/旧上下文已出现的候选再次进入核对表")
    ap.add_argument("--reviewed-files", nargs="*", default=None,
                    help="显式指定已审样本/旧表路径；默认使用已知 gold_review 表和 adjudication jsonl")
    ap.add_argument("--out", default=f"{DATA}/gold_review_sheet_pilot.md")
    ap.add_argument("--title", default="金标人工核对台 · 试点批（对话流指认版）")
    ap.add_argument("--pad-before", type=int, default=3,
                    help="每张卡按猜测跨度向前补多少条连续消息")
    ap.add_argument("--pad-after", type=int, default=4,
                    help="每张卡按猜测跨度向后补多少条连续消息")
    ap.add_argument("--cap", type=int, default=56,
                    help="每张卡最多展示多少条消息；超出时保留头尾并提示截断")
    a = ap.parse_args()
    cards, by_id, idx = build(
        a.n, a.sources, a.offset,
        eligible_only=not a.include_ineligible,
        dedupe_roots=not a.allow_duplicate_roots,
        exclude_reviewed=not a.include_reviewed,
        reviewed_paths=a.reviewed_files if a.reviewed_files is not None else _default_reviewed_paths([a.out]),
    )
    evs = rl.load_events()
    md = render(
        cards, evs, by_id, idx, a.title,
        pad_before=a.pad_before, pad_after=a.pad_after, cap=a.cap,
    )
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"✅ {len(cards)} 张卡 → {a.out}")
    for c in cards:
        print(f"   三料{c['score']}/3  {'+'.join(c['srcs'])}  {c['corrector']}")


if __name__ == "__main__":
    main()
