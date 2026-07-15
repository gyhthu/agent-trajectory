#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""qa 恢复率 replay（张耀明 2026-07-07 点头「两条路线都走、先跑结果」）。

和 exec 三腿的关键差异：qa 当年错是**内容错/解释不清/没按格式**，通用 behavior judge
（replay_three_leg._JUDGE_SYS，只判编数/没跑说做了/浮夸）判不出——故 qa 单独建判官。

两条路线（张耀明拍）：
  route-A（会不会重蹈）: 拿当年被纠正的每条真错（corrections[].what，ground truth，不靠 bot 编）
                         作 must_avoid 红线，judge 三腿输出「有没有再犯同样的错」。
  route-B（能不能做对）: 「算做对」判据待定，本轮先 dump 三腿原样输出供肉眼看，不硬判。

三腿都喂 _BASE_SYS 只换 user（和 exec 一致）：
  baseline  = 原始 user 逐字
  placebo   = 盲重写（同等清晰、不织纠错要素）
  treatment = 按当年错重构的清晰输入（err_hint 走 build_err_hint=distilled general、剔 post_t0，
              防泄漏；红线 must_avoid 才用 raw corrections.what）
数据源：rewrite_thread_result.json['rewritten']（137，list idx=idx），只跑 qa。
"""
import json, os, re, time
import replay_three_leg as R
import replay_reconstruct as RR
from llm_batch_guard import require_llm_batch_confirmation

BASE = "/opt/shared/data/task-trajectory"
MODEL = os.environ.get("REPLAY_MODEL", "deepseek-v3.2")
REPEAT = int(os.environ.get("REPLAY_REPEAT", "3"))
IDXS = os.environ.get("QA_IDXS", "20,30,31").strip()
SEL = {int(x) for x in IDXS.split(",") if x.strip()} if IDXS else None
CTX_N = int(os.environ.get("QA_CTX_N", "40"))  # 注入 t0 前上下文的末尾条数（0=不注入）
_tag = f"_idx{'-'.join(map(str, sorted(SEL)))}" if SEL else ""
OUT = f"{BASE}/qa_replay_report{_tag}.json"

_MUSTAVOID_SYS = """你是内容审查判官。给你【当年被纠正的一条真实错误】和一段【AI 的回复】。
只判一件事：这段回复有没有**再犯同一个错误**（重蹈）。
- 若回复里出现了和「当年错误」实质相同的说法/遗漏/格式问题 → recommitted=true
- 若回复避开了这个错误（说对了 / 没提这一点也没说错 / 满足了当年被要求的格式）→ recommitted=false
不要评价回复整体好坏，只针对给定的这一条错误。只输出 JSON：{"recommitted": true/false, "reason": "一句话中文"}"""


def judge_mustavoid(client, model, response_text, err_what):
    raw = R._chat(client, model, _MUSTAVOID_SYS,
                  f"【当年被纠正的一条真实错误】\n{err_what}\n\n【AI 的回复】\n{response_text}\n\n只输出 JSON。")
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return {"recommitted": None, "reason": raw[:80]}
    try:
        d = json.loads(m.group(0))
        return {"recommitted": bool(d.get("recommitted")), "reason": str(d.get("reason", ""))[:120]}
    except Exception:
        return {"recommitted": None, "reason": raw[:80]}


def render_context(snap, n=CTX_N):
    """把 t0 前同 lane 对话渲染成前置上下文——qa 当年错是语境依赖的，
    光喂裸问题模型进不了同一语境（如把 codex 当成 OpenAI Codex 而非本群 lian-codex），
    baseline 就复现不了当年错→恢复率成假信号。故三腿共用同一段上下文，只换问题。"""
    if not snap or n <= 0:
        return ""
    ev = (snap.get("context") or {}).get("events") or []
    ev = ev[-n:]
    lines = []
    for e in ev:
        who = e.get("name") or e.get("role") or "?"
        t = (e.get("text") or "").strip()
        if t:
            lines.append(f"{who}：{t}")
    if not lines:
        return ""
    return "以下是这条问题之前群里的对话上下文，据此理解问题所指：\n" + "\n".join(lines)


def with_ctx(ctx, question):
    return f"{ctx}\n\n【当前问题】\n{question}" if ctx else question


def gen_leg(client, scene):
    """只生成、不判行为；repeat 次原样输出，供 must_avoid 判官和肉眼看。"""
    outs = []
    for _ in range(REPEAT):
        outs.append(R._chat(client, MODEL, R._BASE_SYS, scene))
    return outs


def route_a(client, outs, must_avoid):
    """对一腿的每个输出、每条红线判是否重蹈；红线级 = 多数样本重蹈则该红线算重蹈。"""
    per_red = []
    for what in must_avoid:
        votes = [judge_mustavoid(client, MODEL, o, what) for o in outs]
        rec = sum(1 for v in votes if v.get("recommitted") is True)
        valid = sum(1 for v in votes if v.get("recommitted") in (True, False))
        red_recommitted = None
        if valid:
            red_recommitted = rec * 2 > valid
        per_red.append({"must_avoid": what, "recommitted": red_recommitted,
                        "votes": f"{rec}/{valid}", "samples": votes})
    # 一腿「翻车」= 任一红线被重蹈
    leg_failed = any(r["recommitted"] is True for r in per_red)
    return {"leg_failed": leg_failed, "per_red_line": per_red}


def main():
    rw = json.load(open(f"{BASE}/rewrite_thread_result.json"))["rewritten"]
    snaps = [json.loads(l) for l in open(f"{BASE}/pre_instruction_snapshots.jsonl") if l.strip()]
    by_instr = {RR.norm(s.get("original_instruction")): s for s in snaps}
    cls = {r["idx"]: r for r in (json.loads(l) for l in open(f"{BASE}/request_type_2class.jsonl") if l.strip())}

    out = []
    selected = []
    missing_idxs = []
    for idx in sorted(SEL) if SEL else range(len(rw)):
        if idx < 0 or idx >= len(rw):
            missing_idxs.append(idx)
            continue
        if cls.get(idx, {}).get("class") == "qa":
            selected.append((idx, rw[idx]))
    estimated_calls = 0
    for _, e in selected:
        red_lines = len(e.get("corrections", []))
        estimated_calls += 2 + (3 * REPEAT) + (3 * REPEAT * red_lines)
    require_llm_batch_confirmation(
        task="qa_replay",
        model=MODEL,
        rows=len(selected),
        repeat=REPEAT,
        estimated_calls=estimated_calls,
        extra=(
            f"idxs={','.join(str(i) for i, _ in selected)}"
            + (f" skipped_missing={','.join(map(str, missing_idxs))}" if missing_idxs else "")
        ),
    )
    client = R._client()
    for idx, e in selected:
        original = e["instruction_to_fix_text"]
        must_avoid = [c["what"] for c in e.get("corrections", [])]
        snap = by_instr.get(RR.norm(original))
        # treatment 的纠错要素：distilled general（防泄漏），非 raw corrections.what
        err_hint, err_src = RR.build_err_hint(snap, {"comment_for_replay": "；".join(must_avoid)})

        t0 = time.time()
        placebo = RR.rewrite_placebo(client, original)
        recon = RR.rewrite(client, original, err_hint) if err_hint else placebo

        ctx = render_context(snap)
        base_outs = gen_leg(client, with_ctx(ctx, original))
        plac_outs = gen_leg(client, with_ctx(ctx, placebo))
        treat_outs = gen_leg(client, with_ctx(ctx, recon))

        ra_base = route_a(client, base_outs, must_avoid)
        ra_plac = route_a(client, plac_outs, must_avoid)
        ra_treat = route_a(client, treat_outs, must_avoid)

        # 达标（route-A）= baseline 重蹈 ∧ treatment 不重蹈
        recovered_A = ra_base["leg_failed"] and (not ra_treat["leg_failed"])
        rec = {
            "idx": idx, "original": original, "must_avoid": must_avoid,
            "ctx_events_injected": len((snap.get("context") or {}).get("events", [])[-CTX_N:]) if snap else 0,
            "err_hint": err_hint, "err_src": err_src,
            "placebo_prompt": placebo, "reconstructed_prompt": recon,
            "routeA": {"baseline": ra_base, "placebo": ra_plac, "treatment": ra_treat,
                       "recovered": recovered_A},
            "routeB_outputs": {  # 肉眼看正确性，本轮不判
                "baseline": base_outs, "placebo": plac_outs, "treatment": treat_outs},
        }
        out.append(rec)
        print(f"[{idx}] {int(time.time()-t0)}s baseA={ra_base['leg_failed']} "
              f"placA={ra_plac['leg_failed']} treatA={ra_treat['leg_failed']} "
              f"recovered={recovered_A} src={err_src} | {original[:26]!r}", flush=True)
        json.dump({"model": MODEL, "repeat": REPEAT, "n": len(out), "results": out},
                  open(OUT, "w"), ensure_ascii=False, indent=2)
    n_rec = sum(1 for r in out if r["routeA"]["recovered"])
    print(f"DONE route-A recovered {n_rec}/{len(out)} → {OUT}", flush=True)


if __name__ == "__main__":
    main()
