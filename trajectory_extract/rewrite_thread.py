#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""rewrite_thread —— 改写任务：在【实际运行的 thread(lane)】里把纠正合并成一条完备指令。

廉莲 2026-07-03 定的独立任务：改写不依赖「应该怎么切」的理想 thread，而在运行时
真给每条消息派的 lane（runtime_lane 重建）里做——这才是模型那一刻真看到的上下文。

两步（承廉莲三步的②③，但 thread 换成运行时 lane）：
  ② 锚定：对每个纠正锚点，在【同 lane 的前序消息】里回指它纠/补的原始指令
     (instruction_to_fix)。窗口=同 lane 前序，绝不跨 lane——模型在主线里根本
     没看到 :s:2 / :fork 的消息，原始指令不可能在那。这是相对旧 arc_builder
     「扁平30条窗口」的实质订正。
  ③ 合并改写：按 (lane, instruction_to_fix) 分组，把指向同一原始指令的多条纠正，
     连同原始指令，LLM 合并改写成一条更顺、更完备的指令(comment)。

铁律(承 turn_intent/arc_builder/runtime_lane)：只报真数。锚点 lane 是真值(msg_task
命中)还是回落主线，如实分开计；上下文捞不到的如实标 miss，绝不硬凑。

复用：runtime_lane(lane 重建) + arc_builder._ANCHOR_SYS(锚定 prompt)
      + turn_intent._client/_LLM_MODEL(deepseek) + task_stitch._strip_feishu
"""
import os
import sys
import json
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import task_stitch as ts  # noqa: E402
import runtime_lane as rl  # noqa: E402
from turn_intent import _client, _LLM_MODEL  # noqa: E402
from arc_builder import _ANCHOR_SYS  # noqa: E402  复用消息级锚定 prompt

DATA = "/opt/shared/data/task-trajectory"
CENSUS = f"{DATA}/session_corrections.jsonl"  # 纠错来源=逐session抽取(session_corrections.py产物,197条,取代旧blob-census;旧35=correction_census.jsonl/旧51=correction_census_0703.jsonl已弃)
OUT = f"{DATA}/rewrite_thread_result.json"
WINDOW = 30  # 同 lane 前序取多少条当锚定候选


def load_census():
    return [json.loads(l) for l in open(CENSUS, encoding="utf-8") if l.strip()]


# ---------- ② 锚定：同 lane 前序窗口 ----------

def _lane_prefix(anchor_mid, lane_of, ordered):
    """取锚点所在 lane、在锚点之前(含锚点)的同 lane 消息序列。"""
    lane = lane_of.get(anchor_mid)
    if lane is None:
        return None, []
    seq = [e for e in ordered if lane_of.get(e["msg_id"]) == lane]
    # 截到锚点为止
    out = []
    for e in seq:
        out.append(e)
        if e["msg_id"] == anchor_mid:
            break
    return lane, out[-(WINDOW + 1):]  # 最后 WINDOW 条 + 锚点


def anchor_one(anchor, lane_of, ordered):
    aid = anchor["anchor_msg_id"]
    lane, window = _lane_prefix(aid, lane_of, ordered)
    if not window or window[-1]["msg_id"] != aid:
        return {"anchor": aid, "status": "miss_context", "fix": None,
                "lane": rl.lane_short(lane) if lane else "?", "task": anchor["task"]}
    cand, id_map = [], {}
    for k, e in enumerate(window):
        tag = f"M{k+1}"
        id_map[tag] = e["msg_id"]
        role = e.get("role", "")
        name = e.get("name", "") or role
        txt = ts._strip_feishu(e.get("text", "") or "")[:120]
        mark = "  ←【本次纠正】" if e["msg_id"] == aid else ""
        cand.append(f"{tag} [{role}/{name}] {txt}{mark}")
    user_msg = (
        "【同一运行时 thread 内的上下文（候选，时间升序，最后一条是本次纠正）】\n"
        + "\n".join(cand)
        + f"\n\n【本次纠正在纠什么】{anchor['what']}"
        + "\n\n请指出这次纠正针对的原始指令是哪条候选（输出其编号）。"
    )
    resp = _client().chat.completions.create(
        model=_LLM_MODEL,
        messages=[{"role": "system", "content": _ANCHOR_SYS},
                  {"role": "user", "content": user_msg}],
        temperature=0,
    )
    raw = resp.choices[0].message.content or ""
    m = re.search(r"\{.*\}", raw, re.S)
    pick = None
    if m:
        try:
            pick = str(json.loads(m.group(0)).get("pick", "")).strip()
        except Exception:
            pick = None
    if not pick or pick.lower() == "none" or pick not in id_map:
        return {"anchor": aid, "status": "no_fix_found", "fix": None,
                "lane": rl.lane_short(lane), "task": anchor["task"], "pick": pick}
    fix_mid = id_map[pick]
    fix_ev = next(e for e in window if e["msg_id"] == fix_mid)
    return {"anchor": aid, "status": "ok", "fix": fix_mid, "lane": rl.lane_short(lane),
            "fix_text": ts._strip_feishu(fix_ev.get("text", "")),
            "fix_name": fix_ev.get("name", ""), "task": anchor["task"],
            "what": anchor["what"], "corrector": anchor.get("corrector", "")}


# ---------- ③ 合并改写：同一 (lane, instruction_to_fix) 的多条纠正 → 一条完备指令 ----------

_REWRITE_SYS = (
    "你是指令改写助手。给你一条【原始指令】和它在同一段对话里陆续收到的若干【纠正/补充】，"
    "请把它们合并、改写成 **一条** 更顺、更完备、一次就能让执行者做对的指令——"
    "即：如果当初就这么说，就不会有后面这些纠正。\n"
    "要求：①保留原始指令的真实意图，把纠正里的约束/口径/边界补进去；"
    "②直接写成用户口吻的一句/一段话指令，不要罗列'纠正1/纠正2'，不要解释；"
    "③只输出改写后的指令正文，不要 markdown、不要前后缀。"
)


def rewrite_group(fix_text, corrections):
    corr_lines = "\n".join(
        f"- （{c.get('corrector','')}）{c['what']}" for c in corrections)
    user_msg = (
        f"【原始指令】\n{fix_text}\n\n"
        f"【随后收到的纠正/补充（共 {len(corrections)} 条）】\n{corr_lines}\n\n"
        "请合并改写成一条完备指令。"
    )
    resp = _client().chat.completions.create(
        model=_LLM_MODEL,
        messages=[{"role": "system", "content": _REWRITE_SYS},
                  {"role": "user", "content": user_msg}],
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()


def run(limit=None):
    census = load_census()
    if limit:
        census = census[:limit]
    ordered = rl.load_events()
    lane_of, prov = rl.assign_lanes(ordered)

    # ② 锚定
    anchored = []
    for n, a in enumerate(census):
        r = anchor_one(a, lane_of, ordered)
        r["lane_prov"] = prov.get(a["anchor_msg_id"], "?")
        anchored.append(r)
        print(f"[②{n+1}/{len(census)}] {r['status']:12s} lane={r['lane']:6s}"
              f"({r['lane_prov']:4s}) {a['task'][:16]:16s} -> {r.get('fix_text','')[:40]}",
              flush=True)

    ok = [r for r in anchored if r["status"] == "ok"]
    # 按 (lane, fix) 分组
    groups = {}
    for r in ok:
        groups.setdefault((r["lane"], r["fix"]), []).append(r)

    # ③ 合并改写
    print("\n" + "=" * 60 + "\n③ 合并改写：")
    rewritten = []
    for (lane, fix), rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        merged = rewrite_group(rs[0]["fix_text"], rs)
        rewritten.append({
            "lane": lane, "instruction_to_fix_msg": fix,
            "instruction_to_fix_text": rs[0]["fix_text"],
            "n_corrections": len(rs),
            "corrections": [{"what": r["what"], "corrector": r["corrector"],
                             "anchor": r["anchor"]} for r in rs],
            "rewritten": merged,
        })
        print(f"  [{len(rs)}纠 lane={lane}] {rs[0]['fix_text'][:34]}\n"
              f"      → {merged[:80]}", flush=True)

    # 统计
    miss = [r for r in anchored if r["status"] == "miss_context"]
    nofix = [r for r in anchored if r["status"] == "no_fix_found"]
    real_lane = sum(1 for r in anchored if r["lane_prov"] == "real")
    print("\n" + "=" * 60)
    print(f"锚点总数              : {len(anchored)}")
    print(f"  lane 真值(msg_task) : {real_lane}   （其余为回落主线重建）")
    print(f"  上下文缺失(miss)    : {len(miss)}")
    print(f"  回指不到原始指令    : {len(nofix)}")
    print(f"  成功回指(ok)        : {len(ok)}")
    print(f"★ 改写产出(完备指令)条数 : {len(rewritten)}  （按 lane+原始指令去重合并）")
    print("=" * 60)

    json.dump({"anchored": anchored, "rewritten": rewritten,
               "n_rewritten": len(rewritten), "n_ok": len(ok),
               "n_real_lane": real_lane, "n_miss": len(miss), "n_nofix": len(nofix)},
              open(OUT, "w"), ensure_ascii=False, indent=1)
    print(f"落盘: {OUT}")
    return anchored, rewritten


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run(limit=lim)
