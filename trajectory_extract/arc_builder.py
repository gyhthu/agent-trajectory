#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""arc_builder —— 把 correction_census 的纠正锚点回指到「被纠的原始指令」。

廉莲三步的第②步执行体：对每个纠正锚点(anchor_msg_id)，在合并 transcript 里
往前定位它纠/补的那条【原始指令】(instruction_to_fix)，做成 pair；再按
instruction_to_fix 去重，数出「消息级」完备 comment 的真条数。

粒度=消息级（张耀明 2026-07-03 拍板先跑）：instruction_to_fix 锚到整条用户
发起消息(msg_id)，哪怕它含多子需求也不拆；同一发起指令下的多次纠正/追问都
回指到它。

数据源(合并去重，按 ts 升序)：
  state/oc_53b8b….json:active_tail_events(近期尾,1230) + events_chunk1_107 + chunk2_raw224
铁律(承 turn_intent)：只报真跑出来的数；锚点上下文捞不到就如实标 miss、报覆盖率，绝不硬凑。

复用：turn_intent._client / _LLM_MODEL(deepseek, 本机 litellm) + task_stitch._strip_feishu
"""
import os
import sys
import json
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import time  # noqa: E402
import task_stitch as ts  # noqa: E402
from turn_intent import _client, _LLM_MODEL  # noqa: E402


def _chat_retry(client, **kw):
    """v4-pro/deepseek 全局限流 3/min 且被本群 live 任务抢占：429 短间隔多次重试。"""
    for attempt in range(40):
        try:
            return client.chat.completions.create(**kw)
        except Exception as ex:  # noqa: BLE001
            if "429" in str(ex) or "RateLimit" in str(ex):
                time.sleep(8)
                continue
            raise
    raise RuntimeError("deepseek 429 重试耗尽")

DATA = "/opt/shared/data/task-trajectory"
CENSUS = f"{DATA}/correction_census.jsonl"
STATE = f"{DATA}/state/oc_53b8b620867a189d8dfe502865dfccc5.json"
CHUNK1 = f"{DATA}/events_chunk1_107.json"
CHUNK2 = f"{DATA}/events_chunk2_raw224.json"
WINDOW = 30  # 锚点往前取多少条消息当候选窗口


def load_census():
    return [json.loads(l) for l in open(CENSUS, encoding="utf-8") if l.strip()]


def load_merged_transcript():
    """多源合并、按 msg_id 去重、按 ts 升序。
    修（2026-07-03）：原来只读 state 的 active_tail_events，漏了 frozen_tasks[].events——
    冻结任务其实带完整正文，reply_chain_gold 一直读两部分，这里没读，致 8 个锚点被误判
    miss_context。现补上 frozen_tasks[].events（与 reply_chain_gold.load_events_from_state 一致）。"""
    srcs = []
    st = json.load(open(STATE, encoding="utf-8"))
    for t in st.get("frozen_tasks", []):
        srcs.append(t.get("events") or [])
    srcs.append(st.get("active_tail_events") or [])
    for p in (CHUNK1, CHUNK2):
        d = json.load(open(p, encoding="utf-8"))
        srcs.append(d if isinstance(d, list) else d.get("events", []))
    seen, evs = set(), []
    for src in srcs:
        for e in src:
            if not isinstance(e, dict):
                continue
            mid = e.get("msg_id")
            if not mid or mid in seen:
                continue
            seen.add(mid)
            evs.append(e)
    evs.sort(key=lambda e: e.get("ts") or 0)
    return evs


_ANCHOR_SYS = (
    "你是对话轨迹标注助手。给你一段群聊上下文（时间升序）和最后发生的一次【纠正】，"
    "请找出这次纠正针对的【原始指令】——即当初引出被纠行为的那条用户发起消息"
    "（提问/下命令/交代任务的原话），不是纠正消息本身、也不是中间任何 bot 回复。\n"
    "粒度=消息级：锚到整条用户消息，哪怕它含多个子需求也不拆。\n"
    '严格输出 JSON：{"pick":"候选编号如 M12，找不到明确原始指令填 none","reason":"≤25字"}，'
    "不要 markdown、不要多余解释。"
)


def anchor_one(anchor, merged, idx_of):
    """对单个锚点回指原始指令。返回 dict(含 instruction_to_fix msg_id 或 miss/none)。"""
    aid = anchor["anchor_msg_id"]
    if aid not in idx_of:
        return {"anchor": aid, "status": "miss_context", "fix": None,
                "task": anchor["task"]}
    i = idx_of[aid]
    lo = max(0, i - WINDOW)
    window = merged[lo:i + 1]  # 含锚点自身
    # 给候选消息编短号 M1.. 映射回 msg_id
    cand, id_map = [], {}
    for k, e in enumerate(window):
        tag = f"M{k+1}"
        id_map[tag] = e.get("msg_id")
        role = e.get("role", "")
        name = e.get("name", "") or role
        txt = ts._strip_feishu(e.get("text", "") or "")[:120]
        mark = "  ←【本次纠正】" if e.get("msg_id") == aid else ""
        cand.append(f"{tag} [{role}/{name}] {txt}{mark}")
    user_msg = (
        "【群聊上下文（候选，时间升序，最后一条是本次纠正）】\n"
        + "\n".join(cand)
        + f"\n\n【本次纠正在纠什么】{anchor['what']}"
        + "\n\n请指出这次纠正针对的原始指令是哪条候选（输出其编号）。"
    )
    client = _client()
    resp = _chat_retry(
        client,
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
                "task": anchor["task"], "pick": pick}
    fix_mid = id_map[pick]
    fix_ev = merged[idx_of[fix_mid]]
    return {"anchor": aid, "status": "ok", "fix": fix_mid,
            "fix_text": ts._strip_feishu(fix_ev.get("text", ""))[:80],
            "fix_name": fix_ev.get("name", ""), "task": anchor["task"]}


def run(limit=None):
    census = load_census()
    merged = load_merged_transcript()
    idx_of = {e["msg_id"]: i for i, e in enumerate(merged) if e.get("msg_id")}
    if limit:
        census = census[:limit]
    results = []
    for n, a in enumerate(census):
        r = anchor_one(a, merged, idx_of)
        results.append(r)
        print(f"[{n+1}/{len(census)}] {r['status']:14s} {a['task'][:18]:18s} "
              f"-> {r.get('fix_text','')[:46]}", flush=True)
    # 统计
    ok = [r for r in results if r["status"] == "ok"]
    miss = [r for r in results if r["status"] == "miss_context"]
    nofix = [r for r in results if r["status"] == "no_fix_found"]
    fixes = {}
    for r in ok:
        fixes.setdefault(r["fix"], []).append(r)
    print("\n" + "=" * 60)
    print(f"锚点总数            : {len(results)}")
    print(f"  上下文缺失(miss)  : {len(miss)}")
    print(f"  回指不到原始指令  : {len(nofix)}")
    print(f"  成功回指(ok)      : {len(ok)}")
    print(f"★ 消息级 instruction_to_fix 去重条数 : {len(fixes)}  (基于 {len(ok)} 个可回指锚点)")
    print("=" * 60)
    print("\n各 instruction_to_fix 收纳的锚点数(即合并了几条纠正)：")
    for fix, rs in sorted(fixes.items(), key=lambda kv: -len(kv[1])):
        print(f"  [{len(rs)}锚] {rs[0]['fix_name']}: {rs[0]['fix_text'][:44]}")
    out = f"{DATA}/arc_msglevel_result.json"
    json.dump({"results": results,
               "n_instruction_to_fix": len(fixes),
               "n_ok": len(ok), "n_miss": len(miss), "n_nofix": len(nofix)},
              open(out, "w"), ensure_ascii=False, indent=1)
    print(f"\n落盘: {out}")
    return results, fixes


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run(limit=lim)
