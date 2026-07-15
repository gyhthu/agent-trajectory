#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""runtime_lane —— 重建每条消息「运行时实际落到的 lane」（模型真看到的 thread）。

廉莲 2026-07-03 拍板：改写任务不在「应该怎么切」的理想 thread 里建，而在
**实际运行的 thread** 里建——即 feishu_agent 运行时按 _resolve_session/_bare_lane
真给这条消息派的 lane。这才是模型那一刻真正看到的上下文。

lane 判定完全照运行时自己的规则，不做任何 LLM 再分割：
  ① msg_task.json 有记录 → 用真值（运行时亲手落盘的 msg_id→lane，ground truth）
  ② 是回复、且回复链 root 拥有独立非主线 lane → 继承该 lane（运行时规则③）
  ③ 否则 → 主线(MAIN=chat_id)。这是运行时对 2 真人群裸消息的默认（恒主线，
     绝不按人裂）——>1100/1230 条落这里，印证运行时根本不细分。

铁律(承 turn_intent/arc_builder)：只报真数。msg_task 按 2天+500条 裁剪，老消息
lane 已被滚掉——命中真值的如实计 real，回落主线的计 main，绝不谎称都是真值。
"""
import os
import sys
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

GROUP = "oc_53b8b620867a189d8dfe502865dfccc5"
DATA = "/opt/shared/data/task-trajectory"
STATE = f"{DATA}/state/{GROUP}.json"
CHUNK1 = f"{DATA}/events_chunk1_107.json"
CHUNK2 = f"{DATA}/events_chunk2_raw224.json"
RECENT_EVENTS = f"{DATA}/runtime_recent_events.jsonl"
# 运行时亲手落盘的 msg_id→lane（本 bot 的 state 目录，非共享区）
MSG_TASK = ("/home/agent/lian-server-bot/skills/feishu-plugin/skills/"
            "feishu-event/scripts/msg_task.json")

MAIN = GROUP  # 主线 lane 就是群基线 chat_id（运行时 _bare_lane 的 main）


def load_events():
    """三源合并、按 msg_id 去重、按 ts 升序（与 arc_builder 同口径）。"""
    st = json.load(open(STATE, encoding="utf-8"))
    # 全量源：active_tail + 已冻结任务(frozen_tasks)里的 events + 其 subreqs 的 events。
    # 之前只读 active_tail_events(185)，是全量的子集 → 老纠错锚点(落在已冻结任务里)
    # 取不到上下文而 miss_context。frozen 事件 schema 与 chunk 同(msg_id/text/ts/
    # parent_id/role/name)，直接并入。全量约 2446 条，覆盖 50/51 纠错锚点。
    srcs = [st.get("active_tail_events", [])]
    for t in st.get("frozen_tasks", []):
        srcs.append(t.get("events", []))
        for sr in t.get("subreqs", []):
            srcs.append(sr.get("events", []))
    for p in (CHUNK1, CHUNK2):
        d = json.load(open(p, encoding="utf-8"))
        srcs.append(d if isinstance(d, list) else d.get("events", []))
    if os.path.exists(RECENT_EVENTS):
        recent = []
        for line in open(RECENT_EVENTS, encoding="utf-8"):
            if line.strip():
                recent.append(json.loads(line))
        srcs.append(recent)
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
    evs.sort(key=lambda e: float(e.get("ts") or 0))
    return evs


def load_real_lanes():
    """运行时落盘的 msg_id→lane（本群）。缺文件返回空。"""
    try:
        d = json.load(open(MSG_TASK, encoding="utf-8"))
        return {k: v[0] for k, v in (d.get(GROUP) or {}).items() if v}
    except Exception:
        return {}


def _root_of(mid, byid):
    """沿 parent_id 走到回复链根。防环。"""
    seen = set()
    e = byid.get(mid)
    while e and e.get("parent_id") and e["parent_id"] in byid and mid not in seen:
        seen.add(mid)
        mid = e["parent_id"]
        e = byid.get(mid)
    return mid


def assign_lanes(events, real=None):
    """返回 (lane_of: msg_id->lane, prov: msg_id->'real'|'inherited'|'main')。"""
    real = real if real is not None else load_real_lanes()
    byid = {e["msg_id"]: e for e in events if e.get("msg_id")}
    lane_of, prov = {}, {}
    for e in events:
        mid = e.get("msg_id")
        if not mid:
            continue
        if mid in real:
            lane_of[mid], prov[mid] = real[mid], "real"
            continue
        # 回复链 root 有独立非主线真值 lane → 继承（运行时规则③）
        if e.get("parent_id"):
            r = _root_of(mid, byid)
            rlane = real.get(r)
            if rlane and rlane != MAIN:
                lane_of[mid], prov[mid] = rlane, "inherited"
                continue
        lane_of[mid], prov[mid] = MAIN, "main"
    return lane_of, prov


def lane_short(lane):
    """MAIN / s:2 / fork:... 短名，便于打印。"""
    if lane == MAIN:
        return "MAIN"
    return lane.split(GROUP, 1)[-1].lstrip(":") if GROUP in lane else lane


if __name__ == "__main__":
    from collections import Counter
    evs = load_events()
    real = load_real_lanes()
    lane_of, prov = assign_lanes(evs, real)
    pc = Counter(prov.values())
    lc = Counter(lane_short(l) for l in lane_of.values())
    print(f"总事件            : {len(evs)}")
    print(f"  真值 lane(real) : {pc['real']}")
    print(f"  回复链继承      : {pc['inherited']}")
    print(f"  回落主线(main)  : {pc['main']}")
    print("lane 分布：")
    for l, n in lc.most_common():
        print(f"  {n:4d}  {l}")
