#!/usr/bin/env python3
"""Reply-chain 免费金标集 —— 把飞书「引用回复」当零成本 ground truth。

一条带 `parent_id` 的消息，天然携带一条 ground truth：**「我和被引用的父消息属于同一
会话/任务」**。这批监督样本不花一分钱标注费，且量大（单群 state 实测 667 条边）。本模块：

  1. 抽回复边：child→parent 且两端都在事件集内（parent 挂空/挂到集外的丢弃）。
  2. 并查集把回复边连成 **gold 共属簇**（连通分量）——注意这是「共属」正标签，
     只断言「这些消息在同一会话」，**不**断言不同簇之间一定分属不同会话（回复链只连不断）。
  3. 免费基线 eval：读当前切分管线的 msg_id→任务 归属，逐条回复边判「child 和 parent
     是否被切进同一任务」，算一致率。这就是「按单条消息路由」新方案要打败的旧基线。
  4. 导出 gold jsonl：每条回复边一条样本（路由到 parent 会话的监督/评测数据）。

数据源二选一：
  · --state <group.json>  默认。直接读增量层落的 state（含 events + frozen_tasks 归属），
    既拿事件又拿「当前切分归属」，一步到位算基线。
  · --hist-file <messages.raw.jsonl>  独立走 task_stitch.fetch_history，只能抽金标、
    不算基线（因为没有切分归属）。用于金标集要比某次 state 更全/更新时。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import task_stitch as ts  # noqa: E402


# ---------------------------------------------------------------- 事件加载

def load_events_from_state(state: dict) -> list[dict]:
    """从增量 state 还原全部输入事件（frozen_tasks[].events + active_tail_events），按 msg_id 去重。"""
    evs: list[dict] = []
    for t in state.get("frozen_tasks", []):
        evs.extend(t.get("events") or [])
    evs.extend(state.get("active_tail_events") or [])
    by: dict = {}
    for e in evs:
        key = e.get("msg_id") or f"ts:{e.get('ts')}:{len(by)}"
        by[key] = e
    return sorted(by.values(), key=lambda e: e.get("ts") or 0)


def assignment_from_state(state: dict) -> dict[str, int]:
    """当前切分归属：msg_id → 任务下标。冻结任务按 frozen_tasks 顺序编号，
    活动尾巴统一编到 -1（同一个「未终结活动任务」，reply 边落在其中算同任务）。"""
    amap: dict[str, int] = {}
    for i, t in enumerate(state.get("frozen_tasks", [])):
        for mid in t.get("member_msg_ids") or []:
            amap[mid] = i
    for e in state.get("active_tail_events") or []:
        mid = e.get("msg_id")
        if mid and mid not in amap:
            amap[mid] = -1
    return amap


# ---------------------------------------------------------------- 并查集

class _DSU:
    def __init__(self):
        self.p: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.p.setdefault(x, x)
        root = x
        while self.p[root] != root:
            root = self.p[root]
        while self.p[x] != root:      # 路径压缩
            self.p[x], x = root, self.p[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


# ---------------------------------------------------------------- 金标 + 基线

def extract_edges(events: list[dict]) -> list[dict]:
    """抽 child→parent 回复边（两端都在集内）。返回边列表，携带两端可读文本供样本用。"""
    by_id = {e.get("msg_id"): e for e in events if e.get("msg_id")}
    edges: list[dict] = []
    for e in events:
        pid = e.get("parent_id")
        mid = e.get("msg_id")
        if not pid or not mid or pid not in by_id:
            continue
        parent = by_id[pid]
        edges.append({
            "child_msg_id": mid,
            "parent_msg_id": pid,
            "child_role": e.get("role"),
            "parent_role": parent.get("role"),
            "child_ts": e.get("ts"),
            "child_text": ts._strip_feishu(e.get("text", "")).strip()[:500],
            "parent_text": ts._strip_feishu(parent.get("text", "")).strip()[:500],
        })
    return edges


def gold_components(edges: list[dict]) -> dict[str, int]:
    """并查集把回复边连成 gold 共属簇，返回 msg_id → 分量 id（稳定编号：按分量最小 msg_id 排序）。"""
    dsu = _DSU()
    for ed in edges:
        dsu.union(ed["child_msg_id"], ed["parent_msg_id"])
    roots = sorted({dsu.find(m) for ed in edges for m in (ed["child_msg_id"], ed["parent_msg_id"])})
    root_to_cid = {r: i for i, r in enumerate(roots)}
    comp: dict[str, int] = {}
    for ed in edges:
        for m in (ed["child_msg_id"], ed["parent_msg_id"]):
            comp[m] = root_to_cid[dsu.find(m)]
    return comp


def score_baseline(edges: list[dict], assignment: dict[str, int]) -> dict:
    """免费基线 eval：逐条回复边判「当前切分是否把 child 和 parent 放进同一任务」。
    只统计两端都有切分归属的边（归属缺失的边无法判、单列）。"""
    same = diff = unknown = 0
    disagreements: list[dict] = []
    for ed in edges:
        ca = assignment.get(ed["child_msg_id"])
        pa = assignment.get(ed["parent_msg_id"])
        if ca is None or pa is None:
            unknown += 1
            continue
        if ca == pa:
            same += 1
        else:
            diff += 1
            disagreements.append({**ed, "child_task": ca, "parent_task": pa})
    scored = same + diff
    return {
        "edges_total": len(edges),
        "edges_scored": scored,
        "edges_unknown_assignment": unknown,
        "agree_same_task": same,
        "disagree_split": diff,
        "agreement_rate": round(same / scored, 4) if scored else None,
        "disagreements": disagreements,
    }


def build(events: list[dict], assignment: dict[str, int] | None) -> dict:
    edges = extract_edges(events)
    comp = gold_components(edges)
    n_comp = len(set(comp.values()))
    out = {
        "events_total": len(events),
        "reply_edges": len(edges),
        "gold_components": n_comp,
        "edges": edges,
        "gold_component_of": comp,
    }
    if assignment is not None:
        out["baseline"] = score_baseline(edges, assignment)
    return out


def write_gold_jsonl(result: dict, path: Path) -> int:
    """每条回复边导出一条样本：路由监督/评测数据（child 应路由到 parent 所在会话）。"""
    comp = result["gold_component_of"]
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for ed in result["edges"]:
            rec = {
                **ed,
                "gold_component": comp.get(ed["child_msg_id"]),
                "label": "same_session",   # 回复边只提供正标签
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--state", help="增量 state json（含 events + 当前切分归属，可算基线）")
    src.add_argument("--hist-file", help="messages.raw.jsonl（只抽金标，无基线）")
    ap.add_argument("--out", help="gold jsonl 输出路径", default=None)
    ap.add_argument("--summary-only", action="store_true", help="只打摘要，不写 jsonl")
    args = ap.parse_args()

    assignment: dict[str, int] | None = None
    if args.state:
        state = json.loads(Path(args.state).read_text(encoding="utf-8"))
        events = load_events_from_state(state)
        assignment = assignment_from_state(state)
    else:
        events = ts.fetch_history("", 0, 0, hist_file=args.hist_file)

    result = build(events, assignment)

    summary = {k: v for k, v in result.items()
               if k not in ("edges", "gold_component_of")}
    if "baseline" in summary:
        summary["baseline"] = {k: v for k, v in summary["baseline"].items()
                               if k != "disagreements"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if not args.summary_only:
        out = Path(args.out) if args.out else (ts.SHARED / "reply_chain_gold.jsonl")
        n = write_gold_jsonl(result, out)
        print(f"[gold] 写出 {n} 条金标样本 → {out}")


if __name__ == "__main__":
    main()
