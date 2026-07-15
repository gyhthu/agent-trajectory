"""A 路线验证·批量子需求切分：把 segment_history 切出的碎簇按 merge-map 并成真任务，
逐个跑 llm_decompose(第②步)，产出 per-task 表 + 一张汇总全量样张。

碎簇问题：segment_history(第①步) 靠〔标签〕话题刀+gap 表面信号，把一摊连续的事
切碎成 1~4 条的续段。A 路线 = 不动第①步，**人工 merge-map 把续段并回真任务**，
先验证第②步(已治干净的子需求切分)在真任务上的产出格式/质量。第①步治本(语义任务切分)是 B 路线。

复用 llm_decompose 的全部核心(build_transcript/llm_decompose/enforce_rollback_purity/render)，
不重写。merge-map 用 1-based 簇号(对应 segment_history 输出顺序)。

用法：
  python3 batch_decompose.py --hist-file <messages.raw.jsonl> --group oc_xxx \
      --merge-map '[["轨迹提取与保存",[1,2,4]],["装antigravity-bot全程",[8,9,10,11,12,13]]]' \
      --model deepseek-v4-pro
不传 --merge-map 时：每个有效消息数 ≥ --min-msgs 的簇各算一个任务。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # data_process 根(脱敏器在此)
import task_stitch as ts  # noqa: E402
from regex_anonymizer import RegexAnonymizer  # noqa: E402
import llm_decompose as ld  # noqa: E402


def _real_n(cluster):
    return sum(1 for e in cluster if not ts._is_noise(e["text"]))


def build_tasks(clusters, merge_map, min_msgs):
    """→ [(task_name, merged_cluster_events_sorted_by_ts), ...]"""
    if merge_map:
        tasks = []
        for name, idxs in merge_map:
            evs = []
            for i in idxs:
                evs += clusters[i - 1]  # 1-based
            evs.sort(key=lambda e: e["ts"])
            tasks.append((name, evs))
        return tasks
    # 无 map：每个 ≥min_msgs 的簇各一个任务
    return [(f"task{i}", c) for i, c in enumerate(clusters, 1)
            if _real_n(c) >= min_msgs]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", required=True)
    ap.add_argument("--hist-file", required=True)
    ap.add_argument("--merge-map", help='JSON: [["任务名",[簇号,...]],...]')
    ap.add_argument("--min-msgs", type=int, default=10, help="无 map 时簇的最小有效消息数")
    ap.add_argument("--model", default=ld._LLM_MODEL)
    ap.add_argument("--out-prefix", default="batch")
    args = ap.parse_args()

    evs = ts.fetch_history(args.group, 0, 0, args.hist_file)
    if not evs:
        raise SystemExit("没拉到历史消息")
    clusters = ts.segment_history(evs)
    merge_map = json.loads(args.merge_map) if args.merge_map else None
    tasks = build_tasks(clusters, merge_map, args.min_msgs)
    print(f"窗口 {len(evs)} 条 → {len(clusters)} 簇 → 归并为 {len(tasks)} 个真任务")

    ts.SHARED.mkdir(parents=True, exist_ok=True)
    anon = RegexAnonymizer()
    summary = ["# 全量样张（A 验证·人工 merge-map 并簇 → 第②步子需求切分）", "",
               f"> 窗口 {len(evs)} 条原始消息 → segment_history {len(clusters)} 簇 "
               f"→ 人工并成 {len(tasks)} 个真任务 → 逐个跑 model={args.model}。", "",
               "| 任务 | 有效条数 | 子需求数 | 回退保真兜底 | 互斥重叠 | 未归属 | 文件 |",
               "|------|---------|---------|-------------|---------|--------|------|"]
    for k, (name, cluster) in enumerate(tasks, 1):
        n = _real_n(cluster)
        try:
            result, rows, meta = ld.llm_decompose(cluster, anon, model=args.model)
            acts = ld.enforce_rollback_purity(result, meta)
            md = ld.render(args.group, f"{k}·{name}", result, rows, acts)
            overlaps, missing = ld.audit_membership(result, len(rows))
            fn = f"{args.out_prefix}_{args.group[:8]}_t{k}_{name}.md"
            (ts.SHARED / fn).write_text(md, encoding="utf-8")
            nsub = len(result.get("subreqs", []))
            guard = f"触发{len(acts)}处" if acts else "无需"
            summary.append(f"| {k}·{name} | {n} | {nsub} | {guard} | "
                           f"{overlaps or '无✅'} | {missing or '无✅'} | {fn} |")
            print(f"  [{k}] {name}: {n}条 → {nsub}子需求 "
                  f"(兜底{guard}, 重叠{overlaps or '无'}, 未归属{missing or '无'})")
        except Exception as ex:  # 单任务失败不拖垮整批，fail-loud 记进表
            summary.append(f"| {k}·{name} | {n} | ❌ | — | — | — | 失败: {ex} |")
            print(f"  [{k}] {name}: ❌ {ex}")

    sfn = f"{args.out_prefix}_{args.group[:8]}_SUMMARY.md"
    (ts.SHARED / sfn).write_text("\n".join(summary), encoding="utf-8")
    print(f"汇总表：{ts.SHARED / sfn}")


if __name__ == "__main__":
    main()
