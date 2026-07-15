"""甲·候选召回消融：加 bge-m3「疑似同任务」候选提示 vs 不加，在两份人工 gold 上比分组质量。

为什么这么设计（张耀明 0629 第4条点破「没有好测试集」）：
  · 绝不拿 frozen_tasks 当真值——它是 LLM 分组自己的产出，拿它评 LLM 分组是循环论证。
  · 只用两份**人工 gold**，且各测一个**相反**的失败模式：
      - gold_接zym（06-23 01:28→03:14）：1 个任务被时间切成多原子段、还跨了两个 bot
        （claude 前半权限/登录 ⟷ codex 后半@排查/转义）。测「**该并**」——失败=过切。
      - gold_并发交织（06-29 00:15→03:00）：A(撤特化测波动) ∥ B(修汇报机制) 同标签同时间窗
        交织。测「**该分**」——失败=过并（06-27 词重叠重演）。
  · LLM 有波动 → 每个条件多跑 N 轮看分布，不单次定论；老实标注样本量小。
  · 打分直接读 result.tasks[].member_segs（1-based 原子段号），按内容关键词给原子段贴标签判定，
    不依赖任何 LLM 产出当真值。

用法：
  python3 eval_candidate_recall.py --rounds 5            # 两份 gold、baseline/treatment 各 5 轮
  python3 eval_candidate_recall.py --rounds 5 --only zym # 只跑接zym
环境：SEG_CANDIDATE_HINTS 由本脚本逐轮设定，无需手设。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import task_stitch as ts  # noqa: E402
import llm_segment as ls  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from regex_anonymizer import RegexAnonymizer  # noqa: E402

GROUP = "oc_53b8b620867a189d8dfe502865dfccc5"
WORK = "/opt/shared/data/task-trajectory/ablation_jia"


def _epoch(y, mo, d, h, mi):
    return int(time.mktime((y, mo, d, h, mi, 0, 0, 0, -1)))


# 两个 gold 窗口（CST）。窄切到 gold 本身那一摊，别把无关任务带进来污染判定。
WINDOWS = {
    "zym": {
        "desc": "接zym-bot进群（该并→理想1任务）",
        "since": _epoch(2026, 6, 23, 1, 28),
        "until": _epoch(2026, 6, 23, 3, 14),
        "kind": "merge",
    },
    "weave": {
        "desc": "并发交织 A∥B（该分→A、B不能焊一坨）",
        "since": _epoch(2026, 6, 29, 0, 15),
        "until": _epoch(2026, 6, 29, 3, 0),
        "kind": "split",
    },
}

# 并发交织 A/B 内容标签（判过并）。A=撤特化测波动，B=修汇报机制。
A_KW = ["撤特化", "特化", "波动", "全距", "σ", "标准差", "v4pro", "v4-pro", "V3", "并行各跑", "重跑"]
B_KW = ["汇报", "主动报", "主动汇报", "cron", "send.sh", "watcher", "session", "回收",
        "自唤醒", "唤醒", "通知", "跑完", "触发不了"]
# 接zym 前半(claude)/后半(codex)标记（判跨bot该并）
EARLY_KW = ["权限", "申请", "链接", "账号", "登录", "sdk", "复用", "发版", "secret", "env"]
LATE_KW = ["没反应", "反应", "转义", "乱码", "隔离", "排查", "bridge", "实例", "开头"]


def _label_atom(rows, anon, kws):
    txt = ls._atom_full_text(rows, anon)
    return sum(txt.count(k) for k in kws)


def build_atoms(win):
    """按窗口导出（一次）→建原子段（确定性，跨轮复用）。"""
    cache = os.path.join(WORK, f"events_{win}.json")
    if os.path.exists(cache):
        evs = json.load(open(cache, encoding="utf-8"))
    else:
        w = WINDOWS[win]
        evs = ts.fetch_history(GROUP, w["since"], w["until"])
        os.makedirs(WORK, exist_ok=True)
        json.dump(evs, open(cache, "w", encoding="utf-8"), ensure_ascii=False)
    atoms = ts.atomic_segments(evs, sim_fn=ts.build_reply_sim_fn())
    return evs, atoms


def score(win, atoms, result, anon):
    """回 (verdict_ok, n_tasks, detail)。"""
    tasks = result.get("tasks", [])
    n_tasks = len(tasks)
    members = [set(ls._seg_ints(t.get("member_segs", []))) for t in tasks]
    if WINDOWS[win]["kind"] == "merge":
        # 该并：gold 真值=这一摊是 1 个任务。判据以「分组成 1 个任务」为准（过切=失败）。
        # 早/晚关键词仅作信息展示（窗口内原子段少、codex后半关键词稀，不当硬门，免脆判）。
        early = {i + 1 for i, rows in enumerate(atoms)
                 if _label_atom(rows, anon, EARLY_KW) > _label_atom(rows, anon, LATE_KW)}
        late = {i + 1 for i, rows in enumerate(atoms)
                if _label_atom(rows, anon, LATE_KW) > _label_atom(rows, anon, EARLY_KW)}
        merged = any((m & early) and (m & late) for m in members)
        ok = (n_tasks == 1)
        return ok, n_tasks, f"(信息)跨bot两半同任务={merged} 早段{sorted(early)} 晚段{sorted(late)}"
    else:
        # 该分：A、B 不能落进同一任务。
        a = {i + 1 for i, rows in enumerate(atoms)
             if _label_atom(rows, anon, A_KW) > _label_atom(rows, anon, B_KW)}
        b = {i + 1 for i, rows in enumerate(atoms)
             if _label_atom(rows, anon, B_KW) > _label_atom(rows, anon, A_KW)}
        over_merged = any((m & a) and (m & b) for m in members)
        ok = bool(a) and bool(b) and not over_merged
        return ok, n_tasks, f"A段{sorted(a)} B段{sorted(b)} 过并(A,B同任务)={over_merged}"


def run(wins, rounds):
    anon = RegexAnonymizer()
    report = {}
    for win in wins:
        evs, atoms = build_atoms(win)
        print(f"\n===== {win} · {WINDOWS[win]['desc']} =====")
        print(f"  原始 {len(evs)} 条 → 原子段 {len(atoms)} 个")
        report[win] = {}
        for cond, flag in [("baseline", False), ("treatment", True)]:
            oks, ntasks_list, details = [], [], []
            for r in range(1, rounds + 1):
                t0 = time.time()
                result, _ = ls.llm_segment(atoms, anon, candidate_hints=flag)
                ok, nt, det = score(win, atoms, result, anon)
                oks.append(ok)
                ntasks_list.append(nt)
                details.append(det)
                tag = "✓" if ok else "✗"
                print(f"  [{cond:9s} r{r}] {tag} 任务数={nt} {det}  ({time.time()-t0:.0f}s)")
            passed = sum(oks)
            report[win][cond] = {
                "pass": passed, "rounds": rounds,
                "ntasks": ntasks_list, "details": details,
            }
            print(f"  → {cond}: 判对 {passed}/{rounds}  任务数{ntasks_list}")
    # 汇总表
    print("\n\n========== 甲·候选召回消融汇总 ==========")
    print(f"{'gold':10s} | {'测什么':22s} | {'baseline':14s} | {'treatment':14s}")
    print("-" * 70)
    for win in wins:
        b = report[win]["baseline"]
        t = report[win]["treatment"]
        kind = "该并(理想1任务)" if WINDOWS[win]["kind"] == "merge" else "该分(A≠B)"
        print(f"{win:10s} | {kind:22s} | 判对{b['pass']}/{b['rounds']} 每轮任务数{b['ntasks']!s:18s} "
              f"| 判对{t['pass']}/{t['rounds']} 每轮任务数{t['ntasks']}")
    out = os.path.join(WORK, f"ablation_jia_{time.strftime('%m%d_%H%M%S')}.json")
    os.makedirs(WORK, exist_ok=True)
    json.dump(report, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n明细落盘：{out}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--only", choices=["zym", "weave"], help="只跑其一")
    args = ap.parse_args()
    wins = [args.only] if args.only else ["zym", "weave"]
    run(wins, args.rounds)


if __name__ == "__main__":
    main()
