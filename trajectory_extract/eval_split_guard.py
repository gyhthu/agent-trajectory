"""拆分守卫消融（LLM 交付物审计版）：baseline（无拆分）vs +拆分守卫，在两份人工 gold 上比分组。

四条件说明：原计划 baseline/仅缝合/仅拆分/双开，但实测**缝合守卫在这两个窄窗是 no-op**
（zym/weave 缝合 on/off 原子段逐字节相同，见 probe_stitch_onoff），且 temperature=0、llm_segment
只吃原子段 → baseline==仅缝合、仅拆分==双开，四条件塌成**两条有效条件**。故只跑有区别的两条。

  · gold_接zym（该并）：拆分守卫不能误伤——理想仍=1 任务。
  · gold_并发交织（该分）：拆分守卫应救回——理想拆成 A、B 两任务。
判分复用 eval_candidate_recall.score（读 result.tasks[].member_segs）。
两条件共用同一次 llm_segment 调用（baseline 产出 → 施加拆分守卫得 +split），省一半配额。

用法：python3 eval_split_guard.py --rounds 5
"""
from __future__ import annotations
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_candidate_recall as ec  # noqa: E402
import llm_segment as ls  # noqa: E402
from regex_anonymizer import RegexAnonymizer  # noqa: E402

WORK = ec.WORK


def run(wins, rounds):
    anon = RegexAnonymizer()
    report = {}
    for win in wins:
        evs, atoms = ec.build_atoms(win)
        print(f"\n===== {win} · {ec.WINDOWS[win]['desc']} =====")
        print(f"  原始 {len(evs)} 条 → 原子段 {len(atoms)} 个")
        report[win] = {"baseline": [], "split": []}
        for r in range(1, rounds + 1):
            t0 = time.time()
            base_result, _ = ls.llm_segment(atoms, anon, candidate_hints=False)
            b_ok, b_nt, b_det = ec.score(win, atoms, base_result, anon)
            split_result = ls.apply_split_guard(atoms, base_result, anon, enabled=True)
            s_ok, s_nt, s_det = ec.score(win, atoms, split_result, anon)
            report[win]["baseline"].append((b_ok, b_nt, b_det))
            report[win]["split"].append((s_ok, s_nt, s_det))
            print(f"  [r{r}] baseline {'✓' if b_ok else '✗'} 任务={b_nt} {b_det}")
            print(f"        +拆分   {'✓' if s_ok else '✗'} 任务={s_nt} {s_det}  ({time.time()-t0:.0f}s)")
    return report


def summarize(report):
    L = []
    L.append("\n========== 拆分守卫消融汇总（LLM 交付物审计版）==========")
    L.append("缝合守卫在这两窄窗经实测为 no-op（原子段逐字节相同），四条件塌成两条：baseline / +拆分")
    L.append(f"{'gold':8s} | {'测什么':18s} | {'baseline 判对':16s} | {'+拆分守卫 判对':16s}")
    L.append("-" * 78)
    for win in report:
        kind = "该并→1任务" if ec.WINDOWS[win]["kind"] == "merge" else "该分→2任务"
        b = report[win]["baseline"]
        s = report[win]["split"]
        bp = sum(1 for ok, _, _ in b if ok)
        sp = sum(1 for ok, _, _ in s if ok)
        bnt = [nt for _, nt, _ in b]
        snt = [nt for _, nt, _ in s]
        L.append(f"{win:8s} | {kind:18s} | {bp}/{len(b)} 任务数{bnt} | {sp}/{len(s)} 任务数{snt}")
    L.append("-" * 78)
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--wins", nargs="+", default=["zym", "weave"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = run(a.wins, a.rounds)
    summary = summarize(rep)
    print(summary)
    out = a.out or os.path.join("/opt/shared/data/task-trajectory/ablation_split_guard",
                                f"ablation_split_{int(time.time())}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"report": rep, "summary": summary}, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n落盘：{out}")


if __name__ == "__main__":
    main()
