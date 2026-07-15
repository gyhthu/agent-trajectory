# -*- coding: utf-8 -*-
"""档A（确定性角色标注注入）验证 harness：在 2 份金标上跑 decompose，打印每子需求边界，
供与人工金标逐段对齐看「S5 的 G6 返工链有没有独立出来 / zym 原来分对的有没有被带歪」。

- chunk1_107：v4-pro（原诊断就是 v4-pro 在这 107 条上把 G5/G6/G7/G9 揉成 S5）→ 看是否拆开。
- zym（143原始/72有效 ≤150 单次路径）：deepseek + v4-pro → 防回归（人工金标=7 子需求）。

跑前后对比：git stash 出基线版 build_transcript 跑一遍存 *_base.log，再跑改后版存 *_A.log。
本脚本只跑「当前工作树版本」；前后由调用方切换代码控制。带429重试由 decompose 内部 retries 兜。
用法：python3 gold_decompose_cmp.py [--rounds N] [--tag base|A]
"""
import json, time, sys, os, argparse
sys.path.insert(0, os.path.expanduser("~/data_process/trajectory_extract"))
import llm_decompose as D
from llm_decompose import decompose_one_task, build_transcript
from regex_anonymizer import RegexAnonymizer

BASE = "/opt/shared/data/task-trajectory"
GOLDS = {
    "zym":     {"ev": f"{BASE}/ablation_jia/events_zym.json", "models": ["deepseek", "deepseek-v4-pro"]},
    "chunk1":  {"ev": f"{BASE}/events_chunk1_107.json",       "models": ["deepseek-v4-pro"]},
}


def fmt(events, model, anon, rounds):
    rows, meta = build_transcript(events, anon)
    n = len(meta)
    ntag = sum(1 for r in rows if any(t in r for t in ("[返工]", "[纠偏]", "[承接]")))

    def tw(idxs):
        idxs = [i for i in idxs if 1 <= i <= n]
        if not idxs:
            return "?"
        a, b = min(idxs), max(idxs)
        ta = time.strftime("%H:%M", time.localtime(meta[a - 1]["ts"]))
        tb = time.strftime("%H:%M", time.localtime(meta[b - 1]["ts"]))
        return f"#{a}-{b} [{ta}-{tb}]"

    out = [f"有效清单 {n} 条，带确定性标签 {ntag} 条"]
    for r in range(1, rounds + 1):
        t0 = time.time()
        try:
            res, _ = decompose_one_task(events, anon, model=model, terminal=False)
        except Exception as e:
            out.append(f"  [r{r}] 失败 {type(e).__name__}: {e}")
            continue
        subs = res.get("subreqs", [])
        out.append(f"  [r{r}] {len(subs)} 子需求 ({time.time() - t0:.0f}s)")
        for s in subs:
            mi = [D._as_seg_int(x) or 0 for x in s.get("member_idx", [])]
            title = s.get("title") or s.get("desc") or str({k: s[k] for k in s if k not in ('member_idx', 'edges')})[:40]
            out.append(f"     {s.get('id', '?'):8s} | {title[:30]:30s} | {len(mi):2d}条 {tw(mi):18s} | {s.get('status', '?')}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--tag", default="A")
    ap.add_argument("--golds", default="zym,chunk1")
    args = ap.parse_args()
    anon = RegexAnonymizer()
    log = [f"# 档A 验证 tag={args.tag} rounds={args.rounds}  {time.strftime('%Y-%m-%d %H:%M')}"]
    for g in args.golds.split(","):
        spec = GOLDS[g]
        events = json.load(open(spec["ev"], encoding="utf-8"))
        log.append(f"\n===== 金标 {g}（原始{len(events)}条）=====")
        for model in spec["models"]:
            log.append(f"\n----- {model} -----")
            block = fmt(events, model, anon, args.rounds)
            log.append(block)
            print(f"[done] {g}/{model}", flush=True)
    text = "\n".join(log) + "\nDONE\n"
    outp = f"{BASE}/gold_decompose_cmp_{args.tag}.log"
    open(outp, "w", encoding="utf-8").write(text)
    print(text)
    print(f"\n落盘: {outp}")


if __name__ == "__main__":
    main()
