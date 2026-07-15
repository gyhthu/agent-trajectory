#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""抽「纠正后 bot 改对的那版回复」当 route-B ground truth（张耀明 2026-07-07 拍：就用它当标答）。

做法：拿每条 correction 的 anchor(纠正那条消息的 msg_id)，在全量事件时间序里定位，
往后找**同 thread 的第一条 bot 实质回复**（跳过「👀 已收到」占位）= 候选 ground truth。
注意：这里只抽候选，**不保证 bot 真改对了**——有的只认错没重答/还错着。真改对与否交下游
judge/人工核（本脚本只落候选 + 供核的原文）。先跑 qa 集。
"""
import json, re
import runtime_lane as rl

BASE = "/opt/shared/data/task-trajectory"
PLACEHOLDER = re.compile(r"^\s*(👀|✅|🕒|🔖|⚠️|收到|已收到|已入队|本群会话上下文)")  # 占位/回执/系统提示


def _is_noise(txt):
    if PLACEHOLDER.match(txt):
        return True
    # 卡片 JSON payload（发图/发卡的原始结构），非文本回复
    if txt.startswith("{") and ('"elements"' in txt or '"tag"' in txt or '"image_key"' in txt):
        return True
    return False


def main():
    rw = json.load(open(f"{BASE}/rewrite_thread_result.json"))["rewritten"]
    cls = {r["idx"]: r for r in (json.loads(l) for l in open(f"{BASE}/request_type_2class.jsonl") if l.strip())}
    qa_idxs = sorted(i for i, r in cls.items() if r.get("class") == "qa")

    evs = rl.load_events()
    pos = {e.get("msg_id"): k for k, e in enumerate(evs)}

    out = []
    for idx in qa_idxs:
        e = rw[idx]
        for ci, c in enumerate(e.get("corrections", [])):
            anchor = c.get("anchor")
            i = pos.get(anchor)
            rec = {"idx": idx, "corr_i": ci, "anchor": anchor,
                   "correction_what": c.get("what"), "corrector": c.get("corrector"),
                   "instruction": e["instruction_to_fix_text"][:100],
                   "gt_msg_id": None, "gt_text": None, "gt_by": None, "note": ""}
            if i is None:
                rec["note"] = "anchor 不在事件序列(可能06-22前老纠正,源追不回)"
                out.append(rec); continue
            thr = evs[i].get("thread_id")
            found = None
            for j in range(i + 1, min(i + 40, len(evs))):  # anchor 后 40 条窗内
                nx = evs[j]
                if nx.get("role") != "bot":
                    continue
                if thr and nx.get("thread_id") and nx.get("thread_id") != thr:
                    continue
                txt = (nx.get("text") or "").strip()
                if not txt or _is_noise(txt) or len(txt) < 20:
                    continue
                found = nx; break
            if found:
                rec["gt_msg_id"] = found.get("msg_id")
                rec["gt_text"] = found.get("text")
                rec["gt_by"] = found.get("name")
            else:
                rec["note"] = "anchor 后 40 条内没找到同 thread 的 bot 实质回复"
            out.append(rec)

    dst = f"{BASE}/qa_ground_truth_candidates.json"
    json.dump(out, open(dst, "w"), ensure_ascii=False, indent=2)
    got = sum(1 for r in out if r["gt_text"])
    print(f"qa 纠正条目 {len(out)}，抽到候选 ground truth {got}，缺 {len(out)-got} → {dst}")
    for r in out:
        head = (r["gt_text"] or r["note"])[:60].replace("\n", " ")
        print(f"  idx{r['idx']}#{r['corr_i']} [{'GT' if r['gt_text'] else '--'}] {head}")


if __name__ == "__main__":
    main()
