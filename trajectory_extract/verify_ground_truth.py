#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ground truth 核对闸（张耀明 2026-07-07 要）。

extract_ground_truth.py 抽的只是「anchor 后同 thread 第一条 bot 实质回复」= 生料，
未必是「针对该纠正的正面改对」——bot 常先认错、然后转去干下一件活（idx31 就抽歪了）。
本闸让 LLM 逐条判：这条候选回复到底是不是**针对这条纠正、正面把错改对了**。
只有过闸(kind=positive_fix)的才当干净 ground truth 供 route-B 用。

判据分四档（对应张耀明拍的病根「没查就臆测」之外的抽取质量问题）：
  positive_fix     —— 真针对该纠正、正面重答/改对了（★唯一过闸）
  only_acknowledged—— 只认错/道歉，没正面把内容改对
  turned_to_other  —— 认错后转去做下一件别的事，答的不是这条纠正
  still_wrong      —— 仍在犯同类错 / 没纠过来
"""
from __future__ import annotations
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import principle_distill as pd

BASE = "/opt/shared/data/task-trajectory"
MODEL = os.environ.get("VERIFY_MODEL", "deepseek-v3.2")

_SYS = """你在核对训练数据的 ground truth。背景：一个飞书群 bot 当年答错被人纠正，我们想拿"它被纠正后改对的那版回复"当标答。
但抽取脚本只是机械地取了"纠正消息之后该 bot 的第一条实质回复"，这条未必真是针对该纠正的正面改对——
bot 常常先认个错、然后就转头去干下一件别的活了。

给你三样东西：①当年的原始指令/问题 ②纠正者指出的错(纠正原话) ③抽到的这条候选回复。
判断这条候选回复属于以下哪一档，只输出 JSON：
- "positive_fix"：真的针对这条纠正，正面把错改对了 / 重新正确回答了原问题。
- "only_acknowledged"：只是认错、道歉、表态会改，但没有正面给出改对后的内容。
- "turned_to_other"：认错后转去做/回答另一件事，回复的实质内容不是针对这条纠正。
- "still_wrong"：仍在犯同类错误，没纠过来。

判"实质内容对不对"要看它有没有正面回应纠正指出的那个点，别被"我明白了/收到/对不起"这类话术带偏。
输出格式：{"kind":"positive_fix|only_acknowledged|turned_to_other|still_wrong","reason":"一句话依据"}"""


def _judge(client, cand):
    user = (
        f"①原始指令/问题：\n{cand.get('instruction','')}\n\n"
        f"②纠正者指出的错（纠正原话）：\n{cand.get('correction_what','')}\n\n"
        f"③抽到的候选回复：\n{(cand.get('gt_text') or '')[:2000]}\n\n"
        "判断属于哪一档，只输出 JSON。"
    )
    raw = pd._chat_with_retry(
        client, MODEL,
        [{"role": "system", "content": _SYS}, {"role": "user", "content": user}],
    ).choices[0].message.content
    s = raw.strip()
    if "```" in s:
        s = s.split("```")[1].lstrip("json").strip() if "```json" in s else s.split("```")[1].strip()
    try:
        j = json.loads(s[s.find("{"):s.rfind("}") + 1])
    except Exception as e:
        return {"kind": "parse_error", "reason": f"{e}: {raw[:120]}"}
    return {"kind": j.get("kind", "?"), "reason": j.get("reason", "")}


def _work(cand):
    cli = pd._client()
    v = _judge(cli, cand)
    return {**{k: cand[k] for k in ("idx", "corr_i", "anchor", "gt_by")},
            "correction_what": cand.get("correction_what"),
            "gt_text": cand.get("gt_text"),
            "kind": v["kind"], "reason": v["reason"],
            "passed": v["kind"] == "positive_fix"}


def main():
    cands = json.load(open(f"{BASE}/qa_ground_truth_candidates.json"))
    todo = [c for c in cands if c.get("gt_text")]
    skipped = [c for c in cands if not c.get("gt_text")]
    print(f"候选 {len(cands)}，有生料待核 {len(todo)}，抽取阶段就缺料 {len(skipped)}", flush=True)

    out = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(_work, c): c for c in todo}
        for n, f in enumerate(as_completed(futs), 1):
            try:
                out.append(f.result())
            except Exception as e:
                c = futs[f]
                out.append({"idx": c["idx"], "corr_i": c["corr_i"], "kind": "error",
                            "reason": str(e)[:120], "passed": False,
                            "correction_what": c.get("correction_what"), "gt_text": c.get("gt_text")})
            if n % 10 == 0:
                print(f"  ...{n}/{len(todo)}", flush=True)
    out.sort(key=lambda r: (r["idx"], r["corr_i"]))

    from collections import Counter
    dist = Counter(r["kind"] for r in out)
    npass = sum(1 for r in out if r["passed"])
    dst = f"{BASE}/qa_ground_truth_verified.json"
    json.dump({"n_candidate": len(cands), "n_have_material": len(todo),
               "n_missing_material": len(skipped), "n_pass": npass,
               "kind_dist": dict(dist), "verified": out}, open(dst, "w"),
              ensure_ascii=False, indent=2)
    print(f"\n核完：{npass}/{len(todo)} 过闸(positive_fix) → 干净 ground truth", flush=True)
    print(f"分档：{dict(dist)}", flush=True)
    print(f"缺料(抽取阶段就没料) {len(skipped)} → 这些 qa 无 ground truth", flush=True)
    print(f"落盘 → {dst}", flush=True)


if __name__ == "__main__":
    main()
