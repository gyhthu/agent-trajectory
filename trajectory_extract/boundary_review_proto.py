#!/usr/bin/env python3
"""保守边界复核·原型（前后对比用，不入生产 llm_decompose）。

设计：对【首切产出的每一段】单独跑一次二元复核——「这一段是不是把多个不同的
【实质交付目标】揉在了一起？」默认 keep（判 1 段），只有能明确点出段内存在
≥2 个不同交付物时才切，并给出每个新交付目标的起始消息编号。

为把「复核机制」从「首切方差」里隔离出来评估，本原型不真跑首切，而是喂两类受控输入：
  · 粗输入：r1 那两个超粗段（S2[9-53]揉G2-G5、S4[44-101]揉G5/G6/G7/G9）→ 看能否拆回金标；
  · 防过切输入：金标 G6[58-81] 24条返工链（绝不能切）→ 看是否默认 keep（档A 在这翻车）。

判别量是【交付物/目标身份】，比首切的返工边界更粗——澄清问答/追问/返工/纠偏/验证探针
一律不算新边界，只有「在建另一个不同的东西」才算。
"""
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/data_process"))  # regex_anonymizer 在父目录
from regex_anonymizer import RegexAnonymizer  # noqa: E402
from openai import OpenAI  # noqa: E402
from llm_decompose import build_transcript, _LLM_BASE, _LLM_KEY  # noqa: E402

_MODEL = os.environ.get("REVIEW_MODEL", "deepseek-v4-pro")
_EV = "/opt/shared/data/task-trajectory/events_chunk1_107.json"

_REVIEW_SYS = """你是对话轨迹切分的【边界复核员】。给你一个已经切好的「子需求段」的消息清单，\
你只回答一个问题：**这一段是不是把多个【不同的实质交付目标】错误地揉在了一起？**

什么叫「不同的实质交付目标」= 在建/在交付【另一个不同的东西】：换了要做的事、换了产物、\
换了交付对象。例：「搭通 LLM 切分通路」和「把切分铺到全量数据」是两个目标；\
「写调研文档」和「修某个 bug」是两个目标。

**以下统统不算新目标边界，必须留在同一段里（这是本复核最重要的纪律）：**
- 对同一个目标的【澄清、追问、答疑、问机制】——哪怕问了十几轮、隔了很久；
- 对同一个目标的【返工、纠偏、再次复现、换实现细节、被提醒后回退重做】——\
这是同一件事的解决过程，是最有价值的因果链，**切碎它=污染训练数据**；
- 【验证探针】（hi/通没通/还报错吗）和它验证的那件事；
- 同一目标下的来回拉扯、反复收窄。

判据：**只有当段内明确出现「上一个东西已经做完/搁下，现在开始做另一个不同的东西」时，才算一个新边界。**
拿不准、感觉只是同一件事的延续或返工 → 一律判不切（keep）。**宁可粗一档，绝不把一个目标的返工链切碎。**

输出严格 JSON（不要 markdown、不要解释）：
{"split": true/false,
 "segments": [{"start_idx": <该子段起始消息#编号>, "goal": "<这段在交付的那个不同的东西，≤16字>"}],
 "reason": "<一句话：为什么切/为什么不切>"}
不切时 segments 只含一个元素（start_idx=整段第一条的编号）。切时按起始编号升序列出每个子段。"""


def _review(seg_lines, span_lo):
    """对一段消息清单跑复核，返回 (split:bool, starts:[int], goals:[str], reason:str)。"""
    client = OpenAI(api_key=_LLM_KEY, base_url=_LLM_BASE)
    user = "子需求段消息清单：\n" + "\n".join(seg_lines)
    last = None
    for _ in range(4):
        try:
            resp = client.chat.completions.create(
                model=_MODEL, temperature=0, max_tokens=1200,
                messages=[{"role": "system", "content": _REVIEW_SYS},
                          {"role": "user", "content": user}])
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```", 2)[1].lstrip("json").strip()
            d = json.loads(raw)
            segs = d.get("segments") or [{"start_idx": span_lo, "goal": "?"}]
            starts = [int(re.search(r"\d+", str(s["start_idx"])).group()) for s in segs]
            goals = [s.get("goal", "?") for s in segs]
            return bool(d.get("split")), starts, goals, d.get("reason", "")
        except Exception as ex:
            last = ex
            m = str(ex)
            if "429" in m or "rate" in m.lower():
                s = re.search(r"in (\d+) seconds", m)
                time.sleep(min((int(s.group(1)) + 5) if s else 35, 60))
            continue
    raise last


def main():
    events = json.load(open(_EV, encoding="utf-8"))
    anon = RegexAnonymizer()
    rows, _ = build_transcript(events, anon)  # rows[i] 对应 #(i+1)
    n = len(rows)
    print(f"# 保守边界复核·前后对比  model={_MODEL}  共{n}条\n")

    # 受控输入：(标签, 起, 止, 该段揉了金标哪几摊, 期望)
    cases = [
        ("粗·r1-S2", 9, 53, "G2+G3+G4+G5(+支线)", "应拆出多个目标"),
        ("粗·r1-S4", 44, 101, "G5+G6+G7+G9", "应拆出多个目标(含G6独立)"),
        ("防过切·金标G6", 58, 81, "G6 单一返工链", "应 keep 不切"),
    ]
    for tag, lo, hi, lumped, expect in cases:
        seg = rows[lo - 1:hi]
        split, starts, goals, reason = _review(seg, lo)
        print(f"===== {tag}  原段 #{lo}-{hi}（{hi-lo+1}条，揉了 {lumped}）=====")
        print(f"  期望：{expect}")
        print(f"  复核：split={split}  切成 {len(starts)} 段")
        for i, (st, g) in enumerate(zip(starts, goals)):
            end = (starts[i + 1] - 1) if i + 1 < len(starts) else hi
            print(f"     └ #{st}-{end}  {g}")
        print(f"  理由：{reason}\n")


if __name__ == "__main__":
    main()
