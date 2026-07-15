#!/usr/bin/env python3
"""两级级联边界复核·原型（前后对比用，不入生产 llm_decompose）。

张耀明 0630 提议：在「保守复核」基础上**多调一次 LLM**，做成级联而非把细判据塞进一个胖 prompt。
  · 第一级 = 交付物复核（沿用 boundary_review_proto._REVIEW_SYS）：粗粒度，问「有没有第二个交付物」，
    默认 keep。已验：捡回 G8/G9 大话题切换，不动 G6。
  · 第二级 = 闭环复核（本文件新增 _LOOP_SYS）：**只在第一级 keep 下来的段上跑**，问更细一档——
    「段内有几个完整的『小目标→尝试→走偏/被纠→做对收口』闭环」。单闭环（哪怕内部反复）一律 keep。

承重判别量 = 完整闭环个数（比交付物细一档、又比档A 的无差别返工边界粗）：
  · G6[58-81] = 1 个闭环（回退保真：发现污染→回退→反复验证→保真做对）→ keep；
  · #44-89   = 3 个闭环（建通路 done → 回退保真 done → 全量铺开）→ 切 3 段。

评估两个失败方向：
  · 偏粗：第一级 keep 的 #44-89（揉 G5/G6/G7）→ 第二级应切成 3 段；
  · 过切：金标 G6[58-81] 单闭环 → 整条级联应 keep（档A 在这翻车）。
"""
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/data_process"))
from regex_anonymizer import RegexAnonymizer  # noqa: E402
from openai import OpenAI  # noqa: E402
from llm_decompose import build_transcript, _LLM_BASE, _LLM_KEY  # noqa: E402
from boundary_review_proto import _REVIEW_SYS, _MODEL  # noqa: E402  复用第一级 prompt

_EV = "/opt/shared/data/task-trajectory/events_chunk1_107.json"

_LOOP_SYS = """你是对话轨迹切分的【闭环复核员】。给你的这一段已经被判定为「同一个交付物/同一个大目标」。\
你只回答一个更细的问题：**这一段里有没有多个完整的「小目标闭环」被串在了一起？**

什么叫一个【完整的小目标闭环】= 一个明确的小目标被提出 → 尝试/执行（中间可以走偏、被纠、返工、反复验证）\
→ 被带到「做完/跑通/收口」的状态。当上一个小目标已经收口、紧接着开启**另一个明确不同的小目标**时，\
两者之间就是一个闭环边界。

例（同一个大目标"做切分"内部仍可能是多个闭环）：
  "搭通切分通路、跑出第一版" 收口 → "发现切碎了返工链、回退把因果保真" 收口 → "把保真版铺到全量" —— 3 个闭环。

**以下一律不算新闭环边界，必须留在同一个闭环里（最重要的纪律）：**
- 同一个小目标内部的澄清/追问/答疑、返工/纠偏/再次复现、换实现细节、被提醒后回退重做；
- 验证探针（hi/通没通/还报错吗）和它验证的那件事；
- 同一小目标下反复拉扯收窄——哪怕来回十几轮，**这仍是一个闭环**。

判据：**只有当「上一个小目标已经被带到做完/跑通/收口，且下一段明确在攻另一个不同的小目标」时，才切一刀。**
拿不准、读起来像同一个小目标的持续解决过程 → 一律 keep。**宁可少切一刀，绝不把一个小目标的返工链切碎。**

输出严格 JSON（不要 markdown、不要解释）：
{"split": true/false,
 "segments": [{"start_idx": <该子闭环起始消息#编号>, "goal": "<这个小目标闭环在做的事，≤16字>"}],
 "reason": "<一句话：为什么切/为什么不切>"}
不切时 segments 只含一个元素（start_idx=整段第一条编号）。切时按起始编号升序列出每个子闭环。"""


def _call(sys_prompt, seg_lines, span_lo):
    client = OpenAI(api_key=_LLM_KEY, base_url=_LLM_BASE)
    user = "段消息清单：\n" + "\n".join(seg_lines)
    last = None
    for _ in range(4):
        try:
            resp = client.chat.completions.create(
                model=_MODEL, temperature=0, max_tokens=1200,
                messages=[{"role": "system", "content": sys_prompt},
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


def _split_to_intervals(lo, hi, starts):
    """把起始编号列表 + 整段终点 → [(s,e), ...]。"""
    starts = sorted(set([lo] + [s for s in starts if lo <= s <= hi]))
    out = []
    for i, s in enumerate(starts):
        e = (starts[i + 1] - 1) if i + 1 < len(starts) else hi
        out.append((s, e))
    return out


def _cascade(rows, lo, hi):
    """整条级联：第一级交付物切分 → 对每个子段跑第二级闭环切分。返回最终区间+各段来源说明。"""
    s1_split, s1_starts, s1_goals, s1_reason = _call(_REVIEW_SYS, rows[lo - 1:hi], lo)
    s1_iv = _split_to_intervals(lo, hi, s1_starts) if s1_split else [(lo, hi)]
    final = []
    trace = [("S1", s1_split, s1_reason, s1_iv, s1_goals if s1_split else ["(整段)"])]
    for (a, b) in s1_iv:
        s2_split, s2_starts, s2_goals, s2_reason = _call(_LOOP_SYS, rows[a - 1:b], a)
        s2_iv = _split_to_intervals(a, b, s2_starts) if s2_split else [(a, b)]
        for j, (c, d) in enumerate(s2_iv):
            g = s2_goals[j] if s2_split and j < len(s2_goals) else "(单闭环)"
            final.append((c, d, g))
        trace.append((f"S2[{a}-{b}]", s2_split, s2_reason, s2_iv, s2_goals if s2_split else ["(单闭环)"]))
    return final, trace


def main():
    events = json.load(open(_EV, encoding="utf-8"))
    anon = RegexAnonymizer()
    rows, _ = build_transcript(events, anon)
    n = len(rows)
    print(f"# 两级级联边界复核·前后对比  model={_MODEL}  共{n}条\n")

    cases = [
        ("偏粗·r1-S4", 44, 101, "G5+G6+G7+G9", "S1拆出G9，S2把#44-89再拆G5/G6/G7"),
        ("偏粗·r1-S2", 9, 53, "G2+G3+G4+G5(+支线)", "应拆出多个小目标闭环"),
        ("过切红线·金标G6", 58, 81, "G6 单一返工链", "整条级联 keep（绝不切）"),
    ]
    for tag, lo, hi, lumped, expect in cases:
        final, trace = _cascade(rows, lo, hi)
        print(f"===== {tag}  原段 #{lo}-{hi}（{hi-lo+1}条，揉了 {lumped}）=====")
        print(f"  期望：{expect}")
        for label, sp, reason, iv, goals in trace:
            ivs = " ".join(f"#{a}-{b}" for a, b in iv)
            print(f"  [{label}] split={sp}  → {ivs}")
            print(f"        理由：{reason}")
        print(f"  ▶ 最终切成 {len(final)} 段：")
        for (a, b, g) in final:
            print(f"     └ #{a}-{b}  {g}")
        print()


if __name__ == "__main__":
    main()
