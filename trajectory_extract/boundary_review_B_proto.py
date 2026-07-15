#!/usr/bin/env python3
"""B·任务态转移标记 + 加强兜底·原型（前后对比用，不入生产）。

张耀明 0630 拍：走 B + 加强兜底。架构 = 高召回提议层 + 投票否决层。
  · 提议层（高召回，激进）：扫「一件事收口→紧接着另一件事开始」的所有候选边界，宁多勿漏。
  · 否决层（高精度，保守 + self-consistency 投票）：对每个候选边界，反问「刀后那段是真·新小目标，
    还是还在返工/澄清/迭代刀前那个目标？」。返工=否决(keep)。**同一判断投 N 票，平票/少数票切→默认 keep。**
    投票直接打掉上轮实测的「单次边界判断 2/3 跳变」方差，且偏置永远倒向「不切」——这是加强兜底的硬结构。

承重测试（chunk1，对齐金标）：
  · 偏粗·#44-89（揉 G5/G6/G7）→ 应切 3 段（边界在 ~58、~82）；
  · 过切红线·G6[58-81] 单返工链 → 内部一刀不许下（提议层可提，否决层必须全否）。
"""
import json
import os
import re
import sys
import time
import argparse
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/data_process"))
from regex_anonymizer import RegexAnonymizer  # noqa: E402
from openai import OpenAI  # noqa: E402
from llm_decompose import build_transcript, _LLM_BASE, _LLM_KEY  # noqa: E402
from boundary_review_proto import _MODEL  # noqa: E402
import task_stitch as ts  # noqa: E402

_EV = "/opt/shared/data/task-trajectory/events_chunk1_107.json"
_VOTES = 3  # 否决层投票数（奇数；平票偏 keep）

_PROPOSE_SYS = """你是对话轨迹切分的【边界提议员】。任务：在这段对话里，把所有「前一件事告一段落、\
紧接着开始另一件事」的位置都找出来——**宁多勿漏，高召回**。

只要读起来像是「前面那个小目标已经做完/跑通/收口/告一段落，后面在攻一个明显不同的新小目标」，\
就提议一个候选边界。你**不需要**纠结它到底算不算真边界——后面有专门的复核员会逐个否决误报。\
你唯一的任务是别漏掉任何可能的任务态转移点。

线索（出现这些往往意味着一件事收口、下一件事开始）：
- 收口信号：跑通了/done/搞定/提交了/这版可以了/先这样/那接下来；
- 转向信号：现在来看…/接下来…/另外…/换个方向/开始做另一个东西，尤其伴随**对象/产物变了**。

输出严格 JSON（不要 markdown）：
{"boundaries": [{"start_idx": <新小目标起始消息#编号>, "hint": "<前面收口了什么、后面要开始什么，≤20字>"}]}
没有任何候选就给空列表。按编号升序。整段第一条不要当边界。"""

_VETO_SYS = """你是对话轨迹切分的【边界否决员】，纪律是**保返工链、宁可不切**。给你「刀前一小段」和「刀后一小段」，\
你只回答一个二元问题：**刀后这段，是在攻一个全新的小目标，还是还在返工/澄清/迭代刀前那个小目标？**

判「真·新小目标」（→放行切）：刀前那个小目标已经被带到做完/跑通/收口，刀后明确在攻**另一件不同的事**\
（目标变了、产物变了）。
判「同一小目标」（→否决 keep，这是默认倾向）：
- 刀后是对刀前那件事的追问/澄清/答疑/补充判据；
- 刀后是返工/纠偏/再次复现/被提醒后回退重做**同一个目标**；
- 刀后是验证探针（通没通/还报错吗）或对同一产物换实现细节；
- 读起来像同一件事还在持续解决——哪怕来回很多轮。

**拿不准就否决。** 把一个小目标的返工链切碎，比漏切一刀严重得多。

输出严格 JSON（不要 markdown、不要解释）：{"new_goal": true/false, "reason": "<一句话>"}"""


def _client():
    return OpenAI(api_key=_LLM_KEY, base_url=_LLM_BASE)


def _chat(sys_prompt, user, max_tokens=900, model=None):
    client = _client()
    last = None
    for _ in range(4):
        try:
            resp = client.chat.completions.create(
                model=model or _MODEL, temperature=0, max_tokens=max_tokens,
                messages=[{"role": "system", "content": sys_prompt},
                          {"role": "user", "content": user}])
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```", 2)[1].lstrip("json").strip()
            return json.loads(raw)
        except Exception as ex:
            last = ex
            m = str(ex)
            if "429" in m or "rate" in m.lower():
                s = re.search(r"in (\d+) seconds", m)
                time.sleep(min((int(s.group(1)) + 5) if s else 35, 60))
            continue
    raise last


def _propose(rows, lo, hi, model=None):
    user = "段消息清单：\n" + "\n".join(rows[lo - 1:hi])
    d = _chat(_PROPOSE_SYS, user, max_tokens=1200, model=model)
    out = []
    for b in d.get("boundaries", []):
        try:
            idx = int(re.search(r"\d+", str(b["start_idx"])).group())
        except Exception:
            continue
        if lo < idx <= hi:
            out.append((idx, b.get("hint", "")))
    return out


def _veto_vote(rows, lo, hi, cut, votes=_VOTES, model=None):
    """对候选切点 cut 投 votes 票。返回 (放行?, 票面, 各票理由)。平票/少数→keep。"""
    before = "\n".join(rows[lo - 1:cut - 1])      # [lo, cut-1]
    after = "\n".join(rows[cut - 1:hi])           # [cut, hi]
    user = f"刀前一段（#{lo}-{cut-1}）：\n{before}\n\n刀后一段（#{cut}-{hi}）：\n{after}"
    vote_values, reasons = [], []
    for _ in range(votes):
        d = _chat(_VETO_SYS, user, max_tokens=300, model=model)
        v = bool(d.get("new_goal"))
        vote_values.append(v)
        reasons.append(("放行" if v else "否决") + ":" + d.get("reason", ""))
    yes = sum(vote_values)
    passed = yes > len(vote_values) // 2          # 必须多数票才放行；平票/少数→keep
    return passed, f"{yes}/{len(vote_values)}", reasons


def _run_case(rows, lo, hi, votes=_VOTES, model=None):
    proposed = _propose(rows, lo, hi, model=model)
    confirmed = []  # 通过否决的切点
    trace = []
    for (cut, hint) in proposed:
        passed, tally, reasons = _veto_vote(rows, lo, hi, cut, votes=votes, model=model)
        trace.append((cut, hint, passed, tally, reasons))
        if passed:
            confirmed.append(cut)
    # 切点 → 区间
    starts = sorted(set([lo] + confirmed))
    iv = []
    for i, s in enumerate(starts):
        e = (starts[i + 1] - 1) if i + 1 < len(starts) else hi
        iv.append((s, e))
    return proposed, trace, iv


def _rows_from_events(events):
    anon = RegexAnonymizer()
    rows, _ = build_transcript(events, anon)
    return rows


def _rows_from_pool(pool_file, line_no=None, t0_msg_id=None, include_t0=True):
    rec = ts.load_pool_record(pool_file, line_no=line_no, t0_msg_id=t0_msg_id)
    events = ts.pool_record_events(rec, include_t0=include_t0)
    rows = []
    for i, e in enumerate(events, 1):
        role = "用户" if e.get("role") == "user" else ("bot" if e.get("role") == "bot" else e.get("role", "?"))
        parent = f" ↩{e.get('parent_id')}" if e.get("parent_id") else ""
        text = ts.gist(e.get("text") or "", 160)
        rows.append(f"#{i} [{role}/{e.get('name') or '?'}]{parent} {text}")
    return rows, rec


def run_boundary_review(rows, lo, hi, votes=_VOTES, model=None):
    proposed, trace, intervals = _run_case(rows, lo, hi, votes=votes, model=model)
    return {
        "lo": lo,
        "hi": hi,
        "votes": votes,
        "model": model or _MODEL,
        "proposed": [{"cut": cut, "hint": hint} for cut, hint in proposed],
        "trace": [{
            "cut": cut,
            "hint": hint,
            "passed": passed,
            "tally": tally,
            "reasons": reasons,
        } for cut, hint, passed, tally, reasons in trace],
        "intervals": [{"lo": a, "hi": b} for a, b in intervals],
    }


def _print_legacy_demo(rows):
    print(f"# B·任务态转移+投票否决·前后对比  model={_MODEL}  votes={_VOTES}\n")

    cases = [
        ("偏粗·#44-89", 44, 89, "G5[44-57]/G6[58-81]/G7[82-89]", "切3段，边界~58/~82"),
        ("过切红线·G6", 58, 81, "G6 单返工链", "内部一刀不下，最终1段"),
    ]
    for tag, lo, hi, gold, expect in cases:
        proposed, trace, iv = _run_case(rows, lo, hi)
        print(f"===== {tag}  #{lo}-{hi}（{hi-lo+1}条）=====")
        print(f"  金标={gold}  期望：{expect}")
        print(f"  提议层捞出 {len(proposed)} 个候选边界：")
        for (cut, hint, passed, tally, reasons) in trace:
            print(f"   · #{cut} [{hint}]  投票 {tally} → {'✂放行' if passed else '🛡否决'}")
            for r in reasons:
                print(f"        {r}")
        ivs = " ".join(f"#{a}-{b}" for a, b in iv)
        print(f"  ▶ 最终切成 {len(iv)} 段：{ivs}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events-json", default=_EV, help="事件 JSON 数组；无 pool 参数时使用")
    ap.add_argument("--pool-file", help="user_corrections_pool*.jsonl")
    ap.add_argument("--pool-line", type=int, help="pool 1基行号")
    ap.add_argument("--pool-t0-msg-id", help="按 t0.msg_id 取 pool 记录")
    ap.add_argument("--pool-no-t0", action="store_true", help="pool 模式不加入 T0")
    ap.add_argument("--lo", type=int, help="起始编号(1基)")
    ap.add_argument("--hi", type=int, help="结束编号(含)")
    ap.add_argument("--votes", type=int, default=_VOTES)
    ap.add_argument("--model", default=_MODEL)
    ap.add_argument("--out-json", help="结构化结果输出路径")
    args = ap.parse_args()

    if args.votes < 1:
        raise SystemExit("--votes must be >= 1")

    pool_meta = None
    if args.pool_file:
        rows, pool_meta = _rows_from_pool(
            args.pool_file,
            line_no=args.pool_line,
            t0_msg_id=args.pool_t0_msg_id,
            include_t0=not args.pool_no_t0,
        )
    else:
        events = json.load(open(args.events_json, encoding="utf-8"))
        rows = _rows_from_events(events)

    if args.lo is None and args.hi is None and not args.out_json and not args.pool_file:
        _print_legacy_demo(rows)
        return

    lo = args.lo or 1
    hi = args.hi or len(rows)
    if not (1 <= lo <= hi <= len(rows)):
        raise SystemExit(f"invalid --lo/--hi for {len(rows)} rows: {lo}-{hi}")

    result = run_boundary_review(rows, lo, hi, votes=args.votes, model=args.model)
    if pool_meta:
        result["pool_line"] = pool_meta.get("_pool_line")
        result["t0_msg_id"] = (pool_meta.get("t0") or {}).get("msg_id") or ""

    if args.out_json:
        out = os.path.abspath(args.out_json)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"文件：{out}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
