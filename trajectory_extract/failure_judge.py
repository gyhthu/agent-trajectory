"""failure-judge —— 独立判「这个子需求到最后到底有没有做对」。

为什么单独一层（张耀明 2026-06-26 拍板的降级版主线）：
  llm_decompose 的 status（一遍过/有返工/被纠偏）口径**默认任务最终做对了**——
  它切的是「小目标→走偏→被纠→做对」的闭环，天然不含「最终根本没做对(failure)」这类。
  但训练里「没达成」的样本绝不能当 SFT 正样本，所以必须有一层**只判最终结果**的审计，
  把 llm_decompose 乐观假设下其实没做对的段揪出来。这就是 failure-judge。

口径与 llm_decompose 正交：
  - llm_decompose.status = **过程**维度（有没有返工/被纠偏）；
  - failure-judge.verdict = **结果**维度（最后满足了没）。
  组装器 map_outcome 把两者合：judge=failure 覆盖乐观假设；status 缺失时用 judge 补。

复用 llm_decompose 的 LLM 基建（litellm 代理，deepseek 不出本机）+ build_transcript（脱敏+编号），不重写。

用法（单段判，多用于组装器内调用；也可 CLI 自测）：
  from failure_judge import judge_subreq
  v = judge_subreq(subreq, meta, anon)   # -> {"verdict","reason","evidence_idx"}
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm_decompose as ld  # noqa: E402  复用 _LLM_* / build_transcript

from openai import OpenAI  # noqa: E402

_SYS_PROMPT = """你是对话结果审计员。给你一个「子需求」内按时间排序的多方对话（已脱敏、带#编号），\
只判一件事：到这段对话结束时，用户提出的这个需求到底有没有被满足。

**只判最终结果，不评过程好坏**（中间走过多少弯路、返工几次都不影响判定，只看最后落点）。

判据：
- success：用户最后明确认可（"行/可以/对了/好的/搞定/通了/没问题"），\
或 bot 给出实质交付后用户无异议、话题自然收尾。
- failure：用户明确表达仍未满足且本段内**没有**后续有效修复（"还是不行/不对/没解决/又错了"之后对话中断或转走）；\
或 bot 明确放弃/说做不出来；或需求悬空（提了之后压根没有任何实质交付就结束）。
- unknown：信息不足，看不出最终成没成（如对话被截断、最后一条是 bot 交付但没有用户回应也无法从内容判断对错）。

**保守铁律**：拿不准就判 unknown，**绝不把"没有明确成功信号"的判成 success**——\
误判 success 会把没做对的样本污染进训练正样本，这是最严重的错误。\
没有用户明确认可、也无法从交付内容本身确认做对的，一律 unknown，不许脑补成功。

严格输出 JSON：{"verdict":"success|failure|unknown","reason":"≤30字依据","evidence_idx":[关键消息#]}。\
不要 markdown 包裹、不要解释。"""


def judge_outcome(evs: list, anon, model: str = ld._LLM_MODEL) -> dict:
    """对一段消息（已是某子需求的成员消息，按 ts 序）判最终结果。
    evs: [{ts,role,name,text,msg_id,...}]；anon: RegexAnonymizer（脱敏在喂模型前完成）。
    返回 {"verdict": success|failure|unknown, "reason": str, "evidence_idx": [int]}。
    evidence_idx 是**本段局部 #编号**（build_transcript 重排，仅供可读核对，不对回全局 member_idx）。
    异常/解析失败 → unknown（fail-soft 到保守值，但 reason 写明原因，不静默吞）。"""
    if not evs:
        return {"verdict": "unknown", "reason": "空段无消息", "evidence_idx": []}
    rows, _meta = ld.build_transcript(evs, anon)
    if not rows:
        return {"verdict": "unknown", "reason": "脱敏后无有效消息", "evidence_idx": []}
    transcript = "\n".join(rows)
    client = OpenAI(api_key=ld._LLM_KEY, base_url=ld._LLM_BASE)
    try:
        resp = client.chat.completions.create(
            model=model, temperature=0,
            messages=[{"role": "system", "content": _SYS_PROMPT},
                      {"role": "user", "content": "子需求消息：\n" + transcript}],
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1].lstrip("json").strip()
        obj = json.loads(raw)
    except Exception as ex:  # noqa: BLE001  判不出按保守值，但 reason 暴露原因
        return {"verdict": "unknown", "reason": f"judge 调用/解析失败: {ex}", "evidence_idx": []}
    v = obj.get("verdict")
    if v not in ("success", "failure", "unknown"):
        return {"verdict": "unknown", "reason": f"verdict 非法({v})", "evidence_idx": []}
    return {"verdict": v, "reason": str(obj.get("reason", ""))[:60],
            "evidence_idx": [int(x) for x in obj.get("evidence_idx", []) if str(x).isdigit()]}


def judge_subreq(subreq: dict, meta: list, anon, model: str = ld._LLM_MODEL) -> dict:
    """从 llm_decompose 的 subreq + meta 取出本段消息再判。
    meta 为 1-based member_idx 对应的消息 meta（llm_decompose 出口）。"""
    members = sorted(int(x) for x in subreq.get("member_idx", []))
    evs = [meta[i - 1] for i in members if 1 <= i <= len(meta)]
    return judge_outcome(evs, anon, model)


def main():
    """CLI 自测：跑一个真实任务的全部子需求，逐段判最终结果。
    复用 llm_decompose 的加载/切分/分段，端到端串一遍。"""
    import argparse
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import task_stitch as ts  # noqa: E402
    from regex_anonymizer import RegexAnonymizer  # noqa: E402

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", required=True)
    ap.add_argument("--task-idx", type=int, required=True)
    ap.add_argument("--hist-file")
    ap.add_argument("--since", type=int)
    ap.add_argument("--until", type=int)
    ap.add_argument("--model", default=ld._LLM_MODEL)
    args = ap.parse_args()

    evs = ts.fetch_history(args.group, args.since or 0, args.until or 0, args.hist_file)
    clusters = ts.segment_history(evs)
    cluster = clusters[args.task_idx - 1]
    anon = RegexAnonymizer()
    result, rows, meta = ld.llm_decompose(cluster, anon, model=args.model)
    ld.enforce_rollback_purity(result, meta)
    print(f"任务{args.task_idx}：{len(result.get('subreqs', []))} 个子需求，逐段判最终结果：")
    for s in result.get("subreqs", []):
        v = judge_subreq(s, meta, anon, model=args.model)
        print(f"  {s.get('id')} [{s.get('title')}] status={s.get('status')} "
              f"→ judge={v['verdict']}（{v['reason']}）")


if __name__ == "__main__":
    main()
