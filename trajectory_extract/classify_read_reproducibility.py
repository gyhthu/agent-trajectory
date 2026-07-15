#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exec 条目「读取目标可复现性」判定（张耀明 2026-07-07 反问触发）。

背景：instruction 判成 exec 只说明"要动手"，不等于沙箱能忠实复现。
沙箱能不能还原，取决于这条 exec **要读什么历史** + **那份料还原不还原得到 t0 态**：

  读取目标分类(LLM 判) ──► 可复现性裁决(确定性规则，吃 snapshot.code_refs join 状态)
  - B_context : 要读的是"前文/自己上一轮产出/群记忆" → 不是磁盘 read，是上下文。
                料就在 snapshot.context / transcript 里，直接注入即可 → reproducible。
  - A_file    : 要读 t0 时的仓内文件/代码/盘上产物 → 须把读取目标还原到 t0 态。
                * A1 git-tracked：靠 code_refs 的 commit join；至少一个仓 status=ok → reproducible。
                * A2 未版本化工作/shared 文件：git 救不了，须另核"mtime<t0 且之后未改"(本脚本标 needs_target_freeze_check)。
                两仓全 join 不到 → unreplayable(环境缺料，非能力失败)。
  - C_live    : 要读活体外部态(飞书云文档/多维表/live API) → 自 t0 起已漂移，沙箱复现不了 → unreplayable。
  - none_exec : 不读历史，纯去跑/写(跑测试、发消息、改文件) → 可复现性由**执行环境**定，不由读取目标定 → exec_env(另论)。

铁律(与 20 条 base 墙教训一脉)：读取目标还原不到 t0 态的，fail-loud 标 unreplayable，
**宁可少测，不拿缺料结果冒充**。绝不硬把 A join 不到的塞进沙箱当"测过了"。

复用基座：replay_three_leg._client/_chat(litellm+deepseek)、pre_instruction_snapshots.jsonl 的 code_refs。
输入 exec 集：request_type_2class.jsonl 里 class==exec 的行(可用 EXEC_IDXS 环境变量补测被归错的，如 idx120)。
"""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replay_three_leg as R

DATA = "/opt/shared/data/task-trajectory"
SNAP = f"{DATA}/pre_instruction_snapshots.jsonl"
CLS = f"{DATA}/request_type_2class.jsonl"
MODEL = os.environ.get("REPLAY_MODEL", "deepseek-v3.2")

_SYS = """你在判断【一条要 bot 执行的历史任务】要读取什么，好决定受控沙箱能不能忠实复现它。
只输出四类之一：
- B_context : 要读的是对话前文 / bot 自己上一轮的产出 / 群记忆——这些是"上下文"，不是磁盘文件。
- A_file    : 要读 t0 时刻仓库里的文件、代码、或盘上已有的产物文件(prompt、脚本、报告、jsonl 等)。
- C_live    : 要读活体外部态——飞书云文档/多维表格、在线 API、实时服务返回、当前群最新消息。
- none_exec : 不需要读任何历史，就是去跑/写/产出(跑测试、发消息、改代码、生成新文件)。
判据：问"要答对/做对这条，被试必须去读的那份东西，本质在哪"。
若既要读盘上文件又要读前文，以**更难复现**的为准(A_file 高于 B_context，C_live 最高)。
只输出一行 JSON：{"read_target":"B_context"|"A_file"|"C_live"|"none_exec","confidence":0.0-1.0,"reason":"≤25字"}"""


def judge_read_target(client, instruction, comment, ctx_peek):
    user = (f"【历史任务(instruction)】\n{instruction}\n\n"
            f"【复现要点提示】\n{comment}\n\n"
            f"【t0 前上下文片段(判断料在不在上下文里)】\n{ctx_peek}\n\n输出 JSON。")
    raw = R._chat(client, MODEL, _SYS, user).strip()
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return {"read_target": "parse_error", "confidence": 0.0, "reason": raw[:40]}
    try:
        d = json.loads(m.group(0))
        rt = d.get("read_target")
        if rt not in ("B_context", "A_file", "C_live", "none_exec"):
            return {"read_target": "parse_error", "confidence": 0.0, "reason": f"bad {rt}"}
        return {"read_target": rt, "confidence": float(d.get("confidence", 0)),
                "reason": str(d.get("reason", ""))[:40]}
    except Exception as e:
        return {"read_target": "parse_error", "confidence": 0.0, "reason": f"{e}"[:40]}


def verdict(read_target, code_refs):
    """确定性裁决：读取目标 + code_refs join 状态 → 可复现性。"""
    joined = [c for c in code_refs if c.get("status") == "ok"]
    if read_target == "B_context":
        return "reproducible", "上下文注入，非磁盘 read"
    if read_target == "A_file":
        if joined:
            return "needs_target_freeze_check", f"{len(joined)}/{len(code_refs)} 仓 join 到；须核具体读取文件是 git-tracked(A1可checkout) 还是 shared 工作文件(A2须 mtime<t0 冻结)"
        return "unreplayable", "两仓全 join 不到 commit，读取目标还原不到 t0 态(环境缺料)"
    if read_target == "C_live":
        return "unreplayable", "活体外部态自 t0 漂移，沙箱复现不了"
    if read_target == "none_exec":
        return "exec_env", "不读历史，可复现性由执行环境定，另论"
    return "unknown", read_target


def main():
    snaps = [json.loads(l) for l in open(SNAP, encoding="utf-8") if l.strip()]
    cls = [json.loads(l) for l in open(CLS, encoding="utf-8") if l.strip()]
    exec_idxs = {c["idx"] for c in cls if c.get("class") == "exec"}
    # 允许手工补入被 judge 归错、经人拍板归 exec 的条目(如 idx120)
    extra = os.environ.get("EXEC_IDXS", "").strip()
    if extra:
        exec_idxs |= {int(x) for x in extra.split(",") if x.strip().isdigit()}
    idxs = sorted(exec_idxs)
    print(f"exec 集 {len(idxs)} 条(含补入 {extra or '无'})", flush=True)
    client = R._client()
    out = []
    for i in idxs:
        s = snaps[i]
        instr = s.get("original_instruction", "")
        comment = s.get("comment_for_replay", "")
        events = (s.get("context") or {}).get("events", [])
        ctx_peek = "\n".join(f"[{e.get('role')}/{e.get('name','')}] {e.get('text','')[:120]}"
                             for e in events[-12:])[:1800]
        rt = judge_read_target(client, instr, comment, ctx_peek)
        v, vr = verdict(rt["read_target"], s.get("code_refs", []))
        row = {"idx": i, "t0_msg_id": (s.get("t0") or {}).get("msg_id", ""),
               "read_target": rt["read_target"], "rt_conf": rt["confidence"], "rt_reason": rt["reason"],
               "verdict": v, "verdict_reason": vr,
               "code_refs": [{"repo": c["repo"], "status": c["status"]} for c in s.get("code_refs", [])],
               "instruction": instr[:70]}
        out.append(row)
        print(f'{i:3d} [{rt["read_target"]:10s}] -> {v:26s} | {instr[:38]!r}', flush=True)
    dst = f"{DATA}/exec_read_reproducibility.jsonl"
    with open(dst, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    import collections
    print(f"\n== read_target: {dict(collections.Counter(o['read_target'] for o in out))}")
    print(f"== verdict    : {dict(collections.Counter(o['verdict'] for o in out))}")
    print(f"== wrote {len(out)} -> {dst}")


if __name__ == "__main__":
    main()
