#!/usr/bin/env python3
"""归一化轨迹 → 干净标准 SFT 转换器（下游层，纯只读转换）。

上游 `antigravity_transcript_adapter.py` 产出的是「带压缩感知 + 保真度」的富归一化轨迹
（保留 __probe__ 探针、conversation_history 摘要、per_turn_context 等分析用元数据）。
那份格式利于质量分析，但**不能直接喂 SFT**：真实任务指令埋在 conversation_history 摘要深处，
`__probe__` 占着 user 位。

本转换器把富轨迹压成**最标准的 OpenAI chat SFT**：
    {"messages": [...], "tools": [...], "meta": {...provenance...}}

三件核心归整：
  1. **指令上浮**：把真实任务指令提到 user 位。来源按优先级（记进 meta.instruction_source）：
       real_user   —— transcript 里真人打的非空、非 __probe__ 用户消息（最高保真）
       history_obj —— conversation_history 顶部（=最近一段）的 `USER Objective` 段落
                      （注意：这是 antigravity 自总结的摘要，非用户逐字原话）
       history_title / injected —— 只抽到标题 / 仅有注入 system 时的兜底
       probe_only  —— 纯探针驱动、无任何显式指令（标出来，下游自行取舍）
  2. **丢噪声**：删掉 __probe__ 探针轮、conversation_history 摘要块（指令已上浮，不再重复留噪）。
  3. **工具调用转 OpenAI 线格式**：assistant.tool_calls 补 `id`+`type:function`、`arguments`
     转 JSON 字符串；紧随的 tool 结果按序绑定 `tool_call_id`（单条 assistant 多调用也对得上）。

注入的 bot 人格 system（injected_system_prompt）若在，作为 `system` 消息保留（真实系统上下文）。
reasoning（thinking）若在，挂在 assistant 消息的 `reasoning` 字段（不塞进 content，保训练可分离）。

诚实边界（照搬上游 fidelity，不粉饰）：history_obj 来源的指令是模型二次总结的 Objective，
不是用户逐字原话；要逐字真实指令得看 codex/claude wire 那两路。每条 meta 里带 instruction_source
与上游 fidelity，训练前可据此分流/加权。
"""
import argparse
import json
import re
import sys


def _extract_top_objective(ch_content):
    """从 conversation_history 抽最近一段（倒序列表第一块）的标题与 USER Objective 段落。
    返回 (title, objective_text)；抽不到则 (None, None)。"""
    if not ch_content or "<conversation_summaries>" not in ch_content:
        return None, None
    body = ch_content.split("<conversation_summaries>", 1)[1]
    body = body.split("</conversation_summaries>", 1)[0]
    # 按「## Conversation 」切块，第一块=最近（倒序）
    blocks = re.split(r"\n##\s+Conversation ", "\n" + body)
    blocks = [b for b in blocks if b.strip()]
    if not blocks:
        return None, None
    first = blocks[0]
    title = None
    if ":" in first:
        title = first.split(":", 1)[1].split("\n", 1)[0].strip()
    obj = None
    if "### USER Objective:" in first:
        obj = first.split("### USER Objective:", 1)[1].strip()
        # 去掉可能残留的下一节标记
        obj = re.split(r"\n##\s", obj)[0].strip()
    return title, (obj or None)


def _surface_instruction(rec):
    """决定 user 位放什么、来源是什么。返回 (instruction_text, source)。"""
    msgs = rec["messages"]
    # 1) 真人非空非探针用户文本
    for m in msgs:
        if m.get("role") == "user":
            c = str(m.get("content", "") or "").strip()
            if c and c != "__probe__":
                return c, "real_user"
    # 2/3) conversation_history 顶部 Objective / 标题
    ch = next((m for m in msgs if m.get("subtype") == "conversation_history"), None)
    if ch:
        title, obj = _extract_top_objective(ch.get("content", ""))
        if obj and title != "__probe__":
            return obj, "history_obj"
        if title and title != "__probe__":
            return title, "history_title"
    # 4) 仅注入 system
    if any(m.get("subtype") == "injected_system_prompt" for m in msgs):
        return "(继续按系统指令自主推进当前任务)", "injected"
    # 5) 纯探针
    return "(probe 探针驱动的自主续跑，transcript 未捕获显式任务指令)", "probe_only"


def to_sft(rec):
    """把一条富归一化轨迹转成 {messages, tools, meta} 标准 SFT。"""
    src_msgs = rec["messages"]
    out_msgs = []

    # system：保留注入的 bot 人格 system（真实系统上下文）
    for m in src_msgs:
        if m.get("subtype") == "injected_system_prompt":
            out_msgs.append({"role": "system", "content": m.get("content", "")})
            break

    # user：上浮后的真实任务指令
    instruction, source = _surface_instruction(rec)
    out_msgs.append({"role": "user", "content": instruction})

    # 中段：assistant / tool，丢 __probe__ 与 conversation_history/system_message 噪声
    call_seq = 0          # 全局自增 call id 号
    pending_ids = []      # 最近一条 assistant 的 tool_call ids，按序绑给随后的 tool 结果
    n_tool_calls = 0
    for m in src_msgs:
        role = m.get("role")
        sub = m.get("subtype", "")
        if role == "user":
            continue  # 所有原始 user（探针/已上浮的指令）不再重复进中段
        if role == "system":
            continue  # conversation_history / system_message / injected 都不进中段
        if role == "assistant":
            am = {"role": "assistant"}
            if m.get("reasoning"):
                am["reasoning"] = m["reasoning"]
            am["content"] = m.get("content", "") or ""
            tcs = m.get("tool_calls") or []
            if tcs:
                oa_calls = []
                pending_ids = []
                for tc in tcs:
                    cid = f"call_{call_seq}"
                    call_seq += 1
                    n_tool_calls += 1
                    pending_ids.append(cid)
                    oa_calls.append({
                        "id": cid,
                        "type": "function",
                        "function": {
                            "name": tc.get("name"),
                            "arguments": json.dumps(tc.get("args", {}), ensure_ascii=False),
                        },
                    })
                am["tool_calls"] = oa_calls
            out_msgs.append(am)
        elif role == "tool":
            cid = pending_ids.pop(0) if pending_ids else None
            tm = {"role": "tool", "content": m.get("content", "") or ""}
            if cid:
                tm["tool_call_id"] = cid
            if m.get("name"):
                tm["name"] = m["name"]
            out_msgs.append(tm)

    return {
        "messages": out_msgs,
        "tools": rec.get("tools"),
        "meta": {
            "source": rec.get("source"),
            "conv_id": rec.get("conv_id"),
            "model": rec.get("model"),
            "created_at": rec.get("created_at"),
            "instruction_source": source,       # 指令上浮来源（见 _surface_instruction）
            "n_model_turns": rec.get("n_model_turns"),
            "n_tool_calls": n_tool_calls,
            "n_compressions": rec.get("n_compressions"),
            "fidelity": rec.get("fidelity"),     # 照搬上游保真度，不粉饰
        },
    }


def main():
    ap = argparse.ArgumentParser(description="富归一化轨迹 → 标准 OpenAI SFT")
    ap.add_argument("infile", help="上游 adapter 产出的归一化 jsonl（如 zym-antigravity-sft.jsonl）")
    ap.add_argument("--out", default=None, help="输出 SFT jsonl 路径（不给则只打印统计+首条预览）")
    ap.add_argument("--min-tool-calls", type=int, default=0,
                    help="过滤：只保留 tool_calls 数 >= 此值的样本（0=不过滤）")
    args = ap.parse_args()

    from collections import Counter
    recs = []
    src_counter = Counter()
    with open(args.infile, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sft = to_sft(json.loads(line))
            if sft["meta"]["n_tool_calls"] < args.min_tool_calls:
                continue
            recs.append(sft)
            src_counter[sft["meta"]["instruction_source"]] += 1

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"输入轨迹 → 输出 SFT 样本: {len(recs)}")
    print(f"指令来源分布:            {dict(src_counter)}")
    roles = Counter()
    for r in recs:
        for m in r["messages"]:
            roles[m["role"]] += 1
    print(f"消息 role 分布:          {dict(roles)}")
    tc = sum(r["meta"]["n_tool_calls"] for r in recs)
    print(f"tool_calls 总数:         {tc}")
    if args.out:
        print(f"\n✅ 已写出: {args.out}")
    else:
        print("\n=== 首条 SFT 预览（messages 前若干条 + tools 数）===")
        r0 = recs[0]
        for m in r0["messages"][:6]:
            c = json.dumps(m, ensure_ascii=False)
            print("  " + (c[:200] + ("…" if len(c) > 200 else "")))
        print(f"  ... tools: {len(r0['tools'] or [])} 个  meta: {r0['meta']['instruction_source']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
