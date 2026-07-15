#!/usr/bin/env python3
"""SubtaskTrajectory 组装器 —— 把「群聊层切分(llm_decompose)」× 「轨迹层 steps(trajectory_aggregate)」
join 成一条符合 schema/subtask_trajectory.schema.json 的子任务训练样本。

设计依据：Agent 轨迹工程全链路整合综述 §5.1（加工侧字段对齐利用侧消费接口）。
四点决策（张耀明 2026-06-25 拍板）：
  1) outcome = LLM-as-judge —— 复用 llm_decompose 的 status 字段（它已判「走偏→被点醒→回退」），
     不另起炉灶；failure（最终没做对）llm_decompose 口径不含，留给 failure-judge 补判(本期返 unknown)。
  2) step 按工具调用聚合 —— model + 紧随 tool = 一个助手回合，聚成 tool_interaction。
  3) 先单 bot 主干 —— multi_agent 段跳过 steps 深加工；orchestration 恒 null。
  4) implicit_feedback 留占位 —— 恒 null，采集管道本期不动。

本期实产：身份/溯源 + quality_tier + outcome(复用 status) + hard_example + steps(聚合+loss_mask)。
占位：orchestration / implicit_feedback / step_reward。
"""
from __future__ import annotations

import json
import os

# ── 决策1：outcome 复用 llm_decompose.status（单一事实源，不重判）──────────────
_STATUS_OUTCOME = {"一遍过": "success", "有返工": "success", "被纠偏": "corrected"}


def map_outcome(status: str, judge_verdict: str | None = None, *,
                terminal: bool = True) -> str:
    """P1·确定性反馈驱动 → schema.outcome（张耀明 2026-06-27 拍板，取代旧两判官覆盖逻辑）。

    修正两条历史误判：
    1)「复杂需求一遍过、用户没反馈」不再被 failure_judge 的 unknown 误降——**沉默即认可**：
       无负向反馈即视为达预期 → success；不再要求「用户明确认可」。
       （旧逻辑里 judge 的 unknown 正是误降来源，这里彻底不参与判定。）
    2) 但「沉默即认可」**只在任务确实终结时成立**。任务还在进行 / 跨段被续上（terminal=False）
       一律 incomplete——绝不给没收尾的活盖成功章（防「看着完成实则没完」式假成功）。

    判定优先级（高→低）：
      - terminal=False         → incomplete（没收尾，成败都不作数，等续上的段一起判）
      - judge_verdict=failure  → failure（failure_judge 拿到明确失败证据，唯一保留的 judge 信号）
      - status=被纠偏           → corrected（过程有歧路但最终做对，保留过程信号）
      - 其余(一遍过/有返工/缺失) → success（沉默即认可）

    judge_verdict 只取 failure 这一强信号；success/unknown 一律不参与（沉默即认可已覆盖）。"""
    if not terminal:
        return "incomplete"
    if judge_verdict == "failure":
        return "failure"
    if status == "被纠偏":
        return "corrected"
    return "success"


def derive_interaction_cost(evs: list) -> dict:
    """量化「bot 用了几次需求才达成」——张耀明 2026-06-26 主线指标（少需求次数达成）。
    确定性口径（不再调 LLM，可核可复现）：
      - user_turns = 本子需求里用户开口的消息条数（首次提需求 + 后续每次打回/追加）；
      - retry_rounds = max(0, user_turns-1) = 首提之外的轮数。
    诚实标注：retry_rounds 是**上界近似**——用户的追加里可能有纯补充信息（非返工），
    但作为第一版「少轮次」指标够用且零成本；要精分纯返工 vs 补充，后续用 LLM 另判，不在本期。"""
    user_turns = sum(1 for e in evs if e.get("role") == "user")
    return {"user_turns": user_turns, "retry_rounds": max(0, user_turns - 1)}


def derive_hard_example(status: str, steps: list) -> dict:
    """难例标记 + 出错 step 定位（iStar 消费 error_step_idx）。
    status=被纠偏 → rollback_corrected；steps 有工具报错 → tool_error_recovery；
    status=有返工 → multi_retry；否则 none。"""
    err_idx = next((s["idx"] for s in steps
                    if s.get("observation") and s["observation"].get("error")), None)
    if status == "被纠偏":
        return {"is_hard": True, "type": "rollback_corrected", "error_step_idx": err_idx}
    if err_idx is not None:
        return {"is_hard": True, "type": "tool_error_recovery", "error_step_idx": err_idx}
    if status == "有返工":
        return {"is_hard": True, "type": "multi_retry", "error_step_idx": err_idx}
    return {"is_hard": False, "type": "none", "error_step_idx": None}


# ── 决策2：step 按工具调用聚合（轨迹层 trajectory_aggregate steps → schema steps）──
def _assistant_text(output_messages):
    """从 trajectory model step 的 output_messages 提取助手纯文本（去 tool_call 结构）。
    兼容 OpenAI(content=str / content=[{type:text,text}]) 两种形态。"""
    if not output_messages:
        return None
    msgs = output_messages if isinstance(output_messages, list) else [output_messages]
    parts = []
    for m in msgs:
        if not isinstance(m, dict):
            parts.append(str(m)); continue
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            for blk in c:
                if isinstance(blk, dict) and blk.get("text"):
                    parts.append(blk["text"])
    txt = "\n".join(p for p in parts if p).strip()
    return txt or None


def _tool_interaction(idx, actor, tool_step, text=None):
    return {
        "idx": idx, "type": "tool_interaction", "actor": actor,
        "action": {"text": text, "reasoning": None,
                   "tool_call": {"name": tool_step.get("tool_name"),
                                 "args": tool_step.get("tool_inputs")}},
        "observation": {"tool_result": tool_step.get("tool_outputs"),
                        "error": tool_step.get("error")},
        "loss_mask": 1, "step_reward": None,
    }


def aggregate_steps(traj_steps: list, bot_id: str) -> list:
    """把轨迹层扁平 steps(type=model/tool) 聚合成 schema steps。
    一个 model step + 其后紧随的 tool steps = 同一助手回合：
      - 回合有 tool → 每个 tool 成一条 tool_interaction，model 文本挂到回合首条 action.text；
      - 回合无 tool → model step 成一条 plain_message(assistant)。
    loss_mask：助手产出(action)=1；observation 永远是 context（token 级掩码下游按 action/observation 结构做）。
    输入需已按时间序（trajectory_aggregate 出口保证 steps.sort by start_ns）。"""
    out, i, n = [], 0, len(traj_steps)
    while i < n:
        s = traj_steps[i]
        if s.get("type") == "model":
            text = _assistant_text(s.get("output_messages"))
            j, tools = i + 1, []
            while j < n and traj_steps[j].get("type") == "tool":
                tools.append(traj_steps[j]); j += 1
            if tools:
                for k, t in enumerate(tools):
                    out.append(_tool_interaction(len(out), bot_id, t, text if k == 0 else None))
                i = j
            else:
                out.append({"idx": len(out), "type": "plain_message", "actor": bot_id,
                            "action": {"text": text, "reasoning": None, "tool_call": None},
                            "observation": None, "loss_mask": 1, "step_reward": None})
                i += 1
        elif s.get("type") == "tool":          # 孤立 tool（前无 model）：仍记为 tool_interaction
            out.append(_tool_interaction(len(out), bot_id, s))
            i += 1
        else:
            i += 1
    return out


# ── 退化路：无轨迹层数据时按群聊消息级产 steps，保证 schema 可满足、可先跑通 ──────
def _chat_steps(evs: list) -> list:
    """群聊层退化 steps：每条消息一个 plain_message。user→loss_mask=0；bot→1。"""
    out = []
    for e in evs:
        is_user = e.get("role") == "user"
        out.append({
            "idx": len(out), "type": "plain_message",
            "actor": "user" if is_user else (e.get("name") or e.get("bot") or "bot"),
            "action": None if is_user else {"text": e.get("text"), "reasoning": None, "tool_call": None},
            "observation": None, "loss_mask": 0 if is_user else 1, "step_reward": None,
        })
    return out


def _infer_tier(steps: list) -> str:
    """quality_tier 启发式占位（TODO：本应由 judge 按 CoT 可信度判）。
    有 reasoning→reasoning_sup；有 tool_interaction/bot action→behavior_clone；否则 recall_only。"""
    if any((s.get("action") or {}).get("reasoning") for s in steps):
        return "reasoning_sup"
    if any(s["type"] == "tool_interaction" or (s.get("action")) for s in steps):
        return "behavior_clone"
    return "recall_only"


# ── 组装：一个 subreq(llm_decompose) + 其消息 + 主导 bot 轨迹 → SubtaskTrajectory ──
def assemble_subtask(subreq: dict, meta: list, source_task_id: str, group_id: str,
                     bot_traj_steps: list | None = None,
                     outcome_judge: dict | None = None,
                     terminal: bool = True) -> dict:
    """subreq: llm_decompose 的一个子需求；meta: llm_decompose 的消息 meta(1-based member_idx)；
    bot_traj_steps: 主导 bot 在本段的轨迹层 steps（单 bot 段且有数据时才组装聚合 steps，
    否则退化为群聊消息级 steps）；
    outcome_judge: failure_judge.judge_subreq 的结果 {verdict,reason,...}（判最终成没成，
    用于覆盖 llm_decompose 的乐观假设；不传则只按 status 映射，向后兼容）。"""
    members = sorted(int(x) for x in subreq.get("member_idx", []))
    evs = [meta[i - 1] for i in members if 1 <= i <= len(meta)]
    tss = [int(e.get("ts") or 0) for e in evs]
    bots = sorted({(e.get("name") or e.get("bot")) for e in evs if e.get("role") != "user"})
    multi_agent = len(bots) > 1
    status = subreq.get("status", "")
    dominant = subreq.get("dominant", "") or (bots[0] if bots else "bot")

    if bot_traj_steps and not multi_agent:
        steps = aggregate_steps(bot_traj_steps, dominant)        # 决策2 聚合
    else:
        steps = _chat_steps(evs)                                 # 退化（multi_agent 或无轨迹）

    judge_verdict = (outcome_judge or {}).get("verdict")
    outcome = map_outcome(status, judge_verdict, terminal=terminal)
    outcome_reason = f"P1·确定性: status={status}, terminal={terminal}"
    if not terminal:
        outcome_reason += " → incomplete(任务未终结,沉默不作认可)"
    elif judge_verdict == "failure":
        outcome_reason += f"; failure_judge=failure({outcome_judge.get('reason', '')})"

    return {
        "subtask_id": f"{source_task_id}#{subreq.get('id', 'S?')}",
        "source_task_id": source_task_id,
        "group_id": group_id,
        "span": {"start": min(tss) if tss else 0, "end": max(tss) if tss else 0},
        "msg_ids": [e.get("msg_id", "") for e in evs],
        "topic_label": subreq.get("title"),
        "participants": sorted({(e.get("name") or e.get("bot") or "?") for e in evs}),
        "multi_agent": multi_agent,
        "quality_tier": _infer_tier(steps),
        "outcome": outcome,                                      # P1·确定性(沉默即认可+terminal gate)
        "outcome_reason": outcome_reason,
        "interaction_cost": derive_interaction_cost(evs),        # 少需求次数达成指标
        "hard_example": derive_hard_example(status, steps),
        "steps": steps,
        "orchestration": None,                                   # 决策3 占位
        "implicit_feedback": None,                               # 决策4 占位
    }


_SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "schema", "subtask_trajectory.schema.json")


def validate(obj: dict):
    """对照 schema 校验（jsonschema 不在则做必填字段兜底检查，不静默放过）。"""
    schema = json.load(open(_SCHEMA_PATH, encoding="utf-8"))
    try:
        import jsonschema
        jsonschema.validate(obj, schema)
        return True
    except ImportError:
        for k in schema["required"]:
            if k not in obj:
                raise AssertionError(f"[schema 校验失败] 缺必填字段: {k}")
        if obj["outcome"] not in schema["properties"]["outcome"]["enum"]:
            raise AssertionError(f"[schema 校验失败] outcome 非法: {obj['outcome']}")
        return True
