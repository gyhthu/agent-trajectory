#!/usr/bin/env python3
"""assemble_subtask_trajectory 单元测试：决策1(outcome 复用) + 决策2(step 聚合) + 端到端 schema 校验。
跑：python test_assemble_subtask_trajectory.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from assemble_subtask_trajectory import (
    map_outcome, derive_hard_example, derive_interaction_cost, aggregate_steps,
    _assistant_text, assemble_subtask, validate,
)


# ── P1·确定性反馈驱动：沉默即认可 ──
def test_map_outcome():
    # 一遍过/有返工/缺失 + 无负向反馈 → success（沉默即认可，不再要求用户明确认可）
    assert map_outcome("一遍过") == "success"
    assert map_outcome("有返工") == "success"
    assert map_outcome("被纠偏") == "corrected"
    assert map_outcome("") == "success"          # P1：无负向信号 → 沉默即认可（旧版误判 unknown）
    assert map_outcome("瞎写") == "success"
    print("✓ P1 outcome 映射(沉默即认可)")


# ── P1：judge 只取 failure 强信号；unknown 不再误降一遍过 ──
def test_map_outcome_with_judge():
    # judge=failure 是唯一保留的 judge 信号：有明确失败证据 → failure
    assert map_outcome("一遍过", "failure") == "failure"
    assert map_outcome("有返工", "failure") == "failure"
    assert map_outcome("被纠偏", "failure") == "failure"
    # 被纠偏 → corrected（过程信号优先于 judge=success）
    assert map_outcome("被纠偏", "success") == "corrected"
    assert map_outcome("一遍过", "success") == "success"
    # 关键回归：复杂需求一遍过 + judge=unknown(用户没反馈) → success，绝不再误降 unknown
    assert map_outcome("一遍过", "unknown") == "success"
    assert map_outcome("有返工", "unknown") == "success"
    assert map_outcome("", "unknown") == "success"
    print("✓ P1 judge：failure 唯一强信号 / unknown 不再误降沉默成功样本")


# ── P1·terminal gate：任务未终结一律 incomplete，不给没收尾的活盖成功章 ──
def test_map_outcome_terminal_gate():
    # 任务还在进行/跨段被续上 → incomplete，盖过一切成败
    assert map_outcome("一遍过", terminal=False) == "incomplete"
    assert map_outcome("被纠偏", terminal=False) == "incomplete"
    assert map_outcome("一遍过", "failure", terminal=False) == "incomplete"
    # terminal=True（默认）才进入沉默即认可
    assert map_outcome("一遍过", terminal=True) == "success"
    print("✓ P1 terminal gate(未终结→incomplete)")


# ── interaction_cost：少需求次数达成指标 ──
def test_interaction_cost():
    evs = [{"role": "user"}, {"role": "bot"}, {"role": "user"}, {"role": "bot"}, {"role": "user"}]
    c = derive_interaction_cost(evs)
    assert c["user_turns"] == 3 and c["retry_rounds"] == 2
    assert derive_interaction_cost([{"role": "user"}, {"role": "bot"}]) == {"user_turns": 1, "retry_rounds": 0}
    assert derive_interaction_cost([{"role": "bot"}]) == {"user_turns": 0, "retry_rounds": 0}
    print("✓ interaction_cost 确定性计数(user_turns / retry_rounds)")


# ── 组装器端到端带 judge：failure 覆盖 + interaction_cost 入样本 + schema 仍过 ──
def test_assemble_with_judge():
    meta = [
        {"ts": 100, "msg_id": "m1", "role": "user", "name": "u", "text": "改简洁点"},
        {"ts": 160, "msg_id": "m2", "role": "bot", "name": "lian-server", "text": "改了"},
        {"ts": 200, "msg_id": "m3", "role": "user", "name": "u", "text": "还是不行，没解决"},
    ]
    subreq = {"id": "S1", "title": "改写", "dominant": "lian-server",
              "status": "一遍过", "member_idx": [1, 2, 3]}
    judge = {"verdict": "failure", "reason": "用户最后明说没解决"}
    obj = assemble_subtask(subreq, meta, "task9", "oc_x", outcome_judge=judge)
    assert validate(obj)
    assert obj["outcome"] == "failure"                    # judge 覆盖了 status=一遍过 的乐观 success
    assert "failure_judge=failure" in obj["outcome_reason"]
    assert obj["interaction_cost"] == {"user_turns": 2, "retry_rounds": 1}
    print("✓ 组装器接 judge：failure 覆盖乐观假设 + interaction_cost 入样本 + schema 过")


# ── 决策2：step 按工具调用聚合 ──
def test_aggregate_model_plus_tool():
    traj = [
        {"type": "model", "start_ns": 1, "output_messages": [{"role": "assistant", "content": "我来查一下"}]},
        {"type": "tool", "start_ns": 2, "tool_name": "Bash", "tool_inputs": {"cmd": "ls"},
         "tool_outputs": "a.txt", "error": None},
    ]
    steps = aggregate_steps(traj, "lian-server")
    assert len(steps) == 1, f"model+tool 应聚成1条, got {len(steps)}"
    s = steps[0]
    assert s["type"] == "tool_interaction"
    assert s["action"]["text"] == "我来查一下"           # model 文本挂到 action
    assert s["action"]["tool_call"]["name"] == "Bash"
    assert s["observation"]["tool_result"] == "a.txt"
    assert s["loss_mask"] == 1
    print("✓ 决策2 model+tool 聚合为 tool_interaction")


def test_aggregate_model_alone():
    traj = [{"type": "model", "start_ns": 1,
             "output_messages": [{"role": "assistant", "content": "纯文本回复"}]}]
    steps = aggregate_steps(traj, "bot")
    assert len(steps) == 1 and steps[0]["type"] == "plain_message"
    assert steps[0]["action"]["text"] == "纯文本回复" and steps[0]["loss_mask"] == 1
    print("✓ 决策2 纯 model → plain_message")


def test_aggregate_multi_tool_in_turn():
    traj = [
        {"type": "model", "start_ns": 1, "output_messages": [{"role": "assistant", "content": "并行两个"}]},
        {"type": "tool", "start_ns": 2, "tool_name": "Read", "tool_inputs": {}, "tool_outputs": "x", "error": None},
        {"type": "tool", "start_ns": 3, "tool_name": "Grep", "tool_inputs": {}, "tool_outputs": "y", "error": None},
    ]
    steps = aggregate_steps(traj, "bot")
    assert len(steps) == 2, "一回合2工具应出2条 tool_interaction"
    assert steps[0]["action"]["text"] == "并行两个"      # 文本只挂首条
    assert steps[1]["action"]["text"] is None
    assert [s["idx"] for s in steps] == [0, 1]
    print("✓ 决策2 一回合多工具 → 文本只挂首条")


def test_assistant_text_block_form():
    assert _assistant_text([{"role": "assistant", "content": [{"type": "text", "text": "块文本"}]}]) == "块文本"
    assert _assistant_text(None) is None
    print("✓ output_messages 块状 content 文本提取")


# ── 难例标记 ──
def test_hard_example():
    assert derive_hard_example("被纠偏", [])["type"] == "rollback_corrected"
    err_steps = [{"idx": 0, "observation": {"error": "TimeoutError"}}]
    he = derive_hard_example("一遍过", err_steps)
    assert he["type"] == "tool_error_recovery" and he["error_step_idx"] == 0
    assert derive_hard_example("有返工", [])["type"] == "multi_retry"
    assert derive_hard_example("一遍过", [])["is_hard"] is False
    print("✓ 难例标记 + error_step_idx 定位")


# ── 端到端：subreq → SubtaskTrajectory → schema 校验 ──
def test_assemble_end_to_end():
    meta = [
        {"ts": 100, "msg_id": "m1", "role": "user", "name": "张耀明", "text": "帮我查个 bug"},
        {"ts": 160, "msg_id": "m2", "role": "bot", "name": "lian-server", "text": "好，我看一下"},
        {"ts": 200, "msg_id": "m3", "role": "bot", "name": "lian-server", "text": "修好了"},
    ]
    subreq = {"id": "S1", "title": "查bug", "dominant": "lian-server",
              "status": "被纠偏", "type": "主线", "member_idx": [1, 2, 3]}
    obj = assemble_subtask(subreq, meta, source_task_id="task42", group_id="oc_x")
    assert validate(obj)
    assert obj["subtask_id"] == "task42#S1"
    assert obj["outcome"] == "corrected"
    assert obj["msg_ids"] == ["m1", "m2", "m3"]
    assert obj["span"] == {"start": 100, "end": 200}
    assert obj["hard_example"]["type"] == "rollback_corrected"
    assert obj["orchestration"] is None and obj["implicit_feedback"] is None
    # 退化路：无轨迹 → 群聊消息级 steps，user loss_mask=0
    assert obj["steps"][0]["actor"] == "user" and obj["steps"][0]["loss_mask"] == 0
    assert obj["steps"][1]["loss_mask"] == 1
    print("✓ 端到端组装 + schema 校验通过")


def test_assemble_with_traj_steps():
    meta = [{"ts": 100, "msg_id": "m1", "role": "user", "name": "u", "text": "do it"},
            {"ts": 160, "msg_id": "m2", "role": "bot", "name": "lian-server", "text": "ok"}]
    subreq = {"id": "S1", "title": "t", "dominant": "lian-server", "status": "一遍过", "member_idx": [1, 2]}
    traj = [{"type": "model", "start_ns": 1, "output_messages": [{"role": "assistant", "content": "跑命令"}]},
            {"type": "tool", "start_ns": 2, "tool_name": "Bash", "tool_inputs": {}, "tool_outputs": "ok", "error": None}]
    obj = assemble_subtask(subreq, meta, "task1", "oc_x", bot_traj_steps=traj)
    assert validate(obj)
    assert len(obj["steps"]) == 1 and obj["steps"][0]["type"] == "tool_interaction"
    assert obj["outcome"] == "success"
    print("✓ 带轨迹层 steps 的单 bot 组装(决策2 生效)")


if __name__ == "__main__":
    test_map_outcome()
    test_map_outcome_with_judge()
    test_map_outcome_terminal_gate()
    test_interaction_cost()
    test_assemble_with_judge()
    test_aggregate_model_plus_tool()
    test_aggregate_model_alone()
    test_aggregate_multi_tool_in_turn()
    test_assistant_text_block_form()
    test_hard_example()
    test_assemble_end_to_end()
    test_assemble_with_traj_steps()
    print("\n全部通过 ✓")
