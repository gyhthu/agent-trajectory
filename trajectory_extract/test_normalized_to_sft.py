#!/usr/bin/env python3
"""normalized_to_sft 的单测：验证指令上浮、噪声剔除、tool_call_id 绑定、OpenAI 线格式。"""
import json

from normalized_to_sft import to_sft, _extract_top_objective


def _base_rec(messages, tools=None):
    return {"source": "antigravity-transcript", "conv_id": "c1", "model": "gemini-3-pro",
            "created_at": "t0", "n_model_turns": 1, "messages": messages,
            "tools": tools or [{"type": "function", "function": {"name": "run_command"}}],
            "fidelity": {"user_prompt": True}}


def test_probe_and_history_dropped_instruction_surfaced():
    ch = ("# Conversation History\n<conversation_summaries>\n"
          "## Conversation abc: Finalizing Perms\n### USER Objective:\n"
          "Resolve denied write access to /mcp.\n</conversation_summaries>")
    rec = _base_rec([
        {"role": "user", "content": "__probe__", "metadata": "time"},
        {"role": "system", "subtype": "conversation_history", "content": ch},
        {"role": "assistant", "content": "checking", "tool_calls": [{"name": "run_command", "args": {"CommandLine": "ls"}}]},
        {"role": "tool", "name": "run_command", "content": "ok"},
    ])
    sft = to_sft(rec)
    roles = [m["role"] for m in sft["messages"]]
    # __probe__ 与 conversation_history 都不进 messages
    assert roles == ["user", "assistant", "tool"], roles
    assert "__probe__" not in json.dumps(sft["messages"], ensure_ascii=False)
    assert "conversation_history" not in json.dumps(sft["messages"], ensure_ascii=False)
    # 指令上浮 = history 顶部 Objective
    assert sft["messages"][0]["content"].startswith("Resolve denied write access")
    assert sft["meta"]["instruction_source"] == "history_obj"


def test_real_user_text_wins():
    rec = _base_rec([
        {"role": "user", "content": "只回三个字：登录通"},
        {"role": "assistant", "content": "登录通"},
    ])
    sft = to_sft(rec)
    assert sft["messages"][0] == {"role": "user", "content": "只回三个字：登录通"}
    assert sft["meta"]["instruction_source"] == "real_user"


def test_tool_call_openai_format_and_id_binding():
    rec = _base_rec([
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"name": "run_command", "args": {"CommandLine": "a"}},
            {"name": "list_dir", "args": {"DirectoryPath": "/x"}}]},
        {"role": "tool", "name": "run_command", "content": "r1"},
        {"role": "tool", "name": "list_dir", "content": "r2"},
    ])
    sft = to_sft(rec)
    asst = sft["messages"][1]
    c0, c1 = asst["tool_calls"]
    # OpenAI 线格式：id + type:function + arguments 为 JSON 字符串
    assert c0["type"] == "function" and c0["function"]["name"] == "run_command"
    assert json.loads(c0["function"]["arguments"]) == {"CommandLine": "a"}
    # tool 结果按序绑定到对应 call id
    tools = [m for m in sft["messages"] if m["role"] == "tool"]
    assert tools[0]["tool_call_id"] == c0["id"]
    assert tools[1]["tool_call_id"] == c1["id"]
    assert sft["meta"]["n_tool_calls"] == 2


def test_injected_system_kept():
    rec = _base_rec([
        {"role": "system", "subtype": "injected_system_prompt", "content": "你=zym-antigravity"},
        {"role": "user", "content": "__probe__"},
        {"role": "assistant", "content": "ok"},
    ])
    sft = to_sft(rec)
    assert sft["messages"][0] == {"role": "system", "content": "你=zym-antigravity"}
    # 无真人文本/history → 兜底 injected
    assert sft["meta"]["instruction_source"] == "injected"


def test_probe_only_marked():
    rec = _base_rec([
        {"role": "user", "content": "__probe__"},
        {"role": "assistant", "content": "ok"},
    ])
    sft = to_sft(rec)
    assert sft["meta"]["instruction_source"] == "probe_only"


def test_extract_top_objective_reverse_chrono():
    ch = ("<conversation_summaries>\n"
          "## Conversation a: Recent Task\n### USER Objective:\nDo the recent thing.\n"
          "## Conversation b: Old Task\n### USER Objective:\nOld login check.\n"
          "</conversation_summaries>")
    title, obj = _extract_top_objective(ch)
    # 倒序：第一块=最近，取 Recent 不取 Old
    assert title == "Recent Task"
    assert obj == "Do the recent thing."


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ✅ {fn.__name__}")
    print(f"\n{len(fns)} passed")
    sys.exit(0)
