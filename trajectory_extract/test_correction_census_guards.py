#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from correction_census import validate_events
from session_corrections import _guard, _reguard_cached_rows


def _row(mid, ts, role, text, name=None):
    return {
        "msg_id": mid,
        "ts": ts,
        "role": role,
        "who": name or role,
        "name": name or role,
        "text": text,
    }


def _event(corrector_msg_id="c", corrector_role="user"):
    return {
        "anchor_msg_id": "b",
        "bot_error_quote": "codex 走 litellm",
        "corrector_msg_id": corrector_msg_id,
        "corrector_role": corrector_role,
        "what": "bot 把 codex reasoning 说错",
    }


def test_validate_events_requires_human_corrector():
    rows = [
        _row("b", 10, "bot", "codex 走 litellm，所以 reasoning 能看", "lian-codex"),
        _row("c", 20, "bot", "我刚才说错了，codex reasoning 看不到", "lian-codex"),
    ]

    kept, dropped = validate_events([_event(corrector_role="bot")], rows)

    assert kept == []
    assert dropped[0]["_drop"].startswith("corrector_role非user")


def test_validate_events_drops_system_or_placeholder_corrector():
    rows = [
        _row("b", 10, "bot", "codex 走 litellm，所以 reasoning 能看", "lian-codex"),
        _row("c", 20, "user", "👀收到，排队中", "张耀明"),
    ]

    kept, dropped = validate_events([_event()], rows)

    assert kept == []
    assert dropped[0]["_drop"] == "corrector是占位/系统回执"


def test_validate_events_drops_demand_change_without_explicit_correction():
    rows = [
        _row("b", 10, "bot", "codex 走 litellm，所以 reasoning 能看", "lian-codex"),
        _row("c", 20, "user", "我改主意了，改写任务和 session 路由是两个独立任务，现在聚焦改写任务", "张耀明"),
    ]

    kept, dropped = validate_events([_event()], rows)

    assert kept == []
    assert dropped[0]["_drop"] == "corrector是需求变更/方向重定向"


def test_validate_events_drops_plan_change_without_explicit_correction():
    rows = [
        _row("b", 10, "bot", "我建议跑盲判 39 条，再看三处决策", "lian-codex"),
        _row("c", 20, "user", "那就不跑盲判 39 条了，只看引用优先模式那 3 处决策。", "张耀明"),
    ]
    ev = {
        "anchor_msg_id": "b",
        "bot_error_quote": "跑盲判 39 条",
        "corrector_msg_id": "c",
        "corrector_role": "user",
        "what": "用户调整评测范围",
    }

    kept, dropped = validate_events([ev], rows)

    assert kept == []
    assert dropped[0]["_drop"] == "corrector是需求变更/方向重定向"


def test_validate_events_drops_plan_change_with_actual_runtime_word():
    rows = [
        _row("b", 10, "bot", "那就不跑盲判 39 条了，只看引用优先模式那 3 处决策。", "lian-codex"),
        _row("c", 20, "user", "我改主意了，改写任务和 session 路由是独立任务。现在聚焦在实际运行里的 thread。", "张耀明"),
    ]
    ev = {
        "anchor_msg_id": "b",
        "bot_error_quote": "不跑盲判 39 条",
        "corrector_msg_id": "c",
        "corrector_role": "user",
        "what": "用户调整评测范围",
    }

    kept, dropped = validate_events([ev], rows)

    assert kept == []
    assert dropped[0]["_drop"] == "corrector是需求变更/方向重定向"


def test_validate_events_drops_design_case_supplement():
    rows = [
        _row("b", 10, "bot", "Plan 模式已开，先并行探两块数据结构。", "lian-codex"),
        _row("c", 20, "user", "你可以写，但对于以下两情况你是怎么考虑的呢？后台跑完没回应不应该算终结。", "张耀明"),
    ]
    ev = {
        "anchor_msg_id": "b",
        "bot_error_quote": "Plan 模式已开",
        "corrector_msg_id": "c",
        "corrector_role": "user",
        "what": "用户补充设计边界",
    }

    kept, dropped = validate_events([ev], rows)

    assert kept == []
    assert dropped[0]["_drop"] == "corrector是需求变更/方向重定向"


def test_validate_events_drops_troubleshooting_feedback():
    rows = [
        _row("b", 10, "bot", "先各查一下，别瞎试。把端口状态和连接输出贴出来。", "lian-codex"),
        _row("c", 20, "user", "curl 显示 connection refused，9324 端口还是没起来。", "张耀明"),
    ]
    ev = {
        "anchor_msg_id": "b",
        "bot_error_quote": "把端口状态和连接输出贴出来",
        "corrector_msg_id": "c",
        "corrector_role": "user",
        "what": "用户回填排障输出",
    }

    kept, dropped = validate_events([ev], rows)

    assert kept == []
    assert dropped[0]["_drop"] == "corrector是排障信息回填"


def test_validate_events_drops_troubleshooting_feedback_with_traceback():
    rows = [
        _row("b", 10, "bot", "把两段输出都贴回来，告诉我 hub 卡在哪。", "lian-codex"),
        _row("c", 20, "user", "德国机打印结果如下：\\nTraceback (most recent call last):\\nConnectionRefusedError: [Errno 111] Connection refused", "张耀明"),
    ]
    ev = {
        "anchor_msg_id": "b",
        "bot_error_quote": "把两段输出都贴回来",
        "corrector_msg_id": "c",
        "corrector_role": "user",
        "what": "用户回填排障输出",
    }

    kept, dropped = validate_events([ev], rows)

    assert kept == []
    assert dropped[0]["_drop"] == "corrector是排障信息回填"


def test_validate_events_keeps_human_correction_question_not_troubleshooting():
    rows = [
        _row("b", 10, "bot", "这个任务标为有返工，状态已经核过。", "lian-codex"),
        _row("c", 20, "user", "为什么“调取李博颉登录对话”状态为“有返工”，它不是只有一条消息吗？", "张耀明"),
    ]
    ev = {
        "anchor_msg_id": "b",
        "bot_error_quote": "标为有返工",
        "corrector_msg_id": "c",
        "corrector_role": "user",
        "what": "用户指出单条消息不可能有返工",
    }

    kept, dropped = validate_events([ev], rows)

    assert dropped == []
    assert len(kept) == 1


def test_validate_events_keeps_human_counter_evidence():
    rows = [
        _row("b", 10, "bot", "codex 走 litellm，所以 reasoning 能看", "lian-codex"),
        _row("c", 20, "user", "不对，codex reasoning 是加密块，实际零可读", "张耀明"),
    ]

    kept, dropped = validate_events([_event()], rows)

    assert dropped == []
    assert len(kept) == 1


def test_validate_events_keeps_human_source_file_correction_question():
    rows = [
        _row(
            "b",
            10,
            "bot",
            "gold_review_sheet_pilot.md 是你标注过的结果文件，可以作为 #4 gold 的来源。",
            "lian-codex",
        ),
        _row(
            "c",
            20,
            "user",
            "这个文件我没标注过啊，你是不是搞错了？找到我标注的结果来说明我错了。",
            "张耀明",
        ),
    ]
    ev = {
        "anchor_msg_id": "b",
        "bot_error_quote": "gold_review_sheet_pilot.md 是你标注过的结果文件",
        "corrector_msg_id": "c",
        "corrector_role": "user",
        "what": "bot 把待填核对台误当成人工标注结果文件",
    }

    kept, dropped = validate_events([ev], rows)

    assert dropped == []
    assert len(kept) == 1


def test_guard_reanchors_bot_self_correction_to_human_trigger():
    rows = [
        _row("b", 10, "bot", "codex 走 litellm，所以 reasoning 能看", "lian-codex"),
        _row("u", 20, "user", "那你拉一条真实 codex call 的 litellm 日志看看", "张耀明"),
        _row("c", 30, "bot", "你说得对，我上一条说错了，codex 实际走 WebSocket", "lian-codex"),
    ]
    ev = _event(corrector_msg_id="c", corrector_role="bot")

    kept = _guard([ev], [], rows)

    assert len(kept) == 1
    assert kept[0]["corrector_msg_id"] == "u"
    assert kept[0]["corrector_role"] == "user"
    assert kept[0]["_corrector_reanchored"] == {"from": "c", "to": "u"}


def test_guard_reanchors_source_file_self_correction_to_human_trigger():
    rows = [
        _row(
            "b",
            10,
            "bot",
            "gold_review_sheet_pilot.md 是你标注过的结果文件，可以作为 #4 gold 的来源。",
            "lian-codex",
        ),
        _row(
            "u",
            20,
            "user",
            "这个文件我没标注过啊，你是不是搞错了？找到我标注的结果来说明我错了。",
            "张耀明",
        ),
        _row(
            "c",
            30,
            "bot",
            "我刚才搞错了，gold_review_sheet_pilot.md 只是核对台，不是你的标注结果。",
            "lian-codex",
        ),
    ]
    ev = {
        "anchor_msg_id": "b",
        "bot_error_quote": "gold_review_sheet_pilot.md 是你标注过的结果文件",
        "corrector_msg_id": "c",
        "corrector_role": "bot",
        "what": "bot 把待填核对台误当成人工标注结果文件",
    }

    kept = _guard([ev], [], rows)

    assert len(kept) == 1
    assert kept[0]["corrector_msg_id"] == "u"
    assert kept[0]["corrector_role"] == "user"
    assert kept[0]["_corrector_reanchored"] == {"from": "c", "to": "u"}


def test_guard_does_not_rescue_pure_bot_self_correction():
    rows = [
        _row("b", 10, "bot", "codex 走 litellm，所以 reasoning 能看", "lian-codex"),
        _row("c", 20, "bot", "我上一条说错了，codex 实际走 WebSocket", "lian-codex"),
    ]
    ev = _event(corrector_msg_id="c", corrector_role="bot")

    kept = _guard([ev], [], rows)

    assert kept == []
    assert ev["_drop"].startswith("corrector_role非user")


def test_guard_does_not_reanchor_summary_confirmation():
    rows = [
        _row("b", 10, "bot", "被裁剪的不是当前用户输入，而是历史上下文。", "lian-codex"),
        _row("u", 20, "user", "我梳理一下：实际上输入给模型的文本拿不到，是吗？", "张耀明"),
        _row("c", 30, "bot", "这三条都对，但漏了一条。", "lian-codex"),
    ]
    ev = {
        "anchor_msg_id": "b",
        "bot_error_quote": "被裁剪的不是当前用户输入",
        "corrector_msg_id": "c",
        "corrector_role": "bot",
        "what": "bot 后续补充漏项",
    }

    kept = _guard([ev], [], rows)

    assert kept == []
    assert ev["_drop"].startswith("corrector_role非user")


def test_reguard_cached_rows_applies_current_corrector_rules():
    rows = [
        _row("b", 10, "bot", "这个任务标为有返工，状态已经核过。", "lian-codex"),
        _row("u", 20, "user", "为什么“调取李博颉登录对话”状态为“有返工”，它不是只有一条消息吗？", "张耀明"),
        _row("c", 30, "bot", "你说得对，这是误标，单条消息不可能有返工。", "lian-codex"),
    ]
    cached = [{
        "anchor_msg_id": "b",
        "bot_error_quote": "标为有返工",
        "corrector_msg_id": "c",
        "corrector_role": "bot",
        "what": "bot 对单条消息错误标记为有返工",
        "_session": 24,
    }]

    kept = _reguard_cached_rows(cached, {24: rows})

    assert len(kept) == 1
    assert kept[0]["corrector_msg_id"] == "u"
    assert kept[0]["corrector_role"] == "user"
