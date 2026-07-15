import recheck_user_corrections_pool as R


def _event(mid, role, text, parent_id=""):
    name = "lian-server" if role == "bot" else role
    return {"msg_id": mid, "role": role, "name": name, "text": text, "parent_id": parent_id}


def test_validate_pick_accepts_ordered_user_bot_user_with_quote():
    by_id = {
        "t0": _event("t0", "user", "请做 A"),
        "b": _event("b", "bot", "A 已经完成，不需要再跑"),
        "c": _event("c", "user", "不对，A 没完成"),
    }
    idx = {"t0": 1, "b": 2, "c": 3}

    ok, reason = R._validate_pick({
        "verdict": "ACCEPT",
        "t0_msg_id": "t0",
        "bot_error_msg_id": "b",
        "corrector_msg_id": "c",
        "bot_error_quote": "A 已经完成",
    }, by_id, idx)

    assert ok
    assert reason == ""


def test_validate_pick_rejects_bad_order_even_when_roles_match():
    by_id = {
        "t0": _event("t0", "user", "请做 A"),
        "b": _event("b", "bot", "A 已经完成，不需要再跑"),
        "c": _event("c", "user", "不对，A 没完成"),
    }
    idx = {"t0": 1, "c": 2, "b": 3}

    ok, reason = R._validate_pick({
        "verdict": "ACCEPT",
        "t0_msg_id": "t0",
        "bot_error_msg_id": "b",
        "corrector_msg_id": "c",
        "bot_error_quote": "A 已经完成",
    }, by_id, idx)

    assert not ok
    assert reason == "bad_order"


def test_validate_pick_matches_quote_after_markdown_emphasis_removed():
    by_id = {
        "t0": _event("t0", "user", "哪些输入拿得到"),
        "b": _event("b", "bot", '拿不到原件，我**存的是"次优替代品"**'),
        "c": _event("c", "user", "应该有 system prompt 吧？"),
    }
    idx = {"t0": 1, "b": 2, "c": 3}

    ok, reason = R._validate_pick({
        "verdict": "ACCEPT",
        "t0_msg_id": "t0",
        "bot_error_msg_id": "b",
        "corrector_msg_id": "c",
        "bot_error_quote": '拿不到原件，我存的是"次优替代品"',
    }, by_id, idx)

    assert ok
    assert reason == ""


def test_resolve_seed_focus_moves_bot_repair_seed_to_human_trigger():
    evs = [
        _event("t0", "user", "怎么处理同任务多次纠正"),
        _event("b", "bot", "每次改写只改一个侧面，各算独立轨迹"),
        _event("c", "user", "PPT 第二页和第三页最好一次说清楚，这样拆很奇怪"),
        _event("ack", "bot", "👀 已收到，开始处理。"),
        _event("repair", "bot", "你说得对，我上一条方向正好反了，应逆向合成一条完备指令"),
    ]
    by_id = {event["msg_id"]: event for event in evs}
    idx = {event["msg_id"]: i for i, event in enumerate(evs)}

    focus = R._resolve_seed_focus("repair", "b", by_id, idx, evs)

    assert focus["focus_corrector_msg_id"] == "c"
    assert focus["repair_msg_id"] == "repair"


def test_resolve_seed_focus_treats_question_challenge_as_human_trigger():
    evs = [
        _event("t0", "user", "transcript 里哪些输入拿得到"),
        _event("b", "bot", "system prompt 拿不到"),
        _event("c", "user", "antigravity 自己落盘的 transcript 应该有 system prompt 吧？", parent_id="b"),
        _event("repair", "bot", "你说得对，transcript 里确实有 system prompt"),
    ]
    by_id = {event["msg_id"]: event for event in evs}
    idx = {event["msg_id"]: i for i, event in enumerate(evs)}

    focus = R._resolve_seed_focus("repair", "b", by_id, idx, evs)

    assert focus["focus_corrector_msg_id"] == "c"
    assert focus["repair_msg_id"] == "repair"


def test_repair_sys_t0_pick_moves_to_nearest_user_before_bot_error():
    evs = [
        _event("u1", "user", "哪些输入拿得到"),
        _event("ack", "bot", "👀 已收到，开始处理。"),
        _event("b", "bot", "system prompt 拿得到一半"),
        _event("c", "user", "应该有 system prompt 吧？"),
    ]
    by_id = {event["msg_id"]: event for event in evs}
    idx = {event["msg_id"]: i for i, event in enumerate(evs)}

    repaired = R._repair_sys_t0_pick({
        "verdict": "ACCEPT",
        "t0_msg_id": "ack",
        "bot_error_msg_id": "b",
        "corrector_msg_id": "c",
        "bot_error_quote": "system prompt 拿得到一半",
    }, by_id, idx, evs)

    assert repaired["t0_msg_id"] == "u1"
    assert repaired["t0_repaired_from"] == "ack"


def test_repair_parent_bot_error_pick_moves_to_replied_bot_when_quote_matches():
    evs = [
        _event("t0", "user", "transcript 里哪些输入拿得到"),
        _event("b", "bot", "拿不到原件，我存的是次优替代品"),
        _event("other_b", "bot", "工具 schema 半个"),
        _event("c", "user", "antigravity 自己落盘的 transcript 应该有 system prompt 吧？", parent_id="b"),
    ]
    by_id = {event["msg_id"]: event for event in evs}
    idx = {event["msg_id"]: i for i, event in enumerate(evs)}

    repaired = R._repair_parent_bot_error_pick({
        "verdict": "ACCEPT",
        "t0_msg_id": "t0",
        "bot_error_msg_id": "other_b",
        "corrector_msg_id": "c",
        "bot_error_quote": "拿不到原件，我存的是次优替代品",
    }, by_id, idx)

    assert repaired["bot_error_msg_id"] == "b"
    assert repaired["bot_error_repaired_from"] == "other_b"


def test_validate_pick_rejects_question_trigger_on_different_reply_line():
    by_id = {
        "t0": _event("t0", "user", "哪些输入拿得到"),
        "b": _event("b", "bot", "system prompt 拿不到"),
        "unrelated": _event("unrelated", "bot", "另一个任务已经完成"),
        "c": _event("c", "user", "应该有 system prompt 吧？", parent_id="unrelated"),
    }
    idx = {"t0": 1, "b": 2, "unrelated": 3, "c": 4}

    ok, reason = R._validate_pick({
        "verdict": "ACCEPT",
        "t0_msg_id": "t0",
        "bot_error_msg_id": "b",
        "corrector_msg_id": "c",
        "bot_error_quote": "system prompt 拿不到",
        "kind": "user_trigger",
    }, by_id, idx)

    assert not ok
    assert reason == "C_not_same_line_with_B"


def test_repair_t0_pick_moves_off_correction_line_to_prior_instruction():
    evs = [
        _event("u1", "user", "先说明 transcript 里哪些输入拿得到"),
        _event("old_b", "bot", "可以"),
        _event("bad_t0", "user", "不是这个，你应该看 transcript", parent_id="old_b"),
        _event("b", "bot", "system prompt 拿不到"),
        _event("c", "user", "应该有 system prompt 吧？", parent_id="b"),
    ]
    by_id = {event["msg_id"]: event for event in evs}
    idx = {event["msg_id"]: i for i, event in enumerate(evs)}

    repaired = R._repair_t0_pick({
        "verdict": "ACCEPT",
        "t0_msg_id": "bad_t0",
        "bot_error_msg_id": "b",
        "corrector_msg_id": "c",
        "bot_error_quote": "system prompt 拿不到",
    }, by_id, idx, evs)

    assert repaired["t0_msg_id"] == "u1"
    assert repaired["t0_repaired_from"] == "bad_t0"


def test_repair_t0_pick_moves_off_bot_error_to_nearest_user_instruction():
    evs = [
        _event("old_u", "user", "先找历史上文"),
        _event("old_b", "bot", "已处理"),
        _event("u1", "user", "那你做吧，以及注意token消耗"),
        _event("b", "bot", "我会直接做旧裁决迁移成新版行号真值"),
        _event("c", "user", "奇怪啊，这个文件我没标注过啊，你是不是搞错了", parent_id="later_b"),
    ]
    by_id = {event["msg_id"]: event for event in evs}
    idx = {event["msg_id"]: i for i, event in enumerate(evs)}

    repaired = R._repair_t0_pick({
        "verdict": "ACCEPT",
        "t0_msg_id": "b",
        "bot_error_msg_id": "b",
        "corrector_msg_id": "c",
        "bot_error_quote": "我会直接做旧裁决迁移成新版行号真值",
    }, by_id, idx, evs)

    assert repaired["t0_msg_id"] == "u1"
    assert repaired["t0_repaired_from"] == "b"


def test_repair_known_bot_error_pick_uses_guarded_anchor_when_model_drifts():
    by_id = {
        "t0": _event("t0", "user", "那你做吧，以及注意token消耗"),
        "b": _event("b", "bot", "我会直接做旧裁决迁移成新版行号真值"),
        "wrong_b": _event("wrong_b", "bot", "我不能把下降 70%-90% 当核实结论说"),
        "c": _event("c", "user", "奇怪啊，这个文件我没标注过啊，你是不是搞错了"),
    }
    idx = {"t0": 1, "b": 2, "wrong_b": 3, "c": 4}

    repaired = R._repair_known_bot_error_pick({
        "verdict": "ACCEPT",
        "t0_msg_id": "t0",
        "bot_error_msg_id": "wrong_b",
        "corrector_msg_id": "c",
        "bot_error_quote": "我不能把下降 70%-90% 当核实结论说",
    }, {
        "anchor_msg_id": "b",
        "bot_error_quote": "我会直接做旧裁决迁移成新版行号真值",
    }, by_id, idx)

    assert repaired["bot_error_msg_id"] == "b"
    assert repaired["bot_error_quote"] == "我会直接做旧裁决迁移成新版行号真值"
    assert repaired["bot_error_repaired_from"] == "wrong_b"


def test_lock_corrector_pick_keeps_focus_c_deterministic():
    repaired = R._lock_corrector_pick({
        "verdict": "ACCEPT",
        "t0_msg_id": "t0",
        "bot_error_msg_id": "b",
        "corrector_msg_id": "bot_admits_error",
        "bot_error_quote": "gold_review_sheet_pilot.md 是你标注过的结果文件",
    }, "human_questions_source")

    assert repaired["corrector_msg_id"] == "human_questions_source"
    assert repaired["corrector_repaired_from"] == "bot_admits_error"


def test_validate_pick_allows_direct_counter_even_when_reply_parent_is_later_bot():
    by_id = {
        "t0": _event("t0", "user", "那你做吧，以及注意token消耗"),
        "b": _event("b", "bot", "我会直接做旧裁决迁移成新版行号真值"),
        "later_b": _event("later_b", "bot", "文件在 gold_review_sheet_pilot.md"),
        "c": _event("c", "user", "奇怪啊，这个文件我没标注过啊，你是不是搞错了", parent_id="later_b"),
    }
    idx = {"t0": 1, "b": 2, "later_b": 3, "c": 4}

    ok, reason = R._validate_pick({
        "verdict": "ACCEPT",
        "t0_msg_id": "t0",
        "bot_error_msg_id": "b",
        "corrector_msg_id": "c",
        "bot_error_quote": "我会直接做旧裁决迁移成新版行号真值",
        "kind": "user_trigger",
    }, by_id, idx)

    assert ok
    assert reason == ""
