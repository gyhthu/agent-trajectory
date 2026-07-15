import build_gold_review_sheet as G


def _ev(i, text, role="user", name="张耀明", parent_id=""):
    return {
        "msg_id": f"m{i}",
        "role": role,
        "name": name,
        "text": text,
        "parent_id": parent_id,
    }


def test_transcript_includes_parent_chain_outside_local_window():
    evs = [
        _ev(0, "真实原始指令"),
        _ev(1, "无关消息一"),
        _ev(2, "无关消息二"),
        _ev(3, "无关消息三"),
        _ev(4, "无关消息四"),
        _ev(5, "好了吗", parent_id="m0"),
        _ev(6, "bot 错误句", role="bot", name="lian-server", parent_id="m5"),
        _ev(7, "它不是只有一条消息吗？", parent_id="m6"),
        _ev(8, "你说得对，刚才误标了", role="bot", name="lian-server", parent_id="m7"),
    ]
    idx = {e["msg_id"]: i for i, e in enumerate(evs)}

    lines, shown = G._transcript(evs, idx, "m5", "m6", "m8", pad_before=0, pad_after=0)

    rendered = "\n".join(lines)
    assert "真实原始指令" in rendered
    assert "省略中间 4 条非回复链消息" in rendered
    assert {"m0", "m5", "m6", "m8"} <= shown


def test_raw_and_t0_maps_prefer_guarded_rows_and_lookup_t0_by_anchor():
    guarded = {
        "anchor_msg_id": "b_guarded",
        "corrector_msg_id": "c",
        "bot_error_quote": "护栏后的错句",
    }
    stale_raw = {
        "anchor_msg_id": "b_raw",
        "corrector_msg_id": "c",
        "bot_error_quote": "旧 raw 错句",
    }
    t0 = {"msg_id": "t0"}
    raw_by_c, t0_by_anchor = G._build_raw_and_t0_maps(
        [guarded, stale_raw],
        [{"t0": t0, "corrections": [{"anchor": "b_guarded"}]}],
    )

    assert raw_by_c["c"] is guarded
    assert t0_by_anchor[raw_by_c["c"]["anchor_msg_id"]] is t0


def test_t0_map_keeps_legacy_corrector_key_for_fallback():
    t0 = {"msg_id": "t0"}
    _raw_by_c, t0_by_anchor = G._build_raw_and_t0_maps(
        [],
        [{"t0": t0, "corrections": [{"anchor": "c"}]}],
    )

    assert t0_by_anchor["c"] is t0


def test_eligible_review_requires_human_c_and_bot_b():
    by_id = {
        "b": _ev("b", "bot 错误句", role="bot", name="lian-server"),
        "u": _ev("u", "真人纠正"),
        "bot_c": _ev("bot_c", "我刚才说错了", role="bot", name="lian-server"),
    }
    good = {"cid": "u", "raw": {"anchor_msg_id": "b"}}
    bad_c = {"cid": "bot_c", "raw": {"anchor_msg_id": "b"}}

    assert G._eligible_for_human_review(good, by_id) == (True, "")
    assert G._eligible_for_human_review(bad_c, by_id) == (False, "C_is_bot")


def test_review_dedupe_key_uses_reply_roots():
    by_id = {
        "root": _ev("root", "原始问题"),
        "b1": _ev("b1", "bot 错一", role="bot", name="lian-server", parent_id="root"),
        "u1": _ev("u1", "纠正一", parent_id="b1"),
        "b2": _ev("b2", "bot 错二", role="bot", name="lian-server", parent_id="root"),
        "u2": _ev("u2", "纠正二", parent_id="b2"),
    }
    c1 = {"cid": "u1", "raw": {"anchor_msg_id": "b1"}}
    c2 = {"cid": "u2", "raw": {"anchor_msg_id": "b2"}}

    assert G._review_dedupe_key(c1, by_id) == G._review_dedupe_key(c2, by_id)


def test_reviewed_overlap_catches_seen_msg_root_and_text():
    events = [
        _ev("root", "原始问题里有一段足够长的唯一文本用于去重"),
        _ev("b", "bot 说错的一段足够长的唯一文本", role="bot", name="lian-server", parent_id="mroot"),
        _ev("u", "真人纠正的一段足够长的唯一文本", parent_id="mb"),
        _ev("other", "另一条完全不同但也足够长的文本用于覆盖文本指纹去重"),
    ]
    by_id = {e["msg_id"]: e for e in events}

    seen_msg = {"msg_ids": {"mu"}, "roots": set(), "texts": set()}
    assert G._reviewed_overlap_reason({"cid": "mu", "raw": {"anchor_msg_id": "mb"}, "t0": {}}, by_id, seen_msg).startswith("seen_msg:")

    seen_root = {"msg_ids": set(), "roots": {"mroot"}, "texts": set()}
    assert G._reviewed_overlap_reason({"cid": "mu", "raw": {"anchor_msg_id": "mb"}, "t0": {}}, by_id, seen_root).startswith("seen_root:")

    seen_text = {
        "msg_ids": set(),
        "roots": set(),
        "texts": {G._text_fingerprint(by_id["mother"]["text"])},
    }
    assert G._reviewed_overlap_reason({"cid": "mother", "raw": {}, "t0": {}}, by_id, seen_text) == "seen_text"
