#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import raw_context_replay as RC
import raw_context_verify as RV


def _e(mid, ts, text, role="user", parent_id=""):
    return {
        "msg_id": mid,
        "ts": ts,
        "role": role,
        "name": role,
        "parent_id": parent_id,
        "text": text,
    }


def _snapshot(t0="t0", anchor="bot", corrector="c"):
    return {
        "status": "ok",
        "group_id": "g",
        "t0": {"msg_id": t0},
        "original_instruction": "原始问题",
        "comment_for_replay": "先核对事实",
        "corrections": [{"anchor": anchor, "corrector": corrector, "what": "bot 说错了"}],
        "leakage_guard": {"distill_audit": []},
    }


def test_build_manifest_uses_parent_thread_before_t0(monkeypatch):
    events = [
        _e("root", 10, "根问题"),
        _e("other", 11, "别的线"),
        _e("reply", 12, "同线追问", parent_id="root"),
        _e("t0", 20, "原始问题", parent_id="reply"),
        _e("bot", 30, "错误回答", role="bot", parent_id="t0"),
        _e("c", 40, "用户纠正", parent_id="bot"),
    ]
    monkeypatch.setattr(RC.PIS, "load_events", lambda: events)
    monkeypatch.setattr(RC.RL, "assign_lanes", lambda evs: (
        {e["msg_id"]: "g" for e in evs},
        {e["msg_id"]: "main" for e in evs},
    ))
    monkeypatch.setattr(RC.RL, "lane_short", lambda lane: "MAIN")

    rows = RC.build_manifest([_snapshot()], context_limit=10)

    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "ok"
    assert row["raw_context_quality"] == RC.QUALITY_EXACT_THREAD
    assert row["context"]["event_ids"] == ["root", "reply"]
    assert row["context"]["materialized"] is False
    assert "events" not in row["context"]
    assert "rendered" not in row["context"]
    assert "baseline_scene" not in row["legs"]
    assert row["system_ref"] == "_BASE_SYS"
    assert "system" not in row
    assert row["legs"]["rewrite_generation"] == "pending"
    assert row["legs"]["placebo_user"] is None


def test_build_manifest_falls_back_to_same_session_window(monkeypatch):
    events = [
        _e("before", 10, "前文"),
        _e("t0", 20, "原始问题"),
        _e("after", 30, "后文"),
    ]
    monkeypatch.setattr(RC.PIS, "load_events", lambda: events)
    monkeypatch.setattr(RC.RL, "assign_lanes", lambda evs: (
        {e["msg_id"]: "g" for e in evs},
        {e["msg_id"]: "main" for e in evs},
    ))
    monkeypatch.setattr(RC.RL, "lane_short", lambda lane: "MAIN")

    rows = RC.build_manifest([_snapshot(anchor="after", corrector="missing")], context_limit=10)

    assert rows[0]["raw_context_quality"] == RC.QUALITY_SAME_SESSION
    assert rows[0]["context"]["event_ids"] == ["before"]
    assert rows[0]["context"]["events_preview"][0]["text_preview"] == "前文"
    assert "rendered" not in rows[0]["context"]


def test_build_manifest_marks_leaked_correction_as_review(monkeypatch):
    events = [
        _e("bot", 10, "提前混入的错误回答", role="bot"),
        _e("t0", 20, "原始问题"),
    ]
    monkeypatch.setattr(RC.PIS, "load_events", lambda: events)
    monkeypatch.setattr(RC.RL, "assign_lanes", lambda evs: (
        {e["msg_id"]: "g" for e in evs},
        {e["msg_id"]: "main" for e in evs},
    ))
    monkeypatch.setattr(RC.RL, "lane_short", lambda lane: "MAIN")

    rows = RC.build_manifest([_snapshot(anchor="bot", corrector="c")], context_limit=10)

    assert rows[0]["status"] == "needs_review"
    assert rows[0]["raw_context_quality"] == RC.QUALITY_MISSING
    assert rows[0]["raw_context_audit"]["leakage_flags"]["b_or_c_event_ids"] == ["bot"]


def test_raw_context_verify_parse_json_normalizes_bad_quality():
    parsed = RV._parse_json('{"ok": true, "quality": "bad", "reject_reason": "", "leakage_flags": []}')

    assert parsed["ok"] is True
    assert parsed["quality"] == RC.QUALITY_MISSING


def test_build_manifest_can_opt_in_to_full_context(monkeypatch):
    events = [
        _e("before", 10, "前文"),
        _e("t0", 20, "原始问题"),
    ]
    monkeypatch.setattr(RC.PIS, "load_events", lambda: events)
    monkeypatch.setattr(RC.RL, "assign_lanes", lambda evs: (
        {e["msg_id"]: "g" for e in evs},
        {e["msg_id"]: "main" for e in evs},
    ))
    monkeypatch.setattr(RC.RL, "lane_short", lambda lane: "MAIN")

    rows = RC.build_manifest([_snapshot()], context_limit=10, embed_full_context=True)

    assert rows[0]["context"]["materialized"] is True
    assert rows[0]["context"]["rendered"] == "M1 [user/user] 前文"
    assert "baseline_scene" in rows[0]["legs"]
    assert "前文" in rows[0]["legs"]["baseline_scene"]


def test_write_jsonl_rejects_large_rows(tmp_path):
    out = tmp_path / "manifest.jsonl"

    try:
        RC.write_jsonl([{"big": "x" * 50}], str(out), max_line_chars=20)
    except ValueError as exc:
        assert "manifest row 0" in str(exc)
    else:
        raise AssertionError("expected oversized row to fail")


def test_slim_manifest_previews_only_tail_events(monkeypatch):
    events = [_e(f"before-{i}", i, f"前文{i}") for i in range(5)]
    events.append(_e("t0", 10, "原始问题"))
    monkeypatch.setattr(RC.PIS, "load_events", lambda: events)
    monkeypatch.setattr(RC.RL, "assign_lanes", lambda evs: (
        {e["msg_id"]: "g" for e in evs},
        {e["msg_id"]: "main" for e in evs},
    ))
    monkeypatch.setattr(RC.RL, "lane_short", lambda lane: "MAIN")

    rows = RC.build_manifest([_snapshot()], context_limit=10, preview_events=2)

    assert rows[0]["context"]["event_ids"] == ["before-0", "before-1", "before-2", "before-3", "before-4"]
    assert [e["msg_id"] for e in rows[0]["context"]["events_preview"]] == ["before-3", "before-4"]
