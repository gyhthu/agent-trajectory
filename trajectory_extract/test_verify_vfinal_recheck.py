import verify_vfinal_recheck as V


def test_to_verify_row_preserves_audit_sources():
    row = {
        "bot_error_msg_id": "b",
        "bot_error_quote": "错句",
        "corrector_msg_id": "c",
        "t0_msg_id": "t0",
        "seed_corrector_msg_id": "seed",
        "focus_corrector_msg_id": "c",
        "what": "错在哪",
        "sources": ["session_full_only_20260710"],
        "seed_whats": ["初筛说明"],
        "kind": "counter",
    }

    out = V._to_verify_row(row)

    assert out["sources"] == ["session_full_only_20260710"]
    assert out["seed_whats"] == ["初筛说明"]
    assert out["kind"] == "counter"
