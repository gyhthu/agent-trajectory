#!/usr/bin/env python3
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pre_instruction_snapshot as P
import principle_distill as PD


def _e(mid, ts, text, role="user", parent_id=""):
    return {
        "msg_id": mid,
        "ts": ts,
        "role": role,
        "name": role,
        "parent_id": parent_id,
        "text": text,
    }


def _fake_distill(general="报数前先核实来源，没依据就说未验证", domain=None, label=None):
    """Deterministic stand-in for the LLM distiller (no network in tests)."""
    def _f(what, model=None, client=None):
        return {
            "source_what": what,
            "general": general,
            "domain": domain,
            "domain_label": label,
            "red_line": {"general_leak": [], "domain_leak": []},
            "raw": None,
        }
    return _f


def test_build_snapshot_uses_only_same_lane_before_t0(tmp_path, monkeypatch):
    state = {
        "frozen_tasks": [{
            "events": [
                _e("main-before", 10, "主线前文"),
                _e("side-before", 11, "旁路线前文"),
            ],
        }],
        "active_tail_events": [
            _e("fix", 20, "原始指令"),
            _e("after", 30, "T0 后纠正"),
        ],
    }
    rewrite = {
        "rewritten": [{
            "lane": "MAIN",
            "instruction_to_fix_msg": "fix",
            "instruction_to_fix_text": "原始指令",
            "rewritten": "完备 comment",
            "corrections": [{"anchor": "after", "what": "后验纠正"}],
        }]
    }
    state_path = tmp_path / "state.json"
    rewrite_path = tmp_path / "rewrite.json"
    state_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    rewrite_path.write_text(json.dumps(rewrite, ensure_ascii=False), encoding="utf-8")

    def fake_lanes(events):
        return {"main-before": "MAIN", "side-before": "s:2", "fix": "MAIN", "after": "MAIN"}

    monkeypatch.setattr(P, "_lane_short_map", fake_lanes)
    monkeypatch.setattr(P, "_git_commit_before", lambda repo, ts: {"repo": repo, "status": "ok"})
    monkeypatch.setattr(P.pd, "_client", lambda: None)
    monkeypatch.setattr(P.pd, "distill_one", _fake_distill(general="通用准则A"))

    rows = P.build_snapshots(
        rewrite_path=str(rewrite_path),
        state_path=str(state_path),
        chunk_paths=[],
        context_limit=10,
        repo_paths=["/repo"],
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "ok"
    assert row["t0"]["msg_id"] == "fix"
    assert [e["msg_id"] for e in row["context"]["events"]] == ["main-before"]
    # baseline never carries the comment; original text stays original (not rewritten).
    assert "T0 后纠正" not in row["prompts"]["baseline"]
    assert "通用准则A" not in row["prompts"]["baseline"]
    # B route: comment_for_replay is the distilled GENERAL principle, not source_comment.
    assert row["comment_for_replay"] == "通用准则A"
    assert "通用准则A" in row["prompts"]["with_comment"]
    assert row["source_comment"] == "完备 comment"  # kept for audit, not injected
    assert "完备 comment" not in row["prompts"]["with_comment"]
    assert row["replay_principles"]["general"] == ["通用准则A"]


def test_build_snapshot_from_vfinal_kept_rows(tmp_path, monkeypatch):
    state = {
        "frozen_tasks": [{
            "events": [
                _e("before", 10, "前文"),
                _e("t0", 20, "原始问题"),
                _e("bot", 30, "错误回答", role="bot"),
                _e("c", 40, "用户纠正", role="user"),
            ],
        }],
        "active_tail_events": [],
    }
    kept = {
        "_t0_msg_id": "t0",
        "anchor_msg_id": "bot",
        "corrector_msg_id": "c",
        "corrector_role": "user",
        "bot_error_quote": "错误回答",
        "what": "bot 把 A 说成 B",
        "task": "vfinal_recheck",
    }
    state_path = tmp_path / "state.json"
    kept_path = tmp_path / "kept.jsonl"
    state_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    kept_path.write_text(json.dumps(kept, ensure_ascii=False) + "\n", encoding="utf-8")

    monkeypatch.setattr(P, "_lane_short_map", lambda events: {
        "before": "MAIN", "t0": "MAIN", "bot": "MAIN", "c": "MAIN",
    })
    monkeypatch.setattr(P, "_git_commit_before", lambda repo, ts: {"repo": repo, "status": "ok"})
    monkeypatch.setattr(P.pd, "_client", lambda: None)
    monkeypatch.setattr(P.pd, "distill_one", _fake_distill(general="回答前先核对事实"))

    rows = P.build_snapshots_from_corrections(
        corrections_path=str(kept_path),
        state_path=str(state_path),
        chunk_paths=[],
        context_limit=10,
        repo_paths=["/repo"],
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "ok"
    assert row["source"] == "vfinal_recheck_verify_kept"
    assert row["t0"]["msg_id"] == "t0"
    assert row["original_instruction"] == "原始问题"
    assert [e["msg_id"] for e in row["context"]["events"]] == ["before"]
    assert row["corrections"][0]["anchor"] == "bot"
    assert row["corrections"][0]["corrector"] == "c"
    assert row["comment_for_replay"] == "回答前先核对事实"
    assert "回答前先核对事实" in row["prompts"]["with_comment"]


def test_b_route_injects_principle_not_source_comment(tmp_path, monkeypatch):
    """Source comment carries the leaky 186/276/318; injected principle must not."""
    state = {
        "frozen_tasks": [],
        "active_tail_events": [
            _e("fix", 20, "你看一个llm精分后，切分的结果差距大不大"),
            _e("bot-after", 26, "编造了 186 个宏观大簇，276 -> 318 个子需求", role="bot"),
            _e("correction", 30, "这是凭空捏造", role="user"),
        ],
    }
    rewrite = {
        "rewritten": [{
            "lane": "MAIN",
            "instruction_to_fix_msg": "fix",
            "instruction_to_fix_text": "你看一个llm精分后，切分的结果差距大不大",
            "rewritten": "重点验证张耀明编造的\"186任务/276→318子需求\"等幻觉数据是否凭空捏造",
            "corrections": [{
                "anchor": "bot-after",
                "what": "zym编造'186任务/276→318子需求'的精分对比数据(幻觉),被戳穿凭空捏造",
                "corrector": "张耀明",
            }],
        }]
    }
    state_path = tmp_path / "state.json"
    rewrite_path = tmp_path / "rewrite.json"
    state_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    rewrite_path.write_text(json.dumps(rewrite, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr(P, "_lane_short_map", lambda events: {
        "fix": "MAIN", "bot-after": "MAIN", "correction": "MAIN",
    })
    monkeypatch.setattr(P, "_git_commit_before", lambda repo, ts: {"repo": repo, "status": "ok"})
    monkeypatch.setattr(P.pd, "_client", lambda: None)
    monkeypatch.setattr(P.pd, "distill_one", _fake_distill(
        general="报数据前必须基于真实产物，拿不出依据就说未验证",
        domain="切分产出的簇不等于真任务，报数前先按语义归并再点数",
        label="切分",
    ))

    rows = P.build_snapshots(
        rewrite_path=str(rewrite_path), state_path=str(state_path),
        chunk_paths=[], context_limit=10, repo_paths=["/repo"],
    )
    row = rows[0]
    assert row["source_comment"].startswith("重点验证")  # leaky source kept for audit
    for leak in ("186", "276", "318"):
        assert leak not in row["comment_for_replay"]
        assert leak not in row["prompts"]["with_comment"]
    assert "报数据前必须基于真实产物" in row["comment_for_replay"]
    assert row["post_t0_failure_evidence"]  # concrete evidence captured, not injected
    assert row["replay_principles"]["domain"][0]["label"] == "切分"
    assert row["leakage_guard"]["distill_audit"]


def test_red_line_bans_only_instance_numbers():
    """Red line = source's own number-tokens, not all digits."""
    what = "把segment_history切出的13个碎簇当成13个真任务,实际只有2-3个真任务"
    assert PD.leaked_number_tokens(what) == {"13", "2-3"}
    # a principle that leaks the instance answer is caught
    assert PD.violates_red_line("这批只有2-3个不是13个", what) == ["13", "2-3"]
    # a clean generalized principle passes
    assert PD.violates_red_line("报数前先按语义归并碎簇再点数，不预设数量", what) == []
    # benign wording with a non-source numeral is NOT a false positive
    assert PD.violates_red_line("第一步先核实来源", what) == []


def test_missing_instruction_event_is_reported(tmp_path, monkeypatch):
    state_path = tmp_path / "state.json"
    rewrite_path = tmp_path / "rewrite.json"
    state_path.write_text(json.dumps({"frozen_tasks": [], "active_tail_events": []}), encoding="utf-8")
    rewrite_path.write_text(
        json.dumps({"rewritten": [{"instruction_to_fix_msg": "missing"}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(P.pd, "_client", lambda: None)
    rows = P.build_snapshots(str(rewrite_path), str(state_path), chunk_paths=[], repo_paths=[])
    assert rows[0]["status"] == "missing_instruction_event"
