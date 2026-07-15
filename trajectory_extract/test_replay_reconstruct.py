#!/usr/bin/env python3
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replay_reconstruct as R


def test_scene_from_snapshot_replaces_only_t0_instruction():
    snap = {
        "original_instruction": "原始指令",
        "prompts": {
            "baseline": "前文包含原始指令这个词\n\n【原始指令】\n原始指令\n\n请按当时状态完成这条原始指令。"
        },
    }

    scene = R._scene_from_snapshot(snap, "重构指令")

    assert "【原始指令】\n重构指令" in scene
    assert "前文包含原始指令这个词" in scene


def test_load_rows_can_read_snapshots(tmp_path):
    path = tmp_path / "snaps.jsonl"
    path.write_text(
        json.dumps({
            "status": "ok",
            "original_instruction": "问法",
            "comment_for_replay": "准则",
        }, ensure_ascii=False) + "\n" +
        json.dumps({
            "status": "missing_instruction_event",
            "original_instruction": "坏行",
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    args = R.parse_args(["--source", "snapshots", "--snapshots", str(path)])

    rows, by_instr = R._load_rows(args)

    assert len(rows) == 1
    assert rows[0]["original_instruction"] == "问法"
    assert rows[0]["comment_for_replay"] == "准则"
    assert by_instr[R.norm("问法")]["status"] == "ok"


def test_classify_legs_requires_err_hint_for_correction_success():
    result = R.classify_legs(
        base_failed=True,
        placebo_failed=True,
        treat_failed=False,
        has_err_hint=False,
    )

    assert result["baseline_reproduced"] is True
    assert result["passed"] is False
    assert result["signal_isolated"] is False
    assert "无合法纠错要素" in result["note"]


def test_classify_legs_marks_signal_isolated_with_err_hint():
    result = R.classify_legs(
        base_failed=True,
        placebo_failed=True,
        treat_failed=False,
        has_err_hint=True,
    )

    assert result["passed"] is True
    assert result["signal_isolated"] is True
    assert result["confounded"] is False
