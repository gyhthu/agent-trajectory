#!/usr/bin/env python3
"""task_resegment.verify_partition —— CHIEF『连续+全覆盖+不重叠』断言回路测试。
跑：python test_task_resegment.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from task_resegment import segment, verify_partition


def mk(ts, topic="A", bot="b1"):
    return {"ts": ts, "topic": topic, "bot": bot, "msg_id": f"m{ts}", "text": f"t{ts}"}


def expect_raise(fn, tag):
    try:
        fn()
    except AssertionError as e:
        assert "[重切校验失败" in str(e), f"{tag}: 异常信息缺标记: {e}"
        return
    raise SystemExit(f"FAIL {tag}: 期望 AssertionError 但未抛")


def test_hard_gap_split():
    rows = [mk(0), mk(60), mk(120), mk(120 + 2000), mk(120 + 2060)]
    segs = segment(rows, 1800)                      # 内部已 verify_partition，能跑通即过校验
    assert len(segs) == 2, f"硬gap应切2段, got {len(segs)}"
    assert sum(len(s["rows"]) for s in segs) == len(rows), "全覆盖失败"
    assert verify_partition(rows, segs)
    print("✓ 硬gap切分 + 全覆盖")


def test_topic_switch():
    rows = [mk(0, "A"), mk(60, "A"), mk(120, "B"), mk(180, "B")]
    segs = segment(rows, 1800)
    assert len(segs) == 2, f"话题持续切换应切2段, got {len(segs)}"
    assert verify_partition(rows, segs)
    print("✓ 话题持续切换切分")


def test_empty():
    assert segment([], 1800) == []
    assert verify_partition([], [])
    print("✓ 空输入")


def test_guard_missing():
    rows = [mk(0), mk(60), mk(120)]
    bad = [{"topic": "A", "rows": rows[:2], "start": 0}]            # 丢了第3条
    expect_raise(lambda: verify_partition(rows, bad), "漏切")
    print("✓ 护栏·漏切(全覆盖失败)")


def test_guard_dup():
    rows = [mk(0), mk(60)]
    bad = [{"topic": "A", "rows": [rows[0], rows[1], rows[1]], "start": 0}]  # 重复一条
    expect_raise(lambda: verify_partition(rows, bad), "重复")
    print("✓ 护栏·重复(不重叠失败)")


def test_guard_empty_seg():
    rows = [mk(0)]
    bad = [{"topic": "A", "rows": [], "start": 0},
           {"topic": "A", "rows": [rows[0]], "start": 0}]
    expect_raise(lambda: verify_partition(rows, bad), "空段")
    print("✓ 护栏·空段")


def test_guard_order():
    a, b = mk(0), mk(1000)
    bad = [{"topic": "A", "rows": [b], "start": 1000},
           {"topic": "A", "rows": [a], "start": 0}]                 # 段序倒置
    expect_raise(lambda: verify_partition([b, a], bad), "段序")
    print("✓ 护栏·段序错乱")


if __name__ == "__main__":
    test_hard_gap_split()
    test_topic_switch()
    test_empty()
    test_guard_missing()
    test_guard_dup()
    test_guard_empty_seg()
    test_guard_order()
    print("\n全部通过 ✓")
