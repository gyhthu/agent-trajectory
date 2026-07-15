#!/usr/bin/env python3
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import incremental_segment as I
import llm_segment as L
from regex_anonymizer import RegexAnonymizer


def _e(ts, mid, role="user", parent=""):
    return {
        "ts": ts, "role": role, "who": role, "name": role,
        "kind": None, "msg_id": mid, "parent_id": parent,
        "task_id": "", "thread_id": "", "text": f"〔数据处理清洗〕{mid}",
    }


def test_split_task_sections():
    md = "# h\n\n---\n\n## 任务 1　A\nx\n\n## 任务 2　B\ny\n"
    assert I.split_task_sections(md) == ["## 任务 1　A\nx\n", "## 任务 2　B\ny\n"]


def test_reply_chain_reopens_old_frozen_task():
    frozen = [
        {"member_msg_ids": ["m1"], "last_ts": 100, "terminal": True},
        {"member_msg_ids": ["m2"], "last_ts": 200, "terminal": True},
    ]
    assert I._reopened_frozen_indexes(frozen, [_e(1000, "m3", parent="m2")]) == {1}


def test_hard_freeze_respects_reopen_and_thaw_window():
    frozen = [
        {"member_msg_ids": ["old"], "last_ts": 100, "terminal": True},
        {"member_msg_ids": ["recent"], "last_ts": 900, "terminal": True},
        {"member_msg_ids": ["open"], "last_ts": 50, "terminal": False},
    ]
    hard = I._hard_frozen_indexes(frozen, watermark_ts=1000, thaw_hours=0, reopened={0})
    assert hard == {1}


def test_confirmation_status_by_last_speaker():
    # 决策2：最后一条是 bot、delivery 未知 → 回退确定性：已完成（沉默即认可）
    assert I.confirmation_status([_e(1, "u1"), _e(2, "b1", role="bot")], True) == "已完成"
    # 决策1：最后一条是用户、非收尾语 → bot 没接 → 未确认
    assert I.confirmation_status([_e(1, "u1")], True) == "未确认"
    # 用户收尾语 → 已确认完成
    assert I.confirmation_status([_e(1, "u1"), _e(2, "搞定了", role="user")], True) == "已完成"
    # 未终结：观测窗右沿
    assert I.confirmation_status([_e(1, "u1")], False) == "未终结"


def test_confirmation_status_consumes_delivery():
    # 乙路线：tail 是 bot，但 LLM 判 delivery=仅占位/无交付 → 未确认（修旧版「bot 收尾即已完成」误判）
    bot_tail = [_e(1, "u1"), _e(2, "b1", role="bot")]
    assert I.confirmation_status(bot_tail, True, delivery="仅占位") == "未确认"
    assert I.confirmation_status(bot_tail, True, delivery="无交付") == "未确认"
    # tail 是 bot 且 LLM 判已交付 → 已完成
    assert I.confirmation_status(bot_tail, True, delivery="已交付") == "已完成"
    # delivery 未知(None) → 回退确定性（bot tail → 已完成），向后兼容旧 state
    assert I.confirmation_status(bot_tail, True, delivery=None) == "已完成"
    # 用户收尾语优先于 delivery（用户明确确认就是已完成）——此处 tail 是用户、delivery 不参与 bot 分支
    assert I.confirmation_status([_e(1, "u1"), _e(2, "搞定了", role="user")], True,
                                 delivery="仅占位") == "已完成"


def test_render_decompose_start_idx_for_single_message_tasks():
    clusters = [[_e(1, "m1")], [_e(2, "m2")]]
    metas = [{"title": "A", "goal": "ga", "segs": [1]},
             {"title": "B", "goal": "gb", "segs": [2]}]
    md, total, _, task_subreqs = L.render_decompose("g", "w", clusters, metas, RegexAnonymizer(), start_idx=4)
    assert total == 2
    assert "## 任务 4　A" in md
    assert "## 任务 5　B" in md
    # C1：单消息任务也产出结构化子需求→msg_id 映射（member=该消息本身）
    assert [t["subreqs"][0]["member_msg_ids"] for t in task_subreqs] == [["m1"], ["m2"]]


def test_run_incremental_state_and_reply_reopen():
    orig_fetch = I.ts.fetch_history
    orig_segment = I.segment_events
    try:
        batches = [
            [_e(100, "m1"), _e(200, "m2", role="bot"), _e(300, "m3")],
            [_e(1000, "m4", parent="m1")],
        ]

        def fake_fetch(group_id, start_s, end_s, hist_file=None):
            return batches.pop(0)

        seen_start_idx = []

        def fake_segment(group_id, events, model=None, start_idx=1, prior_anchors=None):
            seen_start_idx.append(start_idx)
            clusters = [[e] for e in events]
            metas = [{"title": e["msg_id"], "goal": e["msg_id"], "reason": "", "segs": [i + 1]}
                     for i, e in enumerate(events)]
            sections = [f"## 任务 {start_idx + i}　{e['msg_id']}\n" for i, e in enumerate(events)]
            deliveries = [None] * len(events)
            task_subreqs = [{"title": e["msg_id"], "subreqs": [
                {"id": "S1", "title": e["msg_id"], "status": "一遍过", "type": "",
                 "dominant": "", "member_msg_ids": [e["msg_id"]]}]} for e in events]
            return clusters, metas, "segment", "\n".join(sections), len(events), deliveries, task_subreqs

        I.ts.fetch_history = fake_fetch
        I.segment_events = fake_segment
        with tempfile.TemporaryDirectory() as d:
            state_dir = Path(d) / "state"
            out_dir = Path(d) / "out"
            r1 = I.run_incremental("g", state_dir=state_dir, out_dir=out_dir, thaw_hours=0)
            assert r1["mode"] == "full"
            state = I._load_state("g", state_dir)
            assert [t["member_msg_ids"] for t in state["frozen_tasks"]] == [["m1"], ["m2"]]
            assert [e["msg_id"] for e in state["active_tail_events"]] == ["m3"]
            # C1：frozen 每个 task 带结构化 subreqs→msg_id；active_tail 也持久化其 subreqs 映射
            assert [t["subreqs"][0]["member_msg_ids"] for t in state["frozen_tasks"]] == [["m1"], ["m2"]]
            assert state["active_tail_subreqs"][0]["member_msg_ids"] == ["m3"]
            r2 = I.run_incremental("g", state_dir=state_dir, out_dir=out_dir, thaw_hours=0)
            assert r2["mode"] == "incremental"
            # m4 replies to hard-frozen m1, so m1 is reopened into the active set with m4.
            assert r2["reopened"] == 1
            assert r2["active_events"] == 3
            assert seen_start_idx == [1, 2]
            state = I._load_state("g", state_dir)
            all_ids = [ids for t in state["frozen_tasks"] for ids in t["member_msg_ids"]]
            assert "m4" in [e["msg_id"] for e in state["active_tail_events"]]
            assert "m1" in all_ids
    finally:
        I.ts.fetch_history = orig_fetch
        I.segment_events = orig_segment


def test_carry_forward_titles_growing_task():
    """活动尾巴长大（新簇含旧任务全部消息 + 新增）→ 沿用旧标题，不漂移。"""
    prior = [{"msg_ids": ["m1", "m2", "m3"], "title": "轨迹按任务-子需求拆分"}]
    clusters = [[_e(1, "m1"), _e(2, "m2"), _e(3, "m3"), _e(4, "m4"), _e(5, "m5")]]
    metas = [{"title": "聊天轨迹拆分系统开发"}]   # LLM 这窗给了个新标题
    n = I._carry_forward_titles(clusters, metas, prior)
    assert n == 1
    assert metas[0]["title"] == "轨迹按任务-子需求拆分"   # 已沿用旧标题


def test_carry_forward_titles_new_task_keeps_fresh():
    """与任何旧任务都无重叠的全新任务 → 保留这窗 LLM 的新标题。"""
    prior = [{"msg_ids": ["m1", "m2"], "title": "旧任务"}]
    clusters = [[_e(9, "x9"), _e(10, "x10")]]
    metas = [{"title": "全新任务"}]
    n = I._carry_forward_titles(clusters, metas, prior)
    assert n == 0
    assert metas[0]["title"] == "全新任务"


def test_carry_forward_titles_below_threshold_keeps_fresh():
    """重叠不足阈值（旧任务只有少部分留在新簇）→ 不强行沿用。"""
    prior = [{"msg_ids": ["m1", "m2", "m3", "m4", "m5"], "title": "旧任务"}]
    clusters = [[_e(1, "m1"), _e(2, "m2"), _e(99, "z99")]]  # 旧任务只 2/5 留下 = 0.4 < 0.6
    metas = [{"title": "新标题"}]
    assert I._carry_forward_titles(clusters, metas, prior) == 0
    assert metas[0]["title"] == "新标题"


def test_carry_forward_titles_one_to_one():
    """两个旧任务、两个新簇 → 各自匹配各自的标题，不串台、不重复占用同一个锚。"""
    prior = [{"msg_ids": ["a1", "a2"], "title": "任务A"},
             {"msg_ids": ["b1", "b2"], "title": "任务B"}]
    clusters = [[_e(1, "b1"), _e(2, "b2"), _e(3, "b3")],
                [_e(4, "a1"), _e(5, "a2")]]
    metas = [{"title": "X"}, {"title": "Y"}]
    I._carry_forward_titles(clusters, metas, prior)
    assert metas[0]["title"] == "任务B"
    assert metas[1]["title"] == "任务A"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("✓", name)
