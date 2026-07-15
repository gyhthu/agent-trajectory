#!/usr/bin/env python3
"""llm_segment 纯函数回归（不调 LLM）：member_segs 鲁棒解析 + 互斥/覆盖审计 +
原子段→clusters 装配(去重/遗漏补孤) + bot 名兜底脱敏。
跑：python3 test_llm_segment.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm_segment as L
from regex_anonymizer import RegexAnonymizer

T0 = 1_750_000_000


def _atoms(n):
    """造 n 个原子段，每段一条 bot 消息，ts 递增。"""
    return [[{"ts": T0 + i * 3600, "role": "bot", "name": "claude(lian-server)",
              "msg_id": f"m{i}", "parent_id": "", "text": f"〔数据处理清洗〕seg{i} 内容"}]
            for i in range(1, n + 1)]


def test_seg_ints_robust():
    # LLM 可能写 int / "1" / "#1" / "#1,2"
    assert L._seg_ints([1, "2", "#3", "#4,5"]) == [1, 2, 3, 4, 5]
    assert L._seg_ints([]) == []
    assert L._seg_ints(None) == []
    print("✓ _seg_ints 鲁棒解析 int/'1'/'#1'/'#1,2'")


def test_audit_clean():
    atoms = _atoms(5)
    res = {"tasks": [{"id": "T1", "member_segs": [1, 2]},
                     {"id": "T2", "member_segs": ["#3", "#4", "#5"]}]}
    overlaps, missing = L.audit_membership(res, len(atoms))
    assert overlaps == [] and missing == [], f"{overlaps} {missing}"
    print("✓ audit 全覆盖互斥·无告警")


def test_audit_overlap_and_missing():
    atoms = _atoms(5)
    res = {"tasks": [{"id": "T1", "member_segs": [1, 2]},
                     {"id": "T2", "member_segs": [2, 3]}]}  # 2 重叠；4,5 漏
    overlaps, missing = L.audit_membership(res, len(atoms))
    assert overlaps == [2], overlaps
    assert missing == [4, 5], missing
    print("✓ audit 抓重叠[2]+遗漏[4,5]")


def test_assemble_dedup_and_orphan():
    """重叠按首次归属去重；遗漏段各补孤任务，绝不丢数据。"""
    atoms = _atoms(5)
    res = {"tasks": [{"id": "T1", "title": "甲", "member_segs": [1, 2]},
                     {"id": "T2", "title": "乙", "member_segs": ["#2", "#3"]}]}  # 2 重叠, 4/5 漏
    clusters, metas = L.assemble_clusters(atoms, res)
    # 段2 只归 T1（首次）；T2 只剩段3；段4、5 各成孤任务 → 共 4 个 cluster
    assert len(clusters) == 4, f"应 4 任务(甲+乙+2孤), got {len(clusters)}"
    allsegs = sorted(s for m in metas for s in m["segs"])
    assert allsegs == [1, 2, 3, 4, 5], f"全覆盖不丢, got {allsegs}"
    assert sum(m["segs"].count(2) for m in metas) == 1, "段2 不重复归属"
    orphans = [m for m in metas if "孤" in m["title"]]
    assert len(orphans) == 2, "段4、5 补孤任务"
    print("✓ assemble 去重+遗漏补孤·不丢数据")


def test_safe_name_masks_raw_id():
    anon = RegexAnonymizer()
    assert L._safe_name("cli_aaaa7814cd", anon) == "[未登记bot]"
    assert L._safe_name("ou_e5275236c2", anon) == "[未登记bot]"
    # 可读名原样保留
    assert L._safe_name("claude(lian-server)", anon) == "claude(lian-server)"
    print("✓ _safe_name 打码裸 app_id/open_id、保留可读名")


def test_segment_index_samples_tail_and_entities():
    """摘要不再只看前3条：尾部/长消息/实体线索要进入一行摘要。"""
    rows = [[
        {"ts": T0 + 1, "role": "user", "name": "张耀明", "text": "先看一下",
         "msg_id": "m1", "parent_id": ""},
        {"ts": T0 + 2, "role": "bot", "name": "lian-codex", "text": "〔数据处理清洗〕收到",
         "msg_id": "m2", "parent_id": ""},
        {"ts": T0 + 3, "role": "user", "name": "张耀明", "text": "这个先记着",
         "msg_id": "m3", "parent_id": ""},
        {"ts": T0 + 4, "role": "user", "name": "张耀明",
         "text": "真正重点在这里：请检查 trajectory_extract/incremental_segment.py 的增量切分逻辑",
         "msg_id": "m4", "parent_id": ""},
    ]]
    lines, _ = L.build_segment_index(rows, RegexAnonymizer())
    one = lines[0]
    assert "真正重点在这里" in one
    assert "incremental_segment.py" in one
    assert "线索:" in one
    print("✓ segment index 采到尾部重点 + 实体线索")


def test_render_decompose_reports_subreq_membership_audit():
    """合并版 decompose 文档也要暴露子需求重叠/遗漏，不能只在单任务 CLI 里报。"""
    import llm_decompose as D

    orig = D.decompose_one_task

    def fake_decompose_one_task(*args, **kwargs):
        return ({
            "task_goal": "核对增量结果",
            "subreqs": [
                {"id": "S1", "title": "主线", "dominant": "bot", "type": "主线",
                 "status": "一遍过", "member_idx": [1, 2]},
                {"id": "S2", "title": "重叠段", "dominant": "bot", "type": "旁支",
                 "status": "一遍过", "member_idx": [2]},
            ],
            "edges": [],
        }, 3)

    try:
        D.decompose_one_task = fake_decompose_one_task
        cluster = [[
            {"ts": T0 + 1, "role": "user", "name": "张耀明", "text": "看结果",
             "msg_id": "m1", "parent_id": ""},
            {"ts": T0 + 2, "role": "bot", "name": "lian-codex", "text": "在跑",
             "msg_id": "m2", "parent_id": ""},
            {"ts": T0 + 3, "role": "user", "name": "张耀明", "text": "对比呢",
             "msg_id": "m3", "parent_id": ""},
        ]]
        meta = [{"title": "核对结果", "goal": "核对增量结果", "segs": [1], "reason": ""}]
        md, total, _, task_subreqs = L.render_decompose("g", "w", cluster, meta, RegexAnonymizer())
    finally:
        D.decompose_one_task = orig

    assert total == 2
    assert "子需求归属审计" in md
    assert "重叠编号 [2]" in md
    assert "遗漏编号 [3]" in md
    # C1：render_decompose 同时产出结构化子需求→真 msg_id 映射（member_idx 经 _effective 翻成 msg_id）
    subs = task_subreqs[0]["subreqs"]
    valid = {"m1", "m2", "m3"}
    assert all(set(s["member_msg_ids"]) <= valid for s in subs)   # 只落真实 msg_id
    assert "m1" in subs[0]["member_msg_ids"]                       # S1(member_idx=[1,2]) 含首条
    assert subs[0]["status"] == "一遍过" and subs[1]["id"] == "S2"  # 字段随行带出
    print("✓ render_decompose 暴露子需求重叠/遗漏审计 + 结构化 msg_id 映射")


def test_subreq_member_msg_ids_maps_idx_to_real_msgid():
    """C1 纯映射：member_idx（1-based，基于 _effective 顺序）→ 真飞书 msg_id，越界/非数字丢弃。"""
    eff = [{"msg_id": "om_a"}, {"msg_id": "om_b"}, {"msg_id": "om_c"}]
    subs = [{"id": "S1", "title": "t1", "status": "有返工", "type": "主线",
             "dominant": "claude", "member_idx": [1, "#3"]},
            {"id": "S2", "title": "t2", "status": "被纠偏", "type": "收尾",
             "dominant": "廉莲", "member_idx": [2, 99, "x"]}]
    out = L.subreq_member_msg_ids(subs, eff)
    assert out[0]["member_msg_ids"] == ["om_a", "om_c"]      # "#3" 解析成 3
    assert out[1]["member_msg_ids"] == ["om_b"]              # 99 越界、"x" 无数字 → 丢弃
    assert out[0]["status"] == "有返工" and out[1]["dominant"] == "廉莲"
    print("✓ subreq_member_msg_ids 映射正确（含越界/脏值过滤）")


if __name__ == "__main__":
    test_seg_ints_robust()
    test_audit_clean()
    test_audit_overlap_and_missing()
    test_assemble_dedup_and_orphan()
    test_safe_name_masks_raw_id()
    test_render_decompose_reports_subreq_membership_audit()
    test_subreq_member_msg_ids_maps_idx_to_real_msgid()
    print("\n全部通过 ✓")
