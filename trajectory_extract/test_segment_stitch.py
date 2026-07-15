#!/usr/bin/env python3
"""segment_history 短档兜底缝合 + B 终结判定 回归测试。
跑：python3 test_segment_stitch.py

历史：A 跨断档缝合(40min gap 不再硬边界)经真实4天数据实测为负优化(6簇→2簇,过并更重)，
张耀明 2026-06-27 拍板撤掉，任务粒度边界改走 LLM 语义切分(llm_segment.py)。本文件因此
只保留：① segment_history 的短档(≤20min)词重叠兜底缝合不被破坏；② B compute_terminal。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from task_stitch import segment_history, compute_terminal, atomic_segments

HOUR = 3600
DAY = 86400
T0 = 1_750_000_000  # 固定基准 ts，避免依赖 wall-clock


def mk(ts, role, text, mid=None, parent=None):
    return {"ts": ts, "role": role, "who": "u" if role == "user" else "b",
            "name": "张耀明" if role == "user" else "claude(lian-server)",
            "kind": None if role == "user" else "claude",
            "msg_id": mid or f"m{ts}", "parent_id": parent or "",
            "task_id": "", "thread_id": "", "text": text}


# ── 短档兜底缝合（防误切；不触碰跨断档，那交给 llm_segment 治本） ──

def test_short_gap_overlap_merge():
    """短档(≤20min)、同话题、词重叠≥4 → 兜底缝合成 1 簇（防误切）。"""
    a = [mk(T0, "user", "trajectory split outcome 改一下"),
         mk(T0 + 60, "bot", "〔数据处理清洗〕trajectory split outcome 改好")]
    b = [mk(T0 + 18 * 60, "user", "trajectory split outcome 再调"),
         mk(T0 + 18 * 60 + 60, "bot", "〔数据处理清洗〕trajectory split outcome 调好")]
    cls = segment_history(a + b)
    assert len(cls) == 1, f"短档强重叠应缝合成1簇, got {len(cls)}"
    print("✓ 短档·强词重叠·兜底缝合")


def test_long_gap_not_stitched():
    """跨断档(>20min，跨夜)即便同话题强重叠也**不**在本层缝合——治本交给 llm_segment。"""
    a = [mk(T0, "user", "搞 trajectory split outcome terminal 四字段"),
         mk(T0 + 60, "bot", "〔数据处理清洗〕trajectory split outcome terminal 做好")]
    b = [mk(T0 + 12 * HOUR, "user", "trajectory split outcome terminal 再补一版"),
         mk(T0 + 12 * HOUR + 60, "bot", "〔数据处理清洗〕trajectory split outcome terminal 补好")]
    cls = segment_history(a + b)
    assert len(cls) == 2, f"跨断档本层不缝合, 应保持2簇, got {len(cls)}"
    print("✓ 跨断档·本层不缝合（留给 llm_segment）")


def test_different_topic_split():
    """不同话题、长 gap → 两边各自缝成簇、互不并（异话题不合并）。"""
    a = [mk(T0, "user", "改 trajectory split pipeline config"),
         mk(T0 + 60, "bot", "〔数据处理清洗〕trajectory split pipeline config 改好")]
    b = [mk(T0 + 12 * HOUR, "user", "litellm restart service redeploy"),
         mk(T0 + 12 * HOUR + 60, "bot", "〔服务器运维〕litellm restart service redeploy 好")]
    cls = segment_history(a + b)
    assert len(cls) == 2, f"异话题应切2簇, got {len(cls)}"
    # 两个话题不在同一簇里
    topics = [{e["text"][:8] for e in cl} for cl in cls]
    print("✓ 异话题·切分")


def test_parent_reply_stitches_short_gap_even_with_topic_change():
    """普通跨段回复仍是强信号：短距异话题 parent_id 要缝回，防正常回复被误切。"""
    a = [mk(T0, "bot", "〔数据处理清洗〕先给方案", "a0"),
         mk(T0 + 10, "user", "处理轨迹切分", "a1"),
         mk(T0 + 60, "bot", "〔数据处理清洗〕方案细节", "a2")]
    b = [mk(T0 + HOUR, "bot", "〔服务器运维〕服务重启", "b0"),
         mk(T0 + HOUR + 60, "user", "顺手重启服务", "b1"),
         mk(T0 + HOUR + 120, "user", "回复前面那条继续看", "b3", parent="a2")]
    cls = atomic_segments(a + b)
    assert len(cls) == 1, f"短距 parent_id 回复应强缝合, got {len(cls)}"
    print("✓ parent_id 短距异话题仍强缝合")


def test_parent_reply_long_gap_different_topic_downgraded():
    """跨很久且两边明确异话题的 parent_id 可疑，降级给 LLM，不在原子层硬焊死。"""
    a = [mk(T0, "bot", "〔数据处理清洗〕先给方案", "a0"),
         mk(T0 + 10, "user", "处理轨迹切分", "a1"),
         mk(T0 + 60, "bot", "〔数据处理清洗〕方案细节", "a2")]
    b = [mk(T0 + 2 * DAY, "bot", "〔服务器运维〕服务重启", "b0"),
         mk(T0 + 2 * DAY + 60, "user", "重启 litellm 服务", "b1"),
         mk(T0 + 2 * DAY + 120, "user", "这句误回复到了老消息", "b3", parent="a2")]
    cls = atomic_segments(a + b)
    assert len(cls) == 2, f"长距异话题 parent_id 应降级不硬缝合, got {len(cls)}"
    print("✓ parent_id 长距异话题降级为弱信号")


def test_content_axis_high_sim_stitches():
    """换轴(路线A)：异话题回复，但内容相似度高(真续接)→ 强缝合成 1 段。
    用确定性 fake sim_fn，不依赖网络。"""
    a = [mk(T0, "bot", "〔数据处理清洗〕先给方案", "a0"),
         mk(T0 + 10, "user", "处理轨迹切分", "a1"),
         mk(T0 + 60, "bot", "〔数据处理清洗〕方案细节", "a2")]
    b = [mk(T0 + 2 * DAY, "bot", "〔服务器运维〕换个话题", "b0"),
         mk(T0 + 2 * DAY + 60, "user", "接着上面轨迹切分继续", "b3", parent="a2")]
    cls = atomic_segments(a + b, sim_fn=lambda ce, ps: 0.80)   # 高相似=真续接
    assert len(cls) == 1, f"内容高相似应强缝合成1段, got {len(cls)}"
    print("✓ 内容轴·异话题高相似→强缝合（不赌时间）")


def test_content_axis_low_sim_downgrades_even_same_day():
    """换轴根治同天挂错(旧时间轴漏网)：异话题回复、内容相似度低(挂错)→ 降级不焊死，
    即便发生在**同一天**(gap 远 < 24h，旧逻辑会强缝)。这正是 #2 要补的洞。"""
    a = [mk(T0, "bot", "〔数据处理清洗〕轨迹切分方案", "a0"),
         mk(T0 + 10, "user", "处理轨迹切分", "a1"),
         mk(T0 + 60, "bot", "〔数据处理清洗〕方案细节", "a2")]
    b = [mk(T0 + 1800, "bot", "〔服务器运维〕litellm 重启", "b0"),
         mk(T0 + 1860, "user", "重启完了吗", "b1"),
         mk(T0 + 1920, "user", "(点错了)挂到老消息上", "b3", parent="a2")]  # 同天(<24h)挂错
    # 旧时间轴：gap<24h → 强缝合(焊死)；新内容轴：低相似 → 降级
    old = atomic_segments(a + b)                          # sim_fn=None → 回退旧时间轴
    new = atomic_segments(a + b, sim_fn=lambda ce, ps: 0.20)   # 低相似=挂错
    assert len(old) == 1, f"旧时间轴同天挂错会焊死(对照), got {len(old)}"
    assert len(new) == 2, f"新内容轴同天挂错应降级不焊死, got {len(new)}"
    print("✓ 内容轴·同天挂错→降级（旧时间轴会焊死，这是根治的洞）")


def test_isolated_blip_does_not_pollute_segment_topic():
    """孤立话题 blip 被吸收后不得污染段话题（张耀明 2026-07-02 案例）：
    [清洗,清洗,运维,清洗,清洗] 中间那条'运维'是孤立 blip（前后都是清洗），
    整串应为 **1 段**——旧代码在 else 分支写 cur['topic']=eff[i] 会让 blip 把段话题
    改成'运维'，害它后一条'清洗'被误判成新持续切换而多切一刀（切成 2 段）。"""
    topics = ["数据处理清洗", "数据处理清洗", "服务器运维", "数据处理清洗", "数据处理清洗"]
    evs = [mk(T0 + i * 60, "bot", f"〔{tp}〕正文{i}") for i, tp in enumerate(topics)]
    cls = atomic_segments(evs)
    assert len(cls) == 1, f"孤立 blip 应被吸收成 1 段, got {len(cls)}"
    print("✓ 孤立 blip 不污染段话题（1 段）")


def test_real_topic_switch_still_splits():
    """对照：连续两条'运维'=真转场，仍要切开（证明修复没把该切的也吞了）。"""
    topics = ["数据处理清洗", "数据处理清洗", "服务器运维", "服务器运维"]
    evs = [mk(T0 + i * 60, "bot", f"〔{tp}〕正文{i}") for i, tp in enumerate(topics)]
    cls = atomic_segments(evs)
    assert len(cls) == 2, f"真持续转场应切 2 段, got {len(cls)}"
    print("✓ 真转场仍切开（修复无过并回归）")


def test_empty_msg_id_and_parent_no_crash():
    """脏数据回归：有空 msg_id、且有消息 parent_id 为空时，空 parent 绝不当回复边，
    更不能 KeyError 崩掉整群切分（f4e1696 引入的回归，本测试守住）。"""
    a = [mk(T0, "user", "第一条正常消息内容", ""),          # 空 msg_id 脏数据
         mk(T0 + 50000, "user", "隔很久的另一条无关消息", "m2")]  # parent_id 默认空
    cls = atomic_segments(a)        # 不抛异常即通过
    assert len(cls) == 2, f"空 parent 不该把无关段合并, got {len(cls)}"
    print("✓ 空 msg_id/空 parent 不崩、不误合并")


# ── B：终结判定 ──

def test_terminal_last_incomplete():
    """三个任务，最后一个末尾是 bot 回合(用户未回) → 仅末簇 incomplete。"""
    t1 = [mk(T0, "user", "任务甲 alpha"), mk(T0 + 60, "bot", "〔训练评测〕alpha 干完")]
    t2 = [mk(T0 + 2 * HOUR, "user", "任务乙 beta"),
          mk(T0 + 2 * HOUR + 60, "bot", "〔记忆系统〕beta 干完")]
    t3 = [mk(T0 + 4 * HOUR, "user", "任务丙 gamma 还在弄"),
          mk(T0 + 4 * HOUR + 60, "bot", "〔bot基建〕gamma 这就去改")]
    cls = segment_history(t1 + t2 + t3)
    term = compute_terminal(cls)
    last = max(range(len(cls)), key=lambda i: cls[i][0]["ts"])
    assert term[last] is False, "末簇(bot结尾,用户未回)应判未终结"
    assert all(term[i] for i in range(len(cls)) if i != last), "非末簇应判已终结"
    print("✓ B 末簇(无收尾)=incomplete，非末簇=terminal")


def test_terminal_last_resolved():
    """最后一个任务末尾用户明确收尾(搞定/谢谢) → 末簇也算已终结。"""
    t1 = [mk(T0, "user", "任务甲 alpha"), mk(T0 + 60, "bot", "〔训练评测〕alpha 干完")]
    t2 = [mk(T0 + 2 * HOUR, "user", "任务乙 beta 弄一下"),
          mk(T0 + 2 * HOUR + 60, "bot", "〔记忆系统〕beta 弄好了"),
          mk(T0 + 2 * HOUR + 120, "user", "搞定了谢谢")]
    cls = segment_history(t1 + t2)
    term = compute_terminal(cls)
    last = max(range(len(cls)), key=lambda i: cls[i][0]["ts"])
    assert term[last] is True, "末簇有用户明确收尾语应判已终结"
    print("✓ B 末簇有收尾语=terminal")


def test_terminal_empty():
    assert compute_terminal([]) == []
    print("✓ B 空输入")


def test_terminal_interleaved_picks_last_touched():
    """交织长任务（张耀明 2026-06-29 实测修）：A 早起步但一直做到最近(尾10H)，
    B 晚起步却早停(尾3H)。『还在飞的未终结』应是最近被碰的 A，而非『最先开始』口径下的 B。
    旧口径(按簇首事件 ts 取末簇)会把 B 当未终结、A 误判已终结——正是真实数据翻车的形态。"""
    a = [mk(T0, "user", "任务A 长期做"),
         mk(T0 + 10 * HOUR, "bot", "〔数据处理清洗〕A 还在改")]   # A 尾事件最晚 = 最近被碰
    b = [mk(T0 + 2 * HOUR, "user", "任务B 插一脚"),
         mk(T0 + 3 * HOUR, "bot", "〔训练评测〕B 弄完了")]         # B 晚起步但早停
    term = compute_terminal([a, b])   # 传入顺序 [A, B]
    assert term[0] is False, "A 最近被碰(尾10H)→应判未终结(还在飞)"
    assert term[1] is True, "B 早停(尾3H)→应判已终结"
    print("✓ B 交织：未终结落在最近被碰的 A，不是最先开始口径下的 B")


# ── llm_decompose member_idx 鲁棒解析（模型偶发输出 "#1" 不许崩，2026-06-27 实跑暴露） ──

def test_decompose_member_idx_hash_prefix():
    """audit_membership / enforce_rollback_purity 碰到 member_idx=['#1','#2'] 不许 ValueError。"""
    from llm_decompose import _as_seg_int, audit_membership, enforce_rollback_purity
    assert _as_seg_int("#1") == 1 and _as_seg_int(2) == 2
    assert _as_seg_int("#3,4") == 3 and _as_seg_int("noise") is None
    res = {"subreqs": [{"id": "S1", "status": "被纠偏", "type": "旁支",
                        "member_idx": ["#1", "#2"]}], "edges": []}
    meta = [{"role": "user"}, {"role": "assistant", "name": "bot"}]
    assert audit_membership(res, 2) == ([], [])      # 不崩、归属正确
    enforce_rollback_purity(res, meta)                # 不崩
    print("✓ decompose member_idx '#1' 解析不崩")


if __name__ == "__main__":
    test_short_gap_overlap_merge()
    test_long_gap_not_stitched()
    test_different_topic_split()
    test_parent_reply_stitches_short_gap_even_with_topic_change()
    test_parent_reply_long_gap_different_topic_downgraded()
    test_content_axis_high_sim_stitches()
    test_content_axis_low_sim_downgrades_even_same_day()
    test_terminal_last_incomplete()
    test_terminal_last_resolved()
    test_terminal_empty()
    test_terminal_interleaved_picks_last_touched()
    test_decompose_member_idx_hash_prefix()
    print("\n全部通过 ✓")
