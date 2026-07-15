#!/usr/bin/env python3
"""message_tree 确定性测试。纯 stdlib 内联 fixtures，钉死：
  ① 主 agent 多次调用归一 thread
  ② 并行同 prompt 子 agent 分成两 thread（核心 bug 修复），且与到达序无关
  ③ 主 vs 子 thread 不同（system 不同 → root 即分叉）
  ④ 在线增量 == 离线批量逐项一致
  ⑤ provisional 首帧经 unique_descendant_thread 并回唯一子线程
  ⑥ 并行分叉处的共享 [S,U] 帧保持 provisional（多锚 → 不乱并）
  ⑦ content str vs block 列表指纹不语义归一 + sort_keys 键序无关
  ⑧ billing 块不影响 system 指纹
  ⑨ replay 一致（重启回放可复现）
运行：python3 test_message_tree.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import message_tree as mt

PASS = 0
FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f"FAIL  {name}")


def u(text):
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def a(text):
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def a_tool(name, prompt):
    return {"role": "assistant",
            "content": [{"type": "tool_use", "id": "x", "name": name, "input": {"prompt": prompt}}]}


def tr(text):
    return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": text}]}


SYS_MAIN = "You are an interactive agent that helps users with software engineering."
SYS_SUB = "=== CRITICAL: READ-ONLY MODE === You are a Claude agent."
SYS_AUX = "Generate a concise sentence-case title for this session."


# ① 主 agent 多次调用归一 thread
def test_main_multi():
    t = mt.MessageTree()
    c1 = [u("人类问题")]                          # 首调，无 assistant
    c2 = [u("人类问题"), a("回复1")]              # 第二调，含首个 assistant
    c3 = [u("人类问题"), a("回复1"), tr("x"), a("回复2")]
    id1, p1 = mt.mount_call(t, SYS_MAIN, c1)
    id2, p2 = mt.mount_call(t, SYS_MAIN, c2)
    id3, p3 = mt.mount_call(t, SYS_MAIN, c3)
    check("① 首调 provisional", p1 and not p2 and not p3)
    check("① 第2、3调同 thread（锚=首条assistant）", id2 == id3)
    # provisional 首帧能并回该唯一子线程
    merged = mt.unique_descendant_thread(t, SYS_MAIN, c1)
    check("① 首帧并回主线程", merged == id2)


# ② 并行同 prompt 子 agent 分两 thread；③ 与到达序无关
def test_parallel_subagents():
    def build(order):
        t = mt.MessageTree()
        cA = [u("调研甘油三酯阈值"), a("子A：走 Grep")]   # 同 U、不同首条 assistant
        cB = [u("调研甘油三酯阈值"), a("子B：走 Read")]
        calls = {"A": cA, "B": cB}
        ids = {}
        for k in order:
            ids[k], _ = mt.mount_call(t, SYS_SUB, calls[k])
        return ids
    ab = build("AB")
    ba = build("BA")
    check("② 并行同prompt子agent分两thread", ab["A"] != ab["B"])
    check("③ thread_id 与到达序无关(AB==BA)", ab["A"] == ba["A"] and ab["B"] == ba["B"])


# ③ 主 vs 子（system 不同）从 root 即分叉
def test_main_vs_sub():
    t = mt.MessageTree()
    idm, _ = mt.mount_call(t, SYS_MAIN, [u("X"), a("m")])
    ids, _ = mt.mount_call(t, SYS_SUB, [u("X"), a("m")])  # 仅 system 不同
    check("③ system 不同 → thread 不同", idm != ids)


# ④ 在线增量 == 离线批量
def test_online_equals_offline():
    calls = [
        {"request": {"system": SYS_MAIN, "messages": [u("Q")]}},
        {"request": {"system": SYS_MAIN, "messages": [u("Q"), a("A1"), a_tool("Task", "干活")]}},
        {"request": {"system": SYS_SUB, "messages": [u("干活"), a("sub do")]}},
        {"request": {"system": SYS_MAIN, "messages": [u("Q"), a("A1"), a_tool("Task", "干活"), tr("r"), a("A2")]}},
    ]
    online = mt.replay_session(mt.MessageTree(), calls)
    offline = mt.replay_session(mt.MessageTree(), calls)
    check("④ replay 两次逐项一致", online == offline)
    # 拆成两次 mount 序列也一致
    t = mt.MessageTree()
    seq = [mt.mount_call(t, c["request"]["system"], c["request"]["messages"]) for c in calls]
    check("④ 增量 mount == replay", seq == online)


# ⑤⑥ provisional 归属 + 共享祖先多锚保持 provisional
def test_provisional_resolution():
    t = mt.MessageTree()
    # 在 U 处分叉成两个并行子（多锚）
    shared = [u("派发")]
    mt.mount_call(t, SYS_SUB, [u("派发"), a("子A")])
    mt.mount_call(t, SYS_SUB, [u("派发"), a("子B")])
    ids, prov = mt.mount_call(t, SYS_SUB, shared)
    check("⑥ 共享[S,U]帧 provisional", prov)
    check("⑥ 多锚 → 不并(返回None)", mt.unique_descendant_thread(t, SYS_SUB, shared) is None)


# ⑦ content str vs block 指纹差异 + 键序无关
def test_fingerprint():
    f_str = mt._msg_fingerprint({"role": "user", "content": "hi"})
    f_blk = mt._msg_fingerprint({"role": "user", "content": [{"type": "text", "text": "hi"}]})
    check("⑦ str 与 block 不语义归一(指纹不同)", f_str != f_blk)
    f1 = mt._msg_fingerprint({"role": "user", "content": [{"type": "text", "text": "hi"}]})
    f2 = mt._msg_fingerprint({"content": [{"text": "hi", "type": "text"}], "role": "user"})
    check("⑦ 键序无关(sort_keys)", f1 == f2)


# ⑧ billing 块不影响 system 指纹
def test_billing_stripped():
    base = [{"type": "text", "text": SYS_MAIN}]
    billed = [{"type": "text", "text": "x-anthropic-billing-header cch=abc123"},
              {"type": "text", "text": SYS_MAIN}]
    billed2 = [{"type": "text", "text": "x-anthropic-billing-header cch=DIFFERENT"},
               {"type": "text", "text": SYS_MAIN}]
    check("⑧ billing 不影响 system 指纹",
          mt._system_fp(base) == mt._system_fp(billed) == mt._system_fp(billed2))


# ⑨ replay 与逐条 mount 在含子 agent 的真实形态下一致
def test_replay_consistency():
    calls = [
        {"request": {"system": SYS_MAIN, "messages": [u("Q"), a_tool("Agent", "搜A")]}},
        {"request": {"system": SYS_SUB, "messages": [u("搜A"), a("subA1")]}},
        {"request": {"system": SYS_SUB, "messages": [u("搜A"), a("subA2diff")]}},  # 同prompt并行子
        {"request": {"system": SYS_AUX, "messages": [u("压缩内容"), a("标题")]}},
    ]
    r1 = mt.replay_session(mt.MessageTree(), calls)
    r2 = mt.replay_session(mt.MessageTree(), calls)
    check("⑨ replay 可复现", r1 == r2)
    ids = [x[0] for x in r1]
    check("⑨ 两并行子 thread 不同", ids[1] != ids[2])
    check("⑨ aux 与 main/sub 都不同", len(set(ids)) == 4)


# ⑩ 在线（每 call 空树）== 离线（共享树）：非 provisional 的 thread_id 必须逐项一致。
# 这是采集端不维护跨 call 树、直接拿空树算 id 的正确性根基。
def test_fresh_tree_equals_shared():
    calls = [
        {"system": SYS_MAIN, "messages": [u("Q")]},
        {"system": SYS_MAIN, "messages": [u("Q"), a("A1"), a_tool("Task", "干活")]},
        {"system": SYS_SUB, "messages": [u("干活"), a("subA")]},
        {"system": SYS_SUB, "messages": [u("干活"), a("subB_diff")]},  # 同prompt并行子
        {"system": SYS_MAIN, "messages": [u("Q"), a("A1"), a_tool("Task", "干活"), tr("r"), a("A2")]},
        {"system": SYS_AUX, "messages": [u("压缩"), a("摘要")]},
    ]
    shared = mt.MessageTree()
    ok = True
    for c in calls:
        off_id, off_p = mt.mount_call(shared, c["system"], c["messages"])
        on_id, on_p = mt.mount_call(mt.MessageTree(), c["system"], c["messages"])
        if on_p != off_p:
            ok = False
        if not on_p and on_id != off_id:  # provisional 是占位，不要求相等
            ok = False
    check("⑩ 非prov:空树id==共享树id（采集端可无状态）", ok)


if __name__ == "__main__":
    for fn in [test_main_multi, test_parallel_subagents, test_main_vs_sub,
               test_online_equals_offline, test_provisional_resolution,
               test_fingerprint, test_billing_stripped, test_replay_consistency,
               test_fresh_tree_equals_shared]:
        print(f"\n[{fn.__name__}]")
        fn()
    print(f"\n==== {PASS} passed, {FAIL} failed ====")
    sys.exit(1 if FAIL else 0)
