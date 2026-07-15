#!/usr/bin/env python3
"""逐 message 挂树 + 确定性 thread_id（采集端在线盖章 / export 离线分组的单一事实源）。

为什么不用 (system[:200], first_user[:200]) 前缀哈希分桶：那会把**派发 prompt 相同的
并行子 agent 揉成一桶、丢掉其中一条轨迹**，压缩点前后也被切成断链。

实测已确认：子 agent / aux 调用**共享父会话的 session_id**，HTTP 层无任何权威字段能
区分子 agent。所以 thread_id 只能**从消息结构推导**——学 slime 的逐 message 挂树：
一个 session 内所有 call 的 messages 互为前缀共享，每个线程只在尾部追加；把 system 指纹
当 root 下虚拟第 0 层并入路径，逐 message 下钻挂树，**thread_id 锚在「本次 call 路径里
第一条 assistant 节点」**。这一个锚点选择同时满足三条硬约束：

  - 主 agent 多次调用：第一条 assistant 永远是史上第一条回复（前缀不变）→ 同一节点 → 同 id；
  - 并行同 prompt 子 agent：callA=[S,U,A1…] callB=[S,U,B1…]，仅首条 assistant A1≠B1
    → 落 U 下两个不同 assistant 节点 → 不同 id（修好丢数据的核心 bug）；
  - 在线==离线：锚由 call 自身 messages 唯一确定，与别的分支何时到达无关 → 两端一致。

首次 call（messages 只有 [S, U]、还没 assistant）算不出真锚 → 末节点当临时锚、provisional
=True。每个线程的首帧都会这样；export 离线分组时用 unique_descendant_thread 把它并回唯一
子线程（见该函数）。

注：response 不能掺进锚路径——实测 response 的 tool_use 块比下次请求回显的多一个 caller 键，
两边指纹对不上，掺进来反而把首帧锚和后续锚拆开。锚只用 request.messages，保真不归一。
"""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from traj_common import strip_billing_blocks  # noqa: E402  剔 billing 块单一事实源

_UNIT = "\x1f"  # thread_id 路径分隔符（不可能出现在正文里）


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _system_fp(system) -> str:
    """system 指纹：先剔每 call 变的 billing 块，再对剩余结构哈希。前缀 sys: 标层。"""
    clean = strip_billing_blocks(system)
    return "sys:" + _sha(json.dumps(clean, sort_keys=True, ensure_ascii=False))


def _msg_fingerprint(message: dict) -> str:
    """一条 message 的指纹：对 (role, content) 原样哈希。

    content 可能是 str 或 block 列表，sort_keys 消键序噪声；**用原始 message，不剥 thinking
    signature、不做语义归一**——保真，保证采集端与 export 两端字节一致。
    """
    return _sha(json.dumps({"role": message.get("role"), "content": message.get("content")},
                           sort_keys=True, ensure_ascii=False))


def _branch_id(path_keys) -> str:
    """thread_id = t + sha(root→锚 各节点 "role:fp" 串)[:12]，纯内容函数、两端可复现。"""
    return "t" + _sha(_UNIT.join(path_keys))[:12]


class _Node:
    __slots__ = ("fp", "role", "children", "is_anchor", "branch_id")

    def __init__(self, fp, role):
        self.fp = fp
        self.role = role
        self.children = {}      # fp -> _Node，按指纹命中、与插入序无关
        self.is_anchor = False  # 是否某线程的锚（第一条 assistant）
        self.branch_id = None   # 锚节点缓存的 thread_id


class MessageTree:
    """一棵 per-session 的消息树。root 下第一层是 system 指纹节点。"""

    def __init__(self):
        self.root = _Node("__root__", "root")


def _call_seq(system, messages):
    """把一次 call 摊成 [(role, fp), ...]：system 当虚拟第 0 层 + 各 message。"""
    seq = [("system", _system_fp(system))]
    for m in (messages or []):
        seq.append((m.get("role") or "?", _msg_fingerprint(m)))
    return seq


def mount_call(tree: MessageTree, system, messages):
    """把这次 call 逐 message 挂到 tree，返回 (thread_id, provisional)。

    锚 = 路径里第一条 assistant 节点；无 assistant（首调 [S,U]）→ 末节点当临时锚，provisional=True。
    """
    node = tree.root
    keys = []
    anchor = None
    anchor_keys = None
    last = node
    for role, fp in _call_seq(system, messages):
        child = node.children.get(fp)
        if child is None:
            child = _Node(fp, role)
            node.children[fp] = child
        node = child
        last = node
        keys.append(role + ":" + fp)
        if anchor is None and role == "assistant":
            anchor = node
            anchor_keys = list(keys)
    provisional = False
    if anchor is None:               # 整次 call 没有 assistant → 临时锚
        anchor = last
        anchor_keys = list(keys)
        provisional = True
    else:
        anchor.is_anchor = True      # 真锚才标记，供 provisional 归属解析统计
    if anchor.branch_id is None:
        anchor.branch_id = _branch_id(anchor_keys)
    return anchor.branch_id, provisional


def _walk_to_anchor(tree: MessageTree, system, messages):
    """只读重走这次 call 的路径，返回其锚节点（树须已 mount 过）；走不到返回 None。"""
    node = tree.root
    last = node
    for role, fp in _call_seq(system, messages):
        nxt = node.children.get(fp)
        if nxt is None:
            return None
        node = nxt
        last = node
        if role == "assistant":
            return node
    return last


def unique_descendant_thread(tree: MessageTree, system, messages):
    """provisional call 的归属解析：若其锚节点子树里只有一个线程锚，返回该 thread_id，否则 None。

    主线程首帧 [S, U_human]：U_human 子树里只有它唯一的回复 A1 是锚 → 并回主线程。
    并行子 agent 在 U 处分叉（多个锚）→ 该 [S,U] 帧是真·共享祖先，返回 None 保持 provisional。
    """
    node = _walk_to_anchor(tree, system, messages)
    if node is None:
        return None
    ids = set()
    stack = [node]
    while stack:
        n = stack.pop()
        if n.is_anchor and n.branch_id:
            ids.add(n.branch_id)
        stack.extend(n.children.values())
    return next(iter(ids)) if len(ids) == 1 else None


def replay_session(tree: MessageTree, calls):
    """按序对每条 record 调 mount_call（只重建结构、不落盘）。

    供采集端重启 lazy 回放 + export 离线重算共用，保证在线增量==离线批量。
    calls=[{"request": {"system":..., "messages":[...]}}...]，按落盘/ts 序喂入。
    返回 [(thread_id, provisional), ...] 与 calls 等长。
    """
    out = []
    for rec in calls:
        req = rec.get("request") or {}
        out.append(mount_call(tree, req.get("system"), req.get("messages") or []))
    return out
