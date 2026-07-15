#!/usr/bin/env python3
"""轨迹采集共用工具：捕获代理与聚合器都用，保证指纹/角色判定单一事实源。

为什么独立成文件：`system_tools_sha`（去重指纹）和 `classify_role`（主线/子代理
分类）这两件事，代理落盘时和聚合器读盘时都要做，逻辑必须完全一致，否则去重和
分类对不上。放一处，两边 import。
"""
import hashlib
import json

# CC SDK 往 system 里注入的计费头块，text 以此开头。它带的 cch= 每次调用都变，
# 必须在做指纹/分类前剔掉，否则同一个 agent 的每次调用都被当成不同身份。
_BILLING_PREFIX = "x-anthropic-billing-header"


def strip_billing_blocks(system):
    """剔掉 system 里的 x-anthropic-billing-header 块（cch= 逐 call 变，是噪声）。"""
    if isinstance(system, list):
        return [b for b in system
                if not (isinstance(b, dict)
                        and isinstance(b.get("text"), str)
                        and b["text"].startswith(_BILLING_PREFIX))]
    return system


def system_text(system) -> str:
    """把 system（str 或 block 列表）拼成纯文本，供分类正则匹配。已剔 billing。"""
    s = strip_billing_blocks(system)
    if isinstance(s, list):
        return "\n".join(b.get("text", "") for b in s if isinstance(b, dict))
    return s or ""


def system_tools_sha(payload: dict) -> str:
    """sha(剔 billing 后的 system + tools)：同 session 内按此聚类区分主线/子代理/压缩变体。"""
    return hashlib.sha256(
        json.dumps({"system": strip_billing_blocks(payload.get("system")),
                    "tools": payload.get("tools")},
                   ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


# 角色签名：按 system 正文特征句判定。顺序敏感——先匹配专门化子代理/辅助，
# 再落到主线，最后兜底。判定句来自实测各身份的 system 头部。
def classify_role(system) -> tuple:
    """返回 (role, desc)。role ∈ {main, subagent, aux}。

    - main      主线 Claude Code（interactive agent…software engineering），含压缩后变体
    - subagent  主线派生的子代理：文件搜索(Explore)/web 搜索/Task
    - aux       辅助调用：标题生成/裸 SDK helper/未知
    """
    t = system_text(system)
    tl = t.lower()
    # —— 辅助类（先于主线，因其也可能含 "claude code" 字样）——
    if "generate a concise" in tl and "title" in tl:
        return ("aux", "title")
    # —— 子代理类 ——
    if "file search specialist" in tl:
        return ("subagent", "file-search")
    if "performing a web search" in tl or "web search tool use" in tl:
        return ("subagent", "web-search")
    # —— 主线：CC 交互式 agent ——
    if "interactive agent that helps users with software engineering" in tl:
        return ("main", "main")
    # —— 兜底 ——
    if "you are claude code" in tl:
        # 有 CC 前缀但不是主线特征句 → 多半是专门化子代理（Task 等）
        return ("subagent", "task")
    if "claude agent sdk" in tl:
        return ("aux", "sdk")
    return ("aux", "unknown")
