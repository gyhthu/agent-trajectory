#!/usr/bin/env python3
"""导出某次对话的「bot 全量上下文」为可读 markdown（可推飞书文档）。

为什么需要它：bot 在群里犯错时，要复盘它**当时究竟看到了什么**——这只存在于
发往 API 的请求体里（system + tools + 全部 messages + 每个工具返回原文）。
anthropic_capture_proxy.py 已把这些逐次落盘到 api-calls/YYYY-MM-DD.jsonl.gz，
本脚本只做「挑一次调用 + 渲染成人能读的形式」，不重复造数据。

一次 API 调用 = 一段完整上下文快照。同一 session 的「最后一次调用」上下文最全
（含到当前为止所有轮 + 工具返回），所以默认取最后一次。

用法：
  # 今天所有 session 里最近一次调用（哪个 bot 刚出错就看它）
  python3 export_context.py

  # 指定 session（前缀即可，从飞书 transcript 文件名或 !sessions 拿）
  python3 export_context.py --session 7b6439c6

  # 看这个 session 的倒数第 2 次调用（--turn 负数从后数，正数从前数 0 起）
  python3 export_context.py --session 7b6439c6 --turn -2

  # 指定日期、写到文件、完整不截断
  python3 export_context.py --date 2026-06-16 --out /tmp/ctx.md --full

  # 直接推飞书文档（需 FEISHU_APP_ID/SECRET/CHAT_ID 环境变量）
  python3 export_context.py --session 7b6439c6 --feishu
"""
import argparse
import glob
import gzip
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import message_tree as mt  # noqa: E402  逐 message 挂树分组（与采集端共用的单一事实源）

DATA_DIR = Path(os.environ.get("TRAJ_DATA_DIR", "/home/agent/trajectory-data"))
HERE = Path(__file__).resolve().parent
# llm_label_tool 共享数据浏览器(/tools?tab=data-view)读这个根，逐文件 JSON/JSONL 预览。
# 把导出落这里 = 用现成 viewer 看，不用每次塞飞书文档。
SHARED_ROOT = Path(os.environ.get("SHARED_DATA_ROOT", "/opt/shared/data"))
SHARED_SUBDIR = "bot-context"
DATAVIEW_URL = os.environ.get("DATAVIEW_URL", "http://127.0.0.1:5175/tools?tab=data-view")


def iter_api_calls(date):
    """按时间序产出 api-calls 记录。date=None 则扫所有日期文件。"""
    pattern = str(DATA_DIR / "api-calls" / (f"{date}.jsonl.gz" if date else "*.jsonl.gz"))
    for path in sorted(glob.glob(pattern)):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


def pick_calls(session_prefix, date):
    """收集目标 session 的全部调用（按落盘序）。session_prefix=None 时返回全部。"""
    out = []
    for rec in iter_api_calls(date):
        sid = rec.get("session_id") or ""
        if session_prefix and not sid.startswith(session_prefix):
            continue
        out.append(rec)
    return out


def pick_calls_multi(prefixes, date):
    """收集多个 session（一段对话被压缩/重启切成的多个 shard）的全部调用，按 ts 升序合并。
    !ctx 用它：单值 lane 指针会漂到旧 shard，给齐全部 shard 后按时间取真·最新一次调用。"""
    pset = [p for p in (prefixes or []) if p]
    out = []
    for rec in iter_api_calls(date):
        sid = rec.get("session_id") or ""
        if any(sid.startswith(p) for p in pset):
            out.append(rec)
    out.sort(key=lambda r: r.get("ts") or 0)
    return out


def _newest_user_text(rec):
    """该次调用 messages 里最新一条『真实用户消息』的文本（跳过纯 tool_result 回填）。"""
    for m in reversed((rec.get("request") or {}).get("messages") or []):
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, str):
            return c
        for b in (c or []):
            if isinstance(b, dict) and b.get("type") == "text":
                return b.get("text", "")
    return ""


def pick_by_match(calls, text):
    """选『最新用户消息包含该文本』的最后一次调用——用于按看板某轮的提问原文定位该轮。
    某轮的提问只在它自己那几次调用里是『最新用户消息』，下一轮一来就被顶下去，
    故命中区间正好是该轮，取最后一次=该轮收尾调用(其 response 即该轮答复)。
    归一化去空白以穿透回复框架换行。返回索引或 None。"""
    norm = lambda s: "".join((s or "").split())
    needle = norm(text)
    if not needle:
        return None
    hit = [i for i, r in enumerate(calls) if needle in norm(_newest_user_text(r))]
    return hit[-1] if hit else None


def _clip(s, n):
    s = s if isinstance(s, str) else json.dumps(s, ensure_ascii=False)
    if n and len(s) > n:
        return s[:n] + f"\n…〔截断，共 {len(s)} 字，--full 看全文〕"
    return s


def render_blocks(content, maxlen):
    """把一条 message 的 content（字符串或 block 列表）渲染成 markdown 片段。"""
    if content is None:
        return "_(空)_"
    if isinstance(content, str):
        return _clip(content, maxlen)
    parts = []
    for b in content:
        if not isinstance(b, dict):
            parts.append(_clip(str(b), maxlen))
            continue
        t = b.get("type")
        if t == "text":
            parts.append(_clip(b.get("text", ""), maxlen))
        elif t == "thinking":
            parts.append("> 💭 **思考**\n>\n> " + _clip(b.get("thinking", ""), maxlen).replace("\n", "\n> "))
        elif t == "tool_use":
            inp = json.dumps(b.get("input", {}), ensure_ascii=False, indent=2)
            parts.append(f"🔧 **调用工具 `{b.get('name')}`** (id=`{b.get('id','')[:12]}`)\n```json\n{_clip(inp, maxlen)}\n```")
        elif t == "tool_result":
            c = b.get("content")
            if isinstance(c, list):  # tool_result 的 content 也可能是 block 列表
                c = "\n".join(x.get("text", json.dumps(x, ensure_ascii=False))
                              if isinstance(x, dict) else str(x) for x in c)
            err = " ⚠️出错" if b.get("is_error") else ""
            parts.append(f"📥 **工具返回**{err} (for=`{b.get('tool_use_id','')[:12]}`)\n```\n{_clip(c, maxlen)}\n```")
        elif t == "image":
            parts.append("🖼️ _[图片]_")
        else:
            parts.append(f"_[{t}]_ " + _clip(json.dumps(b, ensure_ascii=False), maxlen))
    return "\n\n".join(parts)


def render_system(system, maxlen, full_system):
    if system is None:
        return "_(无 system)_"
    if isinstance(system, str):
        text = system
    else:  # list of blocks
        text = "\n\n".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in system)
    return _clip(text, None if full_system else maxlen)


def build_markdown(rec, all_calls_count, idx, maxlen, full_system):
    req = rec.get("request") or {}
    msgs = req.get("messages") or []
    tools = req.get("tools") or []
    sid = rec.get("session_id") or "?"
    ts = rec.get("ts")
    when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else "?"
    L = []
    L.append(f"# Bot 上下文快照 · session `{sid[:8]}`")
    L.append("")
    L.append(f"- **时间**：{when}")
    L.append(f"- **模型**：{req.get('model')}　**HTTP**：{rec.get('status')}")
    L.append(f"- **本快照**：该 session 第 {idx+1}/{all_calls_count} 次 API 调用"
             f"（= 截至此刻 bot 看到的完整上下文）")
    L.append(f"- **消息数**：{len(msgs)}　**工具数**：{len(tools)}")
    usage = (rec.get("response") or {}).get("usage") if isinstance(rec.get("response"), dict) else None
    if usage:
        L.append(f"- **token 用量**：{json.dumps(usage, ensure_ascii=False)}")
    L.append("")
    L.append("---")
    L.append("## 🧱 System（系统提示）")
    L.append("")
    L.append(render_system(req.get("system"), maxlen, full_system))
    L.append("")
    L.append("## 🧰 可用工具")
    L.append("")
    L.append("、".join(f"`{t.get('name')}`" for t in tools) or "_(无)_")
    L.append("")
    L.append("---")
    L.append("## 💬 对话消息（按序，含工具调用与返回）")
    for i, m in enumerate(msgs):
        role = m.get("role", "?")
        icon = {"user": "👤 user", "assistant": "🤖 assistant", "system": "⚙️ system"}.get(role, role)
        L.append("")
        L.append(f"### [{i}] {icon}")
        L.append("")
        L.append(render_blocks(m.get("content"), maxlen))
    # 把模型这次的回复也附上（response），方便看「上下文→输出」对照
    resp = rec.get("response")
    if isinstance(resp, dict) and resp.get("content"):
        L.append("")
        L.append("---")
        L.append("## 📤 本次模型输出（response）")
        L.append("")
        L.append(render_blocks(resp.get("content"), maxlen))
    return "\n".join(L)


def _clean_content(content):
    """复盘视图减噪：thinking 块去掉 signature（Anthropic 的加密签名串，不可读、
    无信息量）；纯空 thinking 块（无正文只剩签名）整块丢弃。其余块原样保真。"""
    if not isinstance(content, list):
        return content
    out = []
    for b in content:
        if not isinstance(b, dict):
            out.append(b)
            continue
        if b.get("type") == "thinking":
            if not (b.get("thinking") or "").strip():
                continue  # 空思考块：丢
            b = {k: v for k, v in b.items() if k != "signature"}
        out.append(b)
    return out


def build_shared_jsonl(rec, all_calls_count, idx):
    """导成 JSONL（每行一项），供 llm_label_tool data-view 逐行预览。
    保真：消息块原样保留(含 tool_result 正文)，只补一行 meta + 一行 system；
    仅对 thinking 块去签名/丢空块减噪（见 _clean_content）。"""
    req = rec.get("request") or {}
    msgs = req.get("messages") or []
    tools = req.get("tools") or []
    sid = rec.get("session_id") or "?"
    ts = rec.get("ts")
    lines = []
    meta = {
        "_kind": "meta",
        "session_id": sid,
        "model": req.get("model"),
        "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else None,
        "turn": f"{idx + 1}/{all_calls_count}",
        "http_status": rec.get("status"),
        "n_messages": len(msgs),
        "n_tools": len(tools),
        "tools": [t.get("name") for t in tools],
    }
    resp = rec.get("response")
    if isinstance(resp, dict) and resp.get("usage"):
        meta["usage"] = resp["usage"]
    lines.append(meta)
    sysv = req.get("system")
    systext = sysv if isinstance(sysv, str) else "\n\n".join(
        b.get("text", "") if isinstance(b, dict) else str(b) for b in (sysv or []))
    lines.append({"_kind": "system", "text": systext})
    for i, m in enumerate(msgs):
        lines.append({"_kind": "message", "i": i, "role": m.get("role"),
                      "content": _clean_content(m.get("content"))})
    if isinstance(resp, dict) and resp.get("content"):
        lines.append({"_kind": "response", "content": _clean_content(resp.get("content"))})
    return "\n".join(json.dumps(x, ensure_ascii=False) for x in lines)


# ── 多线程轨迹重建（主 agent + 子 agent）──────────────────────────────────
# 同一 session 的全部 API 调用里混着主 agent / 各子 agent / 框架辅助（标题·压缩）。分组靠
# message_tree 逐 message 挂树、按 thread_id 切（治本，替代旧的 (system[:200],first_user[:200])
# 前缀哈希——后者会把同 prompt 并行子 agent 揉成一桶丢数据）；定性靠结构（父边=子 / 无工具
# 一次性调用=aux / 有工具=main），不嗅 system 措辞。子 agent 的完整内部轨迹只存在于它自己
# 那次调用里（主 agent 那次只看得到一个 tool_use+tool_result），故须从整个 session 的调用集
# 合重建才完整。重建出的 threads/turns 结构对齐 llm_label_tool 的 ContextThread/TraceTurn，
# data-view 直接复用 eval 的轨迹查看组件渲染。保真不截断（与 --shared 一贯的保真原则一致）。

def _system_text(system):
    if isinstance(system, list):
        return "\n".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in system)
    return system or ""


def _clean_system(system):
    """剥掉动态 billing header 行（每次 cc_version/cch 不同，会污染线程指纹），留真正人设。"""
    txt = _system_text(system)
    return "\n".join(l for l in txt.split("\n")
                     if not l.startswith("x-anthropic-billing-header")).strip()


def _first_user_text(messages):
    """首条用户消息文本——线程指纹的一半，也是子 agent 的任务指令。"""
    for m in messages:
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, str):
            return c
        for b in (c or []):
            if isinstance(b, dict) and b.get("type") == "text":
                return b.get("text", "")
        return ""
    return ""


def _tool_result_text(content):
    if isinstance(content, list):
        return "\n".join(
            x.get("text", json.dumps(x, ensure_ascii=False)) if isinstance(x, dict) else str(x)
            for x in content)
    return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)


# harness 每轮注入进 user 消息的上下文块（CLAUDE.md/记忆/环境/skill 清单等），
# 角色是 user 但并非真人所打——剥出来单独标 origin=system，别混进『真人』。
_SYSTEM_REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>\s*", re.DOTALL)


def _split_system_reminders(text):
    """把 user 文本里 harness 注入的 <system-reminder> 段从真人正文中剥离，保序返回。
    返回 [(kind, segment), ...]；kind: 'system'=harness 注入上下文 / 'body'=其余正文。"""
    segs = []
    last = 0
    for m in _SYSTEM_REMINDER_RE.finditer(text):
        if m.start() > last:
            segs.append(("body", text[last:m.start()]))
        segs.append(("system", m.group(0)))
        last = m.end()
    if last < len(text):
        segs.append(("body", text[last:]))
    return segs or [("body", text)]


def _turn_origin(turn_type, role, thread_kind):
    """每个 turn 的「来源」分层标记，供前端区分『真人输入』与『agent 调度指令』：
      human         = 真人在群里说的话（只可能出现在主 agent 线程的 user-text）
      agent_dispatch= 父 agent 派发给子 agent 的任务指令（子线程的 user-text——
                      子 agent 视角里它是『user 输入』，但实为上游 agent 生成，非真人）
      framework_aux = 框架辅助调用的 user-text（标题生成/上下文压缩等 aux 线程——既非真人
                      也非任务派发，是 harness 喂给一次性 LLM 调用的处理对象）
      system        = harness 注入的 <system-reminder> 上下文（CLAUDE.md/记忆/环境等，
                      角色挂 user 但非真人；在 _split_system_reminders 处显式标，不走这里）
      model         = 模型自己产出（assistant 文本 / 思考 / 工具调用）
    tool_result 是工具 I/O，已有独立样式，不打 origin。
    关键区分：真人 turn 只在主线程出现；任何非主线程的 user-text 都是 agent 调度，不是人。"""
    if turn_type in ("thinking", "tool_call"):
        return "model"
    if turn_type == "text":
        if role == "user":
            if thread_kind == "main":
                return "human"          # 顶层 CLI 主循环的 user-text = 真人在群里说的话
            if thread_kind == "subagent":
                return "agent_dispatch"  # 父 agent 派发给子 agent 的任务指令
            return "framework_aux"       # aux（标题生成/压缩等框架调用）：既非真人也非任务派发，单独成类
        return "model"
    return None


def _anthropic_turns(messages, thread_kind="main"):
    """Anthropic messages（content blocks）→ TraceTurn[]（对齐 eval schema，保真不截断）。
    thread_kind（main/subagent/aux）决定 user-text 的 origin：主线程=真人，其余=agent 调度。"""
    turns = []
    i = 0

    def _push(turn):
        if "origin" not in turn:  # 调用方已显式标（如 system-reminder）则尊重之
            o = _turn_origin(turn["type"], turn.get("role"), thread_kind)
            if o:
                turn["origin"] = o
        turns.append(turn)

    def _emit_text(role, text):
        """user 文本先剥 harness 注入的 system-reminder（标 system），剩下才算真人/调度。"""
        nonlocal i
        if role != "user":
            if text.strip():
                _push({"type": "text", "id": f"t{i}", "role": role, "text": text})
                i += 1
            return
        for kind, seg in _split_system_reminders(text):
            if not seg.strip():
                continue
            turn = {"type": "text", "id": f"t{i}", "role": role, "text": seg}
            if kind == "system":
                turn["origin"] = "system"
            _push(turn)
            i += 1

    for m in messages:
        role = m.get("role")
        content = _clean_content(m.get("content"))
        if isinstance(content, str):
            _emit_text(role, content)
            continue
        for b in (content or []):
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            if t == "text":
                _emit_text(role, b.get("text", "") or "")
            elif t == "thinking":
                _push({"type": "thinking", "id": f"t{i}", "role": "assistant",
                       "thinking": b.get("thinking", "")})
                i += 1
            elif t == "tool_use":
                _push({"type": "tool_call", "id": f"t{i}", "role": "assistant",
                       "tool_name": b.get("name", "unknown"),
                       "tool_input": json.dumps(b.get("input", {}),
                                                ensure_ascii=False, indent=2),
                       "input_truncated": False, "call_id": b.get("id", ""),
                       "tool_status": None, "tool_title": None})
                i += 1
            elif t == "tool_result":
                _push({"type": "tool_result", "id": f"t{i}",
                       "call_id": b.get("tool_use_id", ""),
                       "output": _tool_result_text(b.get("content")),
                       "is_error": bool(b.get("is_error")), "truncated": False})
                i += 1
            elif t == "image":
                _push({"type": "text", "id": f"t{i}", "role": role, "text": "🖼️ [图片]"})
                i += 1
    return turns


def _response_turns(resp, thread_kind, idx):
    """模型这次调用的输出（rec["response"].content）→ assistant turns，接在输入 turns 之后。
    response 是 assistant 的 content blocks（无外层 role），包成一条 assistant message 复用
    _anthropic_turns 解析（text/thinking/tool_use 都覆盖）。id 加 r{idx}_ 前缀，免与输入 turns 撞键。"""
    if not isinstance(resp, dict):
        return []
    content = resp.get("content")
    if not content:
        return []
    out = _anthropic_turns([{"role": "assistant", "content": content}], thread_kind)
    for t in out:
        t["id"] = f"r{idx}_{t['id']}"
    return out


def _first_user_alltext(messages):
    """首条 user 消息的全部 text 块拼接（含可能夹带的 system-reminder，未剥）。"""
    for m in messages:
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, str):
            return c
        return "\n".join(b.get("text", "") for b in (c or [])
                         if isinstance(b, dict) and b.get("type") == "text")
    return ""


def _first_user_body(messages):
    """首条 user 剥掉 harness 注入的 <system-reminder> 后的真人/派发正文——
    结构化 join key（不依赖 origin/kind）：子 agent 的它 == 父线程那次 Agent/Task 调用 prompt 入参。
    """
    for kind, seg in _split_system_reminders(_first_user_alltext(messages)):
        if kind == "body" and seg.strip():
            return seg
    return ""


def _aux_display_label(sys_clean):
    """aux 的展示名仅用 system 字符串顺手起（标题/压缩），叫错不影响结构——字符串只配 label。"""
    low = sys_clean.lower()
    if ("generate a concise" in low and "title" in low) or "sentence-case title" in low:
        return "框架辅助调用 · 标题生成"
    if "<session>" in low or "summary of the conversation" in low or "compact" in low:
        return "框架辅助调用 · 上下文压缩"
    snippet = " ".join(sys_clean.split())[:24]
    return f"框架辅助调用 · {snippet}" if snippet else "框架辅助调用"


def _dispatch_points(threads):
    """扫所有线程里的 Agent/Task 工具调用 = 每一次「派生子 agent」的派发点。
    返回 [(thread_id, call_id, turn_id, norm_prompt), ...]——norm 去空白以穿透换行差异。"""
    norm = lambda s: "".join((s or "").split())
    pts = []
    for th in threads:
        for t in th["turns"]:
            if t.get("type") == "tool_call" and t.get("tool_name") in ("Agent", "Task"):
                try:
                    prompt = (json.loads(t.get("tool_input") or "{}") or {}).get("prompt", "")
                except (json.JSONDecodeError, TypeError):
                    prompt = ""
                if prompt:
                    pts.append({"thread_id": th["thread_id"], "call_id": t.get("call_id", ""),
                                "turn_id": t.get("id", ""), "norm": norm(prompt), "used": False})
    return pts


def _classify_and_wire(threads):
    """按结构（不读 system 措辞）定 main/sub/aux 并连显式父子边：
      ① subagent = 它的首条 user 正文（_first_user_body，剥 system-reminder 后）命中某线程
         一次未认领的 Agent/Task 派发点 → 连父边（parent_thread_id/…call_id/…turn_id）。
         这是骨架判定的核心：靠「派发指令逐字匹配」而非「system 写没写 you are a claude agent」，
         Anthropic 改措辞也不烂。多个相同 prompt 的并行子 agent 按出现序一一对应、各连各边。
      ② aux  = 其余无父边、且**全程无工具 I/O**的线程——标题生成、上下文压缩这类一次性框架
         调用天然 text-in/text-out、不碰工具。压缩 vs 标题不分，都归 aux。
      ③ main = 其余无父边、**有工具 I/O**的线程（根编排者/主对话；压缩重建后会有多个 epoch，
         都算 main）。若一个有工具的都没有（纯聊天会话），退化为 msg_count 最大者当 main。
    join key 与 kind 判定都不依赖 origin，避开「定性靠 origin、origin 又靠 kind」的循环。
    用「有无工具」而非「system 措辞」分 main/aux：标题/压缩永不带工具，主 epoch 总在干活带工具，
    Anthropic 改 system 措辞也不烂。"""
    norm = lambda s: "".join((s or "").split())
    pts = _dispatch_points(threads)
    for th in threads:
        need = norm(th.get("_join_key", ""))
        if not need:
            continue
        cand = (next((p for p in pts if not p["used"] and p["norm"] == need), None)
                or next((p for p in pts if not p["used"]
                         and (need in p["norm"] or p["norm"] in need)), None))
        if cand:
            cand["used"] = True
            th["kind"] = "subagent"
            th["parent_thread_id"] = cand["thread_id"]
            th["parent_dispatch_call_id"] = cand["call_id"]
            th["parent_dispatch_turn_id"] = cand["turn_id"]
    rest = [th for th in threads if th.get("kind") is None]
    has_tools = lambda t: any(x.get("type") in ("tool_call", "tool_result") for x in t["turns"])
    tooled = [th for th in rest if has_tools(th)]
    if tooled:
        for th in rest:
            th["kind"] = "main" if has_tools(th) else "aux"
    elif rest:  # 纯聊天会话：无任何工具，最长者当 main，其余 aux
        main = max(rest, key=lambda t: t["msg_count"])
        for th in rest:
            th["kind"] = "main" if th is main else "aux"


def _fix_user_origins(th):
    """kind 定了之后，据 kind 把 user-text 的 origin 落实（system-reminder 段保持 origin=system）。
    主=human / 子=agent_dispatch / aux=framework_aux。"""
    o = {"main": "human", "subagent": "agent_dispatch"}.get(th["kind"], "framework_aux")
    for t in th["turns"]:
        if t.get("type") == "text" and t.get("role") == "user" and t.get("origin") != "system":
            t["origin"] = o


def build_threads_payload(calls):
    """从一组 API 调用重建多线程轨迹。

    calls=整 session 的全部调用 → 主 agent + 各子 agent；calls=[单次调用] → 单线程。
    返回结构对齐 eval ContextThread/TraceTurn，data-view 轨迹视图直接渲染。
    """
    # ① 逐 message 挂树分组（治本：替代 (system[:200],first_user[:200]) 前缀哈希）。
    #    并行同 prompt 子 agent 因首条 assistant 不同 → 落不同 thread_id，不再互吞。
    tree = mt.MessageTree()
    metas = []  # (rec, system_raw, msgs)
    for rec in calls:
        req = rec.get("request") or {}
        msgs = req.get("messages") or []
        if not msgs:
            continue
        metas.append((rec, req.get("system"), msgs))
    anchored = [mt.mount_call(tree, sysm, msgs) for (_, sysm, msgs) in metas]
    # ② 分组键 = thread_id；首帧 [S,U]（无 assistant）provisional → 并回其唯一子线程。
    groups = {}
    for (rec, sysm, msgs), (tid, prov) in zip(metas, anchored):
        if prov:
            u = mt.unique_descendant_thread(tree, sysm, msgs)
            if u:
                tid = u
        groups.setdefault(tid, []).append(rec)

    # ③ 每组建 thread（rep=message 最多的那次=最全）；turns 先解析结构，origin 待 kind 定后落实。
    threads = []
    for tid, recs in groups.items():
        rep = max(recs, key=lambda r: len((r.get("request") or {}).get("messages") or []))
        req = rep.get("request") or {}
        msgs = req.get("messages") or []
        sys_clean = _clean_system(req.get("system"))
        fu = _first_user_text(msgs)
        turns = _anthropic_turns(msgs, "main")  # 先按 main 解析（结构 kind 无关），origin 之后修正
        tool_summary = {}
        for t in turns:
            if t["type"] == "tool_call":
                n = t.get("tool_name", "unknown")
                tool_summary[n] = tool_summary.get(n, 0) + 1
        threads.append({
            "thread_id": tid, "kind": None, "label": None,
            "system_head": sys_clean[:400], "first_user": fu[:400],
            "tool_summary": tool_summary, "msg_count": len(msgs), "turns": turns,
            "_rep": rep, "_sys_clean": sys_clean, "_join_key": _first_user_body(msgs),
        })
    # ④ 结构化定 kind + 连父边；据 kind 落实 user-text origin；补「system 指令 + 模型输出」两头。
    _classify_and_wire(threads)
    main_system = None
    for idx, th in enumerate(threads):
        _fix_user_origins(th)
        if th["kind"] == "subagent":
            disp = " ".join((th["_join_key"] or "").split())[:30]
            th["label"] = f"子 agent · {disp}" if disp else "子 agent"
        elif th["kind"] == "main":
            th["label"] = "主 agent"
            if main_system is None:
                main_system = th["_sys_clean"]
        else:
            th["label"] = _aux_display_label(th["_sys_clean"])
        sys_clean = th.pop("_sys_clean")
        rep = th.pop("_rep")
        th.pop("_join_key", None)
        sys_turn = ({"type": "text", "id": "sys", "role": "system", "origin": "system",
                     "text": sys_clean} if sys_clean.strip() else None)
        th["turns"] = ([sys_turn] if sys_turn else []) + th["turns"] + _response_turns(
            rep.get("response"), th["kind"], idx)
    threads.sort(key=lambda t: 0 if t["kind"] == "main" else 1)
    if main_system is None and threads:
        main_system = max((t["system_head"] for t in threads), key=len, default="")
    rec0 = calls[0] if calls else {}
    return {
        "schema": "bot-trajectory-v1",
        "session_id": rec0.get("session_id"),
        "model": (rec0.get("request") or {}).get("model"),
        "system_prompt": main_system or "",
        "threads": threads,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", help="session_id 前缀；缺省=今天最近一次调用（任意 session）")
    ap.add_argument("--sessions",
                    help="逗号分隔的多个 session_id 前缀（一段对话被压缩/重启切成的多个 shard）。"
                         "给齐后按时间在所有 shard 里挑真·最新一次调用，绕开单值 lane 指针漂移。"
                         "与 --session 互斥；--match 时在并集里按 ts 取最后一次命中。")
    ap.add_argument("--date", help="YYYY-MM-DD，缺省=扫所有日期文件")
    ap.add_argument("--turn", type=int, default=-1,
                    help="选该 session 第几次调用：负数从后数(默认 -1=最后)，正数从前数(0 起)")
    ap.add_argument("--match",
                    help="按某轮提问原文定位：选『最新用户消息包含该文本』的最后一次调用"
                         "（看板 #N.k 那轮的提问文本即可，能跳过附和轮、与看板编号对齐）")
    ap.add_argument("--maxlen", type=int, default=4000, help="每块正文截断字数（默认 4000）")
    ap.add_argument("--full", action="store_true", help="不截断任何正文")
    ap.add_argument("--full-system", action="store_true", help="system 提示不截断")
    ap.add_argument("--out", help="输出 markdown 文件路径（缺省打到 stdout）")
    ap.add_argument("--feishu", action="store_true", help="推飞书文档（需 FEISHU_* 环境变量）")
    ap.add_argument("--shared", action="store_true",
                    help=f"导成 jsonl 落 {SHARED_ROOT}/{SHARED_SUBDIR}/，给文件名+data-view 链接（自己浏览）")
    ap.add_argument("--title", help="飞书文档标题")
    args = ap.parse_args()

    maxlen = 0 if args.full else args.maxlen

    if args.sessions:
        prefixes = [p.strip() for p in args.sessions.split(",") if p.strip()]
        calls = pick_calls_multi(prefixes, args.date)
        if not calls:
            sys.exit(f"找不到 session 前缀 {prefixes!r}（date={args.date or '全部'}）的调用")
        if args.match:
            idx = pick_by_match(calls, args.match)
            if idx is None:
                sys.exit(f"匹配不到该轮（提问文本 {args.match!r} 不在 {args.date or '全部'} 的调用里）")
            rec = calls[idx]
        else:
            # 多 shard 已按 ts 升序合并 → --turn 索引即「全对话时间序」，-1=真·最新一次调用。
            try:
                idx = args.turn if args.turn >= 0 else len(calls) + args.turn
                rec = calls[idx]
            except IndexError:
                sys.exit(f"--turn {args.turn} 越界：合并后共 {len(calls)} 次调用")
        count = len(calls)
        session_calls = calls
    elif args.session:
        calls = pick_calls(args.session, args.date)
        if not calls:
            sys.exit(f"找不到 session 前缀 {args.session!r}（date={args.date or '全部'}）的调用")
        if args.match:
            idx = pick_by_match(calls, args.match)
            if idx is None:
                sys.exit(f"匹配不到该轮（提问文本 {args.match!r} 不在 {args.date or '全部'} 的调用里）")
            rec = calls[idx]
        else:
            try:
                idx = args.turn if args.turn >= 0 else len(calls) + args.turn
                rec = calls[idx]
            except IndexError:
                sys.exit(f"--turn {args.turn} 越界：该 session 共 {len(calls)} 次调用")
        count = len(calls)
        session_calls = calls
    else:
        # 缺省：今天（或指定日期）最近一次调用，不分 session
        date = args.date or time.strftime("%Y-%m-%d")
        calls = pick_calls(None, date)
        if not calls:
            sys.exit(f"{date} 无任何 api-call 记录")
        rec = calls[-1]
        sid = rec.get("session_id") or ""
        same = [c for c in calls if (c.get("session_id") or "") == sid]
        count, idx = len(same), len(same) - 1
        session_calls = same

    # --shared：导多线程轨迹 JSON 落共享数据区，给 data-view 链接（轨迹视图渲染）。
    # 不带 --match=整 session 全部调用 → 主 agent + 各子 agent 多线程；带 --match=只导
    # 定位到的那一次调用 → 单线程（看某一轮单次上下文，子 agent 仅以工具 I/O 出现）。
    if args.shared:
        sid = (rec.get("session_id") or "x")[:8]
        src_calls = [rec] if args.match else session_calls
        payload = build_threads_payload(src_calls)
        outdir = SHARED_ROOT / SHARED_SUBDIR
        outdir.mkdir(parents=True, exist_ok=True)
        fname = f"ctx_{sid}_{time.strftime('%m%d_%H%M%S')}.json"
        fpath = outdir / fname
        fpath.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        n_threads = len(payload["threads"])
        n_main = sum(1 for t in payload["threads"] if t["kind"] == "main")
        n_sub = sum(1 for t in payload["threads"] if t["kind"] == "subagent")
        n_aux = sum(1 for t in payload["threads"] if t["kind"] == "aux")
        main_seg = f"{n_main} 个主 agent" + ("（含压缩重建的多个 epoch）" if n_main > 1 else "")
        aux_seg = f" + {n_aux} 个框架辅助调用" if n_aux else ""
        print(f"文件：{fpath}")
        print(f"（{n_threads} 个 agent 线程：{main_seg} + {n_sub} 个子 agent{aux_seg}）")
        print(f"查看：{DATAVIEW_URL}  （左侧树进 {SHARED_SUBDIR}/ 点开 {fname}）")
        return

    md = build_markdown(rec, count, idx, maxlen, args.full or args.full_system)

    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"已写出 {args.out}（{len(md)} 字）", file=sys.stderr)
    elif not args.feishu:
        print(md)

    if args.feishu:
        sid = (rec.get("session_id") or "x")[:8]
        title = args.title or f"Bot上下文快照_{sid}_{time.strftime('%m%d_%H%M')}"
        tmp = args.out or f"/tmp/ctx_{sid}_{int(time.time())}.md"
        if not args.out:
            Path(tmp).write_text(md, encoding="utf-8")
        doc_script = HERE.parent.parent / "feishu-doc" / "scripts" / "md_to_feishu_doc.py"
        r = subprocess.run([sys.executable, str(doc_script), tmp,
                            "--title", title,
                            "--summary", f"session {sid} 的完整上下文快照（system+消息+工具返回）"],
                           capture_output=True, text=True)
        sys.stderr.write(r.stdout + r.stderr)
        if r.returncode != 0:
            sys.exit(f"推飞书文档失败（见上）")


if __name__ == "__main__":
    main()
