#!/usr/bin/env python3
"""按任务把「多模型(claude/codex/antigravity) + 用户」的交互轨迹串成一条有序时间线。

为什么能串、串接键是什么（核过真实数据，非臆测）：
  · 一个任务里多个 bot 参与时，任何**单个 bot 的会话 id 都不能当主键**——claude 用
    session_id、codex 用 thread_id、antigravity 用 transcript 会话，各自独立、互不知情。
  · 唯一横跨所有模型的全局坐标 = **飞书消息层**。而 group-memory(:8765) 的统一消息流
    **已经**把每条消息打上了 {msg_id, task_id, role, bot_id/user_id, timestamp, thread_id}
    ——它本身就是现成的「联邦索引」：按 task_id 聚类即得任务边界，按 bot_id 认出是哪个模型。
  · 于是串接 = ①拉 group-memory 统一流 → ②按 task_id 聚类 → ③任务内按 ts 排成
    人/多bot 交错的时间线 → ④对每个 bot 回合，按 (模型, ts 邻近 + 回复正文匹配) 挂上
    它**自己轨迹库**里那次调用的展开（工具调用/返回、会话 id）。

各模型轨迹库与挂接方式（保真度见 trajectory-fidelity 总账文档）：
  · claude       : ~/trajectory-data/api-calls/*.jsonl.gz   （response.content 文本匹配）
  · lian-codex   : ~/trajectory-data/codex-api-calls/*.jsonl.gz（response.text 文本匹配）
  · antigravity  : ~/.gemini/antigravity/brain/<conv>/.system_generated/logs/transcript.jsonl
                   （transcript 不带飞书 msg_id → 只能按时间窗对齐，全链路最弱一跳）

用法：
  python3 task_stitch.py --group oc_xxx --task '#3'         # 串某任务，markdown 落共享区
  python3 task_stitch.py --group oc_xxx --list             # 先列出该群各任务的参与模型
  python3 task_stitch.py --group oc_xxx --task '#3' --n 120 # 拉更深的历史窗
"""
import argparse
import copy
import glob
import gzip
import json
import os
import time
import urllib.request
from pathlib import Path

GM_URL = os.environ.get("GROUP_MEMORY_URL", "http://127.0.0.1:8765").rstrip("/")
TRAJ = Path(os.environ.get("TRAJ_DATA_DIR", "/home/agent/trajectory-data"))
ANTIGRAV_BRAIN = Path(os.path.expanduser(
    os.environ.get("ANTIGRAV_BRAIN", "~/.gemini/antigravity/brain")))
SHARED = Path(os.environ.get("SHARED_DATA_ROOT", "/opt/shared/data")) / "task-trajectory"
DATAVIEW = os.environ.get("DATAVIEW_URL", "http://127.0.0.1:5175/tools?tab=data-view")

# sender open_id（lian-server app 视角）→ 模型注册表。open_id 是 per-app 的，这里登记的是
# 录制方(lian-server)看到的 bot_id；新增 bot 在这里补一行即可（identity/lookup 可拿名字）。
REGISTRY = {
    "ou_e5275236c27f0604bfdf4a6ca8e10afc": {"name": "claude(lian-server)", "kind": "claude"},
    "ou_cf5045c31e7c7e87562bfe3af592a752": {"name": "lian-codex",          "kind": "codex"},
    "ou_63dba8e072b03356815144f00f0023ee": {"name": "zym-antigravity",     "kind": "antigravity"},
}

# 飞书历史 API(im/v1/messages)里 bot 的 sender 是 app_id（非 group-memory 的 open_id），
# 用户是 open_id。v2「飞书历史当脊柱」用这两张表认人。新增 bot/人在此补一行。
APP_REGISTRY = {
    "cli_aa9678e4f038dcce": {"name": "claude(lian-server)", "kind": "claude"},
    "cli_aabbdfe7b9389cb3": {"name": "lian-codex",          "kind": "codex"},
    "cli_aab361a436229bb3": {"name": "zym-antigravity",     "kind": "antigravity"},
    "cli_aaaa7814cdb8dceb": {"name": "lian-antigravity",    "kind": "antigravity"},
}
USER_REGISTRY = {
    "ou_adff5621f41381371eec5ca9bb45a9ea": "张耀明",
    "ou_7afe4ac1339edb11382e11f60385f3ee": "廉莲",
}


# ── group-memory 统一流 ──

def fetch_stream(group_id, n):
    url = f"{GM_URL}/message/recent?group_id={group_id}&n={n}"
    with urllib.request.urlopen(url, timeout=15) as r:
        data = json.load(r)
    evs = []
    for rec in data.get("results", []):
        m = rec.get("metadata") or {}
        who = m.get("bot_id") or m.get("user_id") or ""
        evs.append({
            "ts": m.get("timestamp") or 0,
            "role": m.get("role") or "?",
            "who": who,
            "name": REGISTRY.get(who, {}).get("name") or _short_user(m),
            "kind": REGISTRY.get(who, {}).get("kind"),
            "msg_id": m.get("msg_id") or "",
            "task_id": m.get("task_id") or "",
            "thread_id": m.get("thread_id") or "",
            "text": rec.get("text") or "",
        })
    evs.sort(key=lambda e: e["ts"])
    return evs


def _short_user(m):
    r = m.get("role")
    return "用户" if r == "user" else (m.get("user_id") or m.get("bot_id") or "?")[:14]


# ── 轨迹库挂接 ──

_CALL_CACHE = {}


def _load_calls(subdir, date):
    """缓存式加载某轨迹库某天的全部 call（按 ts 升序）。"""
    key = (subdir, date)
    if key in _CALL_CACHE:
        return _CALL_CACHE[key]
    out = []
    for p in sorted(glob.glob(str(TRAJ / subdir / f"{date}.jsonl.gz"))):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    out.sort(key=lambda r: r.get("ts") or 0)
    _CALL_CACHE[key] = out
    return out


def _norm(s):
    return "".join((s or "").split())


def _strip_feishu(text):
    """去掉飞书层加的前缀(〔标签〕/@xxx/<at>)，只留模型真正生成的正文起头，供匹配。"""
    import re
    t = re.sub(r"<at\b[^>]*>.*?</at>", "", text)
    t = re.sub(r"^\s*〔[^〕]*〕\s*", "", t)
    t = re.sub(r"@[^\s，。:：]+\s*", "", t)
    return t.strip()


def _claude_resp_text(rec):
    resp = rec.get("response") or {}
    parts = []
    for b in (resp.get("content") or []):
        if isinstance(b, dict) and b.get("type") == "text":
            parts.append(b.get("text", ""))
    return "\n".join(parts)


def _claude_tools(rec):
    resp = rec.get("response") or {}
    return [b.get("name") for b in (resp.get("content") or [])
            if isinstance(b, dict) and b.get("type") == "tool_use"]


def _codex_resp_text(rec):
    resp = rec.get("response") or {}
    if resp.get("text"):
        return resp["text"]
    fr = resp.get("final_response") or {}
    parts = []
    for it in (fr.get("output") or []):
        for c in (it.get("content") or []):
            if isinstance(c, dict) and c.get("text"):
                parts.append(c["text"])
    return "\n".join(parts)


def _claude_user_text(rec):
    """该 claude 调用 input 里最新一条真实用户消息(跳过 tool_result 回填)=触发本回合的人话。"""
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


def _codex_user_text(rec):
    for it in reversed((rec.get("request") or {}).get("input") or []):
        if it.get("role") != "user" and it.get("type") not in (None, "message"):
            continue
        if it.get("role") != "user":
            continue
        for c in (it.get("content") or []):
            if isinstance(c, dict) and c.get("text"):
                return c["text"]
    return ""


def link_trajectory(ev):
    """把一个 bot 回合挂到它自己轨迹库里那次调用。返回挂接信息 dict（含匹配置信度）。"""
    kind = ev["kind"]
    date = time.strftime("%Y-%m-%d", time.localtime(ev["ts"]))
    needle = _norm(_strip_feishu(ev["text"]))[:40]
    if not needle:
        return {"linked": False, "why": "空正文"}

    if kind in ("claude", "codex"):
        subdir = "api-calls" if kind == "claude" else "codex-api-calls"
        gettext = _claude_resp_text if kind == "claude" else _codex_resp_text
        calls = _load_calls(subdir, date)
        # 候选 = 回复正文匹配；并列时取 ts 最接近本消息的那次。
        cands = [c for c in calls if needle and needle in _norm(gettext(c))]
        if not cands:
            # 退一步：只按 ts 邻近（±90s），标低置信。
            near = [c for c in calls if abs((c.get("ts") or 0) - ev["ts"]) <= 90]
            if not near:
                return {"linked": False, "why": f"{date} 库内未匹配到正文/邻近调用",
                        "store": subdir}
            c = min(near, key=lambda c: abs((c.get("ts") or 0) - ev["ts"]))
            return _pack(kind, subdir, c, gettext, conf="低(仅时间邻近)")
        c = min(cands, key=lambda c: abs((c.get("ts") or 0) - ev["ts"]))
        return _pack(kind, subdir, c, gettext, conf="高(正文匹配)")

    if kind == "antigravity":
        # transcript 不带飞书 msg_id：按 transcript mtime 落在本消息 ts 附近的会话对齐。
        best, bestd = None, 1e18
        for tp in glob.glob(str(ANTIGRAV_BRAIN / "*" /
                                ".system_generated" / "logs" / "transcript.jsonl")):
            d = abs(os.path.getmtime(tp) - ev["ts"])
            if d < bestd:
                best, bestd = tp, d
        if best and bestd <= 1800:
            conv = Path(best).parts[-4]
            return {"linked": True, "kind": kind, "store": "antigravity-transcript",
                    "id": conv, "conf": f"低(时间窗对齐, Δ{int(bestd)}s)",
                    "tools": [], "path": best}
        return {"linked": False, "kind": kind, "store": "antigravity-transcript",
                "why": "无 transcript 落在 ±30min 窗内（可能不在本机/已轮转）"}

    return {"linked": False, "why": f"未注册模型类型 {ev['who'][:14]}"}


def _pack(kind, subdir, c, gettext, conf):
    sid = (c.get("session_id") or "")
    if kind == "codex":
        sid = (((c.get("request") or {}).get("client_metadata") or {}).get("thread_id")
               or sid or "?")
    tools = _claude_tools(c) if kind == "claude" else _codex_tools(c)
    ask = _claude_user_text(c) if kind == "claude" else _codex_user_text(c)
    return {"linked": True, "kind": kind, "store": subdir,
            "id": (sid or "?")[:18], "conf": conf,
            "n_msg": len((c.get("request") or {}).get("messages")
                         or (c.get("request") or {}).get("input") or []),
            "tools": tools, "call_ts": c.get("ts"), "ask": ask}


def _codex_tools(rec):
    fr = (rec.get("response") or {}).get("final_response") or {}
    out = []
    for it in (fr.get("output") or []):
        if it.get("type") in ("function_call", "custom_tool_call"):
            out.append(it.get("name") or "?")
    # output_items 兜底
    for it in ((rec.get("response") or {}).get("output_items") or []):
        if isinstance(it, dict) and it.get("type") == "function_call":
            out.append(it.get("name") or "?")
    return out


# ── 渲染 ──

def gist(text, n=70):
    t = _strip_feishu(text).replace("\n", " ")
    return (t[:n] + "…") if len(t) > n else t


def _clean_ask(text):
    """从 bot 输入里的触发用户消息剥掉飞书/系统框架，留人话本体。"""
    import re
    t = text or ""
    # 砍掉系统注入块与框架行
    t = re.split(r"<system-reminder>|【每轮先给问题打标签】|【当前会话】", t)[0]
    t = re.sub(r"\[本条消息[^\]]*\]", "", t)
    t = re.sub(r"\[本条消息 @ 了[^\]]*\]", "", t)
    t = re.sub(r"\[用户「引用回复」[^\]]*\]", "", t)
    t = re.sub(r"---[^-]*你之前说.*?--- 以上 ---", "", t, flags=re.S)
    t = re.sub(r"<at\b[^>]*>.*?</at>", "", t)
    t = " ".join(t.split())
    # 丢掉非人话的伪触发：系统/身份块、harness 续写指令、压缩承接摘要等（非真实用户提问）
    NOISE = ("[System Instructions", "【你是谁】", "Output token limit",
             "<session>", "本地承接摘要", "【群组共识看板")
    if any(t.lstrip().startswith(p) or p in t[:40] for p in NOISE):
        return ""
    # 优先取「用户的意见：」之后的本体（引用回复场景框架后才是真追问）
    m = re.search(r"用户的意见[:：]\s*(.+)$", t)
    return (m.group(1) if m else t).strip()


def render(group_id, task_id, evs):
    L = [f"# 任务轨迹串接 · `{task_id}`", "",
         f"- 群：`{group_id}`",
         f"- 事件数：{len(evs)}",
         f"- 参与方：" + "、".join(sorted({e['name'] for e in evs})),
         "", "---", "",
         "## 统一时间线（人 + 多模型交错；bot 行下挂其自身轨迹）",
         "", "> 注：人的提问从 group-memory 拿不到（其 recent store 仅存 bot），"
         "这里**从触发该 bot 回合的轨迹输入里反推**——人话本就完整保存在 bot 看到的上下文里。",
         ""]
    last_ask = ""
    for e in evs:
        when = time.strftime("%m-%d %H:%M:%S", time.localtime(e["ts"]))
        if e["role"] != "user":
            lk = link_trajectory(e)
            # 反推触发本回合的人话，与上一条不同才插入一个 👤 用户事件
            ask = _clean_ask(lk.get("ask", "")) if lk.get("linked") else ""
            if ask and _norm(ask)[:60] != _norm(last_ask)[:60]:
                L.append(f"### {when}　👤 **用户(反推自轨迹输入)**")
                L.append("")
                L.append(f"> {ask[:160]}")
                L.append("")
                last_ask = ask
            L.append(f"### {when}　🤖 **{e['name']}**　msg=`{e['msg_id'][:14]}`")
            L.append("")
            L.append(f"> {gist(e['text'], 120)}")
            if lk.get("linked"):
                tools = lk.get("tools") or []
                tl = ("，工具：" + "、".join(f"`{t}`" for t in tools)) if tools else "，无工具调用"
                L.append("")
                L.append(f"　↳ 🔗 轨迹[{lk['kind']}] `{lk['store']}` "
                         f"id=`{lk['id']}` 置信={lk['conf']}{tl}")
            else:
                L.append("")
                L.append(f"　↳ ⚠️ 未挂到轨迹：{lk.get('why')}")
        else:
            L.append(f"### {when}　👤 **{e['name']}**　msg=`{e['msg_id'][:14]}`")
            L.append("")
            L.append(f"> {gist(e['text'], 120)}")
        L.append("")
    return "\n".join(L)


def list_tasks(group_id, evs):
    from collections import defaultdict
    tb = defaultdict(set)
    for e in evs:
        if e["task_id"]:
            tb[e["task_id"]].add(e["name"])
    print(f"群 {group_id} 各任务参与模型：")
    for tid, names in sorted(tb.items()):
        flag = "  ★多模型" if len([n for n in names]) > 1 else ""
        print(f"  {tid}: {'、'.join(sorted(names))}{flag}")


# ── v2：飞书历史当脊柱 + 回复链切分 ──
#
# 为什么换源（核过真实数据）：group-memory 的 recent/search 两库都**只存 bot 消息**
# （实测 0 条 user），用它做切分既缺人话、又缺回复边。飞书历史 API(im/v1/messages)是
# 权威源：带所有人的消息 + parent_id/root_id（回复链）——回复链正是任务边界的强信号。

# 飞书历史导出器属 feishu-history skill（留在 bot skills 仓，本提取代码跨仓调用它）。
# 默认指向 bot skills 仓的绝对路径，可用 FEISHU_HIST_EXPORT 覆盖（换机/换布局时）。
_HIST_EXPORT = Path(os.environ.get(
    "FEISHU_HIST_EXPORT",
    "/home/agent/lian-server-bot/skills/feishu-plugin/skills/"
    "feishu-history/scripts/export_raw_jsonl.py"))


def _hist_text(it):
    """从一条飞书历史 item 取可读正文：卡片优先用已解析文本，文本消息取 text 字段。"""
    rt = it.get("_resolved_card_text")
    if rt:
        return rt
    c = (it.get("body") or {}).get("content") or ""
    try:
        j = json.loads(c)
        return j.get("text") or j.get("template") or c
    except (json.JSONDecodeError, TypeError):
        return c


def fetch_history(group_id, start_s, end_s, hist_file=None):
    """拉飞书群历史，转成与 fetch_stream 同构的事件（多 role + 真实人话 + 回复边）。
    hist_file 给定则直接读已导出的 jsonl，免重复拉取（迭代用）。"""
    import subprocess
    import tempfile
    if hist_file:
        path = hist_file
    else:
        d = tempfile.mkdtemp(prefix="stitch_hist_")
        subprocess.run(["python3", str(_HIST_EXPORT), group_id,
                        str(int(start_s)), str(int(end_s)), d],
                       check=True, capture_output=True, text=True)
        path = os.path.join(d, "messages.raw.jsonl")
    evs = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        it = json.loads(line)
        if it.get("msg_type") == "system":
            continue
        # 撤回消息：飞书带权威结构标志 deleted=true —— 语言无关，确定性丢弃。
        # 比匹配文本可靠：#18「This message was recalled」曾漏网，正因旧逻辑只匹配中文
        # 卡「🗑️已撤回」、匹配不上英文系统占位。认标志位一劳永逸。
        if it.get("deleted"):
            continue
        s = it.get("sender") or {}
        st = s.get("sender_type")
        sid = s.get("id") or ""
        if st == "app":
            reg = APP_REGISTRY.get(sid, {})
            role, who, name = "bot", sid, reg.get("name") or sid[:14]
            kind = reg.get("kind")
        elif st == "user":
            role, who = "user", sid
            name = USER_REGISTRY.get(sid) or "用户"
            kind = None
        else:
            continue
        evs.append({
            "ts": int(it.get("create_time") or 0) / 1000.0,   # 飞书毫秒(字符串) → 秒
            "role": role, "who": who, "name": name, "kind": kind,
            "msg_id": it.get("message_id") or "",
            "parent_id": it.get("parent_id") or "",
            "task_id": "", "thread_id": "",
            "text": _hist_text(it),
        })
    evs.sort(key=lambda e: e["ts"])
    return evs


def _is_noise(text):
    t = (text or "").strip()
    if (not t) or t.startswith("👀") or t.startswith("🗑️") or t.startswith("https://"):
        return True
    # 撤回兜底：源头若没带 deleted 结构标志（如 group-memory 二次落盘），认系统占位文本
    if "this message was recalled" in t.lower():
        return True
    # 框架系统提示 / 压缩通知 / 漏进来的裸卡片 JSON —— 非真实任务内容
    if t.startswith(("⚠️ 本群会话上下文", "🧹 上下文已触发压缩", '{"title"', '{"config"',
                     '{"elements"', "📊", "✅ 已")):
        return True
    # 排队/并行调度卡（🕒 已入队 q1… / ⚡ 已把 q1 转为并行…）—— 纯系统 UI，非任务内容
    if t.startswith(("🕒 已入队", "⚡ 已把")):
        return True
    return False


def _topic_of(text):
    import re
    m = re.match(r"\s*〔([^〕]{1,12})〕", text or "")
    tp = m.group(1) if m else ""
    return "" if tp == "闲聊" else tp  # 闲聊不作话题，免污染 carry-forward 切分


# B 路线：抵达解决态的收尾语——用户明确表示这件事了结（满意/收工）。仅用于判「时间线
# 最后一个簇」是否已终结；非末簇靠「时间推进且没再回来」即认定终结，不需要收尾语。
_CLOSURE = ("搞定", "完成了", "做完了", "可以了", "没问题了", "解决了", "结了", "收工",
            "辛苦了", "谢谢", "多谢", "完美", "就这样", "ok了", "好了就", "done", "通过了")

_SUSPICIOUS_REPLY_GAP = 24 * 3600        # 旧时间轴阈值（仅在无 sim_fn 时回退用）
# 内容轴地板：子消息正文 vs 父段正文的余弦相似度低于此 → 判这条回复内容跟父段对不上
# （很可能挂错），降级为弱信号。在 eval_stitch_guard.py 的人工挂错/续接金标上标定。
_REPLY_CONTENT_FLOOR = float(os.environ.get("REPLY_CONTENT_FLOOR", "0.60"))


def _segment_real_topic(seg):
    tp = seg.get("topic") or "?"
    return "" if tp == "?" else tp


def _seg_join_text(seg):
    """父段可读正文拼接（给内容轴算相似度用，截断防超长）。"""
    parts = []
    for e in seg.get("rows", []):
        if _is_noise(e["text"]):
            continue
        t = _strip_feishu(e["text"]).strip()
        if t:
            parts.append(t)
    return "\n".join(parts)[:2000]


def _should_strong_stitch_by_reply(cur_seg, parent_seg, child_event, parent_event, sim_fn=None):
    """parent_id 是强信号，但挂错回复会把两个原子段焊死。

    换轴（张耀明 2026-06-29 第2条根治）：判「该不该强缝合」看**这条回复的内容跟父段对不对得上**，
    不再赌时间差——点错回复跟隔多久无关，旧的「跨≥24h」闸只拦得住跨天挂错、放过更常见的同天挂错。
    同话题（或话题缺失）一律强缝（强信号、便宜短路）；异话题时算子消息 vs 父段的内容相似度，
    低于地板=对不上=很可能挂错 → 降级为弱信号，留给 LLM 语义层。
    sim_fn(child_event, parent_seg)->float|None 由调用方注入（bge-m3）；缺省/失败回退到旧时间轴。
    """
    cur_topic = _segment_real_topic(cur_seg)
    parent_topic = _segment_real_topic(parent_seg)
    if not cur_topic or not parent_topic or cur_topic == parent_topic:
        return True
    if sim_fn is not None:
        sim = sim_fn(child_event, parent_seg)
        if sim is not None:
            return sim >= _REPLY_CONTENT_FLOOR
    # 无内容信号（离线/embedding 不可用）→ 回退旧时间轴行为，向后兼容、绝不崩
    gap = abs((child_event.get("ts") or 0) - (parent_event.get("ts") or 0))
    return gap < _SUSPICIOUS_REPLY_GAP


def build_reply_sim_fn():
    """生产用 sim_fn：bge-m3 算「子消息正文 vs 父段正文」余弦。embedding 不可用时返回 None
    （让 _should_strong_stitch_by_reply 自动回退时间轴）。延迟导入，离线/无网时不影响 import。"""
    try:
        import embed_util
    except Exception:
        return None

    def _sim(child_event, parent_seg):
        ctext = _strip_feishu(child_event.get("text", "")).strip()
        ptext = _seg_join_text(parent_seg)
        if not ctext or not ptext:
            return None
        return embed_util.text_sim(ctext, ptext)

    return _sim


def atomic_segments(evs, gap=2400, sim_fn=None):
    """把历史切成「原子段」——pass①(话题+gap 主切) + pass②(parent_id 回复链缝合/可疑降级)，
    **不做** pass③ 词项兜底合并。原子段 = 不跨硬时间断档/不跨持续话题切换的最小连续活动块，
    刻意「宁过切勿过并」：每个原子段几乎总是单一任务的一段，但同一任务可能被切成多个原子段
    （跨夜/隔时段续接）。llm_segment 在此之上做语义分组把同任务的原子段并回（治本）。

    话题用 carry-forward：bot 的〔标签〕沿用给后续没标签的消息（含用户提问）。"""
    if not evs:
        return []
    # 每条的有效话题（carry-forward）
    eff, last = [], ""
    for e in evs:
        tp = _topic_of(e["text"]) if e["role"] == "bot" else ""
        if tp:
            last = tp
        eff.append(last or "?")
    # ① 话题+gap 主切（话题切换需「持续」：下一条仍是新话题才算，避免单条 blip）
    segs, cur = [], None
    for i, e in enumerate(evs):
        g = (e["ts"] - evs[i - 1]["ts"]) if i else 0
        new = False
        if cur is None or g >= gap:
            new = True
        elif eff[i] != cur["topic"] and eff[i] != "?":
            nxt = eff[i + 1] if i + 1 < len(eff) else eff[i]
            if nxt == eff[i]:
                new = True
        if new:
            cur = {"topic": eff[i], "rows": []}
            segs.append(cur)
        else:
            # 不把「被吸收的孤立 blip / 未知话题'?'」写进段话题：else 分支只在
            # ①同话题 ②非持续的 blip ③'?' 三种情形进入，三种都该保持段话题不变。
            # 旧代码在这里 cur["topic"]=eff[i] 会让 blip 污染段话题，害它「后一条」
            # 被误判成新的持续切换而多切一刀（张耀明 2026-07-02 案例：
            # [清洗,清洗,运维,清洗,清洗] 本该 1 段却切成 2 段）。删掉即修复。
            pass
        cur["rows"].append(e)
    # ② 回复链缝合：跨段的 parent 边把两段并回
    seg_of = {e["msg_id"]: si for si, s in enumerate(segs) for e in s["rows"]}
    event_by_msg = {e["msg_id"]: e for e in evs if e.get("msg_id")}
    uf = list(range(len(segs)))

    def find(x):
        while uf[x] != x:
            uf[x] = uf[uf[x]]
            x = uf[x]
        return x
    for e in evs:
        p = e["parent_id"]
        # p 必须非空：空 parent_id = 没有回复关系，绝不当回复边（否则脏数据里空 msg_id 会被
        # 当成共同父、把无关段误合并，且 event_by_msg[p] 会 KeyError 崩掉整群切分）。
        if p and p in seg_of and find(seg_of[p]) != find(seg_of[e["msg_id"]]):
            child_si, parent_si = seg_of[e["msg_id"]], seg_of[p]
            parent_ev = event_by_msg.get(p)
            if parent_ev is None or _should_strong_stitch_by_reply(segs[child_si], segs[parent_si], e, parent_ev, sim_fn):
                uf[find(child_si)] = find(parent_si)
    from collections import defaultdict
    merged = defaultdict(list)
    for si, s in enumerate(segs):
        merged[find(si)].extend(s["rows"])
    out = []
    for rows in merged.values():
        rows.sort(key=lambda e: e["ts"])
        if all(_is_noise(e["text"]) for e in rows):
            continue  # 纯 ack/链接/空 段丢掉
        out.append(rows)
    out.sort(key=lambda c: c[0]["ts"])
    return out


def segment_history(evs, gap=2400):
    """把一段历史切成若干子任务（启发式·任务粒度边界的 fallback；治本走 llm_segment.py）。

    切法（核过真实数据后定的）：
    主刀 = 话题(〔标签〕) + 时间 gap —— 相邻消息 gap≥gap 秒、或话题持续切换 → 边界；
    缝合 = parent_id 回复链 —— 若一条回复跨过了边界（回复到前一段的消息），一般把两段并回；
           但跨≥24h且两边明确异话题的回复先降级，避免挂错回复把两段硬焊死。
           即「回复边只用来防误切，不当主聚类」（纯回复链太稀疏，bot 卡片多不带 parent）。
    话题用 carry-forward：bot 的〔标签〕沿用给后续没标签的消息（含用户提问）。

    ⚠️ 已知天花板（张耀明 2026-06-27 实测，故撤掉 A 跨断档缝合）：纯启发式（话题标签+gap+
    词重叠）在密集多任务群里必然在「过切/过并」之间摇摆，调不出合理任务边界——跨天续接、
    carry-forward 标签漂移、大簇裸词计数恒超标都治不了。任务粒度的切分治本方案是 LLM 语义
    切分（见 llm_segment.py）；本函数保留为 fallback / llm_segment 的原子段来源。"""
    from collections import defaultdict
    clusters_pass1 = atomic_segments(evs, gap)
    if not clusters_pass1:
        return []

    # ③ 词项重叠兜底缝合（专治没有 parent_id 的跨簇喷射）。短档(≤20min)紧邻簇、词重叠≥4
    #    才并，纯防误切；跨断档（>20min，含跨夜/跨天）的任务续接交给 llm_segment 治本，本层不碰。
    uf2 = list(range(len(clusters_pass1)))
    def find2(x):
        while uf2[x] != x:
            uf2[x] = uf2[uf2[x]]
            x = uf2[x]
        return x

    cluster_terms = []
    for cl in clusters_pass1:
        t = set()
        for e in cl:
            if not _is_noise(e["text"]):
                t |= _terms(e["text"])
        cluster_terms.append(t)

    for i in range(len(clusters_pass1) - 1):
        if find2(i) == find2(i + 1):
            continue
        cl1 = clusters_pass1[i]
        cl2 = clusters_pass1[i + 1]
        # 护栏1：时间窗限制（前后簇间隔 <= 1200秒/20分钟）
        gap = cl2[0]["ts"] - cl1[-1]["ts"]
        if gap > 1200:
            continue
        # 护栏2：词项重叠置信度（score >= 4）
        overlap = cluster_terms[i] & cluster_terms[i + 1]
        if _score(overlap, overlap) >= 4:
            uf2[find2(i + 1)] = find2(i)

    final_merged = defaultdict(list)
    for i, cl in enumerate(clusters_pass1):
        final_merged[find2(i)].extend(cl)
        
    clusters = []
    for rows in final_merged.values():
        rows.sort(key=lambda e: e["ts"])
        clusters.append(rows)
    return clusters


def compute_terminal(clusters):
    """为每个簇判 terminal（任务是否真终结）—— B 路线（张耀明 2026-06-27）。
    喂给 assemble_subtask(terminal=...) → map_outcome 的 terminal gate：未终结一律 incomplete，
    绝不给没收尾的活盖成功章。

    离线全量已知未来，确定性口径：
      · **非**时间线最后一个簇 → terminal=True：时间已推进到别的任务、之后没再回来。
        （同任务的跨断档续段由 llm_segment 语义层缝进同一簇，故这里「没回来」=真结束；
         结束态的好坏由 P1 沉默即认可 / failure_judge 再判，与「是否终结」正交。）
      · 时间线**最后一个**簇 → 默认 terminal=False(incomplete)：观测窗右沿≈“现在”，
        末尾任务往往仍在进行、观测不到它收尾；除非末尾**用户**消息含明确收尾语(_CLOSURE)
        才算已抵达解决态 → terminal=True。末尾是 bot 回合(用户未回)=沉默歧义→仍判未终结
        （区别于非末簇的「definitive 沉默」：那是用户转去别的任务，这是还没来得及回）。

    ⚠️「时间线最后一个簇」= **最近被碰过**的任务（取簇内**最晚**事件 ts），不是「最先开始」。
       早先按 clusters[i][0].ts（最早事件）取，交织场景翻车：一个早起步、却一直做到现在的长
       任务被误判已终结，反把一个晚起步、早就停了的短簇当成「还在飞」（张耀明 2026-06-29 实测：
       「按任务拆分」尾 06-29 仍在动却判已终结，06-26 就停的 blob 反当未终结）。故改取最晚 ts。"""
    if not clusters:
        return []
    last = max(range(len(clusters)), key=lambda i: max(e["ts"] for e in clusters[i]))
    out = [True] * len(clusters)
    resolved = False
    for e in sorted(clusters[last], key=lambda e: e["ts"], reverse=True):
        if _is_noise(e["text"]):
            continue
        if e["role"] == "user":
            t = _norm(_strip_feishu(e["text"]))
            resolved = any(k in t for k in _CLOSURE)
        else:
            resolved = False   # 末尾是 bot 回合、用户未回 → 沉默歧义(可能还没回) → 未终结
        break
    out[last] = resolved
    return out


def _cluster_label(cl):
    from collections import Counter
    topics = Counter(_topic_of(e["text"]) for e in cl
                     if e["role"] == "bot" and _topic_of(e["text"]))
    bots = sorted({e["name"] for e in cl if e["role"] == "bot"})
    dom = topics.most_common(1)[0][0] if topics else "(无标签)"
    ask = next((_strip_feishu(e["text"]) for e in cl
                if e["role"] == "user" and not _is_noise(e["text"])), "")
    if not ask:
        ask = next((_strip_feishu(e["text"]) for e in cl
                    if not _is_noise(e["text"])), "")
    return dom, bots, ask


def render_segmented(group_id, clusters, window_desc):
    L = [f"# 任务重切 + 轨迹串接（v2 · 飞书历史脊柱）", "",
         f"- 群：`{group_id}`",
         f"- 窗口：{window_desc}",
         f"- 重切出子任务：**{len(clusters)}** 个",
         "", "> 源=飞书历史 API（im/v1/messages，带所有人消息+回复链），"
         "切分主刀=parent_id 回复链，bot 回合挂回各自轨迹库。", "",
         "---", ""]
    for k, cl in enumerate(clusters, 1):
        dom, bots, ask = _cluster_label(cl)
        t0 = time.strftime("%m-%d %H:%M", time.localtime(cl[0]["ts"]))
        t1 = time.strftime("%H:%M", time.localtime(cl[-1]["ts"]))
        real = [e for e in cl if not _is_noise(e["text"])]
        L += [f"## 子任务 {k}　〔{dom}〕　{t0}→{t1}",
              f"- {len(real)} 条有效消息 | 参与 bot：{'、'.join(bots) or '—'}",
              f"- 起头：{ask[:80]}", "",
              "<details><summary>展开时间线</summary>", ""]
        last_ask = ""
        for e in cl:
            if _is_noise(e["text"]):
                continue
            when = time.strftime("%m-%d %H:%M:%S", time.localtime(e["ts"]))
            if e["role"] == "user":
                a = _strip_feishu(e["text"])
                if _norm(a)[:60] != _norm(last_ask)[:60]:
                    L += [f"- {when}　👤 **{e['name']}**：{gist(e['text'], 90)}"]
                    last_ask = a
            else:
                lk = link_trajectory(e)
                if lk.get("linked"):
                    tl = ("，工具：" + "、".join(f"`{t}`" for t in (lk.get("tools") or []))
                          ) if lk.get("tools") else ""
                    tag = (f"　↳🔗[{lk['kind']}] `{lk['id']}` {lk['conf']}{tl}")
                else:
                    tag = f"　↳⚠️ {lk.get('why', '未挂接')}"
                L += [f"- {when}　🤖 **{e['name']}**：{gist(e['text'], 90)}",
                      f"  {tag}"]
        L += ["", "</details>", ""]
    return "\n".join(L)


# ── 子需求分解 + 关系边 + 难例标记 ──
# 路线2（廉莲拍板）：最有价值的是「初始做不好→人纠正→做好」的难例。这些数据全藏在
# 子需求内的「返工/纠偏」上——所以分解到子需求层、把人纠正的回合标出来，就是路线2的筛子：
#   带 ↺(返工) / ↦(纠偏) 的子需求 = 高价值难例（留）；一遍过的 = 模型已会（丢）。

# 重复/返工词：人在说「同一件事还没好」——必须是「重复」语义，不含首次报障(如「没反应」单独出现)
_REWORK = ("还是", "仍然", "依旧", "还有", "没变", "一样", "重新",
           "没解决", "没生效", "没起作用", "又出", "又报", "又不", "怎么又")
# 纠偏词：人当场把跑偏的方向掰回来（多字词，压低误命中）
_CORRECTION = ("不对", "不是", "别给", "别直接", "别跟我", "不要", "应该是",
               "错了", "搞错", "看代码", "看下代码", "看下到底", "不该")
# 催促/无内容：人在戳一下等回应，不开新子需求
_FILLER = {"hi", "你好", "在", "在吗", "在么", "能看到吗", "看得到吗", "?", "？",
           "？？？", "??", "ok", "好的", "好", "嗯", "收到", "你好？", "在？"}
# 承接推进：把当前子需求往下一步推（不是新需求），不开新段、也不算难例
_CONT_PREFIX = ("那你先", "那就先", "你先", "先进行", "先跑", "先做", "先备份", "先回答",
                "先将", "接着", "继续", "然后", "那现在", "好的那", "那你继续")
import re as _re
_CMD_RE = _re.compile(r"^(stop|done|parallel|q\d|sessions?|board|!|/|#\d|model)", _re.I)


def _user_class(text):
    """用户消息分类 → filler(催促/噪声) / continuation(承接推进) / rework(返工) /
    correction(纠偏) / fresh(新需求)。rework/correction = 人在纠正 → 该子需求是难例。"""
    t = _norm(_strip_feishu(text))
    if not t or "messagewasrecalled" in t.lower() or t.startswith(("ssh-ed25519", "ssh-rsa")):
        return "filler"
    if _CMD_RE.match(t):            # 看板命令(stop q1 / done / !ctx…)不是需求
        return "filler"
    if t in _FILLER or (len(t) <= 3 and not any(c in t for c in "怎如何为什吗么")):
        return "filler"
    if any(k in t for k in _CORRECTION):
        return "correction"
    if any(k in t for k in _REWORK):
        return "rework"
    # 承接推进：连接词起头 且 (短 或 「…吧」祈使)；长问句(如「那你能…吗」)仍算新需求
    if t.startswith(_CONT_PREFIX) and (len(t) <= 16 or t.endswith("吧")):
        return "continuation"
    return "fresh"


# 指代聚类（把返工/纠偏路由回它真正所属的线程，而非线性挂给最近段）：
# 判别词停用表——通用连接/语气/返工标记本身不是话题判别词，纳入会让两条线误并。
_STOP_BIGRAMS = {
    "这个", "那个", "一下", "一个", "可以", "什么", "怎么", "没有", "就是", "已经",
    "现在", "还是", "仍然", "不是", "不对", "还有", "我们", "你们", "他们", "因为",
    "所以", "但是", "如果", "或者", "这样", "那样", "知道", "觉得", "应该", "这里",
    "那里", "看下", "看看", "目前", "然后", "继续", "先做", "一样", "重新", "搞错",
}


def _terms(text):
    """抽出一条消息的话题判别词集合。强词(S:)=latin/数字 token + 乱码符号''(转义bug 标志)，
    权重 2；弱词(W:)=CJK 二元组，权重 1。用于跨段的「指代/主题」匹配。"""
    import re
    t = _strip_feishu(text)
    out = set()
    for m in re.findall(r"[a-z0-9_]{2,}", t.lower()):
        out.add("S:" + m)
    if re.search(r"[\"'‘’“”]{2,}|''|“”", t):  # 连续引号=@转义bug 的乱码症状，强判别
        out.add("S:__quote__")
    for run in re.findall(r"[一-鿿]{2,}", t):
        for i in range(len(run) - 1):
            bg = run[i:i + 2]
            if bg not in _STOP_BIGRAMS:
                out.add("W:" + bg)
    return out


def _score(a, b):
    return sum(2 if x.startswith("S:") else 1 for x in (a & b))


def _new_sub(e):
    return {"rows": [], "anchor": _strip_feishu(e["text"]) if e["role"] == "user" else "",
            "rework": [], "corrections": [], "terms": set(), "msg_ids": set(),
            "t0": e["ts"], "t1": e["ts"]}


def _route_explain(e, subs, cur, term=True):
    """同 _route，但额外返回判定理由 (why, detail)，供 --trace 解释每条怎么挂的。
       ① parent_id 回复边(强) → ② 词项重叠≥2(指代/同主题, term=True 时) → ③ 兜底=当前段。"""
    p = e.get("parent_id")
    if p:
        for s in subs:
            if p in s["msg_ids"]:
                return s, ("parent", None)
    if term:
        et = _terms(e["text"])
        if et:
            best, bestn, besthit = None, 0, set()
            for s in subs:
                hit = et & s["terms"]
                n = _score(et, hit)
                if n > bestn:
                    best, bestn, besthit = s, n, hit
            if best is not None and bestn >= 2:
                pretty = sorted(x.split(":", 1)[1] for x in besthit)
                return best, ("term", f"≥2命中:{'、'.join(pretty)}")
    return cur, ("fallback", None)


def _route(e, subs, cur, term=True):
    return _route_explain(e, subs, cur, term)[0]


# 分类命中的关键词(给 --trace 显示「凭哪个词判成 rework/correction」)
def _class_kw(text):
    t = _norm(_strip_feishu(text))
    for k in _CORRECTION:
        if k in t:
            return f"correction:{k}"
    for k in _REWORK:
        if k in t:
            return f"rework:{k}"
    if t.startswith(_CONT_PREFIX):
        return "continuation:连接词起头"
    return ""


def _absorb(dst, src):
    """把 src 子需求并进 dst（清理 0-bot 跟进单条用）。"""
    dst["rows"].extend(src["rows"])
    dst["rework"].extend(src["rework"])
    dst["corrections"].extend(src["corrections"])
    dst["terms"] |= src["terms"]
    dst["msg_ids"] |= src["msg_ids"]
    dst["t0"] = min(dst["t0"], src["t0"])
    dst["t1"] = max(dst["t1"], src["t1"])


def decompose_task(cluster, trace=None):
    """把一个任务(cluster)分解成若干子需求。边界=用户的「新需求(fresh)」开新段；
    返工/纠偏/承接/bot回合按「指代路由」(parent_id→词项重叠→兜底)归回所属线程——
    这样交织的并发线程不会串台，每段的真实时间跨度也得以保留(并行∥才浮得出来)。
    trace 给一个 list 时，逐条记录「走了哪条规则、为什么挂到那段」供 --trace 解释。"""
    subs, cur = [], None
    sid = {}  # id(sub) → 创建序号，trace 里指代某段用
    for e in cluster:
        if _is_noise(e["text"]):
            if trace is not None:
                trace.append({"e": e, "skip": "噪声/ack/链接/卡片JSON"})
            continue
        why = None
        if e["role"] == "user":
            cls = _user_class(e["text"])
            if cur is None or cls == "fresh":
                sub = _new_sub(e)        # 每个真·新提问开一段
                subs.append(sub)
                sid[id(sub)] = len(subs)
                act = "🆕开新段(首段)" if cur is None else "🆕开新段(fresh)"
            else:
                sub, why = _route_explain(e, subs, cur)   # 返工/纠偏/承接路由回所属线程
                act = "→并入"
                if cls == "rework":
                    sub["rework"].append(e)
                elif cls == "correction":
                    sub["corrections"].append(e)
            cur = sub                    # 焦点跟着用户走（bot 回合不改焦点）
            if trace is not None:
                trace.append({"e": e, "cls": cls, "kw": _class_kw(e["text"]),
                              "act": act, "why": why, "tgt": sid.get(id(sub))})
        else:  # bot 回合：只认 parent_id 回复边，不靠词重叠（防 mega-blob）
            if cur is None:
                sub = _new_sub(e)
                subs.append(sub)
                sid[id(sub)] = len(subs)
                act = "🆕开新段(任务以bot起头)"
            else:
                sub, why = _route_explain(e, subs, cur, term=False)
                act = "→并入"
            if trace is not None:
                trace.append({"e": e, "cls": "bot", "kw": "", "act": act,
                              "why": why, "tgt": sid.get(id(sub))})
        sub["rows"].append(e)
        sub["terms"] |= _terms(e["text"])
        if e["msg_id"]:
            sub["msg_ids"].add(e["msg_id"])
        sub["t0"] = min(sub["t0"], e["ts"])
        sub["t1"] = max(sub["t1"], e["ts"])
    # 清理：纯跟进单条(0 个 bot 回合)并回它 parent_id 所指的线程——这些是没拿到/没挂上
    # bot 回复的跟进消息("我发版了你再试试"),不该独立成段。
    msg_to_sub = {mid: s for s in subs for mid in s["msg_ids"]}
    for s in list(subs):
        if any(e["role"] == "bot" for e in s["rows"]):
            continue
        tgt = None
        for e in s["rows"]:
            p = e.get("parent_id")
            if p and msg_to_sub.get(p) not in (None, s):
                tgt = msg_to_sub[p]
                break
        if tgt is not None:
            _absorb(tgt, s)
            subs.remove(s)
    subs.sort(key=lambda s: s["t0"])
    for s in subs:
        s["rows"].sort(key=lambda e: e["ts"])   # 路由回来的返工是后到的，按 ts 复原时间线
        s["bots"] = sorted({e["name"] for e in s["rows"] if e["role"] == "bot"})
        s["hard"] = bool(s["rework"] or s["corrections"])  # 难例判定
        if not s["anchor"]:
            s["anchor"] = next((_strip_feishu(e["text"]) for e in s["rows"]
                                if not _is_noise(e["text"])), "")
    return subs


def normalize_event(e):
    """Normalize pool/history event records to the event contract used here."""
    out = copy.deepcopy(e or {})
    out["msg_id"] = str(out.get("msg_id") or "")
    out["parent_id"] = str(out.get("parent_id") or "")
    out["role"] = str(out.get("role") or "?")
    out["name"] = str(out.get("name") or out.get("sender_name") or out.get("who") or "?")
    out["text"] = str(out.get("text") or "")
    out["ts"] = float(out.get("ts") or out.get("timestamp") or 0)
    out.setdefault("who", out.get("user_id") or out.get("bot_id") or "")
    out.setdefault("kind", None)
    return out


def load_pool_record(pool_file, line_no=None, t0_msg_id=None):
    """Load one record from user_corrections_pool*.jsonl by 1-based line or t0 msg_id."""
    if not line_no and not t0_msg_id:
        raise ValueError("line_no or t0_msg_id is required")
    with open(pool_file, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            rec_t0 = ((rec.get("t0") or {}).get("msg_id") or
                      ((rec.get("t0") or {}).get("event") or {}).get("msg_id"))
            if (line_no and i == line_no) or (t0_msg_id and rec_t0 == t0_msg_id):
                rec["_pool_line"] = i
                return rec
    key = f"line {line_no}" if line_no else f"t0_msg_id {t0_msg_id}"
    raise ValueError(f"pool record not found: {key}")


def pool_record_events(rec, include_t0=True):
    """Return snapshot context events plus the T0 user event, de-duplicated by msg_id."""
    events = [normalize_event(e) for e in ((rec.get("context") or {}).get("events") or [])]
    if include_t0:
        t0_event = ((rec.get("t0") or {}).get("event") or {})
        if t0_event:
            events.append(normalize_event(t0_event))
    by_key = {}
    for idx, e in enumerate(events):
        key = e.get("msg_id") or f"idx:{idx}"
        by_key[key] = e
    return sorted(by_key.values(), key=lambda e: (e.get("ts") or 0, e.get("msg_id") or ""))


def _event_brief(e):
    return {
        "msg_id": e.get("msg_id") or "",
        "parent_id": e.get("parent_id") or "",
        "ts": e.get("ts") or 0,
        "role": e.get("role") or "?",
        "name": e.get("name") or "?",
        "text": _strip_feishu(e.get("text") or ""),
    }


def _subreq_brief(sub):
    return {
        "anchor": sub.get("anchor") or "",
        "hard": bool(sub.get("hard")),
        "bots": sub.get("bots") or [],
        "msg_ids": sorted(sub.get("msg_ids") or []),
        "rework_msg_ids": [e.get("msg_id") or "" for e in sub.get("rework") or []],
        "correction_msg_ids": [e.get("msg_id") or "" for e in sub.get("corrections") or []],
        "t0": sub.get("t0") or 0,
        "t1": sub.get("t1") or 0,
        "events": [_event_brief(e) for e in sub.get("rows") or []],
    }


def analyze_pool_record(rec, gap=2400, include_t0=True, sim_fn=None):
    """Deterministic case snapshot analysis for replay data.

    Input is one user_corrections_pool_vfinal... row. Output is structured JSON:
    normalized context events, atomic task segments, and decompose_task subreqs.
    No LLM call and no state write.
    """
    events = pool_record_events(rec, include_t0=include_t0)
    atoms = atomic_segments(events, gap=gap, sim_fn=sim_fn)
    tasks = []
    terminal = compute_terminal(atoms)
    for i, rows in enumerate(atoms, 1):
        trace = []
        subs = decompose_task(rows, trace=trace)
        dom, bots, ask = _cluster_label(rows)
        tasks.append({
            "task_no": i,
            "topic": dom,
            "ask": ask,
            "bots": bots,
            "terminal": terminal[i - 1] if i - 1 < len(terminal) else True,
            "event_count": len(rows),
            "msg_ids": [e.get("msg_id") or "" for e in rows],
            "events": [_event_brief(e) for e in rows],
            "subreq_count": len(subs),
            "subreqs": [_subreq_brief(s) for s in subs],
            "trace": [{
                "msg_id": (r.get("e") or {}).get("msg_id") or "",
                "cls": r.get("cls") or "",
                "kw": r.get("kw") or "",
                "act": r.get("act") or "",
                "why": r.get("why"),
                "target": r.get("tgt"),
                "skip": r.get("skip") or "",
            } for r in trace],
        })
    return {
        "schema_version": 1,
        "source": "user_corrections_pool_snapshot",
        "pool_line": rec.get("_pool_line"),
        "group_id": rec.get("group_id") or "",
        "t0_msg_id": (rec.get("t0") or {}).get("msg_id") or "",
        "bot_error_msg_id": (rec.get("source_vfinal_row") or {}).get("anchor_msg_id") or "",
        "corrector_msg_id": (rec.get("source_vfinal_row") or {}).get("corrector_msg_id") or "",
        "include_t0": include_t0,
        "gap": gap,
        "event_count": len(events),
        "task_count": len(tasks),
        "subreq_count": sum(t["subreq_count"] for t in tasks),
        "events": [_event_brief(e) for e in events],
        "tasks": tasks,
    }


def task_edges(subs):
    """相邻子需求的边：时间重叠→并行∥，否则→顺序⊸。（返工↺是子需求自环，单独在节点上标）"""
    edges = []
    for i in range(len(subs) - 1):
        a, b = subs[i], subs[i + 1]
        edges.append((i, i + 1, "∥并行" if b["t0"] < a["t1"] else "⊸顺序"))
    return edges


def render_decomposed(group_id, clusters, window_desc):
    tasks = [(cl, decompose_task(cl)) for cl in clusters]
    terminal = compute_terminal(clusters)            # B：每个任务是否真终结
    n_sub = sum(len(s) for _, s in tasks)
    n_hard = sum(1 for _, s in tasks for r in s if r["hard"])
    n_incomplete = sum(1 for t in terminal if not t)
    L = ["# 任务→子需求→关系图 + 难例标记（v3 · 路线2筛子）", "",
         f"- 群：`{group_id}`　窗口：{window_desc}",
         f"- 任务 **{len(tasks)}** 个 → 子需求 **{n_sub}** 个；其中**难例 {n_hard}** 个"
         f"（带返工↺/纠偏↦，路线2高价值），平凡 {n_sub - n_hard} 个（一遍过，可丢）",
         f"- 终结判定(B)：**{n_incomplete}** 个任务未终结(⏳incomplete) —— 其子需求 outcome 一律 "
         "incomplete，不盖成功章；其余已终结任务按 P1 沉默即认可判 success/corrected/failure。",
         "", "> 子需求=改写单位。边界:用户新需求(fresh)开段；返工/纠偏/承接按指代路由"
         "(parent_id回复边→词项重叠→兜底)归回所属线程，交织并发不串台。",
         "> 边：顺序⊸ / 并行∥(时间重叠) / 返工↺(自环) / 纠偏↦(人掰方向)。**启发式首版，待校口径。**",
         "", "---", ""]
    for ti, (cl, subs) in enumerate(tasks, 1):
        dom, bots, ask = _cluster_label(cl)
        t0 = time.strftime("%m-%d %H:%M", time.localtime(cl[0]["ts"]))
        t1 = time.strftime("%H:%M", time.localtime(cl[-1]["ts"]))
        hard_here = sum(1 for r in subs if r["hard"])
        tflag = "" if terminal[ti - 1] else "　⏳**未终结(incomplete)**"
        L += [f"## 任务 {ti}　〔{dom}〕　{t0}→{t1}{tflag}",
              f"- 参与 bot：{'、'.join(bots) or '—'} | 子需求 {len(subs)} 个"
              f"（难例 {hard_here}）", ""]
        for k, s in enumerate(subs, 1):
            badge = "🔴难例" if s["hard"] else "⚪平凡"
            st0 = time.strftime("%H:%M", time.localtime(s["t0"]))
            st1 = time.strftime("%H:%M", time.localtime(s["t1"]))
            nlink = sum(1 for e in s["rows"] if e["role"] == "bot"
                        and not _is_noise(e["text"]) and link_trajectory(e).get("linked"))
            nbot = sum(1 for e in s["rows"] if e["role"] == "bot" and not _is_noise(e["text"]))
            L += [f"### r{k} {badge}　{st0}–{st1}　{gist(s['anchor'], 60) or '(无锚)'}",
                  f"- 主导：{'、'.join(s['bots']) or '—'} | bot回合 {nbot}（挂轨迹 {nlink}）"]
            for e in s["rework"]:
                L += [f"  - ↺返工　👤{e['name']}：{gist(_strip_feishu(e['text']), 50)}"]
            for e in s["corrections"]:
                L += [f"  - ↦纠偏　👤{e['name']}：{gist(_strip_feishu(e['text']), 50)}"]
            L += [""]
        es = task_edges(subs)
        if es:
            chain = "　".join(f"r{i+1}─{t}→r{j+1}" for i, j, t in es)
            L += [f"**子需求关系**：{chain}", ""]
        L += ["---", ""]
    return "\n".join(L)


def _why_str(why):
    if not why:
        return ""
    kind, detail = why
    if kind == "parent":
        return "parent_id回复边"
    if kind == "term":
        return f"词重叠({detail})"
    return "兜底·最近段"


def render_trace(group_id, clusters, window_desc, only=None):
    """逐条打印「每条消息走了哪条规则、为什么挂到那段」。段号=建段顺序(非最终r号)。"""
    L = ["# 子需求切分 · 决策明细(--trace)", "",
         f"- 群：`{group_id}`　窗口：{window_desc}",
         "> 看每条消息：**分类**(fresh/rework/correction/continuation/filler/bot, 含命中词) "
         "→ **决策**(开新段 / 并入哪段) → **理由**(parent_id回复边 / 词重叠 / 兜底)。",
         "> 段#=建段先后序号(纯规则的中间态，未经清理/排序，非最终 r 号)。", "", "---", ""]
    for ti, cl in enumerate(clusters, 1):
        if only and ti != only:
            continue
        tr = []
        decompose_task(cl, trace=tr)
        dom, _, _ = _cluster_label(cl)
        t0 = time.strftime("%m-%d %H:%M", time.localtime(cl[0]["ts"]))
        t1 = time.strftime("%H:%M", time.localtime(cl[-1]["ts"]))
        L += [f"## 任务 {ti}　〔{dom}〕　{t0}→{t1}　（{len(cl)} 条原始）", ""]
        for r in tr:
            e = r["e"]
            when = time.strftime("%H:%M:%S", time.localtime(e["ts"]))
            who = ("👤" + e["name"]) if e["role"] == "user" else ("🤖" + e["name"])
            g = gist(e["text"], 54)
            if r.get("skip"):
                L += [f"- `{when}` 　🚫 跳过（{r['skip']}）｜ {g}"]
                continue
            cls = r["cls"]
            kw = f"·{r['kw']}" if r.get("kw") else ""
            tag = f"[{cls}{kw}]"
            seg = f"段#{r['tgt']}" if r.get("tgt") else "段#?"
            why = _why_str(r.get("why"))
            why = f"（{why}）" if why else ""
            if "开新段" in r["act"]:
                L += [f"- `{when}` {who} **{tag}** → {r['act']} = **{seg}**｜ {g}"]
            else:
                L += [f"- `{when}` {who} {tag} → 并入 **{seg}**{why}｜ {g}"]
        L += ["", "---", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", help="飞书 group_id (oc_...)")
    ap.add_argument("--task", help="task_id 或其后缀(如 '#3')；缺省配合 --list")
    ap.add_argument("--n", type=int, default=120, help="拉取最近多少条消息(默认120)")
    ap.add_argument("--list", action="store_true", help="只列各任务参与模型，不串接")
    ap.add_argument("--out", help="输出 md 路径(缺省落共享区)")
    ap.add_argument("--source", choices=["memory", "history", "pool"], default="memory",
                    help="memory=group-memory流(默认) / history=飞书历史脊柱(v2,带人话+回复链切分) / "
                         "pool=user_corrections_pool 快照")
    ap.add_argument("--since", type=int, help="history源：起始 epoch 秒")
    ap.add_argument("--until", type=int, help="history源：结束 epoch 秒")
    ap.add_argument("--hist-file", help="history源：直接读已导出的 jsonl，免重复拉取")
    ap.add_argument("--decompose", action="store_true",
                    help="history源：再下钻一层，每个任务分解成子需求+关系边+难例标记(路线2筛子)")
    ap.add_argument("--trace", type=int, metavar="N",
                    help="history源：打印子需求切分的逐条决策明细(走哪条规则/为何挂某段)；"
                         "N=只看第N个任务(1基)，N=0看全部")
    ap.add_argument("--pool-file", help="pool源：user_corrections_pool*.jsonl")
    ap.add_argument("--pool-line", type=int, help="pool源：1基行号")
    ap.add_argument("--pool-t0-msg-id", help="pool源：按 t0.msg_id 取记录")
    ap.add_argument("--pool-no-t0", action="store_true",
                    help="pool源：只分析 T0 前历史，不把 T0 用户原指令放入事件流")
    ap.add_argument("--json-out", help="pool源：结构化 JSON 输出路径；缺省打印到 stdout")
    args = ap.parse_args()

    # ── v2：飞书历史脊柱 + 回复链重切 ──
    if args.source == "history":
        if not args.group:
            raise SystemExit("history 源需要 --group")
        if not (args.since and args.until) and not args.hist_file:
            raise SystemExit("history 源需要 --since/--until 或 --hist-file")
        evs = fetch_history(args.group, args.since or 0, args.until or 0, args.hist_file)
        if not evs:
            raise SystemExit("飞书历史没拉到消息")
        clusters = segment_history(evs)
        wd = (time.strftime("%m-%d %H:%M", time.localtime(evs[0]["ts"])) + "→" +
              time.strftime("%m-%d %H:%M", time.localtime(evs[-1]["ts"])) +
              f"（{len(evs)}条原始消息）")
        if args.trace is not None:
            md = render_trace(args.group, clusters, wd, only=args.trace or None)
        else:
            md = (render_decomposed if args.decompose else render_segmented)(
                args.group, clusters, wd)
        SHARED.mkdir(parents=True, exist_ok=True)
        out = (Path(args.out) if args.out else
               SHARED / f"resegment_{args.group}_{time.strftime('%m%d_%H%M%S')}.md")
        out.write_text(md, encoding="utf-8")
        nbot = sum(1 for e in evs if e["role"] == "bot" and not _is_noise(e["text"]))
        nlink = sum(1 for cl in clusters for e in cl
                    if e["role"] == "bot" and not _is_noise(e["text"])
                    and link_trajectory(e).get("linked"))
        term = compute_terminal(clusters)
        print(f"重切：{len(evs)} 条 → {len(clusters)} 个子任务"
              f"（未终结 {sum(1 for t in term if not t)} 个）")
        print(f"挂接：{nbot} 个 bot 回合，挂到轨迹 {nlink} 个")
        print(f"文件：{out}\n查看：{DATAVIEW}")
        return

    # ── vfinal pool 快照：服务当前三腿 replay 的只读适配入口 ──
    if args.source == "pool":
        if not args.pool_file:
            raise SystemExit("pool 源需要 --pool-file")
        rec = load_pool_record(args.pool_file, line_no=args.pool_line,
                               t0_msg_id=args.pool_t0_msg_id)
        analysis = analyze_pool_record(rec, include_t0=not args.pool_no_t0)
        text = json.dumps(analysis, ensure_ascii=False, indent=2)
        if args.json_out:
            out = Path(args.json_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text + "\n", encoding="utf-8")
            print(f"pool快照：{analysis['event_count']} 条 → "
                  f"{analysis['task_count']} 任务 / {analysis['subreq_count']} 子需求")
            print(f"文件：{out}")
        else:
            print(text)
        return

    if not args.group:
        raise SystemExit("memory 源需要 --group")
    evs = fetch_stream(args.group, args.n)
    if not evs:
        raise SystemExit("group-memory 没拉到消息")
    if args.list or not args.task:
        list_tasks(args.group, evs)
        return

    sel = [e for e in evs if e["task_id"] == args.task
           or e["task_id"].endswith(args.task)]
    if not sel:
        raise SystemExit(f"任务 {args.task!r} 在最近 {args.n} 条里没有消息")
    task_id = sel[0]["task_id"]
    md = render(args.group, task_id, sel)

    SHARED.mkdir(parents=True, exist_ok=True)
    safe = task_id.replace("#", "_").replace(":", "_")
    out = Path(args.out) if args.out else SHARED / f"task_{safe}_{time.strftime('%m%d_%H%M%S')}.md"
    out.write_text(md, encoding="utf-8")
    n_link = sum(1 for e in sel if e["role"] != "user"
                 and link_trajectory(e).get("linked"))
    n_bot = sum(1 for e in sel if e["role"] != "user")
    print(f"文件：{out}")
    print(f"查看：{DATAVIEW}")
    print(f"统计：{len(sel)} 事件 / {n_bot} 个 bot 回合，挂到轨迹 {n_link} 个")


if __name__ == "__main__":
    main()
