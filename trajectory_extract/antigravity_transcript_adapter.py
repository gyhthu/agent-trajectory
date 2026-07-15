#!/usr/bin/env python3
"""antigravity transcript 离线适配器（形态 1，纯只读）。

antigravity（Google language_server / Gemini 订阅）的模型出口是闭源 Go 二进制走
gRPC/TLS，没有可重定向的 base_url，codex 那套「逼回 HTTP 代理」打不上。但
language_server 自己把每个会话落成明文事件流：
    ~/.gemini/antigravity/brain/<conv_id>/.system_generated/logs/transcript.jsonl
本适配器只读这些已落盘文件，转成与 claude :4319 捕获同构的归一化轨迹（OpenAI-
messages 形态），供质量分析/回放。

零侵入：不碰 bridge / language_server / 桌面进程 / .env / systemd，不写回 brain 目录。

每个 assistant 回合都重建出「压缩感知的会话级输入上下文」（per_turn_context，索引表达）：
未压缩段=逐字原文，压缩段=模型当时真看到的那份 conversation_history 摘要。详见
_build_per_turn_context。

固有保真缺口（transcript 本身就没有，非本脚本能补）：
    - 无闭源端叠在最顶上的官方 system prompt（只拿得到 bridge 注入段）
    - 无 token usage / 采样参数（model id / tools schema 由调用方补齐）
    - thinking 是产品化整理文本，未必等于 raw reasoning token 流
    - 无 token id / logprob（第三方后端物理限制）
    - 非 wire 级逐字节（会话级≈，per_call_boundary 永为 False）
这些缺口逐条记进每条轨迹的 `fidelity` 块，对比时一目了然。
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import Counter

BRAIN_DEFAULT = os.path.expanduser("~/.gemini/antigravity/brain")

# transcript 里「模型回合」与「工具结果」的 type 归类。
# READ_URL_CONTENT 实测会出现（read_url_content 工具的结果），早期漏列会被错分到 unknown。
TOOL_RESULT_TYPES = {
    "RUN_COMMAND", "VIEW_FILE", "LIST_DIRECTORY", "GREP_SEARCH",
    "SEARCH_WEB", "CODE_ACTION", "GENERIC", "READ_URL_CONTENT",
}

# bridge 把 systemPrompt 内联进首条消息：<USER_REQUEST>【系统指令】…【用户消息】<真实消息></USER_REQUEST>
# （续接会话则内联进 CONVERSATION_HISTORY）。这两个标记是接入层固定格式，用来把 system / user 切开。
_SYS_HEADER = "【系统指令】"
_USER_MARKER = "【用户消息】"

# antigravity 固定内置工具集的 schema（OpenAI tools 形态）。与本脚本同目录的 JSON 单一事实源，
# 缺失则 tools_schema 保真位保持 False（不硬编码进代码，便于维护）。
_TOOLS_SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "antigravity_tools_schema.json")


def _load_tools_schema():
    try:
        with open(_TOOLS_SCHEMA_PATH, encoding="utf-8") as f:
            return json.load(f).get("tools") or None
    except (OSError, ValueError):
        return None


def _load_scrub():
    """导入 secret-scrub 的 scrub_text（密钥脱敏单一事实源）。找不到则返回 None，
    由调用方 fail-loud 拒绝发布，绝不静默发布未脱敏数据。"""
    here = os.path.dirname(os.path.abspath(__file__))
    # <root>/feishu-plugin/skills/trajectory-collect/scripts → <root>/llm_label_tool/.../secret-scrub/scripts
    scrub_dir = os.path.normpath(os.path.join(
        here, "..", "..", "..", "..", "llm_label_tool", "skills", "secret-scrub", "scripts"))
    if scrub_dir not in sys.path:
        sys.path.insert(0, scrub_dir)
    try:
        from scrub import scrub_text, harvest_secrets  # noqa: E402
        return scrub_text, harvest_secrets
    except ImportError:
        return None


# PII 伪匿名化（--redact-pii 开启）：保持同一标识在数据集内一致（同人→同代号），不破坏多轮身份信号。
_PII_PATTERNS = [
    ("OPENID", re.compile(r"ou_[0-9a-f]{20,}")),
    ("APPID", re.compile(r"cli_[0-9a-z]{12,}")),
    ("IP", re.compile(r"\b(?:46\.225\.0\.9|8\.218\.177\.165)\b")),
]


def _redact_pii(text):
    """把 open_id/app_id/内部 IP 替换成稳定代号（同值→同代号）。返回 (文本, 命中数)。"""
    import hashlib
    n = 0
    for tag, pat in _PII_PATTERNS:
        def _r(m, _tag=tag):
            nonlocal n
            n += 1
            h = hashlib.sha1(m.group(0).encode()).hexdigest()[:8]
            return f"<{_tag}_{h}>"
        text = pat.sub(_r, text)
    return text, n


def _iter_strings(o):
    """递归取出结构里的所有字符串（用于第一遍 harvest 密钥值）。"""
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from _iter_strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from _iter_strings(v)


def _scrub_record(rec, scrub_text, redact_pii, known_secrets=None):
    """对一条轨迹记录里的所有字符串字段做脱敏（递归）。原地返回新结构 + 累计命中。
    known_secrets：跨记录收集到的密钥明文值，做精确二次替换（清掉散文里的裸引用）。"""
    from collections import Counter
    sec = Counter()
    pii = [0]

    def _walk(o):
        if isinstance(o, str):
            s, c = scrub_text(o, extra_values=known_secrets)
            sec.update(c)
            if redact_pii:
                s, k = _redact_pii(s)
                pii[0] += k
            return s
        if isinstance(o, dict):
            return {k: _walk(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_walk(v) for v in o]
        return o

    # tools schema 是固定模板、无密钥，跳过以省时（且别误伤 schema 里的字段名）
    tools = rec.get("tools")
    rec2 = _walk({k: v for k, v in rec.items() if k != "tools"})
    if tools is not None:
        rec2["tools"] = tools
    return rec2, sec, pii[0]


def _split_user_request(content):
    """拆 antigravity 注入式 user 消息 content。
    返回 (system_prompt 或 None, 真实用户文本, metadata 或 None)。
    bridge 内联格式见上方常量注释；无注入时 system=None、user=去壳后的正文。
    """
    content = content or ""
    meta = None
    m = re.search(r"<ADDITIONAL_METADATA>(.*?)</ADDITIONAL_METADATA>", content, re.S)
    if m:
        meta = m.group(1).strip()
    m = re.search(r"<USER_REQUEST>(.*?)</USER_REQUEST>", content, re.S)
    body = (m.group(1) if m else content).strip()
    system = None
    if _SYS_HEADER in body:
        idx = body.find(_USER_MARKER)
        if idx != -1:
            system, body = body[:idx].strip(), body[idx + len(_USER_MARKER):].strip()
        else:  # 有系统指令但无【用户消息】分隔（罕见）：整块当 system
            system, body = body.strip(), ""
    return system, body, meta


def _clean_arg(v):
    """tool_call 的 args 值是 JSON 编码过的字符串（如 '\"cat x\"' / '500'），尽量解一层。"""
    if isinstance(v, str):
        try:
            return json.loads(v)
        except (ValueError, TypeError):
            return v
    return v


def _norm_tool_calls(tool_calls):
    out = []
    for tc in tool_calls or []:
        if not isinstance(tc, dict):
            continue
        args = {k: _clean_arg(v) for k, v in (tc.get("args") or {}).items()}
        out.append({"name": tc.get("name"), "args": args})
    return out


def _build_per_turn_context(messages):
    """为每个 assistant 回合重建「模型当时真看到的会话级输入上下文」，压缩感知。

    antigravity 把上下文压缩落成带位置的 `conversation_history`(CH) 事件：走到一个 CH，
    它就把此前历史**替换**成这份摘要。所以任一回合的真实输入 =
        最近一次 CH 摘要  +  自该 CH 之后本段的原始消息
    （CH 之前的原始历史已被摘要取代，不在模型视野内）。无 CH 在前时即全量原始前缀。

    系统指令无需特殊处理：bridge 在压缩时把 systemPrompt 重新内联进 CONVERSATION_HISTORY
    摘要（见上方常量注释与下方切分逻辑），故 system 本就随「最近摘要」一并带过来；中途的
    `system_message` 注入是会话内消息，和普通回合一样会被后续压缩取代，不应跨段拉过来。

    返回：
      - 给每条 message 原地打 `seg`（压缩段号，遇 CH +1）；
      - per_turn_context：每个回合一项 {assistant_index, context_indices}，
        context_indices 是指向 messages[] 的**索引列表**（不复制内容：单一事实源、
        避免长会话 O(n²) 膨胀）。下游按索引取 messages[] 即得该回合的精确输入。

    诚实边界：这是「会话级」重建（对话内容对得上），非 wire 级逐字节；且不含闭源端
    叠在最顶上的官方 system（transcript 从不落）。故 per_call_boundary 仍为 False。
    """
    seg = 0
    ch_indices = []      # 所有 conversation_history 的位置（压缩边界）
    for i, m in enumerate(messages):
        if m.get("subtype") == "conversation_history":
            seg += 1
            ch_indices.append(i)
        m["seg"] = seg

    per_turn = []
    for i, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        latest_ch = None
        for c in ch_indices:
            if c < i:
                latest_ch = c
            else:
                break
        lo = (latest_ch + 1) if latest_ch is not None else 0
        ctx = [latest_ch] if latest_ch is not None else []   # 最近一次摘要（含被压进去的 system）
        ctx += [j for j in range(lo, i)                       # 该段内的原始消息
                if messages[j].get("subtype") != "conversation_history"]
        per_turn.append({"assistant_index": i, "context_indices": ctx})
    return per_turn


def adapt_transcript(path, tools_schema=None, model=None):
    """把单个 transcript.jsonl 转成一条归一化轨迹记录。纯读。
    tools_schema/model 由调用方传入（固定工具集 + 推定 model id），用于补齐保真缺口。"""
    conv_id = path.split(os.sep + "brain" + os.sep, 1)[-1].split(os.sep, 1)[0]
    events = []
    bad = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except ValueError:
                bad += 1
    if not events:
        return None

    type_counts = Counter(e.get("type") for e in events)
    messages = []
    # 工具结果按出现顺序排队，回填给最近一次 assistant 的 tool_calls。
    pending_tool_names = []  # 上一条 PLANNER_RESPONSE 还没配到结果的 tool 调用名
    captured_system = False  # 是否已从注入块切出 system prompt（只取首次）

    for e in events:
        etype = e.get("type")
        if etype == "USER_INPUT":
            system, user_text, meta = _split_user_request(e.get("content", ""))
            if system and not captured_system:
                messages.append({"role": "system", "subtype": "injected_system_prompt",
                                 "content": system})
                captured_system = True
            msg = {"role": "user", "content": user_text}
            if meta:
                msg["metadata"] = meta
            messages.append(msg)
        elif etype == "CONVERSATION_HISTORY":
            # 续接会话的 systemPrompt 内联在历史块里，按需切出（仅首次）。
            system, _, _ = _split_user_request(e.get("content", ""))
            if system and not captured_system:
                messages.append({"role": "system", "subtype": "injected_system_prompt",
                                 "content": system})
                captured_system = True
            messages.append({"role": "system", "subtype": "conversation_history",
                             "content": e.get("content", "")})
        elif etype == "SYSTEM_MESSAGE":
            messages.append({"role": "system", "subtype": "system_message",
                             "content": e.get("content", "")})
        elif etype == "PLANNER_RESPONSE":
            tcs = _norm_tool_calls(e.get("tool_calls"))
            msg = {"role": "assistant"}
            if e.get("thinking"):
                msg["reasoning"] = e["thinking"]
            if e.get("content"):
                msg["content"] = e["content"]
            if tcs:
                msg["tool_calls"] = tcs
            messages.append(msg)
            pending_tool_names = [tc.get("name") for tc in tcs]
        elif etype in TOOL_RESULT_TYPES:
            name = pending_tool_names.pop(0) if pending_tool_names else None
            messages.append({"role": "tool", "name": name,
                             "content": e.get("content", "")})
        elif etype == "ERROR_MESSAGE":
            messages.append({"role": "system", "subtype": "error",
                             "content": e.get("error", "")})
        else:
            messages.append({"role": "system", "subtype": etype or "unknown",
                             "content": e.get("content", "")})

    # 压缩感知：给每条 message 打 seg，并重建每个回合的精确输入上下文（索引表达）。
    per_turn_context = _build_per_turn_context(messages)
    n_compressions = sum(1 for m in messages
                         if m.get("subtype") == "conversation_history")

    times = [e.get("created_at") for e in events if e.get("created_at")]
    return {
        "source": "antigravity-transcript",
        "conv_id": conv_id,
        "model": model,                # 推定 model id（调用方传入；transcript 本身不记）
        "created_at": times[0] if times else None,
        "ended_at": times[-1] if times else None,
        "n_events": len(events),
        "n_bad_lines": bad,
        "type_counts": dict(type_counts),
        "n_model_turns": type_counts.get("PLANNER_RESPONSE", 0),
        "n_compressions": n_compressions,   # conversation_history 事件数（压缩次数）
        "messages": messages,               # 每条带 seg（压缩段号）
        # 每个 assistant 回合的压缩感知输入上下文，以索引指向 messages[]（见 _build_per_turn_context）。
        "per_turn_context": per_turn_context,
        "tools": tools_schema,         # 固定内置工具集 schema（调用方传入）
        # 逐条记保真：True=拿到了，False=transcript 固有缺失。
        # system_prompt / tools_schema / model_id 由「离线重建」补齐（见上）——
        # 与 claude api 劫持的差距收敛到仅剩 usage_tokens（usage_tokens/sampling/logprob 物理拿不到）。
        "fidelity": {
            "user_prompt": True,
            "assistant_text": True,
            "reasoning_text": True,        # 有 thinking（产品化文本）
            "tool_call_name_args": True,
            "tool_result": True,
            "timing": True,
            "system_prompt": captured_system,   # ✅ 从接入层注入块切出（切到才置 True）
            "tools_schema": bool(tools_schema),  # ✅ 固定工具集 schema 补齐
            "model_id": bool(model),             # ✅ 推定补齐（非 transcript 原生）
            "usage_tokens": False,         # ❌ 无 input/output/cache token 计数（只有 wire 抓得到）
            "sampling_params": False,      # ❌ 无 temperature/top_p（claude 劫持同样没有，源头不发）
            "raw_reasoning_tokens": False,  # ⚠️ thinking 是整理文本非 raw token 流
            "token_id_logprob": False,     # ❌ 第三方后端物理拿不到（claude 也没有）
            # ✅ 压缩感知重建：每个回合都有「粘性 system+最近摘要+本段原始消息」的精确输入
            # （见 per_turn_context）。未压缩段=逐字原文，压缩段=模型当时真看的那份摘要。
            "per_turn_context_reconstructed": True,
            "per_call_boundary": False,    # ❌ 仍是会话级≈非 wire 级==；不含闭源端官方 system
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default=BRAIN_DEFAULT, help="brain 目录")
    ap.add_argument("--out", default=None, help="输出 jsonl 路径（不给则只打印统计）")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个会话（0=全部）")
    ap.add_argument("--model", default=None,
                    help="推定的 model id（如 gemini-3-pro），写进每条轨迹的 model 字段")
    ap.add_argument("--no-tools", action="store_true",
                    help="不挂固定工具集 schema（默认挂 antigravity_tools_schema.json）")
    ap.add_argument("--publish", default=None, metavar="FILE",
                    help="脱敏后才落地的发布路径（强制过 secret-scrub）。"
                         "用于发到共享区/喂训练——绝不写未脱敏数据。")
    ap.add_argument("--redact-pii", action="store_true",
                    help="同时把 open_id/app_id/内部 IP 伪匿名化（同值→同稳定代号）")
    args = ap.parse_args()

    if args.out and args.publish:
        print("--out（原始）与 --publish（脱敏）互斥，二选一。", file=sys.stderr)
        return 2

    tools_schema = None if args.no_tools else _load_tools_schema()

    scrub_text = harvest_secrets = None
    if args.publish:
        loaded = _load_scrub()
        if loaded is None:  # fail-loud：找不到脱敏器就不发布，绝不静默泄漏
            print("❌ 无法导入 secret-scrub 的 scrub_text，拒绝发布（防泄漏）。\n"
                  "   请确认 llm_label_tool/skills/secret-scrub/scripts/scrub.py 在仓内。",
                  file=sys.stderr)
            return 3
        scrub_text, harvest_secrets = loaded
        if os.sep in args.publish:
            os.makedirs(os.path.dirname(os.path.abspath(args.publish)), exist_ok=True)

    paths = sorted(glob.glob(os.path.join(
        args.brain, "*", ".system_generated", "logs", "transcript.jsonl")))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        print(f"未找到 transcript：{args.brain}", file=sys.stderr)
        return 1

    write_path = args.out or args.publish
    n_conv = n_msg = n_turns = empty = n_with_system = n_compressed = 0
    role_counts = Counter()
    sec_total = Counter()
    pii_total = 0

    # 先把所有记录转出来（发布模式要先 harvest 全量密钥值再两遍扫；非发布模式也一并收集，量小）
    recs = []
    for p in paths:
        rec = adapt_transcript(p, tools_schema=tools_schema, model=args.model)
        if not rec:
            empty += 1
            continue
        recs.append(rec)
        n_conv += 1
        n_msg += len(rec["messages"])
        n_turns += rec["n_model_turns"]
        if rec["fidelity"]["system_prompt"]:
            n_with_system += 1
        if rec.get("n_compressions"):
            n_compressed += 1
        for m in rec["messages"]:
            role_counts[m["role"]] += 1

    # 发布前第一遍：跨全量记录 harvest 密钥明文值（内存保管，绝不落盘），供二次精确替换
    known_secrets = set()
    if args.publish:
        for rec in recs:
            for s in _iter_strings({k: v for k, v in rec.items() if k != "tools"}):
                known_secrets |= harvest_secrets(s)

    out_f = open(write_path, "w", encoding="utf-8") if write_path else None
    for rec in recs:
        if args.publish:
            rec, sec, pii = _scrub_record(rec, scrub_text, args.redact_pii,
                                          known_secrets=known_secrets)
            sec_total.update(sec)
            pii_total += pii
        if out_f:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if out_f:
        out_f.close()
        if args.publish:  # 收紧权限（脱敏后产物仍属内部数据）
            os.chmod(write_path, 0o600)

    print(f"扫描 transcript 文件: {len(paths)}")
    print(f"成功转换会话:        {n_conv}（空文件 {empty}）")
    print(f"归一化消息总数:      {n_msg}")
    print(f"模型回合总数:        {n_turns}")
    print(f"切出 system prompt:  {n_with_system}/{n_conv}")
    print(f"含压缩(CH)的会话:    {n_compressed}/{n_conv}（per_turn_context 已按压缩边界重建）")
    print(f"挂载 tools schema:   {'是（'+str(len(tools_schema))+' 个工具）' if tools_schema else '否'}")
    print(f"model id:            {args.model or '（未指定 --model）'}")
    print(f"消息 role 分布:      {dict(role_counts)}")
    if args.publish:
        print(f"\n🔒 已脱敏发布: {write_path}（权限 600）")
        print(f"   密钥命中并打码: {dict(sec_total) if sec_total else '无'}")
        print(f"   PII 伪匿名化:   {pii_total if args.redact_pii else '（未开 --redact-pii）'}")
    elif args.out:
        print(f"\n⚠️  原始输出（未脱敏，含真实密钥/PII）: {args.out}")
        print("    切勿发到共享区或喂训练；要发布请改用 --publish。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
