#!/usr/bin/env python3
"""codex wire 捕获适配器（把 codex_capture_proxy 落盘的 wire 记录转成归一化轨迹）。

codex（ChatGPT OAuth / gpt-5.5）默认走 WebSocket(wss://chatgpt.com)，无可重定向 base_url；
codex_capture_proxy 用 model_provider=proxycap + supports_websockets=false 逼它退回
HTTP POST /backend-api/codex/responses，逐字节抓真实 wire 请求/响应，落盘：
    ~/trajectory-data/codex-api-calls/<date>.jsonl.gz
每行一条 /responses API call（顶层键 ts/path/status/had_auth/request/response/response_raw）。

本适配器只读这些已落盘文件，把每条 call 转成与 antigravity / claude 同构的归一化轨迹
（OpenAI-messages 形态），字段命名对齐 antigravity_transcript_adapter 便于混用。

与 antigravity 的本质区别：
    antigravity：1 个 transcript = 1 会话事件流 → 1 条会话级记录（per_call_boundary=False）。
    codex：1 个 .jsonl.gz = 多条独立 /responses call，**每条本身就是 wire 级逐字节**
           （per_call_boundary=True）。每条 call = 一个 (input 上下文 → output 补全) 样本。

codex wire 的保真度（逐条记进 fidelity，对比一目了然）：
    ✅ per_call_boundary  : wire 真字节（强于 antigravity 会话级≈）
    ✅ system_prompt      : request.instructions 官方全文（"You are Codex…"）
    ✅ tools_schema       : request.tools 官方 19 条完整定义
    ✅ usage_tokens       : final_response.usage（含 cached_tokens / reasoning_tokens 分列，比 claude 干净）
    ✅ sampling_params    : final_response.temperature / top_p（服务端响应里回显，request 里是 None）
    ❌ raw_reasoning_tokens: reasoning item 是 OpenAI 加密 blob（summary=[]、encrypted_content），
                            client 解不开 → 思考文本物理拿不到。剥块时记 n_reasoning_stripped。
    ❌ token_id_logprob   : output_text.logprobs 为空 []（codex 不下发）

零侵入：只读 trajectory-data，不碰 bridge / lian-codex / .env / proxy。
用法：
    python3 codex_wire_adapter.py --input <file_or_dir> --out  out.jsonl      # 原始（未脱敏）
    python3 codex_wire_adapter.py --input <file_or_dir> --publish pub.jsonl   # 脱敏发布（chmod 600）
    --redact-pii 额外把 open_id/app_id/内部IP 伪匿名化（同值→同稳定代号）
"""
import argparse
import glob
import gzip
import json
import os
import re
import sys
from collections import Counter

DATA_DEFAULT = os.path.expanduser("~/trajectory-data/codex-api-calls")


# ---- 脱敏（与 antigravity adapter 同一套单一事实源，复用 secret-scrub） ----
def _load_scrub():
    """导入 secret-scrub 的 scrub_text / harvest_secrets。找不到返回 None，
    由调用方 fail-loud 拒绝发布，绝不静默发布未脱敏数据。"""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.environ.get("SECRET_SCRUB_SCRIPTS_DIR"),
        os.path.normpath(os.path.join(
            here, "..", "..", "..", "..", "llm_label_tool", "skills", "secret-scrub", "scripts")),
        "/home/agent/lian-codex/skills/llm_label_tool/skills/secret-scrub/scripts",
        "/home/agent/lian-server-bot/skills/llm_label_tool/skills/secret-scrub/scripts",
    ]
    for scrub_dir in candidates:
        if not scrub_dir or not os.path.exists(os.path.join(scrub_dir, "scrub.py")):
            continue
        if scrub_dir not in sys.path:
            sys.path.insert(0, scrub_dir)
        try:
            from scrub import scrub_text, harvest_secrets  # noqa: E402
            return scrub_text, harvest_secrets
        except ImportError:
            continue
    return None


# PII 伪匿名化（--redact-pii）：同值→同稳定代号，不破坏多轮身份信号。
_PII_PATTERNS = [
    ("OPENID", re.compile(r"ou_[0-9a-f]{20,}")),
    ("APPID", re.compile(r"cli_[0-9a-z]{12,}")),
    ("IP", re.compile(r"\b(?:46\.225\.0\.9|8\.218\.177\.165)\b")),
]


def _redact_pii(text):
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
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from _iter_strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from _iter_strings(v)


def _scrub_record(rec, scrub_text, redact_pii, known_secrets=None):
    """递归脱敏一条记录里的所有字符串，跳过 tools（固定模板无密钥，且别误伤 schema 字段名）。"""
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

    tools = rec.get("tools")
    rec2 = _walk({k: v for k, v in rec.items() if k != "tools"})
    if tools is not None:
        rec2["tools"] = tools
    return rec2, sec, pii[0]


# ---- 轨迹重建 ----
def _extract_text(content):
    """从 message 的 content 取纯文本。content 可能是 str，或 [{type:input_text/output_text/..., text}]。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                if isinstance(p.get("text"), str):
                    parts.append(p["text"])
                elif p.get("type") in ("input_text", "output_text") and isinstance(p.get("content"), str):
                    parts.append(p["content"])
            elif isinstance(p, str):
                parts.append(p)
        return "\n".join(parts)
    return ""


def _clean_args(arguments):
    """function_call.arguments 是 JSON 字符串，尽量解成对象（对齐 antigravity 的 args 形态）。"""
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except (ValueError, TypeError):
            return arguments
    return arguments


def _item_to_messages(it, stripped_counter):
    """把一个 /responses item（input 或 output 里的）转成 0~1 条归一化 message。
    type=reasoning 一律剥掉（加密 blob 不可读），只累加 stripped 计数。"""
    if not isinstance(it, dict):
        return []
    t = it.get("type")
    if t == "reasoning":
        stripped_counter[0] += 1
        return []
    if t == "message":
        role = it.get("role") or "assistant"
        return [{"role": role, "content": _extract_text(it.get("content"))}]
    if t == "function_call":
        return [{
            "role": "assistant",
            "tool_calls": [{
                "name": it.get("name"),
                "args": _clean_args(it.get("arguments")),
                "call_id": it.get("call_id") or it.get("id"),
            }],
        }]
    if t == "function_call_output":
        out = it.get("output")
        return [{
            "role": "tool",
            "call_id": it.get("call_id"),
            "content": _extract_text(out) if not isinstance(out, str) else out,
        }]
    # 其它未知类型：保留为 system 杂项，便于排查（不静默丢）
    return [{"role": "system", "subtype": t or "unknown",
             "content": _extract_text(it.get("content"))}]


def adapt_call(rec, source_file=None):
    """把一条 codex wire /responses call 转成一条归一化轨迹记录。纯读。
    messages = [system(instructions)] + input上下文 + output补全；
    completion_start_index 指向 messages[] 里本轮补全（output）的起点，给出干净的
    (context → completion) SFT 切分（这正是 per_call_boundary=True 的 wire 真边界）。"""
    req = rec.get("request") or {}
    resp = rec.get("response") or {}
    fr = resp.get("final_response") if isinstance(resp, dict) else None
    fr = fr if isinstance(fr, dict) else {}

    stripped = [0]
    messages = []

    # 1) 官方 system（base_instructions 全文）
    instr = req.get("instructions")
    if isinstance(instr, str) and instr:
        messages.append({"role": "system", "subtype": "base_instructions", "content": instr})

    # 2) input 上下文（对话历史，含 developer/user/assistant/工具）
    for it in req.get("input", []) or []:
        messages.extend(_item_to_messages(it, stripped))

    # 3) output 补全（本轮模型产出：剥掉加密 reasoning，留 assistant 文本/工具调用）
    completion_start = len(messages)
    output_items = fr.get("output") or []
    for it in output_items:
        messages.extend(_item_to_messages(it, stripped))

    usage = fr.get("usage")
    model = fr.get("model") or req.get("model")
    temperature = fr.get("temperature")
    top_p = fr.get("top_p")

    return {
        "source": "codex-wire",
        "source_file": os.path.basename(source_file) if source_file else None,
        "ts": rec.get("ts"),
        "path": rec.get("path"),
        "status": rec.get("status"),
        "model": model,
        "messages": messages,
        # 本轮补全在 messages[] 里的起点：messages[:completion_start_index]=输入上下文，
        # messages[completion_start_index:]=该 call 的模型补全（SFT target）。wire 真边界。
        "completion_start_index": completion_start,
        "tools": req.get("tools"),
        "usage": usage,
        "sampling": {"temperature": temperature, "top_p": top_p,
                     "reasoning_effort": (req.get("reasoning") or {}).get("effort")},
        "n_reasoning_stripped": stripped[0],   # 剥掉的加密 reasoning 块数（不可读，故剥）
        "fidelity": {
            "user_prompt": True,
            "assistant_text": True,
            "tool_call_name_args": True,
            "tool_result": True,
            "system_prompt": bool(instr),            # ✅ request.instructions 官方全文
            "tools_schema": bool(req.get("tools")),  # ✅ 官方完整 schema
            "model_id": bool(model),
            "usage_tokens": usage is not None,        # ✅ 含 cached/reasoning 分列，比 claude 干净
            "sampling_params": temperature is not None or top_p is not None,  # ✅ final_response 回显
            "reasoning_text": False,        # ❌ reasoning 是加密 blob，summary 空，文本拿不到
            "raw_reasoning_tokens": False,  # ❌ 同上（OpenAI 服务端对 client 上锁）
            "token_id_logprob": False,      # ❌ output_text.logprobs 为空 []
            "per_call_boundary": True,      # ✅ wire 逐字节真边界（强于 antigravity 会话级≈）
        },
    }


def iter_records(path):
    """逐条读一个 .jsonl.gz / .jsonl 捕获文件。"""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError:
                continue


def _gather_files(input_path):
    if os.path.isdir(input_path):
        return sorted(glob.glob(os.path.join(input_path, "*.jsonl.gz")) +
                      glob.glob(os.path.join(input_path, "*.jsonl")))
    return [input_path] if os.path.exists(input_path) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DATA_DEFAULT, help="捕获文件或目录（默认 ~/trajectory-data/codex-api-calls）")
    ap.add_argument("--out", default=None, help="原始输出 jsonl（未脱敏，含真实密钥/PII）")
    ap.add_argument("--publish", default=None, metavar="FILE",
                    help="脱敏后才落地的发布路径（强制过 secret-scrub，chmod 600）")
    ap.add_argument("--redact-pii", action="store_true",
                    help="额外把 open_id/app_id/内部IP 伪匿名化（同值→同稳定代号）")
    args = ap.parse_args()

    if args.out and args.publish:
        print("--out（原始）与 --publish（脱敏）互斥，二选一。", file=sys.stderr)
        return 2

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

    files = _gather_files(args.input)
    if not files:
        print(f"未找到捕获文件：{args.input}", file=sys.stderr)
        return 1

    recs = []
    n_calls = bad = 0
    role_counts = Counter()
    n_stripped_total = 0
    for fp in files:
        for raw in iter_records(fp):
            try:
                rec = adapt_call(raw, source_file=fp)
            except (KeyError, TypeError, AttributeError):
                bad += 1
                continue
            recs.append(rec)
            n_calls += 1
            n_stripped_total += rec["n_reasoning_stripped"]
            for m in rec["messages"]:
                role_counts[m["role"]] += 1

    # 发布前：跨全量 harvest 密钥明文值（内存保管，绝不落盘），供二次精确替换
    known_secrets = set()
    if args.publish:
        for rec in recs:
            for s in _iter_strings({k: v for k, v in rec.items() if k != "tools"}):
                known_secrets |= harvest_secrets(s)

    write_path = args.out or args.publish
    sec_total = Counter()
    pii_total = 0
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
        if args.publish:
            os.chmod(write_path, 0o600)

    print(f"扫描捕获文件:        {len(files)}")
    print(f"转换 /responses call:{n_calls}（坏行 {bad}）")
    print(f"剥掉加密 reasoning:  {n_stripped_total} 块（不可读，故剥）")
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
