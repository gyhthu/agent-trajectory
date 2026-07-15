#!/usr/bin/env python3
"""扫描 agent 轨迹 SFT jsonl 的 token 长度分布。

用 Qwen3 chat template(含 tools 定义)精确算每条样本的训练输入 token 数，
输出长度分布/分位/各 max_length 阈值超长占比，并拆出 tools 固定开销、
标注疑似截断(末轮非 assistant)等合理性信号。

用法: python3 scan_token_len.py <input.jsonl> [out.json]
环境: TOK_MODEL 覆盖 tokenizer(默认 Qwen/Qwen3-8B, 与 Qwen3.x 同族 151k 词表)
"""
import json, os, sys, statistics as st
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoTokenizer
from modelscope import snapshot_download

F = sys.argv[1] if len(sys.argv) > 1 else "/opt/shared/data/zym-antigravity-sft.jsonl"
OUT = sys.argv[2] if len(sys.argv) > 2 else F.rsplit(".", 1)[0] + ".lenscan.json"
MODEL = os.environ.get("TOK_MODEL", "Qwen/Qwen3-8B")

p = snapshot_download(MODEL, allow_patterns=["tokenizer*", "vocab*", "merges*", "*.json"])
tok = AutoTokenizer.from_pretrained(p, trust_remote_code=True)

rows = [json.loads(l) for l in open(F) if l.strip()]
VALID = {"system", "user", "assistant", "tool"}


def ntok(msgs, tools):
    # 注意: transformers 5.x 的 apply_chat_template(tokenize=True) 返回 BatchEncoding(dict),
    # len() 数的是 key 数而非 token 数。统一走"渲染文本 -> encode"，结果稳定不受版本影响。
    txt = tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=False)
    return len(tok(txt, add_special_tokens=False).input_ids)


def clean(msgs):
    out = []
    for m in msgs:
        role, c = m.get("role"), m.get("content")
        if isinstance(c, list):
            c = "\n".join((b.get("text") or b.get("content") or json.dumps(b, ensure_ascii=False))
                          if isinstance(b, dict) else str(b) for b in c)
        elif not isinstance(c, str):
            c = json.dumps(c, ensure_ascii=False) if c is not None else ""
        if role not in VALID:
            role = "assistant" if role in ("model", "ai") else ("tool" if "tool" in str(role) else "user")
        out.append({"role": role, "content": c})
    return out


per = []
for r in rows:
    msgs = clean(r["messages"])
    tools = r.get("tools") or None
    full = ntok(msgs, tools)
    notool = ntok(msgs, None)
    per.append(dict(conv_id=r.get("conv_id"), model=r.get("model"),
                    tokens=full, content_tokens=notool, tools_overhead=full - notool,
                    n_turns=r.get("n_model_turns"), n_tools=len(r.get("tools") or []),
                    n_events=r.get("n_events"), n_bad_lines=r.get("n_bad_lines"),
                    last_role=(r["messages"][-1].get("role") if r["messages"] else None)))

lengths = sorted(x["tokens"] for x in per)


def pct(q):
    k = (len(lengths) - 1) * q / 100
    f = int(k); c = min(f + 1, len(lengths) - 1)
    return int(lengths[f] + (lengths[c] - lengths[f]) * (k - f))


THRESH = [2048, 4096, 8192, 16384, 32768, 40960, 65536]
over = {t: sum(1 for x in lengths if x > t) for t in THRESH}
ov = [x["tools_overhead"] for x in per]

summary = {
    "n_samples": len(lengths),
    "tokenizer": f"{MODEL} (Qwen3 family, 151k vocab)",
    "min": lengths[0], "max": lengths[-1],
    "mean": round(st.mean(lengths), 1), "median": st.median(lengths),
    "p50": pct(50), "p75": pct(75), "p90": pct(90), "p95": pct(95), "p99": pct(99),
    "total_tokens": sum(lengths),
    "over_threshold_count": over,
    "over_threshold_pct": {t: round(over[t] / len(lengths) * 100, 1) for t in THRESH},
    "tools_overhead_min": min(ov), "tools_overhead_max": max(ov),
    "tools_overhead_median": int(st.median(ov)),
    "suspect_truncated": [x["conv_id"] for x in per if x["last_role"] != "assistant"],
    "per_sample": sorted(per, key=lambda x: -x["tokens"]),
}
json.dump(summary, open(OUT, "w"), ensure_ascii=False, indent=2)

# 纯 ASCII 打印, 规避终端对中文抽风
s = summary
print("n=%d  tokenizer=%s" % (s["n_samples"], MODEL))
print("min=%d max=%d mean=%.0f median=%d" % (s["min"], s["max"], s["mean"], s["median"]))
print("p50=%d p75=%d p90=%d p95=%d p99=%d total=%d" % (s["p50"], s["p75"], s["p90"], s["p95"], s["p99"], s["total_tokens"]))
print("tools_overhead min/med/max = %d/%d/%d" % (s["tools_overhead_min"], s["tools_overhead_median"], s["tools_overhead_max"]))
print("suspect_truncated(last_role!=assistant): %d -> %s" % (len(s["suspect_truncated"]), s["suspect_truncated"]))
print("-- over threshold count/pct --")
for t in THRESH:
    print(">%-6d %2d %.1f%%" % (t, over[t], s["over_threshold_pct"][t]))
print("-- per sample: full content tools_ovh turns tools last --")
for x in s["per_sample"]:
    print("%6d %6d %5d  t%-3s x%-2d %s" % (x["tokens"], x["content_tokens"], x["tools_overhead"],
                                           x["n_turns"], x["n_tools"], x["last_role"]))
print("WROTE", OUT)
