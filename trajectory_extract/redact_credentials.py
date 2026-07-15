#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""只抹「真实凭证值」的脱敏器（供 raw 轨迹样例外发内部群用）。
原则：轨迹内容/路径/open_id 不动（内部群无需），只把活凭证替换成占位符——
      按①字段名（authorization/token/secret/...）②值形态（t-tenant / sk-key / Bearer）双保险。
用法：python3 redact_credentials.py <in.jsonl> <out.jsonl>
"""
import json, re, sys

SECRET_KEYS = {"authorization", "api_key", "apikey", "token", "access_token",
               "refresh_token", "secret", "app_secret", "password", "passwd",
               "tenant_access_token", "app_access_token"}
VAL_PATS = [
    (re.compile(r"t-[A-Za-z0-9]{25,}"), "<REDACTED_TENANT_TOKEN>"),
    (re.compile(r"(?<![A-Za-z])sk-[A-Za-z0-9][A-Za-z0-9_\-]{5,}"), "<REDACTED_KEY>"),
    (re.compile(r"[Bb]earer\s+[A-Za-z0-9_\-\.]{10,}"), "Bearer <REDACTED>"),
]
n_field = 0
n_val = 0


def redact_str(s):
    global n_val
    for pat, repl in VAL_PATS:
        s, k = pat.subn(repl, s)
        n_val += k
    return s


def walk(o, keyname=None):
    global n_field
    if isinstance(o, dict):
        return {k: walk(v, k) for k, v in o.items()}
    if isinstance(o, list):
        return [walk(x) for x in o]
    if isinstance(o, str):
        if keyname and keyname.lower() in SECRET_KEYS and o.strip():
            n_field += 1
            return "<REDACTED>"
        return redact_str(o)
    return o


def main():
    inp, out = sys.argv[1], sys.argv[2]
    with open(inp, encoding="utf-8") as f, open(out, "w", encoding="utf-8") as w:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            w.write(json.dumps(walk(obj), ensure_ascii=False) + "\n")
    print(f"脱敏完成 → {out}  (字段名命中 {n_field} 处, 值形态命中 {n_val} 处)")


if __name__ == "__main__":
    main()
