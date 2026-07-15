#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLM verifier for raw-context replay manifests.

The verifier only checks candidate quality.  It must not recover missing
context, rewrite prompts, or answer the original task.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import replay_three_leg as R  # noqa: E402
from llm_batch_guard import require_llm_batch_confirmation  # noqa: E402


DATA = "/opt/shared/data/task-trajectory"
DEFAULT_IN = f"{DATA}/raw_context_replay_manifest.jsonl"
DEFAULT_OUT = f"{DATA}/raw_context_replay_manifest_verified.jsonl"
MODEL = os.environ.get("RAW_CONTEXT_VERIFY_MODEL", os.environ.get("REPLAY_MODEL", "deepseek-v3.2"))

VERIFY_SYS = """你是轨迹回放数据的质检员。你只判断给定 raw-context replay 样本是否可用于高保真复现。
严禁补充缺失消息、严禁回答原任务、严禁改写用户输入。

判定标准：
- ok=true：T0 是原始用户指令；上下文全在 T0 之前；未包含 bot 错误 B、用户纠正 C、最终正确答案；没有明显跨线混入。
- ok=false：T0 不像原始用户指令，或上下文含 B/C/答案泄漏，或跨线混入严重，或证据不足。
- quality 只能从 exact_thread / same_session_window / recovered_by_parent / missing_or_ambiguous 中选一个。

只输出 JSON：
{"ok": true/false, "quality": "...", "reject_reason": "", "leakage_flags": [], "notes": ""}"""


def _read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(rows: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _compact_event(event: dict) -> dict:
    text = event.get("text")
    if text is None:
        text = event.get("text_preview")
    return {
        "msg_id": event.get("msg_id"),
        "role": event.get("role"),
        "name": event.get("name"),
        "parent_id": event.get("parent_id"),
        "ts": event.get("ts"),
        "text": (text or "")[:500],
        "text_chars": event.get("text_chars"),
    }


def _verification_payload(row: dict) -> dict:
    return {
        "idx": row.get("idx"),
        "script_quality": row.get("raw_context_quality"),
        "script_status": row.get("status"),
        "t0": _compact_event(row.get("t0") or {}),
        "parent_chain": (row.get("raw_context_audit") or {}).get("parent_chain"),
        "script_leakage_flags": (row.get("raw_context_audit") or {}).get("leakage_flags"),
        "context_events": [
            _compact_event(e)
            for e in (
                ((row.get("context") or {}).get("events_preview")
                 or (row.get("context") or {}).get("events")
                 or [])[-30:]
            )
        ],
        "corrections": row.get("corrections") or [],
        "source_vfinal_row": row.get("source_vfinal_row"),
    }


def _parse_json(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return {
            "ok": False,
            "quality": "missing_or_ambiguous",
            "reject_reason": "verifier returned non-json",
            "leakage_flags": ["non_json"],
            "notes": raw[:200],
        }
    try:
        data = json.loads(match.group(0))
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "quality": "missing_or_ambiguous",
            "reject_reason": f"bad json: {exc}",
            "leakage_flags": ["bad_json"],
            "notes": raw[:200],
        }
    quality = data.get("quality")
    if quality not in {
        "exact_thread", "same_session_window", "recovered_by_parent", "missing_or_ambiguous",
    }:
        quality = "missing_or_ambiguous"
    return {
        "ok": bool(data.get("ok")),
        "quality": quality,
        "reject_reason": str(data.get("reject_reason") or "")[:300],
        "leakage_flags": data.get("leakage_flags") if isinstance(data.get("leakage_flags"), list) else [],
        "notes": str(data.get("notes") or "")[:300],
    }


def verify_rows(rows: list[dict], limit: int | None = None, model: str = MODEL) -> list[dict]:
    selected = rows[:limit] if limit is not None else rows
    require_llm_batch_confirmation(
        task="raw_context_verify",
        model=model,
        rows=len(selected),
        repeat=1,
        estimated_calls=len(selected),
        extra="verifies locating/leakage only",
    )
    client = R._client()
    out: list[dict] = []
    for row in rows:
        if limit is not None and len(out) >= limit:
            # Preserve remaining rows unverified when caller intentionally samples.
            rest = dict(row)
            rest["llm_verification"] = {"status": "not_run"}
            out.append(rest)
            continue
        payload = json.dumps(_verification_payload(row), ensure_ascii=False, indent=2)
        raw = R._chat(client, model, VERIFY_SYS, f"请质检这个样本：\n{payload}\n\n只输出 JSON。")
        verified = dict(row)
        verified["llm_verification"] = _parse_json(raw)
        out.append(verified)
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=DEFAULT_IN)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--limit", type=int, default=None, help="Verify only the first N rows.")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = _read_jsonl(args.input)
    out = verify_rows(rows, limit=args.limit, model=args.model)
    _write_jsonl(out, args.out)
    verified = sum(1 for r in out if (r.get("llm_verification") or {}).get("status") != "not_run")
    ok = sum(1 for r in out if (r.get("llm_verification") or {}).get("ok") is True)
    print(f"wrote {args.out} verified={verified} ok={ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
