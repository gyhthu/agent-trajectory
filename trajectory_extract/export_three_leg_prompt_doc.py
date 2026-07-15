#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export baseline/treatment/placebo replay prompts from snapshot JSONL to Markdown."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


BASE_SYSTEM = "你是一个严谨的 AI 助手。"
PLACEBO_RULE = "回复时注意排版清晰、分点表达，让阅读者一眼看到重点。"


def _load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _fence(text: str) -> str:
    return "```text\n" + (text or "").replace("```", "` ` `") + "\n```"


def _placebo_prompt(row: dict) -> str:
    legs = row.get("legs") or {}
    if legs.get("placebo_scene"):
        return legs["placebo_scene"]
    if legs.get("placebo_user"):
        return legs["placebo_user"]
    prompts = row.get("prompts") or {}
    baseline = prompts.get("baseline") or ""
    marker = "\n\n【原始指令】"
    if marker not in baseline:
        return baseline + f"\n\n【通用行为准则（placebo，不含纠错信号）】\n{PLACEBO_RULE}"
    head, tail = baseline.split(marker, 1)
    return f"{head}\n\n【通用行为准则（placebo，不含纠错信号）】\n{PLACEBO_RULE}{marker}{tail}"


def _treatment_prompt(row: dict) -> str:
    legs = row.get("legs") or {}
    if legs.get("treatment_scene"):
        return legs["treatment_scene"]
    if legs.get("treatment_user"):
        return legs["treatment_user"]
    prompts = row.get("prompts") or {}
    baseline = prompts.get("baseline") or ""
    hint = row.get("comment_for_replay") or ""
    if not hint:
        return baseline
    marker = "\n\n【原始指令】"
    if marker not in baseline:
        return baseline + f"\n\n【通用行为准则（进 user，不是本题答案）】\n{hint}"
    head, tail = baseline.split(marker, 1)
    return f"{head}\n\n【通用行为准则（进 user，不是本题答案）】\n{hint}{marker}{tail}"


def _merge_by_t0(rows: list[dict]) -> list[dict]:
    grouped: dict[str, dict] = {}
    for row in rows:
        t0id = ((row.get("t0") or {}).get("msg_id")) or row.get("instruction_to_fix_msg") or ""
        if not t0id:
            t0id = f"_missing_{len(grouped)}"
        if t0id not in grouped:
            merged = dict(row)
            merged["corrections"] = list(row.get("corrections") or [])
            merged["_source_rows"] = [row]
            grouped[t0id] = merged
            continue
        merged = grouped[t0id]
        merged["_source_rows"].append(row)
        merged["corrections"].extend(row.get("corrections") or [])
        if merged.get("legs") and row.get("legs"):
            # Raw-context manifests should be pre-merged before export. Keep the
            # first generated legs deterministic and only merge correction audit.
            pass
        hints = [
            h for h in (merged.get("comment_for_replay"), row.get("comment_for_replay"))
            if h
        ]
        seen: set[str] = set()
        merged_hints = []
        for hint in ";".join(hints).replace("；", ";").split(";"):
            hint = hint.strip()
            if hint and hint not in seen:
                seen.add(hint)
                merged_hints.append(hint)
        merged["comment_for_replay"] = "；".join(merged_hints)
    return list(grouped.values())


def _row_summary(row: dict, index: int) -> str:
    t0 = row.get("t0") or {}
    corrections = row.get("corrections") or []
    prompts = row.get("prompts") or {}
    legs = row.get("legs") or {}
    bc_lines = []
    for n, correction in enumerate(corrections, 1):
        bc_lines.append(
            f"  {n}. B=`{correction.get('anchor') or ''}` C=`{correction.get('corrector') or ''}` "
            f"what={correction.get('what') or ''}"
        )
    parts = [
        f"## {index}. {t0.get('msg_id', '')}",
        "",
        f"- status: `{row.get('status', '')}`",
        f"- raw_context_quality: `{row.get('raw_context_quality', '')}`",
        f"- t0_iso: `{t0.get('iso', '')}`",
        f"- merged_bc_count: {len(corrections)}",
        f"- treatment_hint: {row.get('comment_for_replay') or legs.get('err_hint') or '(empty)'}",
        f"- rewrite_generation: `{legs.get('rewrite_generation', '')}`",
        "- corrections:",
        *(bc_lines or ["  (empty)"]),
        "",
        "### baseline",
        _fence(legs.get("baseline_scene") or legs.get("baseline_user") or prompts.get("baseline") or ""),
        "",
        "### treatment",
        _fence(_treatment_prompt(row)),
        "",
        "### placebo",
        _fence(_placebo_prompt(row)),
        "",
    ]
    return "\n".join(parts)


def render(rows: list[dict], source: str) -> str:
    ok = [row for row in rows if row.get("status") == "ok"]
    has_legs = any(row.get("legs") for row in ok)
    merged = ok if has_legs else _merge_by_t0(ok)
    empty_hint = sum(
        1 for row in merged
        if not (row.get("comment_for_replay") or (row.get("legs") or {}).get("err_hint"))
    )
    header = [
        "# v-final 三腿 replay 输入 · 截止 2026-07-10",
        "",
        f"- source: `{source}`",
        f"- raw_rows: {len(rows)}",
        f"- ok_raw_rows: {len(ok)}",
        f"- treatment_units_by_t0: {len(merged)}",
        f"- empty_treatment_hint: {empty_hint}",
        f"- shared_system: `{BASE_SYSTEM}`",
        f"- placebo_rule: {PLACEBO_RULE}",
        "",
        "说明：baseline / treatment / placebo 三腿使用同一个 system；下面保存的是三腿各自的 user 输入。treatment 只注入从纠错事实抽象出来的通用行为准则，不注入 B/C 原文或正确答案。",
        "",
    ]
    body = [_row_summary(row, i) for i, row in enumerate(merged, 1)]
    return "\n".join(header + body)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshots", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    rows = _load_jsonl(args.snapshots)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(render(rows, args.snapshots), encoding="utf-8")
    print(f"wrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
