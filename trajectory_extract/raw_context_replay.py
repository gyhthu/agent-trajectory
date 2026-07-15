#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build raw-context replay manifests.

This is the higher-fidelity replay input builder: it keeps the old invariant
that all legs share the same base system, but replaces the synthetic short
context with the original events visible before T0.

Default mode is deterministic and model-free:
  snapshot/vfinal row -> locate T0 -> choose pre-T0 raw context -> write manifest

Optional ``--generate-rewrites`` calls the existing rewrite LLM helpers to fill
placebo/treatment user prompts.  The LLM is never allowed to locate missing
events or invent context.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import pre_instruction_snapshot as PIS  # noqa: E402
import replay_reconstruct as RR  # noqa: E402
import replay_three_leg as R  # noqa: E402
import runtime_lane as RL  # noqa: E402


DATA = "/opt/shared/data/task-trajectory"
DEFAULT_SNAPSHOTS = f"{DATA}/pre_instruction_snapshots.jsonl"
DEFAULT_OUT = f"{DATA}/raw_context_replay_manifest.jsonl"
DEFAULT_PREVIEW_CHARS = 180
DEFAULT_PREVIEW_EVENTS = 12
DEFAULT_MAX_JSONL_LINE_CHARS = 30000

QUALITY_EXACT_THREAD = "exact_thread"
QUALITY_SAME_SESSION = "same_session_window"
QUALITY_RECOVERED_PARENT = "recovered_by_parent"
QUALITY_MISSING = "missing_or_ambiguous"


def _read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _merge_text_items(*values: str | None) -> str:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        for item in (value or "").replace("；", ";").split(";"):
            item = item.strip()
            if item and item not in seen:
                seen.add(item)
                out.append(item)
    return "；".join(out)


def merge_snapshots_by_t0(snapshots: list[dict]) -> list[dict]:
    """Merge multiple B/C rows under the same v-final t0 into one treatment unit."""
    grouped: dict[str, dict] = {}
    for row in snapshots:
        t0id = ((row.get("t0") or {}).get("msg_id")) or row.get("t0_msg_id") or row.get("_t0_msg_id")
        if not t0id:
            t0id = f"_missing_{len(grouped)}"
        if t0id not in grouped:
            merged = dict(row)
            merged["corrections"] = list(row.get("corrections") or [])
            merged["_source_snapshot_count"] = 1
            grouped[t0id] = merged
            continue

        merged = grouped[t0id]
        merged["_source_snapshot_count"] = int(merged.get("_source_snapshot_count") or 1) + 1
        merged["corrections"].extend(row.get("corrections") or [])
        merged["comment_for_replay"] = _merge_text_items(
            merged.get("comment_for_replay"),
            row.get("comment_for_replay"),
        )

        guard = dict(merged.get("leakage_guard") or {})
        other_guard = row.get("leakage_guard") or {}
        guard["distill_audit"] = list(guard.get("distill_audit") or []) + list(other_guard.get("distill_audit") or [])
        merged["leakage_guard"] = guard
        merged["post_t0_failure_evidence"] = (
            list(merged.get("post_t0_failure_evidence") or [])
            + list(row.get("post_t0_failure_evidence") or [])
        )
    return list(grouped.values())


def _event_ts(event: dict) -> float:
    return PIS._event_ts(event)


def _clean_text(text: str | None) -> str:
    return PIS._clean_text(text)


def _norm(text: str | None) -> str:
    text = _clean_text(text)
    text = re.sub(r"[`*_#>\[\](){}:：,，.。!！?？\s]+", "", text)
    return text[:800]


def _format_event(event: dict, lane_of: dict[str, str] | None = None) -> dict:
    rec = PIS._format_event(event)
    mid = rec.get("msg_id")
    if lane_of and mid:
        rec["lane"] = lane_of.get(mid)
    return rec


def _preview_event(event: dict, lane_of: dict[str, str] | None = None, preview_chars: int = DEFAULT_PREVIEW_CHARS) -> dict:
    rec = _format_event(event, lane_of)
    text = _clean_text(rec.get("text"))
    rec["text_preview"] = text[:preview_chars]
    rec["text_chars"] = len(text)
    rec.pop("text", None)
    return rec


def _snapshot_ref(row: dict, preview_chars: int = DEFAULT_PREVIEW_CHARS) -> dict:
    t0 = row.get("t0") or {}
    return {
        "status": row.get("status"),
        "group_id": row.get("group_id"),
        "t0_msg_id": t0.get("msg_id") or row.get("t0_msg_id") or row.get("_t0_msg_id"),
        "original_instruction_preview": _clean_text(
            row.get("original_instruction") or row.get("instruction") or ""
        )[:preview_chars],
        "original_instruction_chars": len(_clean_text(row.get("original_instruction") or row.get("instruction") or "")),
    }


def _preview_events(
    events: list[dict],
    lane_of: dict[str, str],
    preview_chars: int,
    preview_events: int,
) -> list[dict]:
    if preview_events <= 0:
        return []
    return [_preview_event(e, lane_of, preview_chars) for e in events[-preview_events:]]


def _root_of(mid: str | None, by_id: dict[str, dict]) -> tuple[str | None, bool, list[str]]:
    """Return (root_msg_id, complete_parent_chain, chain_from_root_to_mid)."""
    if not mid or mid not in by_id:
        return None, False, []
    seen: set[str] = set()
    chain = [mid]
    cur = mid
    complete = True
    while cur and cur not in seen:
        seen.add(cur)
        parent = by_id.get(cur, {}).get("parent_id")
        if not parent:
            break
        if parent not in by_id:
            complete = False
            chain.append(parent)
            cur = parent
            break
        chain.append(parent)
        cur = parent
    else:
        complete = False
    chain = list(reversed(chain))
    return chain[0] if chain else mid, complete, chain


def _events_with_root(events: list[dict], root: str | None, by_id: dict[str, dict]) -> list[dict]:
    if not root:
        return []
    out: list[dict] = []
    for event in events:
        mid = event.get("msg_id")
        ev_root, _complete, _chain = _root_of(mid, by_id)
        if ev_root == root:
            out.append(event)
    return out


def _snapshot_t0_event(row: dict, by_id: dict[str, dict]) -> tuple[dict | None, list[dict]]:
    """Use structural id first; fall back to text candidates only for audit."""
    t0 = row.get("t0") or {}
    t0_mid = t0.get("msg_id") or row.get("t0_msg_id") or row.get("_t0_msg_id")
    if t0_mid and t0_mid in by_id:
        return by_id[t0_mid], []

    original = row.get("original_instruction") or row.get("instruction") or ""
    key = _norm(original)
    candidates: list[dict] = []
    if key:
        for event in by_id.values():
            if (event.get("role") or "") != "user":
                continue
            ev_key = _norm(event.get("text"))
            if key and (key in ev_key or ev_key in key):
                candidates.append(event)
    if len(candidates) == 1:
        return candidates[0], candidates
    return None, candidates[:10]


def _correction_msg_ids(row: dict) -> set[str]:
    ids: set[str] = set()
    for correction in row.get("corrections") or []:
        for key in ("anchor", "corrector", "anchor_msg_id", "corrector_msg_id"):
            if correction.get(key):
                ids.add(str(correction[key]))
    src = row.get("source_vfinal_row") or {}
    for key in ("anchor_msg_id", "bot_error_msg_id", "corrector_msg_id"):
        if src.get(key):
            ids.add(str(src[key]))
    return ids


def _choose_context(
    row: dict,
    t0_event: dict,
    events: list[dict],
    by_id: dict[str, dict],
    lane_of: dict[str, str],
    context_limit: int,
) -> tuple[str, list[dict], dict]:
    t0_mid = t0_event.get("msg_id")
    t0_ts = _event_ts(t0_event)
    t0_lane = lane_of.get(t0_mid)
    root, root_complete, chain = _root_of(t0_mid, by_id)

    pre_events = [e for e in events if _event_ts(e) < t0_ts]
    thread_events = _events_with_root(pre_events, root, by_id) if t0_event.get("parent_id") else []
    thread_events = [e for e in thread_events if e.get("msg_id") != t0_mid]

    if thread_events and root_complete:
        quality = QUALITY_EXACT_THREAD
        chosen = thread_events
    elif thread_events:
        quality = QUALITY_RECOVERED_PARENT
        chosen = thread_events
    else:
        quality = QUALITY_SAME_SESSION
        chosen = [e for e in pre_events if not t0_lane or lane_of.get(e.get("msg_id")) == t0_lane]

    chosen = chosen[-context_limit:]
    correction_ids = _correction_msg_ids(row)
    leaked_corrections = sorted(mid for mid in correction_ids if mid in {e.get("msg_id") for e in chosen})
    leaked_after_t0 = [e.get("msg_id") for e in chosen if _event_ts(e) >= t0_ts]
    if leaked_corrections or leaked_after_t0:
        quality = QUALITY_MISSING

    audit = {
        "t0_msg_id": t0_mid,
        "t0_lane": t0_lane,
        "root_msg_id": root,
        "parent_chain_complete": root_complete,
        "parent_chain": chain,
        "context_limit": context_limit,
        "leakage_flags": {
            "has_event_at_or_after_t0": bool(leaked_after_t0),
            "event_ids_at_or_after_t0": leaked_after_t0,
            "has_b_or_c_event": bool(leaked_corrections),
            "b_or_c_event_ids": leaked_corrections,
        },
    }
    return quality, chosen, audit


def _render_context(events: list[dict]) -> str:
    lines: list[str] = []
    for i, event in enumerate(events, 1):
        role = event.get("role") or ""
        name = event.get("name") or role
        text = _clean_text(event.get("text"))
        if text:
            lines.append(f"M{i} [{role}/{name}] {text}")
    return "\n".join(lines)


def _build_user_scene(context_text: str, user_prompt: str) -> str:
    return (
        "你正在复现原始指令到达前一刻的执行状态。下面只包含 T0 前上下文。"
        f"\n\n【T0 前原始上下文】\n{context_text or '(空)'}"
        f"\n\n【当前用户输入】\n{user_prompt}"
    )


def _leg_prompts(
    row: dict,
    context_text: str,
    generate_rewrites: bool,
    client: Any | None,
    embed_full_context: bool,
) -> dict:
    original = row.get("original_instruction") or ""
    err_hint, err_src = RR.build_err_hint(row, {"comment_for_replay": row.get("comment_for_replay", "")})
    placebo = None
    treatment = None
    if generate_rewrites:
        placebo = RR.rewrite_placebo(client, original)
        treatment = RR.rewrite(client, original, err_hint) if err_hint else placebo
    prompts = {
        "err_hint": err_hint,
        "err_src": err_src,
        "baseline_user": original,
        "placebo_user": placebo,
        "treatment_user": treatment,
        "rewrite_generation": "llm_generated" if generate_rewrites else "pending",
    }
    if embed_full_context:
        prompts.update({
            "baseline_scene": _build_user_scene(context_text, original),
            "placebo_scene": _build_user_scene(context_text, placebo) if placebo else None,
            "treatment_scene": _build_user_scene(context_text, treatment) if treatment else None,
        })
    return prompts


def build_manifest(
    snapshots: list[dict],
    context_limit: int = 80,
    generate_rewrites: bool = False,
    embed_full_context: bool = False,
    preview_chars: int = DEFAULT_PREVIEW_CHARS,
    preview_events: int = DEFAULT_PREVIEW_EVENTS,
) -> list[dict]:
    events = PIS.load_events()
    by_id = {e.get("msg_id"): e for e in events if e.get("msg_id")}
    lane_of, lane_prov = RL.assign_lanes(events)
    lane_short = {mid: RL.lane_short(lane) for mid, lane in lane_of.items()}
    client = R._client() if generate_rewrites else None

    rows: list[dict] = []
    for idx, row in enumerate(snapshots):
        t0_event, candidates = _snapshot_t0_event(row, by_id)
        if row.get("status") != "ok" or not t0_event:
            rows.append({
                "schema_version": 2,
                "idx": idx,
                "status": "missing_t0",
                "raw_context_quality": QUALITY_MISSING,
                "original_instruction": row.get("original_instruction") or "",
                "t0_candidates": [_preview_event(c, lane_short, preview_chars) for c in candidates],
                "source_snapshot_ref": _snapshot_ref(row, preview_chars),
                "system_ref": "_BASE_SYS",
            })
            continue

        quality, context_events, audit = _choose_context(
            row=row,
            t0_event=t0_event,
            events=events,
            by_id=by_id,
            lane_of=lane_short,
            context_limit=context_limit,
        )
        context_text = _render_context(context_events) if embed_full_context else ""
        prompts = _leg_prompts(row, context_text, generate_rewrites, client, embed_full_context)
        t0_mid = t0_event.get("msg_id")
        context = {
            "scope": quality,
            "n_events": len(context_events),
            "event_ids": [e.get("msg_id") for e in context_events],
            "events_preview": _preview_events(context_events, lane_short, preview_chars, preview_events),
            "preview_policy": {
                "position": "tail",
                "max_events": preview_events,
                "max_chars_per_event": preview_chars,
            },
            "materialized": bool(embed_full_context),
        }
        if embed_full_context:
            context["events"] = [_format_event(e, lane_short) for e in context_events]
            context["rendered"] = context_text
        rows.append({
            "schema_version": 2,
            "idx": idx,
            "status": "ok" if quality != QUALITY_MISSING else "needs_review",
            "raw_context_quality": quality,
            "group_id": row.get("group_id") or RL.GROUP,
            "t0": _format_event(t0_event, lane_short),
            "lane": lane_short.get(t0_mid),
            "lane_provenance": lane_prov.get(t0_mid),
            "context": context,
            "raw_context_audit": audit,
            "original_instruction": row.get("original_instruction") or _clean_text(t0_event.get("text")),
            "corrections": row.get("corrections") or [],
            "source_vfinal_row": row.get("source_vfinal_row"),
            "legs": prompts,
            "system_ref": "_BASE_SYS",
        })
    return rows


def write_jsonl(rows: list[dict], out_path: str, max_line_chars: int = DEFAULT_MAX_JSONL_LINE_CHARS) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for idx, row in enumerate(rows):
            line = json.dumps(row, ensure_ascii=False)
            if max_line_chars > 0 and len(line) > max_line_chars:
                raise ValueError(
                    f"manifest row {idx} is {len(line)} chars > --max-line-chars {max_line_chars}; "
                    "write a slim manifest or explicitly raise the limit"
                )
            f.write(line + "\n")


def _summary(rows: list[dict]) -> str:
    quality = Counter(r.get("raw_context_quality") for r in rows)
    status = Counter(r.get("status") for r in rows)
    parts = [
        f"rows={len(rows)}",
        "status=" + ",".join(f"{k}:{v}" for k, v in sorted(status.items())),
        "quality=" + ",".join(f"{k}:{v}" for k, v in sorted(quality.items())),
    ]
    return " ".join(parts)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshots", default=DEFAULT_SNAPSHOTS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--context-limit", type=int, default=80)
    ap.add_argument("--preview-chars", type=int, default=DEFAULT_PREVIEW_CHARS,
                    help="Characters kept per context event in slim manifests.")
    ap.add_argument("--preview-events", type=int, default=DEFAULT_PREVIEW_EVENTS,
                    help="Number of tail context events previewed in slim manifests.")
    ap.add_argument("--embed-full-context", action="store_true",
                    help="Opt in to writing full context events and materialized replay scenes.")
    ap.add_argument("--max-line-chars", type=int, default=DEFAULT_MAX_JSONL_LINE_CHARS,
                    help="Fail if any output JSONL row exceeds this many chars; use 0 to disable.")
    ap.add_argument("--generate-rewrites", action="store_true",
                    help="Call replay_reconstruct LLM helpers to fill placebo/treatment user prompts.")
    ap.add_argument("--merge-by-t0", action="store_true",
                    help="Merge rows with the same v-final t0 before building replay units.")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    snapshots = _read_jsonl(args.snapshots)
    if args.merge_by_t0:
        snapshots = merge_snapshots_by_t0(snapshots)
    rows = build_manifest(
        snapshots=snapshots,
        context_limit=args.context_limit,
        generate_rewrites=args.generate_rewrites,
        embed_full_context=args.embed_full_context,
        preview_chars=args.preview_chars,
        preview_events=args.preview_events,
    )
    write_jsonl(rows, args.out, max_line_chars=args.max_line_chars)
    print(f"wrote {args.out} {_summary(rows)}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
