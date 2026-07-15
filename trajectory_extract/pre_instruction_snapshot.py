#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build T0 replay packs for testing whether rewritten comments help.

For each rewritten instruction from rewrite_thread_result.json, this script
finds the original instruction event, treats its timestamp as T0, and emits two
prompt variants over the same pre-T0 world:
  - baseline: pre-T0 context + original instruction
  - with_comment: pre-T0 context + replay-safe comment + original instruction

The output is intentionally model-agnostic JSONL. A later runner can feed the
prompt variants to any model and judge whether baseline fails while comment
passes.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import principle_distill as pd  # noqa: E402
import runtime_lane as rl  # noqa: E402
import task_stitch as ts  # noqa: E402


DATA = "/opt/shared/data/task-trajectory"
GROUP = rl.GROUP
DEFAULT_REWRITE = f"{DATA}/rewrite_thread_result.json"
DEFAULT_OUT = f"{DATA}/pre_instruction_snapshots.jsonl"
DEFAULT_STATE = f"{DATA}/state/{GROUP}.json"
DEFAULT_CHUNKS = [f"{DATA}/events_chunk1_107.json", f"{DATA}/events_chunk2_raw224.json"]
DEFAULT_CORRECTIONS = f"{DATA}/user_corrections_pool_vfinal_recheck_verify_kept.jsonl"


def _read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _event_ts(event: dict) -> float:
    raw = event.get("ts") or event.get("created_at") or 0
    try:
        ts = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if ts > 10_000_000_000:
        return ts / 1000.0
    return ts


def _iso(ts: float) -> str | None:
    if not ts:
        return None
    return _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc).isoformat()


def _clean_text(text: str | None) -> str:
    return ts._strip_feishu(text or "").strip()


def _load_events_from_state(state_path: str) -> list[dict]:
    state = _read_json(state_path)
    events: list[dict] = []
    for task in state.get("frozen_tasks", []) or []:
        events.extend(task.get("events") or [])
    events.extend(state.get("active_tail_events") or [])
    return events


def _load_events_from_json(path: str) -> list[dict]:
    data = _read_json(path)
    if isinstance(data, list):
        return [e for e in data if isinstance(e, dict)]
    return [e for e in (data.get("events") or []) if isinstance(e, dict)]


def load_events(state_path: str = DEFAULT_STATE, chunk_paths: list[str] | None = None) -> list[dict]:
    """Load all known transcript events, dedupe by msg_id, sort by timestamp."""
    chunk_paths = DEFAULT_CHUNKS if chunk_paths is None else chunk_paths
    sources = [_load_events_from_state(state_path)]
    sources.extend(_load_events_from_json(p) for p in chunk_paths)

    seen: set[str] = set()
    events: list[dict] = []
    for source in sources:
        for event in source:
            mid = event.get("msg_id")
            if not mid or mid in seen:
                continue
            seen.add(mid)
            events.append(event)
    events.sort(key=_event_ts)
    return events


def _lane_short_map(events: list[dict]) -> dict[str, str]:
    lane_of, _prov = rl.assign_lanes(events)
    return {mid: rl.lane_short(lane) for mid, lane in lane_of.items()}


def _format_event(event: dict) -> dict:
    return {
        "msg_id": event.get("msg_id"),
        "ts": _event_ts(event),
        "iso": _iso(_event_ts(event)),
        "role": event.get("role") or "",
        "name": event.get("name") or "",
        "parent_id": event.get("parent_id") or "",
        "text": _clean_text(event.get("text")),
    }


def _render_context(events: list[dict]) -> str:
    lines: list[str] = []
    for i, event in enumerate(events, 1):
        role = event.get("role") or ""
        name = event.get("name") or role
        text = _clean_text(event.get("text"))
        lines.append(f"M{i} [{role}/{name}] {text}")
    return "\n".join(lines)


def _git_commit_before(repo: str, ts: float) -> dict:
    repo_path = Path(repo)
    out = {"repo": str(repo_path), "commit_before_t0": None, "status": "missing"}
    if not repo_path.exists():
        return out
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo_path), "rev-list", "-1", f"--before=@{int(ts)}", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception as exc:  # noqa: BLE001
        out.update({"status": "error", "error": str(exc)})
        return out
    if not commit:
        out["status"] = "no_commit_before_t0"
        return out
    try:
        subject = subprocess.check_output(
            ["git", "-C", str(repo_path), "log", "-1", "--format=%s", commit],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        subject = ""
    out.update({"commit_before_t0": commit, "subject": subject, "status": "ok"})
    return out


_CONCRETE_RESULT_RE = re.compile(
    r"(\d|[零一二三四五六七八九十百千万亿]+(?:个|条|项|次|轮|类|层|组|篇|份|任务|子需求)|->|→|=>)"
)
_FAILURE_WORD_RE = re.compile(r"(编造|幻觉|捏造|凭空|错|错误|误判|戳穿)")


def _correction_anchor_event(correction: dict, by_id: dict[str, dict]) -> dict | None:
    anchor = correction.get("anchor")
    if not anchor:
        return None
    return by_id.get(anchor)


def _is_post_t0_failure_evidence(correction: dict, t0: float, by_id: dict[str, dict]) -> bool:
    anchor_event = _correction_anchor_event(correction, by_id)
    if not anchor_event or _event_ts(anchor_event) < t0:
        return False
    text = str(correction.get("what") or "")
    return bool(_CONCRETE_RESULT_RE.search(text) or _FAILURE_WORD_RE.search(text))


def _post_t0_evidence(corrections: list[dict], t0: float, by_id: dict[str, dict]) -> list[dict]:
    """Concrete T0-post failure evidence — kept for audit, never injected."""
    evidence: list[dict] = []
    for correction in corrections:
        if not _is_post_t0_failure_evidence(correction, t0, by_id):
            continue
        anchor_event = _correction_anchor_event(correction, by_id)
        evidence.append({
            "correction": correction,
            "anchor_event": _format_event(anchor_event) if anchor_event else None,
            "reason": "correction anchor is at/after T0 and contains concrete failure/result evidence",
        })
    return evidence


def clean_general(distill_audit: list[dict] | None,
                  post_t0_evidence: list[dict] | None,
                  allow_post_t0_safe_hints: bool = True) -> list[str]:
    """注入用的 distilled GENERAL 准则。

    单一事实源：快照的 comment_for_replay 和 replay_reconstruct 的 err_hint 都调这个，
    默认允许 T0 后纠错被抽象成不含实例答案的通用准则；诊断旧保守口径时可关闭。
    """
    post = {re.sub(r"\s+", "", (ev.get("correction") or {}).get("what") or "")
            for ev in (post_t0_evidence or [])}
    out: list[str] = []
    seen: set[str] = set()
    for a in (distill_audit or []):
        if (
            not allow_post_t0_safe_hints
            and re.sub(r"\s+", "", a.get("source_what") or "") in post
        ):
            continue  # ①剔 post_t0 下游裁决
        g = (a.get("general") or "").strip()  # ②只喂已剥离答案的通用准则
        key = re.sub(r"\s+", "", g)
        if g and key not in seen:
            seen.add(key)
            out.append(g)
    return out


def _dedup_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        key = re.sub(r"\s+", "", it)
        if key and key not in seen:
            seen.add(key)
            out.append(it)
    return out


def _replay_principles(
    corrections: list[dict],
    model: str | None,
    client: Any,
    cache: dict[str, dict],
) -> tuple[list[str], list[dict], list[dict]]:
    """Distill this thread's corrections into general/domain behavior principles.

    B route (张耀明 2026-07-03): the injected comment is an abstracted behavior
    rule, not the completed requirement — carries no instance answer.
    """
    general: list[str] = []
    domain: list[dict] = []
    audit: list[dict] = []
    for correction in corrections:
        what = str(correction.get("what") or "").strip()
        if not what:
            continue
        res = cache.get(what)
        if res is None:
            res = pd.distill_one(what, model=model, client=client)
            cache[what] = res
        audit.append(res)
        if res.get("general"):
            general.append(res["general"])
        if res.get("domain"):
            domain.append({"text": res["domain"], "label": res.get("domain_label")})
    general = _dedup_keep_order(general)
    seen_dom: set[str] = set()
    domain_dedup: list[dict] = []
    for d in domain:
        key = re.sub(r"\s+", "", d["text"])
        if key and key not in seen_dom:
            seen_dom.add(key)
            domain_dedup.append(d)
    return general, domain_dedup, audit


def _prompt_pair(pre_context: str, original: str, comment_for_replay: str) -> dict:
    header = "你正在复现原始指令到达前一刻的执行状态。下面只包含 T0 前上下文。"
    baseline = (
        f"{header}\n\n【T0 前同一运行时 thread 上下文】\n{pre_context or '(空)'}"
        f"\n\n【原始指令】\n{original}\n\n请按当时状态完成这条原始指令。"
    )
    with_comment = (
        f"{header}\n\n【T0 前同一运行时 thread 上下文】\n{pre_context or '(空)'}"
        f"\n\n【通用行为准则（进 system，不是本题答案）】\n{comment_for_replay}"
        f"\n\n【原始指令】\n{original}\n\n请按当时状态完成这条原始指令。"
    )
    return {"baseline": baseline, "with_comment": with_comment}


def build_snapshots(
    rewrite_path: str = DEFAULT_REWRITE,
    state_path: str = DEFAULT_STATE,
    chunk_paths: list[str] | None = None,
    context_limit: int = 80,
    repo_paths: list[str] | None = None,
    model: str | None = None,
) -> list[dict]:
    rewrite = _read_json(rewrite_path)
    rewritten = rewrite.get("rewritten") or []
    events = load_events(state_path, chunk_paths)
    by_id = {e.get("msg_id"): e for e in events if e.get("msg_id")}
    lane_of = _lane_short_map(events)
    repo_paths = repo_paths or ["/home/agent/data_process", "/home/agent/lian-codex-bot"]

    client = pd._client()
    distill_cache: dict[str, dict] = {}
    # Distillation is embarrassingly parallel across unique corrections; pre-warm
    # the cache concurrently (flash is not throttled by the shared v4-pro limit).
    uniq_whats = {
        str(c.get("what") or "").strip()
        for item in rewritten for c in (item.get("corrections") or [])
        if str(c.get("what") or "").strip()
    }
    if uniq_whats:
        from concurrent.futures import ThreadPoolExecutor
        def _one(w: str) -> tuple[str, dict]:
            return w, pd.distill_one(w, model=model, client=client)
        with ThreadPoolExecutor(max_workers=2) as ex:
            for w, res in ex.map(_one, sorted(uniq_whats)):
                distill_cache[w] = res
    rows: list[dict] = []
    for item in rewritten:
        fix_mid = item.get("instruction_to_fix_msg")
        event = by_id.get(fix_mid)
        if not event:
            rows.append({
                "schema_version": 2,
                "instruction_to_fix_msg": fix_mid,
                "status": "missing_instruction_event",
                "source_rewrite": item,
            })
            continue

        t0 = _event_ts(event)
        lane = item.get("lane") or lane_of.get(fix_mid)
        pre_events = [
            e for e in events
            if _event_ts(e) < t0 and (lane is None or lane_of.get(e.get("msg_id")) == lane)
        ][-context_limit:]
        pre_context = _render_context(pre_events)
        original = _clean_text(event.get("text")) or item.get("instruction_to_fix_text", "")
        source_comment = item.get("rewritten", "")
        corrections = item.get("corrections") or []
        general, domain, distill_audit = _replay_principles(
            corrections=corrections, model=model, client=client, cache=distill_cache,
        )
        post_t0_failure_evidence = _post_t0_evidence(corrections, t0, by_id)
        # comment_for_replay injects the GENERAL layer only (agent-agnostic replay),
        # AND excludes generals distilled from post_t0 downstream verdicts —
        # otherwise a t0-post conclusion (e.g. "回退v1") leaks back in via its principle.
        # domain layer is stored for domain-agent variants, not injected here.
        comment_for_replay = "；".join(clean_general(distill_audit, post_t0_failure_evidence))

        rows.append({
            "schema_version": 3,
            "status": "ok",
            "group_id": GROUP,
            "lane": lane,
            "t0": {
                "msg_id": fix_mid,
                "ts": t0,
                "iso": _iso(t0),
                "event": _format_event(event),
            },
            "code_refs": [_git_commit_before(repo, t0) for repo in repo_paths],
            "memory_policy": {
                "rule": "Only memory/core/history entries with updated_at < T0 may be loaded.",
                "historical_memory_export": None,
                "note": "This pack records the filter rule; attach a separately exported pre-T0 memory snapshot if available.",
            },
            "context": {
                "scope": "same_runtime_lane_before_t0",
                "limit": context_limit,
                "n_events": len(pre_events),
                "events": [_format_event(e) for e in pre_events],
            },
            "original_instruction": original,
            "source_comment": source_comment,
            "replay_principles": {"general": general, "domain": domain},
            "comment_for_replay": comment_for_replay,
            "post_t0_failure_evidence": post_t0_failure_evidence,
            "corrections": corrections,
            "prompts": _prompt_pair(pre_context, original, comment_for_replay),
            "leakage_guard": {
                "excluded": "All events with ts >= T0, including correction anchors and later bot summaries.",
                "comment_policy": "B route: comment_for_replay = distilled GENERAL behavior principles (no instance answer). T0 后纠错只能以脱敏后的通用行为准则进入，source_comment kept for audit only, never injected.",
                "red_line": "each principle must not contain any number-token from its own source failure evidence.",
                "distill_audit": distill_audit,
            },
        })
    return rows


def _correction_item_from_vfinal_row(row: dict, by_id: dict[str, dict], lane_of: dict[str, str]) -> dict:
    t0_mid = row.get("_t0_msg_id") or row.get("t0_msg_id")
    t0_event = by_id.get(t0_mid)
    original = _clean_text(t0_event.get("text")) if t0_event else ""
    what = str(row.get("what") or "").strip()
    anchor = row.get("anchor_msg_id") or row.get("bot_error_msg_id")
    corrector = row.get("corrector_msg_id")
    correction = {
        "anchor": anchor,
        "what": what,
        "corrector": corrector,
        "corrector_role": row.get("corrector_role"),
        "bot_error_quote": row.get("bot_error_quote"),
        "source": row.get("task") or "vfinal_recheck",
    }
    return {
        "lane": lane_of.get(t0_mid),
        "instruction_to_fix_msg": t0_mid,
        "instruction_to_fix_text": original,
        "rewritten": what,
        "corrections": [correction],
        "_source_vfinal_row": row,
    }


def build_snapshots_from_corrections(
    corrections_path: str = DEFAULT_CORRECTIONS,
    state_path: str = DEFAULT_STATE,
    chunk_paths: list[str] | None = None,
    context_limit: int = 80,
    repo_paths: list[str] | None = None,
    model: str | None = None,
) -> list[dict]:
    """Build replay snapshots directly from v-final kept correction rows.

    The kept rows already contain the v-final t0/B/C ids and the human-readable
    failure summary.  This adapter converts them to the older rewrite item
    shape, then reuses the same snapshot renderer and leakage policy.
    """
    events = load_events(state_path, chunk_paths)
    by_id = {e.get("msg_id"): e for e in events if e.get("msg_id")}
    lane_of = _lane_short_map(events)
    repo_paths = repo_paths or ["/home/agent/data_process", "/home/agent/lian-codex-bot"]
    source_rows = _read_jsonl(corrections_path)

    rewritten = [
        _correction_item_from_vfinal_row(row, by_id, lane_of)
        for row in source_rows
    ]

    client = pd._client()
    distill_cache: dict[str, dict] = {}
    uniq_whats = {
        str(c.get("what") or "").strip()
        for item in rewritten for c in (item.get("corrections") or [])
        if str(c.get("what") or "").strip()
    }
    if uniq_whats:
        from concurrent.futures import ThreadPoolExecutor
        def _one(w: str) -> tuple[str, dict]:
            return w, pd.distill_one(w, model=model, client=client)
        with ThreadPoolExecutor(max_workers=2) as ex:
            for w, res in ex.map(_one, sorted(uniq_whats)):
                distill_cache[w] = res

    rows: list[dict] = []
    for item in rewritten:
        fix_mid = item.get("instruction_to_fix_msg")
        event = by_id.get(fix_mid)
        if not event:
            rows.append({
                "schema_version": 3,
                "instruction_to_fix_msg": fix_mid,
                "status": "missing_instruction_event",
                "source_vfinal_row": item.get("_source_vfinal_row"),
            })
            continue

        t0 = _event_ts(event)
        lane = item.get("lane") or lane_of.get(fix_mid)
        pre_events = [
            e for e in events
            if _event_ts(e) < t0 and (lane is None or lane_of.get(e.get("msg_id")) == lane)
        ][-context_limit:]
        pre_context = _render_context(pre_events)
        original = _clean_text(event.get("text")) or item.get("instruction_to_fix_text", "")
        source_comment = item.get("rewritten", "")
        corrections = item.get("corrections") or []
        general, domain, distill_audit = _replay_principles(
            corrections=corrections, model=model, client=client, cache=distill_cache,
        )
        post_t0_failure_evidence = _post_t0_evidence(corrections, t0, by_id)
        comment_for_replay = "；".join(clean_general(distill_audit, post_t0_failure_evidence))

        row = {
            "schema_version": 3,
            "status": "ok",
            "source": "vfinal_recheck_verify_kept",
            "group_id": GROUP,
            "lane": lane,
            "t0": {
                "msg_id": fix_mid,
                "ts": t0,
                "iso": _iso(t0),
                "event": _format_event(event),
            },
            "code_refs": [_git_commit_before(repo, t0) for repo in repo_paths],
            "memory_policy": {
                "rule": "Only memory/core/history entries with updated_at < T0 may be loaded.",
                "historical_memory_export": None,
                "note": "This pack records the filter rule; attach a separately exported pre-T0 memory snapshot if available.",
            },
            "context": {
                "scope": "same_runtime_lane_before_t0",
                "limit": context_limit,
                "n_events": len(pre_events),
                "events": [_format_event(e) for e in pre_events],
            },
            "original_instruction": original,
            "source_comment": source_comment,
            "replay_principles": {"general": general, "domain": domain},
            "comment_for_replay": comment_for_replay,
            "post_t0_failure_evidence": post_t0_failure_evidence,
            "corrections": corrections,
            "prompts": _prompt_pair(pre_context, original, comment_for_replay),
            "leakage_guard": {
                "excluded": "All events with ts >= T0, including correction anchors and later bot summaries.",
                "comment_policy": "B route: comment_for_replay = distilled GENERAL behavior principles (no instance answer). T0 后纠错只能以脱敏后的通用行为准则进入，source_comment kept for audit only, never injected.",
                "red_line": "each principle must not contain any number-token from its own source failure evidence.",
                "distill_audit": distill_audit,
            },
            "source_vfinal_row": item.get("_source_vfinal_row"),
        }
        rows.append(row)
    return rows


def write_jsonl(rows: list[dict], out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["rewrite", "vfinal-kept"], default="rewrite")
    ap.add_argument("--rewrite", default=DEFAULT_REWRITE)
    ap.add_argument("--corrections", default=DEFAULT_CORRECTIONS)
    ap.add_argument("--state", default=DEFAULT_STATE)
    ap.add_argument("--chunk", action="append", dest="chunks", default=None,
                    help="Extra event chunk JSON. Repeatable; defaults to known chunk1/chunk2.")
    ap.add_argument("--context-limit", type=int, default=80)
    ap.add_argument("--repo", action="append", dest="repos", default=None,
                    help="Git repo to resolve commit-before-T0. Repeatable.")
    ap.add_argument("--model", default=None, help="Distill model (default flash via principle_distill).")
    ap.add_argument("--out", default=DEFAULT_OUT)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.source == "vfinal-kept":
        rows = build_snapshots_from_corrections(
            corrections_path=args.corrections,
            state_path=args.state,
            chunk_paths=args.chunks,
            context_limit=args.context_limit,
            repo_paths=args.repos,
            model=args.model,
        )
    else:
        rows = build_snapshots(
            rewrite_path=args.rewrite,
            state_path=args.state,
            chunk_paths=args.chunks,
            context_limit=args.context_limit,
            repo_paths=args.repos,
            model=args.model,
        )
    write_jsonl(rows, args.out)
    ok = sum(1 for r in rows if r.get("status") == "ok")
    print(f"wrote {len(rows)} snapshots ({ok} ok) -> {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
