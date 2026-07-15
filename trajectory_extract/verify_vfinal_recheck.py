#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the session-level adversarial verifier on v-final recheck accepts."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import runtime_lane as rl  # noqa: E402
import session_corrections as sc  # noqa: E402


DATA = os.environ.get("TASK_TRAJECTORY_DATA", "/opt/shared/data/task-trajectory")
RECHECK_IN = f"{DATA}/user_corrections_pool_vfinal_recheck.jsonl"
KEPT_OUT = f"{DATA}/user_corrections_pool_vfinal_recheck_verify_kept.jsonl"
DROPPED_OUT = f"{DATA}/user_corrections_pool_vfinal_recheck_verify_dropped.jsonl"


def _load_jsonl(path: str) -> list[dict]:
    rows = []
    p = Path(path)
    if not p.exists():
        return rows
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _backup(path: str) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dst = f"{path}.bak_{stamp}"
    shutil.copy2(path, dst)
    return dst


def _to_verify_row(row: dict) -> dict:
    return {
        "anchor_msg_id": row.get("bot_error_msg_id", ""),
        "bot_error_quote": row.get("bot_error_quote", ""),
        "corrector_msg_id": row.get("corrector_msg_id", ""),
        "corrector_role": "user",
        "what": row.get("what", ""),
        "task": "vfinal_recheck",
        "severity": "minor",
        "_seed_corrector_msg_id": row.get("seed_corrector_msg_id", ""),
        "_focus_corrector_msg_id": row.get("focus_corrector_msg_id", ""),
        "_t0_msg_id": row.get("t0_msg_id", ""),
        "sources": row.get("sources") or [],
        "seed_whats": row.get("seed_whats") or [],
        "kind": row.get("kind", ""),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recheck-in", default=RECHECK_IN)
    ap.add_argument("--kept-out", default=KEPT_OUT)
    ap.add_argument("--dropped-out", default=DROPPED_OUT)
    ap.add_argument("--dry-run", action="store_true", help="Only print counts; do not call LLM or write outputs")
    ap.add_argument("--no-backup", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    recheck_rows = _load_jsonl(args.recheck_in)
    valid_rows = [r for r in recheck_rows if r.get("valid") is True]
    verify_rows = [_to_verify_row(r) for r in valid_rows]
    missing = [
        r for r in verify_rows
        if not r.get("anchor_msg_id") or not r.get("corrector_msg_id") or not r.get("bot_error_quote")
    ]
    print(
        f"[input] recheck_rows={len(recheck_rows)} valid={len(valid_rows)} "
        f"verify_rows={len(verify_rows)} missing_key_fields={len(missing)}",
        flush=True,
    )
    if args.dry_run:
        return 0

    evs = rl.load_events()
    by_id = {e.get("msg_id"): e for e in evs}
    order_idx = {e.get("msg_id"): i for i, e in enumerate(evs)}

    sc.VERIFY_DROPPED.clear()
    kept = sc.verify_events(verify_rows, by_id, order_idx, evs)
    dropped = list(sc.VERIFY_DROPPED)

    backups = []
    if not args.no_backup:
        for path in (args.kept_out, args.dropped_out):
            backed = _backup(path)
            if backed:
                backups.append(backed)

    _write_jsonl(args.kept_out, kept)
    _write_jsonl(args.dropped_out, dropped)

    drop_dist = Counter(
        (r.get("_drop") or (f"_verify_err:{r.get('_verify_err')}" if r.get("_verify_err") else "")).split("(")[0]
        for r in dropped
    )
    verify_errs = sum(1 for r in kept + dropped if r.get("_verify_err"))
    print(f"[output] kept={len(kept)} dropped={len(dropped)} verify_errs={verify_errs}", flush=True)
    print(f"[drop_dist] {dict(drop_dist)}", flush=True)
    if backups:
        print("[backup] " + " ".join(backups), flush=True)
    print(f"[write] kept -> {args.kept_out}", flush=True)
    print(f"[write] dropped -> {args.dropped_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
