#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export a candidate pool from session-level correction detections only."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


DATA = os.environ.get("TASK_TRAJECTORY_DATA", "/opt/shared/data/task-trajectory")
SESSION_CORRECTIONS = f"{DATA}/session_corrections.jsonl"
OUT = f"{DATA}/user_corrections_pool_session_full_only_20260710.jsonl"


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
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def export_pool(corrections: list[dict], source_label: str) -> list[dict]:
    by_corrector: dict[str, dict] = {}
    for row in corrections:
        cid = row.get("corrector_msg_id") or ""
        if not cid:
            continue
        item = by_corrector.setdefault(
            cid,
            {
                "corrector_msg_id": cid,
                "sources": [source_label],
                "whats": [],
                "corrector": row.get("corrector", ""),
            },
        )
        what = row.get("what") or ""
        if what and what not in item["whats"]:
            item["whats"].append(what)
    return list(by_corrector.values())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session-corrections", default=SESSION_CORRECTIONS)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--source-label", default="session_full_only_20260710")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    corrections = _load_jsonl(args.session_corrections)
    pool = export_pool(corrections, args.source_label)
    _write_jsonl(args.out, pool)
    print(
        f"[export] session_corrections={len(corrections)} "
        f"unique_correctors={len(pool)} -> {args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
