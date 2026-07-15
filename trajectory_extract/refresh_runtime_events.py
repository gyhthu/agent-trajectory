#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Refresh lightweight Feishu history events for correction screening.

The correction pipeline needs fresh chat events, but it does not need the
expensive task segmentation/decomposition pass.  This script fetches recent
Feishu history, converts it to the same event schema as ``runtime_lane``, and
writes an append-only de-duplicated cache that ``runtime_lane.load_events()``
merges into its normal sources.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import runtime_lane as rl
import task_stitch as ts


GROUP = rl.GROUP
DATA = Path(rl.DATA)
OUT = Path(rl.RECENT_EVENTS)
OWN_ENV = Path("/home/agent/lian-codex-bot/.env")


def _load_own_feishu_creds(env_path: Path = OWN_ENV) -> None:
    if os.environ.get("FEISHU_APP_ID") and os.environ.get("FEISHU_APP_SECRET"):
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in {"FEISHU_APP_ID", "FEISHU_APP_SECRET"}:
            continue
        os.environ.setdefault(key, value.strip().strip('"').strip("'"))


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _state_watermark() -> int:
    try:
        state = json.loads(Path(rl.STATE).read_text(encoding="utf-8"))
        return int(float(state.get("watermark_ts") or 0))
    except Exception:
        return 0


def refresh(group: str, since: int, until: int, out: Path) -> dict:
    _load_own_feishu_creds()
    existing = _load_jsonl(out)
    fetched = ts.fetch_history(group, since, until)
    by_id = {row.get("msg_id"): row for row in existing if row.get("msg_id")}
    for event in fetched:
        mid = event.get("msg_id")
        if mid:
            by_id[mid] = event
    rows = sorted(by_id.values(), key=lambda e: float(e.get("ts") or 0))
    _write_jsonl(out, rows)
    return {
        "group": group,
        "since": since,
        "until": until,
        "fetched": len(fetched),
        "cached": len(rows),
        "out": str(out),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--group", default=GROUP)
    ap.add_argument("--since", type=int, default=0)
    ap.add_argument("--until", type=int, default=0)
    ap.add_argument("--lookback-hours", type=int, default=24)
    ap.add_argument("--out", default=str(OUT))
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    until = args.until if args.until > 0 else int(time.time())
    since = args.since
    if since <= 0:
        since = max(_state_watermark(), until - args.lookback_hours * 3600)
    result = refresh(args.group, since, until, Path(args.out))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
