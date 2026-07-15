#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trace adjudicated gold triples through correction pipeline artifacts."""
import argparse
import json
import os
from collections import Counter

DATA = "/opt/shared/data/task-trajectory"


STAGES = [
    (
        "session_raw",
        f"{DATA}/session_corrections_raw.jsonl",
        "candidate",
        ("anchor_msg_id", "corrector_msg_id"),
    ),
    (
        "guard_kept",
        f"{DATA}/session_corrections.jsonl",
        "kept",
        ("anchor_msg_id", "corrector_msg_id"),
    ),
    (
        "guard_dropped",
        f"{DATA}/session_corrections_guard_dropped.jsonl",
        "dropped",
        ("anchor_msg_id", "corrector_msg_id"),
    ),
    (
        "session_verify_dropped",
        f"{DATA}/session_corrections_verify_dropped.jsonl",
        "dropped",
        ("anchor_msg_id", "corrector_msg_id"),
    ),
    (
        "pool",
        f"{DATA}/user_corrections_pool.jsonl",
        "candidate",
        ("bot_error_msg_id", "corrector_msg_id"),
    ),
    (
        "session_rerun_pool",
        f"{DATA}/user_corrections_pool_session_rerun_20260710.jsonl",
        "candidate",
        ("bot_error_msg_id", "corrector_msg_id"),
    ),
    (
        "vfinal_recheck",
        f"{DATA}/user_corrections_pool_session_rerun_20260710_vfinal_recheck.jsonl",
        "candidate",
        ("bot_error_msg_id", "corrector_msg_id"),
    ),
    (
        "vfinal_verify_kept",
        f"{DATA}/user_corrections_pool_session_rerun_20260710_vfinal_recheck_verify_kept.jsonl",
        "kept",
        ("anchor_msg_id", "corrector_msg_id"),
    ),
    (
        "vfinal_verify_dropped",
        f"{DATA}/user_corrections_pool_session_rerun_20260710_vfinal_recheck_verify_dropped.jsonl",
        "dropped",
        ("anchor_msg_id", "corrector_msg_id"),
    ),
    (
        "replay_manifest_by_t0",
        f"{DATA}/raw_context_replay_manifest_session_full_only_20260710_by_t0.jsonl",
        "replay",
        (),
    ),
]


def load_jsonl(path):
    rows = []
    if not path or not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def row_t0(row):
    t0 = row.get("t0")
    if isinstance(t0, dict):
        return t0.get("msg_id")
    return row.get("t0_msg_id") or row.get("_t0_msg_id")


def row_b(row, keys):
    for key in keys:
        value = row.get(key)
        if value:
            return value
    return None


def row_c(row, keys):
    for key in keys:
        value = row.get(key)
        if value:
            return value
    return None


def correction_ids_from_manifest(row):
    ids = set()
    for corr in row.get("corrections") or []:
        anchor = corr.get("anchor") or corr.get("bot_error_msg_id") or corr.get("anchor_msg_id")
        corrector = corr.get("corrector_msg_id")
        if anchor:
            ids.add(anchor)
        if corrector:
            ids.add(corrector)
    source = row.get("source_vfinal_row") or {}
    for key in ("bot_error_msg_id", "anchor_msg_id", "corrector_msg_id"):
        if source.get(key):
            ids.add(source[key])
    return ids


def match_stage(row, gold, keys):
    gt0 = gold.get("t0_msg_id")
    gb = gold.get("bot_error_msg_id")
    gc = gold.get("corrector_msg_id")
    ids = {x for x in (gt0, gb, gc) if x}

    if not keys:
        found = set()
        t0 = row_t0(row)
        if t0 in ids:
            found.add(t0)
        found |= correction_ids_from_manifest(row) & ids
        return found

    b = row_b(row, (keys[0], "anchor_msg_id", "bot_error_msg_id"))
    c = row_c(row, (keys[1], "corrector_msg_id"))
    t0 = row_t0(row)
    found = {x for x in (t0, b, c) if x in ids}
    return found


def stage_matches(stage_rows, gold, keys):
    matches = []
    for row in stage_rows:
        found = match_stage(row, gold, keys)
        if not found:
            continue
        matches.append({
            "line": row.get("_line_no"),
            "match": sorted(found),
            "t0": row_t0(row),
            "b": row_b(row, (keys[0], "anchor_msg_id", "bot_error_msg_id")) if keys else None,
            "c": row_c(row, (keys[1], "corrector_msg_id")) if keys else None,
            "drop": row.get("_drop") or row.get("invalid_reason") or "",
        })
    return matches


def match_label(matches, gold):
    gt0 = gold.get("t0_msg_id")
    gb = gold.get("bot_error_msg_id")
    gc = gold.get("corrector_msg_id")
    for match in matches:
        if match.get("t0") == gt0 and match.get("b") == gb and match.get("c") == gc:
            return "exact_t0_b_c"
    for match in matches:
        if match.get("b") == gb and match.get("c") == gc:
            return "exact_b_c"
    for match in matches:
        if match.get("t0") == gt0 and gb in match.get("match", []) and gc in match.get("match", []):
            return "manifest_t0_b_c"
    if any(gb in m.get("match", []) for m in matches) and any(gc in m.get("match", []) for m in matches):
        return "b_and_c_separate"
    if any(gb in m.get("match", []) for m in matches):
        return "b_only"
    if any(gc in m.get("match", []) for m in matches):
        return "c_only"
    if any(gt0 in m.get("match", []) for m in matches):
        return "t0_only"
    return "absent"


def trace(gold_rows, stage_data):
    out = []
    for gold_row in gold_rows:
        gold = gold_row.get("gold") or {}
        rec = {
            "card_no": gold_row.get("card_no"),
            "status": gold_row.get("status"),
            "gold": gold,
            "stages": [],
        }
        for name, path, disposition, keys in STAGES:
            rows = stage_data.get(name, [])
            matches = stage_matches(rows, gold, keys)
            label = match_label(matches, gold)
            rec["stages"].append({
                "stage": name,
                "disposition": disposition,
                "path": path,
                "rows": len(rows),
                "status": label,
                "matches": matches[:5],
                "match_count": len(matches),
            })
        out.append(rec)
    return out


def summarize(records):
    counter = Counter()
    for rec in records:
        final = "absent"
        for stage in rec["stages"]:
            if stage["status"] in {"exact_t0_b_c", "exact_b_c", "manifest_t0_b_c"}:
                final = f"{stage['stage']}:{stage['status']}"
        counter[final] += 1
    return dict(sorted(counter.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", default=f"{DATA}/gold_review_adjudications_target_20260710.jsonl")
    ap.add_argument("--out", help="Write full JSON report to this path")
    args = ap.parse_args()

    gold_rows = load_jsonl(args.gold)
    stage_data = {name: load_jsonl(path) for name, path, _disp, _keys in STAGES}
    records = trace(gold_rows, stage_data)
    report = {
        "gold": args.gold,
        "gold_rows": len(gold_rows),
        "summary": summarize(records),
        "records": records,
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")

    print(json.dumps({"gold_rows": report["gold_rows"], "summary": report["summary"]}, ensure_ascii=False))
    for rec in records:
        print(f"# {rec['card_no']} {rec['gold']}")
        for stage in rec["stages"]:
            if stage["status"] != "absent":
                print(
                    f"  {stage['stage']}: {stage['status']} "
                    f"({stage['disposition']}, matches={stage['match_count']})"
                )


if __name__ == "__main__":
    main()
