#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare adjudicated gold triples against current correction pipeline outputs.

This is a regression gate, not a detector. It only reads human gold and existing
pipeline artifacts, then classifies whether each gold triple was recalled exactly
or where it disappeared.
"""
import argparse
import json
import os
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import build_gold_review_sheet as review  # noqa: E402

DATA = "/opt/shared/data/task-trajectory"
DEFAULT_GOLD = f"{DATA}/gold_review_adjudications_20260708.jsonl"
DEFAULT_OUT = f"{DATA}/gold_review_pipeline_eval_20260708.json"
GUARDED = f"{DATA}/session_corrections.jsonl"
RAW = f"{DATA}/session_corrections_raw.jsonl"
SNAP = f"{DATA}/pre_instruction_snapshots.jsonl"
POOL = f"{DATA}/user_corrections_pool.jsonl"
VFINAL_KEPT = f"{DATA}/user_corrections_pool_vfinal_recheck_verify_kept.jsonl"


def load_jsonl(path):
    rows = []
    if not path or not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def t0_by_anchor(snapshot_rows):
    out = {}
    for snap in snapshot_rows:
        t0 = snap.get("t0") or {}
        t0id = t0.get("msg_id")
        if not t0id:
            continue
        for corr in snap.get("corrections") or []:
            anchor = corr.get("anchor")
            if anchor and anchor not in out:
                out[anchor] = t0id
    return out


def normalize_pipeline_rows(raw_rows, guarded_rows, snapshots, vfinal_kept_rows=None):
    """Return comparable rows with explicit t0/B/C ids where available."""
    by_anchor = t0_by_anchor(snapshots)
    rows = []
    for source, items in (("raw", raw_rows), ("guarded", guarded_rows)):
        for row in items:
            bid = row.get("anchor_msg_id")
            if not bid:
                continue
            rows.append({
                "source": source,
                "t0_msg_id": by_anchor.get(bid),
                "bot_error_msg_id": bid,
                "corrector_msg_id": row.get("corrector_msg_id"),
                "what": row.get("what"),
                "corrector_role": row.get("corrector_role"),
            })
    for row in vfinal_kept_rows or []:
        bid = row.get("anchor_msg_id") or row.get("bot_error_msg_id")
        cid = row.get("corrector_msg_id")
        if not bid or not cid:
            continue
        rows.append({
            "source": "vfinal_verify_kept",
            "t0_msg_id": row.get("_t0_msg_id") or row.get("t0_msg_id"),
            "bot_error_msg_id": bid,
            "corrector_msg_id": cid,
            "what": row.get("what"),
            "corrector_role": row.get("corrector_role"),
        })
    return rows


def _ids(row):
    return (
        row.get("t0_msg_id"),
        row.get("bot_error_msg_id"),
        row.get("corrector_msg_id"),
    )


def classify_gold_row(gold_row, pipeline_rows, pool_correctors):
    return classify_gold_row_for_stage(gold_row, pipeline_rows, pool_correctors)


def classify_gold_row_for_stage(gold_row, pipeline_rows, pool_correctors):
    gold = gold_row.get("gold") or {}
    gt0, gb, gc = _ids(gold)
    if not (gt0 and gb and gc):
        return {
            "card_no": gold_row.get("card_no"),
            "status": gold_row.get("status"),
            "classification": "not_evaluable",
            "missing": gold_row.get("missing") or [],
        }

    exact = [r for r in pipeline_rows if _ids(r) == (gt0, gb, gc)]
    if exact:
        return {
            "card_no": gold_row.get("card_no"),
            "status": "ok",
            "classification": "exact",
            "matches": exact,
        }

    same_b_c = [r for r in pipeline_rows if r.get("bot_error_msg_id") == gb and r.get("corrector_msg_id") == gc]
    same_b = [r for r in pipeline_rows if r.get("bot_error_msg_id") == gb]
    same_c = [r for r in pipeline_rows if r.get("corrector_msg_id") == gc]
    same_t0 = [r for r in pipeline_rows if r.get("t0_msg_id") == gt0]
    machine = gold_row.get("machine") or {}
    machine_guess = {
        "t0_msg_id": machine.get("t0_msg_id"),
        "bot_error_msg_id": machine.get("bot_error_msg_id"),
        "corrector_msg_id": machine.get("corrector_msg_id"),
    }
    if machine_guess == gold:
        guess_status = "same_as_gold"
    elif any(machine_guess.values()):
        guess_status = "differs_from_gold"
    else:
        guess_status = "absent"

    if same_b_c:
        classification = "t0_mismatch"
        matches = same_b_c
    elif same_b:
        classification = "bot_error_recalled_without_gold_corrector"
        matches = same_b
    elif same_c:
        classification = "corrector_recalled_with_wrong_bot_error"
        matches = same_c
    elif same_t0:
        classification = "same_t0_wrong_pair"
        matches = same_t0
    elif gc in pool_correctors:
        classification = "candidate_pool_only"
        matches = []
    else:
        classification = "missed"
        matches = []

    return {
        "card_no": gold_row.get("card_no"),
        "status": "ok",
        "classification": classification,
        "gold": gold,
        "review_machine_guess": machine_guess,
        "review_machine_guess_status": guess_status,
        "matches": matches[:5],
    }


def evaluate(gold_rows, pipeline_rows, pool_rows):
    pool_correctors = {r.get("corrector_msg_id") for r in pool_rows if r.get("corrector_msg_id")}
    final_rows = [row for row in pipeline_rows if row.get("source") != "raw"]
    details = []
    for row in gold_rows:
        detail = classify_gold_row(row, pipeline_rows, pool_correctors)
        final_detail = classify_gold_row_for_stage(row, final_rows, pool_correctors)
        detail["final_classification"] = final_detail["classification"]
        if final_detail.get("matches"):
            detail["final_matches"] = final_detail["matches"]
        details.append(detail)
    counts = Counter(d["classification"] for d in details)
    final_counts = Counter(d["final_classification"] for d in details)
    status_counts = Counter(row.get("status") for row in gold_rows)
    return {
        "summary": {
            "gold_rows": len(gold_rows),
            "gold_status_counts": dict(sorted(status_counts.items())),
            "pipeline_rows": len(pipeline_rows),
            "classification_counts": dict(sorted(counts.items())),
            "final_classification_counts": dict(sorted(final_counts.items())),
        },
        "details": details,
    }


def load_expected_classifications(path):
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "expected_classifications" in data:
        data = data["expected_classifications"]
    return {int(k): v for k, v in data.items()}


def expectation_failures(report, expected):
    by_card = {int(d["card_no"]): d["classification"] for d in report["details"]}
    failures = []
    for card_no, want in sorted(expected.items()):
        got = by_card.get(card_no)
        if got != want:
            failures.append({"card_no": card_no, "expected": want, "actual": got})
    return failures


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", default=DEFAULT_GOLD)
    ap.add_argument("--raw", default=RAW)
    ap.add_argument("--guarded", default=GUARDED)
    ap.add_argument("--snapshots", default=SNAP)
    ap.add_argument("--pool", default=POOL)
    ap.add_argument("--vfinal-kept", default=VFINAL_KEPT,
                    help="v-final verify kept rows; pass an empty string to disable")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--expect-classifications",
                    help="JSON file mapping card_no to expected classification; exits non-zero on mismatch")
    args = ap.parse_args()

    pipeline_rows = normalize_pipeline_rows(
        load_jsonl(args.raw),
        load_jsonl(args.guarded),
        load_jsonl(args.snapshots),
        load_jsonl(args.vfinal_kept),
    )
    report = evaluate(load_jsonl(args.gold), pipeline_rows, load_jsonl(args.pool))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"wrote {args.out}")
    print(json.dumps(report["summary"], ensure_ascii=False, sort_keys=True))
    for detail in report["details"]:
        print(
            f"  #{detail['card_no']} {detail['classification']}"
            f" final={detail['final_classification']}"
        )
    if args.expect_classifications:
        failures = expectation_failures(report, load_expected_classifications(args.expect_classifications))
        if failures:
            print("classification expectation mismatch:", file=sys.stderr)
            for failure in failures:
                print(
                    f"  #{failure['card_no']} expected {failure['expected']} got {failure['actual']}",
                    file=sys.stderr,
                )
            raise SystemExit(1)


if __name__ == "__main__":
    main()
