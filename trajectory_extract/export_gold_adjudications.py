#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export human review adjudications to stable msg_id gold triples.

The review sheet is intentionally human-facing and labels rows as L1..Ln. This
tool replays the same card generation parameters, maps adjudicated L labels back
to msg_id, and writes machine-readable gold records for regression/evaluation.
"""
import argparse
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import build_gold_review_sheet as review  # noqa: E402
import runtime_lane as rl  # noqa: E402

DATA = "/opt/shared/data/task-trajectory"
DEFAULT_ADJUDICATION = f"{DATA}/gold_review_sheet_new_batch_20260708_adjudication.md"
DEFAULT_REVIEW = f"{DATA}/gold_review_sheet_new_batch_20260708.md"
DEFAULT_OUT = f"{DATA}/gold_review_adjudications.jsonl"

_SECTION_RE = re.compile(r"^### #(\d+)\s*$", re.M)
_CARD_RE = re.compile(r"^## #(\d+)\b.*$", re.M)
_REVIEW_LINE_RE = re.compile(r"^- \**(L\d+)\** 〔([^·]+)·([^〕]+)〕.*$")
_LABEL_RE = re.compile(r"真\s*(t0|B|C)：\s*(L\d+)", re.I)
_ACCEPT_RE = re.compile(r"人判：\s*合格候选")
_SOURCE_MSG_RE = re.compile(
    r"(?:裁决来源|人工裁决来源|人工反馈来源|人工反馈消息|来源消息)"
    r"[:：].*?(om_[A-Za-z0-9_]+)"
)


def adjudication_source_msg_id(text):
    """Return the human adjudication source msg_id declared by the notes."""
    m = _SOURCE_MSG_RE.search(text)
    return m.group(1) if m else ""


def require_adjudication_source_msg_id(text):
    """Fail closed unless the adjudication file points to a human source msg."""
    msg_id = adjudication_source_msg_id(text)
    if not msg_id:
        raise ValueError(
            "adjudication file lacks a human source msg_id; add a line like "
            "`人工裁决来源：msg_id=om_xxx` or pass "
            "--allow-unverified-adjudication for diagnostic exports"
        )
    return msg_id


def parse_adjudication(text):
    """Return card_no -> {decision, labels} parsed from the human notes."""
    starts = [(int(m.group(1)), m.start(), m.end()) for m in _SECTION_RE.finditer(text)]
    out = {}
    for i, (card_no, _start, body_start) in enumerate(starts):
        body_end = starts[i + 1][1] if i + 1 < len(starts) else len(text)
        body = text[body_start:body_end]
        labels = {k.lower(): v for k, v in _LABEL_RE.findall(body)}
        out[card_no] = {
            "decision": "accept" if _ACCEPT_RE.search(body) else "drop_or_boundary",
            "labels": labels,
            "note": " ".join(x.strip() for x in body.splitlines() if x.strip())[:500],
        }
    return out


def _line_text_prefix(line):
    body = line.split("　")[-1].strip()
    body = re.sub(r"…〔\+\d+字〕$", "", body).strip()
    return body


def _event_match_key(event):
    return review._flat(event.get("text"), 10000)


def _resolve_review_line(line, evs):
    m = _REVIEW_LINE_RE.match(line)
    if not m:
        return None
    label, role_tag, name = m.groups()
    prefix = _line_text_prefix(line)
    if not prefix or prefix.startswith("⋯"):
        return None
    matches = []
    for event in evs:
        if name and name != (event.get("name") or ""):
            continue
        flat = _event_match_key(event)
        if flat.startswith(prefix) or prefix in flat:
            matches.append(event)
    if len(matches) != 1:
        return {
            "label": label,
            "status": "ambiguous" if matches else "missing",
            "prefix": prefix,
            "matches": [e.get("msg_id") for e in matches],
            "role_tag": role_tag,
            "name": name,
        }
    event = matches[0]
    return {
        "label": label,
        "status": "ok",
        "msg_id": event.get("msg_id"),
        "role_tag": role_tag,
        "name": name,
        "prefix": prefix,
    }


def _resolve_ambiguous_lines_within_card(labels, unresolved, evs):
    """Resolve duplicate rendered lines by choosing the match nearest this card.

    Human review sheets can contain the same long user instruction in multiple
    cards. Once most lines in a card have unique msg_ids, the duplicate belongs
    to the candidate nearest the resolved lines from that same card.
    """
    idx = {e.get("msg_id"): i for i, e in enumerate(evs)}
    event_by_id = {e.get("msg_id"): e for e in evs}
    changed = True
    while changed:
        changed = False
        resolved_positions = sorted(
            idx[rec["msg_id"]]
            for rec in labels.values()
            if rec.get("msg_id") in idx
        )
        if not resolved_positions:
            break
        lo, hi = resolved_positions[0], resolved_positions[-1]
        remaining = []
        for rec in unresolved:
            if rec.get("status") != "ambiguous" or not rec.get("matches"):
                remaining.append(rec)
                continue
            scored = []
            for msg_id in rec["matches"]:
                pos = idx.get(msg_id)
                if pos is None:
                    continue
                if lo <= pos <= hi:
                    dist = 0
                else:
                    dist = min(abs(pos - lo), abs(pos - hi))
                scored.append((dist, pos, msg_id))
            if not scored:
                remaining.append(rec)
                continue
            scored.sort()
            if len(scored) > 1 and scored[0][0] == scored[1][0]:
                remaining.append(rec)
                continue
            event = event_by_id[scored[0][2]]
            labels[rec["label"]] = {
                "label": rec["label"],
                "status": "ok",
                "msg_id": event.get("msg_id"),
                "role_tag": rec.get("role_tag"),
                "name": rec.get("name"),
                "prefix": rec.get("prefix"),
                "resolved_by": "nearest_card_context",
            }
            changed = True
        unresolved = remaining
    return unresolved


def parse_review_sheet(text, evs):
    """Return card_no -> L label metadata by resolving rendered lines to events."""
    starts = [(int(m.group(1)), m.start(), m.end()) for m in _CARD_RE.finditer(text)]
    out = {}
    for i, (card_no, _start, body_start) in enumerate(starts):
        body_end = starts[i + 1][1] if i + 1 < len(starts) else len(text)
        body = text[body_start:body_end]
        labels = {}
        unresolved = []
        for line in body.splitlines():
            rec = _resolve_review_line(line, evs)
            if not rec:
                continue
            if rec["status"] == "ok":
                labels[rec["label"]] = rec
            else:
                unresolved.append(rec)
        unresolved = _resolve_ambiguous_lines_within_card(labels, unresolved, evs)
        out[card_no] = {"labels": labels, "unresolved": unresolved}
    return out


def _card_line_map(card, evs, idx):
    raw = card["raw"] or {}
    t0 = card["t0"] or {}
    records, _shown = review._transcript_records(
        evs, idx, t0.get("msg_id"), raw.get("anchor_msg_id"), card["cid"]
    )
    return {
        r["label"]: r
        for r in records
        if r.get("kind") == "message" and r.get("label")
    }


def export_gold(n, sources, offset, adjudication_text):
    source_msg_id = adjudication_source_msg_id(adjudication_text)
    adjudications = parse_adjudication(adjudication_text)
    cards, _by_id, idx = review.build(n, sources, offset, exclude_reviewed=False)
    evs = rl.load_events()

    rows = []
    for card_no, card in enumerate(cards, 1):
        adj = adjudications.get(card_no)
        if not adj or adj["decision"] != "accept":
            continue
        line_map = _card_line_map(card, evs, idx)
        missing = [
            name for name in ("t0", "b", "c")
            if adj["labels"].get(name) not in line_map
        ]
        raw = card["raw"] or {}
        t0 = card["t0"] or {}
        row = {
            "card_no": card_no,
            "status": "ok" if not missing else "missing_label",
            "missing": missing,
            "sources": card["srcs"],
            "candidate_corrector_msg_id": card["cid"],
            "machine": {
                "t0_msg_id": t0.get("msg_id"),
                "bot_error_msg_id": raw.get("anchor_msg_id"),
                "corrector_msg_id": card["cid"],
                "what": "; ".join(card.get("whats") or []),
            },
            "gold": {},
            "labels": adj["labels"],
            "note": adj["note"],
            "adjudication_source_msg_id": source_msg_id,
        }
        if not missing:
            row["gold"] = {
                "t0_msg_id": line_map[adj["labels"]["t0"]]["msg_id"],
                "bot_error_msg_id": line_map[adj["labels"]["b"]]["msg_id"],
                "corrector_msg_id": line_map[adj["labels"]["c"]]["msg_id"],
            }
        rows.append(row)
    return rows


def export_gold_from_review(review_text, adjudication_text, evs):
    source_msg_id = adjudication_source_msg_id(adjudication_text)
    adjudications = parse_adjudication(adjudication_text)
    review_cards = parse_review_sheet(review_text, evs)
    rows = []
    for card_no, adj in sorted(adjudications.items()):
        if adj["decision"] != "accept":
            continue
        line_map = review_cards.get(card_no, {}).get("labels", {})
        missing = [
            name for name in ("t0", "b", "c")
            if adj["labels"].get(name) not in line_map
        ]
        row = {
            "card_no": card_no,
            "status": "ok" if not missing else "missing_label",
            "missing": missing,
            "labels": adj["labels"],
            "gold": {},
            "note": adj["note"],
            "adjudication_source_msg_id": source_msg_id,
        }
        if not missing:
            row["gold"] = {
                "t0_msg_id": line_map[adj["labels"]["t0"]]["msg_id"],
                "bot_error_msg_id": line_map[adj["labels"]["b"]]["msg_id"],
                "corrector_msg_id": line_map[adj["labels"]["c"]]["msg_id"],
            }
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--offset", type=int, default=8)
    ap.add_argument("--sources", nargs="*", default=[],
                    help="same filter as build_gold_review_sheet.py")
    ap.add_argument("--review", default=DEFAULT_REVIEW,
                    help="rendered review sheet that the adjudication card numbers refer to")
    ap.add_argument("--adjudication", default=DEFAULT_ADJUDICATION)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--regenerate", action="store_true",
                    help="rebuild cards from current inputs instead of reading --review")
    ap.add_argument("--allow-unverified-adjudication", action="store_true",
                    help="diagnostic only: export even if adjudication lacks a human source msg_id")
    args = ap.parse_args()

    adjudication_text = open(args.adjudication, encoding="utf-8").read()
    if not args.allow_unverified_adjudication:
        require_adjudication_source_msg_id(adjudication_text)
    if args.regenerate:
        rows = export_gold(args.n, args.sources, args.offset, adjudication_text)
    else:
        review_text = open(args.review, encoding="utf-8").read()
        rows = export_gold_from_review(review_text, adjudication_text, rl.load_events())
    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"wrote {len(rows)} adjudicated rows ({ok} ok) -> {args.out}")
    for row in rows:
        print(f"  #{row['card_no']} {row['status']} {row.get('gold') or row['missing']}")


if __name__ == "__main__":
    main()
