#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Expand an already-rendered gold review sheet without changing card numbers.

Use this after human adjudication says a card is missing t0/B/C context. The
script reads the original Markdown, resolves its L-lines back to msg_id, then
re-renders wider transcript windows around the original guessed t0/B/C marks.
"""
import argparse
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import build_gold_review_sheet as review  # noqa: E402
import export_gold_adjudications as export_gold  # noqa: E402
import runtime_lane as rl  # noqa: E402

DATA = "/opt/shared/data/task-trajectory"
DEFAULT_REVIEW = f"{DATA}/gold_review_sheet_new_batch_20260708.md"
DEFAULT_OUT = f"{DATA}/gold_review_sheet_new_batch_20260708_expanded.md"

_CARD_RE = re.compile(r"^## #(\d+)\b.*$", re.M)
_TITLE_RE = re.compile(r"^## #\d+\s*(.*)$", re.M)


def iter_card_bodies(text):
    starts = [(int(m.group(1)), m.start(), m.end()) for m in _CARD_RE.finditer(text)]
    for i, (card_no, start, body_start) in enumerate(starts):
        body_end = starts[i + 1][1] if i + 1 < len(starts) else len(text)
        header = text[start:body_start].strip()
        yield card_no, header, text[body_start:body_end]


def resolve_marks(body, evs):
    marks = {}
    for line in body.splitlines():
        rec = export_gold._resolve_review_line(line, evs)
        if not rec or rec.get("status") != "ok":
            continue
        for key, marker in (("t0", "«猜t0?»"), ("b", "«猜B?»"), ("c", "«猜C?»")):
            if marker in line:
                marks[key] = rec["msg_id"]
    return marks


def expand_review(text, evs, cards, pad_before, pad_after, cap):
    idx = {e.get("msg_id"): i for i, e in enumerate(evs)}
    wanted = set(cards or [])
    lines = [
        "# 金标人工核对台 · 缺窗扩窗入口",
        "",
        "只重渲染原核对卡的指定卡号，卡号不随候选池排序漂移。填卡尾 `真t0=L__ 真B=L__ 真C=L__`；不合格就写判据号或 review。",
        "",
    ]
    rendered = 0
    for card_no, header, body in iter_card_bodies(text):
        if wanted and card_no not in wanted:
            continue
        marks = resolve_marks(body, evs)
        trans, shown = review._transcript(
            evs, idx,
            marks.get("t0"), marks.get("b"), marks.get("c"),
            pad_before=pad_before, pad_after=pad_after, cap=cap,
        )
        source = ""
        m = _TITLE_RE.match(header)
        if m:
            source = m.group(1)
        heading_suffix = f"　{source}" if source else ""
        lines.append(f"## #{card_no}{heading_suffix}")
        if trans:
            lines.extend(trans)
        else:
            lines.append("- ❓原卡猜测行未能解析回事件流，需回原卡手查。")
        off = [(nm, mid) for nm, mid in (("t0", marks.get("t0")), ("B", marks.get("b")), ("C", marks.get("c"))) if mid and mid not in shown]
        if off:
            lines.append("")
            lines.append("越界猜测反查：" + "　".join(f"{nm}猜 {mid}" for nm, mid in off))
        lines.append("")
        lines.append("**你填**：判定=☐合格 ☐不合格(判据#__) ☐review　｜　真t0=L__　真B=L__　真C=L__")
        lines.append("\n---\n")
        rendered += 1
    return "\n".join(lines), rendered


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--review", default=DEFAULT_REVIEW)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--cards", type=int, nargs="*", default=[1, 2, 3, 4, 13])
    ap.add_argument("--pad-before", type=int, default=40)
    ap.add_argument("--pad-after", type=int, default=12)
    ap.add_argument("--cap", type=int, default=120)
    args = ap.parse_args()

    text = open(args.review, encoding="utf-8").read()
    md, rendered = expand_review(
        text, rl.load_events(), args.cards,
        pad_before=args.pad_before, pad_after=args.pad_after, cap=args.cap,
    )
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"✅ {rendered} 张扩窗卡 → {args.out}")


if __name__ == "__main__":
    main()
