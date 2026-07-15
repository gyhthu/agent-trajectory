#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Guardrails for expensive batch LLM jobs.

Manual batch runs must surface their cost shape before the first request:
model, row count, repeat count, and estimated call count. Set
CONFIRM_LLM_BATCH=1 only after that summary has been acknowledged.
"""
from __future__ import annotations

import os
import sys


def require_llm_batch_confirmation(
    *,
    task: str,
    model: str,
    rows: int,
    repeat: int,
    estimated_calls: int,
    extra: str = "",
) -> None:
    """Print a compact preflight summary and stop unless explicitly confirmed."""
    summary = (
        f"[llm-batch-preflight] task={task} model={model} "
        f"rows={rows} repeat={repeat} estimated_calls={estimated_calls}"
    )
    if extra:
        summary = f"{summary} {extra}"
    print(summary, flush=True)

    if os.environ.get("CONFIRM_LLM_BATCH") == "1":
        return

    if sys.stdin.isatty():
        typed = input("Type RUN to start this batch LLM job: ").strip()
        if typed == "RUN":
            return

    raise SystemExit(
        "Batch LLM job not started. Confirm in chat first, then rerun with "
        "CONFIRM_LLM_BATCH=1."
    )
