#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Distill failure evidence into replay-safe behavior principles (B route).

For each correction (census `what` = the mistake that actually happened), abstract
a one-sentence *behavior principle* that would keep an agent from repeating this
class of error — WITHOUT leaking this instance's concrete answer.

Two layers (张耀明 2026-07-03, both kept but separated for a multi-agent system):
  - general : one sentence, cross-domain, NO domain jargon, NO instance numbers.
              Injected into every agent's system rules.
  - domain  : one sentence, MAY use domain concepts (簇/任务/子需求...), but still
              NO instance answer values. Injected only into the matching专业 agent.

Red line (both layers): the principle must not contain any number-token that
appears in its own source failure evidence. This bans THIS error's specific
answer (186/276/318/13/142...) precisely, without the糙-regex problem of banning
all digits.

The distiller is model-agnostic over the local litellm proxy; default flash to
avoid the shared v4-pro 3/min limit.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

DATA = "/opt/shared/data/task-trajectory"
DEFAULT_SET = f"{DATA}/principle_set.json"

_LLM_BASE = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:4000/v1")
_LLM_KEY = os.environ.get("LLM_API_KEY", "sk-litellm-master-key")
# deepseek-v3.2 by default: available and fast (~4s). v4-flash's shared deployment
# hits 429 deployment-cooldown under any concurrency, so it's unreliable for batches.
_MODEL = os.environ.get("PRINCIPLE_MODEL", "deepseek-v3.2")
_LLM_CLIENT_TIMEOUT = float(os.environ.get("LLM_CLIENT_TIMEOUT", "90"))
_LLM_HARD_TIMEOUT = int(os.environ.get("LLM_HARD_TIMEOUT", "100"))
_LLM_RETRIES = int(os.environ.get("LLM_RETRIES", "5"))


def _client():
    from openai import OpenAI

    # SDK 层不自动重试（重试交给 _chat_with_retry 统一管），
    # 免得某次请求卡死 socket 把整个串行批量永久堵住。
    return OpenAI(api_key=_LLM_KEY, base_url=_LLM_BASE, timeout=_LLM_CLIENT_TIMEOUT, max_retries=0)


# --- red line: number-tokens present in the source failure evidence ----------
# Match arabic runs, chinese numerals, and ratio arrows so "276->318" is caught.
_NUM_TOKEN_RE = re.compile(r"\d+(?:[.\-/→>~]\d+)*|[零一二三四五六七八九十百千万亿]{1,}")


def leaked_number_tokens(text: str) -> set[str]:
    """Number-like tokens that count as this instance's concrete answer."""
    return {m.group(0) for m in _NUM_TOKEN_RE.finditer(text or "")}


def violates_red_line(principle: str, source_what: str) -> list[str]:
    """Return the source number-tokens that leaked into the principle (empty = clean)."""
    if not principle:
        return []
    leaked = leaked_number_tokens(source_what)
    return sorted(t for t in leaked if t and t in principle)


_SYS = """你是行为准则提炼器。输入是某个 AI agent 在历史对话里犯的一次具体错误（纠错记录）。
你的任务：把这次错误抽象成"经验教训"，让 agent 以后遇到同类情境不再犯，但绝不能泄露这一次的具体答案。

铁律：
1. 提炼的是【行为准则/工作态度】（比如"报数前要核实""没被点名别抢答"），不是【这次的正确答案】。
2. 绝对禁止出现这次错误里的任何具体数字、统计结果、具体实体名（如 186、276→318、13、142 这类）。出现即失败。
3. 分两层输出：
   - general（通用层）：一句话，适用于所有 agent，不带任何专业领域词（不出现"簇/切分/子需求/segment"等），是纯粹的通用工作纪律。
   - domain（领域层）：一句话，允许带这个专业领域的概念，把通用纪律落到该领域的具体操作上；只有当这次错误确实涉及专业领域知识时才产出，否则 domain 置为 null。
   - domain_label：领域层对应的领域标签（如"切分""评测""凭据安全""任务建模"），domain 为 null 时也置 null。
4. 两层都必须是"下次怎么做对"的正向指引，不是"这次错在哪"的复述。
5. 每句尽量短、可执行，不超过 60 字。

只输出 JSON：{"general": "...", "domain": "..."|null, "domain_label": "..."|null}"""


def _extract_json(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


class _HardTimeout(Exception):
    pass


def _chat_with_retry(client, model, messages, retries: int | None = None, hard_timeout: int | None = None):
    """Call the proxy, backing off on 429 deployment cooldowns.

    httpx 的 timeout 对偶发的 socket poll 卡死不总是生效（实测批量里单次请求
    可 16min 只 1s CPU 卡在 do_poll）。故再套一层 SIGALRM 硬超时：poll 不返回
    也会被信号打断，转成可重试异常。仅主线程有效。"""
    import time
    import signal
    import threading
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutTimeout

    retries = _LLM_RETRIES if retries is None else retries
    hard_timeout = _LLM_HARD_TIMEOUT if hard_timeout is None else hard_timeout

    # SIGALRM 只在主线程可用；线程池里的 worker 用单线程看门狗强制超时——
    # 否则 SIGALRM 静默失效、httpx 超时又对 socket poll 卡死不总生效，
    # 一个卡死请求会把 worker 永久冻住（实测攒满 6 worker→整体 0%CPU 挂死）。
    can_alarm = (hasattr(signal, "SIGALRM")
                 and threading.current_thread() is threading.main_thread())

    def _fire(signum, frame):
        raise _HardTimeout(f"hard timeout {hard_timeout}s (SIGALRM)")

    def _do_call():
        return client.chat.completions.create(model=model, messages=messages, temperature=0)

    delay = 4.0
    last_exc = None
    for _ in range(retries):
        prev = signal.signal(signal.SIGALRM, _fire) if can_alarm else None
        if can_alarm:
            signal.alarm(hard_timeout)
        try:
            if can_alarm:
                return _do_call()
            # worker 线程：看门狗跑调用，卡死线程直接丢弃（不 wait），本 worker 继续重试。
            _wd = ThreadPoolExecutor(max_workers=1)
            fut = _wd.submit(_do_call)
            try:
                r = fut.result(timeout=hard_timeout)
                _wd.shutdown(wait=False)
                return r
            except _FutTimeout:
                _wd.shutdown(wait=False)
                raise _HardTimeout(f"hard timeout {hard_timeout}s (watchdog)")
        except Exception as exc:  # noqa: BLE001
            # 429 冷却、超时、连接卡死、SIGALRM 硬超时都重试；其它错误直接抛。
            name = type(exc).__name__
            retryable = ("429" in str(exc) or "RateLimit" in name
                         or "Timeout" in name or "Connection" in name or "APITimeout" in name
                         or isinstance(exc, _HardTimeout))
            if not retryable:
                raise
            last_exc = exc
            time.sleep(delay)
            delay = min(delay * 1.6, 30.0)
        finally:
            if can_alarm:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, prev)
    raise last_exc  # type: ignore[misc]


def distill_one(what: str, model: str | None = None, client=None) -> dict:
    """Abstract one failure `what` into general/domain principles + red-line check."""
    client = client or _client()
    model = model or _MODEL
    resp = _chat_with_retry(
        client,
        model,
        [
            {"role": "system", "content": _SYS},
            {"role": "user", "content": f"【这次犯的错】\n{what}\n\n按两层提炼，只输出 JSON。"},
        ],
    )
    raw = resp.choices[0].message.content or ""
    d = _extract_json(raw)
    general = (d.get("general") or "").strip()
    domain = d.get("domain")
    domain = domain.strip() if isinstance(domain, str) and domain.strip() else None
    domain_label = d.get("domain_label")
    domain_label = domain_label.strip() if isinstance(domain_label, str) and domain_label.strip() else None

    guard = {
        "general_leak": violates_red_line(general, what),
        "domain_leak": violates_red_line(domain or "", what),
    }
    # Red line is hard: a leaking layer is dropped, not shipped.
    if guard["general_leak"]:
        general = ""
    if guard["domain_leak"]:
        domain = None
        domain_label = None
    return {
        "source_what": what,
        "general": general,
        "domain": domain,
        "domain_label": domain_label,
        "red_line": guard,
        "raw": raw[:200] if not general else None,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--what", help="Single failure text to distill (ad-hoc test).")
    ap.add_argument("--model", default=_MODEL)
    args = ap.parse_args(argv)
    if args.what:
        print(json.dumps(distill_one(args.what, model=args.model), ensure_ascii=False, indent=2))
        return 0
    ap.error("provide --what for ad-hoc distill; batch wiring lives in pre_instruction_snapshot")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
