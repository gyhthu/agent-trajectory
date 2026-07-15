#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""session_corrections —— 纠错检测【逐 session 跑】，取代 blob-census 作为纠错来源。

背景（张耀明 2026-07-06 纠偏）：旧法 correction_census.py 是把整段群聊 concat 成
一个几十万字大 blob 喂一次 claude -p，单 pass 召回受限 + blob 越大注意力越稀释
（旧 578 条→35 / 新 1496 条→51，新的反而漏了旧的 28 条真纠错）。张耀明早定的方案是
**按 session 逐个找纠错+对应错误+thread**——每个 session 都被过一遍 = 天然穷举，
没有 blob 召回天花板。本脚本把纠错【来源】从 blob-census 换成逐 session 抽取。

管线（全部复用现成件，DRY）：
  ① rl.load_events()            → 全量 2446 条(active_tail+frozen+chunk，已去重)
  ② route_day.route_day(evs)    → 每条消息 → session（廉莲 #107 路由器，引用线确定性+模型）
  ③ 按 session 分组，每个 session 单独喂 deepseek，用 census 的普查 prompt 逐条枚举纠错
  ④ 汇总去重（按 anchor_msg_id），落 census 同 schema 的 session_corrections.jsonl
下游 rewrite_thread.py 把 CENSUS 指到本文件产物即可，anchor+合并两步不动。

复用：runtime_lane.load_events + route_day.route_day + turn_intent._client/_LLM_MODEL
      + task_stitch._strip_feishu + correction_census.PROMPT（同一普查判据，单一事实源）
"""
import os
import sys
import json
import re
import signal
import time
import threading
import multiprocessing as mp

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import runtime_lane as rl  # noqa: E402
import route_day as rd  # noqa: E402
import task_stitch as ts  # noqa: E402
from turn_intent import _client, _LLM_MODEL  # noqa: E402

DATA = os.environ.get("TASK_TRAJECTORY_DATA", "/opt/shared/data/task-trajectory")
os.makedirs(DATA, exist_ok=True)
ROUTE_CACHE = f"{DATA}/session_route_full.json"       # ② 产物，缓存可复用/可重入
DETECT_CACHE = f"{DATA}/session_corrections_raw.jsonl"  # ③ 逐 session 增量落盘，挂了可重入
OUT = f"{DATA}/session_corrections.jsonl"              # ④ 最终去重产物(census 同 schema)
LLM_TIMEOUT = int(os.environ.get("SESSION_CORRECTIONS_LLM_TIMEOUT", "240"))
LLM_RETRIES = int(os.environ.get("SESSION_CORRECTIONS_LLM_RETRIES", "2"))

# 复用 census 的普查判据（单一事实源），只把「整段对话」换成「本 session 对话」。
# ⚠️ 单一事实源：必须从本目录(_HERE=data_process)的 census 正本(中档护栏)加载，
# 绝不能吃 /opt/shared/data/task-trajectory 下那份未受 git 管理的旧严格孤儿副本——
# 它只认「明确否定=YES」，会把 (b)执行反证/(c)真人质疑触发同线返工 全判 NO（2026-07-08 坐实：
# 中档升级 48b6d9c 一直没在运行时生效，正是对抗核验过杀的头号根因）。_HERE 已在 sys.path[0]，
# 不要再 insert(0, DATA)，否则又劫持回孤儿副本。
from correction_census import (  # noqa: E402
    PROMPT as CENSUS_PROMPT,
    PLACEHOLDER,
    STRONG_CORRECTION,
    VERIFY_PROMPT,
    _is_bot_row,
    _norm,
    validate_events,
)


class _HardTimeout(RuntimeError):
    pass


def _chat_completion(messages, purpose):
    """Call the LLM with a hard timeout so one stuck socket cannot freeze a full run."""
    if os.environ.get("SESSION_CORRECTIONS_FORK_TIMEOUT", "1") == "1":
        return _chat_completion_in_child(messages, purpose)
    can_alarm = hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread()

    def _fire(signum, frame):
        raise _HardTimeout(f"{purpose} timeout after {LLM_TIMEOUT}s")

    last_exc = None
    for attempt in range(1, LLM_RETRIES + 1):
        prev = signal.signal(signal.SIGALRM, _fire) if can_alarm else None
        if can_alarm:
            signal.alarm(LLM_TIMEOUT)
        try:
            resp = _client().chat.completions.create(
                model=_LLM_MODEL,
                messages=messages,
                temperature=0,
                timeout=LLM_TIMEOUT,
            )
            return resp.choices[0].message.content or ""
        except Exception as ex:
            last_exc = ex
            retryable = isinstance(ex, _HardTimeout) or "429" in str(ex) or "RateLimit" in str(ex)
            if attempt < LLM_RETRIES and retryable:
                print(f"    [{purpose} retry {attempt}/{LLM_RETRIES}: {str(ex)[:100]}]", flush=True)
                time.sleep(4 * attempt)
                continue
            raise
        finally:
            if can_alarm:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, prev)
    raise last_exc


def _llm_worker(model, messages, timeout, conn):
    try:
        resp = _client().chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            timeout=timeout,
        )
        conn.send(("ok", resp.choices[0].message.content or ""))
    except Exception as ex:
        conn.send(("err", str(ex)))
    finally:
        conn.close()


def _chat_completion_in_child(messages, purpose):
    ctx = mp.get_context("fork")
    last_exc = None
    for attempt in range(1, LLM_RETRIES + 1):
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        proc = ctx.Process(target=_llm_worker, args=(_LLM_MODEL, messages, LLM_TIMEOUT, child_conn))
        proc.start()
        child_conn.close()
        proc.join(LLM_TIMEOUT)
        if proc.is_alive():
            proc.terminate()
            proc.join(5)
            if proc.is_alive():
                proc.kill()
                proc.join()
            last_exc = _HardTimeout(f"{purpose} timeout after {LLM_TIMEOUT}s")
            retryable = True
        elif parent_conn.poll():
            status, payload = parent_conn.recv()
            if status == "ok":
                parent_conn.close()
                return payload
            last_exc = RuntimeError(payload)
            retryable = "429" in payload or "RateLimit" in payload
        else:
            last_exc = RuntimeError(f"{purpose} worker exited without result, code={proc.exitcode}")
            retryable = True
        parent_conn.close()
        if attempt < LLM_RETRIES and retryable:
            print(f"    [{purpose} retry {attempt}/{LLM_RETRIES}: {str(last_exc)[:100]}]", flush=True)
            time.sleep(4 * attempt)
            continue
        raise last_exc
    raise last_exc


def _render_session(sess_evs):
    """把一个 session 的消息渲染成 [msg_id] role/name: text 的时间序文本。"""
    lines = []
    for e in sess_evs:
        mid = e.get("msg_id", "")
        role = e.get("role", "")
        name = (e.get("name", "") or role)[:10]
        txt = ts._strip_feishu(e.get("text", "") or "")
        lines.append(f"[{mid}] {role}/{name}: {txt}")
    return "\n".join(lines)


def _parse_jsonl(raw):
    out = []
    for line in raw.splitlines():
        line = line.strip().strip("`")
        if not line.startswith("{"):
            m = re.search(r"\{.*\}", line)
            if not m:
                continue
            line = m.group(0)
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


# 单窗字符预算：route_day 的 session 按「任务连续性」归并，长任务会攒成巨型 session
# （实测 S24=959条/45万字，比旧 blob 还大），远超 deepseek 131072 token 上限、且大 blob
# 注意力稀释=正是要治的病。所以巨型 session 必须【内部再按预算分窗】，每窗 bounded 才高召回。
MAX_CHARS = int(os.environ.get("SESSION_CORRECTIONS_MAX_CHARS", "20000"))
# 每窗渲染字符上限。60k 在长 session 里仍会稀释局部纠错；20k 让
# “用户引用某条 bot 回复当场质疑”这类短线索保持在模型注意力中心。
OVERLAP_MSGS = 8         # 窗间按消息重叠，防「纠正句/被纠错」被切到两窗而漏


def _windows(sess_evs):
    """把一个 session 的消息切成若干 <=MAX_CHARS 的窗，相邻窗重叠 OVERLAP_MSGS 条。"""
    wins, cur, cur_chars = [], [], 0
    for e in sess_evs:
        c = len(ts._strip_feishu(e.get("text", "") or ""))
        if cur and cur_chars + c > MAX_CHARS:
            wins.append(cur)
            cur = cur[-OVERLAP_MSGS:] if len(cur) > OVERLAP_MSGS else cur[:]  # 带尾巴重叠
            cur_chars = sum(len(ts._strip_feishu(x.get("text", "") or "")) for x in cur)
        cur.append(e)
        cur_chars += c
    if cur:
        wins.append(cur)
    return wins


def _detect_convo(convo, valid_ids):
    prompt = CENSUS_PROMPT.format(convo=convo)
    raw = _chat_completion([{"role": "user", "content": prompt}], "detect")
    keep = []
    for ev in _parse_jsonl(raw):
        if ev.get("anchor_msg_id") in valid_ids and ev.get("what"):
            keep.append(ev)
    return keep


GUARD_DROPPED = []  # 全局记账：护栏丢弃的事件（张耀明护栏，fail-loud）
VERIFY_DROPPED = []  # 全局记账：对抗核验判掉的事件（corrector 实为确认/追问/无关）

# 对抗核验上下文窗口参数（Fix B，张耀明 2026-07-08）：旧法只喂 anchor+corrector 两条孤立
# 消息、各截 500 字 → 丢中间上下文 + 长回复的否定/错误落在 500 字后判官看不到 = 过杀。
CTX_BEFORE = 3          # 锚点前保留几轮
CTX_AFTER = 2           # 纠正后保留几轮
CTX_MAX_SPAN = 14       # 锚点↔纠正间隔超过这个就省略中段（防巨型 session 窗口爆）
CTX_MSG_CAP = 1200      # 上下文每条正文上限（远高于旧 500 硬截断）
CTX_ANCHOR_CAP = 3500   # 锚点/纠正两条给足额度：错误/否定常落在很靠后的位置

def _render_verify_row(e, aid, cid, rid=None):
    mid = e.get("msg_id", "")
    if mid == aid:
        mark = "▶【bot锚点】"
    elif mid == cid:
        mark = "▶【疑似翻案】"
    elif rid and mid == rid:
        mark = "▶【bot返工】"
    else:
        mark = ""
    role = e.get("role", "")
    name = (e.get("name", "") or role)[:10]
    cap = CTX_ANCHOR_CAP if mid in (aid, cid) else CTX_MSG_CAP
    txt = ts._strip_feishu(e.get("text", "") or "")
    if len(txt) > cap:
        txt = txt[:cap] + f"…〔截断,原长{len(txt)}〕"
    return f"{mark}[{mid}] {role}/{name}: {txt}"

def _verify_context(aid, cid, order_idx, evs, repair_id=None):
    """渲染锚点↔纠正前后一段时间序上下文，标出两条主角。序位取不到返回 None（退回两条）。"""
    ai, ci = order_idx.get(aid), order_idx.get(cid)
    ri = order_idx.get(repair_id) if repair_id else None
    if ai is None or ci is None:
        return None
    endpoints = [ai, ci] + ([ri] if ri is not None else [])
    lo, hi = min(endpoints), max(endpoints)
    start, end = max(0, lo - CTX_BEFORE), min(len(evs) - 1, hi + CTX_AFTER)
    if hi - lo > CTX_MAX_SPAN:  # 跨度过大 → 省略中段，只留两端邻域
        head = range(start, lo + CTX_BEFORE + 1)
        tail = range(hi - CTX_BEFORE, end + 1)
        lines = [_render_verify_row(evs[i], aid, cid, repair_id) for i in head]
        lines.append(f"    …（省略中间 {tail[0] - head[-1] - 1} 轮）…")
        lines += [_render_verify_row(evs[i], aid, cid, repair_id) for i in tail]
        return "\n".join(lines)
    return "\n".join(_render_verify_row(evs[i], aid, cid, repair_id) for i in range(start, end + 1))

def _verify_one(e, by_id, order_idx, evs):
    """第二遍对抗核验：喂锚点↔纠正前后的上下文窗口（含两条主角标注），判 corrector 是否真否定
    anchor。Fix B：不再孤立两条+500 字截断，纠错语义常依赖前后链条，孤立会误杀边缘真纠错。"""
    aid, cid = e.get("anchor_msg_id"), e.get("corrector_msg_id")
    a, c = by_id.get(aid), by_id.get(cid)
    if not a or not c:
        return True  # 缺料交给 validate_events 那关，这里不误杀
    context = _verify_context(aid, cid, order_idx, evs, e.get("_self_correction_msg_id"))
    if context is None:  # 拿不到序位：退回两条，但不做 500 硬截断
        context = "\n".join([_render_verify_row(a, aid, cid), _render_verify_row(c, aid, cid)])
    prompt = VERIFY_PROMPT.format(context=context)
    try:
        raw = _chat_completion([{"role": "user", "content": prompt}], "verify").strip()
        m = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
        v = str(m.get("verdict", "")).upper() == "YES"
        if not v:
            e["_drop"] = f"对抗核验NO({m.get('kind','?')}:{m.get('why','')})"
        return v
    except Exception as ex:
        # 判官挂了不静默放行也不误杀：保留但标记，人工可查（fail-loud）
        e["_verify_err"] = str(ex)[:80]
        return True

def verify_events(events, by_id, order_idx, evs):
    kept = []
    for e in events:
        if _verify_one(e, by_id, order_idx, evs):
            kept.append(e)
        else:
            VERIFY_DROPPED.append(e)
    return kept

REANCHORED = []  # 全局记账：锚点被反查修正的事件（审计用）
CORRECTOR_REANCHORED = []  # 全局记账：C 从 bot 自纠回指到真人触发消息的事件（审计用）

def _resolve_anchor(events, rows):
    """锚点反查修正 + 去重键规范化（张耀明 2026-07-08 坐实：LLM 吐的 anchor_msg_id 常放错——
    安到纠正者身上、指错另一条 bot 消息、或把纠正者的话当成 bot 的错；且它一抖，下游按
    anchor_msg_id 首现去重就整个洗牌 → run 间名单不可复现）。治法=**无条件规范化**：不管 LLM
    原来吐啥，键一律取「bot_error_quote 去本 session bot 消息里反查命中的那条」的 msg_id，
    彻底把 LLM 生成的 id 踢出去重键。
      · 找含 quote 的 bot 消息（whole-session 有效域）：知道 corrector 时间就取早于它的最晚一条
        （紧邻纠正前那条 bot 错误消息），否则按 ts 取最晚匹配——**确定性 tie-break，不吃列表序**。
      · 命中且 != 现锚点 → 重设 anchor_msg_id 并记账（含现锚点本就正确、但规范化到同一条的情形，
        此时 msg_id 不变、不记账）。
      · 无任何 bot 消息含 quote / quote 过短 → 不动，交 validate_events 正确丢弃（错/纠混淆，该杀）。
    幂等：跑第二遍时锚点已是规范化结果，matches/pick 不变 → 不再改动、不重复记账。"""
    by_id = {r["msg_id"]: r for r in rows}
    bot_rows = [r for r in rows if _is_bot_row(r)]
    for e in events:
        q = _norm(e.get("bot_error_quote") or "")
        if len(q) < 4:
            continue
        matches = [r for r in bot_rows if q in _norm(r.get("text", ""))]
        if not matches:
            continue  # quote 不是任何 bot 说的 → 留给护栏丢
        c = by_id.get(e.get("corrector_msg_id"))
        ct = c.get("ts") if c else None
        if ct is not None:
            earlier = [r for r in matches if r.get("ts") is not None and r["ts"] < ct]
            pick = max(earlier, key=lambda r: (r.get("ts") or 0)) if earlier \
                else max(matches, key=lambda r: (r.get("ts") or 0))
        else:
            pick = max(matches, key=lambda r: (r.get("ts") or 0))  # 无 corrector 也确定性取最晚
        if pick["msg_id"] != e.get("anchor_msg_id"):
            e["_reanchored"] = {"from": e.get("anchor_msg_id"), "to": pick["msg_id"]}
            e["anchor_msg_id"] = pick["msg_id"]
            REANCHORED.append(e)
    return events

SELF_CORRECTION = re.compile(
    r"(你说得对|你问到点|你戳|你骂得|你提醒|我上一条|我刚才|我之前|"
    r"我修的方向偏|理解偏|说错|搞错|错了|确实|收回|纠正|误标|漏了|"
    r"坦白|承认|打脸|不成立|逻辑上不成立|不是[^，。；\n]{0,18}而是)"
)
SUMMARY_CONFIRM = re.compile(r"(我梳理一下|我理解|可以理解为|我认这个理由|你先做一版|开始做吧)")


def _is_human_row(r):
    if not r:
        return False
    return not PLACEHOLDER.match(r.get("text", "")) and not _is_bot_row(r)


def _resolve_corrector(events, rows):
    """把 LLM 常见的「C 锚到 bot 自己承认错误那条」回指到真人触发消息。

    边界很窄：只处理当前 corrector 是 bot/sys，且该 bot 文本明显是在自纠/承认前错的情形；
    候选真人必须位于 anchor 与 bot 自纠之间，优先取回复 anchor 的真人，其次取 bot 自纠前
    最近的真人。纯 bot 自纠没有真人触发则继续交给 validate_events 丢弃。
    """
    by_id = {r["msg_id"]: r for r in rows}
    pos = {r["msg_id"]: i for i, r in enumerate(rows)}
    for e in events:
        cid = e.get("corrector_msg_id")
        c = by_id.get(cid)
        if _is_human_row(c):
            continue
        if not c or not SELF_CORRECTION.search(c.get("text", "")):
            continue
        aid = e.get("anchor_msg_id")
        ai, ci = pos.get(aid), pos.get(cid)
        if ai is None or ci is None or ci <= ai:
            continue
        window = rows[ai + 1:ci]
        humans = [r for r in window if _is_human_row(r)]
        if not humans:
            continue
        reply_to_anchor = [r for r in humans if r.get("parent_id") == aid]
        pick = (reply_to_anchor[-1] if reply_to_anchor else humans[-1])
        if SUMMARY_CONFIRM.search(pick.get("text", "")) and not STRONG_CORRECTION.search(pick.get("text", "")):
            continue
        e["_corrector_reanchored"] = {"from": cid, "to": pick["msg_id"]}
        e["_self_correction_msg_id"] = cid
        e["corrector_msg_id"] = pick["msg_id"]
        e["corrector"] = pick.get("name") or pick.get("who") or e.get("corrector", "")
        e["corrector_role"] = "user"
        CORRECTOR_REANCHORED.append(e)
    return events


def _rows_of(sess_evs):
    """把 session 事件构造成 validate_events/_resolve_anchor 用的 rows（单一事实源）。"""
    return [{"msg_id": e.get("msg_id"), "role": e.get("role"),
             "who": e.get("name") or e.get("who"), "name": e.get("name"),
             "text": e.get("text", ""), "ts": e.get("ts"),
             "parent_id": e.get("parent_id")} for e in sess_evs]


def _guard(events, sess_evs, rows=None):
    """张耀明护栏机器校验：anchor 须 bot 实质消息 + bot_error_quote 须命中 anchor 原文。
    用 session 自己的消息当消息集，丢弃凭空造的纠错并记账。
    校验前先过锚点反查修正（_resolve_anchor），救回 LLM 锚点错位的真纠错。
    rows 可由调用方传入复用（巨型 session 窗间去重已构造过，避免重建）。"""
    if rows is None:
        rows = _rows_of(sess_evs)
    events = _resolve_anchor(events, rows)
    events = _resolve_corrector(events, rows)
    kept, dropped = validate_events(events, rows)
    GUARD_DROPPED.extend(dropped)
    return kept


def detect_session(sess_evs):
    """对单个 session 抽全部纠错事件（deepseek）。巨型 session 内部分窗逐窗抽，anchor
    以【整个 session】为有效域（防窗边界丢），跨窗按 anchor_msg_id 去重。返回 census 同 schema。
    出口过张耀明护栏（_guard），挡 idx49/123 式凭空造。"""
    convo = _render_session(sess_evs)
    if len(convo) < 40:  # 太短的 session 不可能有纠错闭环
        return []
    valid_ids = {e.get("msg_id") for e in sess_evs}
    rows = _rows_of(sess_evs)  # 整 session 有效域，窗间去重与护栏共用
    if len(convo) <= MAX_CHARS:
        return _guard(_detect_convo(convo, valid_ids), sess_evs, rows)
    # 巨型 session：内部分窗。窗间去重键必须用【反查后】的稳定 bot msg_id（Fix A，张耀明）——
    # 直接用 LLM 原始 anchor_msg_id 去重，键一抖首现赢家整个洗牌 → S19/S24 名单不可复现。
    # 所以每窗抽完先 _resolve_anchor 规范化，再按规范化后的 anchor_msg_id 去重。
    seen, out = set(), []
    wins = _windows(sess_evs)
    for w in wins:
        evs_w = _resolve_anchor(_detect_convo(_render_session(w), valid_ids), rows)
        for ev in evs_w:
            aid = ev.get("anchor_msg_id")
            if aid and aid not in seen:
                seen.add(aid)
                out.append(ev)
    print(f"    [巨型session分{len(wins)}窗 → {len(out)}纠错]", flush=True)
    return _guard(out, sess_evs, rows)


def _reguard_cached_rows(raw_rows, by_sess):
    """Re-apply current deterministic guards to cached detection rows.

    DETECT_CACHE is intentionally append-only for resumability, but guard logic
    evolves. Cached rows must not bypass newer hard constraints such as
    "corrector must be a real human".
    """
    by_session = {}
    for row in raw_rows:
        if row.get("_empty"):
            continue
        sid = row.get("_session")
        if sid is None:
            GUARD_DROPPED.append({**row, "_drop": "cached row missing _session"})
            continue
        by_session.setdefault(sid, []).append(row)

    kept = []
    for sid, rows_for_session in by_session.items():
        sess_evs = by_sess.get(sid)
        if not sess_evs:
            for row in rows_for_session:
                GUARD_DROPPED.append({**row, "_drop": "cached session missing"})
            continue
        kept.extend(_guard(rows_for_session, sess_evs))
    return kept


def build_route(evs, force=False):
    if os.path.exists(ROUTE_CACHE) and not force:
        cached = json.load(open(ROUTE_CACHE, encoding="utf-8"))
        assign = cached.get("assign") or {}
        max_cached_idx = max((int(k) for k in assign), default=-1)
        if max_cached_idx >= len(evs) - 1:
            print(f"[route] 复用缓存 {ROUTE_CACHE}", flush=True)
            return cached
        print(
            f"[route] 缓存过期：assign覆盖到{max_cached_idx}，当前事件{len(evs)}条，重跑",
            flush=True,
        )
    print(f"[route] 跑 route_day（{len(evs)} 条，model={os.environ.get('ROUTE_MODEL','deepseek')}）…", flush=True)
    res = rd.route_day(evs, respect_reply=True, model=os.environ.get("ROUTE_MODEL", "deepseek"))
    res["event_count"] = len(evs)
    json.dump(res, open(ROUTE_CACHE, "w"), ensure_ascii=False, indent=1)
    print(f"[route] 完成：{res['n_sessions']} 个 session → 缓存 {ROUTE_CACHE}", flush=True)
    return res


def run():
    evs = rl.load_events()
    print(f"[load] 全量事件 {len(evs)} 条", flush=True)
    route = build_route(evs)
    assign = {int(k): v for k, v in route["assign"].items()}  # evs下标 -> session号

    # 按 session 分组（保时间序）
    by_sess = {}
    for i, e in enumerate(evs):
        sid = assign.get(i)
        if sid is None:
            continue
        by_sess.setdefault(sid, []).append(e)
    print(f"[group] {len(by_sess)} 个 session 有消息", flush=True)

    # 断点续跑：已抽过的 session 从 DETECT_CACHE 读回
    done_sessions = set()
    raw_rows = []
    if os.path.exists(DETECT_CACHE):
        for l in open(DETECT_CACHE, encoding="utf-8"):
            if not l.strip():
                continue
            r = json.loads(l)
            raw_rows.append(r)
            done_sessions.add(r["_session"])
        print(f"[resume] 已抽 {len(done_sessions)} 个 session，续跑剩余", flush=True)
        raw_rows = _reguard_cached_rows(raw_rows, by_sess)
        print(f"[resume] 当前护栏重放后保留 {len(raw_rows)} 条缓存纠错", flush=True)

    fcache = open(DETECT_CACHE, "a", encoding="utf-8")
    sids = sorted(by_sess.keys())
    for n, sid in enumerate(sids):
        if sid in done_sessions:
            continue
        try:
            found = detect_session(by_sess[sid])
        except Exception as ex:
            print(f"[③ {n+1}/{len(sids)}] S{sid} ✗ 抽取失败(跳过不落 done): {str(ex)[:120]}", flush=True)
            continue
        # 标记本 session 已处理（哪怕 0 纠错也要落一条哨兵，避免重跑重复抽）
        if not found:
            sentinel = {"_session": sid, "_empty": True}
            fcache.write(json.dumps(sentinel, ensure_ascii=False) + "\n")
            fcache.flush()
            print(f"[③ {n+1}/{len(sids)}] S{sid} ({len(by_sess[sid])}条) → 0 纠错", flush=True)
            done_sessions.add(sid)
            continue
        for ev in found:
            ev["_session"] = sid
            raw_rows.append(ev)
            fcache.write(json.dumps(ev, ensure_ascii=False) + "\n")
        fcache.flush()
        print(f"[③ {n+1}/{len(sids)}] S{sid} ({len(by_sess[sid])}条) → {len(found)} 纠错", flush=True)
    fcache.close()

    # ③.5 第二遍对抗核验（张耀明护栏落地）：机器 check 后再逐条问判官 corrector 真否定否。
    # Fix B：给判官喂锚点↔纠正前后的上下文窗口，需要全量事件的序位索引。
    by_id_all = {e.get("msg_id"): e for e in evs}
    order_idx = {e.get("msg_id"): i for i, e in enumerate(evs)}
    n_before = len(raw_rows)
    raw_rows = verify_events(raw_rows, by_id_all, order_idx, evs)
    print(f"[③.5 对抗核验] {n_before} → {len(raw_rows)}（判掉 {len(VERIFY_DROPPED)}）", flush=True)

    # ④ 去重（按 anchor_msg_id，首现优先）+ 落 census 同 schema
    seen = {}
    for r in raw_rows:
        aid = r.get("anchor_msg_id")
        if not aid or aid in seen:
            continue
        seen[aid] = {
            "anchor_msg_id": aid,
            "bot_error_quote": r.get("bot_error_quote", ""),
            "corrector_msg_id": r.get("corrector_msg_id", ""),
            "corrector": r.get("corrector", ""),
            "corrector_role": r.get("corrector_role", "user"),
            "what": r.get("what", ""),
            "task": r.get("task", ""),
            "severity": r.get("severity", "minor"),
            "_session": r.get("_session"),
        }
        for audit_key in ("_reanchored", "_corrector_reanchored", "_self_correction_msg_id"):
            if audit_key in r:
                seen[aid][audit_key] = r[audit_key]
    with open(OUT, "w", encoding="utf-8") as f:
        for v in seen.values():
            f.write(json.dumps(v, ensure_ascii=False) + "\n")

    # 护栏丢弃存档 + 汇总（张耀明护栏，fail-loud 可审计）
    if GUARD_DROPPED:
        dpath = f"{DATA}/session_corrections_guard_dropped.jsonl"
        with open(dpath, "w", encoding="utf-8") as f:
            for e in GUARD_DROPPED:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        from collections import Counter as _C
        print(f"\n🛡️ 机器护栏丢弃 {len(GUARD_DROPPED)} 条（→ {dpath}）：")
        for reason, c in _C(e["_drop"].split("(")[0] for e in GUARD_DROPPED).most_common():
            print(f"   ×{c}  {reason}")
    if VERIFY_DROPPED:
        vpath = f"{DATA}/session_corrections_verify_dropped.jsonl"
        with open(vpath, "w", encoding="utf-8") as f:
            for e in VERIFY_DROPPED:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        from collections import Counter as _C2
        print(f"\n🔎 对抗核验判掉 {len(VERIFY_DROPPED)} 条（→ {vpath}）：")
        for reason, c in _C2(e["_drop"].split("(")[0] for e in VERIFY_DROPPED).most_common():
            print(f"   ×{c}  {reason}")

    print("\n" + "=" * 60)
    print(f"session 总数        : {len(by_sess)}")
    print(f"逐 session 抽出纠错 : {len(raw_rows)}（含重复 anchor，已过护栏）")
    print(f"★ 去重后纠错事件数  : {len(seen)}  → {OUT}")
    print("=" * 60)
    # 与两份 blob-census 对比
    for name, path in [("旧blob-35", f"{DATA}/correction_census.jsonl"),
                       ("新blob-51", f"{DATA}/correction_census_0703.jsonl")]:
        try:
            bids = {json.loads(l)["anchor_msg_id"] for l in open(path) if l.strip()}
            sids_ = set(seen)
            print(f"  vs {name}: 交集 {len(bids & sids_)} / blob独有 {len(bids - sids_)} / session新增 {len(sids_ - bids)}")
        except Exception as ex:
            print(f"  vs {name}: 比对失败 {ex}")
    return seen


if __name__ == "__main__":
    run()
