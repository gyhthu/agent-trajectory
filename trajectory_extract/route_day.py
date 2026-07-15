#!/usr/bin/env python3
"""日级 session 路由器（廉莲 2026-07-02 提的方案落地版）。

方案（廉莲 #107 原话）：群里所有消息按时间线落盘，从第一条起（第一条=初始 session），
每条**没有引用回复**的新用户请求，自动判它属于哪个已开 session；都不属于就新开。
**有明确引用回复的，无需模型——确定性路由到被引用消息所在的 session。**

两种模式（同一套代码，开关 respect_reply）：
  · respect_reply=True （廉莲规则）：带 parent_id 且父已归属 → 直接continue父的 session，不调模型；
                                     否则调模型判。这是「有回复线无需模型」。
  · respect_reply=False（盲判）    ：无视 parent_id，每条用户请求都调模型。用来量模型
                                     在「没有引用线兜底」时的真实路由质量（对比才有肉）。

只对**用户请求**做路由决策；bot 回复/系统卡片继承其触发用户消息的 session（路由不看它们）。
复用 turn_intent._client() + deepseek，不重造 LLM 客户端。
"""
from __future__ import annotations
import json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import turn_intent as ti
import task_stitch as ts

_MODEL = os.environ.get("ROUTE_MODEL", "deepseek-v4-pro")
_last_call = [0.0]


def _chat(client, **kw):
    """v4-pro 全局限流 3/min 且被本群 live bot 抢占：短间隔多次重试，实时打印。"""
    for attempt in range(40):
        try:
            r = client.chat.completions.create(**kw)
            print(f"    [chat ok attempt={attempt}]", file=sys.stderr, flush=True)
            return r
        except Exception as ex:
            if "429" in str(ex) or "RateLimit" in str(ex):
                print(f"    [429 attempt={attempt}, wait 8s]", file=sys.stderr, flush=True)
                time.sleep(8)
                continue
            raise
    raise RuntimeError("deepseek 429 重试耗尽")

_SYS = (
    "你是群聊会话路由判定器。给你【已开 session 清单】（每条含编号+一句话主题）、"
    "【最近几条消息】、和【当前这条用户请求】。判断：这条用户请求是延续某个已开 session，"
    "还是开一个新 session。\n"
    "判据：session = 一个独立的总目标/交付物，从提出到做完的一整条线，可跨多轮、"
    "中途被打断、隔一段再续。只要还在围绕同一个目标推进（包括追问其中的名词、要求改进、"
    "报错重试、看结果），就算延续同一个 session；只有当用户抛出一个**新的、独立的目标**、"
    "跟所有已开 session 都不是一件事时，才新开。\n"
    '严格输出 JSON：{"session": <已开的编号数字，或字符串"new">, "reason":"≤25字"}。'
    "不要 markdown、不要多余解释。"
)


# 控制命令：操作已排队/已存在任务的元指令（!parallel q1 / !stop q1 / !queue …），
# 不是"真正的用户指令"，不该喂给模型做路由判定（廉莲 2026-07-03 拍板）。
# 它有一个**显式 target instruction**（qK 指向的那条已排队指令），所以它继承的应是
# **qK 那条指令被路由到的 session**（=「q1 的路由」），而不是盲目继承当前活跃 session——
# enqueue 与 !parallel 之间若插了别的指令，last_sid 会漂走，盲继承就会错挂（廉莲 0703 纠偏）。
_CTRL_RE = re.compile(r"^\s*[!/！／]\s*(parallel|stop|cancel|queue|retry|compact|clear|q\d+|p|s)\b", re.I)
# 控制命令显式指向的队列位 qK（`!parallel q1` → 1；`!compact`/`!s` 无显式目标 → None）。
_CTRL_TARGET_RE = re.compile(r"\bq\s*(\d+)\b", re.I)
# 队列入队回执卡（🕒 已入队 qK …）——bot 发的，qK 从这里诞生并绑定到被排队指令的 session。
_ENQUEUE_RE = re.compile(r"已入队\s*q\s*(\d+)")


def _is_control_command(e: dict) -> bool:
    return bool(_CTRL_RE.match(ts._strip_feishu(e.get("text") or "")))


def _ctrl_target_q(e: dict) -> int | None:
    """控制命令显式指向的 qK；无显式目标返回 None。"""
    m = _CTRL_TARGET_RE.search(ts._strip_feishu(e.get("text") or ""))
    return int(m.group(1)) if m else None


def _is_user_request(e: dict) -> bool:
    """只有真·用户指令参与路由。控制命令(!parallel q1 等)虽 role=user，但不是真正的
    用户指令，排除出路由决策——它继承 **其 target instruction(qK) 的路由**（见循环里的
    queue_target 映射），qK 无记录才退回当前活跃 session。"""
    return e.get("role") == "user" and not _is_control_command(e)


def _title_of(session_first_text: str) -> str:
    t = ts._strip_feishu(session_first_text or "").replace("\n", " ").strip()
    t = re.sub(r"@\S+", "", t).strip()
    return t[:40]


def route_day(evs: list[dict], respect_reply: bool = True, model: str = _MODEL,
              client=None) -> dict:
    if client is None:
        client = ti._client()
    idmap = {e["msg_id"]: i for i, e in enumerate(evs) if e.get("msg_id")}
    assign: dict[int, int] = {}          # evs下标 -> session号
    sessions: list[dict] = []            # [{id, title, first_idx}]
    decisions: list[dict] = []           # 每个用户请求一条决策记录

    def open_new(i, e):
        sid = len(sessions) + 1
        sessions.append({"id": sid, "title": _title_of(e["text"]), "first_idx": i})
        assign[i] = sid
        return sid

    last_sid = None  # 最近一条已归属用户消息的 session
    queue_target: dict[int, int] = {}   # qK -> session（qK 那条被排队指令被路由到的 session）
    for i, e in enumerate(evs):
        if not _is_user_request(e):
            # bot/系统消息 + 控制命令：默认继承最近一条用户请求的 session 并登记进 assign。
            # 这样后续用户「引用回复」这条 bot 消息时 parent_i 能在 assign 命中，
            # 确定性引用线路由才生效（用户引用的几乎都是 bot 消息，
            # 不登记 bot 消息 → parent_i 永远 miss → 每条都退回调模型）。
            sess_for_i = last_sid
            # 入队回执「🕒 已入队 qK」：qK 从此绑定到刚被路由的那条指令的 session（=last_sid）。
            mq = _ENQUEUE_RE.search(ts._strip_feishu(e.get("text") or ""))
            if mq and last_sid is not None:
                queue_target[int(mq.group(1))] = last_sid
            # 控制命令：继承 **它的 target instruction(qK)** 的路由，不是盲目继承活跃 session
            # （廉莲 0703 纠偏）。qK 有记录 → 用 queue_target；无记录才退回 last_sid。
            if e.get("role") == "user" and _is_control_command(e):
                tq = _ctrl_target_q(e)
                tgt = queue_target.get(tq) if tq is not None else None
                sess_for_i = tgt if tgt is not None else last_sid
                decisions.append({"idx": i, "mode": "control", "session": sess_for_i,
                                  "target_q": tq,
                                  "target_src": "q-map" if tgt is not None else "active-fallback",
                                  "has_reply": bool(e.get("parent_id")),
                                  "text": e["text"][:60]})
            if sess_for_i is not None and e.get("msg_id"):
                assign[i] = sess_for_i
            continue
        pid = e.get("parent_id")
        parent_i = idmap.get(pid) if pid else None
        det = None
        if respect_reply and parent_i is not None and parent_i in assign:
            det = assign[parent_i]                       # 确定性：路由到父的 session
        if det is not None:
            assign[i] = det
            last_sid = det
            decisions.append({"idx": i, "mode": "reply-det", "session": det,
                              "has_reply": True, "text": e["text"][:60]})
            continue
        if not sessions:
            sid = open_new(i, e)
            last_sid = sid
            decisions.append({"idx": i, "mode": "opener", "session": sid,
                              "has_reply": bool(pid), "text": e["text"][:60]})
            continue
        # 调模型
        prev = []
        for j in range(max(0, i - 6), i):
            pe = evs[j]
            prev.append(f"{pe['name'][:6]}({pe['role']}): "
                        f"{ts._strip_feishu(pe['text'])[:90]}")
        thr = "\n".join(f"  {s['id']}. {s['title']}" for s in sessions)
        um = (f"【已开 session 清单】\n{thr}\n\n"
              f"【最近几条消息（时间升序）】\n" + "\n".join(f"  - {p}" for p in prev) +
              f"\n\n【当前这条用户请求（{e['name'][:6]}）】\n{ts._strip_feishu(e['text'])[:400]}\n\n判定并输出 JSON。")
        resp = _chat(client, model=model, temperature=0,
                     messages=[{"role": "system", "content": _SYS},
                               {"role": "user", "content": um}])
        raw = resp.choices[0].message.content or ""
        m = re.search(r"\{.*\}", raw, re.S)
        sess_out = "new"; reason = raw[:40]
        if m:
            try:
                d = json.loads(m.group(0)); sess_out = d.get("session", "new")
                reason = str(d.get("reason", ""))[:30]
            except Exception:
                pass
        if isinstance(sess_out, str) and sess_out.strip().lower() in ("new", "新"):
            sid = open_new(i, e)
        else:
            try:
                sid = int(sess_out)
                if sid not in {s["id"] for s in sessions}:
                    sid = open_new(i, e)
                else:
                    assign[i] = sid
            except (ValueError, TypeError):
                sid = open_new(i, e)
        last_sid = sid
        decisions.append({"idx": i, "mode": "llm", "session": sid,
                          "has_reply": bool(pid), "reason": reason,
                          "text": e["text"][:60]})
        print(f"  [llm decided] #{i} -> S{sid} ({reason})", file=sys.stderr, flush=True)
    return {"n_sessions": len(sessions), "sessions": sessions,
            "assign": {str(k): v for k, v in assign.items()},
            "decisions": decisions}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--evs", required=True)
    ap.add_argument("--blind", action="store_true", help="无视引用回复,全部调模型")
    ap.add_argument("--model", default=_MODEL)
    ap.add_argument("--out")
    a = ap.parse_args()
    evs = json.load(open(a.evs))
    res = route_day(evs, respect_reply=not a.blind, model=a.model)
    print(json.dumps({"n_sessions": res["n_sessions"],
                      "sessions": res["sessions"]}, ensure_ascii=False, indent=2))
    if a.out:
        json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
        print("wrote", a.out)
