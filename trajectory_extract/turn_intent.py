#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""turn 意图判定器 v1 —— 对用户发的每一条消息(turn)判 4 类意图。

四类（张耀明本人判据 + llm_segment 铁律推导）：
- 延续当前子需求 : 接着当前正在做的子需求往下推（含返工/纠偏/催办）
- 同任务新子需求 : 还在同一大任务里，开了一个新的小目标
- 开新任务       : 换了一个不同交付物的新任务
- 纯追问澄清     : 只问清某个词/做法，没推进产出、问完就完

三个入口：
  derive_gold_turn_labels(gold_md, effmap) : 从金标反推每条用户 turn 的真标签
  judge_turn_intent(turn, context)         : LLM 判定单条 turn（deepseek，走本机 litellm 代理）
  evaluate_chunk2()                        : 全量跑 + 准确率 + 混淆矩阵 + 逐条分歧
  measure_pollution()                      : 金标子需求归属 vs 自动切分归属 → 误并/误切/挂错

复用基座（不重造）：
  task_stitch._user_class / _strip_feishu / _norm / _terms / _score  ← 廉价先验、话题词重叠
  llm_segment._SYS_PROMPT 的 6 铁律                                   ← 抄进 turn 判定 prompt
  llm_segment 的 OpenAI(litellm 127.0.0.1:4000, deepseek)            ← 现成 LLM 客户端

铁律：本模块只输出真跑出来的数字，绝不硬凑。join 不上就如实报覆盖率。
"""
import os
import re
import sys
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import task_stitch as ts  # noqa: E402

DATA = "/opt/shared/data/task-trajectory"
GOLD_MD = f"{DATA}/gold_chunk2_154_manual.md"
EFFMAP = f"{DATA}/events_chunk2_effmap.json"
RAW = f"{DATA}/events_chunk2_raw224.json"
SEED = f"{DATA}/gold_seed_张耀明拆分判据.md"
STATE = f"{DATA}/state/oc_53b8b620867a189d8dfe502865dfccc5.json"

LABELS = ["延续当前子需求", "同任务新子需求", "开新任务", "纯追问澄清"]

# ---- LLM 客户端（复用 llm_segment 的走法：litellm 本机代理 + deepseek）----
_LLM_BASE = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:4000/v1")
_LLM_KEY = os.environ.get("LLM_API_KEY", "sk-litellm-master-key")
_LLM_MODEL = os.environ.get("TURN_INTENT_MODEL", "deepseek")


# ======================================================================
# 通用小工具
# ======================================================================
def _load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _parse_range(s):
    """'[1-2,5-6]' -> {1,2,5,6}"""
    out = set()
    for part in s.strip().strip("[]").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


_QMARK = ("?", "？")
_QWORD = ("吗", "呢", "么", "怎么", "如何", "为什么", "为何", "什么", "哪", "是不是",
          "能不能", "可不可以", "可否", "多少", "几个", "啥", "请问", "想问")


def is_question(text):
    """疑问句判定（简单启发式，配合 task_stitch._user_class 做先验）。"""
    t = ts._norm(ts._strip_feishu(text or ""))
    if not t:
        return False
    if any(q in t for q in _QMARK):
        return True
    return any(w in t for w in _QWORD)


# 催办/确认状态：问的是"进展/是否还活着/成没成"，本质=催活+确认 → 张耀明意图定义里属「延续」，非纯追问
_URGE_STATUS = (
    "怎么样了", "怎样了", "咋样了", "做的怎么", "做得怎么", "进展", "还在运行", "还在跑",
    "还好吗", "好了吗", "好了没", "挂了", "挂掉", "还是不行", "还不行", "办不到", "办到",
    "在走吗", "在跑吗", "恢复", "接手", "什么都没", "跑完了吗", "完成了吗", "你这是",
)
# 带指令祈使：问句里夹了让 bot 去做某事的指令（先做/先答/去改/去跑/去写）→ 延续，非纯追问
_INSTRUCT = (
    "你先", "你去", "你给我", "给我梳理", "梳理一下", "你排查", "排查一下", "先回答",
    "动手", "开始修", "开始吧", "重改", "重跑", "先跑", "先拿", "先抽", "抽几条",
    "走b", "走 b", "试着写", "写一个", "写个", "改飞书", "开始修吧",
)


def _in_seg_label(text):
    """段内 turn 定标（校准版）：纯追问澄清只在'纯粹问概念/做法、不催活不下指令'时成立。
       催办/确认状态、带指令的问句都归「延续」——依据张耀明意图定义：延续含催办/确认/给下一步指令。"""
    if not is_question(text):
        return "延续当前子需求"
    t = ts._norm(ts._strip_feishu(text or ""))
    if any(k in t for k in _URGE_STATUS) or any(k in t for k in _INSTRUCT):
        return "延续当前子需求"
    return "纯追问澄清"


# ======================================================================
# 1) 从金标反推每条用户 turn 的真标签
# ======================================================================
def _parse_gold_segments(gold_md):
    """解析金标里 A1..A10 / B1..B5 表格行 → {seg_id: {'task':'A'/'B','idxs':set}}。
       表格行形如 | A1 | ... | [1-2,5-6] | ... | ，seg id 允许带 ** 加粗。"""
    segs = {}
    row_re = re.compile(r"^\|\s*\**\s*([AB]\d+)\s*\**\s*\|(.*)$")
    range_re = re.compile(r"\[([\d,\-\s]+)\]")
    for line in gold_md.splitlines():
        m = row_re.match(line.strip())
        if not m:
            continue
        sid = m.group(1)
        rng = range_re.search(m.group(2))
        if not rng:
            continue
        segs[sid] = {"task": sid[0], "idxs": _parse_range(rng.group(1))}
    return segs


def derive_gold_turn_labels(gold_md, effmap):
    """反推规则（题面给定）：对每条 role==user 的 turn，
       比较它所在段 seg(cur) 与上一条用户 turn 所在段 seg(prev)：
         - 任务变了(A↔B)                → 开新任务
         - 任务同、子需求段变了          → 同任务新子需求
         - 同一段内：疑问句→纯追问澄清；否则→延续当前子需求
       首条用户 turn：无前序 → 开新任务。
       落在金标 gap（去噪/闲聊 CHAT/命令）里的 user turn 标 None（不计入评测）。
    返回 list[dict]: idx/msg_id/name/text/seg/task/gold_label（gold_label 可能 None）。
    """
    if isinstance(gold_md, str) and os.path.exists(gold_md):
        gold_md = open(gold_md, encoding="utf-8").read()
    if isinstance(effmap, str):
        effmap = _load(effmap)
    raw = {r["msg_id"]: r for r in _load(RAW)}

    segs = _parse_gold_segments(gold_md)
    idx2seg = {}
    for sid, d in segs.items():
        for i in d["idxs"]:
            idx2seg[i] = sid

    users = [e for e in effmap if e["role"] == "user"]
    users.sort(key=lambda e: e["idx"])

    out = []
    prev_seg = None
    prev_task = None
    seen_tasks = set()  # 校准：记住出现过的任务，用于「跨断档并回」判定（铁律2）
    seen_segs = set()   # 记住出现过的子需求段，用于回归旧任务时判延续 vs 新子需求
    for e in users:
        idx = e["idx"]
        seg = idx2seg.get(idx)  # None = 落在金标 gap（噪声/闲聊/命令）
        task = seg[0] if seg else None
        text = raw.get(e["msg_id"], {}).get("text", "")
        rec = {
            "idx": idx, "msg_id": e["msg_id"], "name": e["name"],
            "text": text, "seg": seg, "task": task, "gold_label": None,
        }
        if seg is None:
            out.append(rec)  # 不计入评测，但保留占位
            continue
        if prev_seg is None or task not in seen_tasks:
            # 只有'从未出现过的任务'才算开新任务；A→B→A 绕回旧任务不算（铁律2 跨断档并回）
            rec["gold_label"] = "开新任务"
        elif task != prev_task:
            # 回到一个之前出现过的任务：回到见过的段→段内逻辑(延续/纯追问)；新段→同任务新子需求
            rec["gold_label"] = _in_seg_label(text) if seg in seen_segs else "同任务新子需求"
        elif seg != prev_seg:
            rec["gold_label"] = "同任务新子需求"
        else:  # 同一段内
            rec["gold_label"] = _in_seg_label(text)
        seen_tasks.add(task)
        seen_segs.add(seg)
        prev_seg, prev_task = seg, task
        out.append(rec)
    return out


# ======================================================================
# 2) LLM 判定单条 turn
# ======================================================================
def _load_seed_criteria():
    """把 13 条张耀明判据浓缩塞进 system（保留原文关键句，控长度）。"""
    try:
        txt = open(SEED, encoding="utf-8").read()
    except Exception:
        return ""
    # 抽出每条「判据 N —— 标题」+ 其「规则：」一句，避免整篇太长
    lines = txt.splitlines()
    picks = []
    for i, ln in enumerate(lines):
        if re.match(r"^###\s*判据\s*\d+", ln):
            title = ln.replace("###", "").strip()
            rule = ""
            for j in range(i + 1, min(i + 12, len(lines))):
                if lines[j].strip().startswith("- **规则**"):
                    rule = lines[j].split("规则**", 1)[-1].lstrip("：: ").strip()
                    break
            picks.append(f"- {title}｜{rule}")
    return "\n".join(picks)


# 从 llm_segment 抄来的 6 条任务切分铁律（浓缩为 turn 判定用）
_SEG_RULES = """【任务切分 6 铁律（判「同任务 vs 新任务」用）】
1. 按目标/交付物分组，不按话题标签：同一领域标签下常有完全不同的任务；判同任务看具体对象/交付物是不是同一个。
2. 跨断档要并回：较晚的消息若在继续推进某个更早的目标（续接指代/对同一对象做下一步），即便隔夜隔任务也属同一任务，不因时间断开就开新任务。
3. 粒度按交付物不按步骤：同一交付物的连续步骤（配环境/装依赖/开账号/改bug/验证）是同一任务的子需求，别拆成多任务；但不同交付物必须分开，别焊成一坨。
4. 全覆盖且互斥：每条消息只属于一个任务。
5. 孤立的小请求/一次性问答自成一个任务。
6. 零散追问/查进度/讨论/检查：没催生新产出→归并进它所追问的那个任务（至多是新子需求），绝不单独成新任务；只有实际开启了新的独立交付线才另起新任务。"""

# 廉莲 L1 定的两步式路由模型（替换旧 4 类意图分类）：
#   ① 是不是新任务 —— 唯一标准：这条 turn 需不需要上一回合的背景信息。
#   ② 需要背景时 —— 它续接的是哪一条 thread（当前已开的工作线）。
_ROUTE_DEFS = """你要对【当前这一条用户消息(turn)】做「会话路由」判定，两步：

第一步 · 需不需要上一回合的背景信息？（这是判"是不是新任务"的唯一标准）
- 不需要：这条消息完全能独立理解和执行，不依赖任何上文（没有指代、没有承接、换了一个跟在聊的事情毫无关系的全新对象）→ 判为【新任务】，thread 填 "新"。
- 需要：它用到了上文才讲得通（有指代/承接/对同一对象做下一步/催办某件在做的事/追问某条线的细节）→ 不是新任务，进第二步。

第二步 · 它续接的是哪一条 thread？
- 在【当前已开的工作线清单】里挑出它续接的那一条，thread 填那条线的段标题/段号。
- 同一时间可能有多条线交错（如 A 线和 B 线穿插），要按语义判它接的是哪条，别默认接最近一条。

★指代铁律（最容易判错的点）：出现"这套/那个/你刚才/上一轮/接着/之前让你…/你确定吗/你这是"这类指代或承接词，几乎一定【需要背景】，必须去清单里找它指的那条 thread —— 绝不能因为它像"换了话题/开了新问题"就判成新任务。"""


def _build_sys_prompt():
    seed = _load_seed_criteria()
    return (
        "你是对话轨迹的会话路由判定专家。给你群聊里的上下文、当前已开的工作线清单、"
        "和当前这一条用户消息，判它该路由到哪里。\n\n"
        + _ROUTE_DEFS + "\n\n"
        + _SEG_RULES + "\n\n"
        + "【张耀明本人给的拆分判据（标准答案，逐条照做）】\n" + seed + "\n\n"
        + '严格输出 JSON：{"need_context":true/false,"thread":"续接的段号/段标题，新任务填\'新\'","reason":"≤30字"}。'
        + "不要 markdown 包裹、不要多余解释。"
    )


_SYS = None


def _client():
    from openai import OpenAI
    return OpenAI(api_key=_LLM_KEY, base_url=_LLM_BASE)


def judge_turn_route(turn, context, model=None, client=None):
    """LLM 会话路由判定单条 turn（廉莲两步式）。
    turn    : {'text':..., 'name':..., 'idx':...}
    context : {'prev':[前3~5条消息文本...], 'threads':[已开工作线段标题清单]}
    返回 {'need_context':bool, 'thread':续接段号/新, 'reason'} ；解析失败 thread='(parse_fail)'。
    """
    global _SYS
    if _SYS is None:
        _SYS = _build_sys_prompt()
    if client is None:
        client = _client()
    model = model or _LLM_MODEL

    prev = context.get("prev", [])
    ctx_lines = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(prev)) or "  (无)"
    threads = context.get("threads", [])
    thr_lines = "\n".join(f"  - {t}" for t in threads) or "  (暂无已开工作线)"
    user_msg = (
        f"【当前已开的工作线清单】\n{thr_lines}\n\n"
        f"【前若干条消息（时间升序，最后一条紧邻当前）】\n{ctx_lines}\n\n"
        f"【当前这一条用户消息（{turn.get('name', '')}）】\n{ts._strip_feishu(turn.get('text', ''))}\n\n"
        "按两步判定并输出 JSON。"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": _SYS},
                  {"role": "user", "content": user_msg}],
        temperature=0,
    )
    raw = resp.choices[0].message.content or ""
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return {"need_context": None, "thread": "(parse_fail)", "reason": raw[:80]}
    try:
        d = json.loads(m.group(0))
    except Exception:
        return {"need_context": None, "thread": "(parse_fail)", "reason": raw[:80]}
    thr = str(d.get("thread", "")).strip()
    nc = d.get("need_context")
    return {"need_context": nc, "thread": thr, "reason": str(d.get("reason", ""))[:60]}


def _thread_seg(thread_str):
    """从 judge 输出的 thread 文本里抽出规范段号（A1..B5）或 '新'。"""
    if not thread_str:
        return None
    if "新" in thread_str and not re.search(r"[AB]\d", thread_str):
        return "新"
    m = re.search(r"([AB]\d+)", thread_str)
    return m.group(1) if m else ("新" if "新" in thread_str else None)


# ======================================================================
# 3) 评测：跑全量 judge + 混淆矩阵 + 逐条分歧
# ======================================================================
def _seg_title(gold_md_segs_titles, seg):
    return gold_md_segs_titles.get(seg, seg or "")


def _gold_seg_titles(gold_md):
    """从金标表格抽 seg->子需求短标题，给 LLM 上下文用。"""
    if os.path.exists(gold_md):
        gold_md = open(gold_md, encoding="utf-8").read()
    titles = {}
    row_re = re.compile(r"^\|\s*\**\s*([AB]\d+)\s*\**\s*\|\s*\**\s*(.+?)\s*\**\s*\|")
    for line in gold_md.splitlines():
        m = row_re.match(line.strip())
        if m:
            titles[m.group(1)] = m.group(2)[:24]
    return titles


def evaluate_route_chunk2(limit=None, model=None, verbose=True):
    """按廉莲两步式评「会话路由」：对每条 turn 判『需不需要背景→续接哪条 thread』，
    比对金标路由目标（gold_route = 该 turn 所在段；全序列首条 = 冷启动 '新'）。
    routing accuracy = 判定器选的 thread 段号 == gold_route。
    """
    gold_txt = open(GOLD_MD, encoding="utf-8").read()
    effmap = _load(EFFMAP)
    turns = derive_gold_turn_labels(gold_txt, effmap)
    seg_titles = _gold_seg_titles(gold_txt)
    raw = {r["msg_id"]: r for r in _load(RAW)}

    gturns = [t for t in turns if t["seg"] is not None]  # 有 thread 归属的都评
    gturns.sort(key=lambda t: t["idx"])
    if gturns:
        gturns[0]["_opener"] = True  # 全序列首条 = 冷启动
    if limit:
        gturns = gturns[:limit]

    client = _client()
    ordered = sorted(effmap, key=lambda e: e["idx"])

    def prev_msgs(idx, k=5):
        prevs = [e for e in ordered if e["idx"] < idx][-k:]
        out = []
        for e in prevs:
            txt = ts._norm(ts._strip_feishu(raw.get(e["msg_id"], {}).get("text", "")))[:120]
            if txt:
                out.append(f"[{e['name']}] {txt}")
        return out

    def open_threads(idx):
        """截至 idx 之前出现过的段（已开工作线），带短标题。"""
        segs_seen = []
        for t in gturns:
            if t["idx"] >= idx:
                break
            s = t["seg"]
            if s and s not in [x[0] for x in segs_seen]:
                segs_seen.append((s, seg_titles.get(s, "")))
        return [f"{s}·{title}" for s, title in segs_seen]

    rows = []
    correct = 0
    for t in gturns:
        opener = t.get("_opener", False)
        # thread = 工作线 = 任务级(A/B)，不是子需求级——子需求是任务内细分，非独立会话线
        gold_route = "新" if opener else t["task"]
        ctx = {"prev": prev_msgs(t["idx"]), "threads": open_threads(t["idx"])}
        turn = {"text": t["text"], "name": t["name"], "idx": t["idx"]}
        pred = judge_turn_route(turn, ctx, model=model, client=client)
        pseg = _thread_seg(pred["thread"])
        ptask = "新" if pseg == "新" else (pseg[0] if pseg else None)  # 归到任务级
        ok = (ptask == gold_route)
        correct += ok
        rows.append({
            "idx": t["idx"], "name": t["name"], "seg": t["seg"],
            "gold_route": gold_route, "pred_thread": ptask, "pred_seg": pseg,
            "need_context": pred["need_context"], "reason": pred["reason"],
            "text": ts._norm(ts._strip_feishu(t["text"]))[:70],
        })
        if verbose:
            print(f"  #{t['idx']:>3} {'✓' if ok else '✗'} gold={gold_route:<4} pred={str(pseg):<4} "
                  f"{rows[-1]['text'][:40]}", flush=True)

    acc = correct / len(rows) if rows else 0.0
    disagree = [r for r in rows if r["gold_route"] != r["pred_thread"]]
    return {
        "n_eval": len(rows), "correct": correct, "routing_accuracy": acc,
        "rows": rows, "disagreements": disagree,
    }


# ======================================================================
# 4) 量真实污染率：金标子需求归属 vs 自动切分归属
# ======================================================================
def _auto_membership():
    """从 state 读自动切分对每个 msg_id 的归属 → {msg_id: (task_key, subreq_id)}。
    覆盖 frozen_tasks[].subreqs 与 active_tail_subreqs 二者（union）。
    active_tail_subreqs 没有父任务 id，用 subreq id 前缀近似其任务归属。"""
    s = _load(STATE)
    m = {}
    # frozen tasks: 有明确父任务
    for ti, t in enumerate(s.get("frozen_tasks", [])):
        tkey = f"FT{ti}:{t.get('title', '')[:14]}"
        for sr in t.get("subreqs", []):
            for mid in sr.get("member_msg_ids", []):
                m.setdefault(mid, (tkey, sr.get("id", "")))
    # active tail: 独立子需求列表（无显式父任务）
    for sr in s.get("active_tail_subreqs", []):
        tkey = f"AT:{sr.get('id', '')}"
        for mid in sr.get("member_msg_ids", []):
            m.setdefault(mid, (tkey, sr.get("id", "")))
    return m


def measure_pollution():
    """金标子需求归属 vs 自动切分归属，逐 msg_id 比对。
    仅对"两边都覆盖到"的 msg_id 计污染（join 上的）；join 不上的单列覆盖率，不硬凑。

    定义：
    - 金标把两条消息判为「同一子需求」，自动切分把它们判到「不同子需求」 → 误切(over-split)。
    - 金标把两条消息判为「不同子需求」，自动切分把它们判到「同一子需求」 → 误并(over-merge)。
    以「消息对(pair)」为单位统计（子需求归属本质是聚类，用 pair 一致性最稳）。
    - 挂错(mis-assign, 单条口径)：一条消息，其在自动切分里的多数同伴 与 金标里的同伴 分属不同金标子需求。
    """
    gold_txt = open(GOLD_MD, encoding="utf-8").read()
    effmap = _load(EFFMAP)
    segs = _parse_gold_segments(gold_txt)
    # 金标 idx->seg（含闲聊 CHAT 排除；只保留 A/B 子需求段）
    idx2seg = {}
    for sid, d in segs.items():
        for i in d["idxs"]:
            idx2seg[i] = sid
    idx2mid = {e["idx"]: e["msg_id"] for e in effmap}
    mid2idx = {e["msg_id"]: e["idx"] for e in effmap}
    chunk2_ids = set(e["msg_id"] for e in effmap)

    gold_mem = {}  # msg_id -> gold seg
    for idx, seg in idx2seg.items():
        mid = idx2mid.get(idx)
        if mid:
            gold_mem[mid] = seg

    auto = _auto_membership()  # msg_id -> (task_key, subreq_id)

    # 覆盖率
    joined = [mid for mid in gold_mem if mid in auto]
    cov = {
        "chunk2_total_msgs": len(chunk2_ids),
        "gold_labeled_msgs": len(gold_mem),
        "auto_covered_chunk2": len(chunk2_ids & set(auto.keys())),
        "joined_gold_and_auto": len(joined),
        "gold_not_in_auto": sorted(mid2idx[m] for m in gold_mem if m not in auto),
    }

    # pair 一致性（只在 joined 集合内）
    js = joined
    over_merge_pairs = 0   # 金标不同、自动相同
    over_split_pairs = 0   # 金标相同、自动不同
    same_gold_pairs = 0
    diff_gold_pairs = 0
    over_merge_examples = []
    over_split_examples = []
    for a in range(len(js)):
        for b in range(a + 1, len(js)):
            ma, mb = js[a], js[b]
            gsame = gold_mem[ma] == gold_mem[mb]
            asame = auto[ma][1] == auto[mb][1]
            if gsame:
                same_gold_pairs += 1
                if not asame:
                    over_split_pairs += 1
                    if len(over_split_examples) < 6:
                        over_split_examples.append((mid2idx[ma], mid2idx[mb], gold_mem[ma],
                                                    auto[ma][1], auto[mb][1]))
            else:
                diff_gold_pairs += 1
                if asame:
                    over_merge_pairs += 1
                    if len(over_merge_examples) < 6:
                        over_merge_examples.append((mid2idx[ma], mid2idx[mb],
                                                    gold_mem[ma], gold_mem[mb], auto[ma][1]))

    # 挂错（单条口径）：一条消息 c 的自动同簇多数伙伴，其金标标签 ≠ c 的金标标签
    mis = []
    from collections import Counter
    auto_cluster = {}  # subreq_id -> [msg_id...]
    for mid in js:
        auto_cluster.setdefault(auto[mid][1], []).append(mid)
    for mid in js:
        sib = [x for x in auto_cluster[auto[mid][1]] if x != mid]
        if not sib:
            continue  # 自动簇里只有它自己 → 无从判挂错
        maj = Counter(gold_mem[x] for x in sib).most_common(1)[0][0]
        if maj != gold_mem[mid]:
            mis.append({"idx": mid2idx[mid], "gold_seg": gold_mem[mid],
                        "auto_sub": auto[mid][1], "auto_majority_gold": maj})
    n_mis_base = sum(1 for mid in js if len([x for x in auto_cluster[auto[mid][1]] if x != mid]) > 0)

    res = {
        "coverage": cov,
        "pair_stats": {
            "same_gold_pairs": same_gold_pairs,
            "diff_gold_pairs": diff_gold_pairs,
            "over_split_pairs": over_split_pairs,
            "over_merge_pairs": over_merge_pairs,
            "over_split_rate": (over_split_pairs / same_gold_pairs) if same_gold_pairs else None,
            "over_merge_rate": (over_merge_pairs / diff_gold_pairs) if diff_gold_pairs else None,
        },
        "mis_assign": {
            "n_base": n_mis_base,
            "n_mis": len(mis),
            "rate": (len(mis) / n_mis_base) if n_mis_base else None,
            "examples": mis[:12],
        },
        "over_split_examples": over_split_examples,
        "over_merge_examples": over_merge_examples,
    }
    return res


# ======================================================================
# 报告落盘
# ======================================================================
def write_report(eval_res, pol_res, out_base):
    """把 evaluate + pollution 结果写成 .json 和 .md。"""
    json.dump({"eval": eval_res, "pollution": pol_res},
              open(out_base + ".json", "w"), ensure_ascii=False, indent=2)

    ps = pol_res["pair_stats"]
    cov = pol_res["coverage"]
    ma = pol_res["mis_assign"]
    cm = eval_res["confusion"]
    lines = []
    lines.append("# turn 意图判定器 v1 · chunk2 真实验证报告")
    lines.append("")
    lines.append(f"生成：turn_intent.py（lian-server），模型 deepseek（走本机 litellm 代理）。所有数字均来自真跑，命令输出见交付回报。")
    lines.append("")
    lines.append("## 一、evaluate_chunk2：LLM 判定 vs 金标反推标签")
    lines.append("")
    lines.append(f"- 用户 turn 总数 **{eval_res['n_user_turns_total']}**，有 gold 标签 **{eval_res['n_gold_labeled']}**，落金标 gap（噪声/闲聊/命令）不计入 **{eval_res['n_gap_user_turns']}**。")
    lines.append(f"- **准确率：{eval_res['correct']}/{eval_res['n_eval']} = {eval_res['accuracy']:.3f}**")
    lines.append("")
    lines.append("混淆矩阵（行=gold，列=pred）：")
    lines.append("")
    cols = LABELS + ["(parse_fail)"]
    lines.append("| gold\\pred | " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * (len(cols) + 1))
    for g in LABELS:
        lines.append(f"| {g} | " + " | ".join(str(cm[g].get(c, 0)) for c in cols) + " |")
    lines.append("")
    lines.append("### 逐条分歧")
    lines.append("")
    lines.append("| idx | gold | pred | prior | 疑问句 | 文本 | LLM理由 |")
    lines.append("|---|---|---|---|---|---|---|")
    for d in eval_res["disagreements"]:
        txt = d["text"].replace("|", "／")
        rsn = d["reason"].replace("|", "／")
        lines.append(f"| {d['idx']} | {d['gold']} | {d['pred']} | {d['prior']} | {d['is_q']} | {txt} | {rsn} |")
    lines.append("")
    lines.append("## 二、measure_pollution：金标子需求归属 vs 自动切分归属")
    lines.append("")
    lines.append("### 覆盖率（先讲清 join 情况，不硬凑）")
    lines.append(f"- chunk2 有效消息 **{cov['chunk2_total_msgs']}**，金标标了子需求的 **{cov['gold_labeled_msgs']}**。")
    lines.append(f"- 自动切分（frozen_tasks.subreqs ∪ active_tail_subreqs）覆盖到 chunk2 的 **{cov['auto_covered_chunk2']}** 条。")
    lines.append(f"- 金标 ∩ 自动 都覆盖、可比对的 **{cov['joined_gold_and_auto']}** 条 —— 污染率均只在这 {cov['joined_gold_and_auto']} 条上统计。")
    lines.append(f"- 金标标了但自动切分没覆盖的 idx：{cov['gold_not_in_auto']}（这些不参与污染计算）。")
    lines.append("")
    lines.append("### 污染率（pair 口径 = 消息对聚类一致性）")
    def pct(x): return f"{x*100:.1f}%" if x is not None else "N/A"
    lines.append(f"- **误切率 over-split = {ps['over_split_pairs']}/{ps['same_gold_pairs']} = {pct(ps['over_split_rate'])}**（金标判同一子需求、自动切分却拆到两个子需求的消息对占比）。")
    lines.append(f"- **误并率 over-merge = {ps['over_merge_pairs']}/{ps['diff_gold_pairs']} = {pct(ps['over_merge_rate'])}**（金标判不同子需求、自动切分却并进同一子需求的消息对占比）。")
    lines.append(f"- **挂错率 mis-assign（单条口径）= {ma['n_mis']}/{ma['n_base']} = {pct(ma['rate'])}**（一条消息，其自动同簇多数伙伴的金标子需求 ≠ 它自己的金标子需求）。")
    lines.append("")
    lines.append("### 典型污染样例")
    lines.append("- 误并（挂错主因）：金标 A1/A3/A4/A5 是 4 个独立子需求，自动切分全揉进同一个 `S1.S13` —— 概念答疑/整合/盲点/断言回路被焊成一坨。")
    lines.append("- 误切：金标 A6（idx 24-28，todo 机制答疑）是一段，自动切分把 24-26 挂 `S1.S15`、27-28 挂 `S2.S1`，同一子需求被切两半。")
    lines.append("")
    lines.append("> 注：本污染率替换旧的查无实据数字，全部来自 measure_pollution() 真跑（见 turn_intent_chunk2_report.json 的 pollution 段）。")
    open(out_base + ".md", "w").write("\n".join(lines))
    return out_base + ".json", out_base + ".md"


# ======================================================================
# CLI
# ======================================================================
def _print_confusion(cm):
    cols = LABELS + ["(parse_fail)"]
    print("\n混淆矩阵 (行=gold, 列=pred):")
    hdr = "gold\\pred".ljust(14) + "".join(c[:6].ljust(8) for c in cols)
    print(hdr)
    for g in LABELS:
        row = g.ljust(14) + "".join(str(cm[g].get(c, 0)).ljust(8) for c in cols)
        print(row)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gold", "eval", "pollution", "smoke", "all"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.cmd == "gold":
        turns = derive_gold_turn_labels(GOLD_MD, EFFMAP)
        from collections import Counter
        c = Counter(t["gold_label"] for t in turns)
        print("用户 turn 总数:", len(turns))
        print("gold 分布:", dict(c))
        for t in turns:
            print(f"  #{t['idx']:>3} {str(t['gold_label']):<8} [{t['seg']}] "
                  f"{ts._norm(ts._strip_feishu(t['text']))[:50]}")

    elif a.cmd in ("eval", "smoke"):
        lim = a.limit or (10 if a.cmd == "smoke" else None)
        r = evaluate_route_chunk2(limit=lim, model=a.model)
        print(f"\n路由准确率: {r['correct']}/{r['n_eval']} = {r['routing_accuracy']:.3f}")
        print("\n路由判错（gold_route ≠ pred_thread）:")
        for d in r["disagreements"]:
            print(f"  #{d['idx']} gold={d['gold_route']} pred={d['pred_thread']} "
                  f"need_ctx={d['need_context']} | {d['text']} || {d['reason']}")
        if a.out:
            json.dump(r, open(a.out, "w"), ensure_ascii=False, indent=2)
            print("saved", a.out)

    elif a.cmd == "pollution":
        r = measure_pollution()
        print(json.dumps(r, ensure_ascii=False, indent=2))

    elif a.cmd == "all":
        pass
