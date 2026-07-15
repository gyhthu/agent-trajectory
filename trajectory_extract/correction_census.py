#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""纠错普查 —— 复用张旭 distill 的取数/清洗/渲染函数（DRY），换成「枚举全部纠错事件」的普查 prompt。

区别于 distill.py：distill 只产**能泛化+可机械检查**的 L1 候选（挑剔、逐日常 0 条），
本脚本目标是**数清群里到底有多少次 bot 被纠正**（不设泛化闸），每条纠错事件带：
  anchor_msg_id(bot错误那条) + corrector(谁纠的) + what(一句话) + task(属于哪摊/交付物)。
下游据此：①总数=纠错消息数；②按 task 归并→找被多次纠错的子需求/任务。

用法：python3 correction_census.py --group oc_xxx --n 1200 --out census.jsonl
"""
import os, sys, json, time, argparse
BOT_EVAL = "/opt/shared/data/bot-eval"
sys.path.insert(0, BOT_EVAL)
from distill import fetch_recent, render_convo, extract_jsonl  # 复用张旭的取数/清洗/渲染，单一事实源

PROMPT = """你是 bot 团队的「纠错事件普查员」。下面是某飞书工作群的真实对话（时间序，每行前缀 [msg_id] role/user_id）。

任务：**逐条枚举**对话里**每一次 bot 先前的说法/做法被推翻**的事件——只要 bot 先给了一个具体陈述/做法、随后**它被真正翻掉**，就算一次纠错事件，全部列出。不做筛选、不管能不能泛化、不管是不是通用规则。

判定一次「纠错事件」（要素能对上才算，宁缺勿造）：
- bot 先给了一个**后来被推翻的具体陈述/做法**（=错误回复，锚点）；这条**锚点必须是 bot 说的一句有实质内容的话**，不能是「👀 已收到」「🗑️ 已撤回」这类占位回执，也不能是卡片系统提示。
- 随后这条锚点**被真正翻掉**——命中下列任一（中档，张耀明 2026-07-08 拍）：
  (a) 有人**明确点破它错了**（「不对/不是这个/理解偏了/应该X不是Y/你搞反了」）；
  (b) 出现**与它相反的事实或执行反证**（用户实测报错、git 显示非仓库、「应该优先用X方案」等——用结果/事实否定了它）；
  (c) **真人质疑/追问触发返工改错**：纠正消息必须是真人发的，且后续返工改掉的错与真人质疑同一条线。

★★ 硬护栏（张耀明 2026-07-08 拍，违反即不许输出这条）★★
1. **anchor 那条 bot 消息里必须真的含有「后来被否定的错误陈述」原话**。你必须在 `bot_error_quote` 字段里**逐字摘录**那句 bot 原话（从 [msg_id] 那行的 text 里抠，≤50字）。**摘不出、或只能靠脑补 bot「其实是这么想的」→ 这条不算纠错，丢弃。**
2. **「用户问 X + bot 答 Y」不是纠错**。用户提一个新问题、bot 正常回答，即使答案里含「不是/不必/其实」等字样，也**不是**在纠正 bot 先前的错——因为 bot 先前根本没就这个说过错话。（反例：用户问「门禁必须谷歌邮箱吗？」bot 答「不必」——bot 之前从没说过「必须」，**不是纠错**。）
3. **corrector_msg_id 必须是真人触发纠正那条消息**。纯 bot 自己道歉/改口/自返工、系统提示、压缩提示、占位回执都不算 C；如果真人追问触发后续 bot 改口，`corrector_msg_id` 仍填真人追问/质疑那条，`corrector_role`=user。
4. **追问/澄清本身不是纠错，但要看它有没有同线触发翻案**。「你的意思是不是…」「那X呢」若锚点那句的**对错没被动过**，仍不算；若这句追问之后 bot 把先前的错改了，只有返工所改的错与追问同一条线才算。**纯设计偏好/需求变更**（「我改主意了」「现在聚焦…」「换个方向」这类没有『先前错被翻』的）不算。
5. **corrector 必须是那条否定消息的真实作者**；不确定就不写这条。
6. **anchor（bot 错误那条）必须在时间上早于 corrector（真人翻案/质疑那条）**。你必须给出 `corrector_msg_id`＝那条**真人翻案/质疑消息**（点破错了／给出相反事实-执行反证／同线触发返工，三者之一）的 msg_id，它**必须排在 anchor 之后**。如果 bot 的这句话是**对某人提问的回答、且它之后没有任何真人翻案/质疑**（它是最后一句、无人推翻），那它就没被纠正过——**丢弃**。（这正是挡「用户问X→bot答Y」被误当纠错的关键：bot 的答案 anchor 在用户提问之后，找不到「更晚的翻案」，就不算纠错。）

对**同一件事**被**来回纠正多次**的，每次各列一条、`task` 字段相同。

只输出 JSONL（每行一个 JSON，无解释、无 markdown 包裹）：
{{"anchor_msg_id":"om_xxx（bot 错误回复那条，必须是 bot 有实质内容的消息）","bot_error_quote":"逐字摘录 anchor 那条 bot 消息里后来被翻掉的原话（≤50字，摘不出就别输出这条）","corrector_msg_id":"om_xxx（真人翻案/质疑那条，必须晚于 anchor）","corrector":"谁翻的(user_id或真人名)","corrector_role":"user","what":"一句话：bot错在哪、被翻成什么","task":"这次纠错属于哪摊（同一交付物/目标用同一简短标签）","severity":"minor|major"}}

对话开始：
{convo}
对话结束。现在逐条输出全部纠错事件的 JSONL："""

# 第二遍对抗式核验（张耀明护栏落地）：机器 check 挡不住 LLM 编造「貌似否定」的 corrector，
# 用一个专门存疑的判官判 corrector 到底有没有否定 anchor。
# B 修（张耀明 2026-07-08）：不再只喂 anchor+corrector 两条孤立消息（丢中间上下文+截断），
# 改喂锚点前后的一段时间序上下文窗口，并标出哪条是锚点、哪条是疑似翻案——纠错语义常依赖
# 「用户追问→bot 犯错→用户纠正」的链条，孤立两条会把边缘真纠错误判成 unrelated/newreq。
VERIFY_PROMPT = """你是纠错事件的**存疑核验官**。下面是一段群聊**上下文**（时间序，每行 [msg_id] role/name: 正文）。
其中两条被特别标注：
- 行首带 ▶【bot锚点】的那条 = bot 先前说的一句话（被指为「后来说错、被翻案」的那句）
- 行首带 ▶【疑似翻案】的那条 = 排在它之后、被指为「翻掉了它」的那条消息
- 行首带 ▶【bot返工】的那条 = 若存在，表示疑似翻案之后 bot 自己承认前错/返工改口的后续消息
其余行是上下文，帮你判断，**别把它们当锚点或翻案**。

判断：**疑似翻案**这条，是不是**真的推翻/否定了【bot锚点】的说法或做法**？
中档判据（张耀明 2026-07-08 拍）——默认判 NO，命中下面任一「真翻案」形态才判 YES：
- (a) **明确否定**：直接说锚点错了（「不对」「不是这个」「应该X不是Y」「你搞反了」）→ YES(negate)
- (b) **相反事实/执行反证**：给出与锚点结论相反的事实或运行结果（「报错了」「git 显示这不是仓库」「实测不是这样」「应该优先用X方案」）→ YES(counter)
- (c) **真人质疑触发同线返工**：这条是真人质疑/追问，且上下文显示后续 bot（尤其 ▶【bot返工】）把锚点错误按这条质疑同线改掉 → YES(user_trigger)
判 NO 的情形：
- 它只是**确认/同意/照做**（「可以」「好的」「能登录了」——即使锚点说「不必X」它答「可以用Y」，那是**印证不是否定**）→ NO(confirm)
- 它是**换了话题的新问题/纯追问/需求变更**，锚点那句的对错**没被动过**（「那X呢」「我改主意了」「现在聚焦…」而锚点的说法原封不动）→ NO(newreq)
- 它跟锚点**根本不是一回事** → NO(unrelated)

关键分界：看的**不是**「它是陈述还是提问」，而是**真人这条消息有没有推翻/同线触发推翻锚点那句话**。纯 bot 自纠不能判 YES。
额外硬边界：
- 用户只是按 bot 要求回填排障信息/日志/命令输出/状态（例如 bot 说“先查/贴输出/跑一下”，用户随后贴“拒绝连接/端口没起/还是不行/命令输出”），但没有指出 bot 先前具体说法错了 → NO(troubleshoot)。
- 用户是在调整方案/范围/评测口径（“那就不跑X了，只看Y”“先不做A，改看B”），而不是推翻某句 bot 事实陈述 → NO(newreq)。

只输出一个 JSON：{{"verdict":"YES|NO","kind":"negate|counter|user_trigger|confirm|newreq|unrelated","why":"≤20字"}}

【上下文】
{context}
输出 JSON："""

import re
PLACEHOLDER = re.compile(r"^\s*(👀|🗑️|🗜️|🔖|✅ 已|⚠️ 本群会话|已收到|已撤回|正在压缩|请升级至最新版本)")
DEMAND_CHANGE = re.compile(
    r"(我改主意|改主意了|现在聚焦|重新聚焦|换个方向|改成|先不[做管]|暂时不[做管]|"
    r"那就不[跑做]|不[跑做][^，。；\n]{0,30}了|只看[^，。；\n]{0,40}|"
    r"拆成[^，。；\n]{0,20}独立任务|独立任务|"
    r"对于以下[^，。；\n]{0,30}(情况|case)[^，。；\n]{0,30}怎么考虑)"
)
EXPLICIT_CORRECTION = re.compile(
    r"(不对|不是这个|不是这样|错了|搞反|理解偏|你说错|你搞错|应该[^，。；\n]{0,24}不是|"
    r"实际上不|实际不是|实际并非|实际没有|实际显示|实测|报错|失败|反证|推翻)"
)
STRONG_CORRECTION = re.compile(
    r"(不对|不是这个|不是这样|错了|搞反|理解偏|你说错|你搞错|应该[^，。；\n]{0,24}不是|"
    r"不是[^，。；\n]{0,24}吗|它不是|实际上不|实际不是|实际并非|实际没有|实际显示|实测|反证|推翻)"
)
DEMAND_CHANGE_OVERRIDE = re.compile(
    r"(不对|不是这个|不是这样|错了|搞反|理解偏|你说错|你搞错|"
    r"实际上不|实际不是|实际并非|实际没有|实际显示|实测|反证|推翻)"
)
TROUBLESHOOTING_REQUEST = re.compile(
    r"(先|再|麻烦|帮忙|请)?.{0,12}(查|跑|试|执行|贴|发|看|确认|验证).{0,24}"
    r"(输出|日志|结果|状态|端口|报错|连接|命令|截图|返回|贴回来|卡在哪)"
)
TROUBLESHOOTING_FEEDBACK = re.compile(
    r"(拒绝连接|connection refused|timed out|timeout|端口|没起来|起不来|还是不行|"
    r"报错|error|traceback|日志|输出|结果|状态|curl|ss |ps |grep|npm |python |bash)"
    , re.I
)
SELF_BOT = "lian-server"

def _is_bot_row(r):
    role = (r.get("role") or "").lower()
    who = (r.get("who") or "") + (r.get("name") or "")
    return role in ("bot", "assistant") or any(k in who for k in ("bot", "lian-", "zym", "codex", "antigravity", "claude("))

def _norm(s):
    return re.sub(r"\s+", "", s or "")

def _is_demand_change_without_correction(text):
    """需求重定向不是纠错；同条明确点破旧说法错时留给后续语义核验。"""
    text = text or ""
    return bool(DEMAND_CHANGE.search(text) and not DEMAND_CHANGE_OVERRIDE.search(text))

def _is_troubleshooting_feedback_without_correction(event, rows, by_id):
    """按排障要求回填输出不是纠错；若同条明确否定 bot 结论，留给语义核验。"""
    c = by_id.get(event.get("corrector_msg_id"))
    if not c:
        return False
    text = c.get("text") or ""
    if STRONG_CORRECTION.search(text) or not TROUBLESHOOTING_FEEDBACK.search(text):
        return False
    aid = event.get("anchor_msg_id")
    cid = event.get("corrector_msg_id")
    pos = {r.get("msg_id"): i for i, r in enumerate(rows)}
    ai, ci = pos.get(aid), pos.get(cid)
    if ai is None or ci is None or ci <= ai:
        return False
    start = max(ai, ci - 4)
    prior_bot_text = "\n".join(
        r.get("text") or ""
        for r in rows[start:ci]
        if _is_bot_row(r)
    )
    return bool(TROUBLESHOOTING_REQUEST.search(prior_bot_text))

def validate_events(events, rows):
    """张耀明护栏（机器可验，fail-loud 记账）：
      ①anchor 必须命中一条 bot 消息；②该消息不是占位回执；
      ③bot_error_quote 必须真的能在 anchor 原文里找到（去空白子串匹配，挡脑补）；
      ④corrector_msg_id 必须命中一条真人消息、且**时间晚于 anchor**（挡「用户问X→bot答Y」：
        bot 答案 anchor 在提问之后、找不到更晚的否定 → 不是纠错）。
      ⑤需求变更/方向重定向不是纠错。
    任何一条不满足 → 丢弃并记原因，不静默兜底。"""
    by_id = {r["msg_id"]: r for r in rows}
    kept, dropped = [], []
    for e in events:
        a = by_id.get(e.get("anchor_msg_id"))
        c = by_id.get(e.get("corrector_msg_id"))
        q = e.get("bot_error_quote") or ""
        if a is None:
            e["_drop"] = "anchor不在消息集"; dropped.append(e); continue
        if not _is_bot_row(a):
            e["_drop"] = f"anchor作者非bot({a.get('role')}/{a.get('who')})"; dropped.append(e); continue
        if PLACEHOLDER.match(a.get("text", "")):
            e["_drop"] = "anchor是占位/系统回执"; dropped.append(e); continue
        if len(_norm(q)) < 4:
            e["_drop"] = "bot_error_quote空/过短"; dropped.append(e); continue
        if _norm(q) not in _norm(a.get("text", "")):
            e["_drop"] = "bot_error_quote不在anchor原文(脑补)"; dropped.append(e); continue
        if c is None:
            e["_drop"] = "corrector_msg_id缺失/不在消息集"; dropped.append(e); continue
        if (e.get("corrector_role") or "user") != "user":
            e["_drop"] = f"corrector_role非user({e.get('corrector_role')})"; dropped.append(e); continue
        if PLACEHOLDER.match(c.get("text", "")):
            e["_drop"] = "corrector是占位/系统回执"; dropped.append(e); continue
        if _is_bot_row(c):
            e["_drop"] = f"corrector作者非真人({c.get('role')}/{c.get('who')})"; dropped.append(e); continue
        if _is_demand_change_without_correction(c.get("text", "")):
            e["_drop"] = "corrector是需求变更/方向重定向"; dropped.append(e); continue
        if _is_troubleshooting_feedback_without_correction(e, rows, by_id):
            e["_drop"] = "corrector是排障信息回填"; dropped.append(e); continue
        at, ct = a.get("ts"), c.get("ts")
        if at is not None and ct is not None and not (ct > at):
            e["_drop"] = "corrector不晚于anchor(bot答在后=非纠错)"; dropped.append(e); continue
        kept.append(e)
    return kept, dropped

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="oc_53b8b620867a189d8dfe502865dfccc5")
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--out", default="correction_census.jsonl")
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args()
    rows = fetch_recent(a.group, a.n)
    convo = render_convo(rows)
    prompt = PROMPT.format(convo=convo)
    # 单一模型配置（张耀明 2026-07-08 拍「模型接口统一配置，不要造成结果不同」）：
    # 本 blob 入口过去用 `claude -p`，与 session_corrections 逐 session 检测/对抗核验用的
    # deepseek(litellm) 不是同一个模型 → 同数据出不同结果。焊到同一个 client + 同一个
    # _LLM_MODEL（turn_intent 单一事实源，env TURN_INTENT_MODEL 一处可切），彻底消除分叉。
    from turn_intent import _client, _LLM_MODEL  # 与检测/核验同源
    print(f"[census] 群 {a.group} 拉到 {len(rows)} 条，convo {len(convo)} 字，喂 {_LLM_MODEL}（litellm 本机代理，与检测/核验同模型）…", flush=True)
    t0 = time.time()
    resp = _client().chat.completions.create(
        model=_LLM_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0)
    raw = resp.choices[0].message.content or ""
    print(f"[census] {_LLM_MODEL} 用时 {int(time.time()-t0)}s，输出 {len(raw)} 字", flush=True)
    events = extract_jsonl(raw)
    # 张耀明护栏：机器校验 anchor+bot_error_quote，丢弃凭空造的
    kept, dropped = validate_events(events, rows)
    with open(a.out, "w", encoding="utf-8") as f:
        for e in kept:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    if dropped:
        dpath = a.out + ".dropped.jsonl"
        with open(dpath, "w", encoding="utf-8") as f:
            for e in dropped:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        from collections import Counter as _C
        dc = _C(e["_drop"] for e in dropped)
        print(f"\n🛡️ 护栏丢弃 {len(dropped)} 条（→ {dpath}）：", flush=True)
        for reason, c in dc.most_common():
            print(f"   ×{c}  {reason}", flush=True)
    events = kept
    # 汇总
    from collections import Counter
    tc = Counter(e.get("task","?") for e in events)
    rc = Counter(e.get("corrector_role","?") for e in events)
    print(f"\n✅ 护栏后纠错事件总数：{len(events)} → {a.out}", flush=True)
    print(f"   纠正来源：{dict(rc)}", flush=True)
    multi = {t:c for t,c in tc.items() if c >= 2}
    print(f"\n=== 被多次纠错的『摊/子需求』（同一 task 出现≥2次）：{len(multi)} 个 ===", flush=True)
    for t,c in sorted(multi.items(), key=lambda x:-x[1]):
        print(f"   ×{c}  {t}", flush=True)

if __name__ == "__main__":
    main()
