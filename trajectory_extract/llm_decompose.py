"""方法②：LLM 语义切子需求（治本，替代 task_stitch 的死规则 decompose_task）。

复用 task_stitch 的历史加载 + 任务切分拿到「单个任务的消息」，
**先用 RegexAnonymizer 脱敏**（secret/app_id/open_id/人名 绝不裸送 LLM），
再把脱敏后的消息喂 deepseek，按「子需求=改写单位」口径语义切分。

死规则的天花板（同主题 fresh 跟进拆碎、噪声乱挂、并行线铺平）靠语义理解治本。
对齐靶子：人工金样例 gold_task_*.md。

用法：
  python3 llm_decompose.py --hist-file /tmp/.../messages.raw.jsonl --group oc_xxx --task-idx 11
  python3 llm_decompose.py --since <epoch> --until <epoch> --group oc_xxx --task-idx 11
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# 复用 task_stitch 的加载/切分（单一事实源，不重写）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import task_stitch as ts  # noqa: E402

# 复用确定性脱敏器（在 data_process 根目录）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from regex_anonymizer import RegexAnonymizer  # noqa: E402

from openai import OpenAI  # noqa: E402

# 本机 litellm 代理：免配 key（master key 固定），deepseek 不出本机外网
_LLM_BASE = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:4000/v1")
_LLM_KEY = os.environ.get("LLM_API_KEY", "sk-litellm-master-key")
_LLM_MODEL = os.environ.get("LLM_DECOMPOSE_MODEL", "deepseek")
# 大块路径专用模型：deepseek 对 ~100 条大块高方差（坏JSON/塌成1/碎成51 乱跳，temp=0 压不住），
# v4-pro 稳定落在合理粒度。故小任务(≤big_threshold)留 deepseek 省钱，大块切块时换 v4-pro 换稳定。
# 证据：107条块 deepseek 塌成1/碎成51，v4-pro 稳定7条且识别交织子需求+返工状态（2026-06-30 诊断）。
_LLM_BIG_MODEL = os.environ.get("LLM_DECOMPOSE_BIG_MODEL", "deepseek-v4-pro")

# 确定性角色标注（复用 task_stitch._user_class 的关键词分类，把高精度的"非fresh"信号浮到
# 喂模型的清单里）。只浮 rework/correction/continuation 三类——都由多字关键词确定性触发、误报低，
# 且与既有规则同向（返工/纠偏=归回所属线程、承接=不开新段）。**刻意不标**：
#   · fresh：默认兜底类、含大量长问句，标了会诱导过度开段；
#   · filler：含 hi/你好/??? 这类"验证探针"，prompt 明确要求把它们归进所验证的调试段、不当噪声，
#     打 [催促] 会与那条规则打架（暗示丢弃），故不标。
# 只把"这条是返工/纠偏/承接"的确定性事实告诉模型，开/并段仍由它按切分铁律判。
# 治 2026-06-30 金标暴露的 S5 偏粗：G6 回退返工链（全是 rework/correction）被埋进大段，
# 标记一浮出来，模型几乎不可能再把它并进主线段。
_CLS_TAG = {"rework": "[返工]", "correction": "[纠偏]", "continuation": "[承接]"}
# 消融开关：DECOMPOSE_ROLE_TAGS=1 开标注注入。**默认关**——2026-06-30 金标验证该版不稳定、
# 引入过切（chunk1 两轮 G6 各对一次/各错一次，跟基线同为 1/2，且命中那次把 G6 过切成3段；
# zym 9段过切 vs 基线7段贴金标），未达标不进基线行为；代码留作后续迭代靶子，须显式 =1 才启用。
_ROLE_TAGS_ON = os.environ.get("DECOMPOSE_ROLE_TAGS", "0") != "0"
# 输入格式图例（放用户轮，不动 _SYS_PROMPT 的切分规则）：只解释标注含义，不规定怎么拆。
_TAG_LEGEND = ("（消息里的 [返工]=人说还没好/又出问题、[纠偏]=人说方向错了、"
               "[承接]=把当前事往下推一步，是代码确定性预判的消息角色，"
               "仅供你判断线程归属/返工边界时参考，子需求边界仍按上面的切分铁律裁定）\n")

_SYS_PROMPT = """你是对话轨迹分析专家。给你一个「任务」内按时间排序的多方对话消息（已脱敏），\
把它切成若干「子需求」。

定义：
- 任务 = 一条训练样本：用户一个总目标，agent 多轮工具调用 + 被纠正 → 最终做对。
- 子需求 = 改写单位 = 一个「小目标 → 走偏 → 被纠 → 做对」的完整闭环。\
**同一件事、同一条解决路径上的所有跟进/返工/纠偏都归同一个子需求**，不论中间插了多少别的话、隔了多久。
- 噪声（ok/收到/???/单纯催促、与任务无关的纯寒暄）不单独成子需求。\
**但要分清"验证探针"不是噪声**：当一句 hi/你好 是用来"探 bot 还响不响应、回复还带不带 bug"、\
其后紧跟着 bot 的回复时，这一对是某个调试段的"观察"步（看修没修好、通没通），\
要**归进它所验证的那个调试/排障段**，不许当噪声丢——删了会让排障段看不出"到底通没通"而上下文突兀。\
同一段里这种探针对要么全收、要么全不收，绝不只收一对漏其余。

切分铁律（按重要性，逐条照做）：
1. 【合并】把「同一件事、同一方案内的多轮追问/调试/返工」合并成 1 个子需求，绝不按提问次数拆碎。\
例：「为什么有X」「还会有X吗」「那怎么改」「仍然有X」是同一个 bug 的反复收窄 = 1 个子需求。\
**同一个 bug/报错的不同症状、再次复现、后续修复都算同一子需求**，不许按"现象变体"拆碎\
（例：「回复开头有@标签」和「@标签后面还有残片」是同一个转义 bug 的两种表现 = 1 段，不拆成两段）；\
只有当**解决方案/路径本身被换掉**（见铁律2）才拆。
2. 【方案切换必拆 ★本任务最易漏】当 agent 先走了一条错的/绕远的解决路径，\
后来**被提醒、或自己发现"其实已有现成可用的方法/更轻量的方案"，于是推翻原方案、回退到既有方案重做**——\
这是一个**独立**的子需求（type=收尾或旁支、status=被纠偏），绝不能并进它所推翻的那个主线段。\
判据：解决"路径/方案"本身被换掉了（不是同一方案里继续调 bug），就必须新开一段。\
可执行触发信号——只要某条消息出现下列任一，且它**推翻/替换的是前面已经给出的某套方案**，就从这条起新开一段：\
"停掉/回退/推翻/不用那套/改用XX方案/是我把它讲复杂了/其实不需要这么复杂/我接手/换成"；\
尤其当**主导方在这里发生切换**（如从 lian-codex 接手到 claude），回退动作几乎一定是新子需求的起点。\
这种"走偏→被点醒→想起已有方法→回退做对"的闭环是最有价值的改写样本，漏掉=丢核心信号。\
★★【反幻觉保真·最高优先】"走偏→被提醒→回退"必须切成**两个独立子需求**、用 ↺返工 edge 相连，绝不并成一段：\
(A) 走偏段：agent 先上了复杂/绕远方案（体现它当时**没想起已沉淀的可用方法**），dominant=走偏那方，status=有返工；\
(B) 回退段：被外部提醒后才回退到既有方法，lead_quote 取**那句外部提醒的原话**（不是 agent 自己的话），\
status 必须=被纠偏，dominant=接手回退那方，title 体现"被提醒后回退"，member_idx **从那句外部提醒起**、绝不含前面的走偏消息。\
(C) 必加 edge：{"from":"<走偏段>","to":"<回退段>","type":"↺返工"}。\
把 A、B 并成一段会让训练数据看起来像 agent 一开始就知道轻方案，训出模型**无中生有地声称自己记得某个沉淀方法**——这是最严重的污染，代码侧会强制校验并拆分。
3. 【复盘/收尾必独立】事后复盘、流程总结、"下次怎么优化"这类消息，\
即使主题和前面主线相关，也**单独成段**（type=复盘），绝不揉进它复盘的那个主线段。
4. 【全覆盖且互斥·一条消息恰好归一段】除真噪声外，**每一条非噪声消息必须、且只能归入一个子需求的 member_idx**。\
输出前必做自检：把所有 subreqs 的 member_idx 拼起来——\
(a) 不许有任何编号出现在两个或以上的段里（**禁止重叠**，一条消息归属唯一）；\
(b) 全部非噪声编号都要被覆盖（不漏）。\
若一条消息看似同属两段，按"它实际在推进哪件事"归唯一一段，绝不两边都放。
5. 【配置/准备 vs 排障/调试必分】"把东西配好/装好/接好"（静态配置：填密钥、写.env、发版、装依赖、绑权限）\
和"配好后东西不工作、排查到修好"（动态排障：@没反应、报错、查日志/服务/链路、重启验证）\
是**两件不同的事**，即使前后紧挨、共用一个大目标也**分成两个子需求**——\
前者目标是"配对/就绪"，后者目标是"找出为什么不通、修到通"。\
判据：出现"没反应/不工作/报错/还是不行 → 查 X → 重启/改 Y → 通了"这种排障弧，就从排障起点新开一段。
6. 【并行线分开】时间上交错的并行线（A 问到一半插入 B 的讨论）分成不同子需求，各自连续。\
消息行尾的 `↩回复#N` 表示该消息**精确回复了 #N 那条**——优先用它判断"这条接的是哪条线/属于哪个子需求"，\
比单纯按相邻顺序更可靠；交错对话里靠它把回复挂回正确的线程。

每个子需求输出字段：
- id: "S1","S2"... 按起始时间排序
- title: ≤16字，明确体现这件事（如「@转义bug修复」「回退到轻量隔离方案」）
- lead_quote: 触发这个子需求的那句人话原文（脱敏后的原文，截≤40字）
- lead_role: 触发者角色
- dominant: 主导处理方，**必须是具体的 bot/人名**（从消息里取，如 claude(lian-server)/lian-codex/zym-antigravity/karen），不许只写"bot"
- type: 主线 / 旁支 / 复盘 / 收尾
- status: 一遍过 / 有返工 / 被纠偏 / 未终结
  （未终结 = **只用于全程时间最后那一件还没收尾的事**：末尾是 bot 回合用户还没确认、或用户最后提了 bot 还没回。
   ⚠️ 中段/开头早已被对话越过的事**绝不能标未终结**（对话在它之后还继续=它已了结）；单条消息的开场动作=一遍过。
   绝不要给没收尾的活标 一遍过——那是虚报成功。`未终结` 最终由代码按任务终结判定统一裁定兜底。）
- member_idx: 属于这个子需求的消息编号列表（用输入里的 # 编号）

另输出 edges（子需求间关系）：[{"from":"S1","to":"S4","type":"⊸顺序依赖|∥并行|↺返工|↦派生"}]

**task_goal（任务总目标）用「终态/纠偏后」的结论写**：执行者中途被推翻的误判（如把一段\
"误报→被纠偏→其实是别的原因"的活说成"验证/处理某真问题"）绝不写进总目标，只写这个任务\
最终真正在交付的东西。与 status 由终态裁定同理——别让目标文字复述已被推翻的中途错判。

**delivery（任务交付态·任务级，判 bot 在这件事上最终到底交付了没）**，三选一：
- 已交付：bot 最终给出了实质交付（答案/产物/结论/"已提交/已重启/搞定"等带实际内容的收尾）；
- 仅占位：bot 只留了未兑现的承诺或占位（"我去跑/稍后回填/开始处理/收到"），全程**没出现**对应的实质交付；
- 无交付：任务里 bot 根本没有交付动作（用户请求悬着没人接，或全程只有用户单方发言）。
delivery **只看 bot 这边交付没，不管用户确认没**（确认与否由别处判）；末尾占位但前文已实际交付→已交付。

严格输出 JSON：{"task_goal":"一句话总目标","delivery":"已交付|仅占位|无交付","subreqs":[...],"edges":[...]}。不要 markdown 包裹、不要解释。"""


def build_transcript(cluster, anon: RegexAnonymizer):
    """把一个任务的消息列表 → 脱敏后的带编号清单（供 LLM 读），返回 (lines, meta)。
    丢纯噪声(ack/卡片JSON/压缩提示)，保留短 filler 让 LLM 自己判断忽略。"""
    rows, meta = [], []
    msgid2idx = {}  # 飞书 msg_id → 本清单 #序号（父消息时间在前，遍历到子消息时已登记）
    for e in cluster:
        if ts._is_noise(e["text"]):
            continue
        txt = ts._strip_feishu(e["text"]).strip()
        if not txt:
            continue
        txt = anon.anonymize_text(txt)[:300]  # 脱敏 + 截断控 token
        i = len(meta) + 1
        if e.get("msg_id"):
            msgid2idx[e["msg_id"]] = i
        hhmm = time.strftime("%H:%M:%S", time.localtime(e["ts"]))
        role = "用户" if e["role"] == "user" else "bot"
        name = anon.anonymize_text(e.get("name") or "")
        tag = _CLS_TAG.get(ts._user_class(e["text"]), "") if (_ROLE_TAGS_ON and e["role"] == "user") else ""
        pid = e.get("parent_id")
        if pid and pid in msgid2idx:
            rep = f"↩回复#{msgid2idx[pid]}"      # 精确回复目标
        elif pid:
            rep = "↩回复前文(父消息是噪声/不在本任务内)"  # 父被过滤或跨任务
        else:
            rep = ""
        rows.append(f"#{i} [{hhmm}] {role}({name}){tag}{rep}: {txt}")
        meta.append(e)
    return rows, meta


def llm_decompose(cluster, anon: RegexAnonymizer, model=_LLM_MODEL):
    rows, meta = build_transcript(cluster, anon)
    transcript = "\n".join(rows)
    client = OpenAI(api_key=_LLM_KEY, base_url=_LLM_BASE)
    resp = client.chat.completions.create(
        model=model, temperature=0, max_tokens=8000,  # 给足输出额度，防大任务 JSON 截断
        messages=[{"role": "system", "content": _SYS_PROMPT},
                  {"role": "user", "content": (_TAG_LEGEND if _ROLE_TAGS_ON else "") + "任务消息：\n" + transcript}],
    )
    raw = resp.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1].lstrip("json").strip()
    return json.loads(raw), rows, meta


def _effective(cluster):
    """与 build_transcript 一致的非噪声过滤——保证分块计数和喂模型的清单口径相同。"""
    out = []
    for e in cluster:
        if ts._is_noise(e["text"]):
            continue
        if not ts._strip_feishu(e["text"]).strip():
            continue
        out.append(e)
    return out


def _split_by_gaps(eff, target):
    """把一个超大任务的有效消息切成 ~target 条/块，切点落在**最大时间 gap** 处，
    尽量不把同一个子需求（连续活动）拦腰切断。返回若干 event 子列表（覆盖且互斥）。"""
    import math
    n = len(eff)
    if n <= target:
        return [eff]
    nchunks = math.ceil(n / target)
    size = math.ceil(n / nchunks)
    cuts = []
    for k in range(1, nchunks):
        ideal = k * size
        lo, hi = max(1, ideal - size // 3), min(n - 1, ideal + size // 3)
        best, best_gap = ideal, -1
        for i in range(lo, hi + 1):
            gap = eff[i]["ts"] - eff[i - 1]["ts"]
            if gap > best_gap:
                best_gap, best = gap, i
        cuts.append(best)
    chunks, prev = [], 0
    for c in sorted(set(cuts)):
        if c > prev:
            chunks.append(eff[prev:c])
            prev = c
    chunks.append(eff[prev:])
    return [c for c in chunks if c]


def enforce_terminal_tail(result, terminal):
    """代码级兜底·终结状态归一（`未终结` 的**唯一裁定点**，单一事实源 compute_terminal）：

    `未终结` 是结构事实——"观测窗右沿没收尾"，**只可能落在全程时间最后那一段**；
    LLM 拿到这个词后会到处乱标（给中段/开头的单条消息也盖 `未终结`，物理上不成立：
    对话在它之后明明还继续了）。所以代码统一裁定，不信任 LLM 的自由标注：

    1. **撤销误标**：凡 status=`未终结` 但**不是全局尾段**的子需求 → 一律订正回证据态
       （有 `↺返工` 边连到它 → `有返工`，否则 → `一遍过`）；
    2. **强制尾段**：任务 compute_terminal=False（未收尾）时，把全局尾段强制 `未终结`；
       compute_terminal=True（已收尾）时，连尾段若被标 `未终结` 也撤掉。

    返回订正动作列表（写进产物审计，透明可核）。"""
    subs = result.get("subreqs", [])
    if not subs:
        return []

    def _maxidx(s):
        vals = [v for v in (_as_seg_int(x) for x in s.get("member_idx", [])) if v is not None]
        return max(vals) if vals else -1

    rework_ids = set()
    for e in result.get("edges", []):
        if "返工" in str(e.get("type", "")):
            rework_ids.add(e.get("from"))
            rework_ids.add(e.get("to"))

    tail = max(subs, key=_maxidx)
    actions = []
    # 1. 撤销非尾段的 `未终结` 误标
    for s in subs:
        if s is tail:
            continue
        if s.get("status") == "未终结":
            old = s["status"]
            s["status"] = "有返工" if s.get("id") in rework_ids else "一遍过"
            actions.append(f"{s.get('id')}「{s.get('title','')}」非尾段却标 未终结（对话在其后仍继续）"
                           f" → 订正为 {s['status']}")
    # 2. 尾段：按 compute_terminal 统一裁定
    if not terminal:
        if tail.get("status") != "未终结":
            old = tail.get("status")
            tail["status"] = "未终结"
            actions.append(f"{tail.get('id')}「{tail.get('title','')}」是未终结任务的收尾子需求 → status {old}→未终结")
    else:
        if tail.get("status") == "未终结":
            tail["status"] = "有返工" if tail.get("id") in rework_ids else "一遍过"
            actions.append(f"{tail.get('id')}「{tail.get('title','')}」任务已终结，尾段误标 未终结 → 订正为 {tail['status']}")
    return actions


def decompose_one_task(cluster, anon: RegexAnonymizer, model=_LLM_MODEL,
                       big_model=_LLM_BIG_MODEL,
                       terminal=True, big_threshold=150, chunk_target=100, retries=3):
    """单任务下钻子需求的**统一入口**（render_decompose 调它）：
    - 小任务（≤big_threshold 有效消息）：直接 decompose + 回退保真兜底（用 model=deepseek，便宜稳）。
    - 大任务：按时间 gap 切块逐块 decompose，member_idx 全局偏移、S-id 加块前缀、edges 重映射后合并
      —— 根治「几百条大任务输出 JSON 撞 max_tokens → 截断/自保摘成 1 子需求」的坍缩。
      **大块逐块改用 big_model=v4-pro**：deepseek 对 ~100 条大块高方差（坏JSON/塌成1/碎成51），
      v4-pro 稳定落在合理粒度并识别交织子需求+返工状态（2026-06-30 诊断，见 _LLM_BIG_MODEL 注释）。
    每块/每次调用对瞬时坏 JSON 重试 retries 次，仍坏才抛。返回 (result, n_eff)。"""
    def _one(sub, mdl):
        last_exc = None
        last_res = last_n = None
        overlap_retried = False   # 大段重叠只额外重试一次拆分，再不行就确定性消解
        for _ in range(retries):
            try:
                res, rows, meta = llm_decompose(sub, anon, model=mdl)
                enforce_rollback_purity(res, meta)   # 先拆走偏/回退段(建 ↺返工 边)
                enforce_status_sanity(res)           # 再订正单消息误标(尊重上面的边)
            except Exception as ex:
                last_exc = ex
                # 429/限流是 cooldown（"Try again in N seconds"）：立刻重试必然全撞同一冷却窗→整组失败。
                # 按报错里的 N 退避后再重试；非限流错（坏 JSON）不睡，快速重试。
                _msg = str(ex)
                if "429" in _msg or "ratelimit" in _msg.lower() or "rate limit" in _msg.lower():
                    import re as _re
                    _m = _re.search(r"in (\d+) seconds", _msg)
                    time.sleep(min((int(_m.group(1)) + 5) if _m else 35, 60))
                continue
            last_res, last_n = res, len(rows)
            # 子需求重叠治理：枢纽当场合法消解；大段重叠 force=False 时回 True 触发本循环重试
            if not resolve_overlaps(res, len(rows), force=overlap_retried):
                return res, len(rows)
            overlap_retried = True   # 花掉那一次重试
        if last_res is not None:     # 重试用尽仍有大段重叠 → 确定性兜底消解后返回
            resolve_overlaps(last_res, last_n, force=True)
            return last_res, last_n
        raise last_exc

    eff = _effective(cluster)
    if len(eff) <= big_threshold:
        res, n = _one(cluster, model)          # 小任务：deepseek（便宜稳，无大块高方差）
        enforce_terminal_tail(res, terminal)   # 未终结任务的收尾段 → 未终结
        return res, n

    chunks = _split_by_gaps(eff, chunk_target)
    merged = {"task_goal": "", "subreqs": [], "edges": []}
    offset = 0
    for ci, chunk in enumerate(chunks, 1):
        res, n = _one(chunk, big_model)        # 大块：v4-pro（换稳定，治 deepseek 大块高方差坍缩）
        if not merged["task_goal"]:
            merged["task_goal"] = res.get("task_goal", "")
        idmap = {}
        for s in res.get("subreqs", []):
            new_id = f"S{ci}.{s.get('id', 'S?')}"
            idmap[s.get("id")] = new_id
            merged["subreqs"].append(dict(
                s, id=new_id,
                member_idx=[(_as_seg_int(x) or 0) + offset for x in s.get("member_idx", [])]))
        for e in res.get("edges", []):
            merged["edges"].append({
                **e,
                "from": idmap.get(e.get("from"), e.get("from")),
                "to": idmap.get(e.get("to"), e.get("to"))})
        offset += n
    enforce_terminal_tail(merged, terminal)   # 未终结任务的收尾段 → 未终结（跨块取全局最后一段）
    return merged, offset


def _as_seg_int(x):
    """member_idx 鲁棒解析：模型可能写 1 / "1" / "#1" → 一律抽出数字。无数字返回 None。
    与 llm_segment._seg_ints 同源防御（同一个 `#` 前缀坑，两处都要扛）。"""
    import re
    m = re.search(r"\d+", str(x))
    return int(m.group()) if m else None


def audit_membership(result, n_rows):
    """互斥+覆盖审计（fail-loud，不静默）：返回 (重叠编号, 遗漏编号)。
    重叠 = 一条消息归了多段（违反互斥）；遗漏 = 非噪声消息没归任何段。
    遗漏里含被模型判为噪声的，需人工区分；重叠是硬错，必须为空。"""
    from collections import Counter
    cnt = Counter()
    for s in result.get("subreqs", []):
        for x in s.get("member_idx", []):
            v = _as_seg_int(x)
            if v is not None:
                cnt[v] += 1
    overlaps = sorted(k for k, v in cnt.items() if v > 1)
    missing = sorted(set(range(1, n_rows + 1)) - set(cnt))
    return overlaps, missing


# ── 子需求重叠治理（按重叠形状分流，互斥守 member_idx、关系挪 edges）────────────
# 一条消息本应只归一个子需求。LLM 偶发让两段共享消息，按重叠的「形状」分两类治：
#   · 小边界枢纽（两段只共享 ≤PIVOT_MAX 条，典型如「A 先放一放、先看 B」这一句）
#     —— 这是**合法**的跨界衔接，不是错：枢纽消息归它**开启**的那段（时间靠后的 B），
#        从靠前的 A 里剥掉，A↔B 的关系改记到 edges（不靠共享 member_idx 表达）。
#   · 大段连续重叠（共享 >PIVOT_MAX 条，典型如伞状段套了它自己的细分段）—— 真出错：
#        先让上层重试一次拆分；仍重叠才确定性消解——让**范围最窄**的那段留住消息、其余剥掉。
# 这样互斥永远守在 member_idx 层（下游统计/训练不重复计数），信息一条不丢（关系进 edges）。
PIVOT_MAX = 2
_PIVOT_EDGE_TYPE = "⤼枢纽(搁置转接)"

# ── 软归属探针（Q1·先 a 后 b）─────────────────────────────────────────────
# 默认关：互斥硬约束不变，生产管线一字不动。
# 置 TRAJ_SOFT_MEMBERSHIP=1 时打开「软归属」——保留 LLM 吐出的跨段重叠不消解，
# 让「一条消息本就同属多个子需求」（如一句话抛 4 问）的信号能流到下游被观测。
# 这是廉价探针：跑一遍看多归属到底出现多频、有没有用，再决定要不要走 b（按子需求重切）。
# member_idx 结构本就是 list、C1 的成员本就支持多归属，此开关只是不再把重叠当错剥掉。
SOFT_MEMBERSHIP = os.environ.get("TRAJ_SOFT_MEMBERSHIP", "").strip().lower() in ("1", "true", "yes", "on")


def _seg_members(s):
    """子需求的成员消息编号（去重、升序、剥掉 #/字符串外壳）。"""
    return sorted({v for v in (_as_seg_int(x) for x in s.get("member_idx", []))
                   if v is not None})


def _seg_span(s):
    """成员范围宽度（max-min）；空段视为无穷宽（消解时最先被剥）。"""
    m = _seg_members(s)
    return (m[-1] - m[0]) if m else float("inf")


def _strip_members(s, drop):
    """从子需求里删掉指定编号（保留原始写法的其余项）。"""
    drop = set(drop)
    s["member_idx"] = [x for x in s.get("member_idx", []) if _as_seg_int(x) not in drop]


def _msg_to_subs(subreqs):
    """消息编号 → 含它的子需求下标列表（按当前 member_idx 现算）。"""
    from collections import defaultdict
    m2s = defaultdict(list)
    for si, s in enumerate(subreqs):
        for v in _seg_members(s):
            m2s[v].append(si)
    return m2s


def _ensure_pivot_edge(result, from_id, to_id):
    """A↔B 已无共享 member_idx 后，把它们的衔接关系记进 edges（去重，不覆盖已有边）。"""
    if not from_id or not to_id or from_id == to_id:
        return
    for e in result.get("edges", []):
        if e.get("from") == from_id and e.get("to") == to_id:
            return
    result.setdefault("edges", []).append(
        {"from": from_id, "to": to_id, "type": _PIVOT_EDGE_TYPE})


def resolve_overlaps(result, n_rows, force=False, pivot_max=PIVOT_MAX):
    """按重叠形状治理子需求重叠（原地改 result）。返回 large_remain:bool——
    True = 存在「大段连续重叠」且本轮 force=False 未消解（调用方应重试一次拆分）。
    枢纽（小边界）总是当场合法消解；大段重叠仅在 force=True 时确定性消解（最窄段胜）。"""
    subreqs = result.get("subreqs", [])
    # 先把每段内部的重复编号去掉，避免自重叠误报
    for s in subreqs:
        seen, dedup = set(), []
        for x in s.get("member_idx", []):
            v = _as_seg_int(x)
            if v is not None and v in seen:
                continue
            if v is not None:
                seen.add(v)
            dedup.append(x)
        s["member_idx"] = dedup
    # 软归属探针：段内去重后即返回，跨段重叠一律保留（不剥、不记枢纽边），供下游观测多归属
    if SOFT_MEMBERSHIP:
        return False
    if len(subreqs) < 2:
        return False
    overlaps, _ = audit_membership(result, n_rows)
    if not overlaps:
        return False

    # 把共享消息按「子需求对」归拢，统计每对共享几条 → 判枢纽还是大段重叠
    from collections import defaultdict
    pair_shared = defaultdict(set)
    for m, subs in _msg_to_subs(subreqs).items():
        if len(subs) >= 2:
            for i in range(len(subs)):
                for j in range(i + 1, len(subs)):
                    pair_shared[frozenset((subs[i], subs[j]))].add(m)

    has_large = False
    # 第一遍·枢纽：共享 ≤pivot_max → 归靠后的段(它开启的)，从靠前的段剥掉 + 记边
    for pair, shared in pair_shared.items():
        if len(shared) > pivot_max:
            has_large = True
            continue
        a, b = sorted(pair, key=lambda si: (_seg_members(subreqs[si]) or [10**9])[0])
        _strip_members(subreqs[a], shared)
        _ensure_pivot_edge(result, subreqs[a].get("id"), subreqs[b].get("id"))

    # 第二遍·大段重叠：真错。force=False 先交给上层重试一次拆分
    if has_large and not force:
        return True
    # force=True（重试后仍重叠）→ 确定性消解：每条仍被多段占的消息，最窄段留、其余剥
    if has_large:
        for m, subs in list(_msg_to_subs(subreqs).items()):
            if len(subs) < 2:
                continue
            winner = min(subs, key=lambda si: _seg_span(subreqs[si]))
            for si in subs:
                if si != winner:
                    _strip_members(subreqs[si], {m})

    _prune_empty_subreqs(result)
    return False


def _prune_empty_subreqs(result):
    """剥到没成员的子需求删掉，并清掉引用它的 edges（信息已并入留存段/枢纽边）。"""
    kept, dropped = [], set()
    for s in result.get("subreqs", []):
        if _seg_members(s):
            kept.append(s)
        else:
            dropped.add(s.get("id"))
    if dropped:
        result["subreqs"] = kept
        result["edges"] = [e for e in result.get("edges", [])
                           if e.get("from") not in dropped and e.get("to") not in dropped]


def enforce_rollback_purity(result, meta):
    """代码级兜底·反幻觉保真红线（不靠模型自觉）：
    若一个『回退段』(status=被纠偏 且 type∈{收尾,旁支}) 把它前面的 bot 走偏也并了进来，
    就**强制拆**成「走偏段 ↺返工→ 回退段」两段并补因果边——因为合并会抹掉
    "agent 当时没想起已沉淀方法、被提醒才回退" 这个真实约束，训出无中生有的伪先知。

    判据（纯结构、低误报）：回退段 member_idx 里，第一条『用户(外部提醒)』消息之前
    若存在实质 bot 消息，则这段 bot 前缀 = 被并入的走偏，从提醒处切开。
    回退段已从提醒起头(纯净)或无外部提醒 → 不动。
    切分只在已有 member 内做划分(不增不减)，故不破坏互斥/覆盖。
    返回 code 执行的动作列表(写进产物审计，透明可核)。"""
    def _is_user(idx):
        return meta[idx - 1].get("role") == "user"

    actions, new_subreqs = [], []
    for s in result.get("subreqs", []):
        members = sorted(v for v in (_as_seg_int(x) for x in s.get("member_idx", []))
                         if v is not None)
        rollback = (s.get("status") == "被纠偏"
                    and s.get("type") in ("收尾", "旁支"))
        if not rollback or len(members) < 2:
            new_subreqs.append(s)
            continue
        remind_pos = next((j for j, idx in enumerate(members) if _is_user(idx)), None)
        if not remind_pos:  # None(无外部提醒) 或 0(已从提醒起头，纯净)
            new_subreqs.append(s)
            continue
        prefix, rest = members[:remind_pos], members[remind_pos:]
        if not any(not _is_user(i) for i in prefix):  # 前缀无实质 bot 走偏
            new_subreqs.append(s)
            continue
        old_id = s.get("id", "S?")
        stray_id, back_id = old_id + "a", old_id + "b"
        stray_bot = next((meta[i - 1].get("name") for i in reversed(prefix)
                          if not _is_user(i)), "") or "(走偏方)"
        stray = {"id": stray_id, "title": "走偏：" + s.get("title", "")[:12],
                 "lead_quote": "", "lead_role": "bot", "dominant": stray_bot,
                 "type": "旁支", "status": "有返工", "member_idx": prefix}
        back = dict(s, id=back_id, member_idx=rest)
        # 旧 id 上的现有 edge 重指到回退段(回退是延续结果)，避免悬空
        for e in result.get("edges", []):
            if e.get("from") == old_id:
                e["from"] = back_id
            if e.get("to") == old_id:
                e["to"] = back_id
        result.setdefault("edges", []).append(
            {"from": stray_id, "to": back_id, "type": "↺返工"})
        new_subreqs += [stray, back]
        actions.append(
            f"{old_id} 回退段并入了前置走偏 {prefix} → 强制拆为 "
            f"{stray_id}(走偏,dominant={stray_bot}) ↺返工→ {back_id}(回退)，并补因果边")
    if actions:
        result["subreqs"] = new_subreqs
    return actions


def enforce_status_sanity(result):
    """代码级兜底·状态自洽：单条消息的子需求若标 `有返工`/`被纠偏`，但**没有任何 `↺返工`
    因果边**连到它 → 订正为 `一遍过`。
    依据：返工/纠偏 = 「先做一版 → 再返工/被纠正」的闭环，单条消息内不可能发生；
    若闭环是跨段表达的（有 ↺返工 edge 连到本段，如代码拆出的走偏/回退段），则**保留不动**。
    返回订正动作列表（写进产物审计，透明可核）。"""
    rework_ids = set()
    for e in result.get("edges", []):
        if "返工" in str(e.get("type", "")):
            rework_ids.add(e.get("from"))
            rework_ids.add(e.get("to"))
    actions = []
    for s in result.get("subreqs", []):
        mem = [v for v in (_as_seg_int(x) for x in s.get("member_idx", [])) if v is not None]
        if (len(mem) < 2 and s.get("status") in ("有返工", "被纠偏")
                and s.get("id") not in rework_ids):
            old = s["status"]
            s["status"] = "一遍过"
            actions.append(f"{s.get('id')}「{s.get('title','')}」单条消息#{mem} 无 ↺返工 边却标 {old} → 订正为 一遍过")
    return actions


def render(group_id, task_idx, result, rows, purity_actions=None):
    L = [f"# task{task_idx} 子需求 · LLM 语义切（方法②, model={_LLM_MODEL}）", "",
         "> 复用 task_stitch 加载/任务切分 → RegexAnonymizer 脱敏 → deepseek 语义分段。",
         "> 脱敏在喂模型**之前**完成：secret/app_id/open_id/人名 均不裸送。", "",
         f"**任务总目标**：{result.get('task_goal','')}", "",
         "## 子需求", "",
         "| id | title | 主导 | type | status | lead(脱敏原文) | 消息# |",
         "|----|-------|------|------|--------|----------------|-------|"]
    for s in result.get("subreqs", []):
        idx = ",".join(str(x) for x in s.get("member_idx", []))
        L.append(f"| {s.get('id','')} | {s.get('title','')} | {s.get('dominant','')} "
                 f"| {s.get('type','')} | {s.get('status','')} "
                 f"| {s.get('lead_quote','')} | {idx} |")
    L += ["", "## 子需求间关系 (edges)", ""]
    for e in result.get("edges", []):
        L.append(f"- {e.get('from','')} —{e.get('type','')}→ {e.get('to','')}")
    overlaps, missing = audit_membership(result, len(rows))
    _ov_label = (f"- 多归属编号（软归属探针·合法多归）：{overlaps or '无'}"
                 if SOFT_MEMBERSHIP else
                 f"- 重叠编号（硬错，应为空）：{overlaps or '无 ✅'}")
    L += ["", "## 归属审计（互斥+覆盖）", "",
          _ov_label,
          f"- 未归属编号（含被判噪声，需人工区分）：{missing or '无'}"]
    L += ["", "## 回退保真强制（代码兜底·反幻觉红线）", ""]
    if purity_actions:
        for a in purity_actions:
            L.append(f"- ⚠️ 代码已强制拆分：{a}")
        L.append("- 注：走偏段的 title/dominant 为代码机械填充，member_idx 划分精确，"
                 "如需更贴切的标题可人工微调（不影响保真结构）。")
    else:
        L.append("- 无需干预：回退段均已纯独立（或本任务无被纠偏回退）✅")
    L += ["", "---", "", "## 喂给模型的脱敏消息清单（可复核脱敏是否干净）", ""]
    L += rows
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", required=True)
    ap.add_argument("--task-idx", type=int, required=True, help="第几个任务(1基)")
    ap.add_argument("--hist-file", help="已导出的 messages.raw.jsonl，免重拉")
    ap.add_argument("--since", type=int)
    ap.add_argument("--until", type=int)
    ap.add_argument("--model", default=_LLM_MODEL)
    ap.add_argument("--out")
    args = ap.parse_args()

    evs = ts.fetch_history(args.group, args.since or 0, args.until or 0, args.hist_file)
    if not evs:
        raise SystemExit("没拉到历史消息")
    clusters = ts.segment_history(evs)
    if not (1 <= args.task_idx <= len(clusters)):
        raise SystemExit(f"task-idx 超界：共 {len(clusters)} 个任务")
    cluster = clusters[args.task_idx - 1]

    terminal = ts.compute_terminal(clusters)[args.task_idx - 1]
    anon = RegexAnonymizer()
    result, rows, meta = llm_decompose(cluster, anon, model=args.model)
    purity_actions = enforce_rollback_purity(result, meta)
    sanity_actions = enforce_status_sanity(result)
    tail_actions = enforce_terminal_tail(result, terminal)
    md = render(args.group, args.task_idx, result, rows,
                purity_actions + sanity_actions + tail_actions)

    ts.SHARED.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else \
        ts.SHARED / f"llmdecompose_{args.group[:8]}_task{args.task_idx}.md"
    out.write_text(md, encoding="utf-8")
    overlaps, missing = audit_membership(result, len(rows))
    print(f"任务{args.task_idx}：{len(rows)} 条有效消息 → {len(result.get('subreqs',[]))} 个子需求")
    if overlaps:
        if SOFT_MEMBERSHIP:
            print(f"🔬 软归属探针：{len(overlaps)} 条消息多归属（编号 {overlaps}）—— 已保留供观测")
        else:
            print(f"⚠️ 互斥违规：编号 {overlaps} 归了多段（需重跑或人工修）")
    if purity_actions:
        print(f"🛡️ 回退保真兜底触发 {len(purity_actions)} 处，已代码强制拆分：")
        for a in purity_actions:
            print(f"   - {a}")
    print(f"文件：{out}")


if __name__ == "__main__":
    main()
