"""任务粒度的 LLM 语义切分（治本，替代 segment_history 的启发式任务边界）。

为什么要它（张耀明 2026-06-27 实测拍板）：纯启发式（话题标签+时间 gap+词重叠）在密集多任务
群里调不出合理任务边界——跨天续接接不回、carry-forward 的〔标签〕漂移让话题 gate 失灵、
大簇裸词计数恒超标 → 要么过切成碎片、要么过并成 2 坨。任务「是不是同一件事」是语义问题，
启发式顶不住，必须让 LLM 读懂目标来判。

两段式（治本同时扛得住规模，1 次 LLM 调用搞定 4 天上千条）：
  Stage1 原子段（确定性，复用 task_stitch.atomic_segments）：按硬时间断档 + 持续话题切换
         切成「最小连续活动块」，宁过切勿过并——每块几乎总是单一任务的一段。
  Stage2 LLM 语义分组：把每个原子段压成一行摘要（时间段/话题/参与方/起头人话/混合要点/线索），
         让 LLM 按「同一个目标=同一任务」把原子段分组（**允许跨天、跨其它任务、跨漂移标签
         地把非相邻原子段并回同一任务**）。输出 task→成员原子段；代码侧做互斥+覆盖审计。

脱敏：摘要喂模型**之前**用 RegexAnonymizer 脱敏（secret/app_id/open_id/人名 不裸送）。

用法：
  python3 llm_segment.py --group oc_xxx --hist-file /tmp/.../messages.raw.jsonl
  python3 llm_segment.py --group oc_xxx --since <epoch> --until <epoch>
  python3 llm_segment.py --group oc_xxx --hist-file ... --decompose   # 任务切完再下钻子需求
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# 复用 task_stitch 的加载/原子分段/渲染（单一事实源，不重写）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import task_stitch as ts  # noqa: E402

# 复用确定性脱敏器（在 data_process 根目录）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from regex_anonymizer import RegexAnonymizer  # noqa: E402

from openai import OpenAI  # noqa: E402

# 本机 litellm 代理：免配 key（master key 固定），deepseek 不出本机外网
_LLM_BASE = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:4000/v1")
_LLM_KEY = os.environ.get("LLM_API_KEY", "sk-litellm-master-key")
# 分组器默认 deepseek-v4-pro：实测 σ0.9（V3 deepseek 为 σ2.85），且标题污染 0/6（V3 4/6），
# 把「装bot」这类同一交付物的执行子步骤稳定并成 1 任务。decompose 走自己的 LLM_DECOMPOSE_MODEL
# （默认 deepseek，见 llm_decompose.py）——两个模型独立可调，不再共用一个默认。
_LLM_MODEL = os.environ.get("LLM_SEGMENT_MODEL", "deepseek-v4-pro")

_SYS_PROMPT = """你是对话轨迹分析专家。给你一个多人群聊里**按时间排好的若干「原子段」摘要**\
（已脱敏，每段是一小块连续活动）。你的任务：把这些原子段按「同一个任务」分组。

定义：
- 任务 = 用户一个**独立的总目标 / 一个交付物**，从提出到（试图）做完的一整条线。一条训练样本对应一个任务。
- 一个任务可能横跨多天、中间被别的任务打断、隔几小时甚至隔夜再续 —— 这些散落的原子段都属于
  同一个任务，要分到同一组。
- **任务边界看「交付物/产出」**：把某个 bot 装好并跑通、实现并验证某个功能、采集到某批数据——
  从提出到（试图）交付的一整条线，**中间所有支撑步骤（配环境/装依赖/开账号/SSH免密/建群/改bug/
  验证登录）都属于这一个任务**，它们是任务内部的子需求层（留给下一步 decompose 去切），
  在这一步**不要各自拆成独立任务**。

分组铁律（按重要性逐条照做）：
1. 【按目标分组，不按话题标签】每段开头的〔标签〕（如〔数据处理清洗〕〔bot基建〕）只是粗略
   领域，**同一个标签下常有好几个完全不同的任务**（如「装 antigravity bot」和「修 litellm 鉴权」
   都可能挂同一个领域标签）。判同任务看的是**具体目标/对象是不是同一个**（同一个 bug、同一个
   功能、同一个文件/服务/脚本），不是标签是否相同。标签相同但目标不同 → 不同任务。
2. 【跨断档续接要并回】一个较晚的原子段如果在**继续推进**某个更早原子段的目标（明显的续接
   指代「之前那个/继续/接着上次/我们现在做到哪」，或就是在对同一对象做下一步），即便中间隔了
   很久、隔了夜、夹了别的任务，也要**并进那个更早的任务组**，不要因为时间断开就新开任务。
3. 【粒度：按交付物，不按步骤】判任务边界看**交付物/产出是不是同一个**，两个方向都不能犯：
   - 反「过切」：同一个交付物的连续步骤**别拆成多任务**。例「给 X 装 antigravity bot」——
     开账号→SSH 免密→建群→修安装器→验证登录，是**一个任务**（这些步骤是它的子需求），
     把这一次连续安装拆成 10+ 个任务是严重过切。
   - 反「过并」：**不同交付物必须分开**，即便话题/对象相邻相关。例即便都围绕「轨迹」，
     『采集/获取轨迹』『按任务拆分轨迹』『对齐某类 bot 的轨迹保真度』是三个不同产出 = 三个任务，
     别焊成一坨；『教 yaoming 登录平台』和『给 X 装 bot』也是两个不同交付物，要分开。
   一句话判据：**不同的最终产出 = 不同任务；服务于同一产出的不同步骤 = 同一任务的子需求。**
   焊不同产出成一坨、和把同一产出过切成碎步骤，**都是污染**。
4. 【全覆盖且互斥】每个原子段编号必须、且只能属于一个任务的 member_segs。输出前自检：把所有
   任务的 member_segs 拼起来，(a) 不许有编号出现在两个任务里；(b) 所有编号都要被覆盖（不漏）。
5. 【孤段也成任务】一个原子段如果不属于任何其它任务（独立的小请求/一次性问答），它自己就是
   一个任务，不要硬塞进别的组。
6. 【零散追问/插话：先归并，除非催生新产出】群聊里常有零散的追问、查进度、讨论、检查类插话
   （如「拆分进度到哪了」「讨论一下粒度需求」「检查下两判官有没有问题」）。这类段最容易被时拆
   时并、是波动的主因，按下面口径**确定性处理**：
   - 这条追问/讨论若**没有引出一条新的独立产出**——只是在问/议/查某个已有任务的状态或细节——
     就**归并进它所追问的那个任务**。它在那个任务内部**至多是一个新的子需求**（留给 decompose
     去识别），在这一步**绝不单独成一个任务**。
   - 只有当这条追问**实际开启了一条新的、独立的交付线**（开始做一个之前没有的新东西）时，
     才另起新任务。
   一句话判据：**追问 / 讨论 / 检查 ≠ 新任务，除非它催生了新产出**；问的是哪个任务，就并进哪个
   任务。同样几条插话，别这次并、下次拆。

每个任务输出字段：
- id: "T1","T2"... 按该任务**最早**原子段的时间排序
- title: ≤20字，明确体现这个任务的目标（如「给张耀明装 antigravity bot」「修 litellm db 鉴权」）
- goal: 一句话总目标（≤40字）
- member_segs: 属于这个任务的原子段编号列表（用输入里的 # 编号，按时间升序）
- reason: ≤30字，为什么这些段是同一任务（尤其跨断档并回的，点明续接信号）

严格输出 JSON：{"tasks":[...]}。不要 markdown 包裹、不要解释。"""


def _safe_name(name, anon: RegexAnonymizer):
    """bot 名兜底脱敏：未登记 bot 在 _cluster_label 里会退化成裸 app_id[:14]，截断后逃过
    RegexAnonymizer 的 cli_+15位 规则 → 这里把仍像裸 id(cli_/ou_ 起头) 的名字一律打码，
    防 app_id/open_id 裸送 LLM。正常可读名（已登记 bot/人名）原样保留。"""
    import re
    s = anon.anonymize_text(name or "")
    if re.match(r"^(cli_|ou_)[a-zA-Z0-9]+", s):
        return "[未登记bot]"
    return s


def _meaningful_rows(rows):
    return [e for e in rows if not ts._is_noise(e["text"]) and ts._strip_feishu(e["text"]).strip()]


def _brief_msg(e, anon):
    txt = ts._strip_feishu(e["text"]).strip()
    txt = anon.anonymize_text(txt).replace("\n", " ")[:80]
    role = "用户" if e["role"] == "user" else "bot"
    return f"{role}:{txt}"


def _seg_gist(rows, anon, n=4):
    """取首尾+长消息混合要点，避免主题埋在第4条以后时摘要完全看不到。"""
    meaningful = _meaningful_rows(rows)
    selected, seen = [], set()

    def add(idx):
        if idx < 0 or idx >= len(meaningful):
            return
        e = meaningful[idx]
        key = e.get("msg_id") or (e.get("ts"), e.get("role"), e.get("text"))
        if key in seen:
            return
        seen.add(key)
        selected.append(e)

    # 首条保留起因；末条保留结果/当前状态；最长两条补足中后段主题。
    add(0)
    add(len(meaningful) - 1)
    for idx, _ in sorted(
        enumerate(meaningful),
        key=lambda item: len(ts._strip_feishu(item[1]["text"]).strip()),
        reverse=True,
    ):
        add(idx)
        if len(selected) >= n:
            break

    out = []
    for e in selected[:n]:
        out.append(_brief_msg(e, anon))
    return out


def _seg_entities(rows, anon, limit=6):
    """抽少量稳定实体/文件名/命令词，给 LLM 聚类增加非位置线索。"""
    import re
    found = []
    seen = set()
    for e in rows:
        if ts._is_noise(e["text"]):
            continue
        txt = anon.anonymize_text(ts._strip_feishu(e["text"]))
        for m in re.findall(r"[A-Za-z0-9_./-]{4,}|[\u4e00-\u9fffA-Za-z0-9_-]+(?:bot|任务|轨迹|切分|配置|登录|安装|重启)", txt):
            token = m.strip(".,;:，。；：()[]{}<>")
            if len(token) < 4 or token in seen:
                continue
            seen.add(token)
            found.append(token[:40])
            if len(found) >= limit:
                return found
    return found


def build_segment_index(atoms, anon: RegexAnonymizer):
    """每个原子段 → 一行摘要（编号/时间段/话题/参与方/起头人话/混合要点/线索），脱敏。"""
    lines = []
    for i, rows in enumerate(atoms, 1):
        dom, bots, ask = ts._cluster_label(rows)
        t0 = time.strftime("%m-%d %H:%M", time.localtime(rows[0]["ts"]))
        t1 = time.strftime("%m-%d %H:%M", time.localtime(rows[-1]["ts"]))
        nreal = sum(1 for e in rows if not ts._is_noise(e["text"]))
        ask_a = anon.anonymize_text(ask or "").replace("\n", " ")[:70]
        bots_a = "、".join(_safe_name(b, anon) for b in bots) or "—"
        gist = "｜".join(_seg_gist(rows, anon))
        entities = "、".join(_seg_entities(rows, anon)) or "—"
        lines.append(
            f"#{i} 〔{dom}〕 {t0}→{t1} | {nreal}条 | bot:{bots_a}\n"
            f"   起头: {ask_a}\n"
            f"   要点: {gist}\n"
            f"   线索: {entities}")
    return lines, atoms


def _atom_full_text(rows, anon):
    """嵌入用：脱敏后全部有意义正文拼接（截断），比 4 行 gist 更接近全文。
    与 offline_bge_candidate_probe.atom_text 同口径（候选召回单一事实源）。"""
    parts = []
    for e in rows:
        if ts._is_noise(e["text"]):
            continue
        t = anon.anonymize_text(ts._strip_feishu(e["text"])).strip()
        if t:
            role = "用户" if e["role"] == "user" else "bot"
            parts.append(f"{role}:{t}")
    return "\n".join(parts)[:2000] or "(空)"


def candidate_pairs(atoms, anon, topk=8, floor=0.55):
    """bge-m3 算原子段两两余弦，回 floor 以上的 top-K「疑似同任务」候选对。
    embedding 不可用（服务挂/未配）→ 返回 []（调用方退回无提示，绝不崩）。
    **只做候选召回**：阈值是「召回↔token」旋钮，不是正确性裁决——最终由 LLM 按交付物拍板
    （离线验证见 offline_bge_candidate_probe.py / 文档§四，铁律：候选-only 永不自动合并）。"""
    import embed_util
    texts = [_atom_full_text(rows, anon) for rows in atoms]
    M = embed_util.embed_texts(texts)
    if M is None:
        return []
    pairs = []
    n = len(atoms)
    for i in range(n):
        for j in range(i + 1, n):
            sim = float((M[i] * M[j]).sum())
            if sim >= floor:
                pairs.append((sim, i + 1, j + 1))  # 1-based 与摘要 #编号 对齐
    pairs.sort(reverse=True)
    return pairs[:topk]


def _candidate_hint_block(atoms, anon, topk, floor):
    """把候选对渲染成**中性、防锚定**的辅助提示；无候选或 embed 不可用 → 空串。"""
    pairs = candidate_pairs(atoms, anon, topk=topk, floor=floor)
    if not pairs:
        return ""
    rows = "\n".join(f"  #{i} ⟷ #{j}（文字相近度 {sim:.2f}）" for sim, i, j in pairs)
    return (
        "\n\n【候选参考 · 仅供辅助，不是结论】\n"
        "下列原子段对「文字表述较接近」，**可能也可能不**属于同一任务。请仍严格按上面的交付物\n"
        "铁律自行判断：内容相近≠同一任务（不同任务常用词相似），表述不同≠不同任务（同一任务\n"
        "跨阶段用词会变）。**绝不要因为某对出现在这里，就把不同交付物的段强行并到一起。**\n"
        + rows)


def llm_segment(atoms, anon: RegexAnonymizer, model=_LLM_MODEL,
                candidate_hints=None, hint_topk=8, hint_floor=0.55):
    """LLM 把原子段分组成任务。返回 (result_dict, index_lines)。

    candidate_hints: True/False 显式开关；None 时读环境变量 SEG_CANDIDATE_HINTS（默认关，
      baseline 行为完全不变）。开启时在摘要清单后追加 bge-m3 候选对中性提示（甲·候选召回）。"""
    if candidate_hints is None:
        candidate_hints = os.environ.get("SEG_CANDIDATE_HINTS", "0") == "1"
    lines, _ = build_segment_index(atoms, anon)
    index = "\n".join(lines)
    user_content = "原子段摘要清单：\n" + index
    if candidate_hints:
        user_content += _candidate_hint_block(atoms, anon, hint_topk, hint_floor)
    client = OpenAI(api_key=_LLM_KEY, base_url=_LLM_BASE)
    resp = client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "system", "content": _SYS_PROMPT},
                  {"role": "user", "content": user_content}],
    )
    raw = resp.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1].lstrip("json").strip()
    return json.loads(raw), lines


_SPLIT_AUDIT_PROMPT = """你是对话轨迹分析专家。下面若干「原子段」被初步归为**同一个任务**，\
现在请你做一次独立复核：它们是否真的服务于**同一个最终交付物**？

判据（交付物铁律）：
- 同一交付物的不同步骤/阶段（如 申请权限→配环境→联调，环环相扣指向一个成果）→ 保持一组。
- 其实是 **≥2 条各自独立、只是时间上并行或交错**的工作线（例如同时在调试两个互不依赖的功能、
  各有各的目标和产出）→ **必须拆成多组**。

**默认倾向拆分**：只有当你确信这些段共同指向同一个可交付成果、彼此是支撑/递进关系时才合并；
只要看出两条互不依赖、可以各自单独交付的工作线，就拆开。表面话题相近、同一时段、同一个 bot/服务，
**都不是合并的理由**——看的是产出是不是同一个。

严格输出 JSON：{"groups": [[编号,...], ...]}。每个输入编号恰好出现一次、全覆盖。不要 markdown、不要解释。"""


def _split_audit_llm(atoms, seg_ids, anon, model=_LLM_MODEL):
    """对一个多段任务做交付物审计：返回拆分后的分组 list[list[int]]（1-based 段号）。
    LLM 挂/解析失败 → 返回 [seg_ids]（不拆，fail-safe）；漏的段补回，绝不丢数据。"""
    seg_ids = sorted(set(seg_ids))
    if len(seg_ids) < 2:
        return [seg_ids]
    parts = [f"【原子段 #{s}】\n{_atom_full_text(atoms[s - 1], anon)}" for s in seg_ids]
    user = "初步归为同一任务的原子段：\n\n" + "\n\n".join(parts)
    try:
        client = OpenAI(api_key=_LLM_KEY, base_url=_LLM_BASE)
        resp = client.chat.completions.create(
            model=model, temperature=0,
            messages=[{"role": "system", "content": _SPLIT_AUDIT_PROMPT},
                      {"role": "user", "content": user}],
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1].lstrip("json").strip()
        groups = json.loads(raw).get("groups", [])
    except Exception:
        return [seg_ids]  # fail-safe：不拆
    out, seen = [], set()
    for g in groups:
        ints = [x for x in _seg_ints(g) if x in seg_ids and x not in seen]
        for x in ints:
            seen.add(x)
        if ints:
            out.append(ints)
    missing = [s for s in seg_ids if s not in seen]
    if missing:
        out.append(missing)  # 漏段补回，绝不丢数据
    return out or [seg_ids]


def apply_split_guard(atoms, result, anon, model=_LLM_MODEL, enabled=None):
    """拆分守卫（默认关，env SEG_SPLIT_AUDIT=1 开启）：对分组产出里每个多段任务做 LLM 交付物
    审计，判为 ≥2 个独立并行交付物的拆成多任务。与缝合守卫互补——缝合管「漏并」、本守卫管「过并」。
    bge-m3 相似度版已被实证证伪（weave 表面词比 zym 真同任务还相似，无可行地板），故改走意图层 LLM 审计。"""
    if enabled is None:
        enabled = os.environ.get("SEG_SPLIT_AUDIT", "0") == "1"
    if not enabled:
        return result
    new_tasks = []
    for t in result.get("tasks", []):
        seg_ids = sorted(set(_seg_ints(t.get("member_segs", []))))
        if len(seg_ids) < 2:
            new_tasks.append(t)
            continue
        groups = _split_audit_llm(atoms, seg_ids, anon, model)
        if len(groups) <= 1:
            new_tasks.append(t)
        else:
            for g in groups:
                nt = dict(t)
                nt["member_segs"] = g
                nt["_split_audit"] = f"拆自「{t.get('title', '')}」"
                new_tasks.append(nt)
    return {"tasks": new_tasks}


def _seg_ints(members):
    """鲁棒解析 member_segs：LLM 可能写 1 / "1" / "#1" / "#1,2" → 一律抽成 int 列表。"""
    import re
    out = []
    for x in members or []:
        for m in re.findall(r"\d+", str(x)):
            out.append(int(m))
    return out


def audit_membership(result, n_segs):
    """互斥+覆盖审计（fail-loud）：返回 (重叠编号, 遗漏编号)。
    重叠=一个原子段归了多任务(硬错,应空)；遗漏=某段没归任何任务(补成孤任务)。"""
    cnt = Counter()
    for t in result.get("tasks", []):
        for x in _seg_ints(t.get("member_segs", [])):
            cnt[x] += 1
    overlaps = sorted(k for k, v in cnt.items() if v > 1)
    missing = sorted(set(range(1, n_segs + 1)) - set(cnt))
    return overlaps, missing


def assemble_clusters(atoms, result):
    """task→成员原子段 映射回 clusters（与 segment_history 输出同构：每个 cluster = 行列表）。
    去重(若 LLM 误把一段归多任务，按首次出现归属)；遗漏的段各自补成孤任务，绝不丢数据。"""
    seen = set()
    tasks = sorted(result.get("tasks", []),
                   key=lambda t: min(_seg_ints(t.get("member_segs", [])), default=10**9))
    clusters, metas = [], []
    for t in tasks:
        idxs = []
        for x in _seg_ints(t.get("member_segs", [])):
            if 1 <= x <= len(atoms) and x not in seen:
                seen.add(x)
                idxs.append(x)
        if not idxs:
            continue
        rows = []
        for x in idxs:
            rows.extend(atoms[x - 1])
        rows.sort(key=lambda e: e["ts"])
        clusters.append(rows)
        metas.append({"title": t.get("title", ""), "goal": t.get("goal", ""),
                      "reason": t.get("reason", ""), "segs": idxs})
    # 遗漏段：各自成孤任务（fail-loud 已在 audit 报出）
    for x in range(1, len(atoms) + 1):
        if x not in seen:
            clusters.append(list(atoms[x - 1]))
            metas.append({"title": "(LLM 未归类·孤段)", "goal": "", "reason": "未归类补孤任务",
                          "segs": [x]})
    order = sorted(range(len(clusters)), key=lambda i: clusters[i][0]["ts"])
    return [clusters[i] for i in order], [metas[i] for i in order]


def render(group_id, clusters, metas, lines, result, window_desc):
    overlaps, missing = audit_membership(result, len(lines))
    terminal = ts.compute_terminal(clusters)
    n_inc = sum(1 for t in terminal if not t)
    L = [f"# 任务切分 · LLM 语义分组（治本, model={_LLM_MODEL}）", "",
         f"- 群：`{group_id}`　窗口：{window_desc}",
         f"- 原子段(Stage1) **{len(lines)}** 个 → LLM 分组(Stage2)出任务 **{len(clusters)}** 个"
         f"（其中未终结 ⏳{n_inc} 个）",
         f"- 归属审计：重叠(硬错应空) {overlaps or '无 ✅'}；未归类补孤任务 {missing or '无'}",
         "", "> Stage1=task_stitch.atomic_segments(硬gap+持续话题切，宁过切)；"
         "Stage2=按目标语义分组(允许跨天/跨任务/跨漂移标签并回)。脱敏后再喂模型。",
         "", "---", ""]
    for ti, (cl, m) in enumerate(zip(clusters, metas), 1):
        dom, bots, ask = ts._cluster_label(cl)
        t0 = time.strftime("%m-%d %H:%M", time.localtime(cl[0]["ts"]))
        t1 = time.strftime("%m-%d %H:%M", time.localtime(cl[-1]["ts"]))
        nreal = sum(1 for e in cl if not ts._is_noise(e["text"]))
        tflag = "" if terminal[ti - 1] else "　⏳**未终结(incomplete)**"
        L += [f"## 任务 {ti}　{m['title'] or dom}{tflag}",
              f"- 目标：{m['goal']}",
              f"- 跨度：{t0}→{t1} | {nreal} 条有效 | 参与 bot：{'、'.join(bots) or '—'}",
              f"- 成员原子段：{m['segs']}　{('| 并组理由：' + m['reason']) if m['reason'] else ''}",
              ""]
    L += ["---", "", "## 喂给模型的脱敏原子段清单（可复核脱敏 + 分组判据）", ""]
    L += lines
    return "\n".join(L)


def subreq_member_msg_ids(subs, eff):
    """把每个子需求的 member_idx（1-based，基于 llm_decompose._effective 的顺序）翻成真飞书 msg_id。
    单一事实源：member_idx→msg_id 的对应就是「非噪声消息按时间序 i↔eff[i-1]」，
    与 build_transcript 的编号口径同源（两者都用 _effective 过滤，见 llm_decompose 注释）。
    供 C 回合级纠错标注把子需求焊回具体消息（active-tail 子需求原先只有本地#、无法 join）。"""
    import llm_decompose as ld
    out = []
    for s in subs:
        ids = []
        for x in s.get("member_idx", []):
            i = ld._as_seg_int(x)
            if i is not None and 0 < i <= len(eff) and eff[i - 1].get("msg_id"):
                ids.append(eff[i - 1]["msg_id"])
        out.append({"id": s.get("id", ""), "title": s.get("title", ""),
                    "status": s.get("status", ""), "type": s.get("type", ""),
                    "dominant": s.get("dominant", ""), "member_msg_ids": ids})
    return out


def render_decompose(group_id, window_desc, clusters, metas, anon, model=None, start_idx=1):
    """对每个任务 cluster 下钻子需求（复用 llm_decompose 核心），产出「任务→子需求」合并文档。
    单条消息以下的任务不调 LLM（无内部闭环结构），直接记 1 个平凡子需求。
    model=None 时取 decompose 自己的默认（LLM_DECOMPOSE_MODEL，默认 deepseek），
    与分组器模型解耦——分组用强模型(v4pro)整合，子需求拆分仍用 deepseek。"""
    import llm_decompose as ld  # 单一事实源：子需求拆分逻辑只在 llm_decompose
    if model is None:
        model = ld._LLM_MODEL
    L = [f"# 任务 → 子需求　两级拆分（model={model}）", "",
         f"- 群：`{group_id}`　窗口：{window_desc}",
         f"- {len(clusters)} 个任务，逐个下钻子需求（子需求=「小目标→走偏→被纠→做对」的改写单位）",
         "", "---", ""]
    terminal = ts.compute_terminal(clusters)  # 单一事实源：任务是否真终结（决定收尾子需求是否 未终结）
    total_sub = 0
    deliveries: list[str | None] = []  # 与 clusters 对齐：每任务的 LLM 交付判断，供增量层 confirmation_status 消费
    task_subreqs: list[dict] = []  # 与 clusters 对齐：{title, subreqs:[{id,title,status,type,dominant,member_msg_ids}]}
                                   # 结构化子需求→真 msg_id 映射，落进 state 供 C 回合级纠错标注 join（不再靠本地#人工核）
    for offset, (cl, m) in enumerate(zip(clusters, metas)):
        ti = start_idx + offset
        nreal = sum(1 for e in cl if not ts._is_noise(e["text"]))
        t0 = time.strftime("%m-%d %H:%M", time.localtime(cl[0]["ts"]))
        t1 = time.strftime("%m-%d %H:%M", time.localtime(cl[-1]["ts"]))
        is_terminal = terminal[offset]
        tflag = "" if is_terminal else "　⏳**未终结(incomplete)**"
        L.append(f"## 任务 {ti}　{m['title']}{tflag}")
        meta_tail = f"　| 跨度：{t0}→{t1} | {nreal} 条有效 | 原子段 {m['segs']}"
        if nreal < 2:
            total_sub += 1
            deliveries.append(None)  # 单消息任务无 decompose，delivery 未知 → confirmation_status 回退确定性判
            st = "未终结" if not is_terminal else "一遍过"
            # 单消息任务无 decompose，目标只能回退到 segment 的 goal
            L.append(f"- 目标：{m['goal']}{meta_tail}")
            L += [f"- **1 个子需求**（任务本身，单条消息无内部闭环）· status={st}", ""]
            eff = ld._effective(cl)
            # 单消息任务无需 LLM，平凡子需求即成功态（decompose_ok=True，不会被判失败重跑）
            task_subreqs.append({"title": m["title"], "decompose_ok": True, "subreqs": [
                {"id": "S1", "title": m["title"], "status": st, "type": "", "dominant": "",
                 "member_msg_ids": [e.get("msg_id") for e in eff if e.get("msg_id")]}]})
            continue
        try:
            # 统一入口：大任务自动分块 decompose 防坍缩 + 瞬时坏 JSON 重试 + 回退保真兜底 + 未终结收尾
            res, n_eff = ld.decompose_one_task(cl, anon, model=model, terminal=is_terminal)
        except Exception as ex:  # 重试仍失败才 fail-loud：不静默吞，标在文档里
            deliveries.append(None)  # decompose 失败，delivery 未知 → 回退确定性判
            L.append(f"- 目标：{m['goal']}{meta_tail}")  # decompose 失败，目标回退 segment goal
            L += [f"- ⚠️ 子需求拆分失败（重试3次仍坏）：{ex}", ""]
            # decompose_ok=False：把「失败」和「合法地就是没子需求」区分开，供增量层判定重跑（不再靠 subreqs 空不空去猜）
            task_subreqs.append({"title": m["title"], "decompose_ok": False, "subreqs": []})  # 拆分失败，无子需求映射
            continue
        # 【单一数据源】任务目标用 decompose 的终态 task_goal——它读全量消息、按终态/纠偏后结论生成，
        # 比 segment 那行有损摘要的 goal 更准、且不复述中途误判；task_goal 为空才回退 segment goal。
        # 同时不再单列"总目标"，消除"目标 vs 总目标"两源打架的重复。
        task_goal = (res.get("task_goal") or "").strip() or m["goal"]
        deliveries.append((res.get("delivery") or "").strip() or None)  # LLM 任务级交付判断
        L.append(f"- 目标：{task_goal}{meta_tail}")
        subs = res.get("subreqs", [])
        total_sub += len(subs)
        # decompose_ok=True：LLM 正常返回即成功（哪怕 subreqs 合法为空），下次增量不再重跑
        task_subreqs.append({"title": m["title"], "decompose_ok": True,
                             "subreqs": subreq_member_msg_ids(subs, ld._effective(cl))})
        L.append(f"- **{len(subs)} 个子需求**：")
        L += ["", "| id | 子需求 | 主导 | type | status | 消息# |",
              "|---|---|---|---|---|---|"]
        for s in subs:
            idx = ",".join(str(x) for x in s.get("member_idx", []))
            L.append(f"| {s.get('id','')} | {s.get('title','')} | {s.get('dominant','')} "
                     f"| {s.get('type','')} | {s.get('status','')} | {idx} |")
        overlaps, missing = ld.audit_membership(res, n_eff)
        if overlaps or missing:
            L.append(f"- ⚠️ 子需求归属审计：重叠编号 {overlaps or '无'}；遗漏编号 {missing or '无'}")
        edges = res.get("edges", [])
        if edges:
            estr = "；".join(f"{e.get('from','')}{e.get('type','→')}{e.get('to','')}" for e in edges)
            L.append(f"- 子需求关系：{estr}")
        L.append("")
    L[3] = f"- {len(clusters)} 个任务 → 共 **{total_sub}** 个子需求（子需求=「小目标→走偏→被纠→做对」的改写单位）"
    return "\n".join(L), total_sub, deliveries, task_subreqs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", required=True)
    ap.add_argument("--hist-file", help="已导出的 messages.raw.jsonl，免重拉")
    ap.add_argument("--since", type=int)
    ap.add_argument("--until", type=int)
    ap.add_argument("--model", default=_LLM_MODEL)
    ap.add_argument("--out")
    ap.add_argument("--decompose", action="store_true",
                    help="任务切完，对每个任务再下钻子需求(调 llm_decompose)")
    args = ap.parse_args()

    evs = ts.fetch_history(args.group, args.since or 0, args.until or 0, args.hist_file)
    if not evs:
        raise SystemExit("没拉到历史消息")
    atoms = ts.atomic_segments(evs, sim_fn=ts.build_reply_sim_fn())
    if not atoms:
        raise SystemExit("原子分段为空")

    anon = RegexAnonymizer()
    result, lines = llm_segment(atoms, anon, model=args.model)
    clusters, metas = assemble_clusters(atoms, result)
    wd = (time.strftime("%m-%d %H:%M", time.localtime(evs[0]["ts"])) + "→" +
          time.strftime("%m-%d %H:%M", time.localtime(evs[-1]["ts"])) +
          f"（{len(evs)}条原始）")
    md = render(args.group, clusters, metas, lines, result, wd)

    ts.SHARED.mkdir(parents=True, exist_ok=True)
    out = (Path(args.out) if args.out else
           ts.SHARED / f"llmsegment_{args.group}_{time.strftime('%m%d_%H%M%S')}.md")
    out.write_text(md, encoding="utf-8")
    overlaps, missing = audit_membership(result, len(lines))
    print(f"切分：{len(evs)} 条 → 原子段 {len(atoms)} → LLM 任务 {len(clusters)} 个")
    term = ts.compute_terminal(clusters)
    print(f"未终结(B)：{sum(1 for t in term if not t)} 个")
    if overlaps:
        print(f"⚠️ 互斥违规：原子段 {overlaps} 归了多任务（已按首次归属去重）")
    if missing:
        print(f"⚠️ 未归类：原子段 {missing} LLM 没归任何任务（已各补孤任务，未丢数据）")
    print(f"文件：{out}\n查看：{ts.DATAVIEW}")

    if args.decompose:
        print(f"\n下钻子需求：对 {len(clusters)} 个任务逐个调 llm_decompose …")
        # 不传 args.model：decompose 独立走 LLM_DECOMPOSE_MODEL（默认 deepseek），与分组器解耦
        sub_md, total_sub, _, _ = render_decompose(args.group, wd, clusters, metas, anon)
        sub_out = (ts.SHARED /
                   f"llmdecompose_all_{args.group[:8]}_{time.strftime('%m%d_%H%M%S')}.md")
        sub_out.write_text(sub_md, encoding="utf-8")
        print(f"子需求：{len(clusters)} 任务 → 共 {total_sub} 个子需求\n文件：{sub_out}")


if __name__ == "__main__":
    main()
