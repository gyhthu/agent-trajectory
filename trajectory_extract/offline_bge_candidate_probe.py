"""离线验证：bge-m3 向量相似度当「候选召回」靠不靠谱（张耀明 0629 让做着试试看）。

要回答的问题（Q1）：原子段被时间切开后，bge-m3 能不能把「本属同一任务、却被切到不同
原子段」的那些对召回出来（= 价值），同时不至于把不同任务的原子段也大量拉成高相似（= 噪声）？
这关系到它会不会重蹈 06-27 词重叠覆辙——结论的前提是：**bge-m3 只做候选、不自动合并**，
最终由 LLM 按交付物拍板。所以这里量两件事：
  1) 召回：被时间切开、但 LLM 后来归到同一任务的「跨断档原子段对」，相似度排不排得高；
  2) 噪声：相似度≥阈值的对里，有多少其实是不同任务（这些会被喂给 LLM 当候选，靠 LLM 驳回）。

参考标注：用增量状态里 7 个 frozen_task 的 member_msg_ids 当「同任务」真值——它是上一步
LLM 语义分组的产出（不是人工金标，但是目前规模最大的成片参考；小样人工金标另在 gold_*.md）。

数据源：incremental 状态文件的 frozen_tasks[].events（898 条带任务归属的真实消息）。
嵌入：本机 litellm 的 bge-m3（deepseek 不出本机），每段嵌「脱敏后的全部有意义正文」。
**纯离线测量，不改任何生产口径、不接任何线上链路。**

用法：python3 offline_bge_candidate_probe.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import task_stitch as ts  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from regex_anonymizer import RegexAnonymizer  # noqa: E402

STATE = "/opt/shared/data/task-trajectory/state/oc_53b8b620867a189d8dfe502865dfccc5.json"
EMB_BASE = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:4000/v1")
EMB_KEY = os.environ.get("LLM_API_KEY", "sk-litellm-master-key")
EMB_MODEL = os.environ.get("EMB_MODEL", "bge-m3")
THRESHOLDS = [0.5, 0.6, 0.7, 0.8]


def atom_text(rows, anon):
    """嵌入用：脱敏后的全部有意义正文拼接（截断），比 4 行 gist 更接近全文。"""
    parts = []
    for e in rows:
        if ts._is_noise(e["text"]):
            continue
        t = anon.anonymize_text(ts._strip_feishu(e["text"])).strip()
        if t:
            role = "用户" if e["role"] == "user" else "bot"
            parts.append(f"{role}:{t}")
    return "\n".join(parts)[:2000] or "(空)"


def main():
    d = json.load(open(STATE, encoding="utf-8"))
    anon = RegexAnonymizer()

    # 1) 取带任务归属的事件，建 msg_id→参考任务 映射
    msg_task = {}
    titles = []
    all_evs = []
    seen = set()
    for ti, t in enumerate(d["frozen_tasks"]):
        titles.append(t["title"])
        for e in t["events"]:
            mid = e.get("msg_id") or ""
            if mid:
                msg_task[mid] = ti
            key = mid or (e["ts"], e["role"], e["text"][:20])
            if key in seen:
                continue
            seen.add(key)
            all_evs.append(e)
    all_evs.sort(key=lambda e: e["ts"])
    print(f"参考任务 {len(titles)} 个，带归属事件 {len(all_evs)} 条")

    # 2) 原子切分（确定性，复用生产链路）
    atoms = ts.atomic_segments(all_evs)
    print(f"原子段 {len(atoms)} 个")

    # 3) 每个原子段的参考任务 = 成员多数票
    def ref_task(rows):
        from collections import Counter
        c = Counter()
        for e in rows:
            mid = e.get("msg_id") or ""
            if mid in msg_task:
                c[msg_task[mid]] += 1
        return c.most_common(1)[0][0] if c else -1

    atom_ref = [ref_task(r) for r in atoms]
    texts = [atom_text(r, anon) for r in atoms]

    # 4) 嵌入（一次批量）
    cli = OpenAI(base_url=EMB_BASE, api_key=EMB_KEY)
    embs = []
    B = 32
    for i in range(0, len(texts), B):
        r = cli.embeddings.create(model=EMB_MODEL, input=texts[i:i + B])
        embs.extend([x.embedding for x in r.data])
    M = np.array(embs, dtype=np.float32)
    M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    sim = M @ M.T

    # 5) 统计所有「非相邻」原子段对（i+1<j：被至少一个其它原子段隔开 = 时间上被切开）
    n = len(atoms)
    pairs = []  # (sim, i, j, same_task)
    for i in range(n):
        for j in range(i + 1, n):
            if atom_ref[i] < 0 or atom_ref[j] < 0:
                continue
            same = atom_ref[i] == atom_ref[j]
            pairs.append((float(sim[i, j]), i, j, same, j == i + 1))

    same_nonadj = [p for p in pairs if p[3] and not p[4]]   # 跨断档同任务对 = 要召回的
    print(f"\n候选对总数 {len(pairs)}；其中『跨断档同任务对』(被切开却属同一任务) = {len(same_nonadj)} 个")
    print("（注：相邻同任务对不算，本就连着；这 %d 个才是 bge-m3 要补回来的价值目标）" % len(same_nonadj))

    print("\n阈值 | 跨断档同任务召回 | 候选对精确率(命中同任务/全部≥阈值) | 跨任务误召回数")
    for T in THRESHOLDS:
        rec_hit = sum(1 for p in same_nonadj if p[0] >= T)
        above = [p for p in pairs if p[0] >= T]
        prec_hit = sum(1 for p in above if p[3])
        cross = sum(1 for p in above if not p[3])
        rec = rec_hit / len(same_nonadj) if same_nonadj else 0
        prec = prec_hit / len(above) if above else 0
        print(f" {T:.2f} | {rec_hit}/{len(same_nonadj)} = {rec:5.0%}      | {prec_hit}/{len(above)} = {prec:5.0%}            | {cross}")

    # 6) 看最危险的「跨任务高相似」前几对（这些会当候选喂 LLM，靠 LLM 按交付物驳回）
    cross_hi = sorted([p for p in pairs if not p[3]], key=lambda x: -x[0])[:6]
    print("\n最高相似的『跨任务』对（潜在误召回，需 LLM 按交付物驳回）：")
    for s, i, j, _, _ in cross_hi:
        print(f"  sim={s:.3f}  #{i}「{titles[atom_ref[i]][:18]}」 ⟷ #{j}「{titles[atom_ref[j]][:18]}」")

    # 7) 看跨断档同任务对里相似度最低的几对（bge-m3 也救不回的，会漏召）
    miss = sorted(same_nonadj, key=lambda x: x[0])[:6]
    print("\n跨断档同任务对里相似度最低的（bge-m3 会漏召、仍得靠 LLM 全局读）：")
    for s, i, j, _, _ in miss:
        print(f"  sim={s:.3f}  #{i} ⟷ #{j}  同属「{titles[atom_ref[i]][:18]}」")


if __name__ == "__main__":
    main()
