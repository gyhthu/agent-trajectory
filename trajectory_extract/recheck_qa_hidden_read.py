#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全量复检 qa 集，揪「表面问答、实则答对必须先读盘/查历史产物/查代码/查 wire」的隐性 exec
（张耀明 2026-07-07：idx20 就是这类误判——'codex 呢'表面像问答，答对却必须开盘查 wire）。

判据（和 request_type 的 exec 定义一致）：答对的前提是否**必须先去读本机的具体文件/历史产物/
代码/wire 抓包才能确认**。是→隐性 exec（误判），凭通用知识+对话上下文就能答→真 qa。
只重判、不改原产物；输出候选清单供人工/张耀明过目。
"""
import json, re, sys
import replay_three_leg as R
import replay_reconstruct as RR

BASE = "/opt/shared/data/task-trajectory"
MODEL = "deepseek-v3.2"
CTX_N = 20

_SYS = """你在复检一条历史「用户要求」当初被归成 qa（问答类）对不对。
只判一件事：**要正确回答这条，模型是不是必须先去读「这台机器/这个项目」的具体文件、历史产物、
代码或 wire 抓包才能答准？**
- 若答对的前提是先读盘核对本地具体东西（某文件内容、某段代码实现、某次抓包、bot 自己之前产出的
  prompt/脚本/数据的实际内容）→ hidden_exec=true（其实是 exec、误判成了 qa）。
  典型：问"某本群 bot 的运行轨迹能不能截获""你之前构造的X覆盖了Y没""某实现走的是哪条通路"——
  答案是项目特定的实现事实，凭通用知识猜不准、必须开盘查。
- 若凭通用知识 / 概念理解 / 已给的对话上下文就能答对，不需要去读本地具体文件 → hidden_exec=false（真 qa）。
  典型：解释一个通用概念、讨论设计思路、问业界通行做法、基于上下文already给出的信息作判断。
只输出一行 JSON：{"hidden_exec": true/false, "reason": "≤25字中文"}"""


def render_ctx(snap, n=CTX_N):
    if not snap:
        return ""
    ev = ((snap.get("context") or {}).get("events") or [])[-n:]
    lines = [f"{e.get('name') or e.get('role') or '?'}：{(e.get('text') or '').strip()}"
             for e in ev if (e.get("text") or "").strip()]
    return "【对话上下文】\n" + "\n".join(lines) if lines else ""


def judge(client, instr, ctx):
    user = f"{ctx}\n\n【被复检的用户要求】\n{instr}\n\n只输出 JSON。" if ctx else \
           f"【被复检的用户要求】\n{instr}\n\n只输出 JSON。"
    raw = R._chat(client, MODEL, _SYS, user)
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return {"hidden_exec": None, "reason": raw[:60]}
    try:
        d = json.loads(m.group(0))
        return {"hidden_exec": bool(d.get("hidden_exec")), "reason": str(d.get("reason", ""))[:60]}
    except Exception:
        return {"hidden_exec": None, "reason": raw[:60]}


def main():
    rw = json.load(open(f"{BASE}/rewrite_thread_result.json"))["rewritten"]
    snaps = [json.loads(l) for l in open(f"{BASE}/pre_instruction_snapshots.jsonl") if l.strip()]
    by_instr = {RR.norm(s.get("original_instruction")): s for s in snaps}
    cls = {r["idx"]: r for r in (json.loads(l) for l in open(f"{BASE}/request_type_2class.jsonl") if l.strip())}

    client = R._client()
    qa_idxs = [i for i, r in cls.items() if r.get("class") == "qa"]
    print(f"复检 {len(qa_idxs)} 条 qa", flush=True)
    flagged = []
    for i, idx in enumerate(sorted(qa_idxs)):
        e = rw[idx]
        instr = e["instruction_to_fix_text"]
        snap = by_instr.get(RR.norm(instr))
        v = judge(client, instr, render_ctx(snap))
        v["idx"] = idx
        v["instruction"] = instr[:120]
        if v["hidden_exec"] is True:
            flagged.append(v)
        print(f"[{i+1}/{len(qa_idxs)}] idx{idx} hidden_exec={v['hidden_exec']} {v['reason']}", flush=True)

    out = f"{BASE}/qa_recheck_hidden_read.json"
    json.dump({"flagged": flagged, "n_qa": len(qa_idxs)}, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"\nDONE 疑似隐性 exec {len(flagged)}/{len(qa_idxs)} → {out}", flush=True)
    for f in flagged:
        print(f"  idx{f['idx']}: {f['reason']} | {f['instruction'][:50]}", flush=True)


if __name__ == "__main__":
    main()
