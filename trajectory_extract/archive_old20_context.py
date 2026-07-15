#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重写「老版20条三腿replay上下文归档.md」。
本轮改动（张耀明 2026-07-06 要求）：把每条 ①输入 里的「上文」从
原来的「触发句+前2条群上下文」改成三段可读说明：
  【当时在处理什么 thread】 ← session_route_full.json 的 session title
  【baseline 该复现的原错误】 ← 137 快照 corrections[].what（10条有真原文）/
                               无真快照的用蒸馏准则反推并如实标注 + 本次 baseline 判官观测
  【为什么被纠正】          ← 137 快照 source_comment（真原话）/ 无则用蒸馏准则说明
数据源全部落盘、可复跑。
"""
import json, re

BASE = "/opt/shared/data/task-trajectory"
OUT = f"{BASE}/老版20条三腿replay上下文归档.md"
_BASE_SYS = "你是一个严谨的 AI 助手。"

def norm(s): return re.sub(r"\s+", "", (s or ""))

full = json.load(open(f"{BASE}/replay_option1_full.json"))
rows = full["results"]
placebo_rule = full["placebo_rule"]
model = full["model"]; repeat = full["repeat"]

snaps = [json.loads(l) for l in open(f"{BASE}/pre_instruction_snapshots.jsonl") if l.strip()]
by_instr = {norm(s.get("original_instruction")): s for s in snaps}

route = json.load(open(f"{BASE}/session_route_full.json"))
sess_title = {s["id"]: s["title"] for s in route["sessions"]}
decisions = route["decisions"]

def find_thread(instr):
    key = norm(instr)[:14]
    if not key:
        return None
    for de in decisions:
        if key in norm(de.get("text")):
            return sess_title.get(de.get("session"))
    return None

def leg_failed(r, k): return r["legs"][k]["failed"]

def code_block(text):
    # 4 反引号外层 fence，防回复内自带 ``` 截断
    return "````text\n" + (text or "").rstrip() + "\n````"

def rep(r, k):
    leg = r["legs"][k]
    s = leg["samples"][0]
    v = s.get("verdict", {})
    fc, n = leg.get("failed_count"), leg.get("n")
    flag = "翻车" if leg["failed"] else "没翻"
    return flag, fc, n, v.get("reason", ""), s.get("response", "")

L = []
w = L.append
w("# 老版20条三腿replay·输入+输出全档")
w("")
w(f"模型 **{model}**（生成三腿回复+行为判官同一个，验机制用非最终真模型）｜repeat={repeat}｜总账 **2/20 达标**（第 4、7 条）")
w("")
w("## 三腿到底喂了模型什么（读前必看）")
w("")
w("每条 replay 给模型 **system + user 两段**：")
w(f"- **user 消息（三腿完全相同）** = 当时那句触发指令原文。")
w(f"- **system 底座（三腿相同）** = `{_BASE_SYS}`")
w("- **三腿唯一差别 = 底座后注入的那行【行为准则】**：🅰 baseline 不注；🅱 treatment 注本条蒸馏出的纠错准则；🅾 placebo 注一条与错误无关的正经规则做对照。")
w("- **达标** = 🅰翻车 ∧ 🅱没翻 ∧ 🅾仍翻。")
w("")
w("每条 ①输入 开头的【当时在处理什么 thread / baseline 该复现的原错误 / 为什么被纠正】说明来源：thread 主题取自会话路由的 session 标题；原错误与纠正原话取自 137 前置快照（`corrections[].what` / `source_comment`），**10/20 有快照真原文**，另 10 条当年走的是更早批次快照、已被覆盖，无原始 what 原文，则用当年蒸馏出的准则反推错误类型并**如实标注**，另附本次 baseline 腿判官的实际观测。")
w("")
w("---")
w("")

for i, r in enumerate(rows, 1):
    instr = r["original_instruction"]
    snap = by_instr.get(norm(instr))
    thread = find_thread(instr) or "（消息流未定位到所属 thread）"
    principle = r["comment_for_replay"]

    达标 = leg_failed(r, "baseline") and not leg_failed(r, "treatment") and leg_failed(r, "placebo")
    note = r.get("verdict_note", "")
    head = "✅达标" if 达标 else "❌不达标"
    w(f"## {i}. {head} — {note}")
    w("")
    w("### ① 喂给模型的输入")
    w("")
    # 三段说明
    w(f"**【当时在处理什么 thread】** {thread}")
    w("")
    if snap:
        whats = [c.get("what") for c in snap.get("corrections", []) if c.get("what")]
        sc = snap.get("source_comment")
        corrector = None
        for c in snap.get("corrections", []):
            if c.get("corrector"):
                corrector = c["corrector"]; break
        w("**【baseline 该复现的原错误】**（137 快照真原文）")
        for x in whats:
            w(f"- {x}")
        bf, fc, n, reason, _ = rep(r, "baseline")
        w(f"- ▸ 本次 baseline 判官观测：**{bf}**（{fc}/{n}）— {reason}")
        w("")
        w(f"**【为什么被纠正】**（{corrector or '用户'} 当时的纠正原话，137 快照 source_comment）")
        w(f"> {sc or '（快照未记 source_comment）'}")
    else:
        w("**【baseline 该复现的原错误】**（该条无 137 快照真原文，以下为当年蒸馏准则反推的错误类型）")
        w(f"- 反推：违反了准则「{principle}」所约束的行为")
        bf, fc, n, reason, _ = rep(r, "baseline")
        w(f"- ▸ 本次 baseline 判官观测：**{bf}**（{fc}/{n}）— {reason}")
        w("")
        w("**【为什么被纠正】**（无快照原话，据蒸馏准则说明）")
        w(f"> 当年这处纠错蒸馏出的行为准则为：{principle}")
    w("")
    w(f"**触发指令原文（三腿相同的 user 消息）**：")
    w(code_block(instr))
    w("")
    w("**三腿各自的 system（只有【行为准则】那行不同）**：")
    w(f"- 🅰 baseline：`{_BASE_SYS}`（无准则）")
    w(f"- 🅱 treatment：`{_BASE_SYS}` + `【行为准则】{principle}`")
    w(f"- 🅾 placebo：`{_BASE_SYS}` + `【行为准则】{placebo_rule}`")
    w("")
    w("### ② 三腿实际输出")
    w("")
    for k, tag, expect in [("baseline", "🅰 BASELINE", "应翻车才有可测的错"),
                            ("treatment", "🅱 TREATMENT", "应没翻=准则起效"),
                            ("placebo", "🅾 PLACEBO", "应仍翻=排除安慰剂")]:
        flag, fc, n, reason, resp = rep(r, k)
        ok = "✓合预期" if ((k == "treatment" and flag == "没翻") or (k != "treatment" and flag == "翻车")) else "✗违预期"
        w(f"**{tag}** — 判官裁决：{flag}（{fc}/{n}）·{ok}（{expect}）｜理由：{reason}")
        w(code_block(resp))
        w("")
    w("---")
    w("")

open(OUT, "w").write("\n".join(L))
print("written", OUT, "bytes", len("\n".join(L).encode()))
print("137 hits:", sum(1 for r in rows if norm(r["original_instruction"]) in by_instr))
print("thread located:", sum(1 for r in rows if find_thread(r["original_instruction"])))
