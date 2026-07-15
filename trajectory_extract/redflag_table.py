#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全量红旗表（张耀明 2026-07-08 拍：先跑红旗全量表）。

对全 137 条捕获链结果跑**能自动判的信号**，产出一张预警表——
每条标它踩没踩红旗、踩了哪条。**红旗只是初判/预警灯，不下结论**：
亮灯=值得开原文细核，不等于该踢（idx127 就是 gt_by≠本bot 亮灯但纠正真针对本 bot 的假阳性）。

信号（都便宜、纯规则/字段查，不调 LLM）：
  R1 gt_by≠本bot   : 被当标答的「下一条 bot 回复」作者不是 lian-server（只 qa 侧有 GT 候选才可判）
                     → 纠正主体八成是别的 bot（idx64/66 一类误抓），但也可能只是 GT 抽错作者(假阳性)。
  R2 纠正提别的bot : correction.what 里点名 codex/antigravity/zym → R1 照不到 exec 侧时的主体错配代理信号。
  R3 寒暄          : original_instruction 像问候（你好/hi/hello/在吗）。
  R4 叫停/中止      : 像喊停（!stop/停/别发了/够了/取消/闭嘴）或情绪叫停。
  R5 元层追问自身    : 像追问 bot 自己上一轮（为什么你/你确定/你刚/你思考…为什么）——drop(a) 型。

正典对齐：pre_instruction_snapshots.jsonl 行序 == idx（已验 mismatch=0）。
铁律 fail-loud：任何输入文件缺失/idx 对不齐立即抛错，不静默兜底。
"""
import json, os, re, sys
from collections import defaultdict

DATA = "/opt/shared/data/task-trajectory"
SNAP = f"{DATA}/pre_instruction_snapshots.jsonl"
RTYPE = f"{DATA}/request_type_2class.jsonl"
GTCAND = f"{DATA}/qa_ground_truth_candidates.json"
OUT_MD = f"{DATA}/全量红旗表_137.md"
OUT_JSONL = f"{DATA}/redflag_table_137.jsonl"

SELF_BOT = "lian-server"        # 本 bot 标识（gt_by 里的写法：claude(lian-server)）
OTHER_BOTS = ("codex", "antigravity", "zym")

# —— 规则模式（粗筛，宁可多亮灯让人再看，不追求精确）——
RE_GREET = re.compile(r"(^|\s)(你好|您好|hi|hello|hey|在吗|早上好|早|哈喽)\b", re.I)
RE_STOP = re.compile(r"(!stop|/stop|停一下|停下|别发了|别发|够了|取消|闭嘴|打住|先停|停止发|你怎么这么蠢)")
RE_META = re.compile(r"(为什么你|你为什么|你确定|你刚才?为什么|你思考时|你怎么(又|会|能)|你动不动)")


def load_jsonl(p):
    if not os.path.exists(p):
        sys.exit(f"[FATAL] 缺文件 {p}")
    return [json.loads(l) for l in open(p) if l.strip()]


def main():
    snaps = load_jsonl(SNAP)
    rtype = load_jsonl(RTYPE)
    if not os.path.exists(GTCAND):
        sys.exit(f"[FATAL] 缺文件 {GTCAND}")
    gtcand = json.load(open(GTCAND))

    if len(snaps) != 137:
        sys.exit(f"[FATAL] snapshot 不是 137 条，实为 {len(snaps)}")

    rt_by_idx = {r["idx"]: r for r in rtype}
    # 验对齐
    for i, s in enumerate(snaps):
        r = rt_by_idx.get(i)
        if not r or r.get("t0_msg_id") != s["t0"]["msg_id"]:
            sys.exit(f"[FATAL] idx {i} 对不齐 request_type")

    # idx -> set(gt_by)
    gt_by_idx = defaultdict(set)
    for c in gtcand:
        gt_by_idx[c["idx"]].add(c.get("gt_by") or "?")

    rows = []
    for i, s in enumerate(snaps):
        r = rt_by_idx[i]
        cls = r["class"]
        lane = s.get("lane", "")
        instr = (s.get("original_instruction") or "").strip()
        corrs = s.get("corrections") or []
        whats = " ｜ ".join((c.get("what") or "") for c in corrs)
        correctors = ",".join(sorted({(c.get("corrector") or "?") for c in corrs}))
        gtset = gt_by_idx.get(i, set())

        flags = []
        # R1 gt_by≠本bot（只有 qa 侧有 GT 候选才可判）
        r1_applicable = len(gtset) > 0
        if r1_applicable and not any(SELF_BOT in (g or "") for g in gtset):
            flags.append("R1:gt_by≠本bot")
        # R2 纠正 what 里点名别的 bot
        if any(b in whats.lower() for b in OTHER_BOTS):
            flags.append("R2:纠正提别bot")
        # R3 寒暄
        if RE_GREET.search(instr):
            flags.append("R3:寒暄")
        # R4 叫停
        if RE_STOP.search(instr) or RE_STOP.search(whats):
            flags.append("R4:叫停")
        # R5 元层追问自身
        if RE_META.search(instr):
            flags.append("R5:元追问")

        rows.append({
            "idx": i, "class": cls, "lane": lane,
            "moved_from": r.get("moved_from") or "",
            "n_corr": len(corrs), "correctors": correctors,
            "gt_by": sorted(gtset),
            "flags": flags, "n_flags": len(flags),
            "instruction": instr, "correction_what": whats,
        })

    # 落 jsonl
    with open(OUT_JSONL, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 统计
    flagged = [r for r in rows if r["flags"]]
    by_flag = defaultdict(list)
    for r in flagged:
        for fl in r["flags"]:
            by_flag[fl.split(":")[0]].append(r["idx"])
    by_class_flagged = defaultdict(int)
    for r in flagged:
        by_class_flagged[r["class"]] += 1

    # 落 md：按 class 分组，组内亮灯优先、按 idx
    def esc(t):
        return (t or "").replace("|", "／").replace("\n", " ").strip()

    lines = []
    lines.append("# 全量红旗表 · 137 条（张耀明 2026-07-08）\n")
    lines.append("> **红旗=预警灯，只是初判、不下结论**：亮灯=值得开原文细核，≠该踢。")
    lines.append("> 假阳性实例 idx127（R1 亮但纠正真针对本 bot、应归意图类）。\n")
    lines.append("## 汇总\n")
    lines.append(f"- 全 137 条，**亮灯 {len(flagged)} 条**，其中 "
                 f"qa {by_class_flagged.get('qa',0)} / exec {by_class_flagged.get('exec',0)} / drop {by_class_flagged.get('drop',0)}")
    for fl in ("R1", "R2", "R3", "R4", "R5"):
        ids = sorted(set(by_flag.get(fl, [])))
        name = {"R1": "gt_by≠本bot", "R2": "纠正提别bot", "R3": "寒暄", "R4": "叫停", "R5": "元追问"}[fl]
        lines.append(f"- **{fl} {name}**：{len(ids)} 条 — idx {ids}")
    lines.append("")

    for cls in ("qa", "exec", "drop"):
        crows = [r for r in rows if r["class"] == cls]
        crows.sort(key=lambda r: (-r["n_flags"], r["idx"]))
        cf = [r for r in crows if r["flags"]]
        lines.append(f"\n## {cls}（{len(crows)} 条，亮灯 {len(cf)}）\n")
        lines.append("| idx | 红旗 | 挪自 | gt_by | 纠正者 | instruction | correction.what |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in crows:
            fl = "🚩" + " ".join(x.split(":")[1] for x in r["flags"]) if r["flags"] else ""
            gtb = ",".join(g.replace("claude(lian-server)", "本bot") for g in r["gt_by"]) or "-"
            lines.append(f"| {r['idx']} | {fl} | {r['moved_from']} | {esc(gtb)} | {esc(r['correctors'])} "
                         f"| {esc(r['instruction'])[:44]} | {esc(r['correction_what'])[:60]} |")

    open(OUT_MD, "w").write("\n".join(lines) + "\n")

    print(f"[ok] 137 条，亮灯 {len(flagged)} 条")
    for fl in ("R1", "R2", "R3", "R4", "R5"):
        ids = sorted(set(by_flag.get(fl, [])))
        print(f"  {fl}: {len(ids)}  idx={ids}")
    print(f"  按类亮灯: qa={by_class_flagged.get('qa',0)} exec={by_class_flagged.get('exec',0)} drop={by_class_flagged.get('drop',0)}")
    print(f"[out] {OUT_MD}")
    print(f"[out] {OUT_JSONL}")


if __name__ == "__main__":
    main()
