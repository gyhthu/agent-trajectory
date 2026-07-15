#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重构 user 输入的三腿 replay（张耀明 2026-07-06 定的真需求；07-06 补回 placebo）。

和三腿 replay 的关键区别：**不改 system、改 user**。
每条：
  baseline  : system=朴素底座, user = 原始 user 输入（逐字）         → 应复现原错
  placebo   : system=朴素底座, user = 盲重写（同等清晰、不织纠错要素）→ 控制「仅重写」混淆变量
  treatment : system=朴素底座, user = 按当时错误重构后的清晰输入      → 看能否让模型做对
达标 = baseline 翻车 ∧ treatment 没翻。
信号隔离（纠错要素真起作用）= baseline 翻 ∧ treatment 没翻 ∧ placebo 仍翻；
若 placebo 也没翻 → 「仅把话重写清楚」就够、纠错要素非必需（混淆，需警示）。
placebo 盲于 err_hint，只按字面把原话说清楚、不加原文没有的要求，用来和 treatment 对照。

数据源：
  - 默认兼容 replay_option1_full.json 的 20 条（每条都有逐字 original_instruction）；
  - --snapshots 可直接读取 pre_instruction_snapshots.jsonl，用 T0 前上下文复现当时场景。
错误信号来自快照中 replay-safe 的 distilled general；没有合法纠错要素则 treatment 退化为 placebo。
重构 prompt 现跑现生成（同一 REPLAY_MODEL），落盘可复跑。
判官/模型复用 replay_three_leg。
"""
import argparse, json, os, re, sys, time
import replay_three_leg as R
import pre_instruction_snapshot as PIS

BASE = "/opt/shared/data/task-trajectory"
MODEL = os.environ.get("REPLAY_MODEL", "deepseek-v3.2")
REPEAT = int(os.environ.get("REPLAY_REPEAT", "3"))
# 只跑指定 1-based 序号（逗号分隔）验证用；空=全跑。子集跑落独立报告名，不覆盖全量正本。
IDXS = os.environ.get("REPLAY_IDXS", "").strip()
SEL = {int(x) for x in IDXS.split(",") if x.strip()} if IDXS else None
_tag = f"_idx{'-'.join(map(str, sorted(SEL)))}" if SEL else ""
DEFAULT_OUT = f"{BASE}/reconstruct_twoleg_report{_tag}.json"
DEFAULT_SENT = f"{BASE}/RECONSTRUCT_DONE{_tag}"

def norm(s): return re.sub(r"\s+", "", (s or ""))

def build_err_hint(snap, r):
    """构造 treatment 的错误提示（唯一事实源）。
    ①剔 post_t0：post_t0_failure_evidence 标记的下游裁决 correction 不进 err_hint；
    ②改喂 distilled general：用快照 distill_audit 里已剥离实例答案的通用准则，不用原始 what。
    回退：无快照→蒸馏 comment_for_replay；有快照但过滤后无干净 general→空串
         （treatment 无合法纠错要素可织，退化为 placebo，如实标记而非硬塞下游结论）。
    """
    if not snap:
        return r["comment_for_replay"], "蒸馏准则(无快照)"
    # 共用快照侧同一个过滤器（①剔 post_t0 ②喂 distilled general），杜绝两处逻辑漂移。
    generals = PIS.clean_general(
        (snap.get("leakage_guard") or {}).get("distill_audit"),
        snap.get("post_t0_failure_evidence"))
    npt = len(snap.get("post_t0_failure_evidence") or [])
    if generals:
        return "；".join(generals), f"快照distilled-general(剔{npt}条post_t0)"
    return "", f"过滤后无干净general(post_t0={npt})→treatment退化placebo"

_REWRITE_SYS = """你是 bot 团队的「需求规格师」。给你一条飞书工作群里的【原始用户输入】，以及这条输入当时导致执行 bot 犯的【错误】。
你的任务：站在任务发起人的口吻（第二人称「你」），把这条需求**一次交代清楚**，让执行 bot 不会再犯这个错。
按错误性质自行织入相关要素（只织相关的）：背景前提 / 团队偏好（先看代码别猜、报数必真跑亲核、要轻方案）/ 防幻觉（没做就说没做、不许编数）/ 防疑神疑鬼 / 验收判据 / 范围边界 / 复用优先（先 grep 现成的）/ 权限边界 / 术语说人话。
要求：正面把正确做法和边界说死，不要写成「请注意别犯X错」的空泛训诫；2~6 句；具体可直接当任务交代用。
只输出重构后的用户输入本身，不要解释、不要 markdown。"""

def rewrite(client, original, err_hint):
    user = f"【原始用户输入】\n{original}\n\n【当时导致的错误】\n{err_hint}\n\n现在输出重构后的、交代清楚的用户输入："
    return R._chat(client, MODEL, _REWRITE_SYS, user).strip()

# placebo：盲于当时错误，只把原话「说清楚」，语气/详尽程度对齐 treatment，但不加任何原文没有的要求。
# 作用=隔离「仅重写清晰/写长」这个混淆变量：若 placebo 也能救回，说明起作用的是重写本身而非纠错要素。
_PLACEBO_SYS = """你是 bot 团队的「需求规格师」。给你一条飞书工作群里的【原始用户输入】。
请站在任务发起人的口吻（第二人称「你」），把这条需求**用清楚、完整的话重新表述一遍**，让执行 bot 一看就懂要做什么。
只按字面意思把原话说明白，**不要添加原文没有的具体要求、约束、背景或验收标准**——不知道的别补，别替发起人做决定。
2~6 句；只输出重写后的用户输入本身，不要解释、不要 markdown。"""

def rewrite_placebo(client, original):
    user = f"【原始用户输入】\n{original}\n\n现在输出清楚重述后的用户输入："
    return R._chat(client, MODEL, _PLACEBO_SYS, user).strip()


def _load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def _load_rows(args):
    snaps = _load_jsonl(args.snapshots)
    by_instr = {norm(s.get("original_instruction")): s for s in snaps}
    if args.source == "snapshots":
        rows = [{
            "idx": i,
            "original_instruction": s.get("original_instruction", ""),
            "comment_for_replay": s.get("comment_for_replay", ""),
            "_snapshot": s,
        } for i, s in enumerate(snaps, 1) if s.get("status") == "ok"]
        return rows, by_instr
    rows = json.load(open(args.input, encoding="utf-8"))["results"]
    return rows, by_instr


def _scene_from_snapshot(snap, rewritten_prompt=None):
    """Render the same T0-before scene as the snapshot, optionally replacing T0."""
    if not snap:
        return None
    context = ((snap.get("prompts") or {}).get("baseline") or "")
    original = snap.get("original_instruction") or ""
    if not rewritten_prompt or not original or original not in context:
        return context
    marker = "【原始指令】\n"
    pos = context.find(marker)
    if pos < 0:
        return context.replace(original, rewritten_prompt, 1)
    start = pos + len(marker)
    end = context.find("\n\n", start)
    if end < 0:
        end = len(context)
    t0_text = context[start:end]
    return context[:start] + t0_text.replace(original, rewritten_prompt, 1) + context[end:]


def _scene_for_leg(snap, original, prompt):
    if snap:
        scene = _scene_from_snapshot(snap, prompt)
        if scene:
            return scene
    return prompt or original


def classify_legs(base_failed, placebo_failed, treat_failed, has_err_hint):
    """Classify replay result; no err_hint means treatment has no correction signal."""
    baseline_reproduced = bool(base_failed)
    passed = bool(has_err_hint and base_failed and treat_failed is False)
    is_isolated = passed and (placebo_failed is True)
    is_confound = passed and (placebo_failed is False)
    if not has_err_hint:
        note = "无合法纠错要素：treatment 退化 placebo，只统计 baseline 是否复现，不计纠错有效"
    elif is_isolated:
        note = "达标+信号隔离：原话翻车、纠错重构做对、盲重写仍翻（纠错要素起作用）"
    elif is_confound:
        note = "达标但混淆：盲重写也做对了 → 仅把话说清楚就够、纠错要素非必需"
    elif not base_failed:
        note = "baseline 未复现原错 → 无可测的错"
    elif treat_failed:
        note = "重构后仍翻车 → 重构没救回"
    else:
        note = "判定不全"
    return {
        "has_err_hint": bool(has_err_hint),
        "baseline_reproduced": baseline_reproduced,
        "passed": passed,
        "signal_isolated": is_isolated,
        "confounded": is_confound,
        "note": note,
    }


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["legacy", "snapshots"], default="legacy")
    ap.add_argument("--input", default=f"{BASE}/replay_option1_full.json")
    ap.add_argument("--snapshots", default=f"{BASE}/pre_instruction_snapshots.jsonl")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--sentinel", default=DEFAULT_SENT)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rows, by_instr = _load_rows(args)

    client = R._client()
    out = []
    ok = 0            # 达标：base 翻 ∧ treat 没翻
    isolated = 0      # 信号隔离：base 翻 ∧ treat 没翻 ∧ placebo 仍翻
    confound = 0      # 混淆：base 翻 ∧ treat 没翻 ∧ placebo 也没翻（仅重写就够）
    n_run = 0
    for i, r in enumerate(rows, 1):
        if SEL and i not in SEL:
            continue
        n_run += 1
        original = r["original_instruction"]
        snap = r.get("_snapshot") or by_instr.get(norm(original))
        err_hint, err_src = build_err_hint(snap, r)

        t0 = time.time()
        placebo = rewrite_placebo(client, original)
        # err_hint 为空=无合法纠错要素，treatment 与 placebo 同构（如实退化，不硬造）。
        recon = rewrite(client, original, err_hint) if err_hint else placebo
        base_scene = _scene_for_leg(snap, original, original)
        placebo_scene = _scene_for_leg(snap, original, placebo)
        treatment_scene = _scene_for_leg(snap, original, recon)
        base_leg = R._run_leg(client, MODEL, R._BASE_SYS, base_scene, REPEAT)
        plac_leg = R._run_leg(client, MODEL, R._BASE_SYS, placebo_scene, REPEAT)
        treat_leg = R._run_leg(client, MODEL, R._BASE_SYS, treatment_scene, REPEAT)
        bf, pf, tf = base_leg["failed"], plac_leg["failed"], treat_leg["failed"]
        classification = classify_legs(bf, pf, tf, bool(err_hint))
        passed = classification["passed"]
        is_isolated = classification["signal_isolated"]
        is_confound = classification["confounded"]
        ok += passed
        isolated += is_isolated
        confound += is_confound
        rec = {
            "idx": i, "original_instruction": original, "err_hint": err_hint,
            "err_src": err_src, "reconstructed_prompt": recon, "placebo_prompt": placebo,
            "scene_source": "snapshot" if snap else "instruction_only",
            "t0_msg_id": ((snap or {}).get("t0") or {}).get("msg_id"),
            "baseline": base_leg, "placebo": plac_leg, "treatment": treat_leg,
            "base_failed": bf, "placebo_failed": pf, "treat_failed": tf,
            **classification,
        }
        out.append(rec)
        print(f"[{i}] {int(time.time()-t0)}s base={bf} placebo={pf} treat={tf} "
              f"passed={passed} isolated={is_isolated} src={err_src} | {original[:24]!r}", flush=True)
        json.dump({"model": MODEL, "repeat": REPEAT, "n": n_run, "selected": sorted(SEL) if SEL else None,
                   "source": args.source, "snapshots": args.snapshots,
                   "passed_count": ok, "isolated_count": isolated, "confound_count": confound,
                   "results": out},
                  open(args.out, "w"), ensure_ascii=False, indent=2)

    open(args.sentinel, "w").write(f"passed {ok}/{n_run} isolated {isolated} confound {confound}")
    print(f"DONE passed {ok}/{n_run} isolated {isolated} confound {confound} → {args.out}", flush=True)

if __name__ == "__main__":
    main()
