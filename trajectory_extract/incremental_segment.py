#!/usr/bin/env python3
"""Incremental task segmentation for Feishu group trajectories.

This module is an orchestration layer over the existing pipeline:
task_stitch.atomic_segments -> llm_segment.llm_segment/assemble_clusters ->
llm_segment.render_decompose.  It keeps stable state by Feishu msg_id instead
of position-based segment numbers, so frozen historical tasks do not drift when
new messages arrive.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm_segment as seg  # noqa: E402
import task_stitch as ts  # noqa: E402
import export_tasks as et  # noqa: E402  # 切分完自动导出 md/jsonl 供下游

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from regex_anonymizer import RegexAnonymizer  # noqa: E402


STATE_VERSION = 1
DEFAULT_THAW_HOURS = int(os.environ.get("INCR_THAW_HOURS", "24"))
DEFAULT_STATE_DIR = Path(os.environ.get("INCR_STATE_DIR", str(ts.SHARED / "state")))


def _state_path(group_id: str, state_dir: Path = DEFAULT_STATE_DIR) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", group_id)
    return state_dir / f"{safe}.json"


def _load_state(group_id: str, state_dir: Path = DEFAULT_STATE_DIR) -> dict | None:
    path = _state_path(group_id, state_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_state(group_id: str, state: dict, state_dir: Path = DEFAULT_STATE_DIR) -> Path:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = _state_path(group_id, state_dir)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    return path


def _event_msg_ids(events: list[dict]) -> list[str]:
    return sorted({e.get("msg_id", "") for e in events if e.get("msg_id")})


def _last_ts(events: list[dict]) -> float:
    return max((float(e.get("ts") or 0) for e in events), default=0.0)


def _carry_forward_titles(clusters: list[list[dict]], metas: list[dict],
                          anchors: list[dict], thresh: float = 0.6) -> int:
    """跨增量窗稳定任务标题：把这一窗新切出的簇按 msg_id 重叠匹配到上一窗的同一任务，
    重叠足够高就**沿用上一窗的标题**，根治「同一任务每窗换一个新标题、串不成时间线」。

    判据用「旧任务的 msg_id 被新簇包含的比例」`|旧∩新|/|旧|`，不用 Jaccard——
    任务在增量里只增不减（活动尾巴会越长越大），用包含率比 Jaccard 稳：尾巴长大后
    Jaccard 会被新增消息稀释掉，但旧消息几乎全留在新簇里，包含率仍接近 1。

    **只改 title 字符串，绝不动成员归属/任务边界**——对评测金标（按边界/归属判）中性。
    返回实际沿用（改写）了几条标题。"""
    if not anchors:
        return 0
    used: set[int] = set()
    changed = 0
    for i, cluster in enumerate(clusters):
        cset = set(_event_msg_ids(cluster))
        if not cset:
            continue
        best_j, best_score = -1, 0.0
        for j, a in enumerate(anchors):
            if j in used or not a.get("title"):
                continue
            aset = set(a.get("msg_ids") or [])
            if not aset:
                continue
            score = len(aset & cset) / len(aset)   # 旧任务被新簇包含的比例
            if score > best_score:
                best_score, best_j = score, j
        if best_j >= 0 and best_score >= thresh:
            old_title = anchors[best_j].get("title") or ""
            if old_title and metas[i].get("title") != old_title:
                metas[i]["title"] = old_title
                changed += 1
            used.add(best_j)
    return changed


def _title_anchors_from(records: list[dict]) -> list[dict]:
    """从任务记录（_task_record 产物，含 title + member_msg_ids）抽标题锚，供下一窗沿用。"""
    return [{"msg_ids": r.get("member_msg_ids") or [], "title": r.get("title") or ""}
            for r in records]


def _meaningful_tail(events: list[dict]) -> dict | None:
    for e in sorted(events, key=lambda x: x.get("ts") or 0, reverse=True):
        if not ts._is_noise(e.get("text", "")) and ts._strip_feishu(e.get("text", "")).strip():
            return e
    return None


def confirmation_status(events: list[dict], terminal: bool, delivery: str | None = None) -> str:
    """增量态的确认标签（与 llm_decompose 的离线 status 质量标签**正交**：那条判执行质量，这条判闭环确认态）。

    按张耀明 2026-06-29 两条决策 + 乙路线（交付判断搭 decompose 的 LLM 便车）：
      · 任务未终结（观测窗右沿、可能还在飞）              → 未终结
      · 最后一条是【用户】且非收尾语 → bot 没接              → 未确认（决策1：bot 欠一次回复，
        轨迹末尾停在用户那条请求上、bot 的 👀 占位是噪声被滤）
      · 最后一条是【用户】且含收尾语(_CLOSURE)→ 用户确认   → 已完成
      · 最后一条是【bot】：不再一律「已完成」——查 delivery（LLM 读全文判 bot 到底交付没）：
          - delivery∈{仅占位,无交付}：bot 末尾只是占位承诺、没实质交付 → 未确认（决策1，修旧版误判）
          - delivery=已交付 或 未知(None，单消息/decompose失败/旧state) → 已完成（决策2：沉默即认可）

    delivery 由 llm_decompose 的同一次 LLM 调用产出，零额外成本；未知时回退到原确定性行为。
    未确认态供后续「主动补回复/对话中主动提醒」逻辑消费（提醒触发本身后面再做）。
    """
    if not terminal:
        return "未终结"
    tail = _meaningful_tail(events)
    if tail is None:
        return "已完成"
    if tail.get("role") == "user":
        t = ts._norm(ts._strip_feishu(tail.get("text", "")))
        if any(k in t for k in ts._CLOSURE):
            return "已完成"        # 用户收尾语 → 已确认完成
        return "未确认"            # bot 没接这条用户消息 → 决策1（结构事实，无歧义）
    # tail 是 bot：交付判断交给 LLM（delivery），不再用「谁最后说话」瞎猜
    if delivery in ("仅占位", "无交付"):
        return "未确认"            # bot 末尾是占位承诺/无实质交付 → 决策1（乙路线修旧版误判）
    return "已完成"                # delivery=已交付 或 未知 → 决策2 沉默即认可


def _task_record(cluster: list[dict], meta: dict, render_md: str, terminal: bool,
                 delivery: str | None = None, subreqs: list[dict] | None = None,
                 decompose_ok: bool = True) -> dict:
    return {
        "title": meta.get("title", ""),
        "goal": meta.get("goal", ""),
        "reason": meta.get("reason", ""),
        "member_msg_ids": _event_msg_ids(cluster),
        "last_ts": _last_ts(cluster),
        "terminal": bool(terminal),
        "delivery": delivery,
        "confirmation_status": confirmation_status(cluster, terminal, delivery),
        "render_md": render_md,
        # 结构化子需求→真 msg_id（C1）：纠错标注按此精确 join 到子需求，不再靠 .md 本地#人工核
        "subreqs": subreqs or [],
        # decompose 成功没（False=429/报错留下的洞）：冻结复用时失败的强制重下钻，v4-pro 恢复后自愈
        "decompose_ok": bool(decompose_ok),
        "events": cluster,
    }


def _decompose_failed(task: dict) -> bool:
    """这个冻结任务的子需求下钻是不是失败了（需要下次增量重跑）。
    新 state：直接看显式 decompose_ok=False。
    旧 state（无该字段）：回退推断——多消息任务(nreal>=2)却 0 子需求 = 当年 429 留的洞。
    单消息任务本就只有 1 个平凡子需求，不会误判。"""
    ok = task.get("decompose_ok")
    if ok is not None:
        return ok is False
    subs = task.get("subreqs") or []
    if subs:
        return False
    nreal = sum(1 for e in (task.get("events") or []) if not ts._is_noise(e.get("text", "")))
    return nreal >= 2


def split_task_sections(decompose_md: str) -> list[str]:
    """Split a combined decompose markdown into per-task sections.

    Header material is intentionally omitted; callers build a fresh combined
    header when merging frozen and active sections.
    """
    sections: list[str] = []
    cur: list[str] = []
    for line in decompose_md.splitlines():
        if line.startswith("## 任务 "):
            if cur:
                sections.append("\n".join(cur).rstrip() + "\n")
            cur = [line]
        elif cur:
            cur.append(line)
    if cur:
        sections.append("\n".join(cur).rstrip() + "\n")
    return sections


def _reopened_frozen_indexes(frozen_tasks: list[dict], new_events: list[dict]) -> set[int]:
    msg_to_idx = {}
    for i, task in enumerate(frozen_tasks):
        for msg_id in task.get("member_msg_ids") or []:
            msg_to_idx[msg_id] = i
    reopened: set[int] = set()
    for e in new_events:
        parent_id = e.get("parent_id")
        if parent_id in msg_to_idx:
            reopened.add(msg_to_idx[parent_id])
    return reopened


def _hard_frozen_indexes(frozen_tasks: list[dict], watermark_ts: float,
                         thaw_hours: int, reopened: set[int]) -> set[int]:
    cutoff = float(watermark_ts or 0) - thaw_hours * 3600
    out = set()
    for i, task in enumerate(frozen_tasks):
        if i in reopened:
            continue
        if _decompose_failed(task):
            continue  # 下钻失败的洞不许硬冻结 → 事件回流 active 区被重下钻，自愈
        if bool(task.get("terminal", True)) and float(task.get("last_ts") or 0) < cutoff:
            out.add(i)
    return out


def _window_desc(events: list[dict]) -> str:
    if not events:
        return "(空窗口)"
    return (time.strftime("%m-%d %H:%M", time.localtime(events[0]["ts"])) + "→" +
            time.strftime("%m-%d %H:%M", time.localtime(events[-1]["ts"])) +
            f"（{len(events)}条原始）")


def segment_events(group_id: str, events: list[dict], model: str | None = None, start_idx: int = 1,
                   prior_anchors: list[dict] | None = None):
    events = sorted(events, key=lambda e: e["ts"])
    atoms = ts.atomic_segments(events, sim_fn=ts.build_reply_sim_fn())
    if not atoms:
        return [], [], "", "", 0, [], []
    anon = RegexAnonymizer()
    result, lines = seg.llm_segment(atoms, anon, model=model or seg._LLM_MODEL)
    clusters, metas = seg.assemble_clusters(atoms, result)
    if prior_anchors:
        # 跨窗标题沿用：在渲染前改 metas[i]["title"]，让 seg_md / decomp_md 都用稳定标题
        _carry_forward_titles(clusters, metas, prior_anchors)
    wd = _window_desc(events)
    seg_md = seg.render(group_id, clusters, metas, lines, result, wd)
    decomp_md, total_sub, deliveries, task_subreqs = seg.render_decompose(
        group_id, wd, clusters, metas, anon, start_idx=start_idx)
    return clusters, metas, seg_md, decomp_md, total_sub, deliveries, task_subreqs


def _combined_decompose_md(group_id: str, window_desc: str, sections: list[str]) -> str:
    body = "\n".join(s.rstrip() for s in sections if s.strip()).rstrip()
    return "\n".join([
        "# 任务 → 子需求　增量两级拆分",
        "",
        f"- 群：`{group_id}`　窗口：{window_desc}",
        f"- 任务数：{len(sections)}（冻结任务原样复用，活动尾巴重新下钻）",
        "",
        "---",
        "",
        body,
        "",
    ])


def _write_outputs(group_id: str, seg_md: str, decomp_md: str, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%m%d_%H%M%S")
    seg_out = out_dir / f"llmsegment_incremental_{group_id[:8]}_{stamp}.md"
    decomp_out = out_dir / f"llmdecompose_incremental_{group_id[:8]}_{stamp}.md"
    seg_out.write_text(seg_md, encoding="utf-8")
    decomp_out.write_text(decomp_md, encoding="utf-8")
    return seg_out, decomp_out


def _build_state(group_id: str, events: list[dict], clusters: list[list[dict]], metas: list[dict],
                 sections: list[str], thaw_hours: int, deliveries: list | None = None,
                 task_subreqs: list | None = None) -> dict:
    terminal = ts.compute_terminal(clusters)
    deliveries = deliveries or []
    task_subreqs = task_subreqs or []
    frozen_tasks = []
    active_tail_events: list[dict] = []
    active_tail_subreqs: list[dict] = []
    for i, (cluster, meta) in enumerate(zip(clusters, metas)):
        is_terminal = terminal[i] if i < len(terminal) else True
        section = sections[i] if i < len(sections) else ""
        delivery = deliveries[i] if i < len(deliveries) else None
        subs = task_subreqs[i].get("subreqs") if i < len(task_subreqs) else None
        ok = task_subreqs[i].get("decompose_ok", True) if i < len(task_subreqs) else True
        # active_tail = compute_terminal 标的那个未终结簇（最近被碰的任务），不靠列表末位
        # （簇按首事件 ts 排序，末位≠最近被碰，交织长任务会错；认 not is_terminal 最稳）。
        if not is_terminal:
            active_tail_events = cluster
            active_tail_subreqs = subs or []  # C1：活动尾巴的子需求→msg_id 也持久化，解 A 的 23 条 join 缺口
            continue
        frozen_tasks.append(_task_record(cluster, meta, section, is_terminal, delivery, subs, ok))
    return {
        "schema_version": STATE_VERSION,
        "group_id": group_id,
        "watermark_ts": _last_ts(events),
        "thaw_hours": thaw_hours,
        "frozen_tasks": frozen_tasks,
        "active_tail_events": active_tail_events,
        "active_tail_subreqs": active_tail_subreqs,
    }


def _export_after_segment(group_id: str, state: dict) -> str | None:
    """切分成功后自动导出该群 frozen_tasks 为 md/jsonl（下游 SFT/分析）。
    fail-soft：导出出错只记警告、绝不拖垮切分主流程（切分才是本脚本的核心产物）。"""
    try:
        res = et.export_group_from_state(state, group_id)
        print(f"[export] {res['tasks']} 任务 → {len(res['md_files'])} md / "
              f"{len(res['jsonl_files'])} jsonl：{res['out_dir']}")
        return res["out_dir"]
    except Exception as ex:  # noqa: BLE001
        print(f"[export] ⚠️ 导出失败（不影响切分）：{ex}", file=sys.stderr)
        return None


def run_incremental(group_id: str, since: int = 0, until: int = 0, hist_file: str | None = None,
                    state_dir: Path = DEFAULT_STATE_DIR, out_dir: Path = ts.SHARED,
                    thaw_hours: int = DEFAULT_THAW_HOURS, model: str | None = None) -> dict:
    state = _load_state(group_id, state_dir)
    start = since
    if state and not hist_file:
        start = int(state.get("watermark_ts") or 0)
    # until<=0 → 取「现在」。飞书 list-messages 的 end_time 不能传 0
    # （那会被当成截止到 epoch 0 → 拉回空），定时跑必须给上界。hist_file 模式走文件、不受此影响。
    if until <= 0 and not hist_file:
        until = int(time.time())
    new_events = ts.fetch_history(group_id, start, until, hist_file)
    if state and not new_events:
        return {"changed": False, "state_path": str(_state_path(group_id, state_dir))}

    if not state:
        events = new_events
        clusters, metas, seg_md, decomp_md, total_sub, deliveries, task_subreqs = segment_events(
            group_id, events, model=model)
        sections = split_task_sections(decomp_md)
        new_state = _build_state(group_id, events, clusters, metas, sections, thaw_hours,
                                 deliveries, task_subreqs)
        new_state["title_anchors"] = [
            {"msg_ids": _event_msg_ids(c), "title": m.get("title") or ""}
            for c, m in zip(clusters, metas)]
        state_path = _write_state(group_id, new_state, state_dir)
        seg_out, decomp_out = _write_outputs(group_id, seg_md, decomp_md, out_dir)
        export_dir = _export_after_segment(group_id, new_state)
        return {
            "changed": True, "mode": "full", "events": len(events), "tasks": len(clusters),
            "subtasks": total_sub, "state_path": str(state_path),
            "segment_path": str(seg_out), "decompose_path": str(decomp_out),
            "export_dir": export_dir,
        }

    frozen = list(state.get("frozen_tasks") or [])
    reopened = _reopened_frozen_indexes(frozen, new_events)
    hard = _hard_frozen_indexes(frozen, float(state.get("watermark_ts") or 0), thaw_hours, reopened)

    hard_sections = [frozen[i].get("render_md", "") for i in range(len(frozen)) if i in hard]
    active_events: list[dict] = []
    for i, task in enumerate(frozen):
        if i not in hard:
            active_events.extend(task.get("events") or [])
    active_events.extend(state.get("active_tail_events") or [])
    active_events.extend(new_events)
    # De-dup by msg_id while preserving latest event object and chronological order.
    by_msg = {}
    for e in active_events:
        key = e.get("msg_id") or f"ts:{e.get('ts')}:{len(by_msg)}"
        by_msg[key] = e
    active_events = sorted(by_msg.values(), key=lambda e: e["ts"])

    clusters, metas, seg_md, active_decomp_md, total_sub, deliveries, task_subreqs = segment_events(
        group_id, active_events, model=model, start_idx=len(hard_sections) + 1,
        prior_anchors=state.get("title_anchors") or [])
    active_sections = split_task_sections(active_decomp_md)
    combined_sections = hard_sections + active_sections
    decomp_md = _combined_decompose_md(group_id, _window_desc(active_events or new_events), combined_sections)

    terminal = ts.compute_terminal(clusters)
    new_frozen = [frozen[i] for i in range(len(frozen)) if i in hard]
    active_tail_events: list[dict] = []
    active_tail_title = ""
    active_tail_subreqs: list[dict] = []
    for i, (cluster, meta) in enumerate(zip(clusters, metas)):
        is_terminal = terminal[i] if i < len(terminal) else True
        section = active_sections[i] if i < len(active_sections) else ""
        delivery = deliveries[i] if i < len(deliveries) else None
        subs = task_subreqs[i].get("subreqs") if i < len(task_subreqs) else None
        ok = task_subreqs[i].get("decompose_ok", True) if i < len(task_subreqs) else True
        # active_tail = 那个未终结簇（最近被碰的任务），不靠列表末位（同 _build_state 的理由）。
        if not is_terminal:
            active_tail_events = cluster
            active_tail_title = meta.get("title", "")   # 留着写进锚，下一窗沿用
            active_tail_subreqs = subs or []            # C1：活动尾巴子需求→msg_id 持久化
            continue
        new_frozen.append(_task_record(cluster, meta, section, is_terminal, delivery, subs, ok))

    state["schema_version"] = STATE_VERSION
    state["watermark_ts"] = max(float(state.get("watermark_ts") or 0), _last_ts(new_events))
    state["thaw_hours"] = thaw_hours
    state["frozen_tasks"] = new_frozen
    state["active_tail_events"] = active_tail_events
    state["active_tail_subreqs"] = active_tail_subreqs
    # 重建标题锚（本窗全部任务：冻结 + 活动尾巴），供下一窗沿用，保持标题跨窗连贯
    anchors = _title_anchors_from(new_frozen)
    if active_tail_events:
        anchors.append({"msg_ids": _event_msg_ids(active_tail_events), "title": active_tail_title})
    state["title_anchors"] = anchors
    state_path = _write_state(group_id, state, state_dir)
    seg_out, decomp_out = _write_outputs(group_id, seg_md, decomp_md, out_dir)
    export_dir = _export_after_segment(group_id, state)
    return {
        "changed": True, "mode": "incremental", "new_events": len(new_events),
        "active_events": len(active_events), "hard_frozen": len(hard_sections),
        "reopened": len(reopened), "active_tasks": len(clusters),
        "subtasks": total_sub, "state_path": str(state_path),
        "segment_path": str(seg_out), "decompose_path": str(decomp_out),
        "export_dir": export_dir,
    }


def run_pool_snapshot(pool_file: str, line_no: int | None = None, t0_msg_id: str | None = None,
                      out_path: str | None = None, include_t0: bool = True) -> dict:
    """Analyze one vfinal pool case with the deterministic layers only.

    This is a read-only adapter for replay cases. It deliberately does not load
    or write incremental state, and it does not call the LLM segmenter.
    """
    rec = ts.load_pool_record(pool_file, line_no=line_no, t0_msg_id=t0_msg_id)
    result = ts.analyze_pool_record(rec, include_t0=include_t0,
                                    sim_fn=ts.build_reply_sim_fn())
    if out_path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        result = dict(result)
        result["out_path"] = str(out)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--group")
    ap.add_argument("--hist-file", help="已导出的 messages.raw.jsonl；用于首跑/测试")
    ap.add_argument("--since", type=int, default=0)
    ap.add_argument("--until", type=int, default=0)
    ap.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    ap.add_argument("--out-dir", default=str(ts.SHARED))
    ap.add_argument("--thaw-hours", type=int, default=DEFAULT_THAW_HOURS)
    ap.add_argument("--model", default=None)
    ap.add_argument("--snapshot-pool-file", help="只读分析 user_corrections_pool*.jsonl 的一条 case")
    ap.add_argument("--snapshot-pool-line", type=int, help="snapshot 模式：1基行号")
    ap.add_argument("--snapshot-t0-msg-id", help="snapshot 模式：按 t0.msg_id 取记录")
    ap.add_argument("--snapshot-out", help="snapshot 模式：结构化 JSON 输出路径")
    ap.add_argument("--snapshot-no-t0", action="store_true",
                    help="snapshot 模式：只分析 T0 前历史，不把 T0 用户原指令放入事件流")
    args = ap.parse_args()
    if args.snapshot_pool_file:
        result = run_pool_snapshot(
            args.snapshot_pool_file,
            line_no=args.snapshot_pool_line,
            t0_msg_id=args.snapshot_t0_msg_id,
            out_path=args.snapshot_out,
            include_t0=not args.snapshot_no_t0,
        )
        print(json.dumps({
            "changed": False,
            "mode": "pool_snapshot",
            "pool_line": result.get("pool_line"),
            "events": result.get("event_count"),
            "tasks": result.get("task_count"),
            "subtasks": result.get("subreq_count"),
            "out_path": result.get("out_path"),
        }, ensure_ascii=False, indent=2))
        return
    if not args.group:
        raise SystemExit("--group is required unless --snapshot-pool-file is used")
    result = run_incremental(
        args.group, since=args.since, until=args.until, hist_file=args.hist_file,
        state_dir=Path(args.state_dir), out_dir=Path(args.out_dir),
        thaw_hours=args.thaw_hours, model=args.model,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("changed"):
        print(f"查看：{ts.DATAVIEW}")


if __name__ == "__main__":
    main()
