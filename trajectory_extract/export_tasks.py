#!/usr/bin/env python3
"""子任务切分轨迹导出器——把某群 state json 里已冻结的任务（frozen_tasks）导出成
人可读的 md + 下游可用的 jsonl，供 SFT / 分析消费。

数据源（单一事实源）：/opt/shared/data/task-trajectory/state/{群}.json
产物落到：           /opt/shared/data/task-trajectory/exports/{群}/

每个 frozen_task 产两个文件：
  {序号}_{title简化}.md    —— 人看。优先直接用任务自带 render_md；为空则用 events 拼。
  {序号}_{title简化}.jsonl —— 每行一个子需求对象，喂下游：
      {task_title, subreq_id, subreq_title, subreq_type, status, member_msg_ids,
       events:[归属该子需求的原文消息]}
    归属靠 subreq.member_msg_ids 与 task.events 的 msg_id join。
另出一张 {群}_index.jsonl：每任务一行摘要。

脱敏铁律：md 拼接段 / jsonl 的 events 文本一律过 RegexAnonymizer（open_id/app_id/人名
不裸落盘），但保留真实 msg_id——下游 join / 纠错标注按真 msg_id 精确对齐（见 C1 决策）。
render_md 本身是切分时已生成的渲染视图、不含裸 id，原样用。

用法：
  python3 export_tasks.py --group oc_53b8b620867a189d8dfe502865dfccc5
  python3 export_tasks.py --group oc_xxx --state-dir /path/to/state --out-dir /path/to/exports
被 incremental_segment.run_incremental 每次切分成功后自动调用（export_group_from_state）。
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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # data_process 根
import task_stitch as ts  # noqa: E402
from regex_anonymizer import RegexAnonymizer  # noqa: E402

DEFAULT_STATE_DIR = Path(os.environ.get("INCR_STATE_DIR", str(ts.SHARED / "state")))
DEFAULT_EXPORT_DIR = Path(os.environ.get("TRAJ_EXPORT_DIR", str(ts.SHARED / "exports")))


def _safe_group(group_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", group_id)


def _slug_title(title: str, n: int = 30) -> str:
    """任务标题→文件名安全片段：去飞书标记/空白，保留中英数，截断。"""
    t = ts._strip_feishu(title or "").strip()
    t = re.sub(r"\s+", "", t)
    t = re.sub(r'[/\\:*?"<>|]', "", t)     # 文件名非法字符
    t = t.strip(".")                       # 别以点开头/结尾
    return (t[:n] or "untitled")


def _fmt_ts(ts_val) -> str:
    if not ts_val:
        return "—"
    return time.strftime("%m-%d %H:%M:%S", time.localtime(float(ts_val)))


def _anon_event(e: dict, anon: RegexAnonymizer) -> dict:
    """脱敏一条 event 用于落盘：文本/who/name 过脱敏，msg_id 原样保留(join 用)。"""
    return {
        "msg_id": e.get("msg_id", ""),
        "role": e.get("role", ""),
        "who": anon.anonymize_text(e.get("name") or e.get("who") or ""),
        "ts": e.get("ts"),
        "text": anon.anonymize_text(e.get("text") or ""),
    }


def _events_md(events: list[dict], anon: RegexAnonymizer) -> str:
    """render_md 为空时的兜底：把 events 按时间拼成可读 markdown。"""
    lines = []
    for i, e in enumerate(sorted(events, key=lambda x: x.get("ts") or 0), 1):
        when = _fmt_ts(e.get("ts"))
        who = anon.anonymize_text(e.get("name") or e.get("who") or "")
        role = "用户" if e.get("role") == "user" else "bot"
        txt = ts._strip_feishu(e.get("text") or "").strip()
        txt = anon.anonymize_text(txt).replace("\n", " ")
        lines.append(f"#{i} [{when}] {role}({who}): {txt}")
    return "\n\n".join(lines)


def _subreq_rows(task: dict, anon: RegexAnonymizer) -> list[dict]:
    """把一个任务的 subreqs join 上 events，产出 jsonl 行对象列表。
    join：subreq.member_msg_ids ∩ task.events(by msg_id)。"""
    by_msg = {e.get("msg_id"): e for e in (task.get("events") or []) if e.get("msg_id")}
    title = ts._strip_feishu(task.get("title") or "").strip()
    rows = []
    for sr in task.get("subreqs") or []:
        ids = sr.get("member_msg_ids") or []
        evs = [_anon_event(by_msg[m], anon) for m in ids if m in by_msg]
        rows.append({
            "task_title": title,
            "subreq_id": sr.get("id", ""),
            "subreq_title": ts._strip_feishu(sr.get("title") or "").strip(),
            "subreq_type": sr.get("type", ""),
            "status": sr.get("status", ""),
            "dominant": anon.anonymize_text(sr.get("dominant") or ""),
            "member_msg_ids": ids,
            "events": evs,
        })
    return rows


def export_group_from_state(state: dict, group_id: str,
                            export_dir: Path = DEFAULT_EXPORT_DIR) -> dict:
    """遍历 state 的 frozen_tasks，对每个任务导出 md + jsonl，并汇总 index.jsonl。
    返回 {group_id, out_dir, tasks, md_files, jsonl_files, index_path}。"""
    anon = RegexAnonymizer()
    safe_g = _safe_group(group_id)
    out_dir = export_dir / safe_g
    out_dir.mkdir(parents=True, exist_ok=True)

    frozen = list(state.get("frozen_tasks") or [])
    md_files, jsonl_files, index_rows = [], [], []

    for k, task in enumerate(frozen, 1):
        title = task.get("title") or ""
        slug = _slug_title(title)
        stem = f"{k}_{slug}"

        # --- md：优先 render_md，空则用 events 拼 ---
        md = (task.get("render_md") or "").strip()
        if not md:
            md = (f"## 任务 {k}　{ts._strip_feishu(title).strip()}\n\n"
                  f"- 目标：{ts._strip_feishu(task.get('goal') or '').strip() or '—'}"
                  f" | 状态：{task.get('confirmation_status', '—')}"
                  f" | 交付：{task.get('delivery') or '—'}\n\n"
                  f"### 原文消息（render_md 为空，events 兜底）\n\n"
                  + _events_md(task.get("events") or [], anon))
        md_path = out_dir / f"{stem}.md"
        md_path.write_text(md.rstrip() + "\n", encoding="utf-8")
        md_files.append(str(md_path))

        # --- jsonl：每行一个子需求 ---
        rows = _subreq_rows(task, anon)
        jsonl_path = out_dir / f"{stem}.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        jsonl_files.append(str(jsonl_path))

        index_rows.append({
            "task_idx": k,
            "task_title": ts._strip_feishu(title).strip(),
            "confirmation_status": task.get("confirmation_status", ""),
            "terminal": task.get("terminal"),
            "delivery": task.get("delivery"),
            "n_events": len(task.get("events") or []),
            "n_subreqs": len(task.get("subreqs") or []),
            "last_ts": task.get("last_ts"),
            "md": os.path.basename(str(md_path)),
            "jsonl": os.path.basename(str(jsonl_path)),
        })

    index_path = out_dir / f"{safe_g}_index.jsonl"
    with index_path.open("w", encoding="utf-8") as f:
        for r in index_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {
        "group_id": group_id,
        "out_dir": str(out_dir),
        "tasks": len(frozen),
        "md_files": md_files,
        "jsonl_files": jsonl_files,
        "index_path": str(index_path),
    }


def export_group(group_id: str, state_dir: Path = DEFAULT_STATE_DIR,
                 export_dir: Path = DEFAULT_EXPORT_DIR) -> dict:
    """从磁盘读某群 state 后导出。"""
    safe = _safe_group(group_id)
    path = state_dir / f"{safe}.json"
    if not path.exists():
        raise SystemExit(f"state 不存在：{path}")
    state = json.loads(path.read_text(encoding="utf-8"))
    return export_group_from_state(state, state.get("group_id") or group_id, export_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", required=True)
    ap.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_EXPORT_DIR))
    args = ap.parse_args()
    res = export_group(args.group, Path(args.state_dir), Path(args.out_dir))
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("md_files", "jsonl_files")}, ensure_ascii=False, indent=2))
    print(f"→ {len(res['md_files'])} 个 md, {len(res['jsonl_files'])} 个 jsonl, "
          f"1 个 index：{res['out_dir']}")


if __name__ == "__main__":
    main()
