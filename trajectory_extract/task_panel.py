#!/usr/bin/env python3
"""任务状态面板——把所有群 state 里的任务按状态汇总成一张可读表（张耀明 2026-06-29 要求）。

三种状态（与 incremental_segment.confirmation_status / compute_terminal 完全一致，不另立口径）：
  · 未终结 = 观测窗右沿、还在飞的活（= state 的 active_tail，刻意没封存）。每群至多一个。
  · 未确认 = 已终结，但 bot 末尾只留占位/没看到实质交付，或末尾是用户消息 bot 没接。
  · 已完成 = 已终结且看到交付（delivery=已交付）或用户收尾语认可。
未终结 不在 frozen_tasks 里（它在 active_tail_events），故本面板单独把 active_tail 渲成「未终结」行，
不重跑 LLM、不额外烧钱——纯读 state 落盘，是 incremental 产物的只读视图。

用法：
  python3 task_panel.py                 # 扫默认 state 目录，渲染 markdown 到 stdout
  python3 task_panel.py --out PATH       # 同时写到 PATH（默认落共享目录供 data-view 零 SSH 看）
  python3 task_panel.py --json           # 输出结构化 JSON（供别的工具消费）
"""
import argparse
import json
import time
from pathlib import Path

DEFAULT_STATE_DIR = Path("/opt/shared/data/task-trajectory/state")
DEFAULT_OUT = Path("/opt/shared/data/task-trajectory/task_panel.md")

# 状态展示顺序与图标（未终结排最前——最该被人盯的活）
STATUS_ORDER = ["未终结", "未确认", "已完成"]
STATUS_ICON = {"未终结": "⏳", "未确认": "❓", "已完成": "✅"}


def _fmt_ts(ts: float | int | None) -> str:
    if not ts:
        return "—"
    return time.strftime("%m-%d %H:%M", time.localtime(float(ts)))


def _active_tail_record(state: dict) -> dict | None:
    """把 active_tail_events 渲成一条「未终结」任务行（不重跑 LLM，只摘时间窗+首尾消息）。"""
    evs = state.get("active_tail_events") or []
    evs = [e for e in evs if e.get("text")]
    if not evs:
        return None
    evs = sorted(evs, key=lambda e: e.get("ts") or 0)
    # 标题取「最后一条用户提问」——active_tail 是还没语义切分的原始 blob，
    # 最近的用户诉求最能代表「此刻在飞的活」（比第一条 06-xx 的旧消息准）。
    last_user = next((e["text"] for e in reversed(evs) if e.get("role") == "user"), evs[-1]["text"])
    title = (last_user or "").strip().splitlines()[0][:40]
    last = evs[-1]
    return {
        "confirmation_status": "未终结",
        "title": title or "(进行中)",
        "goal": "",
        "terminal": False,
        "delivery": None,
        "last_ts": last.get("ts"),
        "first_ts": evs[0].get("ts"),
        "n_events": len(evs),
        "last_role": last.get("role"),
    }


def collect(state_dir: Path) -> list[dict]:
    """扫所有群 state，返回 [{group_id, tasks:[...]}]，tasks 含 frozen + active_tail。"""
    rows = []
    for path in sorted(state_dir.glob("*.json")):
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            # fail-loud：坏 state 不静默跳过，明着记一条错行
            rows.append({"group_id": path.stem, "error": str(exc), "tasks": []})
            continue
        tasks = list(state.get("frozen_tasks") or [])
        tail = _active_tail_record(state)
        if tail:
            tasks = tasks + [tail]
        rows.append({
            "group_id": state.get("group_id") or path.stem,
            "watermark": state.get("watermark_ts"),
            "tasks": tasks,
        })
    return rows


def render_md(rows: list[dict]) -> str:
    out = ["# 任务状态面板", ""]
    out.append(f"_刷新于 {time.strftime('%Y-%m-%d %H:%M:%S')} · 数据源：增量切分 state（只读视图）_")
    out.append("")
    out.append("**状态口径**：⏳未终结=还在飞的活（观测窗右沿）｜❓未确认=已结束但没看到实质交付/bot没接｜✅已完成=已交付或用户认可")
    out.append("")

    # 全局状态计数
    total = {s: 0 for s in STATUS_ORDER}
    for r in rows:
        for t in r["tasks"]:
            st = t.get("confirmation_status")
            if st in total:
                total[st] += 1
    summary = " ｜ ".join(f"{STATUS_ICON[s]}{s} {total[s]}" for s in STATUS_ORDER)
    out.append(f"**合计**：{summary}")
    out.append("")

    for r in rows:
        if r.get("error"):
            out.append(f"## ⚠️ 群 `{r['group_id']}` — state 读取失败：{r['error']}")
            out.append("")
            continue
        out.append(f"## 群 `{r['group_id']}`　（水位 {_fmt_ts(r.get('watermark'))}）")
        out.append("")
        tasks = r["tasks"]
        if not tasks:
            out.append("_（无任务）_")
            out.append("")
            continue
        # 按状态分组，未终结排最前
        by_status = {s: [] for s in STATUS_ORDER}
        for t in tasks:
            by_status.setdefault(t.get("confirmation_status") or "未确认", []).append(t)
        for st in STATUS_ORDER:
            group = by_status.get(st) or []
            if not group:
                continue
            out.append(f"### {STATUS_ICON.get(st, '·')} {st}（{len(group)}）")
            out.append("")
            out.append("| 任务 | 最后活动 | 交付 |")
            out.append("|---|---|---|")
            for t in sorted(group, key=lambda x: x.get("last_ts") or 0, reverse=True):
                title = (t.get("title") or "(无题)").replace("|", "/").strip()[:50]
                dl = t.get("delivery") or "—"
                extra = ""
                if st == "未终结" and t.get("n_events"):
                    span = f"{_fmt_ts(t.get('first_ts'))}→{_fmt_ts(t.get('last_ts'))}"
                    extra = f"（原始尾巴{t['n_events']}条·{span}·末{'用户' if t.get('last_role') == 'user' else 'bot'}）"
                out.append(f"| {title}{extra} | {_fmt_ts(t.get('last_ts'))} | {dl} |")
            out.append("")
    return "\n".join(out).rstrip() + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="markdown 输出路径；设为 - 则只打印不写文件")
    ap.add_argument("--json", action="store_true", help="输出结构化 JSON 而非 markdown")
    args = ap.parse_args()

    rows = collect(Path(args.state_dir))
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2, default=lambda o: None))
        return
    md = render_md(rows)
    print(md)
    if args.out and args.out != "-":
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
