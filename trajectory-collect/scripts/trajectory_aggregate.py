#!/usr/bin/env python3
"""把 collector 落盘的数据聚合成轨迹 JSONL（管道A 的「归」环节）。

两个来源（--source 选择，默认 all）：
- openclaw：spans/*.jsonl 按 traceId 聚合。一条 trace ≈ gateway 处理一条消息的
  完整过程（invoke → 模型调用×N → 工具调用×N），正文在 openclaw.content.* 属性里。
- claude-code：logs/*.jsonl（Claude Code 内置 OTel 事件）按 session.id 聚合。
  内置 OTel 只导出 user_prompt 正文 + api_request 计量 + tool 结构（无模型输出正文），
  正文靠 join 本地 transcript（~/.claude/projects/*/<session-id>.jsonl）补全——
  OTel 提供计量与 join key，transcript 提供完整 messages（含 tool_use/tool_result）。

输出每条轨迹含 RL 四要素：
  prompt / 模型输出 / 工具反馈 / token 计量
reward 列暂空，由用户后续行为（评分、纠偏、采纳）按 session 回填。

用法:
  python3 trajectory_aggregate.py                 # 聚合全部 -> trajectories/
  python3 trajectory_aggregate.py --sample 3      # 聚合并漂亮打印 3 条抽样核验
  python3 trajectory_aggregate.py --date 2026-06-12 --source claude-code
"""
import argparse
import glob
import gzip
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from traj_common import classify_role, system_tools_sha  # noqa: E402

DATA_DIR = Path(os.environ.get("TRAJ_DATA_DIR", "/home/agent/trajectory-data"))
# Claude Code 本地 transcript 根目录（正文 join 源）
CLAUDE_PROJECTS_DIR = Path(os.environ.get(
    "CLAUDE_PROJECTS_DIR", str(Path.home() / ".claude" / "projects")))


def publish(src: Path, dest_name: str, publish_dir: str | None):
    """把聚合产出按「来源-角色-日期」命名复制到共享目录（--publish / TRAJ_PUBLISH_DIR）。"""
    if not publish_dir:
        return
    dest = Path(publish_dir) / dest_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    print(f"  [publish] {src.name} -> {dest}")

CONTENT_KEYS = {
    "openclaw.content.system_prompt": "system_prompt",
    "openclaw.content.input_messages": "input_messages",
    "openclaw.content.output_messages": "output_messages",
    "openclaw.content.tool_inputs": "tool_inputs",
    "openclaw.content.tool_outputs": "tool_outputs",
    "openclaw.content.tool_definitions": "tool_definitions",
}


def load_spans(date: str | None):
    pattern = str(DATA_DIR / "spans" / (f"{date}.jsonl" if date else "*.jsonl"))
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def maybe_json(text):
    """内容属性可能是 JSON 字符串，尽量解析成结构。"""
    if isinstance(text, str) and text[:1] in "[{":
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            pass
    return text


def build_trajectory(trace_id: str, spans: list) -> dict:
    spans.sort(key=lambda s: s["start_ns"])
    traj = {
        "trace_id": trace_id,
        "start_ns": spans[0]["start_ns"],
        "end_ns": max(s["end_ns"] for s in spans),
        "agent": None, "channel": None, "provider": None, "model": None,
        "system_prompt": None, "tool_definitions": None,
        "steps": [],          # 按时间排列的 model / tool 步骤
        "tokens": {},         # input/output/total 等累计
        "reward": None,       # 占位：后续用户反馈回填
        "span_names": sorted({s["name"] for s in spans}),
    }
    for s in spans:
        a = s["attributes"]
        for k in ("agent", "channel", "provider", "model"):
            traj[k] = traj[k] or a.get(f"openclaw.{k}")
        if a.get("openclaw.content.system_prompt"):
            traj["system_prompt"] = a["openclaw.content.system_prompt"]
        if a.get("openclaw.content.tool_definitions"):
            traj["tool_definitions"] = maybe_json(a["openclaw.content.tool_definitions"])

        if s["name"] in ("openclaw.model.call", "openclaw.model.usage") or \
                a.get("gen_ai.operation.name") in ("chat", "text_completion"):
            step = {
                "type": "model",
                "span": s["name"],
                "start_ns": s["start_ns"],
                "model": a.get("gen_ai.request.model") or a.get("openclaw.model"),
                "input_messages": maybe_json(a.get("openclaw.content.input_messages")),
                "output_messages": maybe_json(a.get("openclaw.content.output_messages")),
            }
            for tk in ("input", "output", "total", "cache_read", "cache_write"):
                v = a.get(f"openclaw.tokens.{tk}") or a.get(f"gen_ai.usage.{tk}_tokens")
                if v is not None:
                    step[f"tokens_{tk}"] = v
                    traj["tokens"][tk] = traj["tokens"].get(tk, 0) + int(v)
            if any(v is not None for k, v in step.items() if k not in ("type", "span", "start_ns")):
                traj["steps"].append(step)
        elif s["name"] == "openclaw.tool.execution":
            traj["steps"].append({
                "type": "tool",
                "start_ns": s["start_ns"],
                "tool_name": a.get("gen_ai.tool.name") or a.get("openclaw.toolName"),
                "tool_inputs": maybe_json(a.get("openclaw.content.tool_inputs")),
                "tool_outputs": maybe_json(a.get("openclaw.content.tool_outputs")),
                "error": a.get("openclaw.errorCategory"),
            })
    traj["steps"].sort(key=lambda x: x["start_ns"])
    return traj


def load_logs(date: str | None):
    pattern = str(DATA_DIR / "logs" / (f"{date}.jsonl" if date else "*.jsonl"))
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def load_request_captures(date: str | None) -> dict:
    """读 anthropic_capture_proxy 落盘的 requests/*.jsonl，按 session_id 归组。

    system/tools 只存在于发往 API 的请求体里（transcript 和内置 OTel 都没有），
    由捕获代理在请求线上截获。一个 session 可能有多个变体（主线 vs 子代理、
    ToolSearch 动态加载后 tools 变化），按时间序保留全部。
    """
    by_session = defaultdict(list)
    pattern = str(DATA_DIR / "requests" / (f"{date}.jsonl" if date else "*.jsonl"))
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                by_session[rec["session_id"]].append(rec)
    for recs in by_session.values():
        recs.sort(key=lambda r: r.get("ts", 0))
    return by_session


def load_api_calls(date: str | None):
    """读捕获代理的全量轨 api-calls/*.jsonl.gz（每行 = 一次完整 API 调用）。"""
    pattern = str(DATA_DIR / "api-calls" / (f"{date}.jsonl.gz" if date else "*.jsonl.gz"))
    for path in sorted(glob.glob(pattern)):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def load_transcript_messages(path: str) -> list:
    """从 Claude Code 本地 transcript 提取完整对话正文（含 tool_use/tool_result）。"""
    msgs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") in ("user", "assistant") and isinstance(rec.get("message"), dict):
                m = rec["message"]
                msgs.append({"role": m.get("role"), "content": m.get("content"),
                             "timestamp": rec.get("timestamp")})
    return msgs


def build_cc_trajectory(session_id: str, events: list,
                        request_captures: list | None = None) -> dict:
    """把一个 Claude Code 会话的 OTel 事件聚合成轨迹，并 join transcript 补正文。"""
    events.sort(key=lambda e: e.get("time_ns", 0))
    traj = {
        "source": "log",          # 日志路（OTEL+transcript join，vs 捕获代理路 source=api）
        "brain": "claude-code",
        "role": "main",           # 日志路只有主线——子代理不写 transcript，仅 api 路可见
        "agent_desc": "main",
        "session_id": session_id,
        "start_ns": events[0].get("time_ns"),
        "end_ns": events[-1].get("time_ns"),
        "model": None,
        "system": None,       # 系统提示词（capture proxy 从请求体截获，首个变体）
        "tools": None,        # 工具 schema 全量（同上）
        "system_tools_variants": None,  # 同 session 的全部变体（主线/子代理）
        "prompts": [],        # user_prompt 正文（OTEL_LOG_USER_PROMPTS=1）
        "steps": [],          # api_request / tool 步骤（计量级）
        "tokens": {},
        "cost_usd": 0.0,
        "messages": None,     # transcript join 的完整正文
        "transcript_path": None,
        "reward": None,       # 占位：后续用户反馈回填
    }
    if request_captures:
        # 首个变体 = 主线首请求的 system/tools；后续变体多为子代理或工具集变化
        traj["system"] = request_captures[0].get("system")
        traj["tools"] = request_captures[0].get("tools")
        if len(request_captures) > 1:
            traj["system_tools_variants"] = [
                {"ts": r.get("ts"), "model": r.get("model"),
                 "system": r.get("system"), "tools": r.get("tools")}
                for r in request_captures]
    for e in events:
        a = e["attributes"]
        name = a.get("event.name")
        if name == "user_prompt":
            traj["prompts"].append(a.get("prompt") or f"(len={a.get('prompt_length')})")
        elif name == "api_request":
            traj["model"] = traj["model"] or a.get("model")
            step = {"type": "model", "time_ns": e.get("time_ns"),
                    "model": a.get("model"), "request_id": a.get("request_id"),
                    "query_source": a.get("query_source"),
                    "duration_ms": a.get("duration_ms")}
            for tk in ("input", "output", "cache_read", "cache_creation"):
                v = a.get(f"{tk}_tokens")
                if v is not None:
                    step[f"tokens_{tk}"] = v
                    traj["tokens"][tk] = traj["tokens"].get(tk, 0) + int(v)
            traj["cost_usd"] += float(a.get("cost_usd") or 0)
            traj["steps"].append(step)
        elif name == "tool_result":
            traj["steps"].append({
                "type": "tool", "time_ns": e.get("time_ns"),
                "tool_name": a.get("tool_name"), "tool_use_id": a.get("tool_use_id"),
                "success": a.get("success"), "duration_ms": a.get("duration_ms"),
            })
        elif name == "api_error":
            traj["steps"].append({"type": "api_error", "time_ns": e.get("time_ns"),
                                  "model": a.get("model"), "error": a.get("error")})
    hits = glob.glob(str(CLAUDE_PROJECTS_DIR / "*" / f"{session_id}.jsonl"))
    if hits:
        traj["transcript_path"] = hits[0]
        traj["messages"] = load_transcript_messages(hits[0])
    return traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="只聚合某天（YYYY-MM-DD），默认全部")
    ap.add_argument("--sample", type=int, default=0, help="抽样打印 N 条轨迹核验")
    ap.add_argument("--content-only", action="store_true",
                    help="只输出带正文内容（content capture 命中）的轨迹")
    ap.add_argument("--source", choices=["openclaw", "claude-code", "api-hijack", "all"],
                    default="all",
                    help="聚合哪个来源（openclaw=spans 按 traceId；claude-code=logs 按 "
                         "session.id；api-hijack=api-calls 按 session 归组，slime 形态）")
    ap.add_argument("--publish", default=os.environ.get("TRAJ_PUBLISH_DIR"),
                    help="把产出按 来源-角色-日期 命名发布到该共享目录"
                         "（如 /opt/shared/data/transcript），也可用 TRAJ_PUBLISH_DIR")
    args = ap.parse_args()

    if args.source in ("openclaw", "all"):
        aggregate_openclaw(args)
    if args.source in ("claude-code", "all"):
        aggregate_claude_code(args)
    if args.source in ("api-hijack", "all"):
        aggregate_api_hijack(args)


def aggregate_openclaw(args):
    by_trace = defaultdict(list)
    for span in load_spans(args.date):
        by_trace[span["trace_id"]].append(span)

    out_path = DATA_DIR / "trajectories" / f"{args.date or 'all'}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trajectories = []
    for tid, spans in by_trace.items():
        traj = build_trajectory(tid, spans)
        if args.content_only and not (traj["system_prompt"] or traj["steps"]):
            continue
        trajectories.append(traj)
    trajectories.sort(key=lambda t: t["start_ns"])

    with open(out_path, "w", encoding="utf-8") as f:
        for t in trajectories:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    with_content = sum(1 for t in trajectories
                       if t["system_prompt"] or any(s.get("input_messages") or s.get("tool_inputs")
                                                    for s in t["steps"]))
    print(f"[openclaw] traces={len(by_trace)} trajectories={len(trajectories)} "
          f"含正文={with_content} -> {out_path}")
    if trajectories:  # 空天不发布，避免共享目录堆空文件
        publish(out_path, f"log-openclaw-{args.date or 'all'}.jsonl", args.publish)

    for t in trajectories[-args.sample:] if args.sample else []:
        print("\n" + "=" * 70)
        print(f"trace={t['trace_id']} agent={t['agent']} channel={t['channel']} "
              f"model={t['model']} tokens={t['tokens']} spans={t['span_names']}")
        sp = t["system_prompt"]
        print(f"[system_prompt] {str(sp)[:200] if sp else '(未捕获)'}")
        for s in t["steps"]:
            if s["type"] == "model":
                print(f"  [model {s.get('model')}] in={str(s.get('input_messages'))[:150]}")
                print(f"     out={str(s.get('output_messages'))[:150]}")
            else:
                print(f"  [tool {s.get('tool_name')}] in={str(s.get('tool_inputs'))[:120]} "
                      f"out={str(s.get('tool_outputs'))[:120]}")


def aggregate_claude_code(args):
    by_session = defaultdict(list)
    for rec in load_logs(args.date):
        if rec.get("attributes", {}).get("session.id"):
            by_session[rec["attributes"]["session.id"]].append(rec)
    captures = load_request_captures(args.date)

    date_label = args.date or "all"
    out_path = DATA_DIR / "trajectories" / f"{date_label}.log-main.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trajectories = []
    for sid, events in by_session.items():
        traj = build_cc_trajectory(sid, events, captures.get(sid))
        if args.content_only and not (traj["prompts"] or traj["messages"]):
            continue
        trajectories.append(traj)
    trajectories.sort(key=lambda t: t["start_ns"] or 0)

    with open(out_path, "w", encoding="utf-8") as f:
        for t in trajectories:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    joined = sum(1 for t in trajectories if t["messages"])
    with_sys = sum(1 for t in trajectories if t["system"] is not None)
    print(f"[claude-code] sessions={len(by_session)} trajectories={len(trajectories)} "
          f"transcript已join={joined} system/tools已join={with_sys} -> {out_path}")
    publish(out_path, f"log-main-{date_label}.jsonl", args.publish)

    for t in trajectories[-args.sample:] if args.sample else []:
        print("\n" + "=" * 70)
        print(f"session={t['session_id']} model={t['model']} tokens={t['tokens']} "
              f"cost=${t['cost_usd']:.4f} steps={len(t['steps'])}")
        for p in t["prompts"]:
            print(f"[prompt] {str(p)[:150]}")
        if t["messages"]:
            print(f"[transcript] {len(t['messages'])} messages <- {t['transcript_path']}")
            for m in t["messages"][-2:]:
                print(f"  [{m['role']}] {str(m['content'])[:150]}")
        else:
            print("[transcript] (未找到——正文缺失，只有计量)")


def aggregate_api_hijack(args):
    """api-calls 全量轨（捕获代理路，source=api）→ 按角色拆成独立轨迹。

    一个 session_id 里混着主线 + 子代理(文件搜索/web 搜索/Task) + 辅助(标题/SDK)，
    它们共用同一个 session_id，只能靠 system 正文签名区分（classify_role，已剔 billing）。
    按 (session × role × desc) 聚组，每组 = 一条独立 agent 轨迹，按 role 分文件落盘：
      api-main / api-subagent / api-aux，每行带 source/role/brain/agent_desc 字段。
    """
    by_session = defaultdict(list)
    for rec in load_api_calls(args.date):
        by_session[rec["session_id"]].append(rec)
    if not by_session:
        print("[api-hijack] 无 api-calls 数据（代理升级前的流量没有全量轨）")
        return

    date_label = args.date or "all"
    role_files = {r: DATA_DIR / "trajectories" / f"{date_label}.api-{r}.jsonl.gz"
                  for r in ("main", "subagent", "aux")}
    role_files["main"].parent.mkdir(parents=True, exist_ok=True)
    handles = {r: gzip.open(p, "wt", encoding="utf-8") for r, p in role_files.items()}
    counts = {r: {"traj": 0, "calls": 0} for r in role_files}
    n_calls = 0
    try:
        for sid in sorted(by_session, key=lambda s: by_session[s][0].get("ts", 0)):
            calls = sorted(by_session[sid], key=lambda r: r.get("ts", 0))
            n_calls += len(calls)
            # 按 (role, desc) 聚组——同一身份的调用归一条轨迹（主线含压缩后变体）
            groups = defaultdict(list)
            for c in calls:
                role, desc = classify_role(c["request"].get("system"))
                groups[(role, desc)].append(c)
            for (role, desc), gcalls in groups.items():
                # 重算指纹（剔 billing 后）——存盘 sha 可能是修复前的污染值，这里统一
                shas = []
                for c in gcalls:
                    s = system_tools_sha(c["request"])
                    if s not in shas:
                        shas.append(s)
                traj = {
                    "source": "api",          # 捕获代理路（vs 日志路 source=log）
                    "brain": "claude-code",
                    "role": role,             # main / subagent / aux
                    "agent_desc": desc,       # main / file-search / web-search / task / title / sdk
                    "session_id": sid,        # 与主线同一 session（子代理寄生其中）
                    "agent_shas": shas,       # 本身份的 system+tools 指纹（压缩会有多个）
                    "start_ts": gcalls[0].get("ts"),
                    "end_ts": gcalls[-1].get("ts"),
                    "n_calls": len(gcalls),
                    "calls": gcalls,          # 每 call 自带完整 request/response
                    "reward": None,           # 占位：后续用户反馈回填
                }
                handles[role].write(json.dumps(traj, ensure_ascii=False) + "\n")
                counts[role]["traj"] += 1
                counts[role]["calls"] += len(gcalls)
    finally:
        for h in handles.values():
            h.close()

    summary = " ".join(f"{r}={counts[r]['traj']}轨/{counts[r]['calls']}call"
                       for r in ("main", "subagent", "aux"))
    print(f"[api-hijack] sessions={len(by_session)} calls={n_calls} -> {summary}")
    for r, p in role_files.items():
        if counts[r]["traj"]:  # 空角色（如当天无子代理）不发布，避免共享目录堆空文件
            publish(p, f"api-{r}-{date_label}.jsonl.gz", args.publish)

    if args.sample:
        for r in ("main", "subagent"):
            print("\n" + "=" * 70 + f"\n[api-{r}] 抽样:")
            with gzip.open(role_files[r], "rt", encoding="utf-8") as f:
                for line in list(f)[:args.sample]:
                    t = json.loads(line)
                    c0 = t["calls"][0]
                    req = c0["request"]
                    print(f"  session={t['session_id'][:8]} role={t['role']} "
                          f"desc={t['agent_desc']} calls={t['n_calls']} "
                          f"model={req.get('model')} tools={len(req.get('tools') or [])}")


if __name__ == "__main__":
    main()
