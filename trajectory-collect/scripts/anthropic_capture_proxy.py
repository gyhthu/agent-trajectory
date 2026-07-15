#!/usr/bin/env python3
"""Anthropic API 透传捕获代理（管道B：API 劫持式全量记录，不改流量）。

为什么需要它：`system`/`tools` 只存在于发往 API 的请求体里，模型输出的
on-wire 原文只存在于响应里——本地 transcript 和内置 OTel 都拿不全——所以
在请求线上截获是唯一拿法。claude CLI 设 ANTHROPIC_BASE_URL 指到本代理即接入。
参考 slime 的 API 劫持思路：每次 API 调用 = 一条完整训练样本
（full request + full response），主线 agent / 子代理 / compaction / haiku
工具调用全走同一个 /v1/messages 端点，天然全覆盖；按 sha(system+tools)
聚类即可区分不同 agent 身份。

- 监听 BIND_HOST:PORT（默认 127.0.0.1:4319，安全规则：默认绝不绑 0.0.0.0）
- 所有请求原样转发 UPSTREAM（默认 https://api.anthropic.com），响应流式回传
- POST */v1/messages（带 session 标记的）双轨落盘：
  1. requests/YYYY-MM-DD.jsonl：system+tools 去重轻量轨（聚合器 join 用）
  2. api-calls/YYYY-MM-DD.jsonl.gz：全量轨——request 完整 payload +
     response（SSE 流重建成最终 message，含 thinking/tool_use/usage）。
     请求体带全上下文前缀，必须 gzip（多 member 追加，gzip.open 可直接读）
- 绝不落盘任何请求头（Authorization/OAuth token 不经过磁盘）
- 捕获环节出错只打日志不阻断转发（观测不能搞挂生产 bot）；转发出错返回 502

下游：trajectory_aggregate.py
  --source claude-code  按 session join system/tools 进轨迹
  --source api-hijack   api-calls 按 session 归组，每 call 一条 slime 形态样本
"""
import asyncio
import gzip
import json
import os
import re
import sys
import time
from pathlib import Path

import aiohttp
from aiohttp import web

sys.path.insert(0, str(Path(__file__).resolve().parent))
from traj_common import system_tools_sha  # noqa: E402  指纹单一事实源（已剔 billing 块）
import message_tree as mt  # noqa: E402  逐 message 挂树算 thread_id（与 export 共用单一事实源）

BIND_HOST = os.environ.get("BIND_HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "4319"))
UPSTREAM = os.environ.get("UPSTREAM", "https://api.anthropic.com").rstrip("/")
DATA_DIR = Path(os.environ.get("TRAJ_DATA_DIR", "/home/agent/trajectory-data"))

# metadata.user_id 两种已知格式：
# 新版 = JSON 串 {"device_id":...,"account_uuid":...,"session_id":"<uuid>"}
# 旧版 = user_<hash>_account_<uuid>_session_<uuid>
_SESSION_RE = re.compile(r"session_([0-9a-f-]{36})")


def extract_session_id(user_id: str):
    if user_id.startswith("{"):
        try:
            return json.loads(user_id).get("session_id")
        except json.JSONDecodeError:
            pass
    m = _SESSION_RE.search(user_id)
    return m.group(1) if m else None


# system_tools_sha 来自 traj_common（剔 billing 块后哈希）——与聚合器分类用同一指纹。

# 进程内去重：session_id -> {sha, ...}（重复无害，聚合器也会去重）
_seen: dict = {}
_write_lock = asyncio.Lock()

# 这些头由 aiohttp/HTTP 栈自行管理，不透传
_HOP_HEADERS = {"host", "content-length", "transfer-encoding", "connection",
                "keep-alive", "proxy-authenticate", "proxy-authorization",
                "te", "trailers", "upgrade"}


async def append_jsonl(subdir: str, rec: dict, compress: bool = False):
    day = time.strftime("%Y-%m-%d")
    suffix = ".jsonl.gz" if compress else ".jsonl"
    out = DATA_DIR / subdir / f"{day}{suffix}"
    out.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(rec, ensure_ascii=False) + "\n"
    async with _write_lock:
        if compress:  # 每次追加一个 gzip member，gzip.open 读时自动拼接
            with gzip.open(out, "at", encoding="utf-8") as f:
                f.write(line)
        else:
            with open(out, "a", encoding="utf-8") as f:
                f.write(line)


async def capture_system_tools(path: str, session_id: str, payload: dict, sha: str):
    """轻量轨：system/tools 按 session+hash 去重落盘（聚合器 join 用）。"""
    if payload.get("system") is None and payload.get("tools") is None:
        return
    if sha in _seen.setdefault(session_id, set()):
        return
    _seen[session_id].add(sha)
    await append_jsonl("requests", {
        "ts": time.time(),
        "session_id": session_id,
        "model": payload.get("model"),
        "path": path,
        "sha": sha,
        "system": payload.get("system"),
        "tools": payload.get("tools"),
    })
    print(f"[capture] session={session_id[:8]} model={payload.get('model')} "
          f"tools={len(payload.get('tools') or [])} sha={sha[:8]}", flush=True)


def reconstruct_sse_message(text: str):
    """从 SSE 事件流重建最终 message（content blocks + stop_reason + usage）。"""
    msg = None
    blocks: list = []
    partial_json: dict = {}
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data:
            continue
        ev = json.loads(data)
        t = ev.get("type")
        if t == "message_start":
            msg = ev["message"]
            blocks = list(msg.get("content") or [])
        elif t == "content_block_start":
            i = ev["index"]
            while len(blocks) <= i:
                blocks.append(None)
            blocks[i] = ev["content_block"]
            partial_json[i] = ""
        elif t == "content_block_delta":
            i, d = ev["index"], ev["delta"]
            dt = d.get("type")
            if dt == "text_delta":
                blocks[i]["text"] = blocks[i].get("text", "") + d["text"]
            elif dt == "thinking_delta":
                blocks[i]["thinking"] = blocks[i].get("thinking", "") + d["thinking"]
            elif dt == "input_json_delta":
                partial_json[i] = partial_json.get(i, "") + d["partial_json"]
            elif dt == "signature_delta":
                blocks[i]["signature"] = blocks[i].get("signature", "") + d["signature"]
        elif t == "content_block_stop":
            i = ev["index"]
            if partial_json.get(i):
                try:
                    blocks[i]["input"] = json.loads(partial_json[i])
                except json.JSONDecodeError:  # 截断等异常，保留原文不丢
                    blocks[i]["input_raw"] = partial_json[i]
        elif t == "message_delta":
            if msg is not None:
                msg.update(ev.get("delta") or {})
                if ev.get("usage"):
                    msg.setdefault("usage", {}).update(ev["usage"])
    if msg is not None:
        msg["content"] = blocks
    return msg


async def capture_api_call(session_id: str, payload: dict, sha: str,
                           status: int, resp_headers, resp_body: bytes):
    """全量轨：一次 API 调用 = 一条完整样本（request 全量 + response 重建）。"""
    if resp_headers.get("Content-Encoding") == "gzip":
        resp_body = gzip.decompress(resp_body)
    rec = {
        "ts": time.time(),
        "session_id": session_id,
        "sha": sha,            # sha(system+tools)：聚类区分主线/子代理/compaction
        "status": status,
        "request": payload,    # 全量：model/system/tools/messages/metadata/...
        "response": None,
    }
    # 结构化 thread_id：逐 message 挂树、锚在本次 call 路径里第一条 assistant 节点。
    # 锚路径只由本次 call 自身 (system, messages 到首条 assistant) 决定、与树历史无关——
    # 故用一棵临时空树即可算出确定性 id，无需跨 call 状态/锁/重启回放（实测 == export 离线共享树
    # 算出的 id）。首调 [S,U] 无 assistant → provisional 占位，export 离线按结构并回唯一子线程。
    # 全程 try/except 只报错不阻断：采集是所有 bot 共用的旁路，绝不因算 id 出错影响落盘。
    try:
        tid, provisional = mt.mount_call(mt.MessageTree(), payload.get("system"),
                                         payload.get("messages") or [])
        rec["thread_id"] = tid
        if provisional:
            rec["thread_id_provisional"] = True
    except Exception as e:
        print(f"[capture-ERROR] thread_id: {e!r}", file=sys.stderr, flush=True)
    ctype = (resp_headers.get("Content-Type") or "").split(";")[0]
    text = resp_body.decode("utf-8", errors="replace")
    if status == 200 and ctype == "text/event-stream":
        rec["response"] = reconstruct_sse_message(text)
        if rec["response"] is None:  # 重建失败必须留原文，不能丢数据
            rec["response_raw"] = text
    else:
        try:
            rec["response"] = json.loads(text)
        except json.JSONDecodeError:
            rec["response_raw"] = text
    await append_jsonl("api-calls", rec, compress=True)
    usage = (rec["response"] or {}).get("usage") if isinstance(rec["response"], dict) else None
    print(f"[api-call] session={session_id[:8]} sha={sha[:8]} status={status} "
          f"msgs={len(payload.get('messages') or [])} usage={usage}", flush=True)


async def proxy(request: web.Request) -> web.StreamResponse:
    body = await request.read()
    payload = session_id = sha = None
    if request.method == "POST" and request.path.endswith("/v1/messages"):
        try:
            payload = json.loads(body)
            user_id = (payload.get("metadata") or {}).get("user_id") or ""
            session_id = extract_session_id(user_id)
            if session_id:
                sha = system_tools_sha(payload)
                await capture_system_tools(request.path, session_id, payload, sha)
            else:
                # 无 session 标记（配额探测等）不捕获，但要可见，便于发现格式变化
                print(f"[capture-skip] no session in user_id={user_id!r} "
                      f"model={payload.get('model')}", flush=True)
        except Exception as e:  # 捕获失败只报错，不阻断转发
            print(f"[capture-ERROR] request: {e!r}", file=sys.stderr, flush=True)
            payload = session_id = None

    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in _HOP_HEADERS}
    try:
        async with request.app["client"].request(
            request.method, UPSTREAM + request.path_qs,
            headers=headers, data=body if body else None,
        ) as upstream:
            resp_headers = {k: v for k, v in upstream.headers.items()
                            if k.lower() not in _HOP_HEADERS}
            resp = web.StreamResponse(status=upstream.status,
                                      headers=resp_headers)
            await resp.prepare(request)
            buf = bytearray()  # 边透传边缓冲，eof 后重建响应落全量轨
            async for chunk in upstream.content.iter_chunked(16384):
                await resp.write(chunk)
                if session_id:
                    buf.extend(chunk)
            await resp.write_eof()
    except Exception as e:  # 转发失败必须显式报错，禁止静默
        print(f"[proxy-ERROR] {request.method} {request.path}: {e!r}",
              file=sys.stderr, flush=True)
        return web.json_response(
            {"type": "error",
             "error": {"type": "proxy_error", "message": repr(e)}},
            status=502)

    if session_id:
        try:
            await capture_api_call(session_id, payload, sha,
                                   upstream.status, upstream.headers, bytes(buf))
        except Exception as e:  # 响应已回传完毕，捕获失败只报错
            print(f"[capture-ERROR] api-call: {e!r}", file=sys.stderr, flush=True)
    return resp


async def healthz(_request: web.Request) -> web.Response:
    return web.Response(text="ok")


async def make_app() -> web.Application:
    app = web.Application(client_max_size=256 * 1024 * 1024)
    # auto_decompress=False：原始字节透传，Content-Encoding 头保持一致
    # total=None：模型流式响应可长达数十分钟，不设总超时
    app["client"] = aiohttp.ClientSession(
        auto_decompress=False,
        timeout=aiohttp.ClientTimeout(total=None, connect=30),
    )

    async def close_client(app):
        await app["client"].close()

    app.on_cleanup.append(close_client)
    app.router.add_get("/healthz", healthz)
    app.router.add_route("*", "/{tail:.*}", proxy)
    return app


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Anthropic capture proxy listening on {BIND_HOST}:{PORT} "
          f"-> {UPSTREAM}, capture -> {DATA_DIR}/{{requests,api-calls}}/", flush=True)
    web.run_app(make_app(), host=BIND_HOST, port=PORT, print=None)


if __name__ == "__main__":
    main()
