#!/usr/bin/env python3
"""Codex (ChatGPT 后端 / Responses 协议) 透传捕获代理。

与 anthropic_capture_proxy.py 是同一套骨架（透传不改流量、原始字节双向、捕获出错
不阻断转发、转发出错显式 502），只把上游从 Anthropic Messages 换成 codex 在
**ChatGPT 登录态**下打的 Responses 协议端点。

劫持点（已实测确认）：
- codex 在 auth_mode=chatgpt 时，base URL 由配置键 `chatgpt_base_url` 决定，
  默认 `https://chatgpt.com/backend-api/codex`，实际请求打到 `<base>/responses`。
  （注意：`openai_base_url` 只对 API-key provider 生效，chatgpt 态用的是
  `chatgpt_base_url`——codex-sdk 的 baseUrl 选项映射到前者，对 chatgpt 态无效，
  必须用 SDK 的 config 选项传 chatgpt_base_url，或 CLI `--config chatgpt_base_url=`。）
- OAuth：codex 子进程自带 `Authorization: Bearer <chatgpt access_token>`，
  本代理原样透传（绝不落盘任何请求头），刷新由 codex 自己用 refresh_token 完成。

接入：把 codex 的 chatgpt_base_url 指到 `http://127.0.0.1:PORT/backend-api/codex`，
本代理把收到的 path（含 /responses）拼到 UPSTREAM 主机透传上去。

落盘（DATA_DIR/codex-api-calls/YYYY-MM-DD.jsonl.gz，多 member gzip）：
  每条 = 一次 API 调用的 request 全量 payload + response（SSE 流原文 + 尽力重建）。
  不落任何请求头；OAuth token 不经磁盘。

这是 off-policy 字符串级捕获（够 SFT / 质量分析）；ChatGPT 后端不返回
token id / logprob，拿不到 token 级——要 token 级须自 serve 模型（见 slime 分析）。
"""
import asyncio
import gzip
import json
import os
import sys
import time
from pathlib import Path

import aiohttp
from aiohttp import web

BIND_HOST = os.environ.get("BIND_HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "4320"))
# 上游主机（不含 /backend-api/codex —— 那段由 codex 发来的 path 自带）。
UPSTREAM = os.environ.get("CODEX_UPSTREAM", "https://chatgpt.com").rstrip("/")
DATA_DIR = Path(os.environ.get("TRAJ_DATA_DIR", "/home/agent/trajectory-data"))

_write_lock = asyncio.Lock()

# 这些头由 HTTP 栈自行管理，不透传（Host 不透传 → 上游拿到正确的 chatgpt.com）
_HOP_HEADERS = {"host", "content-length", "transfer-encoding", "connection",
                "keep-alive", "proxy-authenticate", "proxy-authorization",
                "te", "trailers", "upgrade"}


async def append_jsonl(subdir: str, rec: dict, compress: bool = True):
    day = time.strftime("%Y-%m-%d")
    suffix = ".jsonl.gz" if compress else ".jsonl"
    out = DATA_DIR / subdir / f"{day}{suffix}"
    out.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(rec, ensure_ascii=False) + "\n"
    async with _write_lock:
        if compress:
            with gzip.open(out, "at", encoding="utf-8") as f:
                f.write(line)
        else:
            with open(out, "a", encoding="utf-8") as f:
                f.write(line)


def reconstruct_responses_sse(text: str):
    """从 Responses 协议 SSE 流尽力重建最终输出（output items + usage）。

    Responses 流事件名以 response.* 为主：response.created / response.output_item.added /
    response.output_text.delta / response.reasoning_summary_text.delta /
    response.function_call_arguments.delta / response.completed 等。
    这里做尽力重建：拼出最终文本/工具调用，拿到末尾 response.completed 里的完整对象。
    解析不出就交给上层保留原文，绝不丢数据。
    """
    final_response = None
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    output_items: list[dict] = []   # 从 output_item.done 收集的最终 output 项（含工具调用）
    events_seen = 0
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            ev = json.loads(data)
        except json.JSONDecodeError:
            continue
        events_seen += 1
        t = ev.get("type", "")
        if t == "response.output_text.delta":
            text_parts.append(ev.get("delta", ""))
        elif t in ("response.reasoning_summary_text.delta",
                   "response.reasoning_text.delta"):
            reasoning_parts.append(ev.get("delta", ""))
        elif t == "response.output_item.done":
            # ChatGPT 后端 response.completed 的 output 数组是空的，最终 output 项
            # （message / function_call / reasoning 等）只在 output_item.done 里逐个吐出。
            # 收集这些 done 项才是完整的 output（含工具调用 name/call_id/arguments）。
            item = ev.get("item")
            if isinstance(item, dict):
                output_items.append(item)
        elif t in ("response.completed", "response.incomplete", "response.failed"):
            final_response = ev.get("response", final_response)
    if final_response is None and not events_seen:
        return None
    # final_response.output 兜底：后端给空数组时用流式收集的 output_items 补全，
    # 这样结构化形态对工具调用也完整（SFT 需要）。后端若真给了 output 就尊重它。
    if isinstance(final_response, dict) and not final_response.get("output") and output_items:
        final_response["output"] = output_items
    return {
        "final_response": final_response,           # response.completed 的完整对象（含 usage/output/status）
        "text": "".join(text_parts) or None,         # 拼接出的可读正文（sidecar）
        "reasoning": "".join(reasoning_parts) or None,
        "output_items": output_items or None,        # 流式收集的最终 output 项（含工具调用）
        "events_seen": events_seen,
    }


async def capture_api_call(path: str, req_body: bytes, status: int,
                           resp_headers, resp_body: bytes, had_auth: bool):
    payload = None
    try:
        payload = json.loads(req_body) if req_body else None
    except json.JSONDecodeError:
        payload = {"_raw": req_body.decode("utf-8", errors="replace")}

    if resp_headers.get("Content-Encoding") == "gzip":
        try:
            resp_body = gzip.decompress(resp_body)
        except OSError:
            pass
    text = resp_body.decode("utf-8", errors="replace")

    rec = {
        "ts": time.time(),
        "path": path,
        "status": status,
        "had_auth": had_auth,          # OAuth 头是否随请求透传过来（不落 token 本身）
        "request": payload,            # 全量 request payload（含 model/input/instructions/tools…）
        "response": None,
    }
    # 按内容嗅探而非 Content-Type：HTTP responses 端点回的不是 text/event-stream，
    # 但流体就是 Responses SSE（event:/data: response.*）。先试 SSE 重建，不成再当 JSON。
    is_sse = "data:" in text and "response." in text
    rebuilt = reconstruct_responses_sse(text) if is_sse else None
    if rebuilt is not None:
        rec["response"] = rebuilt
        rec["response_raw"] = text     # SSE 原文一并留底（token 级以下的完整 on-wire）
    else:
        try:
            rec["response"] = json.loads(text)
        except json.JSONDecodeError:
            rec["response_raw"] = text  # 重建/解析都不成必须留原文，绝不丢数据
    await append_jsonl("codex-api-calls", rec, compress=True)
    model = (payload or {}).get("model") if isinstance(payload, dict) else None
    print(f"[codex-api] {path} status={status} auth={had_auth} model={model} "
          f"resp_bytes={len(resp_body)}", flush=True)


async def proxy(request: web.Request) -> web.StreamResponse:
    body = await request.read()
    had_auth = "authorization" in (k.lower() for k in request.headers)
    is_capture = request.method == "POST" and request.path.endswith("/responses")

    # codex 把请求体用 zstd 压缩发来；aiohttp 服务端已据 content-encoding 解压，
    # request.read() 拿到的是明文 body。故转发时必须丢掉 content-encoding（否则上游
    # 会按 zstd 去解明文而失败）。content-length 已在 _HOP_HEADERS 里，由 HTTP 栈重算。
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in _HOP_HEADERS and k.lower() != "content-encoding"}
    # 上游响应也别用 zstd（aiohttp 客户端不支持，会在解析响应头时抛 ContentEncodingError）。
    # 强制 identity：既能透传又能直接 capture 明文，不丢数据。
    headers["Accept-Encoding"] = "identity"
    try:
        async with request.app["client"].request(
            request.method, UPSTREAM + request.path_qs,
            headers=headers, data=body if body else None,
        ) as upstream:
            resp_headers = {k: v for k, v in upstream.headers.items()
                            if k.lower() not in _HOP_HEADERS}
            resp = web.StreamResponse(status=upstream.status, headers=resp_headers)
            await resp.prepare(request)
            buf = bytearray()
            async for chunk in upstream.content.iter_chunked(16384):
                await resp.write(chunk)
                if is_capture:
                    buf.extend(chunk)
            await resp.write_eof()
    except Exception as e:  # 转发失败必须显式报错，禁止静默
        print(f"[proxy-ERROR] {request.method} {request.path}: {e!r}",
              file=sys.stderr, flush=True)
        return web.json_response(
            {"error": {"type": "proxy_error", "message": repr(e)}}, status=502)

    if is_capture:
        try:
            await capture_api_call(request.path, body, upstream.status,
                                   upstream.headers, bytes(buf), had_auth)
        except Exception as e:  # 响应已回传，捕获失败只报错
            print(f"[capture-ERROR] {e!r}", file=sys.stderr, flush=True)
    return resp


async def healthz(_request: web.Request) -> web.Response:
    return web.Response(text="ok")


async def make_app() -> web.Application:
    app = web.Application(client_max_size=256 * 1024 * 1024)
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
    try:
        from aiohttp.http_parser import HAS_ZSTD
    except Exception:
        HAS_ZSTD = "?"
    print(f"Codex capture proxy listening on {BIND_HOST}:{PORT} -> {UPSTREAM}, "
          f"capture -> {DATA_DIR}/codex-api-calls/ (HAS_ZSTD={HAS_ZSTD})", flush=True)
    web.run_app(make_app(), host=BIND_HOST, port=PORT, print=None)


if __name__ == "__main__":
    main()
