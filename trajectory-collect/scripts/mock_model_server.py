#!/usr/bin/env python3
"""mock OpenAI 兼容模型服务：管道A 端到端回归测试夹具。

让 OpenClaw 走原生 openai-completions provider 调本服务，
验证 diagnostics-otel 的 captureContent（input/output/system_prompt）→ span → 聚合链路。
只回固定文本，带 usage 计量。监听 127.0.0.1:4099。
"""
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BIND_HOST = os.environ.get("BIND_HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "4099"))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):  # noqa: N802
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        req = json.loads(body or "{}")
        n_in = sum(len(str(m.get("content", ""))) for m in req.get("messages", [])) // 4
        text = "mock 回复：轨迹采集端到端测试 OK（本回复由 mock_model_server 生成）"
        resp = {
            "id": f"chatcmpl-mock-{time.time_ns()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.get("model", "mock-1"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": n_in, "completion_tokens": 20,
                      "total_tokens": n_in + 20},
        }
        payload = json.dumps(resp, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        print(f"[mock] {fmt % args}", flush=True)


if __name__ == "__main__":
    print(f"mock model server on {BIND_HOST}:{PORT}", flush=True)
    ThreadingHTTPServer((BIND_HOST, PORT), Handler).serve_forever()
