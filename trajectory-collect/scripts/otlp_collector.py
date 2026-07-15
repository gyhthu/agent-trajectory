#!/usr/bin/env python3
"""OTLP/HTTP collector：接收 OpenClaw diagnostics-otel 导出的 trace/log（http/protobuf），
原样落盘为 JSONL（管道A：被动轨迹采集的「收」环节）。

- 监听 BIND_HOST:PORT（默认 127.0.0.1:4318，安全规则：默认绝不绑 0.0.0.0）
- POST /v1/traces  → spans/YYYY-MM-DD.jsonl（每行一个 span，含 resource/attributes 全量）
- POST /v1/logs    → logs/YYYY-MM-DD.jsonl
- POST /v1/metrics → 200 丢弃（管道A 不需要 metrics）
- 数据目录 TRAJ_DATA_DIR（默认 /home/agent/trajectory-data）

下游：trajectory_aggregate.py 按 traceId 聚合成轨迹 JSONL。
"""
import gzip
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from opentelemetry.proto.collector.trace.v1 import trace_service_pb2
from opentelemetry.proto.collector.logs.v1 import logs_service_pb2
from opentelemetry.proto.common.v1 import common_pb2

BIND_HOST = os.environ.get("BIND_HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "4318"))
DATA_DIR = Path(os.environ.get("TRAJ_DATA_DIR", "/home/agent/trajectory-data"))

_write_lock = threading.Lock()


def any_value_to_py(v: common_pb2.AnyValue):
    kind = v.WhichOneof("value")
    if kind is None:
        return None
    if kind == "array_value":
        return [any_value_to_py(x) for x in v.array_value.values]
    if kind == "kvlist_value":
        return {kv.key: any_value_to_py(kv.value) for kv in v.kvlist_value.values}
    if kind == "bytes_value":
        return v.bytes_value.hex()
    return getattr(v, kind)


def attrs_to_dict(attrs):
    return {kv.key: any_value_to_py(kv.value) for kv in attrs}


def append_jsonl(subdir: str, records: list):
    day = time.strftime("%Y-%m-%d")
    path = DATA_DIR / subdir / f"{day}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with _write_lock, open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def parse_traces(body: bytes) -> list:
    req = trace_service_pb2.ExportTraceServiceRequest()
    req.ParseFromString(body)
    out = []
    for rs in req.resource_spans:
        resource = attrs_to_dict(rs.resource.attributes)
        for ss in rs.scope_spans:
            for span in ss.spans:
                out.append({
                    "trace_id": span.trace_id.hex(),
                    "span_id": span.span_id.hex(),
                    "parent_span_id": span.parent_span_id.hex() or None,
                    "name": span.name,
                    "start_ns": span.start_time_unix_nano,
                    "end_ns": span.end_time_unix_nano,
                    "status_code": span.status.code,
                    "status_message": span.status.message or None,
                    "attributes": attrs_to_dict(span.attributes),
                    "events": [
                        {"name": e.name, "time_ns": e.time_unix_nano,
                         "attributes": attrs_to_dict(e.attributes)}
                        for e in span.events
                    ],
                    "resource": resource,
                })
    return out


def parse_logs(body: bytes) -> list:
    req = logs_service_pb2.ExportLogsServiceRequest()
    req.ParseFromString(body)
    out = []
    for rl in req.resource_logs:
        resource = attrs_to_dict(rl.resource.attributes)
        for sl in rl.scope_logs:
            for rec in sl.log_records:
                out.append({
                    "trace_id": rec.trace_id.hex() or None,
                    "span_id": rec.span_id.hex() or None,
                    "time_ns": rec.time_unix_nano,
                    "severity": rec.severity_text or rec.severity_number,
                    "body": any_value_to_py(rec.body),
                    "attributes": attrs_to_dict(rec.attributes),
                    "resource": resource,
                })
    return out


class OTLPHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _read_body(self) -> bytes:
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            chunks = []
            while True:
                size_line = self.rfile.readline().strip()
                size = int(size_line.split(b";")[0], 16)
                if size == 0:
                    self.rfile.readline()  # trailing CRLF
                    break
                chunks.append(self.rfile.read(size))
                self.rfile.readline()  # chunk 尾 CRLF
            return b"".join(chunks)
        return self.rfile.read(int(self.headers.get("Content-Length", 0)))

    def do_POST(self):  # noqa: N802
        body = self._read_body()
        if self.headers.get("Content-Encoding") == "gzip":
            body = gzip.decompress(body)
        try:
            if self.path.rstrip("/") == "/v1/traces":
                spans = parse_traces(body)
                append_jsonl("spans", spans)
                print(f"[traces] +{len(spans)} spans (body={len(body)}B)", flush=True)
                if not spans and len(body) > 0 and os.environ.get("DEBUG_DUMP"):
                    dump = DATA_DIR / "debug" / f"traces-{time.time_ns()}.bin"
                    dump.parent.mkdir(parents=True, exist_ok=True)
                    dump.write_bytes(body)
            elif self.path.rstrip("/") == "/v1/logs":
                logs = parse_logs(body)
                append_jsonl("logs", logs)
                print(f"[logs] +{len(logs)} records", flush=True)
            elif self.path.rstrip("/") == "/v1/metrics":
                pass  # 管道A 不需要 metrics，丢弃
            else:
                self.send_error(404)
                return
        except Exception as e:  # 报错必须显示，禁止静默失败
            print(f"[ERROR] {self.path}: {e!r}", file=sys.stderr, flush=True)
            self.send_error(400, str(e))
            return
        # OTLP/HTTP 成功响应：空 protobuf 消息即合法
        self.send_response(200)
        self.send_header("Content-Type", "application/x-protobuf")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):  # noqa: N802 健康检查
        if self.path == "/healthz":
            payload = b"ok"
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.send_error(404)

    def log_message(self, *args):  # 静默 access log（数据量日志已自行打印）
        pass


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((BIND_HOST, PORT), OTLPHandler)
    print(f"OTLP collector listening on {BIND_HOST}:{PORT}, data -> {DATA_DIR}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
