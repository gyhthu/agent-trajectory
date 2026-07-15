# trajectory-collect：Agent 轨迹采集（管道A：OTEL 被动归集 + 请求捕获）

把本机两类大脑的每轮 agent 交互（prompt / 模型输出 / 工具调用 / token / system
prompt / tools schema）被动采集落盘，聚合成可用于 SFT / PRM / off-policy 训练的轨迹 JSONL。

## 架构

```
OpenClaw gateway（diagnostics-otel 插件,        Claude Code（内置 OTel：feishu-event bot、
  captureContent 全开）                           手动 CLI、cron headless 全覆盖）
   │  OTLP http/protobuf traces                    │  OTLP http/protobuf logs（事件）
   └────────────→ 127.0.0.1:4318 ←────────────────┘
otlp_collector.py（systemd --user: trajectory-collector.service）
   │  spans/YYYY-MM-DD.jsonl          logs/YYYY-MM-DD.jsonl
   │
   │  Claude Code → ANTHROPIC_BASE_URL=127.0.0.1:4319（settings.json env）
   │  anthropic_capture_proxy.py（systemd --user: trajectory-capture-proxy.service）
   │  透传 api.anthropic.com，截获请求体 system+tools → requests/YYYY-MM-DD.jsonl
   ▼
trajectory_aggregate.py
   │  --source openclaw     按 traceId 聚合 → trajectories/<date>.jsonl
   │  --source claude-code  按 session.id 聚合 + join transcript 正文（日志路，只有主线）
   │                        + join requests 的 system/tools → <date>.log-main.jsonl
   │  --source api-hijack   api-calls 全量请求按 (session×角色) 拆（捕获代理路）
   │                        → <date>.api-{main,subagent,aux}.jsonl.gz
   │  --publish <共享目录>   按「来源-角色-日期」命名复制到共享目录（见下）
   ▼  每行一条轨迹（四要素齐 + source/role/brain/agent_desc + reward 占位）
```

## 两条采集路 × 角色（核心区分）

- **来源 source**：`log`（OTEL+transcript join，去重后线性对话，可读，做 SFT/PRM）
  vs `api`（捕获代理截 HTTP，逐 call request/response 精确，做 off-policy/slime replay）。
  谁也不全包含谁：log 含被压缩裁掉的历史、api 含每步真实条件+响应，且采集路独立互为兜底。
- **角色 role**：`main`（主线 CC）/ `subagent`（文件搜索·web 搜索·Task）/ `aux`（标题·SDK）。
  **子代理与主线共用同一个 session_id**（寄生其中），只能靠 system 正文签名区分
  （`traj_common.classify_role`）。**日志路看不到子代理**——子代理不写 transcript，
  只在 API 线上出现，所以拆主线/子代理只有 api 路做得到。
- **指纹必须剔 billing 块**：CC SDK 往 system 注入 `x-anthropic-billing-header`，其
  `cch=` 每 call 变；不剔则每 call 都被当独立身份（143 call→143「身份」，实际仅 6）。
  `traj_common.system_tools_sha`/`classify_role` 已统一剔除（代理与聚合器共用）。

## 共享目录发布命名（`/opt/shared/data/transcript/`）

`--publish` 按 `{source}-{role}-{date}` 命名，文件名+行内字段双重区分：
`log-main-<date>.jsonl`、`api-main-<date>.jsonl.gz`、`api-subagent-<date>.jsonl.gz`、
`api-aux-<date>.jsonl.gz`（openclaw 非空时 `log-openclaw-<date>.jsonl`）。

## 常用命令

```bash
# 聚合今天的轨迹并抽 3 条核验（--source openclaw / claude-code / api-hijack / all）
TRAJ_DATA_DIR=/home/agent/trajectory-data \
  python3 scripts/trajectory_aggregate.py --date $(date +%F) --sample 3

# 聚合并发布到共享目录（按 来源-角色-日期 命名，供查质量/训练取数）
TRAJ_DATA_DIR=/home/agent/trajectory-data \
  python3 scripts/trajectory_aggregate.py --date $(date +%F) --source all \
  --publish /opt/shared/data/transcript

# collector / 捕获代理 状态与日志
systemctl --user status trajectory-collector trajectory-capture-proxy
journalctl --user -u trajectory-capture-proxy -f
```

## 关键配置（已生效）

- `~/.openclaw/openclaw.json` → `diagnostics.otel`：`endpoint=http://127.0.0.1:4318`、
  `traces:true, metrics:false, logs:false`、`sampleRate:1.0`、`captureContent.*` 六开关全开
  （正文走 span 的 `openclaw.content.*` 属性，不需要 log body）。
- 插件：`openclaw plugins install clawhub:@openclaw/diagnostics-otel`（要求 runtime ≥2026.6.5）。
- `~/.claude/settings.json` → `env`：`CLAUDE_CODE_ENABLE_TELEMETRY=1`、
  `OTEL_LOGS_EXPORTER=otlp`、`OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf`、
  `OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318`、`OTEL_LOG_USER_PROMPTS=1`。
  settings env 随每次新会话生效（bridge / CLI / cron 全覆盖），不用重启服务。
- `~/.claude/settings.json` → `env.ANTHROPIC_BASE_URL=http://127.0.0.1:4319`：
  所有 claude 进程经捕获代理透传上游。代理挂了 systemd 2s 自动拉起；要彻底
  绕开就删这行 env。
- collector / 代理 默认绑 `127.0.0.1`（BIND_HOST 可改，绝不默认 0.0.0.0）。

## 坑（已踩）

- **Node OTLP exporter 用 chunked 传输**：collector 只读 Content-Length 会拿到空 body、
  解出 0 span，且连接被读坏后 exporter 停发。必须处理 `Transfer-Encoding: chunked`。
- runtime 2026.6.1 装不了 diagnostics-otel（要求 ≥2026.6.5），已升级 openclaw 至 2026.6.5。
- 一轮 span 不是即时到达：BatchSpanProcessor 分批 flush，等 ~1 分钟再聚合。
- **Claude Code 内置 OTel 不导出模型输出正文**：只有 user_prompt 正文 + api_request
  token/cost 计量 + tool 结构（成败/字节数）。正文靠按 session.id join 本地 transcript
  （`~/.claude/projects/*/<session-id>.jsonl`，含完整 messages/thinking/tool 正文）补全，
  聚合脚本已内置该 join。
- **system prompt 和 tools schema 哪里都没有**：transcript 和 OTel 均不含，只存在于
  发往 API 的请求体里——靠 4319 捕获代理在请求线上截获（管道B 轻量版：只记录不改流量）。
- **metadata.user_id 是 JSON 串**：新版 CLI 格式为
  `{"device_id":...,"account_uuid":...,"session_id":"<uuid>"}`，不是旧的
  `user_..._session_<uuid>` 下划线拼接；代理两种都兼容。

## 设计文档

飞书：《Agent 训练数据归集方案》（轨迹采集与训练数据归集 文件夹）
