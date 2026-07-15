# Agent 轨迹洞察（一）：Codex

> 本文解释两件事：Codex 的模型流量在什么位置被“劫持”，以及捕获到的 Responses wire 记录如何被归一化为可分析、可用于行为克隆（BC）/SFT 的轨迹。
>
> 代码基线：`openai-codex` commit `2f7d89b1419bf7064346855b0acde23514b1ebc5`（2026-07-13）。

## 结论先行

这套方案没有修改 Codex 的 agent loop、上下文组装、工具执行或 OAuth 刷新逻辑，也不是 TLS MITM。它利用 Codex 已有的 model provider 配置，在**模型客户端选择传输协议和构造 Responses 请求 URL 的边界**插入一个本地透明代理：

1. 自定义 provider 设置 `supports_websockets=false`，使 Codex 不走 Responses WebSocket，直接进入 HTTP Responses 分支；
2. provider 的 `base_url` 指向 `127.0.0.1:4320/backend-api/codex`，于是 `POST .../responses` 先到本地代理；
3. 代理把请求原样转发到 `https://chatgpt.com`，同时旁路记录请求 payload 和响应 SSE；
4. `codex_wire_adapter.py` 把每次 `/responses` 调用转换为一条 `context → completion` 轨迹，统一消息角色、工具调用、usage、采样参数和保真度标记。

因此，更准确的表述不是“侵入 Codex 内部埋点”，而是：

> **在 Codex 的 Responses transport/provider seam 上做配置型流量改道，再以透明反向代理旁路采集。**

## 1. 三份核心代码各自负责什么

| 代码 | 职责 | 所在阶段 |
|---|---|---|
| `openai-codex/` | 决定 provider、base URL、ChatGPT OAuth、Responses WebSocket/HTTP 分支，并组装真实请求 | Codex 运行时 |
| `trajectory-collect/scripts/codex_capture_proxy.py` | 透明转发 HTTP 流量，旁路捕获 `/responses` 请求和 SSE 响应 | 在线采集 |
| `trajectory_extract/codex_wire_adapter.py` | 把每条 wire call 转为统一的 OpenAI-messages 形态，并按需脱敏发布 | 离线归一化 |

完整链路如下：

```mermaid
flowchart LR
    U["用户任务"] --> A["Codex agent loop"]
    A --> C["上下文与工具定义组装"]
    C --> M["ModelClientSession.stream"]

    M --> G{"provider.supports_websockets?"}
    G -- "true：默认路径" --> WS["Responses WebSocket"]
    WS --> CG["ChatGPT Codex backend"]

    G -- "false：采集路径" --> HTTP["HTTP Responses client"]
    HTTP --> P["127.0.0.1:4320<br/>codex_capture_proxy.py"]
    P --> CG
    CG --> P
    P --> HTTP

    P -. "旁路落盘，不阻断回包" .-> RAW["codex-api-calls/YYYY-MM-DD.jsonl.gz"]
    RAW --> AD["codex_wire_adapter.py"]
    AD --> N["归一化 trajectory JSONL"]
    N --> BC["质量分析 / BC / SFT"]
```

## 2. 劫持到底插在 Codex 的什么位置

### 2.1 第一刀：在 transport capability gate 关闭 WebSocket

Codex 的 `ModelClientSession.stream()` 优先尝试 Responses WebSocket，但是否启用由 provider 能力位控制：

- `responses_websocket_enabled()` 在 `provider.supports_websockets == false` 时直接返回 `false`；
- `stream()` 因而跳过 `stream_responses_websocket()`；
- 随后进入 `stream_responses_api()`，即 HTTP `POST /responses` 分支。

这不是“先让 WebSocket 失败再抓 fallback”，而是**在发起请求前通过 provider capability 明确选择 HTTP**。相应源码位置：

- `openai-codex/codex-rs/core/src/client.rs:926`：WebSocket 能力判断；
- `openai-codex/codex-rs/core/src/client.rs:1774`：WS/HTTP 总分流；
- `openai-codex/codex-rs/core/src/client.rs:1395`：HTTP Responses 实现。

#### 如果继续走 WebSocket，就一定无法捕获吗？

**不是原理上不能，而是当前这份代理抓不到。**原因要分两种情况看：

1. **使用默认 ChatGPT provider URL**：Codex 会直接建立到 ChatGPT 的 `wss://.../responses` 长连接，流量根本不经过 `127.0.0.1:4320`，本地代理自然看不到。
2. **把 provider base URL 指向本机，但仍声明支持 WebSocket**：Codex 会把 `http://127.0.0.1:4320/.../responses` 自动转换为 `ws://127.0.0.1:4320/.../responses`。流量虽然到了本机，但现有 `codex_capture_proxy.py` 只实现普通 HTTP 请求和 SSE 响应转发，没有实现 WebSocket Upgrade、双向 frame relay 和 frame 级落盘，因此仍不能正确代理和采集。

当前捕获器的数据模型也明确只认：

```text
POST .../responses → 一个 JSON request body → 一条 SSE response stream
```

而 WebSocket 的数据模型是：

```text
一次 HTTP 101 Upgrade → 一条可跨多次调用复用的长连接
                      → client 发送 response.create JSON frame
                      ← server 返回多条 response.* JSON frame
```

HTTP 模式下“一次 POST 就是一条 call”，边界天然清楚；WebSocket 模式下，连接边界不等于调用边界，还需要按 `response.create`、response id、终止事件、重连和 prewarm frame 重新拼 call。

如果要保留 WebSocket 也可以另写一套 WS-aware capture relay，它至少需要：

- 接受 Codex 的 WebSocket Upgrade；
- 再向 ChatGPT 建立上游 `wss` 连接；
- 双向透明转发 text/binary/ping/pong/close frame；
- 从出站 `response.create` frame 提取 instructions、input、tools；
- 从入站 `response.*` frame 重建 output、usage 和结束状态；
- 处理连接复用、prewarm（`generate=false`）、断线重连和一次连接内的多个 response；
- 继续保证鉴权 header 只透传、不落盘。

所以设置 `supports_websockets=false` 的真正意义是：**把流量降到现有采集器已经完整支持、且每次调用边界天然明确的 HTTP POST + SSE 协议**，而不是声称 WebSocket 天生无法观测。

#### WebSocket 和 HTTP Responses 的运行差异

两条路径承载的核心 Responses 语义相同，都可以流式返回 `response.*` 事件。差异主要不在“能否流式”，而在**连接和上下文如何复用**：

| 维度 | HTTP POST + SSE | Responses WebSocket |
|---|---|---|
| 通信形态 | 一次 POST 对应一条响应流 | 一次 Upgrade 后保持双向长连接 |
| 请求单位 | 每次调用发送一个 HTTP body | 每次调用发送一个 `response.create` frame |
| 响应单位 | SSE `data:` event | WebSocket JSON frame |
| 连接复用 | HTTP client 可复用底层 TCP/TLS，但每次仍是独立 HTTP request | Codex 显式缓存并复用同一个 Responses WS connection |
| 首次请求准备 | 正常发起 POST | 可提前 preconnect/prewarm，正式请求到来时复用连接 |
| 多轮上下文 | 当前采集路径看到每次请求的完整 `input` | 若上下文是上一轮的严格扩展，可发 `previous_response_id + incremental input` |
| 调用边界 | POST 开始到 SSE 结束，天然明确 | 必须按 `response.create`、response id 和完成事件切分 |
| 代理实现难度 | 普通反向代理加 SSE buffer 即可 | 需要 Upgrade、双向 frame relay 和有状态重建 |

给非网络背景的同事，可以把 HTTP 理解成“每次办事都单独提交一张表”，把 WebSocket 理解成“电话接通后保持通话”：

```mermaid
flowchart TB
    subgraph HTTP["HTTP：每次重新办一件事"]
        H1["请求 1"] --> H2["响应 1"]
        H3["请求 2"] --> H4["响应 2"]
        H5["请求 3"] --> H6["响应 3"]
    end

    subgraph WS["WebSocket：一条持续通话的连接"]
        C["Codex"] <--> S["模型服务器"]
        C -. "工具结果、新指令" .-> S
        S -. "文本、工具调用" .-> C
    end
```

#### 为什么 Codex 原版优先选择 WebSocket

当前源码没有一句产品说明直接写“选择 WebSocket 是为了降低多少毫秒”，但实现明确围绕以下优化设计；“降低交互延迟和重复传输”是由这些机制可以直接支持的工程判断：

1. **连接预热**：Codex 可先发送 `generate=false` 的 v2 `response.create`，完成连接与服务端状态准备，但不做正式生成。用户请求真正开始时可以复用该连接和 warmup response id。
2. **长连接复用**：`ModelClientSession` 缓存 WebSocket connection。一次 turn 中可能发生多次“模型输出工具调用 → 本地执行工具 → 把结果交回模型”，无需为每一步重新建立 WS 会话。
3. **增量发送上下文**：当新请求的 input 是上一请求及其 response items 的严格扩展，Codex 会设置 `previous_response_id`，并只发送新增 item。例如上一轮已经发送过 user message 和 assistant tool call，下一轮可以只发新的 tool result，而不是重复上传全部历史。
4. **服务端连续性**：Codex保存 `x-codex-turn-state` 等 turn 级状态用于 sticky routing，使连续请求更容易落在兼容的服务端上下文上。
5. **仍保留 HTTP fallback**：若 provider 不支持 WebSocket、服务端要求切换，或 WS 失败达到回退条件，Codex仍能使用 HTTP Responses，不把可用性押在单一传输上。

这对 coding agent 特别有价值，因为一次用户任务通常不是“一问一答”，而是：

```mermaid
flowchart LR
    U["用户指令"] --> M1["模型调用 1"]
    M1 --> T1["执行工具"]
    T1 --> M2["模型调用 2"]
    M2 --> T2["执行工具"]
    T2 --> M3["模型调用 3"]
    M3 --> A["最终回答"]
```

WebSocket 把这串高频模型调用放在可预热、可复用、可增量续接的通道上。需要注意，增量传输主要减少客户端重复上传和协议开销，**不等于模型少看了历史，也不能据此推断 token 计费一定下降**；服务端仍需恢复完整上下文来完成推理。

对轨迹采集而言，这也解释了为什么强制 HTTP 很方便：HTTP 请求里有完整上下文，天然得到一条独立的 `(context → completion)` 样本；若直接采 WebSocket，后续 frame 可能只有 `previous_response_id + delta`，采集器必须维护连接级状态，把多个 frame 和先前响应重新拼成模型当时真正可见的完整上下文。

### 2.2 第二刀：在 provider base URL 把 HTTP 请求导向本机

HTTP 分支本身仍然只是 Codex 的标准 Responses client。真正决定请求发往哪里的，是 `ModelProviderInfo::to_api_provider()`：

- provider 显式设置了 `base_url` 时，Codex优先使用该值；
- 未设置时，ChatGPT 登录态才回落到内置的 ChatGPT Codex URL；
- Responses client 再在 base URL 后拼接 `responses` endpoint。

因此，按当前源码最直接、最可验证的配置形态是：

```toml
model_provider = "proxycap"

[model_providers.proxycap]
name = "proxycap"
base_url = "http://127.0.0.1:4320/backend-api/codex"
wire_api = "responses"
requires_openai_auth = true
supports_websockets = false
```

这里两个配置缺一不可：

- `supports_websockets=false` 解决“走哪种传输”；
- `base_url=http://127.0.0.1:4320/...` 解决“HTTP 发到哪里”。

`requires_openai_auth=true` 让 provider 继续使用 Codex 的 OpenAI/ChatGPT 登录态。OAuth bearer token 和账号头由 Codex 的 auth provider 附加，代理只负责透传；token 刷新仍由 Codex 自己完成。

### 2.3 第三刀：代理只在 `/responses` 上旁路取证

`codex_capture_proxy.py` 对所有路径都可以透传，但只有同时满足以下条件才捕获：

```text
method == POST && path.endswith("/responses")
```

所以健康检查或其他后端请求不会混进轨迹数据。捕获点位于：

```text
Codex 已完成 instructions/input/tools 组装
                    ↓
            HTTP 请求已经序列化
                    ↓
        [本地代理在这里复制 request/response]
                    ↓
             ChatGPT Codex backend
```

这也是其保真度较高的原因：捕获内容就是模型端实际收到的逻辑 payload，而不是事后从 UI 日志反推。

## 3. 一次模型调用如何穿过代理

```mermaid
sequenceDiagram
    autonumber
    participant A as Codex agent loop
    participant M as ModelClientSession
    participant P as codex_capture_proxy.py
    participant O as chatgpt.com
    participant D as gzip JSONL

    A->>M: Prompt(instructions, input, tools)
    Note over M: supports_websockets=false
    M->>P: POST /backend-api/codex/responses<br/>Authorization + zstd body
    Note over P: aiohttp 已将请求体解压为明文<br/>不记录任何请求头
    P->>O: POST 同一路径<br/>透传鉴权，重算长度，Accept-Encoding: identity
    O-->>P: Responses SSE chunks
    loop 每个 16 KiB chunk
        P-->>M: 立即流式回传
        P->>P: 同时追加到旁路 buffer
    end
    M-->>A: 标准 ResponseStream events
    P->>P: 重建 final_response / output_items / text
    P->>D: 追加一条 JSONL gzip member
```

### 3.1 请求侧处理

代理调用 `request.read()` 读取 body。Codex 的请求可使用 zstd；aiohttp server 已按 `Content-Encoding` 解压，所以此时拿到的是 JSON 明文。转发时代理必须移除：

- 原来的 `Content-Encoding`，避免上游把明文再次当 zstd 解压；
- `Host`、`Content-Length`、`Transfer-Encoding` 等 hop-by-hop/连接级头，由新的 HTTP 栈重建。

其余头，包括 `Authorization`，原样传给上游。代理只落一个布尔值 `had_auth`，**不落任何 header，也不落 bearer token**。

### 3.2 Codex 发给 ChatGPT 的 HTTP 请求字段

这里的“顶层字段”专指 `POST /backend-api/codex/responses` 的 JSON body，不包括 `Authorization`、`Content-Encoding` 等 HTTP header。当前源码中的 `ResponsesApiRequest` 一共有 15 个顶层候选字段；带 `Option` 的字段为空时不会序列化，`instructions` 为空字符串时也会省略。因此，不是每次抓包都会同时看到全部 15 个字段。

| 顶层字段 | 类型 | 是否稳定出现 | 作用与兼容注意事项 |
|---|---|---:|---|
| `model` | string | 是 | 请求的模型 ID。ChatGPT 后端可能通过响应头或最终 response 告知实际路由模型。 |
| `instructions` | string | 标准 Responses 是 | Codex 的 base/system instructions；Responses Lite 中为空或省略，内容改放进 `input` 的 developer message。 |
| `input` | array | 是 | 模型本轮可见的上下文，由多种 `ResponseItem` 组成，不等同于 Chat Completions 的纯 messages 数组。 |
| `tools` | array / null | 标准 Responses 通常是 | 工具定义及 JSON Schema；Responses Lite 中改放进 `input[].type=additional_tools`。 |
| `tool_choice` | string | 是 | 当前 Codex 固定为 `"auto"`，允许模型决定是否调用工具。 |
| `parallel_tool_calls` | boolean | 是 | 是否允许并行工具调用；Responses Lite 下固定为 `false`。 |
| `reasoning` | object / null | 通常是 | 推理强度、摘要和上下文范围，子字段见下文。 |
| `store` | boolean | 是 | 当前仅 Azure Responses endpoint 会设为 `true`，ChatGPT/普通自定义 provider 通常为 `false`。 |
| `stream` | boolean | 是 | HTTP 分支固定为 `true`，服务端以 SSE 流返回事件。 |
| `stream_options` | object | 条件出现 | OpenAI provider 且启用并发 reasoning summary 时，控制摘要的流式交付方式。 |
| `include` | array | 是 | 当前固定包含 `"reasoning.encrypted_content"`，请求服务端附带不可读的 reasoning continuation state。 |
| `service_tier` | string | 条件出现 | 服务等级；不支持的自部署服务应忽略，而不是因此拒绝整个请求。 |
| `prompt_cache_key` | string | 通常是 | Codex 生成的 prompt cache 标识，用于提高相同前缀的缓存命中。 |
| `text` | object | 条件出现 | 输出 verbosity 或 JSON Schema 格式控制。 |
| `client_metadata` | object | 通常是 | Codex/OpenAI 使用的客户端元数据；非 OpenAI provider 可安全忽略未识别键。 |

对应的源码定义在 `openai-codex/codex-rs/codex-api/src/common.rs:215-239`，实际赋值在 `openai-codex/codex-rs/core/src/client.rs:890-906`。

#### 一个接近真实的完整请求例子

下面是“已经发生过一次工具调用，Codex 把调用结果送回模型继续生成”的示意。实际 instructions 和 tools 通常很长，这里只保留结构：

```json
{
  "model": "gpt-5-codex",
  "instructions": "You are Codex, a coding agent. Follow repository instructions...",
  "input": [
    {
      "type": "message",
      "role": "user",
      "content": [
        {"type": "input_text", "text": "检查这个仓库的测试失败原因"}
      ]
    },
    {
      "type": "reasoning",
      "id": "rs_123",
      "summary": [],
      "content": null,
      "encrypted_content": "opaque-encrypted-blob"
    },
    {
      "type": "message",
      "role": "assistant",
      "phase": "commentary",
      "content": [
        {"type": "output_text", "text": "我先运行测试并定位第一个失败。"}
      ]
    },
    {
      "type": "function_call",
      "id": "fc_123",
      "call_id": "call_123",
      "name": "exec_command",
      "arguments": "{\"cmd\":\"pytest -q\"}"
    },
    {
      "type": "function_call_output",
      "call_id": "call_123",
      "output": "1 failed, 23 passed ..."
    }
  ],
  "tools": [
    {
      "type": "function",
      "name": "exec_command",
      "description": "Run a command in a PTY",
      "parameters": {
        "type": "object",
        "properties": {
          "cmd": {"type": "string"},
          "workdir": {"type": "string"}
        },
        "required": ["cmd"],
        "additionalProperties": false
      }
    }
  ],
  "tool_choice": "auto",
  "parallel_tool_calls": true,
  "reasoning": {
    "effort": "medium",
    "summary": "auto"
  },
  "store": false,
  "stream": true,
  "include": ["reasoning.encrypted_content"],
  "prompt_cache_key": "session-derived-cache-key",
  "text": {"verbosity": "medium"},
  "client_metadata": {"originator": "codex_cli_rs"}
}
```

第一次请求通常只有 instructions、用户输入和 tools；后续请求才逐渐累积 assistant message、reasoning、function call 和 function call output。一次 Codex turn 因而可能对应多次这样的 HTTP call。

#### `instructions`：它是什么，为什么重要

标准 Responses 中，`instructions` 是独立于 `input` 的顶层字符串，由 `prompt.base_instructions.text` 写入。它通常包含 Codex 的身份、工作方式、工具使用约束、输出要求和安全边界，是模型实际看到的最高层基础指令之一：

```json
{
  "instructions": "You are Codex, a coding agent. You and the user share a workspace..."
}
```

需要区分三类容易混在一起的指令：

| 语义来源 | 标准 Responses wire 位置 | 建议归一化语义 |
|---|---|---|
| Codex base instructions | 顶层 `instructions` | `role=system, subtype=base_instructions` |
| 运行时 developer instructions | `input[].type=message, role=developer` | `role=developer, subtype=runtime_developer_instructions` |
| 用户请求 | `input[].type=message, role=user` | `role=user` |

标准 Responses 例子：

```json
{
  "instructions": "Codex base instructions...",
  "input": [
    {
      "type": "message",
      "role": "developer",
      "content": [
        {"type": "input_text", "text": "本仓库修改后必须运行单元测试。"}
      ]
    },
    {
      "type": "message",
      "role": "user",
      "content": [
        {"type": "input_text", "text": "修复登录失败。"}
      ]
    }
  ]
}
```

Responses Lite 不再使用顶层 `instructions`，而是把同一份 base instructions 搬成 developer message：

```json
{
  "input": [
    {
      "type": "additional_tools",
      "role": "developer",
      "tools": []
    },
    {
      "type": "message",
      "role": "developer",
      "content": [
        {"type": "input_text", "text": "Codex base instructions..."}
      ]
    }
  ],
  "parallel_tool_calls": false,
  "reasoning": {"context": "all_turns"}
}
```

因此，不能把所有 `role=developer` 都简单等同为同一种 prompt：Lite 的第一条 Codex 注入 developer message 在语义上对应标准 Responses 的顶层 base instructions，其他 developer message 才可能是 repo、管理员或运行时附加指令。

#### `input` 中不同 `type` 的含义

当前 Codex 的 `ResponseItem` 枚举能够识别下列 wire type。常规代码 agent 轨迹最重要的是前六类；后面的类型只在相应能力启用时出现。

| `input[].type` | 关键字段 | 含义 |
|---|---|---|
| `message` | `role`, `content`, `phase?` | developer/user/assistant 消息；最基本的上下文项。 |
| `additional_tools` | `role`, `tools` | Responses Lite 把顶层 tools 搬入 input 后的表示。 |
| `reasoning` | `summary`, `content?`, `encrypted_content?` | 上一轮推理项；ChatGPT 常见为密文，兼容服务也可返回明文 reasoning text。 |
| `function_call` | `name`, `arguments`, `call_id` | 模型请求调用一个 JSON Schema function tool。 |
| `function_call_output` | `call_id`, `output` | Codex 执行 function 后回传的结果。 |
| `custom_tool_call` | `name`, `input`, `call_id` | freeform/custom tool 调用，输入不一定是 JSON object。 |
| `custom_tool_call_output` | `call_id`, `output` | custom tool 的执行结果。 |
| `local_shell_call` | `call_id`, `status`, `action` | Responses 原生 shell action。 |
| `agent_message` | `author`, `recipient`, `content` | agent 之间的定向消息，内容可能是明文或加密内容。 |
| `tool_search_call` | `call_id`, `execution`, `arguments` | 动态搜索可用工具。 |
| `tool_search_output` | `call_id`, `status`, `tools` | 工具搜索结果。 |
| `web_search_call` | `status`, `action` | 服务端内建 web search 事件。 |
| `image_generation_call` | `status`, `result`, `revised_prompt?` | 服务端图片生成结果。 |
| `compaction` | `encrypted_content` | 服务端产生的加密上下文压缩项；旧名 `compaction_summary` 也可解析。 |
| `compaction_trigger` | 无 | 请求服务端执行压缩的控制项，不是持久输出。 |
| `context_compaction` | `encrypted_content?` | 另一种上下文压缩表示。 |

`message.content[]` 自身还有第二层 type：

```json
{
  "type": "message",
  "role": "user",
  "content": [
    {"type": "input_text", "text": "解释这张图"},
    {"type": "input_image", "image_url": "data:image/png;base64,...", "detail": "high"}
  ]
}
```

| `content[].type` | 方向 | 关键字段 |
|---|---|---|
| `input_text` | 输入 | `text` |
| `input_image` | 输入 | `image_url`, `detail?` |
| `output_text` | assistant 历史输出 | `text` |

其中 assistant message 还可能带 `phase=commentary | final_answer`。这个字段用于区分过程性说明和终局回答，但 provider/model 并不总是返回，归一化时必须允许 `phase` 缺失。

#### 工具调用必须保持 `call_id` 闭环

工具调用的 `arguments` 在 wire 上是“包含 JSON 的字符串”，不是已经解析好的对象：

```json
{
  "type": "function_call",
  "call_id": "call_abc",
  "name": "exec_command",
  "arguments": "{\"cmd\":\"rg --files\"}"
}
```

Codex 执行后，下一次请求使用相同 `call_id` 回传结果：

```json
{
  "type": "function_call_output",
  "call_id": "call_abc",
  "output": "README.md\nsrc/main.rs\n"
}
```

`function_call.call_id` 与 `function_call_output.call_id` 是 agent 工具闭环的关联键。兼容 Qwen/vLLM 时，即使 reasoning 暂时不支持，也必须先保证 call name、arguments 和 call_id 能按 Responses 协议稳定往返，否则 Codex 无法继续 agent loop。

#### `reasoning`：请求配置和历史 item 是两回事

顶层 `reasoning` 是“本次如何推理”的配置：

```json
{
  "reasoning": {
    "effort": "medium",
    "summary": "auto",
    "context": "all_turns"
  }
}
```

而 `input[].type=reasoning` 是“上一轮已经产生的推理项”：

```json
{
  "type": "reasoning",
  "id": "rs_123",
  "summary": [],
  "content": null,
  "encrypted_content": "opaque-encrypted-blob"
}
```

二者不能混为一个字段。Codex 请求 `include=["reasoning.encrypted_content"]` 后，ChatGPT 可以把不可读状态作为 reasoning item 返回；下一轮 Codex 原样放回 `input`，它的用途是延续推理状态，不是给客户端解密阅读。自部署 Qwen 若返回明文，可把 `content` 写成 `[{"type":"reasoning_text","text":"..."}]`，不需要伪造 `encrypted_content`。

### 3.3 ChatGPT 传给 Codex 的顶层字段

HTTP 请求设置了 `stream=true`，所以 ChatGPT 不是一次性返回一个普通 JSON，而是在 response body 中连续发送 SSE 事件。每个事件的基本 wire 形态是：

```text
event: response.output_text.delta
data: {"type":"response.output_text.delta", ...}

```

Codex 和捕获代理真正解析的是每行 `data:` 后面的 JSON。事件顶层至少有 `type`，大多数还带 `sequence_number`；其余字段由事件类型决定，不能把所有事件按同一个 schema 解析。

下面的字段表由两份脱敏真实样本交叉验证：

- `codex_wire_raw_single_call_2026-07-13_redacted.json`：返回加密 reasoning 和最终 assistant message；
- `codex_responses_lite_raw_with_tool_call_2026-07-15_redacted.json`：Responses Lite，请求包含 custom tool 历史，本轮返回加密 reasoning 和 `function_call(name=wait)`。

#### SSE 生命周期和 item 边界事件

| SSE `type` | 顶层字段 | 作用 |
|---|---|---|
| `response.created` | `type`, `response`, `sequence_number` | response 已创建；此时 `response.status` 通常为 `in_progress`，真实样本的 `response.output=[]`。 |
| `response.in_progress` | `type`, `response`, `sequence_number` | 服务端进入生成阶段；真实样本的 `response.output` 仍为空。 |
| `response.output_item.added` | `type`, `item`, `output_index`, `sequence_number` | 开始生成一个 reasoning、message 或 tool-call item；item 通常还不完整。 |
| `response.output_item.done` | `type`, `item`, `output_index`, `sequence_number` | 一个完整结构化 item 已完成，是重建最终 output 的关键事件。 |
| `response.completed` | `type`, `response`, `sequence_number` | 本轮成功结束，包含最终状态、实际模型和 usage。ChatGPT 实测 `response.output=[]`，完整 items 仍需从 `output_item.done` 收集。 |
| `response.incomplete` | `type`, `response`, `sequence_number` | 因 token/context 等原因未完整结束；原因在 `response.incomplete_details`。 |
| `response.failed` | `type`, `response`, `sequence_number` | 本轮失败；错误在 `response.error`。 |

两份样本都验证了如下顺序骨架：

```mermaid
sequenceDiagram
    participant S as ChatGPT
    participant C as Codex

    S-->>C: response.created
    S-->>C: response.in_progress
    S-->>C: output_item.added(reasoning)
    S-->>C: output_item.done(reasoning)
    S-->>C: output_item.added(message 或 function_call)
    S-->>C: 内容或参数 delta...
    S-->>C: 内容或参数 done
    S-->>C: output_item.done(message 或 function_call)
    S-->>C: response.completed
```

#### Assistant 文本 SSE 事件

普通文本回答样本实际出现：

| SSE `type` | 顶层字段 | 作用 |
|---|---|---|
| `response.content_part.added` | `type`, `content_index`, `item_id`, `output_index`, `part`, `sequence_number` | 开始一个 message content part；`part.type=output_text`。 |
| `response.output_text.delta` | `type`, `content_index`, `delta`, `item_id`, `logprobs`, `obfuscation`, `output_index`, `sequence_number` | 文本增量；代理拼接 `delta` 得到可读正文。 |
| `response.output_text.done` | `type`, `content_index`, `item_id`, `logprobs`, `output_index`, `sequence_number`, `text` | 当前 output text 的完整文本。 |
| `response.content_part.done` | `type`, `content_index`, `item_id`, `output_index`, `part`, `sequence_number` | 当前 content part 完成。 |

一个简化的文本 delta 是：

```json
{
  "type": "response.output_text.delta",
  "content_index": 0,
  "delta": "我先检查",
  "item_id": "msg_...",
  "logprobs": [],
  "obfuscation": "...",
  "output_index": 1,
  "sequence_number": 8
}
```

`delta` 才是正文增量；`item_id + output_index + content_index` 负责定位它属于哪个 message/content part，`sequence_number` 用于恢复事件顺序。样本虽然带 `logprobs` 键，但值为空，不能据此声称拿到了 token-level logprob。

#### Function call 参数 SSE 事件

带工具调用的 Lite 样本实际出现：

| SSE `type` | 顶层字段 | 作用 |
|---|---|---|
| `response.function_call_arguments.delta` | `type`, `delta`, `item_id`, `obfuscation`, `output_index`, `sequence_number` | function arguments 的字符串增量；该样本共 19 个 delta。 |
| `response.function_call_arguments.done` | `type`, `arguments`, `item_id`, `output_index`, `sequence_number` | 完整 arguments JSON 字符串。 |

完整 function call 最终仍由 `response.output_item.done.item` 给出：

```json
{
  "id": "fc_...",
  "type": "function_call",
  "status": "completed",
  "arguments": "{\"cell_id\":\"...\",\"yield_time_ms\":10000,\"max_tokens\":10000}",
  "call_id": "call_...",
  "name": "wait",
  "internal_chat_message_metadata_passthrough": {"turn_id": "..."},
  "metadata": {"turn_id": "..."}
}
```

这里有三个不能破坏的约束：

1. `arguments` 在 wire 上是“包含 JSON 的字符串”，不是已解析对象；
2. `name` 必须对应请求 tools 中声明的 function name；
3. `call_id` 必须在下一次请求的 `function_call_output.call_id` 中原样返回，才能闭合工具调用。

本轮 response 只产生 function call；Codex 执行工具之后，执行结果会放进**下一次** HTTP 请求，而不会出现在同一次 SSE response 中。

#### Reasoning SSE 事件

Responses wire 中与可读 reasoning 有关的事件如下。两份 ChatGPT 样本均未实际出现这些事件，因此表中同时标出当前 Codex/代理的显式处理能力，避免把“协议存在”误写成“本次实测返回”：

| SSE `type` | 主要顶层字段 | 当前处理方式 |
|---|---|---|
| `response.reasoning_summary_part.added` | `type`, `item_id?`, `output_index?`, `summary_index`, `sequence_number?` | Codex 建立一个 summary part；代理只计入 `events_seen`。 |
| `response.reasoning_summary_text.delta` | `type`, `delta`, `item_id`, `output_index`, `summary_index`, `sequence_number` | Codex 消费摘要增量；代理拼接 `delta` 到 `response.reasoning`。 |
| `response.reasoning_summary_text.done` | `type`, `text`, `item_id`, `output_index`, `summary_index`, `sequence_number` | Codex 消费完整摘要；代理当前不单独处理 done。 |
| `response.reasoning_text.delta` | `type`, `delta`, `item_id`, `output_index`, `content_index`, `sequence_number` | Codex 消费明文 reasoning 增量；代理拼接 `delta` 到 `response.reasoning`。 |
| `response.reasoning_text.done` | `type`, `text`, `item_id`, `output_index`, `content_index`, `sequence_number` | 协议可出现；当前代理不单独处理，当前 Codex SSE mapper 也未显式映射该 done 事件。原文仍保存在 `response_raw`。 |

但两份 ChatGPT 样本均未出现任何可读 reasoning delta，而是直接在 `output_item.done` 返回加密 reasoning item：

```json
{
  "id": "rs_...",
  "type": "reasoning",
  "content": [],
  "encrypted_content": "opaque-encrypted-blob",
  "internal_chat_message_metadata_passthrough": {"turn_id": "..."},
  "summary": [],
  "metadata": {"turn_id": "..."}
}
```

因此样本中的 `response.reasoning=null` 只表示“代理没有拼到可读 reasoning delta”，不表示模型没有进行 reasoning。带工具调用的样本同时记录了 `usage.output_tokens_details.reasoning_tokens=12`。

#### `response.output_item.done.item` 的三种核心结构

| `item.type` | 关键字段 | 两份样本中的结果 |
|---|---|---|
| `reasoning` | `id`, `content`, `encrypted_content`, `summary`, `metadata` | 两份都有；content/summary 为空，encrypted content 非空。 |
| `message` | `id`, `status`, `content`, `phase`, `role`, `metadata` | 普通调用样本有；`role=assistant`, `phase=final_answer`。 |
| `function_call` | `id`, `status`, `arguments`, `call_id`, `name`, `metadata` | Lite 工具样本有；调用 `wait`，arguments 为 JSON 字符串。 |

完成后的 assistant message 实际结构还包含 annotations/logprobs：

```json
{
  "id": "msg_...",
  "type": "message",
  "status": "completed",
  "content": [
    {
      "type": "output_text",
      "annotations": [],
      "logprobs": [],
      "text": "..."
    }
  ],
  "phase": "final_answer",
  "role": "assistant",
  "metadata": {"turn_id": "..."}
}
```

#### `response.created/in_progress/completed.response` 的顶层字段

两份样本中的 `response` 对象具有同一组顶层字段：

| 字段组 | 实际字段 |
|---|---|
| 身份与状态 | `id`, `object`, `created_at`, `status`, `background`, `completed_at`, `error`, `incomplete_details` |
| 模型与输入回显 | `model`, `instructions`, `tools`, `tool_choice`, `parallel_tool_calls`, `reasoning`, `text` |
| 生成控制 | `max_output_tokens`, `max_tool_calls`, `temperature`, `top_p`, `frequency_penalty`, `presence_penalty`, `top_logprobs`, `truncation` |
| 状态与缓存 | `previous_response_id`, `prompt_cache_key`, `prompt_cache_retention`, `store`, `service_tier` |
| 安全与工具统计 | `moderation`, `safety_identifier`, `tool_usage` |
| 输出与计费 | `output`, `usage` |
| 其余透传 | `user`, `metadata` |

其中 `response.completed.response.usage` 是归一化最重要的计量字段：

```json
{
  "input_tokens": 39358,
  "input_tokens_details": {
    "cache_write_tokens": 0,
    "cached_tokens": 38656
  },
  "output_tokens": 45,
  "output_tokens_details": {
    "reasoning_tokens": 12
  },
  "total_tokens": 39403
}
```

注意，当前样本没有 `end_turn`；Codex 客户端允许该字段缺失并使用兜底结束逻辑，不能把它列成 ChatGPT 当前稳定返回字段。

### 3.4 代理如何解析 SSE

代理一边把上游 chunk 写回 Codex，一边把同一批字节复制到旁路 buffer。这样在线响应优先，采集不会阻塞 agent：

- 转发失败：显式返回 `502 proxy_error`；
- 捕获/落盘失败：响应已经返回给 Codex，只记录 `capture-ERROR`；
- 不依赖响应 `Content-Type`，只要正文同时出现 `data:` 和 `response.` 就尝试重建 SSE。

`reconstruct_responses_sse()` 当前执行四类操作：

| wire 事件 | 解析动作 | 落盘位置 |
|---|---|---|
| 每个可解析的 `data:` JSON | `events_seen += 1` | `response.events_seen` |
| `response.output_text.delta` | 拼接 `delta` | `response.text` |
| `response.reasoning_summary_text.delta` / `response.reasoning_text.delta` | 拼接可读 delta | `response.reasoning` |
| `response.output_item.done` | 收集完整 `item` | `response.output_items[]` |
| `response.completed/incomplete/failed` | 保存其中的 `response` | `response.final_response` |

`response.function_call_arguments.delta/done` 目前没有单独落成 sidecar；正常完成时不会丢失，因为完整 `arguments` 会再次出现在 `response.output_item.done.item` 中。若流在 item done 之前异常中断，只保留 `response_raw` 才能事后恢复已收到的参数 delta。

### 3.5 为什么要同时保存重建结果和 `response_raw`

两份真实样本都验证了一个容易误读的现象：

```text
原始 response.created.response.output     = []
原始 response.in_progress.response.output = []
原始 response.completed.response.output   = []
```

完整输出实际分散在多个 `response.output_item.done` 中。代理先收集 `output_items`，再在 `final_response.output` 为空时回填：

```text
普通样本 output_items = [reasoning, message]
工具样本 output_items = [reasoning, function_call]

重建后 final_response.output = output_items
```

所以捕获文件里的 `response.final_response.output` 非空是**代理重建结果**，不能反推原始 `response.completed` 也携带了完整 output。`response_raw` 必须保留，保证重建器漏识别新事件、流提前断开或未来协议新增字段时仍能重放原始 wire。

## 4. SSE 解析后落盘的字段

每次 `/responses` 调用写成一行，按日期追加到：

```text
$TRAJ_DATA_DIR/codex-api-calls/YYYY-MM-DD.jsonl.gz
```

落盘记录不是 ChatGPT 原样返回的单个 response，而是“请求 + 原始 SSE + 代理重建视图”的组合：

```json
{
  "ts": 1780000000.0,
  "path": "/backend-api/codex/responses",
  "status": 200,
  "had_auth": true,
  "request": {
    "model": "...",
    "instructions": "...",
    "input": [],
    "tools": [],
    "reasoning": {}
  },
  "response": {
    "final_response": {
      "id": "resp_...",
      "status": "completed",
      "model": "...",
      "output": [
        {"type": "reasoning", "encrypted_content": "..."},
        {"type": "message", "role": "assistant", "content": []}
      ],
      "usage": {}
    },
    "text": "拼接后的 assistant 正文；工具调用轮可能为 null",
    "reasoning": null,
    "output_items": [
      {"type": "reasoning"},
      {"type": "message 或 function_call"}
    ],
    "events_seen": 123
  },
  "response_raw": "event: ...\ndata: ..."
}
```

#### 捕获记录顶层字段

| 字段 | 来源 | 含义 |
|---|---|---|
| `ts` | 代理本机 | 完成该 API call 捕获时的 Unix timestamp。 |
| `path` | HTTP request | 实际路径，例如 `/backend-api/codex/responses`。 |
| `status` | HTTP response | 上游 HTTP 状态码。 |
| `had_auth` | HTTP request header 检查 | 是否收到 Authorization header；只保存布尔值，不保存 token。 |
| `request` | Codex request body | 完整 JSON 请求，包括 model、instructions/input、tools、reasoning 等。 |
| `response` | 代理解析结果 | SSE 的结构化重建视图，子字段见下表。 |
| `response_raw` | ChatGPT response body | 完整 SSE 原文，保留事件边界、delta、sequence number 和代理未识别字段。 |

#### `response` 重建字段

| 字段 | 类型 | 生成方式 | 注意事项 |
|---|---|---|---|
| `final_response` | object / null | 取最后一个 `response.completed/incomplete/failed` 的 `response` 对象 | 若其 `output` 为空，代理用收集到的 `output_items` 回填，因此它不是严格原样 wire 对象。 |
| `text` | string / null | 顺序拼接所有 `response.output_text.delta.delta` | 工具调用轮没有 assistant 正文时为 `null`。 |
| `reasoning` | string / null | 拼接可读 reasoning summary/text delta | 仅有 `encrypted_content` 时为 `null`，不能据此判断模型没有 reasoning。 |
| `output_items` | array / null | 收集所有 `response.output_item.done.item` | 保存完整 reasoning、message、function call 等结构化输出。 |
| `events_seen` | integer | 成功 JSON 解析的 `data:` 事件数 | 计数包括代理未专门处理、但成功解析的事件。 |

#### `final_response.output` 与 `output_items` 的关系

两者在当前 ChatGPT 捕获中通常内容相同，但来源不同：

```mermaid
flowchart LR
    RAW["response_raw SSE"] --> DONE["逐个 output_item.done.item"]
    DONE --> ITEMS["response.output_items[]"]
    RAW --> TERM["response.completed.response<br/>原始 output=[]"]
    TERM --> FINAL["response.final_response"]
    ITEMS -->|"final_response.output 为空时回填"| FINAL
```

- 需要忠实审计 ChatGPT 原始返回：读取 `response_raw`；
- 需要方便归一化本轮完成输出：读取 `response.final_response.output`；
- 需要确认每个 item 来自哪个 done 事件：结合 `response_raw` 的 `output_index/sequence_number` 与 `response.output_items`。

#### 普通文本轮与工具调用轮的落盘差异

| 字段 | 普通文本样本 | Lite 工具调用样本 |
|---|---|---|
| `response.text` | assistant 正文 | `null` |
| `response.reasoning` | `null`（仅密文） | `null`（仅密文） |
| `response.output_items[0]` | encrypted reasoning | encrypted reasoning |
| `response.output_items[1]` | assistant message | `function_call(name=wait)` |
| 原始 completed `output` | `[]` | `[]` |
| 重建 final response `output` | `[reasoning, message]` | `[reasoning, function_call]` |

这里的粒度是 **per API call**：一个 `.jsonl.gz` 文件包含多次调用，每一行才是一条独立轨迹候选。文件边界只是日期/存储边界，不是会话边界。

## 5. 轨迹归一化具体做了哪些事

`codex_wire_adapter.py` 的目标不是简单改字段名，而是把 Responses item 流变成统一、可切分、可审计的 trajectory record。

```mermaid
flowchart TD
    R["一条 wire call"] --> Q["读取 request 与 final_response"]
    Q --> S["instructions → system/base_instructions"]
    S --> I["request.input → 历史 messages"]
    I --> B["记录 completion_start_index"]
    B --> O["final_response.output → 本轮 completion"]
    O --> X["剥离加密 reasoning<br/>累计 n_reasoning_stripped"]
    X --> E["提取 model / usage / sampling / tools"]
    E --> F["写入逐字段 fidelity"]
    F --> P{"输出模式"}
    P -- "--out" --> RAW["原始 JSONL<br/>未脱敏"]
    P -- "--publish" --> SEC["全量密钥 harvest + scrub"]
    SEC --> PII["可选稳定 PII 伪匿名化"]
    PII --> PUB["发布 JSONL<br/>chmod 600"]
```

### 5.1 建立单次调用的真实训练边界

归一化消息按以下顺序构造：

```text
messages = [system instructions] + request.input + final_response.output
```

在追加 output 之前记录：

```text
completion_start_index = len(messages)
```

于是：

```text
messages[:completion_start_index]   = 本次请求的完整输入上下文
messages[completion_start_index:]   = 本次模型产生的 completion / target
```

这是 wire call 的真实边界，不需要从时间间隔、UI turn 或 session 日志猜测。对 SFT/BC 而言，这是适配器最重要的产物。

### 5.2 把 Responses item 映射为统一 message

| Responses item | 归一化结果 | 关键处理 |
|---|---|---|
| 顶层 `instructions` | `role=system, subtype=base_instructions` | 只适用于标准 Responses；保留 Codex 官方 system 全文 |
| `type=message` | 原角色 message | `content` 数组抽取并拼接纯文本；包括 `role=developer`，但目前不识别其语义子类型 |
| `type=function_call` | `role=assistant, tool_calls[]` | 保留 `name`、`call_id`；JSON 字符串 arguments 尽量解析为对象 |
| `type=function_call_output` | `role=tool` | 用 `call_id` 与调用对齐，保留工具输出文本 |
| `type=reasoning` | 不进入 messages | 加密/不可读，计入 `n_reasoning_stripped` |
| 未知 item type | `role=system, subtype=<type>` | 不静默丢弃类型，但只抽取 content；对含其他结构字段的类型仍可能丢数据 |

文本抽取兼容三种常见形态：纯字符串、带 `text` 的 content part，以及 `input_text`/`output_text` part。

### 5.3 保留工具定义，而不只保留工具调用

`request.tools` 被整体放入顶层 `tools`。这很关键：只看 transcript 往往只能看到“调用了什么”，而 wire request 还能回答：

- 模型当时可见哪些工具；
- 每个工具的描述和 JSON schema 是什么；
- 一次调用是否受工具集合变化影响。

因此可以分析的不只是 action sequence，还包括 **action space**。

### 5.4 补齐运行元数据

适配器输出：

- `model`：优先使用服务端 `final_response.model`，否则回退请求模型；
- `usage`：保留服务端 usage 原结构，包括 input/output、cache、reasoning 等可用分项；
- `sampling.temperature/top_p`：从最终响应回显中取；
- `sampling.reasoning_effort`：从请求的 `reasoning.effort` 取；
- `ts/path/status/source/source_file`：保留来源和调用状态，支持审计及失败样本分析。

### 5.5 显式记录“拿到了什么、没拿到什么”

每条轨迹都带 `fidelity`，避免下游把缺失字段误当成“空值即真实值”：

| 能力 | Codex wire 轨迹 | 原因 |
|---|:---:|---|
| user / assistant 文本 | ✅ | request 与 SSE output 可见 |
| 工具调用 name/args | ✅ | `function_call` 可见 |
| 工具结果 | ✅ | `function_call_output` 可见 |
| system prompt | 条件 ✅ | `request.instructions` 存在时为真 |
| tool schema | 条件 ✅ | `request.tools` 存在时为真 |
| model id | 条件 ✅ | request 或 response 有值时为真 |
| usage token | 条件 ✅ | final response 有 usage 时为真 |
| sampling 参数 | 条件 ✅ | response 回显 temperature/top_p 时为真 |
| per-call boundary | ✅ | 一行就是一次真实 `/responses` call |
| reasoning 明文 | ❌ | reasoning item 是不可读/加密内容 |
| token id / logprob | ❌ | ChatGPT backend 不向该客户端下发 |

这种逐条 fidelity 比写一个全局说明更可靠，因为失败响应、协议升级或缺字段调用可能与正常样本不同。

### 5.6 剥离 reasoning，而不是伪造或错误训练

ChatGPT 的两份当前实测样本中，`type=reasoning` 都是空 summary/content 加 encrypted content。适配器的现有策略是：

1. 不把密文塞进 `messages`；
2. 不尝试把密文当可读 CoT；
3. 不用其他模型生成的解释冒充原始 reasoning；
4. 只累计 `n_reasoning_stripped`，给下游保留“这里原本存在 reasoning item”的事实。

所以这套数据适合学习 Codex 的**可观察行为**：回复、工具选择、参数和工具结果衔接；它不适合宣称提供了原始思维链监督。

这条策略只对“ChatGPT 返回 encrypted-only reasoning”的当前样本成立，不能推广成“所有 provider 的 reasoning 都不可读”。自部署 Qwen 等兼容服务可能在标准 Responses `reasoning.content[].reasoning_text` 中返回明文；当前 adapter 仍会无条件删除，详见不足三。

### 5.7 发布前的脱敏是 fail-loud

适配器提供两种互斥输出：

- `--out`：原始 JSONL，明确标为可能含密钥/PII；
- `--publish`：必须成功导入统一 `secret-scrub` 才允许写文件，否则返回错误并拒绝发布。

发布流程会先跨全量记录 harvest 密钥明文，再递归 scrub 所有字符串；可选 `--redact-pii` 用稳定哈希把 open_id、app_id、内部 IP 替换为同值同代号，从而在保护身份的同时保留跨轮关联。发布文件最终设置为权限 `0600`。

`tools` schema 被排除在字符串 scrub 之外，以避免固定字段名/描述被误伤；这依赖“工具模板本身不含秘密”的前提，若未来工具描述动态注入租户数据，需要重新评估。

### 5.8 当前归一化的三个已确认不足

#### 不足一：没有识别 Responses Lite wire 变体

这个判断是正确的，而且不只是少了一个格式标签，而是会导致 system/tool schema 的实质性误读或丢失。

标准 Responses 的请求形态是：

```json
{
  "instructions": "Codex base instructions...",
  "tools": [{"type": "function", "name": "..."}],
  "input": [
    {"type": "message", "role": "user", "content": []}
  ]
}
```

Responses Lite 则把 instructions 和 tools 都移进 `input`：

```json
{
  "input": [
    {
      "type": "additional_tools",
      "role": "developer",
      "tools": [{"type": "function", "name": "..."}]
    },
    {
      "type": "message",
      "role": "developer",
      "content": [{"type": "input_text", "text": "Codex base instructions..."}]
    },
    {"type": "message", "role": "user", "content": []}
  ]
}
```

同时，Lite 请求还具有这些差异：

- 顶层 `instructions` 为空或被省略；
- 顶层 `tools` 为 `null` 或被省略；
- `reasoning.context="all_turns"`；
- `parallel_tool_calls=false`；
- HTTP 请求通过 `x-openai-internal-codex-responses-lite: true` 标识，WebSocket 则通过 client metadata 标识；
- 输入图片会去掉 detail，远程图片可能被替换为说明文本。

当前 adapter 的具体问题是：

1. 只从顶层 `request.instructions` 建 system message，所以 Lite 的 base instructions 不会被识别为 `subtype=base_instructions`；
2. 只从顶层 `request.tools` 填充归一化 `tools`，所以 Lite 的 `additional_tools.tools` 会丢失；
3. `_item_to_messages()` 不认识 `type=additional_tools`，会把它降为一个内容为空的 `role=system, subtype=additional_tools`，仅保留类型名，没有保留真正重要的 `tools` 数组；
4. `_item_to_messages()` 不认识 `custom_tool_call`，会把它降为空 system message，丢失 `name/input/call_id`；
5. `_item_to_messages()` 不认识 `custom_tool_call_output`，会把它降为空 system message，丢失结构化或字符串 `output` 以及 `call_id`；
6. `fidelity.system_prompt` 和 `fidelity.tools_schema` 会被错误标成 `false`，即使这些信息实际上存在于 Lite input 中；
7. 输出中没有 `wire_variant=responses_lite`，下游无法区分两种协议布局。

`codex_responses_lite_raw_with_tool_call_2026-07-15_redacted.json` 对问题 4、5 给出了真实证据：34 个 input items 中包含 8 个 `custom_tool_call` 和 8 个 `custom_tool_call_output`。当前 adapter 会把这 16 个 item 变成内容为空的 system 杂项，导致历史 exec 工具名、输入、输出以及调用关联全部丢失。本轮新产生的 `function_call(name=wait)` 能被当前 adapter 识别，但它所依赖的历史上下文已经失真。

不同工具定义必须按不同 item 类型归一化：

| tool 定义 | 模型调用 item | 执行结果 item | 归一化要求 |
|---|---|---|---|
| `type=function` | `function_call` | `function_call_output` | 解析 JSON 字符串 `arguments`，保留 `name/call_id`。 |
| `type=custom` | `custom_tool_call` | `custom_tool_call_output` | 保留 freeform `input`，兼容 output 为字符串或 content-item 数组。 |
| `type=namespace` | namespace 下的 function/custom call | 对应 output | 保留 namespace provenance，不能压平后产生同名冲突。 |

注意，当前捕获文件不保存请求 header，所以 adapter 不能依赖 Lite header 做唯一判断。可以按结构识别：

```text
顶层 instructions/tools 缺失
+ input[0].type == "additional_tools"
+ input[0].role == "developer"
```

更完整的改进应当是：

- 增加 `wire_variant: responses | responses_lite | unknown`；
- Lite 模式从 `additional_tools.tools` 恢复顶层归一化 `tools`；
- 将紧随 `additional_tools` 的、由 Codex 注入的 developer base-instructions message 标记为 `subtype=base_instructions`；
- 保留其原始 wire role，例如增加 `original_role="developer"`，避免为了 canonical role 丢 provenance；
- 增加 `custom_tool_call/custom_tool_call_output` 映射，保留 `call_id` 闭环以及 output 的 string/array 原始形态；
- fidelity 按恢复后的语义字段判断，而不是只检查标准 Responses 的顶层键；
- 对 Lite、标准 Responses、function/custom tool 和畸形/混合格式分别增加 adapter 测试。

#### 不足二：`role=developer` 只做了语法保留，没有做语义归一化

这个判断也成立，但需要说明：当前 adapter **并没有把 developer role 改错或直接丢掉**。`_item_to_messages()` 对 `type=message` 使用原始 `role`，所以普通 developer message 会得到：

```json
{"role": "developer", "content": "..."}
```

问题在更深一层：developer message 在 Codex 中不是单一来源，它可能承载：

- Responses Lite 的 base instructions；
- repo/user 配置中的 developer instructions；
- permissions、sandbox、环境上下文；
- skills、plugins、apps、协作模式等动态能力说明；
- 模型切换、时间提醒或其他运行时注入信息。

当前 adapter 将这些全部压平为同一种 developer message，并把多段 content 拼成纯文本。因此会损失：

- developer message 的语义来源和 subtype；
- content part 的原始类型与边界；
- “基础系统指令”和“运行时开发者约束”的区分；
- 下游只接受 `system/user/assistant/tool` 时的明确兼容策略。

但不能简单把所有 developer message 合并进 system：

- 多条 developer message 的相对顺序可能影响模型可见语义；
- 有些是稳定基础指令，有些是每轮动态上下文；
- 合并会让 provenance、缓存分析和配置版本分析变得困难；
- Lite 的 base instructions 虽然 wire role 是 developer，语义上却对应标准 Responses 的顶层 instructions。

建议采用“双层表示”而不是粗暴改 role：

```json
{
  "role": "system",
  "original_role": "developer",
  "subtype": "base_instructions",
  "content": "...",
  "source": "responses_lite_input"
}
```

对于普通 developer instructions，则可以保留：

```json
{
  "role": "developer",
  "original_role": "developer",
  "subtype": "runtime_developer_instructions",
  "content": "..."
}
```

如果目标训练格式原生支持 developer role，应保持 `role=developer`；如果目标格式只支持 system/user/assistant/tool，再在最终导出层做有记录的映射，而不是在 wire adapter 层无痕合并。无法可靠判断 subtype 时应标为 `developer_unknown`，不能仅凭文本猜成 base instructions。

因此，前两个不足实际上互相关联：**Responses Lite 把原本位于顶层的 system/tools 搬进了 developer input，而当前 adapter 既不识别 Lite，也没有足够细的 developer 语义模型。**

#### 不足三：自定义模型的明文 reasoning 会被无条件删除

当前 `_item_to_messages()` 没有检查 reasoning item 的实际内容，而是对所有 `type=reasoning` 执行同一逻辑：

```python
if t == "reasoning":
    stripped_counter[0] += 1
    return []
```

这对 ChatGPT 当前 encrypted-only 样本是合理的安全默认值，但接入自部署 Qwen、gpt-oss 或其他兼容模型后可能造成真实数据丢失。例如服务端可以按标准 Responses 结构返回：

```json
{
  "id": "rs_...",
  "type": "reasoning",
  "content": [
    {
      "type": "reasoning_text",
      "text": "先检查仓库结构，再定位失败测试。"
    }
  ],
  "summary": [],
  "encrypted_content": null
}
```

当前 adapter 会把这段可读文本和密文 item 一样整项删除，并错误增加 `n_reasoning_stripped`。此外，捕获代理已经会将 `response.reasoning_text.delta` 和 `response.reasoning_summary_text.delta` 拼入 `response.reasoning` sidecar，但 `adapt_call()` 当前完全不读取该字段，所以即使 SSE 中出现明文也无法进入归一化结果。

需要区分三种不同语义，不能都叫“reasoning 文本”：

| wire 来源 | 实际语义 | 建议处理 |
|---|---|---|
| `reasoning.content[].type=reasoning_text` | 模型公开的原始/明文 reasoning | 保留，标记 `reasoning_kind=raw`。 |
| `reasoning.summary[].type=summary_text` | 服务端生成的可读推理摘要，不等于原始 CoT | 保留，标记 `reasoning_kind=summary`。 |
| `reasoning.encrypted_content` | 客户端不可读的续推状态 | 不写入训练 message；只记录存在性、数量和可选字节长度。 |
| `response.reasoning` sidecar | 代理从 reasoning text/summary delta 拼出的便利视图 | 仅作结构化 item 缺失时的兜底；当前 sidecar 混合两类 delta，须标记 `reasoning_kind=unknown_or_mixed`。 |

推荐按以下优先级解析，避免重复和语义误标：

```text
1. 优先读取 final_response.output[].reasoning.content
2. 再读取同一 item 的 reasoning.summary
3. 只有结构化 item 没有明文时，才使用 response.reasoning sidecar
4. 只有 encrypted_content 时才执行 stripped 计数
5. item 与 sidecar 文本相同或重叠时去重，不能重复生成两条 reasoning
```

建议的归一化表示为：

```json
{
  "role": "assistant",
  "subtype": "reasoning",
  "reasoning_kind": "raw",
  "content": "先检查仓库结构，再定位失败测试。",
  "source": "response_item.content",
  "item_id": "rs_..."
}
```

摘要必须单独标记：

```json
{
  "role": "assistant",
  "subtype": "reasoning",
  "reasoning_kind": "summary",
  "content": "模型先检查结构，然后定位失败测试。",
  "source": "response_item.summary",
  "item_id": "rs_..."
}
```

reasoning message 应保持其在 output items 中的原始顺序，尤其要保留它位于 tool call/message 之前还是之间的边界。`fidelity.reasoning_text` 只能在拿到 `reasoning_kind=raw` 时设为 `true`；只有 summary 时应新增或使用独立的 `reasoning_summary` fidelity，不能把摘要冒充原始 reasoning。

还需注意非标准兼容形态：某些服务可能返回 Chat Completions 风格 `reasoning_content`，或把思考包在 assistant 文本的 `<think>...</think>` 中。Codex 当前走 Responses wire，优先要求服务端兼容层映射成标准 reasoning item；若 adapter 增加兜底解析，必须记录 `source` 和原始格式，不能无痕改写。

### 5.9 字段变化图：从 wire record 到 normalized trajectory

下面按“保留、重组、派生、剥离”四类标出字段变化。这里的 raw 侧表示 `codex_capture_proxy.py` 的标准 Responses record；Responses Lite 的例外见上一节。

```mermaid
flowchart LR
    subgraph RAW["原始 wire record"]
        RM["ts / path / status"]
        RMO["request.model"]
        RI["request.instructions"]
        RIN["request.input[]"]
        RT["request.tools[]"]
        RE["request.reasoning.effort"]
        RO["response.final_response.output[]"]
        RU["response.final_response.usage"]
        RS["response temperature / top_p"]
        RR["type=reasoning<br/>encrypted_content"]
        RRAW["response_raw SSE"]
    end

    subgraph OP["归一化操作"]
        K["① 保留 / 改路径"]
        M["② 重组为 messages"]
        B["③ 计算补全边界"]
        D["④ 派生审计字段"]
        X["⑤ 剥离不可训练内容"]
    end

    subgraph NORM["归一化 trajectory"]
        NM["ts / path / status / model"]
        NSYS["messages[]<br/>system: base_instructions"]
        NCTX["messages[]<br/>input context"]
        NOUT["messages[]<br/>current completion"]
        NT["tools[]"]
        NU["usage"]
        NS["sampling<br/>temperature / top_p / reasoning_effort"]
        NB["completion_start_index"]
        NF["source / source_file / fidelity"]
        NX["n_reasoning_stripped"]
    end

    RM --> K --> NM
    RMO --> K
    RI --> M --> NSYS
    RIN --> M --> NCTX
    RO --> M --> NOUT
    RT --> K --> NT
    RU --> K --> NU
    RE --> D --> NS
    RS --> D
    NCTX --> B --> NB
    NOUT --> B
    RM --> D --> NF
    RR --> X --> NX
    RRAW -. "只留在 raw capture<br/>不进入 normalized JSONL" .-> X

    classDef kept fill:#dbeafe,stroke:#2563eb,color:#172554;
    classDef reshape fill:#fef3c7,stroke:#d97706,color:#451a03;
    classDef derived fill:#dcfce7,stroke:#16a34a,color:#052e16;
    classDef stripped fill:#fee2e2,stroke:#dc2626,color:#450a0a;
    class K,NM,NT,NU kept;
    class M,NSYS,NCTX,NOUT reshape;
    class B,D,NS,NB,NF derived;
    class X,NX,RR,RRAW stripped;
```

图例：

- 蓝色：原值保留，只调整所在路径或字段组织；
- 黄色：把 Responses item 重组为统一 message；
- 绿色：adapter 新计算或补充的字段；
- 红色：因不可读、不可训练或体积原因不进入 normalized JSONL，但用计数保留事实。

#### 用 `work/1.json` 看一次真实归一化结果

`work/1.json` 本身已经是归一化后的记录，不包含对应的 raw record；所以下图右侧数字直接读取该文件，左侧结构依据 adapter 的确定性映射回标，不冒充原始抓包统计。

```mermaid
flowchart LR
    subgraph BEFORE["标准 Responses 逻辑输入"]
        BI["request.instructions"]
        BIN["request.input[]<br/>历史 message / function call / tool result / reasoning"]
        BO["final_response.output[]<br/>本轮模型输出"]
        BT["request.tools[]"]
        BM["model / usage / sampling"]
    end

    subgraph CHANGE["adapter 变化"]
        C1["instructions 前置为 system"]
        C2["不同 item 统一为 role messages"]
        C3["在追加 output 前记 index"]
        C4["reasoning 密文剥离并计数"]
        C5["工具和运行元数据提到顶层"]
    end

    subgraph AFTER["work/1.json"]
        A0["messages[0]<br/>system × 1"]
        A1["messages[1:76]：输入上下文 × 75<br/>developer 1 · user 2<br/>assistant 41 · tool 31"]
        A2["completion_start_index = 76"]
        A3["messages[76]：本轮 completion × 1<br/>assistant.tool_calls → exec_command"]
        A4["tools = 12 个 schema"]
        A5["n_reasoning_stripped = 10"]
        A6["usage：77,163 input / 87 output<br/>71,040 cached"]
        A7["sampling：temperature 1.0<br/>top_p 0.98 · effort medium"]
        A8["fidelity：10 项可见能力为 true<br/>reasoning/logprob 3 项为 false"]
    end

    BI --> C1 --> A0
    BIN --> C2 --> A1
    BO --> C2 --> A3
    A1 --> C3 --> A2
    A3 --> C3
    BIN --> C4 --> A5
    BO --> C4
    BT --> C5 --> A4
    BM --> C5 --> A6
    C5 --> A7
    C5 --> A8

    classDef source fill:#e0e7ff,stroke:#4f46e5,color:#1e1b4b;
    classDef transform fill:#fef3c7,stroke:#d97706,color:#451a03;
    classDef result fill:#dcfce7,stroke:#16a34a,color:#052e16;
    classDef loss fill:#fee2e2,stroke:#dc2626,color:#450a0a;
    class BI,BIN,BO,BT,BM source;
    class C1,C2,C3,C5 transform;
    class C4,A5 loss;
    class A0,A1,A2,A3,A4,A6,A7,A8 result;
```

这个样例最直观地展示了 `completion_start_index` 的作用：77 条归一化 message 中，前 76 条是本次调用的 context，最后 1 条才是训练 target。它也说明“剥离 reasoning”不是静默删除：10 个不可读 reasoning item 没进入 messages，但数量保留在 `n_reasoning_stripped` 中。

## 6. 归一化后的输出 schema

```json
{
  "source": "codex-wire",
  "source_file": "2026-07-15.jsonl.gz",
  "ts": 1780000000.0,
  "path": "/backend-api/codex/responses",
  "status": 200,
  "model": "...",
  "messages": [],
  "completion_start_index": 4,
  "tools": [],
  "usage": {},
  "sampling": {
    "temperature": 1.0,
    "top_p": 0.98,
    "reasoning_effort": "medium"
  },
  "n_reasoning_stripped": 1,
  "fidelity": {}
}
```

可直接导出：

```bash
python3 trajectory_extract/codex_wire_adapter.py \
  --input ~/trajectory-data/codex-api-calls \
  --out codex-raw.jsonl
```

发布脱敏版本：

```bash
python3 trajectory_extract/codex_wire_adapter.py \
  --input ~/trajectory-data/codex-api-calls \
  --publish codex-publish.jsonl \
  --redact-pii
```

## 7. 这套轨迹能回答哪些洞察问题

### Agent 行为

- 面对什么上下文时，Codex选择文本回答还是工具调用？
- 多个工具都可用时，它如何选择工具和组织参数？
- 工具失败或输出异常后，下一步是重试、换工具、修正参数还是向用户解释？
- 同一任务中，一次模型调用携带了多少历史上下文，completion 从哪里开始？

### Prompt 与工具系统

- system instructions 的版本变化是否改变行为？
- tool schema 的描述、参数约束变化是否改变调用正确率？
- 工具集合变大后，误选率是否上升？

### 成本与效率

- input/output/cached/reasoning token 的结构如何？
- 哪些任务产生过长上下文或重复输入？
- 哪些调用只有很短 completion，却携带很大上下文？

### 数据质量

- 哪些样本缺 system、tools、usage 或采样参数？
- 哪些样本包含被剥离的 reasoning item？
- 失败状态的调用是否应进入训练集，还是单独用于故障分析？

## 8. 保真度与诚实边界

### 能保证的

- 每条记录对应一次真实 HTTP `/responses` 调用；
- request payload 包含模型实际可见的 instructions、input 和 tools；
- response 同时保留尽力重建结果与原始 SSE；
- completion 边界来自同一次 wire call，不是启发式切分；
- OAuth token 不落盘，发布模式对密钥脱敏失败会拒绝输出。

### 不能保证的

- 不是 token 级 rollout：没有 token id 和 logprob；
- 对当前 ChatGPT encrypted-only 样本，没有可读原始 reasoning，不能做真实 CoT 监督；自定义 provider 即使返回明文，当前 adapter 也会误删，修复不足三后才能保留；
- “逐字节”应理解为代理读取到的 HTTP 逻辑 body/SSE 字节。请求 zstd 已由 aiohttp 解码，不是保存原始压缩包；
- WS 改为 HTTP SSE 可能改变传输时延与流式平滑度，尽管逻辑 payload 和上游模型不变；
- 代理是新的在线依赖：代理不可用时，该模型请求会失败并返回 502；
- 当前代理先把整段响应缓存在内存中再落盘，大响应会增加进程内存占用。

## 9. 关于 `chatgpt_base_url` 的版本说明

`codex_capture_proxy.py` 文件头写的是“ChatGPT 登录态通过 `chatgpt_base_url` 改道”。这描述了当时部署/SDK 接入的实测方式，但当前纳入分析的 `openai-codex` 源码显示：

1. `Config.chatgpt_base_url` 的默认值确实是 `https://chatgpt.com/backend-api/`；
2. 模型请求转为 `ApiProvider` 时，若 `ModelProviderInfo.base_url` 已显式设置，会优先采用 provider `base_url`；
3. 只有 provider 未设置 `base_url` 时，才根据 ChatGPT auth mode 使用内置 ChatGPT Codex URL。

因此，Wiki 采用“**自定义 `proxycap` provider 的显式 `base_url` 是模型 `/responses` 改道点**”这一当前源码可验证的说法。若生产 bridge 仍把 `CODEX_CAPTURE_PROXY_URL` 映射为 `chatgpt_base_url`，应在部署仓补充那段映射代码作为证据；它不在本仓库中，不能仅凭环境变量名推断。

## 10. 代码证据索引

| 结论 | 代码位置 |
|---|---|
| ChatGPT/API auth mode 决定默认 base URL，显式 provider base URL 优先 | `openai-codex/codex-rs/model-provider-info/src/lib.rs:241-259` |
| 默认 OpenAI provider 支持 Responses WebSocket | `openai-codex/codex-rs/model-provider-info/src/lib.rs:329-363` |
| `supports_websockets=false` 关闭 WS | `openai-codex/codex-rs/core/src/client.rs:926-937` |
| Codex 优先 WS、否则 HTTP Responses | `openai-codex/codex-rs/core/src/client.rs:1768-1824` |
| HTTP 分支构建并发送标准 Responses request | `openai-codex/codex-rs/core/src/client.rs:1378-1460` |
| HTTP `ResponsesApiRequest` 的 15 个顶层候选字段 | `openai-codex/codex-rs/codex-api/src/common.rs:215-239` |
| 顶层字段的实际赋值、省略条件和固定值 | `openai-codex/codex-rs/core/src/client.rs:824-907` |
| `input` 支持的 `ResponseItem` 类型及字段 | `openai-codex/codex-rs/protocol/src/models.rs:933-1164` |
| `message.content` 支持 input_text/input_image/output_text | `openai-codex/codex-rs/protocol/src/models.rs:843-858` |
| Responses Lite 把 tools/base instructions 移入 developer input | `openai-codex/codex-rs/core/src/client.rs:834-863` |
| Responses Lite 去除图片 detail | `openai-codex/codex-rs/core/src/client_common.rs:51-101` |
| Codex 内部 Responses SSE 事件类型 | `openai-codex/codex-rs/codex-api/src/common.rs:73-121` |
| Codex 对 output/reasoning/function-call SSE 的映射 | `openai-codex/codex-rs/codex-api/src/sse/responses.rs:326-469` |
| 代理 SSE 重建 | `trajectory-collect/scripts/codex_capture_proxy.py:67-120` |
| 代理透传、流式回包和旁路 buffer | `trajectory-collect/scripts/codex_capture_proxy.py:164-204` |
| Responses item → message 映射 | `trajectory_extract/codex_wire_adapter.py:130-188` |
| 当前 adapter 对所有 reasoning 无条件删除 | `trajectory_extract/codex_wire_adapter.py:158-166` |
| 当前 adapter 未实现 custom tool，未知类型降为空 system 杂项 | `trajectory_extract/codex_wire_adapter.py:158-188` |
| 当前 adapter 只从顶层 instructions/tools 归一化 system/tool schema | `trajectory_extract/codex_wire_adapter.py:191-255` |
| context/completion 切分及输出 schema | `trajectory_extract/codex_wire_adapter.py:191-255` |
| 发布脱敏与权限控制 | `trajectory_extract/codex_wire_adapter.py:279-361` |

## 11. 一句话评价

Codex 这条采集链路的优势是：**离真实模型 wire 很近、每次调用边界准确、system/tools/usage 都较完整，而且原生 Responses 格式离统一 IR 最近**。它的硬边界也同样明确：**ChatGPT 当前只暴露 encrypted reasoning，拿不到可读 CoT 和 token 级训练信号；自定义模型可以暴露明文 reasoning，但必须先修复 adapter 的无条件剥离逻辑。**
