# antigravity 轨迹离线归集为 SFT 训练样本

> 负责人：张耀明（数据）· 实现：lian-server（claude）
> 起于 2026-06-23 群内讨论，本文档随进展更新。

---

## 一、初衷：为什么做这件事

我们已经能把 **claude / codex** 的真实工作轨迹采下来做训练数据——靠 API 劫持：
设 `ANTHROPIC_BASE_URL=127.0.0.1:4319`，让本地代理把每次 `/v1/messages` 调用的
**完整请求 + 重建的响应**逐条落盘（`api-calls/*.jsonl.gz`），每条就是一个天然闭合的
`(prompt, completion)` 训练样本。

**antigravity（Google 的 agent IDE / 飞书接入的 bot）也想纳入同一套训练数据**，
但它够不到上面那套劫持：

- antigravity 的模型出口是 Google `language_server` 闭源 Go 二进制，走 gRPC/TLS
  **直连**，**没有可重定向的 base_url**——`127.0.0.1:4319` 代理那条路物理打不通。
- Go 静态链接 `crypto/tls`，基本无视系统证书库/系统代理且大概率证书 pinning，
  **TLS-MITM 代理也拦不到**。

所以目标变成：**在不碰网络层的前提下，离线把 antigravity 的轨迹重建成与 claude
劫持同等保真度的 SFT 闭合样本。**

---

## 二、方法：离线重建 + 脱敏发布

### 2.1 数据来源
antigravity 把每次会话的事件流落在本地：
```
~/.gemini/antigravity/brain/<conv_id>/.system_generated/logs/transcript.jsonl
```
这是**事后事件日志**（USER_INPUT / PLANNER_RESPONSE / TOOL_RESULT / CONVERSATION_HISTORY…），
不是「每次 API call 的真实请求体」。adapter 读它、归一化成训练记录。

### 2.2 三个 🟡 字段是怎么来的（核心：不是问 antigravity 要的官方原件）
| 字段 | 来源方式 | 诚实边界 |
|---|---|---|
| **system prompt** | 飞书 bridge 在首条 USER_INPUT 里把系统指令内联成 `<USER_REQUEST>【系统指令】…【用户消息】<真实消息>`，adapter 切 `【系统指令】`→`【用户消息】` 之间为 system | 是 **bridge 注入的那段**，非 antigravity 从 Google 收到的官方底层 system（在闭源 language_server 里，离线拿不到）。对飞书轨迹=模型当时实际看到的 system 前缀，SFT 对。客户端原生对话**无此包裹**，故格式不一定通用 |
| **tools schema** | transcript 只记「按名调用+参数」，无 schema。从 5000+ 真实 tool_call 统计 15 个固定内置工具名 + 各自参数键并集，手写成 OpenAI tools 形态存 `antigravity_tools_schema.json` | **工具名/参数键实证**，但 `description/required/type` 是反推补的，**非官方 schema**。SFT 够用 |
| **model_id** | `--model gemini-3-pro` 推定 | transcript 不记 model |

### 2.3 保真度对标 claude 劫持（逐字段核过真实数据，非脑补）
| 字段 | claude 劫持 | antigravity 离线 | 结论 |
|---|---|---|---|
| messages / thinking / tool_call / tool_result | ✅ | ✅ | 闭合 |
| system / tools / model | ✅ | 🟡（见 2.2） | 可补 |
| **sampling**(temperature/top_p…) | ❌ **claude 自己也没有** | ❌ | **不是差异项** |
| **usage**(token 计数) | ✅ 很全 | ❌ | 唯一硬缺口 |
| token_id / logprob | ❌ | ❌ | 两边都没有 |

**关键结论**：相对 claude 劫持，离线真正缺的只有 `usage` token 计数，而它**只有 eBPF
wire 抓包（`ecapture gotls`）能补**。→ **张耀明拍板 SFT 不需要 usage → 不碰 eBPF。**

### 2.4 「闭合」的精确程度（诚实边界）
- **SFT 级闭合 = 能拿**：每个模型回合都有自洽输入（system+tools+前序 messages）+ 完整输出。
- **wire 级逐字节闭合（off-policy 精确回放）= 拿不到**：transcript 是事后事件流，且
  antigravity 内部做上下文压缩（CONVERSATION_HISTORY 即证据），重建 messages 前缀
  ≈ 而非 == 真实输入。每条记录 `fidelity.per_call_boundary=False` 已标注。
  **张耀明用途是 SFT，所以是「能」。**

### 2.5 脱敏发布（必做，不可跳过）
raw 轨迹里混着 tool 输出当场捕获的**真实密钥**（实扫到真 LLM key、Bearer×11、JWT×7、
DB 明文口令 `postgres123`/`ada123`）+ 大量 PII。**直接发共享区或喂训练 = 扩散秘密，
SFT 模型会复述。** 发布路径强制：
1. **两遍扫**：第一遍跨全量 `harvest_secrets` 收密钥明文值（仅内存）；第二遍精确全局替换，
   连散文里反引号裸引用的口令都清；
2. `--redact-pii`：open_id/app_id/IP 伪匿名化（同值→同稳定短码，保留多轮身份信号）；
3. 防误杀：harvest 严格门槛只收不透明密钥形，排除路径/代码标识符；
4. fail-loud：脱敏器导入失败直接拒绝发布。

---

## 三、已完成的工作（push 正本 main）

| commit | 内容 |
|---|---|
| `c6d0c14` | adapter 补到 claude 平价：切 system prompt、挂 15 内置工具 schema、补 model_id、修 `READ_URL_CONTENT` 错分 bug |
| `ef92c3e` | 脱敏发布路径 `--publish`（强制两遍扫 secret-scrub）+ `--redact-pii`；扩 scrub.py 加 `scrub_text`/`harvest_secrets`/口令模式；补 `test_scrub_text.py` 5/5 |

**代码位置**
- adapter：`feishu-plugin/skills/trajectory-collect/scripts/antigravity_transcript_adapter.py`
- 工具 schema：同目录 `antigravity_tools_schema.json`
- 脱敏器：`llm_label_tool/skills/secret-scrub/scripts/scrub.py`

**已实跑验证**
- lian-antigravity（德国机飞书实例）：218 会话全量 `--publish` 跑通，重扫密钥清零、行行合法 JSON。
- 张耀明德国机飞书实例：13 会话已脱敏发布到 `/opt/shared/data/zym-antigravity-sft.jsonl`（chmod 644，团队可读）。

**标准跑法**
```bash
python3 antigravity_transcript_adapter.py \
  --brain <brain目录> --model gemini-3-pro \
  --redact-pii --publish /opt/shared/data/<目标>.jsonl
```

---

## 四、后续待办

1. **【主线】归集张耀明客户端全量历史**
   - 现象：德国机 brain 只有 13 条且全在今天 07:42~08:49——因为**德国机实例今天才接入飞书**。
   - 真相：张耀明之前都在 **antigravity 客户端（本地 Windows 电脑）** 对话，主力历史在
     `%USERPROFILE%\.gemini\antigravity\brain`，不在德国机。
   - **下一步**：Windows PowerShell 清点客户端会话数/行数 → 贴回群 → 经香港跳板 scp 一个
     样本到德国机验格式 → 全量搬运 + 跑。

2. **【风险点·必验】客户端原生格式 ≠ 飞书 bridge 格式**
   - adapter 的 system prompt 切割依赖飞书 bridge 特有的 `<USER_REQUEST>【系统指令】…【用户消息】…` 包裹；
   - 客户端 IDE 直接交互**无此包裹**，system 承载方式可能不同，且可能有 `messages/` 形态的
     第二种存储（已在会话 `a5519929` 见到）。
   - **必须先拿一条客户端真实 transcript 验格式**，再决定要不要给客户端格式补适配分支。
     主体（messages/tool/thinking）大概率通用，但 system 切割不敢打包票。

3. **【沉淀】把「德国机一键脱敏发布」写进 SKILL.md** 当标准流程（张耀明确认后做）。

4.（可选）若将来需要 `usage` token 计数做 reward/计费对齐，才上 `ecapture gotls` eBPF
   抓包——当前 SFT 用途不需要，已否决。

---

## 五、一句话现状
**方法已跑通、脱敏已焊死、德国机飞书那 13 条已交付；真正的数据量在张耀明 Windows
客户端，卡在「先验客户端格式、再全量搬运」这一步。**
