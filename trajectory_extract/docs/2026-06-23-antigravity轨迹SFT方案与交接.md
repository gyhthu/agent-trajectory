# antigravity 轨迹归集为 SFT 训练数据 —— 方案与交接文档

> 版本：2026-06-23　起草：lian-server(claude)　数据负责人：张耀明

## 0. 一句话
把 antigravity（Google 的 agent IDE / 飞书接入 bot）的工作轨迹，**离线**重建成与 claude API 劫持同等保真度的 SFT 训练样本，并强制脱敏发布。**核心代码已就绪、纯 Python 标准库**，待完成三件事：①客户端格式适配 ②工具化（飞书发文件即处理）③可选的持续采集。

## 1. 初衷
- claude / codex 的训练数据靠 **API 劫持**：设 `ANTHROPIC_BASE_URL=127.0.0.1:4319`，本地代理把每次 `/v1/messages` 的**完整请求+重建响应**逐条落盘，每条即一个天然闭合的 `(prompt, completion)` 样本。
- **antigravity 够不到这套**：模型出口是闭源 `language_server`（Go 二进制），走 gRPC/TLS 直连，**无可重定向 base_url**；Go 静态 TLS + 证书 pinning，**MITM 代理也拦不到**。
- 故目标改为：**不碰网络层，离线把 antigravity 轨迹重建成同保真 SFT 样本。**

## 2. 方法
### 2.1 数据源
```
~/.gemini/antigravity/brain/<conv_id>/.system_generated/logs/transcript.jsonl
```
是**事后事件日志**（USER_INPUT / PLANNER_RESPONSE / TOOL_RESULT / CONVERSATION_HISTORY…），不是每次 API call 的真实请求体。adapter 读它、归一化成训练记录。

### 2.2 三个补齐字段的来源（诚实，非官方原件）
| 字段 | 来源 | 边界 |
|---|---|---|
| **system** | 飞书 bridge 在首条 USER_INPUT 注入 `<USER_REQUEST>【系统指令】…【用户消息】<真实消息>`，adapter 切 `【系统指令】`→`【用户消息】` | **仅飞书 bridge 格式有**；**客户端原生格式无此包裹 → 见待做任务 A** |
| **tools** | 从 5000+ 真实 tool_call 反推 15 个固定内置工具的名+参数键，手写成 OpenAI tools 形态存 `antigravity_tools_schema.json` | 工具名/参数键实证，`description/required/type` 反推，非官方 schema |
| **model** | `--model gemini-3-pro` 推定 | transcript 不记 model |

### 2.3 保真度对标 claude（逐字段实测，非脑补）
| 字段 | claude 劫持 | antigravity 离线 |
|---|---|---|
| messages/thinking/tool_call/tool_result | ✅ | ✅ |
| system/tools/model | ✅ | 🟡（见 2.2） |
| sampling(temperature/top_p…) | ❌ claude 自己也没有 | ❌（非差异项） |
| usage(token 计数) | ✅ | ❌ 唯一硬缺口，只 eBPF(`ecapture gotls`) 能补 → **SFT 不需要，已否决** |
| token_id/logprob | ❌ | ❌ |

### 2.4 闭合边界（诚实）
- **SFT 级闭合 = 能拿**：每个模型回合有自洽输入（system+tools+前序 messages）+ 完整输出。
- **wire 级逐字节闭合 = 拿不到**：transcript 是事后流，且 antigravity 内部做上下文压缩，重建 messages 前缀 **≈ 而非 ==** 真实输入，每条 `fidelity.per_call_boundary=False` 已标注。SFT 用途不受影响。

### 2.5 脱敏（必做，不可跳过）
raw 轨迹混着 tool 输出当场捕获的**真实密钥**（实扫到 LLM key、Bearer、JWT、DB 明文口令）+ PII。**直接发布/喂训练会被 SFT 模型复述**。发布路径强制：
1. **两遍扫**：第一遍跨全量 `harvest_secrets` 收密钥明文值（仅内存）；第二遍精确全局替换（连散文反引号裸引用都清）；
2. `--redact-pii`：open_id/app_id/IP 伪匿名化（同值→同稳定短码，保留多轮身份信号）；
3. 防误杀：harvest 严格门槛只收不透明密钥形，排除路径/代码标识符；
4. **fail-loud**：脱敏器导入失败直接拒绝发布。

## 3. 已完成（代码就绪）
**代码位置（正本仓 gitee `karenliancau/cbu-llm-skills`）**
- adapter：`feishu-plugin/skills/trajectory-collect/scripts/antigravity_transcript_adapter.py`
- 工具 schema：同目录 `antigravity_tools_schema.json`
- 脱敏器：`llm_label_tool/skills/secret-scrub/scripts/scrub.py`（已加 `scrub_text` / `harvest_secrets` / 口令模式）

**commit**（早期独立验证过、可信）
- `c6d0c14` adapter 补到 claude 平价（切 system、挂 tools schema、补 model、修 READ_URL_CONTENT 错分）
- `ef92c3e` `--publish` 脱敏发布路径 + 扩 scrub.py + `test_scrub_text.py` 5/5

**依赖**：纯 Python 标准库（`re/sys/argparse/json/os/glob/collections/hashlib`），**零 pip 依赖**，Windows 装个 Python 3 即可跑。

**已实跑验证**：德国机飞书实例 218 会话、张耀明德国机飞书实例 13 会话，`--publish` 跑通、重扫密钥清零、行行合法 JSON。

**运行命令（标准）**
```bash
python antigravity_transcript_adapter.py \
  --brain <brain目录> --model gemini-3-pro \
  --redact-pii --publish <输出>.jsonl
# 验格式时用 --out（不脱敏）替代 --publish，看终端打印的「切出 system prompt: X/N」
```

## 4. 待做（交接任务）
### 任务 A · 客户端格式适配【核心技术点】
- **背景**：张耀明历史主力数据在他 **Windows 客户端** `%USERPROFILE%\.gemini\antigravity\brain`（不在德国机；德国机实例今天才接入飞书，只有 13 条今天的）。
- **问题**：客户端原生对话**没有飞书 bridge 那层 `【系统指令】` 包裹**，现成 adapter 的 system 切割大概率失效。
- **做法**：
  1. 拿一条客户端 transcript 跑 `--out`（不脱敏），看「切出 system prompt: X/N」；
  2. 若 X≈0：读 adapter `adapt_transcript()` 的 system 切割逻辑，**打开一条客户端 transcript.jsonl 看首条事件真实结构**，定位 system prompt 落在哪个 type（可能是独立 SYSTEM 事件 / 某 config 字段 / 确实没有）；
  3. 给 adapter 补一个「客户端格式」system 切割分支，重跑到能切出；
  4. messages/tool/thinking 主体大概率通用，重点只在 system。
- **产出**：改动 diff 并回正本仓。

### 任务 B · 工具化【张耀明明确要求】
目标：**别每次手动**，做成人人能用的工具。两种形态（可都做，先 B1 应急）：
- **B1 · 自包含 CLI**：把 adapter+schema+scrub 打成单文件/单包（内联或同目录），双击/一行就跑，跨平台分发。
- **B2 · 飞书发文件即处理**（最优，零安装）：
  ```
  用户把 brain.zip 拖进群发给 bot → bot 下载 → 解压 → adapter --redact-pii --publish → 回传脱敏 SFT
  ```
  - 前提：确认/补 **feishu-event 的「接收文件并下载」能力**（`feishu_agent.py`；飞书协议支持 `file_key`→下载 API，我们实现待确认，历史上图片接收有坑 14005）；
  - 自动脱敏，raw 不外泄。
- **沉淀进 `trajectory-collect/SKILL.md`** 成团队公共能力。

### 任务 C · 持续采集【可选】
本地 brain 自动同步到德国机（Syncthing / 计划任务 scp）+ 德国机 cron 定时 `adapter --publish`，配一次永久自动更新训练数据。

## 5. 关键约束（红线）
1. **对外只给 `--publish` 产物，绝不给 `--out` raw**（raw 含真密钥）。
2. raw 落地目录设私有（600）、不进 git、不团队可见；脱敏成品才进可训练区。
3. 改了 adapter（尤其客户端适配）→ diff 并回正本仓，保持单一事实源。

## 6. 代码获取
```bash
git clone https://gitee.com/karenliancau/cbu-llm-skills.git
cd cbu-llm-skills/feishu-plugin/skills/trajectory-collect/scripts
```
（德国机的具体 IP / 账号、共享区路径私下同步，勿入公开文档）
