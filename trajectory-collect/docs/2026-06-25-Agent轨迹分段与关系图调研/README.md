# Agent 轨迹分段与关系图构建 —— 业界/学界调研综述

> 版本：2026-06-25　起草：lian-server(claude)　数据负责人：张耀明
> 关联代码：`trajectory-collect/scripts/task_stitch.py` / `task_resegment.py`（飞书历史脊柱 + 回复链重切 + 子需求分解）
> 关联文档：`../训练数据归集方案.md`、`../2026-06-24-三家agent轨迹劫持保真度总账.md`
> 重点论文全文翻译见同目录 `全文翻译/` 子文件夹（见文末「附录·论文索引」）

> **飞书云文档版（已发 teamagent 数据处理群，2026-06-25）**：
> - 综述（主入口）：https://dcn0fouqbzgw.feishu.cn/docx/DVLddZyIiopQsaxdxRGcl00Bnic
> - CHIEF 译文：https://dcn0fouqbzgw.feishu.cn/docx/KdZsdh8lsoh68rx0mpHcIbJFnhf
> - Kummerfeld 对话解缠译文：https://dcn0fouqbzgw.feishu.cn/docx/Cfksdz9xnobLM6xCv4kcuLccnhf
> - Def-DTS 译文：https://dcn0fouqbzgw.feishu.cn/docx/Jln1dApxKowwOlxByB0cn3runOd

---

## 0. 一句话定性

我们这套「轨迹拆分 + 关系图」恰好踩在三个**互不交流的学术圈**的交叉点上：

1. **对话分段圈**（dialogue topic segmentation / disentanglement）——管「一段流怎么切成多个任务/线程」；
2. **离线 RL / Agent 训练数据圈**——管「原始日志怎么造成 SFT/RL 轨迹」；
3. **流程挖掘圈**（process mining）——管「事件日志怎么还原成流程图/依赖图」。

结论先行：**没有现成端到端方案覆盖我们的全链路，但每一块都有可直接抄的成熟范式。** 本文按我们正在做的三件事（`task_resegment` 重切 / `task_stitch` 拼接 / 回复链重切）组织，每条只留「强相关 + 有具体方法可落地」的。

---

## 一、轨迹拆分（对应 `task_resegment` / 回复链重切）

### 1.1 最对口的一篇：CHIEF —— agent 执行日志 → 子任务切分

《From Flat Logs to Causal Graphs》(arXiv 2602.23701)。它解决的正是「扁平 agent 日志 → 子任务节点」，范式几乎是我们 `task_resegment` 的现成蓝本：

- **硬约束「连续 + 全覆盖 + 不重叠」**——正是离线重切该守的不变量：切出来的子任务段必须连续、覆盖全部 step、互不重叠。我们重切器应把这条写成断言校验。
- **二阶段防幻觉**：先让 LLM 草拟分段 → 再用 reflection 校验每段 step 区间是否真对齐原始 log，不对齐就迭代。**我们现在若只让 LLM 一次性切，缺的就是这道「对齐回原始 log」的校验回路。**
- **coarse-to-fine 两级粒度**：先粗切大任务，再细切子任务——和我们「任务→子需求」两层分解同构。

→ 全文翻译：`全文翻译/译文-CHIEF-从扁平日志到因果图.md`

### 1.2 回复链重切有学术正本：Conversation Disentanglement

Kummerfeld 2019《A Large-Scale Corpus for Conversation Disentanglement》(arXiv 1810.11118, ACL 2019)。把「从交织消息流分离出线程」严格归约为 **回复关系预测（reply-to prediction）**：为每条消息找 parent，parent 链确定后线程树自然恢复。这正是 `task_stitch` v2「飞书脊柱 + 回复链」在做的事。

两个直接可搬的点：
1. **标准两步法** = 先 link prediction（判每条消息回的是哪条）→ 再贪心聚类成线程；
2. 它点名的**头号难点 = 长距离回复**（回的不是上一条，而是很久之前那条）——正是我们扁平飞书流最易切错处，**要专门构造测试用例覆盖**。

补充：近期 LLM 路线（reply-link 交给微调/zero-shot LLM 判）迁到开发者聊天可到 ~0.90 F1，可作我们规则重切之上的**兜底校验层**。

→ 全文翻译：`全文翻译/译文-Kummerfeld2019-对话解缠语料库.md`

### 1.3 三个可直接落地的工程细节

- **切之前先做指代消解/改写**（UR-DTS，arXiv 2409.07672，中文原文）：agent 日志全是「它 / 上面那个 / 这个结果」，不补全直接算相似度会失真。先 utterance rewriting 还原省略与指代，再切。
- **「是否新任务」别一次性硬问**（Def-DTS，arXiv 2505.21033）：拆成「先给每个 event 打 domain-agnostic 意图标签 → 再判相邻意图是否跳变」，比直接问 LLM「这里是不是新任务」更稳，且其设计**优化 recall**——对我们而言**漏切比多切更伤下游**（漏切=两个任务粘连成脏样本，多切=顶多碎一点还能合）。Def-DTS 有开源 prompt 模板，可直接复用。
  → 全文翻译（含 prompt 模板）：`全文翻译/译文-DefDTS-演绎推理对话主题分割.md`
- **评估用分段领域标准指标 Pk / WindowDiff**，别只看 F1。F1 对「切点偏一格」零容忍，而 Pk/WindowDiff 用滑动窗口容忍近邻误差，更贴合「切点大致对就行」的真实诉求。

---

## 二、轨迹关系图构建（对应「轨迹图 / 关系边」）

业界把轨迹转图分三条路线，**对我们最有价值的是「依赖图」路线**：

1. **流程挖掘路线（最低成本 baseline）**：把多轮日志当事件日志，用 `pm4py` 的 **directly-follows graph (DFG)** 自动发现流程图，再用 variants 发现「同类任务的不同执行路径」。现成库，适合先跑出时序骨架，零建模成本。
2. **依赖图路线（对我们最有价值）**：节点 = 子任务/工具调用，边 = **数据依赖**（B 用了 A 的输出）+ **控制依赖**（B 因 A 的结果才触发）。这正是我们 v3「关系边」要建的东西。判依赖的两种做法：①符号法——追工具输出在后续输入里的字面/引用复现（精确但脆）；②LLM 法——让模型读两个 step 判「B 是否依赖 A」（鲁棒但要控幻觉，建议加「指出依赖的具体字段」强制其给证据）。
3. **状态机路线**：把轨迹抽象成「状态 + 转移」，适合高度结构化的固定流程，对我们这种开放式 agent 任务**过拟合、不推荐**。

落地建议：**先 DFG 出骨架（便宜、可视化排错），再在子任务节点之间补数据/控制依赖边（LLM + 字段级证据）。** 关系边的 schema 已在 v3（commit 65f9470）开了头，沿用。

---

## 三、Agent 训练数据 pipeline（业界怎么从原始日志造 SFT/RL 轨迹）

业界主流 pipeline 的共性骨架（OpenAI-messages / verl 原生格式圈）：

1. **canonical IR 先行**：所有来源先归一到统一中间表示（我们已定 = OpenAI-messages 超集 = verl 原生），再谈下游。多源差异在归一层吸收，下游只认 IR。
2. **轨迹 = (state, action, observation)\* 序列**：state=上下文、action=模型输出（含 tool_call）、observation=工具返回。SFT 取 action 段做监督；RL 还要 reward。
3. **质量分层而非一刀切**：业界普遍按「可作推理监督 / 仅可作行为克隆 / 仅可作召回」给轨迹打质量档——和我们三家 agent 总账里「带 CoT 只能切 claude，BC 三家都能喂」的结论一致。
4. **难例标记（hard-example mining）**：把「工具报错后自我纠正」「多次试错才成功」的轨迹单独标出——这类对 agent 能力提升最大。我们 v3 已加难例标记字段（commit 65f9470）。
5. **去噪/脱敏在 IR 之后、入集之前**：固定一道脱敏关卡（我们已有 `processed/` 脱敏产物）。

→ 对我们的直接借鉴：**pipeline 顺序锁死为「采集 → 归一到 IR → 重切/分段 → 建关系图 → 质量分层 + 难例标记 → 脱敏 → 入集」**，每一步产物可独立校验、可回放。

---

## 四、多源 / 多 agent 轨迹对齐与拼接（对应 `task_stitch`）

我们要把「飞书群历史（人 + 多个 bot 的对话）+ 各 bot 自落盘轨迹」拼成完整任务轨迹，业界对口经验：

- **以一条权威时间轴为脊柱（spine）做对齐**——我们选「飞书群历史」当脊柱是对的：它是唯一含全部参与者、带可信时间戳的流。各 bot 自落盘轨迹按时间 + 内容指纹挂回脊柱。
- **跨源对齐靠「锚点匹配」**：用消息内容指纹 / 工具调用 ID / 时间窗三者交叉确认同一事件在不同源的对应，避免单靠时间戳错配（不同机器时钟漂移）。
- **多 agent 交错要先解缠再拼接**：多个 bot 在同一群并发回复 → 先用第一节的回复链解缠分出每个任务线程，再在线程内按 agent 拼接。**顺序是「先解缠、后拼接」，不能反。**
- **缺口显式标注、绝不静默丢**：某源缺某段（如 bot 动态卡 API 返占位、reasoning 被服务端上锁）→ 标 `unresolved` 留空位，不要假装连续（呼应总账文档与「禁止静默失败」原则）。

---

## 五、给我们的行动清单（落地优先级）

| 优先级 | 动作 | 对应代码 | 来源 |
|---|---|---|---|
| P0 | `task_resegment` 加「切完校验回原始 log（连续+全覆盖+不重叠）」断言回路 | `task_resegment.py` | CHIEF §1.1 |
| P0 | 回复链重切**专门构造长距离回复测试用例** | `task_stitch.py` + tests | Kummerfeld §1.2 |
| P1 | 切分前置一道指代消解/改写 | 新增预处理步 | UR-DTS §1.3 |
| P1 | 「是否新任务」改为「打意图标签→判跳变」两段式，优化 recall | `task_resegment.py` | Def-DTS §1.3 |
| P1 | 评估指标加 Pk / WindowDiff | eval 脚本 | §1.3 |
| P2 | 关系图先 `pm4py` DFG 出骨架，再补依赖边（带字段级证据） | v3 关系边 | §二 |
| P2 | pipeline 顺序固化 + 每步产物可独立校验 | 全链路 | §三 |

---

## 附录·论文索引

| 论文 | arXiv | 全文翻译 | 对口我们哪块 |
|---|---|---|---|
| From Flat Logs to Causal Graphs (CHIEF) | [2602.23701](https://arxiv.org/abs/2602.23701) | `全文翻译/译文-CHIEF-从扁平日志到因果图.md` | task_resegment 重切范式 |
| A Large-Scale Corpus for Conversation Disentanglement | [1810.11118](https://arxiv.org/abs/1810.11118) | `全文翻译/译文-Kummerfeld2019-对话解缠语料库.md` | 回复链重切学术正本 |
| Def-DTS: Deductive Reasoning for DTS | [2505.21033](https://arxiv.org/abs/2505.21033) | `全文翻译/译文-DefDTS-演绎推理对话主题分割.md` | 意图跳变判断 + prompt |
| UR-DTS (Utterance Rewriting) | [2409.07672](https://arxiv.org/abs/2409.07672) | 中文原文，无需翻译（见 arXiv PDF） | 切前指代消解 |

> 评估指标补充阅读：Pk（Beeferman 1999）、WindowDiff（Pevzner & Hearst 2002）——分段任务通用指标，比 F1 容忍近邻误差。
