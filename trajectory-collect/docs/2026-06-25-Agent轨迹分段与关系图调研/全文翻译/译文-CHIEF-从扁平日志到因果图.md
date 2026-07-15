# 【全文翻译】From Flat Logs to Causal Graphs: Hierarchical Failure Attribution for LLM-based Multi-Agent Systems

> 原文 arXiv: 2602.23701  https://arxiv.org/abs/2602.23701
> 译者：lian-server(claude)  译于 2026-06-25
> 框架名：CHIEF。本译文为团队内部归档，仅供学习参考；以原文为准。

---

# 从扁平日志到因果图：面向基于 LLM 的多智能体系统的分层失败归因

**作者**

Yawen Wang（王雅雯）¹²³⁴∗，Wenjie Wu（吴文杰）⁵∗，Junjie Wang（王俊杰）¹²³⁴†，Qing Wang（王青）¹²³⁴†

- ¹ 中国科学院软件研究所，中国北京
- ² 复杂系统建模与仿真技术国家重点实验室，中国北京
- ³ 综合信息系统技术国家级重点实验室（Science & Technology on Integrated Information System Laboratory），中国北京
- ⁴ 中国科学院大学，中国北京
- ⁵ 武汉理工大学，中国武汉

邮箱：{yawen2018, junjie, wq}@iscas.ac.cn，louis_wu@whut.edu.cn

（∗ 这些作者对本工作贡献相同；† 通讯作者。）

arXiv:2602.23701v1 [cs.AI]，2026 年 2 月 27 日

---

## 摘要（Abstract）

由 LLM 驱动的多智能体系统（Multi-Agent Systems，MAS）在复杂领域展现出了卓越的能力，但同时也存在固有的脆弱性与不透明的失败机制。

现有的失败归因（failure attribution）方法——无论是依赖直接提示（direct prompting）、代价高昂的重放（replay），还是有监督微调（supervised fine-tuning）——通常都把执行日志当作扁平的序列（flat sequences）来处理。这种线性视角无法解耦 MAS 内在的错综复杂的因果联系，从而导致可观测性弱（weak observability）、责任边界模糊（ambiguous responsibility boundaries）。

为了解决这些挑战，我们提出 CHIEF，一个把混乱的轨迹（chaotic trajectories）转化为结构化的分层因果图（hierarchical causal graph）的新颖框架。它随后采用分层的、由「神谕」引导的回溯（hierarchical oracle-guided backtracking），通过合成出的虚拟神谕（synthesized virtual oracles）来高效地剪枝搜索空间。最后，它通过一种渐进式因果筛查策略（progressive causal screening strategy）实现反事实归因（counterfactual attribution），以严格地把真正的根因（true root causes）与被传播的症状（propagated symptoms）区分开来。

在 Who&When 基准上的实验表明，CHIEF 在智能体级（agent-level）和步骤级（step-level）准确率上都超越了八个强大且达到当前最优（state-of-the-art）水平的基线方法。消融实验（ablation studies）进一步证实了我们提出的每个模块的关键作用。

---

## 1 引言（Introduction）

大语言模型（Large Language Models，LLM）的出现，赋予了智能体在感知（perception，Zheng 等人，2024）、规划（planning，Erdogan 等人，2025）和推理（reasoning，Putta 等人，2024）方面出色的能力。在这些能力的基础上，多智能体系统（MAS）应运而生，用来编排（orchestrate）各类专用智能体，在诸如软件工程（Ma 等人，2025；Wang 等人，2025）以及通用型现实任务（Mialon 等人，2024；Yoran 等人，2024）等复杂领域取得了卓越的性能。

然而，把自主智能体与各种工具集成在一起会引入固有的脆弱性。近期研究揭示，失败率高达 86.7%（Cemri 等人，2025），错误会沿着错综复杂的依赖关系传播，使系统变得不可靠且不透明。

**（图 1：由一个未被满足的约束所引发的失败日志。尽管编排器（Orchestrator）在第 14 步标注了约束（红色），网页浏览智能体（Websurfer）在执行时（第 16 步）却忽略了这些约束。）**

因此，失败归因（Failure Attribution）——即识别导致任务失败的根因（root cause）和应负责任的智能体——已经成为一项至关重要却又极具挑战性的任务。自从 Who&When 基准（Zhang 等人，2025d）被提出以来，已有各种方法被提出来自动化这一诊断过程。基于 LLM 的直接方法（Zhang 等人，2025d；Banerjee 等人，2025）往往难以在冗长的上下文中捕捉到细粒度的因果线索；基于频谱（spectrum-based）的方法（Ge 等人，2025）由于需要反复重放轨迹，会产生高昂得令人却步的 token 成本；基于微调（fine-tuning-based）的方法（Zhang 等人，2025a、b）则带来巨大的训练开销与泛化风险。

更关键的是，当前的方法受限于把 MAS 轨迹当作扁平序列来处理，忽略了图 1 中所展示的固有的结构复杂性。与简单的线性文本不同，这些日志代表着智能体的观察（observations）、思考（thoughts）、动作（actions）和结果（results）的稠密交织，它们由严格的数据依赖（例如网页搜索结果）和智能体交互（例如编排器与执行器之间的交互）相互关联。这种结构抽象（structural abstraction）和因果解耦（causal disentanglement）的缺失，使得现有方法在面对失败归因中固有的三大诊断障碍时举步维艰：

**不透明的因果流（Opaque Causal Flows）：** 如图 1 所示，原始的 MAS 日志呈现出工具执行、环境反馈与智能体间通信的稠密交织。在没有结构化解析的情况下，隐含的依赖关系与因果联系都淹没在这冗长的文本中，导致归因分析的可观测性很弱。

**稀疏的中间监督（Sparse Intermediate Supervision）：** 与拥有模块化单元测试（modular unit tests）的传统软件调试不同，MAS 轨迹缺乏中间的真值（ground truth）。由于正确性只能在最终结果处被观察到，要在长程（long-horizon）轨迹中精确定位失败步骤，就变成了一场「大海捞针」（needle in a haystack）式的搜索。

**模糊的责任边界（Ambiguous Responsibility Boundaries）：** 正如图 1 所例示的，一处约束违反（例如「步行 5 分钟」）触发了失败（第 19 步），但究竟错误源自编排器的疏漏（第 14 步）还是执行器的疏忽（第 16 步），却并不清楚。这种「错误表现之处」与「错误引入之处」之间的差异，使得在没有因果抽象的情况下，难以把根因从被传播的症状中区分出来。

为了解决这些障碍，我们提出 CHIEF，一个面向基于 LLM 的 MAS 的因果分层失败归因框架（Causal HIErarchical Failure attribution framework）。我们摒弃了把日志当作扁平序列处理的范式，转而显式地把混乱的轨迹重构为一个结构化的图，从而把归因转化为一个透明且系统化的分而治之（divide-and-conquer）过程。

**第一，** 我们构建一个分层因果图（Hierarchical Causal Graph）。CHIEF 把任务分解为子任务（subtasks），应用观察-思考-动作-结果（Observation-Thought-Action-Result，OTAR）解析来解耦交织在一起的智能体行为，并显式地建模步骤之间的数据依赖。

**第二，** 我们提出分层的、由神谕引导的回溯（Hierarchical Oracle-Guided Backtracking）。这个模块充当一种有组织的分而治之策略，用一种自顶向下（top-down）的搜索来取代线性扫描。通过用合成出的虚拟神谕来验证子任务，我们绕过了对细粒度动作的逐一检查，高效地剪枝搜索空间，从而精确定位失败步骤。

**第三，** 我们通过一种渐进式因果筛查策略来实现反事实归因（Counterfactual Attribution）。通过依据因果作用域（causal scope）与依赖性质（dependency nature）来过滤责任，并应用一个面向偏离的可逆性检查（deviation-aware check for reversibility），我们严格地把真正的根因从被传播的症状中区分出来。

我们的实验评测利用了 Who&When 基准（Zhang 等人，2025d），它涵盖 127 种多样的 MAS 架构。CHIEF 展现出卓越的性能，超越了被归为四种范式的 8 个基线。在手工构造（hand-crafted）子集上，它取得了 77.59% 的智能体级准确率和 29.31% 的步骤级准确率；在算法生成（algorithm-generated）子集上，则分别取得了 76.80% 和 52.00%。此外，消融实验隔离了我们所提模块各自的影响，验证了分层因果图与虚拟神谕对于有效的失败归因至关重要。

我们的贡献总结如下：

- 一种新颖的方法 CHIEF，它把混乱的轨迹转化为一个结构化的图，从而实现一个透明且系统化的分而治之归因过程。
- 一种分层因果图的构建方法，它建立起一个结构化的基础，既能促进精确的失败归因，也能服务于对 MAS 的其他理解与分析。
- 在 Who&When 基准上的大量实验，证明了其相对于现有基线的卓越性能。
- 在我们网站上的复现包¹，以支持可复现性与未来研究。

（¹ https://anonymous.4open.science/r/CHIEF-86B8）

---

## 2 相关工作（Related Work）

### 2.1 基于 LLM 的多智能体系统（LLM-based Multi-Agent Systems）

基于 LLM 的 MAS 通过编排专用智能体，在复杂任务上取得了卓越的性能（Li 等人，2023；Hong 等人，2024；Wu 等人，2023）。现有框架大致分为两类：依赖预定义标准操作流程（standard operating procedures）的**手工构造系统**（例如 AutoGen，Wu 等人，2023；MetaGPT，Hong 等人，2024；ChatDev，Qian 等人，2024）；以及自主优化智能体角色与拓扑（topologies）的**自动化系统**（例如 AgentPrune，Zeng 等人，2025；AFlow，Zhang 等人，2025c）。

尽管在推理与助手类任务上取得了成功（Zhuge 等人，2024；Mialon 等人，2024），但智能体与工具之间错综复杂的编排引入了固有的脆弱性，错误会沿着不透明的依赖关系传播。我们把关注点从系统构建转移开来，去攻克自动化失败归因这一关键的下游任务。

### 2.2 LLM 智能体的失败归因（Failure Attribution for LLM Agents）

失败归因对系统调试至关重要，但现有方法面临各自鲜明的局限。早期工作（Zhang 等人，2025d）采用「LLM 即评判者」（LLM-as-a-Judge）范式，但无法处理冗长的日志，常常只能得到低于 10% 的准确率。ECHO（Banerjee 等人，2025）虽然引入了分层上下文（hierarchical context）和共识投票（consensus voting），但它仅仅把层级结构当作一种静态表示（static representation），常常把可见的症状误当作隐藏的根因。

基于频谱的 FAMAS（Ge 等人，2025）依赖反复重放来做统计归因。虽然它对长轨迹有效，但成本高昂，且只能得出相关性（correlations）而无法给出因果解释。AgenTracer（Zhang 等人，2025a）和 GraphTracer（Zhang 等人，2025b）这类方法则在合成的失败数据上微调专用模型（例如 8B 参数规模）。尽管它们能达到很高的准确率，但需要在数据生成与模型训练上付出可观的前期成本，并且在应用于未见过的（unseen）智能体日志时面临泛化风险。

与先前工作不同，CHIEF 重构出一个分层因果图，使得高效的单遍（one-pass）推理成为可能，无需昂贵的重放或额外的训练即可产出令人满意的归因。

---

## 3 问题形式化（Problem Formulation）

**LLM MAS。** 我们遵循标准的基于轮次（turn-based）的 LLM MAS 协议（Hong 等人，2024；Wu 等人，2023），即在每个时间步恰好有一个智能体执行一个动作。该 MAS 定义为 𝓜 = ⟨𝓝, 𝓢, 𝓐, 𝓟⟩，由 N 个智能体 𝓝 = {i}ᵢ₌₀ᴺ 构成。在第 t 步，活跃智能体 iₜ 执行一个动作 aₜ，通过转移概率 𝓟(s_{t+1} | sₜ, aₜ) 把系统从状态 sₜ 转移到 s_{t+1}。这产生一条轨迹 τ = {sₜ, (iₜ, aₜ)}ₜ₌₀ᵀ，其结果 Z(τ) ∈ {0, 1}（其中 1 表示失败）。

**失败归因。** 对于一条失败的轨迹 τ（即 Z(τ) = 1），失败归因问题就是把特定的「智能体-步骤对」(i, t) 识别为失败的根因。我们首先定义决定性错误指示函数（decisive error indicator）：

Δ_{i,t}(τ) = 𝕀( Z(τ̃^{(i,t)}) = 0 )

其中 τ̃^{(i,t)} 是把智能体 i 在第 t 步的动作修正之后所得到的反事实轨迹（counterfactual trajectory）。Δ_{i,t} = 1 意味着该修正导致了成功（即 Z(τ̃^{(i,t)}) = 0），从而把 (i, t) 识别为一处决定性错误；否则为 0。

实际上，由于错误传播（error propagation），一条轨迹可能包含多处决定性错误。根因被定义为时间序列上最早的那处决定性错误（Zhang 等人，2025d）：

> (i\*, t\*) = arg min_{(i,t): Δ_{i,t}(τ)=1} t　　（公式 1）

因此，目标是构造一个映射 𝓕，能够从原始轨迹 τ 中准确地恢复出真值根因 (i\*, t\*)。

---

## 4 方法（Method）

为了化解 MAS 诊断的复杂性，我们提出一个因果分层失败归因框架。如图 2 所示，我们通过三个阶段把归因转化为一个结构化的、分而治之的推理过程：（1）**分层因果图构建（Hierarchical Causal Graph Construction）**，用来解析扁平的轨迹；（2）**分层的、由神谕引导的回溯（Hierarchical Oracle-Guided Backtracking）**，用来自顶向下地识别错误候选；（3）**反事实归因（Counterfactual Attribution）**，用来把根因从被传播的症状中区分出来。

**（图 2：CHIEF 的总览。）**

### 4.1 分层因果图构建（Hierarchical Causal Graph Construction）

原始的 MAS 日志虽然表面上是扁平的文本序列，但其内在拥有一个潜在的执行拓扑（latent execution topology）。它把环境反馈、高层规划、底层执行等内容交织在一起，全部由隐式的数据/控制依赖（data/control dependencies）、智能体交互以及逻辑/时序关系相互关联。由于错误恰恰是沿着这些隐藏的路径——而非线性的文本——传播的，因此我们构建一个分层因果图（Hierarchical Causal Graph，HCG），记作 𝓖 = (𝓥, 𝓔)，来显式地映射这一结构。

#### 4.1.1 分层节点（Hierarchical Node）

节点集合 𝓥 = 𝓥_sub ∪ 𝓥_agt 把轨迹抽象为两种粒度的语义单元：

**子任务节点（Subtask Node）** Sₖ ∈ 𝓥_sub 被定义为一个高层的逻辑抽象，表示一个独立的任务阶段，由图 5(a) 中详述的结构化属性来刻画。为了把原始轨迹 τ 划分为一个既逻辑自洽、又忠实于执行日志的序列 {Sₖ}ₖ₌₀ᴷ，我们采用一个两步流程：

**（1）基于 RAG 的任务分解（RAG-based Task Decomposition）。** 为了确保合理性，我们从已有的基准（即 GAIA，Mialon 等人，2024；AssistantBench，Yoran 等人，2024）中构建一个知识库，库中填充了人工标注的逐步解答范例（step-by-step solution exemplars）。我们利用检索增强生成（Retrieval-Augmented Generation，RAG）来检索相关的分解原型（decomposition prototypes），把它们作为少样本（few-shot）提示，引导 LLM 生成一个模块化的任务计划。实现细节与示例提示见附录 A。

**（2）轨迹对齐的反思（Trajectory-Aligned Reflection）。** 为了防止幻觉式（hallucinated）的任务分解，一个反思机制会验证所生成的 Sₖ 与原始日志 τ 之间的对齐情况。不匹配的子任务会经历迭代式的精化（iterative refinement），以确保分解严格遵循实际的执行流程。

**智能体节点（Agent Node）** a ∈ 𝓥_agt 被定义为一个原子化的执行单元，表示子任务内某个特定的智能体实例。为了严格地刻画该智能体的行为，我们使用 OTAR 元组为每个节点赋予结构化属性：⟨观察（Observation），思考（Thought），动作（Action），结果（Result）⟩，它是对 TAR 模式（TAR schema，Bouzenia 和 Pradel，2025）的扩展。这些属性的各个组成部分由一个基于 LLM 的解析器从原始轨迹 τ 中抽取出来。我们在附录 B 中详述 OTAR 解析，并在图 5(b) 中图示智能体节点的属性。

#### 4.1.2 因果边构建（Causal Edge Construction）

边集合 𝓔 = 𝓔_sub ∪ 𝓔_agt ∪ 𝓔_step 在三个层级上捕捉因果依赖。具体而言，**子任务边（Subtask Edges，𝓔_sub）** 连接相邻的子任务，用来建模任务的高层逻辑演进（logical progression）；而**智能体边（Agent Edges，𝓔_agt）** 连接智能体节点，用来表示智能体间的协作。在最细的粒度上，我们通过映射执行步骤之间的数据流依赖（例如变量引用）来显式地构建**步骤边（Step Edges，𝓔_step）**。这种多层级的连通性建立起了追踪错误传播所必需的路径。

我们根据边的类型来构建不同的边。子任务边和智能体边利用**反事实模式（Counterfactual Patterns）** Φ，即 Bias(u) —Φ→ Anomaly(v)，把上游的潜在偏离 Bias(u) 与下游可观察的错误 Anomaly(v) 关联起来。相比之下，步骤边充当**数据快照（data snapshots）**，显式地记录确切的上游输出与下游输入。把它们结合起来，我们就为后续的回溯阶段提供了全面的证据。我们在附录 C 中详述其实现，并在图 5(c) 中图示边的属性。

### 4.2 分层的、由神谕引导的回溯（Hierarchical Oracle-Guided Backtracking）

为了在图 𝓖 中精确定位根因 (a\*, x\*)（即智能体-步骤对），我们提出一个两阶段的策略，包含**子任务虚拟神谕合成（Subtask Virtual Oracle Synthesis）** 和**分层失败回溯（Hierarchical Failure Backtracking）**。我们首先为每个子任务节点 Sₖ 合成一个虚拟神谕 𝓞ₖ，把它作为一种中间监督来在子任务级验证执行的正确性。随后我们执行一个自顶向下的回溯过程，逐步把失败范围从粗粒度的子任务收窄到细粒度的智能体步骤。

#### 4.2.1 子任务虚拟神谕合成（Subtask Virtual Oracle Synthesis）

对于每个子任务节点 Sₖ ∈ 𝓥_sub，我们合成一个虚拟神谕 𝓞ₖ，作为理想的中间验证器（intermediate verifier）。我们把它形式化为一个结构化的语义元组：

𝓞ₖ = ⟨𝓖_sub, 𝓟_pre, 𝓔_key, 𝓒_acc⟩

其中 𝓖_sub 定义了从任务指令派生而来的、该子任务的具体**目标（Goal）**；𝓟_pre 勾勒出依赖于上游输出与环境状态的**前置条件（Preconditions）**；𝓔_key 突出在推理过程中必须被验证的**关键证据（Key Evidence）**；𝓒_acc 则确立了判定该子任务是否成功的**接受准则（Acceptance Criteria）**。

我们把 𝓞ₖ 的合成建模为一个由 LLM 参数化的生成函数 𝓕_gen：

> 𝓞ₖ = 𝓕_gen( 𝓘, 𝓞_{<k}, τ_{>k}, τ_s )　　（公式 2）

其输入包括：任务指令 𝓘、先前神谕的历史 𝓞_{<k}、后续尚未处理的轨迹 τ_{>k}（即在已完成的子任务之后剩下的步骤），以及通过 RAG 检索到的相似任务分解范例 τ_s。实现细节见附录 D。

#### 4.2.2 分层失败回溯（Hierarchical Failure Backtracking）

基于因果图与合成出的神谕，我们执行一个自顶向下的回溯过程，逐步收窄失败范围。该过程在三个层级上识别失败候选：

**子任务级（Subtask Level）。** 为了追踪失败源头，我们按照逆拓扑序（reverse topological order）遍历子任务节点。对于每个子任务 Sₖ，我们采用一个基于 LLM 的语义评估器（semantic evaluator）𝓕_eval，通过把其实际执行输出与神谕的目标（𝓖_sub）和接受准则（𝓒_acc）相比较，来得到二元判定。输出为 1 表示存在差异，从而把该子任务识别为失败候选：

> 𝓒_sub = { Sₖ ∈ 𝓥_sub | 𝓕_eval(Sₖ, ⟨𝓖_sub, 𝓒_acc⟩) = 1 }　　（公式 3）

**智能体级（Agent Level）。** 对于每个候选子任务 Sₖ ∈ 𝓒_sub，我们下钻到其构成的各个智能体。我们应用语义评估器 𝓕_eval 来评估每个智能体（a ∈ Sₖ）的 OTAR 元组与神谕的前置条件（𝓟_pre）和关键证据（𝓔_key）的一致性。表现出偏离（即 𝓕_eval 输出 1）的智能体被识别为候选：

> 𝓒_agt = { a ∈ Sₖ | 𝓕_eval(a_otar, ⟨𝓟_pre, 𝓔_key⟩) = 1 }　　（公式 4）

**步骤级（Step Level）。** 给定候选子任务 Sₖ ∈ 𝓒_sub 内的候选智能体 a ∈ 𝓒_agt，我们对其执行的步骤集合 𝓧 进行最终审查。我们采用 𝓕_eval 来验证每个步骤 x ∈ 𝓧，把其实际执行细节（例如输入/输出）与该智能体汇总后的 OTAR（a_otar）以及子任务的神谕约束（𝓞ₖ）进行交叉比对。表现出偏离（即 𝓕_eval 输出 1）的步骤被精确定位为失败候选：

> 𝓒_step = { x ∈ 𝓧 | 𝓕_eval(x, ⟨a_otar, 𝓞ₖ⟩) = 1 }　　（公式 5）

基于 LLM 的语义评估器 𝓕_eval 的详细实现见附录 E。

### 4.3 反事实归因（Counterfactual Attribution）

在识别出失败候选之后，本阶段把它们归因到其真正的源头。为了系统性地解开错误传播路径，我们采用一种渐进式因果筛查策略。该方法依据**因果作用域**（局部 vs. 非局部，local vs. non-local）通过**局部归因（Local Attribution）** 来区分责任；依据**依赖性质**（控制 vs. 数据，control vs. data）通过**规划-控制归因（Planning-Control Attribution）** 与**数据流归因（Data-Flow Attribution）** 来区分责任。最后，**面向偏离的归因（Deviation-Aware Attribution）** 充当一个有效性过滤器（validity filter），剪除那些随后被自我纠正的瞬态错误（transient errors）。下面给出每个阶段的依据（rationale），技术实现细节见附录 F。

#### 4.3.1 局部归因（Local Attribution）

本阶段验证局部化的错误是否源自步骤 x 本身，而非从上游传播而来。我们通过应用绑定在依赖边上的反事实模式 Φ，来识别上游的因果触发者（causal triggers）𝓢_cause：

> 𝓢_cause = { x′ ∈ Pre(x) | Bias(x′) —Φ→ Anomaly(x) }　　（公式 6）

𝓢_cause 表示任何能够在因果上解释当前 x 处错误的先前步骤 x′ ∈ Pre(x)（即 x 的前驱步骤）。我们基于这个集合来进行归因：

- 若 𝓢_cause = ∅，意味着步骤 x 接收到了有效的输入，却产生了错误的输出。因此，我们把根因在局部归到 x。
- 若 𝓢_cause ≠ ∅，则错误是从上游传播而来。我们排除 x，转入规划-控制阶段与数据流阶段，去精确定位非局部的源头。

#### 4.3.2 规划-控制归因（Planning-Control Attribution）

若错误被判定为非局部的，我们首先调查控制流错误（control flow errors），它们常常表现为冗余的循环行为（cyclic behaviors）。由于这些循环往往模糊了规划错误与执行错误之间的界限，我们的核心思想是判别智能体究竟是在尝试调整其计划，还是仅仅未能执行一个有效的计划。

为了解耦责任，我们把重复出现的步骤序列聚合为一个**循环组（Loop Group）**，并检查迭代式重新规划（re-planning）与重新执行（re-execution）步骤背后的具体理由。当规划者（编排器）在反复收到错误信号的情况下，仍然生成语义上完全相同的思考或命令，表明它未能更新控制流时，我们把责任归给**规划者（planner / orchestrator）**。反之，当规划者主动提出有效的策略转变（例如修改工具参数或 API）以打破循环，而执行者却始终产出异常结果时，我们把责任归给**执行者（executor）**。图 12 为这两种失败模式提供了具体示例。

#### 4.3.3 数据流归因（Data-Flow Attribution）

除了控制流之外，非局部的错误也可能通过数据依赖来传播。我们利用步骤级的数据流边 𝓔_step 以及 OTAR 元组中显式的变量引用，重建错误传播路径。我们沿着这条路径审计数据一致性（data consistency），回溯定位那个最先把有效的上游输入「污染」（corrupted）成异常结果的具体步骤。这有效地把**数据生成者**（即真正的根因，例如一个被幻觉出来的事实）与下游的**数据传播者**（即症状，例如基于坏输入的计算错误）解耦开来，确保归因瞄准的是源头而非表现。

#### 4.3.4 面向偏离的归因（Deviation-Aware Attribution）

MAS 常常表现出自我纠正（self-correcting）的能力。为了严格地把根因从瞬态波动（transient fluctuations）中区分出来，我们评估所识别错误的**因果可逆性（Causal Reversibility）**。对于一个上游可疑步骤 xₜ，我们检查其后续轨迹 τ_{>t} 中是否存在自我纠正的证据。若某个后续步骤 x_{t+k} 成功地重新满足了神谕准则（即系统返回到了一个有效状态），那么 xₜ 处最初的偏离就被认定为可逆的，并不被赋予任何责任。因此，我们优先把归因指向**不可逆的错误（irreversible errors）**。

---

## 5 实验设置（Experiment Setup）

### 5.1 基准（Benchmark）

我们的评测利用 Who&When（Zhang 等人，2025d），据我们所知，这是目前唯一公开的 MAS 失败归因基准。该数据集派生自 GAIA（Mialon 等人，2024）和 AssistantBench（Yoran 等人，2024）中的通用型任务，包含 184 条失败日志，分为两个子集：

（1）一个**算法生成（algorithm-generated）子集**，包含来自 126 种多样架构的 126 条日志，由 CaptainAgent（Song 等人，2024）创建；

（2）一个**手工构造（hand-crafted）子集**，包含来自 Magentic-One（Fourney 等人，2024）系统的 58 条日志。

真值的可靠性通过人类专家的多轮共识标注（multi-round consensus annotation）来保证。

### 5.2 基线（Baselines）

我们把 CHIEF 与被归为四种范式的八个有代表性的方法进行比较：

**随机（Random）：** 一个下界基线，从日志中随机选择一个智能体和一个步骤。

**基于 LLM 的提示（LLM-based Prompting）：** 我们纳入 Who&When 基准（Zhang 等人，2025d）中提出的三种策略：（1）一次性（All-at-once，直接预测），（2）逐步（Step-by-step，顺序判断），（3）二分查找（Binary-search，递归收窄）。我们另外纳入 ECHO（Banerjee 等人，2025），它把分层上下文抽取与共识投票相结合。值得注意的是，ECHO 主要把层级结构用于静态上下文表示和目标一致性检查（objective consistency checks），而非构建一个因果图来做自顶向下的回溯。

**基于频谱的方法（Spectrum-based Method）：** 我们纳入 FAMAS（Ge 等人，2025），它通过对反复重放的轨迹进行变化的统计分析来工作。

**基于微调的方法（Fine-tuning-based Methods）：** 我们纳入两个 8B 参数规模的微调模型：AgenTracer（Zhang 等人，2025a），通过在反事实重放上进行多粒度强化学习（multi-granular reinforcement learning）来训练；以及 GraphTracer（Zhang 等人，2025b），它利用信息依赖图（information dependency graphs）来构造合成样本以做有监督的归因。

**（表 1：Who&When 基准上的主要结果。我们报告两个子集上的智能体级和步骤级准确率（%）。每个单元格报告两个值：左侧对应「带 𝓖」（w/ 𝓖）的设置，右侧对应「不带 𝓖」（w/o 𝓖）的设置，其中 𝓖 表示可以访问任务真值（即正确的结果）。最佳结果以粗体表示。）**

| 类型 | 方法 | 手工构造-智能体级↑ | 手工构造-步骤级↑ | 算法生成-智能体级↑ | 算法生成-步骤级↑ |
|---|---|---|---|---|---|
| 启发式 | Random | 12.00 / 12.00 | 4.20 / 4.20 | 29.10 / 29.10 | 19.10 / 19.10 |
| 基于 LLM 的提示 | All-at-Once | 50.00 / 48.28 | 5.17 / 5.17 | 61.11 / 59.52 | 13.49 / 15.87 |
| | Step-by-Step | 36.00 / 34.30 | 6.60 / 6.90 | 39.70 / 28.30 | 27.40 / 17.80 |
| | Binary Search | 51.70 / 36.20 | 6.90 / 6.90 | 44.10 / 30.10 | 24.00 / 16.60 |
| | ECHO | 68.40 / 67.90 | 28.10 / 26.80 | 68.80 / 67.20 | 28.80 / 27.20 |
| 基于频谱 | FAMAS | 62.07 / – | 41.38 / – | 55.56 / – | 23.81 / – |
| 基于微调 | AgenTracer | 69.10 / 63.82 | 20.70 / 20.68 | 69.62 / 63.73 | 42.90 / 37.30 |
| | GraphTracer | 74.91 / 69.74 | 28.63 / 27.97 | 76.64 / 67.42 | 49.97 / 44.35 |
| | **CHIEF（本文）** | **77.59 / 72.41** | **29.31 / 29.31** | **76.80 / 68.80** | **52.00 / 45.60** |

### 5.3 评测指标（Evaluation Metrics）

遵循标准协议（Zhang 等人，2025d；Ge 等人，2025；Zhang 等人，2025a），我们报告两种粒度上的准确率：

（1）**智能体级准确率（Agent-level Accuracy）：** 正确识别出失败责任智能体的案例所占的比例。

（2）**步骤级准确率（Step-level Accuracy）：** 正确归因到确切根因步骤的案例所占的比例。

为了缓解随机性，所有结果都是三次独立运行的平均值，采用严格的 top-1 准则（即真值目标被排在第一位）。

### 5.4 实现细节（Implementation Details）

我们使用 Python 3.11 实现 CHIEF。除非另有说明，所采用的基座 LLM 为 DeepSeek-V3.2（thinking）。对于基线，我们优先采用官方实现及其默认配置；对于因版本差异或随机性而难以复现的基线，则采用其论文中报告的结果。实验在一台配备 Intel i7-10700 CPU、一块 NVIDIA TITAN RTX GPU 和 32GB 内存的服务器上进行。

---

## 6 结果（Results）

### 6.1 主要结果：基线对比（Main Results: Baseline Comparison）

如表 1 所示，CHIEF 在 Who&When 基准上相对于 8 个基线展现出压倒性的性能，在所有指标上都超越了现有方法，仅有一处微小的例外。CHIEF 显著超越了所有直接提示基线（All-at-once、Step-by-step、Binary Search）。虽然 ECHO 得益于分层结构，但它把层级结构的使用局限于静态上下文表示。CHIEF 推进了这一范式，显式地构建一个因果图来引导自顶向下的回溯，从而实现了更优的归因准确率。

此外，CHIEF 以较低的计算成本超越了那些昂贵的基线。虽然基于频谱的 FAMAS 依赖代价高昂的重放，在手工构造子集上取得了最佳的步骤级准确率（在该子集上，更长的轨迹使得可靠的统计分析成为可能），但 CHIEF 在所有其他设置上的统治性表现证明了因果推理更高效——它以零重放成本（zero replay cost）交付了稳健的性能。类似地，微调方法（AgenTracer、GraphTracer）会带来高昂的训练成本且难以泛化，但仍然不及 CHIEF。这表明，把扁平日志结构化为因果图，相比参数更新（parameter updates）能更有效地释放 LLM 的推理能力。

CHIEF 对于真值（𝓖）的缺失高度鲁棒。虽然能访问 𝓖 会让所有方法都受益，但在相同设置下，CHIEF 始终取得更优的结果（FAMAS 的步骤级准确率除外²）。这种稳定性来自我们的虚拟神谕，它提供了有效的中间监督，确保在不严重依赖最终任务结果的情况下也能实现精确归因。

（² 由于 FAMAS 在其论文中只报告了带真值的结果，表 1 中缺失的条目以 – 表示。）

### 6.2 成本效率分析（Cost Efficiency Analysis）

**（表 2：在「带 𝓖」设置下，Who&When 两个子集上每个案例的平均 token 成本。越低越好。）**

| 方法 | 手工构造 | 算法生成 |
|---|---|---|
| All-at-Once | 21,581 | 5,833 |
| Step-by-Step | 87,720 | 6,533 |
| Binary Search | 34,659 | 5,226 |
| FAMAS | 431,620 | 116,660 |
| ECHO | 53,701 | 25,642 |
| **CHIEF** | 55,085 | 19,504 |

**（表 3：使用各种基座 LLM 时，CHIEF 在 Who&When 基准（带 𝓖）上的性能。我们报告两个子集上的智能体级和步骤级准确率（%），括号中标注了模型的思考等级（thinking level）。）**

| 类型 | 基座 LLM | 手工构造-智能体级↑ | 手工构造-步骤级↑ | 算法生成-智能体级↑ | 算法生成-步骤级↑ |
|---|---|---|---|---|---|
| 开源 | Qwen3-235B-A22B-Instruct-2507 | 63.79 | 22.41 | 69.04 | 32.53 |
| | Kimi-k2-0905-preview | 67.24 | 24.13 | 68.25 | 42.06 |
| | Deepseek-V3.2(thinking) | 77.58 | 29.31 | 76.80 | 52.00 |
| 闭源 | GPT-5.2(medium) | 68.96 | 24.13 | 66.67 | 43.65 |
| | Claude-4.5-Sonnet(standard thinking) | 68.96 | 27.58 | 63.49 | 51.58 |
| | Gemini-3-Flash-Preview(medium) | 70.68 | 29.31 | 69.84 | 50.79 |

为了在准确率之外评估成本效率，我们呈现了不同范式在每条失败日志上的平均 token 消耗，见表 2。相比直接提示（例如 All-at-once），CHIEF 带来的 token 增长是中等的（2.5～3 倍），与 ECHO 的成本相当。然而这一成本是值得的，因为 CHIEF 取得了显著更优的归因准确率。基于频谱的 FAMAS 由于反复重放而成本高昂，消耗的 token 是 CHIEF 的 6～8 倍。相比之下，CHIEF 采用「单遍」因果图推理，绕过了昂贵的环境再交互，大幅降低了开销。虽然微调方法（AgenTracer、GraphTracer）可能提供更低的推理成本³，但它们需要在错误注入数据生成（error-injection data generation）与模型微调上付出高昂的前期成本。反观 CHIEF，它直接开箱即用（off-the-shelf）就能交付卓越的性能，绕过了微调流水线的可观开销。

（³ 由于这些微调模型未开源，我们没有报告它们的推理成本。）

### 6.3 基座 LLM 的影响（Impact of Base LLM）

我们评估 CHIEF 在各种基座 LLM 上的泛化能力，涵盖闭源模型（例如 GPT-5.2、Claude 4.5 Sonnet、Gemini-3）和开放权重（open-weight）模型（例如 Qwen 3、Kimi-k2、DeepSeek-V3.2），结果详见表 3。

归因准确率与基座 LLM 的指令跟随（instruction-following）和推理能力正相关。我们采用作为默认骨干（backbone）的开放权重模型 DeepSeek-V3.2 在所有设置上都取得了最高的性能。在闭源模型中，Gemini-3 在两个子集上都领先，唯有 Claude 4.5 在算法生成子集上交付了最佳的步骤准确率。值得注意的是，像 GPT-5.2 这样的闭源模型表现不如高思考等级（high-thinking）的开放模型 DeepSeek-V3.2。这主要是因为我们为节省成本而把它们配置在了较低的思考等级。此外，我们的提示设计可能并未针对特定模型做最优适配。总的来说，这些结果表明 CHIEF 能驾驭基座 LLM 的能力来构建精确的因果图和虚拟神谕，进而释放出 LLM 内在的推理潜能以实现有效的归因。

### 6.4 消融实验（Ablation Study）

为了评估我们所提各组件的独立影响，我们定义三个模块：M1（分层因果图，Hierarchical Causal Graph）、M2（分层的、由神谕引导的回溯，Hierarchical Oracle-Guided Backtracking）、M3（反事实归因，Counterfactual Attribution）。以 M1 作为结构基础（朴素基线 All-at-Once 除外），我们评估三种配置：

- ❶ **仅 M1（Only M1）：** 利用因果图，但在图上通过直接提示来做归因，不含 M2 和 M3。
- ❷ **M1+M2：** 引入虚拟神谕以做精确的自顶向下回溯，但不含 M3。
- ❸ **M1+M3：** 直接在图上应用反事实推理，但缺乏 M2 所提供的中间监督与搜索空间剪枝。

**（表 4：在「带 𝓖」设置下，CHIEF 在 Who&When 基准上的消融实验。我们评估各模块（M1–M3）的增量贡献，报告两个子集上的智能体级和步骤级准确率（%）。）**

| CHIEF 变体 | 手工构造-Agent↑ | 手工构造-Step↑ | 算法生成-Agent↑ | 算法生成-Step↑ |
|---|---|---|---|---|
| All-at-Once | 50.00 | 5.17 | 61.11 | 13.49 |
| Only M1 | 37.93 | 18.96 | 66.66 | 24.60 |
| M1+M2 | 51.72 | 22.41 | 61.11 | 34.12 |
| M1+M3 | 50.00 | 22.41 | 65.07 | 26.98 |
| **CHIEF** | 77.59 | 29.31 | 76.80 | 52.00 |

结果呈现在表 4 中。首先，单独引入 HCG（M1）在算法生成子集上能持续提升性能，但在手工构造子集（其特点是轨迹更长）上结果好坏参半：步骤级准确率上升了，但相比 All-at-Once，智能体级准确率却下降了。这表明，在没有引导式推理（M2/M3）的情况下，稠密的结构信息会引发认知过载（cognitive overload），妨碍 LLM 有效地利用该图。其次，把 M1 与 M2 或 M3 中的任一个结合，都能恢复性能。M1+M2 能实现精确的回溯，但由于缺乏因果归因，有把症状与根因混淆的风险。反之，M1+M3 应用了反事实推理，但苦于没有神谕引导的剪枝，只能依赖原始的直觉去扫描整个图。第三，CHIEF 的完整版本在所有指标上都取得了更优的准确率，证明了各模块的互补作用：M1 提供结构基础，M2 通过自顶向下的回溯高效地收窄搜索空间，M3 通过反事实分析严格地验证根因。这种整体性的集成，对于在冗长且无结构的 MAS 轨迹中实现精确归因是不可或缺的。

---

## 7 结论（Conclusion）

本文提出了 CHIEF，一个面向基于 LLM 的 MAS 的因果分层失败归因框架。通过把扁平日志重构为一个分层因果图，CHIEF 借助虚拟神谕引导的回溯与反事实推理，实现了高效的自顶向下归因。在 Who&When 基准上的大量实验表明，CHIEF 在智能体级和步骤级准确率上都超越了八个基线。这些结果凸显了因果结构对于在冗长且无结构的 MAS 日志中实现有效失败归因的重要性。

---

## 局限性（Limitations）

一个主要的局限在于，CHIEF 的有效性依赖于分层因果图与虚拟神谕的保真度（fidelity）。上游的不准确（例如被幻觉出来的边）可能传播到最终的诊断中。此外，我们的评测目前局限于 Who&When 基准——这是当前唯一可用的公开数据集——因此有必要在未来于更广泛的系统上进行验证。不过，在这一基准上相对于八个有代表性基线的性能优势，已在相当程度上缓解了这些威胁（threats）。

其次，CHIEF 聚焦于识别单一的决定性根因，这与 Who&When 基准中的假设保持一致。我们的方法是否适用于累积式错误传播（cumulative error propagation）——即失败源自一连串细微偏离的累加、而非单一的灾难性错误——仍有待未来研究的验证。

---

## 参考文献（References）

（按原文要求，参考文献列表不翻译，此处略。完整文献见原文 arXiv:2602.23701。）

---

## 附录 A　基于 RAG 的任务分解细节（Details for RAG-based Task Decomposition）

我们详述用于任务分解的 RAG 组件。我们从一个参考知识库中检索语义相似的范例，并把它们作为分解原型注入提示，以鼓励生成可验证的子任务阶段。

该知识库由两个公开数据集构建而成：（1）GAIA，我们利用其中全部 165 个提供了显式步骤标注（Steps）的实例，作为分解模板；（2）AssistantBench，我们从中挑选 33 个高质量实例，其解释（explanation）字段包含丰富的隐式子目标（implicit sub-goals）和验证线索（verification trails），用作分解指南。在检索时，我们在初始检索阶段与最终提示插入阶段都使用固定数量的 2 个范例。我们在嵌入（embeddings）上计算余弦相似度（cosine similarity）以识别相关条目，同时排除任何源自当前评测任务的知识库实例，以防止数据污染（data contamination）。

**知识库构建（Knowledge Base Construction）。** 为了构建一个统一的知识库，我们把 GAIA 和 AssistantBench 数据集中的条目规范化为一种可检索的纯文本格式。具体而言，每个 GAIA 条目由问题（Question）与其推理步骤（Steps）拼接而成，而每个 AssistantBench 条目则由任务描述（Task）与其详细解释（Explanation）组合而成。这种一致的文本格式便于高效的检索与利用，如图 3 所示。

**（图 3：知识库构建说明与格式。）**
```
- GAIA 条目格式：{Question} + {Steps}
- AssistantBench 条目格式：{Task} + {Explanation}
```

**检索到的范例（Retrieved Example）。** 这些检索到的范例展示了不同基准在呈现任务及其推理过程时的不同风格惯例。来自 AssistantBench 的范例使用一个「Task」字段作为主查询，后接一个「Explanation」字段，其推理隐式地融入了子任务分解和工具使用建议。相比之下，GAIA 的范例使用一个「Question」字段作为任务描述，并显式地以带编号的「Steps」来以结构化、顺序化的方式勾勒子任务。格式上的这些差异有助于模型适应构造提示和推理轨迹的多样方式。检索到的范例展示在图 4 中。

**（图 4：注入提示中的检索范例。）**
```
[注入范例 1]
来源：AssistantBench
Task: Which gyms (not including gymnastics centers) in West Virginia are within 5 miles (by car) of the Mothman Museum?
Explanation: You can use Google Maps to find the Mothman Museum and then search nearby gyms within 5 miles, ...

[注入范例 2]
来源：GAIA
Question: A 5-man group made up of one tank, one healer, and three DPS is doing a dungeon ...
Steps: 1. Searched "WoW classes" on Google. 2. Opened ... 3. Identified the relevant classes ... 4. Listed the classes in alphabetical order ...
```

**完整提示（Full Prompt）。** 该提示旨在引导 LLM 执行基于 RAG 的任务分解，通过融入原始问题、真值、多智能体对话历史以及检索到的参考范例，从而确保所生成的子任务既语义有意义、又忠实于实际的执行轨迹。它进一步强制执行一个结构化的自我反思（self-reflection）过程——包含草稿优化、证据对齐和最终优化。完整的提示模板见图 5。

**（图 5：基于 RAG 的任务分解提示。以下为原文英文提示模板，按惯例保留原文。）**
```
You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real-world problem.
The problem is: {question}
The correct answer for the problem is: {ground_truth}
Here is the conversation in JSON format: {history_text}
There are total {len(history_text)} steps, each entry provides the agent output and its role.
Here is the retrieved reference example: {rag_text}
Based on this conversation and retrieved example, please decompose the reasoning into semantic subtasks.
You must perform a self-reflection process to optimize your decomposition:
1. Draft Optimization: propose an initial set of subtasks.
2. Evidence Alignment: ensure each subtask's step range aligns with the dialogue.
3. Final Optimization: ensure step ranges are continuous, cover all steps, and do NOT overlap.
```

**（图 6：分层因果图的一个示例。其中 (a) 子任务节点；(b) 智能体节点；(c) 边。）**

---

## 附录 B　OTAR 解析提示（Prompt for OTAR Parsing）

该提示旨在引导 LLM 把多智能体执行轨迹解析为针对预定义子任务内每个智能体的结构化 OTAR（观察、思考、动作、结果）元组，从而能够严格地刻画智能体行为，这是对原始 TAR 模式的扩展。它强制执行一种严格的输出格式，要求子任务名称的精确匹配、识别每个子任务步骤范围内的活跃智能体，以及把它们的观察、思考、动作和结果直接从对话历史中抽取出来并做详细拆解。提示本身采用一种高度受约束的、模板驱动的结构，配有清晰的分节指令，以确保跨所有子任务都能产出一致且可解析的 OTAR 标注。OTAR 解析的提示模板见图 7。

**（图 7：OTAR 解析提示。以下为原文英文提示模板。）**
```
You are an AI assistant tasked with analyzing multi-agent execution traces.
The problem is: {question}
The correct answer for the problem is: {ground_truth}
Here is the conversation in JSON format: {history_text}
There are total {len(history_text)} steps, each entry provides the output of the agent and its role.
Below are the subtasks: {subtasks_text}
Your job:
1. For each subtask, identify the agents that actively perform actions within its step_range.
2. Summarize each agent's behavior into Action / Observation / Thought / Result.
For EACH subtask, you MUST answer in the following strict format:
The Subtask Name: (must exactly match one of the given subtask names)
Agents:
- Agent: <agent_name>
-- Action: <what this agent did>
-- Observation: <what they saw>
-- Thought: <their reasoning>
-- Result: <their output>
(repeat '- Agent' blocks if multiple agents in this subtask)
```

---

## 附录 C　边构建提示（Prompt for Edge Construction）

**子任务 & 智能体边提示（Subtask & Agent Edge Prompt）。** 为了构建多层级因果图的结构化依赖，该提示专门设计用于生成子任务边和智能体边。其核心机制在于定义具体的边类型和反事实模式，显式地勾勒子任务之间的逻辑演进，以及子任务内智能体之间的协作与错误传播路径。子任务 & 智能体边构建的提示模板见图 8。

**（图 8：子任务 & 智能体边构建提示。以下为原文英文提示模板。）**
```
You are an expert in causal reasoning and multi-agent task analysis.
The problem is: {question}
The correct answer for the problem is: {ground_truth}
Here is the conversation in JSON format: {history_text}
There are total {len(history_text)} steps, each entry provides the output of the agent and its role.
Below are the subtasks with their agents: {subtasks_agents_text}
Your job:
1. Construct causal edges ONLY for consecutive subtask pairs: (S1->S2), (S2->S3), ... (subtask-subtask edges)
2. For each subtask, construct causal edges BETWEEN agents inside this subtask only (no cross-subtask agent edges)
3. Use the agent-level DAG to describe how observations, reasoning, and decisions flow from one agent to another
For EACH subtask-subtask edge (consecutive pairs) and EACH agent-agent edge (per subtask),
you MUST output ONE block in the following exact format:
From: <upstream subtask id (e.g., S1) or upstream agent name>
To: <downstream subtask id (e.g., S2) or downstream agent name>
Type: <subtask edge type: data_dependency / logical_prereq; or agent edge type: obs_dependency / reasoning_continuation / decision_dependency / environment_feedback / memory_ref / loop_control>
Counterfactual_Patterns:
- Bias: ...
  Anomaly: ...
Guidelines:
- You may output zero, one, or multiple Failure Modes items for each edge
- Output all subtask edges first, then agent edges by subtask
Additional Guidelines for Counterfactual Pattern Construction:
1. Counterfactual patterns must correspond to 4 attribution scenarios: Local (whether the error of the current edge is caused by itself), Planning-Control (planner/executor responsibility in control flow/loops), Data-Flow (the starting point of data corruption in data streams), and Deviation-Aware (whether the deviation is reversible).
2. For Subtask/Agent edges, explicitly bind the potential bias of the upstream node (Bias) to the observable anomaly of the downstream node (Anomaly), e.g., "If the Thought of the upstream Agent has hallucinations (Bias), the Result of the downstream Agent will have data errors (Anomaly)".
3. When constructing Planning-Control counterfactuals, distinguish between planner responsibility (repeating failed strategies) and executor responsibility (execution anomalies under valid strategies).
4. When constructing Data-Flow counterfactuals, associate the consistency corruption node of specific data items.
5. When constructing Deviation-Aware counterfactuals, mark whether the deviation is reversible (whether subsequent steps self-correct).
```

（图 8 提示中文要点：反事实模式须对应 4 种归因场景——局部（当前边的错误是否由自身引起）、规划-控制（控制流/循环中规划者/执行者的责任）、数据流（数据流中数据污染的起点）、面向偏离（偏离是否可逆）；对子任务/智能体边须把上游节点的潜在偏差 Bias 显式绑定到下游节点的可观察异常 Anomaly。）

**步骤边提示（Step Edge Prompt）。** 为了用具体的数据证据补充抽象的因果模式，该提示专门设计用于构建步骤边。其核心机制在于严格匹配上游步骤的输出数据与下游步骤的输入数据，显式地捕捉执行步骤之间具体的数据流，从而为因果回溯提供确定性的证据线索。步骤边构建的提示模板见图 9。

**（图 9：步骤边构建提示。以下为原文英文提示模板。）**
```
You are an AI assistant tasked with analyzing multi-agent execution traces.
The problem is: {question}
The correct answer for the problem is: {ground_truth}
Here is the conversation in JSON format: {history_text}
There are total {len(history_text)} steps, each entry provides the output of the agent and its role.
Below are the subtasks: {subtasks_text}
Your job:
1. Identify step-level edges (meaningful data passing between steps)
2. For each step edge, identify the upstream step (data producer) and downstream step (data consumer)
You MUST answer in the following strict format:
- Upstream: <integer step id where the data is produced>
  output_data: "short description of the data (e.g., 'distance=30' or 'parsed_goal')"
  data_type: "text/numeric/list/boolean"
  step_id: <step id in the upstream step>
  agent_id: <agent id in the upstream step>
- Downstream: <integer step id where the data is used>
  input_data: "short description of the data (e.g., 'distance=30' or 'parsed_goal')"
  data_type: "text/numeric/list/boolean"
  step_id: <step id in the downstream step>
  agent_id: <agent id in the downstream step>
Guidelines:
- Step Edges should capture meaningful data passing from upstream to downstream step
- You may output zero, one, or multiple Step Edges items
```

---

## 附录 D　神谕合成提示（Prompt for Oracle Synthesis）

为了给分层回溯生成中间监督信号，该提示专门设计用于合成子任务虚拟神谕。其核心机制是利用先前已生成神谕的约束以及后续尚未处理轨迹的上下文，通过顺序生成（sequential generation）和全局自检（global self-checking）来为每个子任务构建结构化且内部一致的验证准则。神谕合成的提示模板见图 10。

**（图 10：神谕合成提示。以下为原文英文提示模板。）**
```
You are an AI assistant tasked with synthesizing a complete set of Subtask Virtual Oracles for hierarchical failure diagnosis in a multi-agent trajectory.
The problem is: {question}
The correct answer for the problem is: {ground_truth}
Here is the conversation in JSON format: {history_text}
There are total {len(history_text)} steps, each entry provides the output of the agent and its role.
Here is the retrieved reference example: {rag_text}
Given subtask plan produced by the decomposition stage: {subtasks}
Your goal: Generate ALL virtual oracles for ALL subtasks.
IMPORTANT thinking rule:
- You must synthesize oracles in subtask order: k = 1..K.
- While generating the oracle for the current subtask, treat all previously generated oracles as "previous oracle constraints" (internal memory). Use them to ensure consistency, avoid contradictions, and define valid preconditions.
You MUST perform a self-check process:
1) Draft All Oracles (sequential):
- For k=1..K, draft the oracle using the Problem, the retrieved reference example, the subsequent unprocessed trajectory, and the previously generated oracle constraints.
2) Global Consistency Check:
- No oracle contradicts the problem instruction or earlier oracles.
- Precondition must only depend on information that is available before or at the beginning of this subtask (i.e., upstream outputs, environment states, or outcomes implied by earlier oracles).
- Key Evidence must include only essential evidence that should be checked during this subtask.
- Acceptance Criteria must be checkable after execution and falsifiable.
3) Finalize All Oracles.
Output format:
For each subtask, output exactly the following block, repeated K times in order:
-Subtask Name: <copy exactly from subtask plan>
-Oracle:
  Goal: <what this subtask should achieve>
  Precondition: <each item must reference upstream outputs/environment states only>
  Key Evidence: <critical facts/claims/tool-return fields/intermediate quantities to verify>
  Acceptance Criteria: <checkable post-hoc; define pass/fail>
```

---

## 附录 E　分层回溯提示（Prompt for Hierarchical Backtracking）

为了实现从粗到细粒度的故障定位，该提示专门设计用于执行分层回溯。其核心机制是利用因果图与虚拟神谕，在三个层级（子任务、智能体、步骤）上依次进行语义评估与比较，逐步收窄候选错误节点，从而精确定位根因。分层回溯的提示模板见图 11。

**（图 11：分层回溯提示。以下为原文英文提示模板。）**
```
You are an AI assistant tasked with analyzing a multi-agent conversation solving a real-world problem and performing hierarchical failure backtracking.
The problem is: {question}
The correct answer for the problem is: {ground_truth}
Here is the conversation in JSON format: {history_text}
There are total {len(history_text)} steps, each entry provides the output of the agent and its role.
Here is the causal graph describing the hierarchical reasoning structure: {graph}
Your job:
1) Subtask-level backtracking:
- Traverse subtasks in reverse topological order according to the subtask edges in the graph. For each subtask, determine its actual execution output from the conversation slice within its step range. Compare the actual execution output against this subtask's oracle Goal and Acceptance Criteria. Internally make a binary discrepancy decision (0/1). If discrepancy=1, include this subtask in the candidate error subtasks set.
2) Agent-level backtracking:
- For each candidate subtask, evaluate each constituent agent. Compare the agent OTAR against the oracle Preconditions and Key Evidence of that subtask. Internally make a binary discrepancy decision (0/1). If discrepancy=1, include this agent in the candidate error agents set.
3) Step-level backtracking:
- For each candidate agent within a candidate subtask, evaluate its executed steps in that subtask.
- For each step, extract concrete execution details (prioritize tool input/output, intermediate computed values, cited facts, or explicit variable references).
- Cross-check the step's execution details against:
  (i) the agent's OTAR summary
  (ii) the full oracle checklist of this subtask (Goal, Preconditions, Key Evidence, Acceptance Criteria).
- Internally make a binary discrepancy decision (0/1). If discrepancy=1, include this step as a candidate error step.
Now, please strictly follow this output format:
Candidate Error Subtasks: [Id1, Id2, ...]
Candidate Error Agents: [agent1, agent2, ...]
Candidate Error Steps: [step_Id1, step_Id2, ...]
```

---

## 附录 F　反事实归因细节（Details for Counterfactual Attribution）

本附录为规划-控制归因提供两个示意性示例。两个示例都呈现出循环式失败（cyclic failures），但责任归属不同：一个归因于规划者（没有有意义的策略更新），另一个归因于执行者（有有效的策略转变，但执行异常）。

**（图 12(a)：归咎于规划者的示例。）** 在第 4 步，规划者发出了一个正确的意图：查询关于版本 v2 的信息。WebSurfer 随后检索到了关键证据——v1 已被弃用（deprecated），并提供了指向 v2 的链接。在正确的控制流下，规划者此时本应指示执行者去跟进这个 v2 链接。然而在第 6 步，规划者仍然执着于 v1，生成了一个偏离的指令，要求继续围绕 v1 进行搜索，这把轨迹拖入了循环行为，并产生了多个后续的循环组。在这些重复中，尽管反复收到失败信号，规划者仍持续地发出语义上等价的「探索 v1」的思考或命令，表明缺乏有效的策略更新。最终，WebSurfer 被迫产出了一个基于 v1 的错误答案。因此，本案例满足我们「在收到失败反馈后仍坚持一个失败的方法」的判据，我们把根因归于**规划者责任（Planner Responsibility）**。

**（图 12(b)：归咎于执行者的示例。）** 在第 10 步，规划者显式地发出了一个查询指令，并强调了完整的约束集合：「24 小时 + 自助打印 + 目标区域」。然而在第 11 步，WebSurfer 忽略了「自助」（self-service）这一约束，返回了一个无效的候选。作为回应，在第 12 步，规划者识别出了这一疏漏并重申了约束；尽管如此，在第 13 步，WebSurfer 又重复了类似的执行偏离，把轨迹推入了又一个循环，形成了多个循环组。与图 12(a) 不同，规划者反复检测到错误，并提出了旨在打破循环的合理策略转变，而执行者却持续地忽略或误解约束的某些部分，从而不断产出异常结果。因此，本案例符合我们「规划有效但执行异常」的判据，我们把根因归于**执行者责任（Executor Responsibility）**。

此外，反事实归因的提示模板系统性地整合了四种核心归因策略：它首先采用局部归因来判定错误是否源自局部。若不是，它接着利用规划-控制归因来剖析循环行为中的责任，并利用数据流归因来回溯到数据污染的源头。最后，面向偏离的归因充当一个有效性过滤器，排除那些之后被系统自我纠正的瞬态偏离。反事实归因的完整提示模板见图 13。

**（图 13：反事实归因提示。以下为原文英文提示模板。）**
```
You are an AI assistant tasked with analyzing a multi-agent conversation solving a real-world problem and performing counterfactual failure attribution.
The problem is: {question}
The correct answer for the problem is: {ground_truth}
Here is the multi-agent conversation: {history_text}
There are total {len(history_text)} steps, each entry provides an agent's output.
Here is the structured candidate_set: {candidate_set}
Here is the graph: {dag_graph}
Your job is to identify the SINGLE most responsible reasoning mistake that directly leads to the wrong final result.
You MUST follow this internal reasoning procedure over the candidate_error_steps:
Stage A Local Attribution (local vs upstream propagation):
- For each candidate step x:
- Use the graph's predecessor relations and edge-attached counterfactual patterns to judge whether there exists any upstream step that can causally explain the anomaly at x under the oracle constraints.
- If no upstream causal trigger can explain x AND x received valid inputs yet produced an incorrect output, treat x as a strong local-origin root-cause candidate.
Stage B Planning-Control Attribution (control / loop responsibility):
- Use loop information from the graph (e.g., loop groups and entry/internal/exit roles) to analyze whether the failure is caused by redundant cyclic behavior.
- Distinguish planner vs executor responsibility:
- Planner responsibility: despite repeated error signals, the planner repeats semantically identical thoughts/commands and fails to adapt strategy.
- Executor responsibility: the planner proposes valid strategy shifts, yet execution still yields abnormal results.
Stage C Data-Flow Attribution (data dependency responsibility):
- Use data-flow information in the graph to trace how key data items are produced and consumed.
- Decide whether the candidate step:
- fabricates data (no upstream basis),
- misinterprets upstream data,
- misuses otherwise correct data.
- If the error propagates via data, prefer attributing responsibility to the earliest step where valid upstream inputs are first corrupted into an abnormal result.
Stage D Final Screening (Reversibility and Irrecoverability):
- Deviation-aware filter (reversibility): Check whether the deviation introduced by a candidate step is later self-corrected such that oracle constraints are re-satisfied (system returns to a valid state). If the deviation is reversible (self-corrected later), assign it minimal responsibility compared to irreversible deviations.
- Irrecoverability tie-break (first point blocking recovery): Prefer the candidate step that first makes it hard or impossible to restore the correct reasoning path through conventional means, rather than the earliest deviation.
Final decision rule;
- Select ONE step that best satisfies: (1) true origin (local-origin or earliest data corruption) AND (2) passes Final Screening (irreversible and/or first irrecoverable point) AND (3) strongest downstream impact / irrecoverability.
- After deciding the single most responsible step:
- Determine the agent name at that step (from candidate_set and conversation context).
- Determine the exact step index (step_id).
- Provide a concise explanation (2-3 sentences) that explicitly mentions which attribution stages (Local / Planning-Control / Data-Flow / Final Screening) support this choice.
Now, please answer in this exact plain text format:
Agent Name: (your final prediction, a single agent name)
Step Number: (your final prediction, a single integer step id)
Reason for Mistake: (your explanation, summarize in 2-3 sentences)
No special symbols, no extra commentary.
```

（图 13 提示中文要点：对每个候选错误步骤依次走 4 个阶段——A 局部归因判断异常是局部产生还是上游传播；B 规划-控制归因区分规划者责任（反复重复语义相同的思考/命令、不调整策略）与执行者责任（规划者给出有效策略转变但执行仍异常）；C 数据流归因追踪关键数据的产生与消费，判断候选步是「捏造数据/误解上游数据/误用本来正确的数据」，并优先归因到最早把有效上游输入污染成异常结果的步骤；D 最终筛查含可逆性过滤（被后续自我纠正的可逆偏离责任最小）与不可恢复性 tie-break（优先选第一个使正确推理路径难以通过常规手段恢复的步骤）。最终只输出单个最负责的智能体名、步骤号和 2-3 句理由。）

---

*（全文翻译完）*
