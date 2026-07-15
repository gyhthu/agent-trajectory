# 【全文翻译】Def-DTS: Deductive Reasoning for Open-domain Dialogue Topic Segmentation（演绎推理做开放域对话主题分割）

> 原文 arXiv: 2505.21033  https://arxiv.org/abs/2505.21033
> 译者：lian-server(claude)  译于 2026-06-25
> 本译文为团队内部归档，仅供学习参考；以原文为准。

---

## 标题

**Def-DTS：用于开放域对话主题分割的演绎推理（Def-DTS: Deductive Reasoning for Open-domain Dialogue Topic Segmentation）**

## 作者与机构

Seungmin Lee¹, Yongsang Yoo¹·², Minhwa Jung¹·³, Min Song¹·⁴（通讯作者）

- ¹ 延世大学（Yonsei University）
- ² 乐天 INNOVATE（LOTTE INNOVATE）
- ³ LG 电子（LG Electronics）
- ⁴ Onoma AI

邮箱：
- ¹ {elplaguister, 4n3mone, minalang, min.song}@yonsei.ac.kr
- ² yongsang.yoo@lotte.net
- ³ minalang.jung@lge.com
- ⁴ min.song@onomaai.com

> 脚注：本文是已被 ACL 2025 Findings 录用的 camera-ready 版本的预印本，正式版将出现在 ACL Anthology。

---

## 摘要（Abstract）

对话主题分割（Dialogue Topic Segmentation, DTS）的目标是把对话切分为若干内部连贯的片段（segment）。DTS 在各种 NLP 下游任务中发挥着关键作用，但长期受困于几个慢性问题：数据短缺、标注歧义，以及近期所提解决方案不断增加的复杂度。另一方面，尽管大语言模型（Large Language Models, LLMs）与推理策略（reasoning strategies）已取得长足进步，它们却很少被应用到 DTS 上。

本文提出 **Def-DTS：用于开放域对话主题分割的演绎推理（Deductive Reasoning for Open-domain Dialogue Topic Segmentation）**，它利用基于 LLM 的多步演绎推理（multi-step deductive reasoning）来提升 DTS 性能，并借助中间结果实现案例分析（case study）。我们的方法采用结构化提示（structured prompting）来完成：双向上下文摘要（bidirectional context summarization）、话语意图分类（utterance intent classification），以及演绎式的主题切换检测（deductive topic shift detection）。在意图分类过程中，我们提出了一个可泛化的意图列表（generalizable intent list），用于做领域无关（domain-agnostic）的对话意图分类。

在多种对话场景下的实验表明，Def-DTS 始终优于传统方法和最新的（state-of-the-art）方法，且每一个子任务都对性能提升有贡献，尤其是在降低第二类错误（type 2 error，即漏检）方面。我们还探索了自动标注（auto-labeling）的潜力，强调了 LLM 推理技术在 DTS 中的重要性。

> 脚注：我们的代码与提示词已公开：https://github.com/ElPlaguister/Def-DTS

---

## 1 引言（Introduction）

（图 1：一段对话中主题切换的示例。触发主题切换的线索用红色高亮标出。）

对话主题分割（DTS）是一项旨在把一段对话切分为若干片段的任务，其中每个片段聚焦于一个连贯的主题。图 1 展示了单段对话内部主题切换的一个例子。DTS 对各种自然语言处理（NLP）任务都至关重要，包括：回复预测（response prediction）（Lin 等, 2020; Xu 等, 2021b; He 等, 2022）、回复生成（response generation）（Li 等, 2016; Xu 等, 2021a; Liu 等, 2022）、对话状态追踪（dialogue state tracking）（Das 等, 2024）、摘要（summarization）（Bokaei 等, 2016; Chen 与 Yang, 2020; Qi 等, 2021; Zhong 等, 2022）、问答（question answering）（Yoon 等, 2018; Zhang 等, 2022），以及机器阅读理解（machine reading comprehension）（Ma 等, 2024）。

尽管关注度日益增长，DTS 仍然受困于若干慢性难题。第一，标注数据的短缺使得多数近期 DTS 研究只能走无监督（unsupervised）路线，而这通常只能得到次优的性能。第二，片段标注上的歧义阻碍了有效方法的发展。最后，近期的研究——如 DialSTART（Gao 等, 2023）、UR-DTS（Hou 等, 2024）——分别在前人工作 CSM（Xing 与 Carenini, 2021）和 DialSTART（Gao 等, 2023）的基础上提出了增量式（incremental）方法，但这些方法需要更多参数、更高复杂度。这一演进路径说明，DTS 是一个具有挑战性、且常常被低估的问题。

正当 DTS 在自身的复杂性中挣扎时，NLP 领域则随着大语言模型（LLMs）与推理方法论的兴起而取得了显著进展。然而，即便考虑到这些 LLM 强大的问题求解能力以及 DTS 所带来的挑战，推理策略却很少被应用到 DTS 领域。原因在于，DTS 在 NLP 中长期被当作一个轻量级的子任务来对待。尽管如此，随着 AI 驱动的聊天服务的兴起，对更先进 DTS 模块的需求正在增长。具备推理能力的 LLM 恰好适合满足这一需求，使得基于 LLM 的 DTS 成为一条可行的解决路径。

为了给主题切换建立清晰的判定标准、并简化复杂的子任务，我们提出 Def-DTS：一种用于开放域对话主题分割的演绎推理方法，使用基于 LLM 的多步推理。Def-DTS 采用结构化提示，引导 LLM 在话语级别（utterance level）依次完成：双向上下文摘要、话语意图分类，以及演绎式主题切换检测，并着重强调领域无关的意图分类。

为了评估 Def-DTS，我们在三个对话数据集上进行了测试，覆盖开放域（open-domain）与任务导向（task-oriented）两种设定，并使用三项关键指标。我们的方法以显著优势持续超越传统方法以及最新的无监督、监督（supervised）和基于提示（prompt-based）的技术。消融实验（ablation studies）证实，每个子任务都能提升整体性能，其中中间步骤的意图分类尤其能改善真正例（true-positive）检测。最后，我们探索了基于 LLM 的 DTS 自动标注。

我们的贡献有四点：

- 我们**首次**将 LLM 推理技术引入 DTS，把前人方法论中的洞见整合成一个连贯且具有演绎性的提示设计。

- 我们把 DTS 重新表述（reformulate）为一项话语级的意图分类任务——将意图分类作为多步推理过程的核心组件来实现，从而支持灵活、任务无关（task-agnostic）的提示。

- 我们的方法在几乎所有对比基线上都实证地展现出更优的性能，凸显了提示工程（prompt engineering）在 DTS 中的有效性。

- 通过对本方法推理结果的深入分析，我们揭示了 LLM 推理在 DTS 中面临的挑战，并讨论了把 LLM 用作 DTS 自动标注器（auto-labeler）的可能性。

---

## 2 相关工作（Related Works）

### 2.1 对话主题分割（Dialogue Topic Segmentation）

对话主题分割把对话切分为连贯的主题单元。由于标注数据集有限，研究者们尽管面对其复杂性，仍大多依赖无监督方法（Xing 与 Carenini, 2021）。早期方法如 TextTiling（Hearst, 1997）通过词汇相似度来检测主题切换，后来用基于嵌入（embedding-based）的方法做了改进（Song 等, 2016）。

近期研究强调主题连贯性（topical coherence）与相似度打分（similarity scoring）。CSM（Xing 与 Carenini, 2021）利用基于 BERT 的连贯性建模，而 Dial-START（Gao 等, 2023）引入 SimCSE（Gao 等, 2021）来度量主题相似度。SumSeg（Artemiev 等, 2024）通过摘要抽取关键信息，并施加平滑（smoothing）来处理主题波动。UR-DTS（Hou 等, 2024）则通过改写话语来恢复缺失的指代（references），从而增强分割效果。尽管有这些进展，数据稀缺和性能局限仍然存在。

为应对这一点，SuperDialSeg（Jiang 等, 2023）引入了一种监督式方法，使用大规模的 DGDS 数据集（Feng 等, 2020, 2021）。然而，可用于开放域对话的数据集仍然有限。LLM 也在影响 DTS。S3-DST（Das 等, 2024）把结构化提示用于对话状态追踪与分割，但缺乏对多样化 DTS 设定的通用适用性。

### 2.2 LLM 推理时的推理策略（Reasoning Strategy at LLM Inference）

LLM 的进步（Brown 等, 2020）催生了围绕"系统 2 推理"（System 2 reasoning）（Kahneman, 2011）整合的研究，包括上下文学习（in-context learning）（Brown 等, 2020）和思维链提示（chain-of-thought prompting）（Wei 等, 2022）。这些技术让 LLM 能够处理复杂任务，例如符号数学（symbolic mathematics）（Yang 等, 2024）、检索增强生成（retrieval-augmented generation, RAG）（Lewis 等, 2020）和数据生成（data generation）（Adler 等, 2024）。研究表明，中间推理步骤能显著提升多跳推理（multi-hop reasoning）（Wang 等, 2023）和数学问题求解（Imani 等, 2023）等领域的性能。在此基础上，我们把基于 LLM 的推理整合进 DTS，借助其潜力来提升这一本质上复杂的任务中的分割准确率。

---

## 3 Def-DTS

（图 2：本方法 Def-DTS 的提示配置与整体流程。(a) 我们使用一个包含意图专属示例的通用意图列表，以实现领域无关的归类。(b) 我们采用 XML 结构化的输入输出格式，以稳定地提供对话内容。(c) 我们指示 LLM 在一次推理中对每条话语执行多步推理。）

### 3.1 整体流程（Overall Flow）

我们的方法（图 2）以一个结构化的提示格式，对每条话语应用多个子任务。提示模板（图 2b）包含四部分：合法标签列表（valid label list）、任务描述（task description）、输出格式（output format）和输入（input）。如图 2c 所示，它由三个主要子任务组成：(i) 双向上下文抽取（bidirectional context extraction）、(ii) 话语意图分类（utterance intent classification）、(iii) 演绎式主题切换分类（deductive topic shift classification）。每个子任务都以演绎的方式执行，进一步细节见算法 1（Algorithm 1）。

**算法 1 Def-DTS**

```
输入定义：
1: ClassifyIntent —— 给定一条话语及其上下文和意图池，对该话语的意图进行分类
2: Summarize —— 对给定的对话上下文进行摘要
3: 意图池 X = {x1, x2, …, xm}
4: 结构化对话 D = {di}, i=1..N，其中 di = {ui, si}（话语 ui 和说话人 si）
5: 结果 R = {ri}, i=1..N，其中 ri = {Pi, Qi, Xi, Ti}
6:       （前文上下文 Pi、后文上下文 Qi、意图 Xi、主题切换标签 Ti）

7: R ← ∅，S ← StructureDialogue(D)
8: for i ← 1 to N do
9:     Pi ← ExtractContext(D, max(1, i-2), i)        # 抽取前文上下文（前 2 条）
10:    Qi ← ExtractContext(D, i+1, min(i+3, N))      # 抽取后文上下文（后 3 条）
11:    Xi ← ClassifyIntent(D[i], Pi, Qi, X)          # 分类当前话语意图
12:    Ti ← ClassifyTopicShift(Xi)                   # 由意图演绎主题切换
13:    ri ← {Pi, Qi, Xi, Ti}
14:    R ← R ∪ {ri}
15: end for
16: return R

18: function ClassifyTopicShift(x)
19:    return x ∈ {"introduce_topic", "change_topic"} ? "YES" : "NO"
20: end function

22: function ExtractContext(D, start, end)
23:    context ← Summarize(D[start:end])
24:    return "U_start - U_end", context
25: end function
```

### 3.2 结构化格式（Structured Format）

我们以先前的 DST（对话状态追踪）研究（Das 等, 2024）为基准，采用 XML 格式的结构化模板。提示的格式化能通过提供结构化、定义良好的格式来增强对标注指令的遵循度，从而改善与任务的对齐、便于解析（parsing），进而减少后处理（post-processing）的需求（Das 等, 2024）。我们相信，我们这套演绎式、多步骤的方法同样能从这些优势中受益——增强与标注指令的对齐、减少后处理工作量。

为此，我们使用基于 XML 的结构化提示来标准化 LLM 的输出，以提升解析效率、最小化后处理。输入模板（图 2b）把话语组织在 `<Ux>` 元素中，每个元素包含说话人信息和话语内容。输出（图 2c）遵循同样的结构化格式，确保标注的一致性。完整模板见附录 8（即附录 A.1 的提示模板）。

### 3.3 双向上下文抽取（Bidirectional Context Extraction）

在 Def-DTS 的第一阶段，我们指示 LLM 为每条话语分别对其前文和后文对话进行摘要。考虑双向上下文（bidirectional context）是许多方法中的常用做法，例如 BERT 架构（Devlin 等, 2019），在无监督设定中也被频繁采用（Gao 等, 2023; Hou 等, 2024），这一策略已被证明对理解对话上下文行之有效。虽然 Das 等（2024）只抽取前文上下文以防止上下文遗忘（contextual forgetting），但我们对此做了改进：同时抽取前文和后文上下文，以实现上下文感知（context-aware）的对话主题分割，并防止上下文遗忘。

我们选用固定窗口大小（fixed window size），以确保在没有预先定义片段的无监督和真实环境中的适用性——当片段边界不可得时，这是一种稳健的做法。如图 2c(i) 所示，我们指示模型用 `<preceding_context>` 标签摘要不超过 2 个前序轮次（turns），用 `<subsequent_context>` 标签摘要不超过 3 个后续轮次。窗口大小取 -2:-1（前文）和 1:3（后文），是为了在上下文信息量与 token 效率之间取得平衡，这一做法与 Gao 等（2023）所用的方法类似。通过对每段上下文范围做摘要，我们既节省 token，又能保留细微的主题关系。

每个元素由 `<range> … </range>`（用于界定摘要范围）和 `<context>…</context>`（实际摘要内容）组成。这样的摘要方式既能捕捉相关对话，又能保持上下文简洁。我们的实验表明，这一做法能有效地帮助观察细微的主题关系。

### 3.4 话语意图分类（Utterance Intent Classification）

**表 1：用于开放域对话的话语意图列表。我们用这个列表对话语进行归类，并据此演绎地分类主题切换标签。**

| 意图（Intent） | 描述（Description） |
|---|---|
| JUST_COMMENT（仅评论） | 对前文进行评论，而没有任何提问。**不是**主题切换 |
| JUST_ANSWER（仅回答） | 回答前一条话语。**不是**主题切换 |
| DEVELOP_TOPIC（展开主题） | 把对话展开到相似且包含性的子主题。**不是**主题切换 |
| INTRODUCE_TOPIC（引入主题） | 引入一个相关但不同的主题。**是**主题切换 |
| CHANGE_TOPIC（更换主题） | 彻底更换话题。**是**主题切换 |

由于 DTS 数据集存在歧义问题，单纯给出主题分类的描述不足以传达主题变化的精确定义。因此我们提出：用一个定义良好、互相区分的标签列表来对话语进行分类（类似于意图分类），将对 DTS 有益。尽管大多数意图分类任务都是在任务导向对话（Task Oriented Dialogue, TOD）（Liu 与 Lane, 2016; Chen 与 Luo, 2023; Fong 与 Ong, 2023）的语境下进行的，但 TOD 通常涉及特定领域的数据集，难以泛化到其他领域或更开放式的对话。

作为应对这一问题的途径，我们注意到 Xie 等（2021）在其标注指南中识别出了五种会话回应模式（patterns of conversational responses）。这些模式反映了话语的自然特征，从而能对各种形式的对话进行意图分类。详细的意图模式与描述见表 1。

受这项研究启发，我们指示模型通过话语意图分类来检测主题变化。具体而言，在完成双向上下文抽取之后，如图 2c(ii) 所示，模型在考虑此前生成的双向上下文的基础上，把当前话语归类为预定义通用意图池（图 2a 上方框）中的某个意图。

此外，我们通过为每个意图提供示例对话来增强模型对意图的理解，如图 2a 中"通用意图池（General intent pool）"所示。这些意图专属的示例为模型提供了检测主题变化的有用指引，使其能在后续子任务——演绎式主题切换——中推导出结果。同时，为了通过来自传统文本分割方法的统计频率分析来证明我们意图标签的重要性（见 5.5 节），我们做了相应的语言学检验。使用这一技术，我们在各种对话设定下都观察到了显著的性能提升，并能对每个意图进行细致分析。意图池构建的细节见附录 A.3–A.5。

### 3.5 演绎式主题切换分类（Deductive Topic Shift Classification）

最后，如图 2c(iii) 所示，模型基于上一步意图分类结果中的演绎指引，预测主题是否正在发生变化。这一任务以"强制（enforced）"的方式处理：模型依据表 1 中每个意图描述末尾的说明（即"不是主题切换"或"是主题切换"），从上一步的意图分类结果出发，演绎地输出预先确定的标签。

我们之所以明确指示模型执行这一步，有两个原因。第一，它简化了解析过程，使任务易于处理。第二，它确保模型显式地输出主要目标的结果，使其在逐一完成各子任务的过程中，始终聚焦于主要目标（DTS）。

---

## 4 实验（Experiments）

### 4.1 数据集（Datasets）

**表 2：对话主题分割数据集统计。**

| 数据集 | 样本数 | 每段对话话语数(avg) | (min) | (max) | 每段对话片段数(avg) | 片段平均长度(avg.len) |
|---|---|---|---|---|---|---|
| TIAGE | 100 | 15.6 | 14 | 16 | 4.2 | 3.8 |
| SuperDialseg | 1322 | 12.1 | 7 | 19 | 4.0 | 3.0 |
| Dialseg711 | 711 | 26.2 | 7 | 47 | 4.9 | 5.4 |

我们在三个数据集上评估本方法：TIAGE、SuperDialseg、Dialseg711，以验证本方法在开放域和任务导向两种设定下的性能。数据集统计见表 2。

**TIAGE**（Xie 等, 2021）是唯一公开可得的、带主题分割的日常对话数据集，源自 PersonaChat（Zhang 等, 2018），专为建模开放域对话中的主题切换而设计。

**SuperDialseg**（Jiang 等, 2023）是一个大规模对话分割数据集，基于文档对齐（document-grounded）的语料构建，提供了一个用于识别基于文档的对话中分割点的框架。

**Dialseg711**（Xu 等, 2021b）是一个真实对话数据集，由 MultiWOZ（Budzianowski 等, 2018）和 Stanford Dialog Dataset（Eric 等, 2017）自动标注而来，方法是把不同主题的对话拼接起来，因此具有清晰的主题差异；由于其合成（synthetic）特性，片段边界处的连贯性很低。

### 4.2 评估指标（Evaluation Metrics）

与早期研究（Xing 与 Carenini, 2021; Jiang 等, 2023; Artemiev 等, 2024）一样，我们采用 Pk 误差（Beeferman 等, 1997）、WindowDiff（WD）误差（Pevzner 与 Hearst, 2002）和 F1 分数。Pk 误差通过一个在预测上滑动的窗口，统计是否存在错误分配的片段来计算。WD 误差通过在滑动窗口内比较金标准（gold labels）与预测的边界数量来计算。注意：Pk 或 WD 越低代表性能越高。

### 4.3 对比方法（Comparison Methods）

我们提出了一种基于提示工程的精细化 DTS 方法，并把它的性能与多种无监督、监督和基于 LLM 的方法进行对比。

首先，我们与一个随机基线（random baseline）对比，该基线根据随机选取的片段数量，任意分配片段边界。其次，我们与基于 TextTiling 的知名无监督学习方法对比，如连贯性打分模型（Coherence Scoring Model, CSM）（Xing 与 Carenini, 2021）、DialSTART（Gao 等, 2023）、SumSeg（Artemiev 等, 2024）。

我们也对比监督学习方法。我们选取了基础的 BERT 模型（Devlin 等, 2019）、先进且高性能的 RoBERTa 模型（Liu, 2019）、以及 RetroTS-T5（Xie 等, 2021）系统来做对比分析。

最后，我们把本方法与近期提出的基于 LLM 的方法对比。这些方法我们均使用 gpt-4o 来运行。我们选取了 SuperDialseg（Jiang 等, 2023）中所用的 PlainText 提示，以及 S3-DST（Das 等, 2024）的提示。每一项实际对比所用方法论的细节，在附录 D.1 中讨论。

### 4.4 实验结果（Experimental Results）

**表 3：三个数据集上的性能。由于 Dialseg711 数据集没有训练/验证划分，因此该数据集的监督学习部分无结果。每组方法中的最佳结果以粗体标出；所有方法中的最佳性能以红色文字标注（此处用【】标出）。**（↓ 越低越好，↑ 越高越好）

| 方法 | TIAGE Pk↓ | TIAGE WD↓ | TIAGE F1↑ | SuperDialseg Pk↓ | SuperDialseg WD↓ | SuperDialseg F1↑ | Dialseg711 Pk↓ | Dialseg711 WD↓ | Dialseg711 F1↑ |
|---|---|---|---|---|---|---|---|---|---|
| **无监督学习方法** | | | | | | | | | |
| Random | 0.526 | 0.664 | 0.237 | 0.494 | 0.649 | 0.266 | 0.533 | 0.714 | 0.204 |
| TextTiling | 0.469 | 0.488 | 0.204 | 0.441 | 0.453 | 0.388 | 0.470 | 0.493 | 0.245 |
| TextTiling+Glove | 0.486 | 0.511 | 0.236 | 0.519 | 0.524 | 0.353 | 0.399 | 0.438 | 0.436 |
| CSM | 0.400 | 0.420 | 0.427 | 0.462 | 0.467 | 0.381 | 0.278 | 0.302 | 0.610 |
| DialSTART | 0.482 | 0.528 | 0.378 | 0.373 | 0.412 | 0.627 | 0.179 | 0.198 | 0.733 |
| SumSeg | 0.482 | 0.496 | 0.075 | 0.479 | 0.485 | 0.119 | 0.477 | 0.483 | 0.070 |
| **监督学习方法** | | | | | | | | | |
| BERT | 0.418 | 0.435 | 0.124 | 0.214 | 0.225 | 0.725 | - | - | - |
| RoBERTa | 0.265 | 0.287 | 0.572 | **0.185** | **0.192** | **0.784** | - | - | - |
| RetroTS-T5 | 0.280 | 0.317 | 0.576 | 0.227 | 0.237 | 0.733 | - | - | - |
| **基于 LLM 的方法** | | | | | | | | | |
| Plain Text | 0.445 | 0.485 | 0.185 | 0.412 | 0.427 | 0.048 | 0.333 | 0.353 | 0.010 |
| S3-DST_uttr | 0.439 | 0.498 | 0.265 | 0.442 | 0.469 | 0.404 | 0.087 | 0.109 | 0.790 |
| **Def-DTS（本文）** | **【0.232】** | **【0.256】** | **【0.699】** | **0.315** | **0.324** | **0.686** | **【0.015】** | **【0.018】** | **【0.979】** |

实验结果见表 3。Def-DTS 在基于 LLM 的方法中始终展现出更优的性能，并在 TIAGE 和 Dialseg711 数据集上取得最新最优（state-of-the-art）结果——这两个数据集与我们"分析通用开放域对话"的目标高度契合。相比之下，其他基于 LLM 的方法不仅不如监督学习方法，甚至不如部分无监督方法，说明单纯使用 LLM 并不能保证成功。

**TIAGE**

与近期基于 LLM 的方法 S3-DST_uttr 相比，我们的方法在 Pk 和 WD 误差上都降低了 0.2 以上，同时 F1 分数提升超过 0.4，效果可观。此外，Def-DTS 在 TIAGE 上甚至超越了监督式方法，在所有指标上都领先 10% 以上，从而凸显了本方法在无需额外训练的情况下、即便在领域无关设定下，也能在各种对话环境中取得高性能的有效性。

**SuperDialseg**

我们的方法超越了所有无监督方法。尽管相比用监督学习训练的模型性能略低，但它在使用提示式技术的无监督方法中取得了最佳结果。

**Dialseg711**

我们的方法同样表现出更优的性能。值得注意的是，它超越了最强的基于 LLM 的方法 S3-DST_uttr，凸显了本方法的通用适用性和稳健性。

总体而言，在所有受测数据集上的这些一致提升，确认了本方法对多样开放域对话场景的稳健有效性——它不仅在无监督设定下表现出色，还超越了此前领先的基于 LLM 的方法。这进一步确立了本方法即便在具有挑战性的、领域无关条件下，也能交付高性能的潜力。

---

## 5 分析与讨论（Analysis and Discussion）

### 5.1 消融实验（Ablation Study）

**表 4：消融实验。**

| 方法 | TIAGE Pk↓ | TIAGE WD↓ | TIAGE F1↑ |
|---|---|---|---|
| w/o all（去掉全部子任务） | 0.295 | 0.333 | 0.605 |
| w/o intent（去掉意图分类） | 0.316 | 0.342 | 0.524 |
| w/o examples（去掉意图示例） | 0.287 | 0.308 | 0.617 |
| w/o context（去掉上下文抽取） | 0.263 | 0.296 | 0.682 |
| w/o bidirectional（去掉后文，仅前文） | 0.269 | 0.301 | 0.659 |
| **Def-DTS（完整）** | **0.232** | **0.256** | **0.699** |

为评估本方法各部分的贡献，我们做了消融实验，结果见表 4。

在 **w/o all**（去掉全部组件）情形下，模型被指示在没有上下文抽取、也没有意图分类的情况下检测主题切换。在 **w/o intent**（去掉意图）情形下，模型在对每条话语做上下文抽取之后再检测主题切换。我们观察到 w/o intent 的表现竟然比 w/o all 更差，说明仅依赖对话上下文来预测主题切换并不能得到最优性能。

在 **w/o examples**（去掉示例）情形下，这本质上就是 Def-DTS 但不给意图提供示例。w/o examples 的表现优于 w/o intent，说明在用于主题切换预测之前、先把上下文加工成意图，能带来显著优势。

在 **w/o context**（去掉上下文）情形下，模型被指示在对每条话语做意图分类之后再检测主题切换——这与 w/o intent 情形正好相反。这一结果表明：在合适示例的支持下，意图分类对单条话语的主题切换预测有显著影响。

在 **w/o bidirectional**（去掉双向，仅前文）情形下，上下文抽取步骤不考虑后文上下文。与完整的 Def-DTS 和 w/o context 情形相比，这一情形表现更低，凸显了考虑双向上下文对意图分类和主题切换检测的关键作用。

综上，Def-DTS 的每个模块都对性能提升有贡献，而当所有模块一起应用时，它们协同作用，带来可观的性能增长。

### 5.2 结构化格式的对比研究（Comparative Study for Structured Format）

**表 5：结构化格式的对比研究。**

| 输入/输出格式（I/O Format） | TIAGE Pk↓ | TIAGE WD↓ | TIAGE F1↑ |
|---|---|---|---|
| NL（自然语言） | 0.274 | 0.302 | 0.640 |
| JSON | 0.259 | 0.292 | 0.658 |
| XML | **0.232** | **0.256** | **0.699** |

为考察消融实验未覆盖的结构化 I/O 格式的影响，我们用三种不同格式——自然语言（Natural Language, NL）、JSON、XML——表达完全相同的提示，并在表 5 中对比性能。结果显示，结构化格式（XML 和 JSON）不仅在解析上有优势，在任务性能上也优于 NL。这些发现实证地支持了 Das 等（2024）提出的假设：XML 能在对话处理中提供结构上的好处。

### 5.3 意图分类准确率（Intent Classification Accuracy）

**表 6：TIAGE 基准上的意图级混淆矩阵。**（TP=真正例，FP=假正例，TN=真负例，FN=假负例，Acc=准确率）

| 意图 | TP | FP | TN | FN | Acc |
|---|---|---|---|---|---|
| JUST_COMMENT | 0 | 1 | 498 | 35 | 0.93 |
| JUST_ANSWER | 0 | 1 | 456 | 23 | 0.95 |
| DEVELOP_TOPIC | 0 | 0 | 119 | 47 | 0.71 |
| INTRODUCE_TOPIC | 189 | 68 | 0 | 0 | 0.73 |
| CHANGE_TOPIC | 21 | 6 | 0 | 0 | 0.78 |

由于我们的方法直接从话语意图演绎主题切换，因此分析意图分类结果至关重要。我们考察混淆矩阵（表 6），以识别模型难以处理的话语类型。由于没有关于意图的真值（ground truth），只提供了主题切换标签，因此正确与否由"预测的意图是否与主题切换标签一致"来判定。例如，如果一条话语本应是主题切换，却被分类为意图 JUST_COMMENT、主题切换为 NO，那么它计为一个假负例（False Negative, FN）。

大多数结果在主题切换分类上表现良好，只有两类例外：JUST_COMMENT 和 JUST_ANSWER 中的正例。这些发现表明：模型的主要难点在于把细微的主题差异判定为真正的切换——相比其他话语类型，这是导致性能下降的更显著因素。其他数据集（SuperDialseg、Dialseg711）的分析见附录 B。

### 5.4 意图级对比（Intent Level Comparison）

（图 3：(a) MATCHED INTENT（匹配意图）表示——仅在本方法判断正确的情形下，把话语按本方法的意图分类结果分组后，其他方法在这些分组上的准确率。(b) MISMATCHED CASE（错配情形）表示——仅在本方法判断错误的情形下，其他方法与本方法共同出错的计数。）

我们对比了各种方法在本方法所预测的不同意图类别上的表现，如图 3 所示。

在 (a) MATCHED INTENT 中：对于**不含**主题切换的话语，当本方法也正确时，其他方法约能达到 80–85% 的准确率。然而，对于**含**主题切换的话语，在本方法正确的情形下，其他方法的准确率下降到约 20–50%。

在 (b) MISMATCHED CASE 中：对于**不含**主题切换的话语，在本方法出错的情形下，其他方法仍能正确分类其中 50%。然而，对于**含**主题切换的话语，本方法漏掉的情形中，其他方法也有 80% 没能分类正确。

这表明：检测**真正发生主题切换**的话语，比检测不含主题切换的话语要困难得多。本方法在不含主题切换的情形下大约领先其他方法 20%，在含主题切换的情形下领先超过 40%。总而言之，虽然本方法在所有情形下都提升了准确率，但在处理含主题切换的话语时，提升幅度更大。

### 5.5 意图标签的语言学检验（Linguistic Test for Intent Labels）

为了证明意图标签对主题切换的影响，我们采用了统计语言学（statistical linguistics）的方法。传统文本分割使用停顿（pauses）、提示词（cue words）和指代名词短语（referential noun phrases）来识别边界（Passonneau 与 Litman, 1997）。Galley 等（2003）发现提示短语（cue phrases）与主题分割之间存在显著相关性。在此基础上，我们假设：诸如 "introduce topic（引入主题）" 和 "change topic（更换主题）" 这类标签中的提示词，与它们在数据中的整体频率相关。卡方检验（χ² test）得到 χ²(32) = 76.2263，p < 0.001，证实了显著的相关关系。这验证了我们的标签是语言学上信息丰富的语篇边界（discourse boundaries）标记，并为在新数据集中筛选主题切换数据提供了一个判据。

### 5.6 本地 LLM 的性能对比（Performance Comparison for Local LLMs）

**表 7：本地 LLM 上的性能。Llama 和 Qwen 分别采用 Llama-3.1-70B-Instruct 和 Qwen2.5-72B-Instruct。所有方法中的最佳性能以粗体标出。**

| 模型 | TIAGE Pk↓ | TIAGE WD↓ | TIAGE F1↑ | SuperDialseg Pk↓ | SuperDialseg WD↓ | SuperDialseg F1↑ | Dialseg711 Pk↓ | Dialseg711 WD↓ | Dialseg711 F1↑ |
|---|---|---|---|---|---|---|---|---|---|
| Plain Text + Llama | 0.472 | 0.515 | 0.215 | 0.492 | 0.495 | 0.026 | 0.350 | 0.373 | 0.032 |
| Plain Text + Qwen | 0.495 | 0.533 | 0.162 | 0.485 | 0.487 | 0.059 | 0.422 | 0.434 | 0.012 |
| S3-DST_uttr + Llama | 0.456 | 0.474 | 0.143 | 0.490 | 0.512 | 0.072 | 0.158 | 0.190 | 0.553 |
| S3-DST_uttr + Qwen | - | - | - | - | - | - | - | - | - |
| Def-DTS + Llama | **0.307** | **0.339** | **0.552** | **0.384** | **0.385** | **0.432** | **0.029** | **0.039** | **0.941** |
| Def-DTS + Qwen | 0.327 | 0.345 | 0.530 | 0.433 | 0.434 | 0.171 | 0.102 | 0.208 | 0.729 |

我们用本地 LLM（而非 GPT-4）进行了实验，具体为 Llama 3.1 和 Qwen 2.5，确切模型名见表 7。实验中我们测试了三种提示：Plain Text、S3-DST_uttr、Def-DTS。为保证效率，我们从每个数据集随机抽取 100 个样本。在本实验中，Def-DTS 在所有数据集上都取得了最高性能。在 Qwen 的情形中，所有数据集都观察到了格式错误（formatting errors）。虽然 plain（自然语言）是非结构化方法、没有出现错误，但其性能相对较低。这些结果表明，Def-DTS 在不同 LLM 上都能保持高准确率。更多 LLM 的实验见附录 C.1–C.2。

### 5.7 自动标注可能性的讨论（Discussion for Possibility of Auto-Labeling）

鉴于迄今已提出多种自动标注方法论，我们评估提示工程是否可能成为一种可行的自动标注方法。我们做了一个初步实验来评估用提示工程做 DTS 的可行性。我们用 Cohen's Kappa 分数，把 GPT-4 在 Def-DTS 下生成的片段标签与正确标签进行对比。结果显示 Kappa 分数为：TIAGE 0.485、SuperDialseg 0.429、Dialseg711 0.975，表明 TIAGE 和 SuperDialseg 上为中等一致（moderate agreement），Dialseg711 上为几乎完美一致（almost perfect agreement）。值得注意的是，我们在 TIAGE 上的标注结果超过了真实人工标注者之间观察到的 0.479 一致性分数。尽管鉴于中等一致性仍需改进，但这些发现表明，本方法仍可作为一个最起码可用的标注器（minimal annotator）。

---

## 6 结论（Conclusion）

以往的 DTS 方法一直受困于若干挑战，包括数据短缺、片段标注歧义，以及日益复杂的模型架构。与此同时，"用 LLM 进行推理"这一颇有前景的方法尚未在 DTS 语境下被探索。为解决这些问题，我们提出 Def-DTS，它把 LLM 与精细的推理策略结合起来。Def-DTS 既包含双向上下文抽取这一前人研究中的关键组件，又引入了话语意图分类这一新颖任务。该方法在开放域对话设定和任务导向对话设定下都展现出显著的性能提升。通过在话语意图分类任务中提供数据集专属的示例，本方法在多样化数据集上的效能得到增强，能够在不同对话语境下实现自适应的性能。通过其主要发现和多样化分析，我们证明了 LLM 推理是一条颇有前景的 DTS 路径。它不仅凸显了本方法的潜力，还从统计上勾勒出了未来研究需要解决的挑战。

在后续研究中，我们打算探索 DTS 自动标注的可行性，并通过 LLM 推理来考察 DTS 与其他 NLP 下游任务整合的潜力。

---

## 7 局限性（Limitations）

首先，尽管我们通过统计语言学实验证明了当前意图标签的重要性，但我们无法完全排除存在更合适意图标签的可能性。此外，由于我们提供的只是选取代表性示例的首个方法，更深入地探索"如何选取最优示例"的方法论，仍是本研究未来的一步工作。

为了在各种对话设定下提升意图与示例的质量，必须首先解决一个根本问题，即为 DTS 提供高质量的数据集。对话应包含周全的标注标准和真实的对话领域，以应对我们面临的局限。然而，人工标注不仅依然昂贵，还存在标注不一致或歧义的风险。我们相信，借助精细指南、使用 LLM 进行自动标注，将在打造一个更可持续、更可靠的 DTS 环境中发挥关键作用。

---

## 致谢（Acknowledgments）

本研究得到韩国创意内容振兴院（KOCCA）通过"文化、体育与旅游 R&D 计划"提供的资助支持，该资助由韩国文化体育观光部于 2024 年拨款（项目名称：开发面向同人创作的生成式 AI 故事平台，项目编号：RS-2024-00442270）。本研究部分得到由韩国政府（MSIT）资助的 IITP 资助支持（编号 RS-2020-II201361，人工智能研究生院计划（延世大学））。

---

## 参考文献（References）

（按任务要求，参考文献列表不翻译，此处略。完整文献见原文 arXiv:2505.21033。）

---

# 附录（Appendix）

## 附录 A 提示词（Prompts）

### A.1 提示模板（Prompt Template）

**表 8：我们为主数据集 TIAGE 提供的提示模板。每个数据集相比其他数据集有不同特征，因此我们针对每个数据集对原始模板中的意图池做了修改。**

> 说明：以下为工程可直接复用的关键资产——**先给出原样英文 prompt，再附中文翻译**。

#### 原文 Prompt（英文，原样保留）

```xml
<valid_utterance_intent>
<item>
<name>JUST_COMMENT</name>
<desc>Commenting on the preceding context without any asking. Not a topic shift</desc>
<example>
<speaker1>My dad works for the New York Times.</speaker1>
<speaker2>Oh wow! You know, I dabble in photography; maybe you can introduce us sometime.</speaker2>
<speaker1>Photography is the greatest art out there. (not a topic shift)</speaker1>
</example>
</item>
<item>
<name>JUST_ANSWER</name>
<desc>Answering preceding utterance. Not a topic shift</desc>
<example>
<speaker1>Do you teach cooking? </speaker1>
<speaker2>No, since I'm a native of Mexico, I teach Spanish. (not a topic shift)</speaker2>
</example>
</item>
<item>
<name>DEVELOP_TOPIC</name>
<desc>Developing the conversation to similar and inclusive sub-topics. Not a topic shift</desc>
<example>
<speaker1>Pets are cute!</speaker1>
<speaker2>I heard that Huskies are difficult dogs to take care of. (not a topic shift)</speaker2>
</example>
</item>
<item>
<name>INTRODUCE_TOPIC</name>
<desc>Introducing a relevant but different topic. A topic shift</desc>
<example>
<speaker1>You are an artist? What kind of art, I do American Indian stuff.</speaker1>
<speaker2> I love to eat too, sometimes too much. (a topic shift)</speaker2>
</example>
</item>
<item>
<name>CHANGE_TOPIC</name>
<desc>Completely changing the topic. A topic shift</desc>
<example>
<speaker1>What do you do for fun?</speaker1>
<speaker2>I drive trucks so me and my buds go truckin in the mud.</speaker2>
<speaker1>Must be fun! My version of that's running around a library!</speaker1>
<speaker2>That's cool! I love that too. Do you have a favourite animal? Chickens are my favourite. I love them. (topic shift)</speaker2>
</example>
</item>
</valid_utterance_intent>

<valid_topic_shift_label>
<item>
<name>YES</name>
<desc>The current utterance has **weak OR no topical** relation to the preceding conversation context OR is the first utterance in the conversation, marking the beginning of a new dialogue segment.</desc>
</item>
<item>
<name>NO</name>
<desc>The current utterance has **relevant OR equal** topic to the preceding conversation context.</desc>
</item>
</valid_topic_shift_label>

## TASK ##
You are given a dialogue starting with U. From utterance number 0, you have to answer the following sub-tasks for each utterance.

1. Summarize the preceding and subsequent context in <=3 sentences seperately
The range of the context should be previous or next 1-3 utterances except for the case of the first or last utterance.
For example, given current utterance number is 2, preceding range is 0-1, subsequent range is 3-5.

2. Output the utterance_intent
Use the list <valid_utterance_intent> … </valid_utterance_intent> to categorize utterance.
Consider topical difference between preceding and subsequent context.

3. Output the topic_shift_label
Use the list <valid_topic_shift_label> … </valid_topic_shift_label>.

## OUTPUT FORMAT ##
<U{utterance number}>
<preceding_context>
<range>{range of utterances referred in context}</range>
<context>{context of the previous 1-3 utterances}</context>
</preceding_context>
<subsequent_context>
<range>{range of utterances referred in context}</range>
<context>{context of the next 1-3 utterances}</context>
</subsequent_context>
<utterance_intent>{valid utterance intent}</utterance_intent>
<topic_shift_label>{valid topic shift label}</topic_shift_label>
</U{utterance number}>

## INPUT ##
{XML-structured dialogue}

## OUTPUT ##
```

#### 中文翻译（对照）

合法话语意图列表 `<valid_utterance_intent>`：

- **JUST_COMMENT**：对前文进行评论，而没有任何提问。不是主题切换。
  - 示例：说话人1"我爸在《纽约时报》工作。" / 说话人2"哇！你知道吗，我也玩点摄影；改天你可以把我介绍给他。" / 说话人1"摄影是世上最伟大的艺术。（不是主题切换）"
- **JUST_ANSWER**：回答前一条话语。不是主题切换。
  - 示例：说话人1"你教做饭吗？" / 说话人2"不，我是墨西哥本地人，我教西班牙语。（不是主题切换）"
- **DEVELOP_TOPIC**：把对话展开到相似且包含性的子主题。不是主题切换。
  - 示例：说话人1"宠物真可爱！" / 说话人2"我听说哈士奇是很难照顾的狗。（不是主题切换）"
- **INTRODUCE_TOPIC**：引入一个相关但不同的主题。是主题切换。
  - 示例：说话人1"你是艺术家？哪种艺术，我做美洲原住民题材的东西。" / 说话人2"我也很爱吃，有时吃太多了。（是主题切换）"
- **CHANGE_TOPIC**：彻底更换话题。是主题切换。
  - 示例：说话人1"你平时怎么找乐子？" / 说话人2"我开卡车，所以我和哥们儿会去泥地里飙车。" / 说话人1"听起来很好玩！我的版本是在图书馆里跑来跑去！" / 说话人2"很酷！我也喜欢。你有最喜欢的动物吗？鸡是我的最爱，我爱它们。（主题切换）"

合法主题切换标签列表 `<valid_topic_shift_label>`：

- **YES**：当前话语与前文对话上下文**关系微弱或无关**，或者是对话中的第一条话语，标志着一个新对话片段的开始。
- **NO**：当前话语与前文对话上下文**相关或主题相同**。

**## 任务 ##**
给定一段以 U 开头的对话。从第 0 号话语开始，你必须对每条话语回答以下子任务：

1. 分别用不超过 3 句话摘要前文和后文上下文。上下文范围应为前/后 1–3 条话语（首条或末条话语除外）。例如，当前话语编号为 2 时，前文范围为 0–1，后文范围为 3–5。

2. 输出 utterance_intent（话语意图）。使用 `<valid_utterance_intent>…</valid_utterance_intent>` 列表对话语归类。要考虑前文与后文上下文之间的主题差异。

3. 输出 topic_shift_label（主题切换标签）。使用 `<valid_topic_shift_label>…</valid_topic_shift_label>` 列表。

**## 输出格式 ##**（结构同上方 XML：`<U{话语编号}>` 内含 `<preceding_context>`（带 `<range>` 和 `<context>`）、`<subsequent_context>`、`<utterance_intent>`、`<topic_shift_label>`）

**## 输入 ##** {XML 结构化对话}
**## 输出 ##**

### A.2 其他数据集的意图标签（Intent Labels for other datasets）

**表 9：SuperDialseg 数据集的话语意图列表。**

| 意图 | 描述 |
|---|---|
| DIFFERENT_QUESTION（不同提问） | 提问与前文不相似或主题不同的内容。**是**主题切换 |
| RELEVANT_QUESTION（相关提问） | 提问与前文相似或主题连贯的内容。**不是**主题切换 |
| ANSWERING（回答中） | 回答前一条话语。**不是**主题切换 |
| ADDITIONAL_COMMENT（追加评论） | 同一说话人在前一条话语之外追加的评论。**不是**主题切换 |

对于 Dialseg711 数据集，我们从原始意图列表中删除了名为 INTRODUCE_TOPIC 的意图。对于 SuperDialseg 数据集，主题切换发生在"话语指向与前一条话语不同的文档"时。如表 9 所示，我们把意图从原始版本完全更换，以适配基于文档对齐（document-grounded）的对话设定中的主题转换。

### A.3 TIAGE 回应模式的修改（Modification of TIAGE Response Patterns）

**表 10：不同对话场景下的意图标签。**

| 场景 | 是否主题切换 |
|---|---|
| 对前文进行评论 | 否 |
| 问题回答 | 否 |
| 把对话展开到子主题 | 否 |
| 引入一个相关但不同的主题 | 是 |
| 彻底更换话题 | 是 |

TIAGE（Xie 等, 2021）最初是为"不使用后文上下文的实时主题切换检测"而设计的。因此，它的会话回应模式列表无法原样用于我们的全对话分割任务。为解决这一点，我们提出两种方法。第一，我们不再仅用紧邻上下文来分类每条话语，而是同时检索前文和后文上下文来辅助决策。这种双向视角能捕捉一条话语与"此前所说"和"随后所说"的关系，从而实现更精确的意图分类。第二，我们把 TIAGE 原始的回应模式列表改造以适配我们的分类目标——重组模式（如 Asking 提问类）、并精炼每个意图的描述（如 Relevant 相关、Inclusive 包含）。这一改造确保了对"话语是延续主题、还是以细微方式发生切换"的更细粒度检测。通过这些增强，我们在对整段对话做主题分割时取得了显著更好的性能，超越了直接使用 TIAGE 原始列表所得的结果。

### A.4 意图示例的构建（Construction of Intent Examples）

对于 TIAGE 数据集，我们直接使用了其论文中的示例。对于其他数据集，我们从训练（Train）划分中随机选取了符合以下规则的对话片段：

- 每个示例选取 2–3 条连续话语。
- 确保示例中最后一条话语对应目标话语意图。
- 所有示例都从单段对话中抽取。
- 保持话语长度简洁（在 100 字符以内）。

这一领域无关（domain-independent）的指南，可以作为"为对话量身定制最佳示例"的初步探路者，尽管它未必完美。

### A.5 意图池构建原则（Intent Pool Construction Principles）

在构建意图标签、描述和说明性示例时，我们发现：当提示具备以下两个关键特征时，能产生最具泛化性、最有效的性能：

- 意图列表允许对话语做**互斥（mutually exclusive）**的归类。
- 解释和示例能清晰区分每条话语所引发的**主题切换程度（degree of topic shifts）**。

附录 8（即 A.1）所采用的意图规范满足这些条件，针对其他数据集量身定制的意图（附录 A.2）也同样按照这些原则来设计。

---

## 附录 B 其他数据集上的分析（Analysis on other datasets）

### B.1 消融实验（Ablation Study）

**表 11：用于识别本方法中各子任务有效性的消融实验。**

| 方法 | SuperDialseg Pk↓ | SuperDialseg WD↓ | SuperDialseg F1↑ | Dialseg711 Pk↓ | Dialseg711 WD↓ | Dialseg711 F1↑ |
|---|---|---|---|---|---|---|
| w/o all | 0.378 | 0.382 | 0.467 | 0.007210 | 0.009416 | 0.987245 |
| w/o intent | 0.363 | 0.364 | 0.448 | 0.093486 | 0.127211 | 0.701330 |
| w/o examples | 0.338 | 0.341 | 0.646 | 0.005826 | 0.012322 | 0.984733 |
| w/o context | 0.327 | 0.331 | 0.635 | 0.005800 | 0.008464 | 0.989770 |
| Def-DTS | 0.317 | 0.322 | 0.674 | 0.009024 | 0.013738 | 0.982143 |

如表 11 所示，我们发现缺失某些模块会导致性能下降。为高效评估，我们分别为 SuperDialseg 和 Dialseg711 的消融实验各随机抽样 100 段对话。这批数据与 5.6 节所用相同。

特别是在 SuperDialseg 数据集上，我们观察到：增加任意子任务都能带来一致的提升。然而，对于 Dialseg711 数据集，上下文抽取模块的存在对性能提升至关重要——甚至 w/o（去掉某模块的）情形都超越了我们全部组件齐备的方法。我们推测，这个问题源于对局部上下文的过度集中（over-concentration for local context），这与 TIAGE 的消融结果一致；此外，对于具有清晰主题切换信号的 Dialseg711 而言，仅仅预测标签就足以解决问题。归根结底，我们的意图分类模块在所有数据集上都提升了性能。

### B.2 意图分类准确率（Intent Classification Accuracy）

**表 12：其他数据集的意图级混淆矩阵。**

SuperDialseg：

| 意图 | TP | FP | TN | FN | Acc |
|---|---|---|---|---|---|
| DIFFERENT_QUESTION | 2456 | 688 | 83 | 192 | 0.74 |
| RELEVANT_QUESTION | 1 | 0 | 811 | 989 | 0.45 |
| ANSWERING | 0 | 2 | 7819 | 168 | 0.98 |
| ADDITIONAL_COMMENT | 0 | 0 | 1264 | 211 | 0.86 |

Dialseg711：

| 意图 | TP | FP | TN | FN | Acc |
|---|---|---|---|---|---|
| JUST_COMMENT | 0 | 6 | 5067 | 14 | 0.996 |
| JUST_ANSWER | 0 | 6 | 7675 | 8 | 0.998 |
| DEVELOP_TOPIC | 0 | 3 | 2359 | 13 | 0.993 |
| CHANGE_TOPIC | 2708 | 66 | 0 | 0 | 0.976 |

我们对其他数据集做了详细的准确率分析，结果见表 12。

对于 SuperDialseg，如表 9 所示应用了四个新的意图池。ADDITIONAL_COMMENT、ANSWERING、RELEVANT_QUESTION 被归为非主题切换情形，而 DIFFERENT_QUESTION 被归为主题切换情形。然而，对于 DIFFERENT_QUESTION 的情形，指令遵循（instruction following）执行得并不好。对于 RELEVANT_QUESTION 的情形，除一个例外外指令遵循执行良好，但其准确率相对较低。RELEVANT_QUESTION 与 DIFFERENT_QUESTION 在解释上的差异，可能与实际数据集特征和主题变化有关。相比之下，ANSWERING 和 ADDITIONAL_COMMENT 的情形显示出相当高的分类准确率。这一对比表明，改进"提问类（Question-type）"意图将带来整体性能的提升。

对于 Dialseg711，整体准确率比其他数据集更高，其中演绎指令未被执行的话语不到 1%。在三个意图（除 CHANGE_TOPIC 外）的结果中，所有错误情形里，分别有 18% 的 DEVELOP_TOPIC、30% 的 JUST_COMMENT、42% 的 JUST_ANSWER 是因指令遵循错误而被误分类的。我们认为，这一问题可以通过为指令遵循增加额外指令或修改提示来解决。

---

## 附录 C 额外实验（Additional Experiments）

### C.1 本地 LLM 实验（Experiments for Local LLMs）

**表 13：本地 LLM 上的性能。Llama 8B、Qwen 7B、Qwen 32B 分别采用 Llama-3.1-8B-Instruct、Qwen2.5-7B-Instruct、Qwen2.5-32B-Instruct。Pk、WD、F1 仅对格式正确的输出计算。Error 列为格式错误数量。**

| 方法 | 模型 | TIAGE Pk↓ | TIAGE WD↓ | TIAGE F1↑ | TIAGE Error | SuperDialseg Pk↓ | SuperDialseg WD↓ | SuperDialseg F1↑ | SuperDialseg Error | Dialseg711 Pk↓ | Dialseg711 WD↓ | Dialseg711 F1↑ | Dialseg711 Error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Plain Text | Llama 8B | 0.529 | 0.604 | 0.303 | 1 | 0.497 | 0.504 | 0.036 | 0 | 0.350 | 0.373 | 0.032 | 0 |
| Plain Text | Qwen 7B | 0.509 | 0.563 | 0.249 | 2 | 0.517 | 0.522 | 0.132 | 1 | 0.486 | 0.513 | 0.069 | 0 |
| Plain Text | Qwen 32B | 0.476 | 0.515 | 0.221 | 0 | 0.466 | 0.471 | 0.083 | 0 | 0.391 | 0.415 | 0.015 | 0 |
| S3-DST_uttr | Llama 8B | 0.460 | 0.460 | 0.018 | 0 | 0.472 | 0.494 | 0.076 | 0 | 0.188 | 0.196 | 0.705 | 0 |
| S3-DST_uttr | Qwen 7B | 0.563 | 0.860 | 0.299 | 57 | 0.578 | 0.952 | 0.351 | 47 | 0.582 | 0.759 | 0.093 | 71 |
| S3-DST_uttr | Qwen 32B | 0.430 | 0.455 | 0.211 | 22 | 0.431 | 0.443 | 0.106 | 57 | 0.237 | 0.270 | 0.497 | 83 |
| Def-DTS | Llama 8B | 0.474 | 0.525 | 0.218 | 1 | 0.473 | 0.527 | 0.398 | 0 | 0.246 | 0.268 | 0.464 | 10 |
| Def-DTS | Qwen 7B | 0.462 | 0.465 | 0.022 | 20 | 0.473 | 0.474 | 0.071 | 14 | 0.162 | 0.170 | 0.715 | 44 |
| Def-DTS | Qwen 32B | 0.338 | 0.374 | 0.501 | 38 | 0.392 | 0.400 | 0.429 | 1 | 0.221 | 0.343 | 0.435 | 18 |

Def-DTS 依赖 LLM 的推理能力，因此模型规模会显著影响性能。表 7 显示 S3-DST 在 Qwen 70B 上出现格式错误，这一开始让我们不太愿意测试更小的 LLM。然而，考虑到小型 LLM（sLLM）能力和应用的不断增长，我们针对 Llama 8B、Qwen 7B、Qwen 32B 做了额外实验。

实验结果见表 13。尽管 Def-DTS 在较小模型上以及在 Dialseg711 上有所吃力，但它在较大模型上通过发挥 LLM 推理能力展现出了更大的提升。不过，我们承认，把 Def-DTS 应用到更小的 LLM 上需要做一些调整，例如额外的参数修改。

### C.2 闭源 LLM 的性能（Performance for Closed-source LLMs）

**表 14：额外闭源 LLM 的性能。**

| 模型 | TIAGE Pk↓ | TIAGE WD↓ | TIAGE F1↑ |
|---|---|---|---|
| R1（Deepseek-R1） | 0.286 | 0.331 | 0.644 |
| V3（Deepseek-V3） | 0.259 | 0.204 | 0.674 |
| GPT-4o | 0.232 | 0.256 | 0.699 |

我们用额外的闭源 LLM 评估了 Def-DTS 的性能，包括 Deepseek-R1（Guo 等, 2025）和 Deepseek-V3（Liu 等, 2024），如表 14 所示。三个模型都取得了比 4.4 节中所报告的基于 LLM 的方法更好的性能，并且——除 R1 外——甚至超越了监督基线。这证明了 Def-DTS 在各种闭源 LLM 上的通用适用性。有趣的是，尽管 R1 是专为推理而优化的，它却不如并非专为此类能力设计的 GPT-4o 和 V3。我们把这归因于我们的提示策略：通过提供明确的主题切换判据、说明性示例和清晰定义的推理路径，任务被组织成一种"减少了对复杂、主动推理之需求"的形式。

---

## 附录 D 实验细节（Details for Experiment）

### D.1 实现细节（Details for Implementation）

我们实验所用的模型：闭源 LLM 为 gpt-4o，开源 LLM 为 Llama-3.1-70B-Instruct 和 Qwen2.5-72B-Instruct。起初我们考虑了两个闭源模型：gpt-4o 和 Claude-3.5-sonnet。但 Claude 在初步评估中相比 gpt-4o 准确率较差，且没有先前研究把其方法应用到 Claude 系列上，因此被排除。对于开源 LLM 的推理，我们使用了由 4 块 NVIDIA A100 80GB GPU 组成的计算基础设施。我们在实验中没有采用任何模型专属的调优或量化（quantization）技术，从而保持了模型原始的架构与参数。除了为实验可复现性把温度（temperature）设为 0 之外，我们保留了最初声明的超参数。

### D.2 复现细节（Details for Reproduce）

**表 15：各方法原论文中报告的性能。其中 DialSTART 指 Gao 等（2023）的主要结果，SumSeg 指 Artemiev 等（2024）的主要结果，Plain Text 指 Jiang 等（2023）主要结果的 ChatGPT 变体，S3-DST_turn 指 Das 等（2024）的主要结果。**

| 模型 | TIAGE Pk↓ | TIAGE WD↓ | TIAGE F1↑ | SuperDialseg Pk↓ | SuperDialseg WD↓ | SuperDialseg F1↑ | Dialseg711 Pk↓ | Dialseg711 WD↓ | Dialseg711 F1↑ |
|---|---|---|---|---|---|---|---|---|---|
| **无监督学习方法** | | | | | | | | | |
| DialSTART | - | - | - | - | - | - | 0.179 | 0.198 | - |
| SumSeg | 0.438 | 0.455 | - | 0.469 | 0.480 | - | - | - | - |
| **基于 LLM 的方法** | | | | | | | | | |
| Plain Text (GPT-3.5) | 0.496 | 0.560 | 0.362 | 0.318 | 0.347 | 0.658 | 0.290 | 0.355 | 0.690 |
| S3-DST_turn | - | - | - | - | - | - | 0.009 | 0.008 | - |

对于 SuperDialseg 论文（Jiang 等, 2023），我们获取了 Random、TextTiling、TextTiling+Glove、CSM、BERT、RoBERTa、RetroTS-T5 等方法的实验结果。

我们复现了 PlainText（Jiang 等, 2023）、S3-DST（Das 等, 2024）、DialSTART（Gao 等, 2023）、SumSeg（Artemiev 等, 2024）的实验结果——这些方法要么缺少 F1 分数，要么没有在某些数据集上做实验。

在复现过程中，我们保持了包括随机种子在内的所有设置，未做任何参数修改。然而，我们观察到与原始实验不同的结果。它们原始的实验结果列于表 15。

对于基于 LLM 方法的复现，我们在以下情形做了必要修改：

- **Plain Text（Jiang 等, 2023）**：由于 Plain Text 是唯一公开了系统提示（system prompt）的方法论，为公平起见我们也不使用系统提示。我们对比了使用和不使用系统提示的结果，发现除解析上的不便外，性能几乎相同。

- **S3-DST（Das 等, 2024）**：S3-DST 以"轮次（turn）"为基础构建提示，而我们以"话语（utterance）"为基础进行。当对话有连续话语或奇数条话语时，他们的做法不适用于我们的方法。因此，我们把基于轮次的推理修改为基于话语的推理。

所有最终使用的提示都附在我们的代码仓库中。

---

*（本文档由 LaTeXML 于 2025 年 5 月 27 日生成的 HTML 源翻译而来。）*
