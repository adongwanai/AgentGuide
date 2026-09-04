---
type: 论文清单
status: 已发布
level: 进阶
topic:
  - Agent
  - 科研
  - 记忆
---

# 一口气读完Agent Memory的21篇论文：从理论到实践的完整指南

> **Agent记忆系统全景图 | 含2025最新综述 | 从入门到精通 | 算法工程师与开发工程师必读**

---

## 📖 前言：为什么Agent需要记忆？

想象一下，如果你每次和朋友聊天都不记得之前说过什么，每次做事都要从零开始学习，这样的生活会是什么样？AI Agent也是如此。尽管大语言模型（LLM）具备强大的推理能力，但它们面临着**上下文窗口限制**、**无法持续学习**、**缺乏个性化**等核心挑战。

Agent Memory（智能体记忆）就是为了解决这些问题而生的。本文将带你一口气读完这个领域最核心的21篇论文（包含2025年12月最新综述），从理论框架到工程实践，建立对Agent记忆系统的完整认知。

---

## 🎯 核心问题：关于Agent Memory我们需要考虑什么?

在深入论文之前,让我们先建立一个清晰的认知框架:

### 1. 如何获取记忆？
- 通过和用户交互（对话记录）
- 通过环境交互（任务执行）
- 通过工具调用（外部知识）

### 2. 怎么组织记忆？
- **模型参数**：知识固化在权重中
- **模型上下文**：当前会话的工作记忆
- **外部数据库**：长期存储的知识库

### 3. 怎么利用记忆？
- **RAG**（检索增强生成）
- **Few-shot**（少样本学习）
- **Fine-tuning**（微调）

---

## 🧠 记忆类型：人脑与AI的类比

### 人类记忆系统

正如人类利用长短期记忆进行有效的交互和学习一样，Agent的记忆机制通常划分为：

- **短期记忆（Short-term Memory）**：决定Agent在微观任务上的即时表现
- **长期记忆（Long-term Memory）**：作为持久知识库，决定Agent在宏观时间尺度上的智能深度和个性化水平

通过两者配合，Agent才能表现出连贯性、上下文感知能力，才会显得更智能。

### Agent记忆的分类体系

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/Z6bicxIx5naKN2bKgz1kjLz55MLy6TB9t6VSO8BCC0DJEI8Niaj3HpvawCkG9HMsYCMjd7lCVaErHEd09ic4eBxpw/640?wx_fmt=other&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=2)

智能体内的记忆存储区主要包括：

1. **上下文（Context）**：短期记忆或工作记忆区，窗口有限且容易被遗忘
2. **LLM参数**：蕴含了智能体的大部分知识，属于长期记忆区
3. **外挂记忆存储**：通过外挂存储的方式来对记忆进行扩展，也属于长期记忆区

### 存储形式的差别

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/Z6bicxIx5naKN2bKgz1kjLz55MLy6TB9tf1xibrSpuDSZQZRLMJochFQC5ydkJxJZmBZ9q06qsfUhXVA4Cob0XHg/640?wx_fmt=other&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=3)

智能体内的记忆主要以两种形式存在：

- **参数形式（Parametric）**：KV-Cache（短期）、LLM权重（长期）
- **非参数形式（Non-parametric）**：外挂记忆存储（长期）

---

## 🔄 Agent Memory工作流程

Agent通常通过以下几步来有效地管理记忆：

1. **记忆存储（Memory Storage）**：设计策略存储重要的交互信息
2. **记忆更新（Memory Update）**：随着交互不断更新，优化响应
3. **记忆检索（Memory Retrieval）**：根据当下需求检索相关内容

---

## 🛠️ Agent Memory实现方式

### 1. 物理外挂
即外置数据库和RAG，需要检索当前query相关的内容。
- **代表产品**：Mem0、Zep
- **优点**：即插即用
- **缺点**：不够end-to-end

### 2. Memory as Reasoning/Tool
通过训练Reasoning或Tool的方式动态更新context。
- **代表产品**：MemAgent、memory-R1
- **优点**：更接近end-to-end
- **缺点**：不够灵活

### 3. 参数更新
LLM本身就是一个Memory体，通过更新参数来更新记忆。
- **代表论文**：MemoryLLM、WISE
- **优点**：最本质的记忆方式
- **缺点**：最难实现

---

## 📚 论文阅读指南

### 两大流派

**🎓 模型驱动派**（算法岗推荐）：
- 改造模型内部结构
- 从底层嵌入记忆能力
- 适合做算法创新、发论文

**🛠️ 应用驱动派**（开发岗推荐）：
- 在应用层构建记忆系统
- 不改模型，加"外挂"
- 适合做工程落地、快速验证

### 推荐阅读顺序

**开发岗（快速上手）**：
```
1. Memory in the Age of AI Agents (2025最新综述，建立完整框架)
   ↓
2. MemGPT (应用驱动的代表)
   ↓
3. Mem0 (生产级实现)
   ↓
4. Zep (时序知识图谱)
```

**算法岗（深入研究）**：
```
1. Memory in the Age of AI Agents (2025最新综述，三维框架)
   ↓
2. Agent Memory Survey (2024经典综述，六大操作)
   ↓
3. Memorizing Transformers (模型驱动的开山之作)
   ↓
4. MemoryLLM (可更新记忆)
   ↓
5. MemGPT (对比应用驱动)
   ↓
6. 其他前沿论文
```

---

## 🎓 Part 1: 应用驱动派核心论文（10篇）

### 1. MemGPT ⭐⭐⭐⭐⭐ 必读第一篇

**论文标题**：MemGPT: Towards LLMs as Operating Systems
**发表时间**：2023年10月
**机构**：UC Berkeley
**论文链接**：https://arxiv.org/abs/2310.08560
**GitHub**：https://github.com/cpacker/MemGPT (22k+ Stars)

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/Z6bicxIx5naKN2bKgz1kjLz55MLy6TB9t1hRIUtsO5h8qiaKSa8xdQibPBsENTKLN2UpF4Bwm6oic6CibMOIMibYxQAA/640?wx_fmt=other&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=5)

**核心思想**：
把LLM当作操作系统，借鉴操作系统的虚拟内存管理机制：
- **主内存**（Main Context）：固定大小的上下文窗口
- **外部存储**（External Context）：无限容量的外部数据库
- **页面置换算法**：智能决定哪些信息保留在主内存

**技术创新**：

1. **分层存储架构**：
   - **System Instructions**：存储静态的系统提示词
   - **Working Context**：存储key facts、preferences和重要的用户信息
   - **FIFO**：存储滚动的历史对话记录

2. **自主管理机制**：
   - **递归摘要总结**：基于当前Summary和要移除的消息生成新Summary
   - **记忆更新与提取**：通过Function Executor完成，完全自驱动
   - LLM自主决定何时swap in/out

**适合场景**：
- 长期对话（跨会话记忆）
- 复杂任务追踪
- 研究助手

---

### 2. MemoryBank ⭐⭐⭐⭐ 遗忘曲线的先驱

**论文标题**：MemoryBank: Enhancing Large Language Models with Long-Term Memory
**发表时间**：2023年
**论文链接**：https://arxiv.org/abs/2305.10250

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/Z6bicxIx5naKN2bKgz1kjLz55MLy6TB9tVyINlrn3eV1IN4SP2B5K5pZ2hFZXXkrTHiahu8aLSibqOGMGV7TSJS4w/640?wx_fmt=other&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=4)

**核心思想**：
Memory Bank是比较早期的Memory研究，为了让Chat机器人更具人性化。主要特点：

1. **对话总结和定期回顾**：存储摘要以便搜索记忆
2. **遗忘机制**：借鉴艾宾浩斯遗忘曲线理论，模仿人类认知过程

**底层存储分为三个部分**：

1. **In-Depth Memory Storage**：记录多轮对话的详细内容，包含时间戳
2. **Hierarchical Event Summary**：将对话内容压缩成简洁的摘要，形成层次化记忆结构
3. **Dynamic Personality Understanding**：通过长期互动不断评估和更新对用户个性的理解

**检索方式**：
每轮对话和其摘要为一个记忆片段，通过Embedding存储到向量数据库。当前对话上下文作为检索条件进行相似向量检索。

**适合场景**：
- 长期伴侣型AI（如Character.AI）
- 情感陪伴
- 个人助理

---

### 3. Zep ⭐⭐⭐⭐⭐ 时序知识图谱的标杆

**论文标题**：ZEP: A TEMPORAL KNOWLEDGE GRAPH ARCHITECTURE FOR AGENT MEMORY
**发表时间**：2025年1月
**论文链接**：https://arxiv.org/pdf/2501.13956
**GitHub**：https://github.com/getzep/zep

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/Z6bicxIx5naKN2bKgz1kjLz55MLy6TB9tiayHMVqpthF8SjYc2F4vmobvkDhNalibyQLMWZyYuA3kySxBT92TZgqg/640?wx_fmt=other&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=6)

**核心思想**：
ZEP号称更满足企业内的需求，能够根据对话和Business Data来获取更实时的信息。核心在于自研了图引擎Graphiti，提供时间感知的知识图谱。

**知识图谱三层结构**：

1. **情境子图（Episodic Memory）**：
   - Node是情境
   - Edge连接到语义实体
   - 包含原始输入数据（消息、问题、JSON格式）

2. **语义子图（Semantic Memory）**：
   - Node是语义实体
   - Edge表示语义实体之间的关系
   - 从情境子图中提取

3. **社区子图（Community Memory）**：
   - Node表示强连接实体的Clusters
   - 包含对聚类的高层次概括
   - 参考GraphRAG思路

**记忆更新的创新点**：

1. **记忆去重**：通过LLM判重，避免存储冲突的记忆
2. **时间信息提取与边失效机制**：记录事实数据的产生时间和作用时间，当有矛盾事实时标记为失效
3. **标签传播算法**：动态构建社区子图

**记忆提取的三步法**：

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/Z6bicxIx5naKN2bKgz1kjLz55MLy6TB9tW5usibnFLpOtc0zpVfpNWJWgDXnJAcvgWupMhrMYKoGO2KMUialmSILw/640?wx_fmt=other&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=7)

1. **Search**：多种检索方法结合（语义相似度、BM25、广度优先搜索）
2. **Reranker**：对搜索结果重新排序
3. **Constructor**：将节点和边转换为文本上下文

**性能表现**：
- DMR Benchmark: 94.8% vs MemGPT的93.4%
- LongMemEval benchmark: 准确度提升18.5%

**适合场景**：
- 需要追溯历史的场景
- 事件驱动的应用
- 长期用户关系管理

---

### 4. A-MEM ⭐⭐⭐⭐ 卡片笔记法的启发

**论文标题**：A-MEM: Agentic Memory for LLM Agents
**发表时间**：2024年
**论文链接**：https://arxiv.org/abs/2409.09908

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/Z6bicxIx5naKN2bKgz1kjLz55MLy6TB9t6fSxyVL2NXQYqibtU5MO7J6A3aNIYgouAQQCZyph1WeuU3jVWqjRuvA/640?wx_fmt=other&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=8)

**核心思想**：
通过实现『Zettelkasten知识管理方法』（卡片笔记法）来提升智能体的记忆能力。每个卡片是一个知识点，包含：
- 原始交互内容
- 时间戳
- 关键词
- 标签
- 上下文描述
- 链接集合

**主要实现方法**：

1. **Note Construction**：从最新交互中提取新记忆，生成Notes
2. **Link Generation & Memory Evolution**：检索与新记忆最相关的历史记忆，决定是否建立连接
3. **Memory Retrieval**：分析Query提取Keywords，利用Keywords从记忆网络中检索

**性能表现**：
在LOCOMO数据集上表现优于MemGPT和MemoryBank

---

### 5. Mem0 ⭐⭐⭐⭐⭐ 生产级首选

**论文标题**：Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory
**发表时间**：2025年4月
**论文链接**：https://arxiv.org/abs/2504.19413
**GitHub**：https://github.com/mem0ai/mem0 (29k+ Stars)
**注**：开源项目始于2023年7月，论文发表于2025年4月

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/Z6bicxIx5naKN2bKgz1kjLz55MLy6TB9tHhCx04Ny58WmTEcUzLuHLL5CqxyR17BbvdxOCmH7x9TsZMUkd33sGQ/640?wx_fmt=other&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=9)

**核心思想**：
图增强的记忆框架，提供两种实现：
- **Mem0**：不使用图的实现
- **Mem0-G**：基于Graph的实现

**Mem0的主要步骤**：

1. **记忆生成**：上下文感知的生成方式（当前问答 + 最近M条消息 + 会话Summary）
2. **记忆更新**：检索语义相似的N个记忆，让LLM判断是否需要增加、修改或删除

![图片](https://mmbiz.qpic.cn/mmbiz_png/Z6bicxIx5naKN2bKgz1kjLz55MLy6TB9tZKVc0VicDZzfjl6RnN6icsiaHXvZrkasSRHpQbGyspNjs7NxluEiapqQ3w/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=10)

**Mem0-G的创新点**：

1. **两阶段流水线**：
   - 实体提取模块
   - 关系生成器模块

2. **冲突检测机制**：识别潜在的冲突关系，将其标记为无效而非物理删除

3. **双重检索机制**：
   - **实体中心方法**：识别关键实体，构建完整子图
   - **语义三元组方法**：将整个查询编码为Embedding向量

**性能表现**：

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/Z6bicxIx5naKN2bKgz1kjLz55MLy6TB9tVnMJ6icEcq2IYW5mFjDhnlh1NkDUyn30DNDl2zl3zslcI6NmLXtHZsQ/640?wx_fmt=other&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=11)

在LOCOMO数据集上综合表现最好，Open Domain和Temporal问题Graph的效果更高。

**独特优势**：
- ✅ 开箱即用（10行代码集成）
- ✅ 支持多种后端（Redis、Qdrant、PostgreSQL）
- ✅ 自动去重和更新

**适合场景**：
- 个性化推荐
- 对话Agent
- 用户画像

---

### 6. MemOS ⭐⭐⭐⭐⭐ 记忆操作系统

**论文标题**：MemOS: A Memory OS for AI System
**发表时间**：2025年5月
**论文链接**：https://arxiv.org/abs/2505.22101
**机构**：记忆张量（亿元天使轮融资）

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/Z6bicxIx5naKN2bKgz1kjLz55MLy6TB9t6OwjCzQgjqmb25eIlibJyxCcHdyaVj18yynErJC68LrK2TJG9VDEGxw/640?wx_fmt=other&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=13)

**核心愿景**：
构建面向下一代智能体的记忆基础设施：
1. **记忆作为系统资源**：将记忆抽象为可调度管理的一流系统资源
2. **演进作为核心能力**：构建记忆与模型协同进化的基础设施
3. **治理作为安全基础**：建立全生命周期的记忆治理机制

**记忆分类创新**：

将记忆划为三种类型，支持类型间动态转换：

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/Z6bicxIx5naKN2bKgz1kjLz55MLy6TB9tANPssgH9ticLtSQVECibtzb199XmfR99p9huM1rJyozicCicv25v0k7kgw/640?wx_fmt=other&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=14)

1. **纯文本记忆（Plaintext Memory）**：长期记忆区的显式记忆
2. **激活记忆（Activation Memory）**：短期记忆区的隐式记忆（KV-Cache）
3. **参数记忆（Parameter Memory）**：长期记忆区的隐式记忆

**动态转换机制**：

- 纯文本 → 激活：频繁使用的记忆预先转换为KV-Cache，降低TTFT延迟
- 纯文本/激活 → 参数：通过蒸馏或LoRA固化到模型参数
- 参数 → 纯文本：过时知识卸载为纯文本形式

**统一抽象：MemCube**

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/Z6bicxIx5naKN2bKgz1kjLz55MLy6TB9tE7LflAiaMszC03qabv3fK7PGLOIq7WZALetCwfMatF2Q68GZNwjZukg/640?wx_fmt=other&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=15)

使用MemCube对不同类型的记忆做统一抽象，Metadata包含动态指标（如usage），通过追踪这些指标判断记忆是『热』还是『冷』。

**系统架构**：

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/Z6bicxIx5naKN2bKgz1kjLz55MLy6TB9t6MoE6472Y53mBxl1PCoro1WSB0o9dDf6d9Oib1AygQPOFvwZcZJKazw/640?wx_fmt=other&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=16)

**关键优化点**：

1. **MemReader**：识别任务意图、时间范围、主题实体，转换为结构化的MemoryCall
2. **多视角记忆结构**：
   - 标签系统
   - 知识图谱
   - 语义分层（private、shared、global）
3. **混合检索与动态调度**：支持标量和向量的混合检索

**性能表现**：
在LOCOMO测试数据集上取得SOTA成绩

---

### 7. MIRIX ⭐⭐⭐⭐ 多模块记忆框架

**论文标题**：MIRIX: Multi-Agent Memory System for LLM-Based Agents
**发表时间**：2025年7月
**论文链接**：https://arxiv.org/abs/2507.07957

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/Z6bicxIx5naKN2bKgz1kjLz55MLy6TB9traB8P5ACRiaMwtw6icwZgxNruAPkh8Suia4nfGAkpcnJ6sCwQL28ImRuA/640?wx_fmt=other&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=17)

**核心思想**：
借鉴人类记忆系统的『分工协作』理念，将记忆分为6个不同的组件，采取Multi-Agent架构进行管理。

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/Z6bicxIx5naKN2bKgz1kjLz55MLy6TB9tDUZQTIp7iazfIgT4GJhqzmgAK9BdTFweZtzwKQ3qh3KCRdtHs6yUObg/640?wx_fmt=other&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=18)

**优化策略**：

1. **记忆更新**：
   - 通过元记忆管理器（Meta Memory Manager）进行路由
   - 根据输入分析确定相关的记忆组件
   - 路由到对应的记忆管理器

2. **记忆提取**：
   - 先从所有记忆组件进行粗略检索，返回高层次摘要
   - Chat Agent确定需要进一步搜索的组件
   - 执行详细搜索返回完整内容

**核心理念**：
区分不同类型的记忆，采取不同的存储结构和检索方式，通过路由机制提升效率。

---

### 8. HippoRAG ⭐⭐⭐⭐ 神经生物学启发

**论文标题**：HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models
**发表时间**：2024年5月
**机构**：The Ohio State University、Stanford University
**会议**：NeurIPS 2024
**论文链接**：https://arxiv.org/abs/2405.14831
**GitHub**：https://github.com/OSU-NLP-Group/HippoRAG

![图片](https://mmbiz.qpic.cn/mmbiz_png/Z6bicxIx5naKN2bKgz1kjLz55MLy6TB9tZKVc0VicDZzfjl6RnN6icsiaHXvZrkasSRHpQbGyspNjs7NxluEiapqQ3w/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=10)

**核心思想**：
模拟人脑海马体的记忆形成机制：
- **海马索引**：快速定位相关记忆
- **新皮层存储**：长期知识存储
- **模式分离**：区分相似但不同的记忆

**技术实现**：
- Personalized PageRank（PPR）模拟海马索引
- 知识图谱存储
- 多跳检索

**实验效果**：
- 在多跳问答任务上超越传统RAG 20%+
- 检索准确率显著提升

**适合场景**：
- 多跳推理
- 复杂关系理解

---

### 9. Generative Agents ⭐⭐⭐⭐⭐ 拟人化记忆流

**论文标题**：Generative Agents: Interactive Simulacra of Human Behavior
**发表时间**：2023年
**机构**：斯坦福大学
**论文链接**：https://arxiv.org/abs/2304.03442

**核心思想**：
提出了**Memory Stream（记忆流）**架构，是现代Agent记忆架构的基础。

**机制**：
智能体的所有观察流被记录在无限长的列表中。决策时根据三个标准检索记忆：

1. **Recency（新近性）**：最近发生的事情更重要
2. **Importance（重要性）**：核心事件权重高于琐事
3. **Relevance（相关性）**：与当前情境语义相关

**亮点**：
引入了**Reflection（反思）**机制，定期从流水账记忆中提炼出高级观点（High-level insights），形成分层记忆结构。

**适合场景**：
- 社交模拟
- NPC（游戏角色）
- 游戏世界

---

### 10. Voyager ⭐⭐⭐⭐ 技能库作为记忆

**论文标题**：Voyager: An Open-Ended Embodied Agent with Large Language Models
**发表时间**：2023年5月
**机构**：Caltech、NVIDIA、USC等多机构合作
**论文链接**：https://arxiv.org/abs/2305.16291
**GitHub**：https://github.com/MineDojo/Voyager

**核心思想**：
提出了**Skill Library（技能库）**作为记忆，关注"程序性记忆"——如何记住"怎么做一件事"。

**机制**：
在Minecraft游戏中：
- 智能体探索世界并编写代码完成任务
- 代码验证成功后作为"技能"存入向量数据库
- 遇到类似任务时检索并复用代码片段

**亮点**：
解决了灾难性遗忘问题，实现能力的滚雪球式增长。

**性能表现**：
- 获取3.3倍更多独特物品
- 行进2.3倍更长距离
- 解锁关键技能树里程碑速度提升15.3倍

**适合场景**：
- 开放世界探索
- 编码任务
- 技能积累

**注**：这是研究项目，非NVIDIA官方产品

---

## 🔬 Part 2: 模型驱动派核心论文（5篇）

### 11. Memorizing Transformers ⭐⭐⭐⭐⭐ 开山之作

**论文标题**：Memorizing Transformers
**发表时间**：2022年3月
**机构**：Google Research
**论文链接**：https://arxiv.org/abs/2203.08913
**GitHub**：https://github.com/lucidrains/memorizing-transformers-pytorch

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/Z6bicxIx5naKN2bKgz1kjLz55MLy6TB9t1hRIUtsO5h8qiaKSa8xdQibPBsENTKLN2UpF4Bwm6oic6CibMOIMibYxQAA/640?wx_fmt=other&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=5)

**核心思想**：
首次在Transformer中引入外部记忆，通过KNN检索历史信息：
- 每个token都存储到外部记忆库
- 推理时通过KNN检索相关历史
- 融合外部记忆和内部注意力

**算法流程**：
```
1. 编码：将历史token存入记忆库
2. 检索：KNN搜索最相关的k个记忆
3. 融合：Attention(Q, K_memory, V_memory)
4. 输出：结合内部和外部注意力
```

**实验效果**：
- 在长文本任务上困惑度降低4%
- 外推能力显著提升

**算法岗价值**：
- 理解外部记忆的设计思想
- KNN检索的优化策略
- Attention融合机制

---

### 12. MemoryLLM ⭐⭐⭐⭐⭐ 可更新记忆

**论文标题**：MemoryLLM: Towards Self-Updatable Large Language Models
**发表时间**：2024年2月
**会议**：ICML 2024
**机构**：多机构合作（UCSD、Amazon等）
**论文链接**：https://arxiv.org/abs/2402.04624
**GitHub**：https://github.com/wangyu-ustc/MemoryLLM

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/Z6bicxIx5naKN2bKgz1kjLz55MLy6TB9tVyINlrn3eV1IN4SP2B5K5pZ2hFZXXkrTHiahu8aLSibqOGMGV7TSJS4w/640?wx_fmt=other&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=4)

**核心思想**：
在每层Transformer中插入可更新的Memory Tokens：
- **固定参数**：预训练的模型参数（不变）
- **Memory Tokens**：可读写的记忆单元（可更新）
- **终身学习**：持续学习新知识，对抗遗忘

**技术细节**：
1. Memory Tokens设计
2. 读写机制（Read/Write Gates）
3. 更新策略（何时更新、如何更新）
4. 遗忘控制

**适合研究方向**：
- 终身学习
- 模型编辑
- 知识更新

---

### 13. Memory³ ⭐⭐⭐⭐ 分层记忆

**论文标题**：Memory³: Language Modeling with Explicit Memory
**发表时间**：2024年7月
**论文链接**：https://arxiv.org/abs/2407.01178

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/Z6bicxIx5naKN2bKgz1kjLz55MLy6TB9tiayHMVqpthF8SjYc2F4vmobvkDhNalibyQLMWZyYuA3kySxBT92TZgqg/640?wx_fmt=other&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=6)

**核心思想**：
模拟人脑，将记忆分为三层：
- **感知记忆**：原始输入
- **短期记忆**：工作上下文
- **长期记忆**：知识存储

**算法设计**：
每一层有不同的编码方式和检索策略

**创新点**：
首次提出分层记忆的完整框架

---

### 14. WISE ⭐⭐⭐⭐ 双参数体系

**论文标题**：WISE: Rethinking the Knowledge Memory for Lifelong Model Editing
**发表时间**：2024年5月
**机构**：浙江大学
**论文链接**：https://arxiv.org/abs/2405.14768
**GitHub**：https://github.com/zjunlp/EasyEdit

![[Pasted image 20251223000704.png]]

**核心思想**：
- **主记忆**：预训练知识（冻结）
- **侧记忆**：后续编辑的知识（可更新）

**适合场景**：
- 知识编辑
- 事实更新
- 避免灾难性遗忘

---

### 15. Titans ⭐⭐⭐⭐ 学习遗忘

**论文标题**：Titans: Learning to Memorize at Test Time
**发表时间**：2024年12月（2025年1月公开）
**机构**：Google Research
**论文链接**：https://arxiv.org/abs/2501.00663
**GitHub**：https://github.com/lucidrains/titans-pytorch（非官方实现）

![图片](https://github.com/lucidrains/titans-pytorch/blob/main/fig2.png?raw=true)

**核心思想**：
设计神经网络模块，学习何时存储、何时遗忘：
- **存储门控**：决定哪些信息值得存储
- **遗忘门控**：决定何时清理记忆
- **检索门控**：决定调用哪些记忆

**技术突破**：

1. **测试时学习（Test-Time Learning）**：
   - 模型在推理过程中持续学习和更新参数
   - 而非在预训练后冻结权重

2. **MIRAS框架**：
   - 在模型运行时更新核心记忆
   - 处理超过2M的上下文窗口

3. **Surprise指标**：
   - 基于梯度误差计算"惊讶度"
   - 高误差表示信息新颖，应该记忆

**创新点**：
让模型学会"断舍离"，避免记忆过载

**性能表现**：
- 在语言建模、常识推理、基因组学、时间序列任务上超越Transformer
- 在针刺问答（Needle-in-Haystack）任务中准确率更高

---

## 📚 Part 3: 综述与理论（4篇）

### 16. Memory in the Age of AI Agents ⭐⭐⭐⭐⭐ 2025年最新综述

**论文标题**：Memory in the Age of AI Agents: A Survey Forms, Functions and Dynamics
**发表时间**：2025年12月
**作者**：Yuyang Hu 等46位研究者
**论文链接**：https://arxiv.org/abs/2512.13564
**GitHub**：https://github.com/Shichun-Liu/Agent-Memory-Paper-List

**核心贡献**：

这是目前**最新、最全面**的Agent Memory综述，解决了现有研究碎片化的问题，提出了统一的分类框架。

#### 🔑 核心框架：三个维度

**1. 记忆形式（Forms）- 记忆如何实现**

- **Token级记忆（Token-level Memory）**：
  - 直接将记忆以文本token形式存储在上下文中
  - 典型方法：Prompt Engineering、In-Context Learning
  - 优点：简单直接、易于理解和调试
  - 缺点：受限于上下文窗口长度

- **参数化记忆（Parametric Memory）**：
  - 将记忆固化在模型参数中
  - 典型方法：Fine-tuning、LoRA、Memory Tokens
  - 优点：访问速度快、不占用上下文
  - 缺点：更新成本高、可能遗忘

- **潜在记忆（Latent Memory）**：
  - 以隐式表示形式存储（如KV-Cache、Embedding）
  - 典型方法：KV-Cache复用、向量数据库检索
  - 优点：压缩效率高、检索灵活
  - 缺点：可解释性差

**2. 记忆功能（Functions）- 记忆存储什么**

- **事实型记忆（Factual Memory）**：
  - 存储客观事实和知识
  - 例如："用户的生日是1990年5月1日"
  - 对应人类的**语义记忆**

- **经验型记忆（Experiential Memory）**：
  - 存储过去的交互经历和事件
  - 例如："上次用户询问了关于Python的问题"
  - 对应人类的**情景记忆**

- **工作型记忆（Working Memory）**：
  - 存储当前任务的中间状态
  - 例如："当前正在执行第3步，已完成前2步"
  - 对应人类的**工作记忆/短期记忆**

**3. 记忆动态（Dynamics）- 记忆如何演化**

- **记忆形成（Formation）**：
  - 如何从原始输入中提取和编码记忆
  - 关键技术：摘要、实体提取、关键信息识别

- **记忆演化（Evolution）**：
  - 如何更新、合并、遗忘记忆
  - 关键技术：冲突检测、重要性评估、遗忘机制

- **记忆检索（Retrieval）**：
  - 如何根据需求找到相关记忆
  - 关键技术：语义检索、时序检索、混合检索

#### 🎯 关键洞察

**与传统分类的区别**：

| 传统分类 | 本综述框架 | 优势 |
|:---|:---|:---|
| 短期/长期记忆 | Forms + Functions + Dynamics | 更精细、更全面 |
| 模糊的"记忆类型" | 三个正交维度 | 清晰区分实现、内容、过程 |
| 静态视角 | 动态演化视角 | 关注记忆生命周期 |

**明确区分的概念**：

1. **Agent Memory vs LLM Memory**：
   - LLM Memory：模型预训练知识（静态）
   - Agent Memory：智能体运行时记忆（动态）

2. **Agent Memory vs RAG**：
   - RAG：检索静态文档库
   - Agent Memory：动态更新的个性化记忆

#### 📊 基准测试汇总

论文汇总了主流的Memory评测基准：

- **LOCOMO**：长对话记忆评测
- **LongMemEval**：企业级长期记忆评测
- **MemoryBench**：多任务记忆能力评测
- **DMR Benchmark**：动态记忆检索评测

#### 🛠️ 开源框架总结

论文系统梳理了开源Memory框架：

| 框架 | 记忆形式 | 记忆功能 | 核心特点 |
|:---|:---|:---|:---|
| MemGPT | Token级 + 潜在 | 事实 + 经验 | OS隐喻、分层存储 |
| Mem0 | 潜在（图谱） | 事实 + 经验 | 知识图谱、生产级 |
| Zep | 潜在（时序图） | 事实 + 经验 | 时间感知、企业级 |
| MemOS | 三种形式 | 三类功能 | 动态转换、OS架构 |

#### 🚀 新兴研究前沿

论文指出了5个重要研究方向：

1. **记忆自动化（Memory Automation）**：
   - 自动决定何时存储、更新、遗忘
   - 减少人工设计记忆策略

2. **强化学习集成（RL Integration）**：
   - 将记忆作为奖励信号的一部分
   - 学习最优记忆管理策略

3. **多模态记忆（Multimodal Memory）**：
   - 融合文本、图像、音频、视频记忆
   - 跨模态检索和推理

4. **多智能体记忆（Multi-Agent Memory）**：
   - 多个Agent之间的记忆共享
   - 协作记忆、冲突解决

5. **可信度问题（Trustworthiness）**：
   - 记忆的隐私保护
   - 记忆的真实性验证
   - 记忆的可解释性

#### 💡 为什么这篇综述重要？

1. **最新**：2025年12月发表，涵盖最前沿研究
2. **最全**：46位作者联合撰写，视角全面
3. **最系统**：首次提出Forms-Functions-Dynamics三维框架
4. **最实用**：提供基准测试和开源框架的完整梳理

**使用建议**：
- ⭐ **必读第一篇**：建立Agent Memory的完整认知体系
- 📖 配合GitHub论文列表：追踪最新研究进展
- 🎯 确定研究方向：通过三维框架找到创新点
- 🔧 选择工具：根据Forms-Functions选择合适的框架

---

### 17. Agent Memory Survey ⭐⭐⭐⭐⭐ 经典综述

**论文标题**：大模型智能体记忆机制综述
**发表时间**：2024年4月
**论文链接**：https://arxiv.org/pdf/2404.13501.pdf
**GitHub**：https://github.com/nuster1128/LLM_Agent_Memory_Survey

**核心贡献**：

1. **记忆分类框架**：
   - 参数化记忆
   - 上下文记忆（结构化/非结构化）

2. **六大操作**：
   - 巩固（Consolidation）
   - 更新（Update）
   - 索引（Indexing）
   - 遗忘（Forgetting）
   - 检索（Retrieval）
   - 压缩（Compression）

3. **研究主题**：
   - Memory架构设计
   - Memory操作优化
   - Memory评估方法
   - Memory应用场景

**使用建议**：
- 理解记忆操作的完整生命周期
- 掌握记忆系统的工程实现

---

### 18. CoALA ⭐⭐⭐⭐⭐ 认知架构框架

**论文标题**：Cognitive Architectures for Language Agents (CoALA)
**发表时间**：2023年
**论文链接**：https://arxiv.org/abs/2309.02427

**核心贡献**：
将Agent的记忆正式结构化为四类：

1. **Working Memory（工作记忆）**：当前的上下文
2. **Episodic Memory（情景记忆）**：过去的经历
3. **Semantic Memory（语义记忆）**：事实性知识
4. **Procedural Memory（程序性记忆）**：技能和规则

**亮点**：
统一了所有论文的术语，是理解Agent架构的通用蓝图

---

### 19. Multimodal Memory Survey ⭐⭐⭐⭐

**GitHub**：https://github.com/patrick-tssn/Awesome-Multimodal-Memory

**核心内容**：
收录400+篇多模态记忆论文：
- 视觉记忆（Visual Memory）
- 机器人记忆（Robotic Memory）
- 多模态上下文建模
- 音频、视频、图像、3D记忆

**适合方向**：
- 多模态Agent
- 具身智能
- VLM应用

---

## 🔧 Part 4: 其他重要论文（2篇）

### 20. Reflexion ⭐⭐⭐⭐ 语言反馈学习

**论文标题**：Reflexion: Language Agents with Verbal Reinforcement Learning
**发表时间**：2023年
**论文链接**：https://arxiv.org/abs/2303.11366

**核心贡献**：
提出基于**语言反馈的短期记忆窗口**。

**机制**：
智能体在行动失败后，会生成自我反思（Self-reflection）的文本摘要，存储在滑动窗口的记忆缓冲区中。

**亮点**：
下次尝试时，智能体会读取之前的"教训"，避免重蹈覆辙。典型的短期工作记忆优化。

**适合场景**：
- 复杂推理
- 解题
- 纠错

---

### 21. MemLong ⭐⭐⭐⭐ 超长文本处理

**论文标题**：MemLong: Memory-Augmented Retrieval for Long Context Modeling
**发表时间**：2024年8月
**论文链接**：https://arxiv.org/abs/2408.16967

**核心贡献**：
检索记忆模块 + 可控注意力

**适合场景**：
- 超长文本处理
- 文档分析

---

## 📊 Part 5: 技术演进趋势总结

### 两大流派完整对比

| 维度       | 模型驱动                                                                  | 应用驱动                                        |
| :------- | :-------------------------------------------------------------------- | :------------------------------------------ |
| **代表论文** | Memorizing Transformers<br/>MemoryLLM<br/>Memory³<br/>WISE<br/>Titans | MemGPT<br/>Mem0<br/>Zep<br/>MemOS<br/>MIRIX |
| **核心思路** | 改造模型内部结构                                                              | 应用层记忆管理                                     |
| **实现方式** | Memory Tokens<br/>KNN检索<br/>门控机制                                      | 外部数据库<br/>知识图谱<br/>向量检索                     |
| **优势**   | • 性能上限高<br/>• 读取效率高<br/>• 深度集成                                        | • 落地快<br/>• 易扩展<br/>• 模型无关                  |
| **劣势**   | • 研发成本高<br/>• 需要重新训练<br/>• 通用性差                                       | • 依赖底层模型<br/>• 可能有延迟<br/>• 幻觉问题             |
| **适合岗位** | 🔬 算法工程师                                                              | 🛠️ 开发工程师                                   |

---

### 经过验证的优化手段

从上述Memory产品的演进趋势看，以下优化手段被验证有效：

#### 1. 精细化的记忆管理
记忆在场景、分类和形式上有明确的区分，『分而治之』的思路被证明是有效的优化手段，这和Multi-Agent的优化思路类似。

#### 2. 组合多种记忆存储结构
记忆底层存储结构可以大致分为：
- **结构化信息**：Metadata或Tag
- **纯文本**：Text-Chunk、Summary、情境记录
- **知识图谱**：实体关系网络

分别构建标签索引、全文索引、向量索引和图索引提升检索效果。不同的存储结构对应不同的场景，记忆框架由集成单一结构演进到组合多种架构。

#### 3. 记忆检索优化
检索方式逐步演进：
- 单一检索 → 混合检索
- 针对Embedding和Reranker进行调优
- 多阶段检索（粗检索 → 精检索）

---

## 🎯 如何选择论文阅读？

### 按目标选择

| 你的目标 | 推荐论文 | 阅读顺序 |
|:---|:---|:---|
| **快速了解全貌** | Memory in the Age of AI Agents (2025) | 直接读最新综述 |
| **做工程落地** | 2025综述 → MemGPT → Mem0 → Zep | 先建立框架后实践 |
| **做算法创新** | 2025综述 → 2024综述 → Memorizing Transformers → MemoryLLM | 从框架到细节 |
| **发论文** | 2025综述 + 最新5篇（2024-2025） | 找到gap |

---

### 开发岗阅读策略

**重点关注**：
- ✅ 系统架构（怎么设计的）
- ✅ 实现细节（怎么实现的）
- ✅ 性能指标（效果如何）
- ❌ 可以跳过数学推导

**阅读顺序**：
```
Abstract → Introduction → System Design → Experiments
```

**时间分配**：
- 一篇论文：30-60分钟
- 重点：架构图 + 代码实现

---

### 算法岗阅读策略

**重点关注**：
- ✅ 问题定义（解决什么问题）
- ✅ 算法设计（创新点是什么）
- ✅ 数学原理（为什么有效）
- ✅ 实验设计（如何验证）
- ✅ 局限性（未来工作）

**阅读顺序**：
```
Abstract → Introduction → Method（重点！）→ Experiments → Conclusion
```

**时间分配**：
- 一篇论文：2-4小时（精读）
- 做笔记、画图、推公式

---

## 💡 典型论文流派与应用场景对照表

| **论文名称** | **核心隐喻** | **记忆类型** | **适用场景** |
|---|---|---|---|
| **Generative Agents** | 人类日记本 | 完整的情景记忆流 | 社交模拟、NPC、游戏 |
| **MemGPT** | 操作系统（OS） | 分层（RAM/Disk） | 长对话助手、文档分析 |
| **Voyager** | 代码库（Library） | 程序性记忆（技能） | 开放世界探索、编码任务 |
| **Reflexion** | 错题本 | 短期反思缓冲区 | 复杂推理、解题、纠错 |
| **MemoryBank** | 艾宾浩斯曲线 | 动态遗忘机制 | 情感陪伴、个人助理 |
| **Zep** | 时间图谱 | 时序知识图谱 | 历史追溯、事件驱动 |
| **Mem0** | 知识图谱 | 图增强记忆 | 个性化推荐、用户画像 |
| **MemOS** | 记忆OS | 三种记忆类型 | 企业级应用 |
| **HippoRAG** | 海马体 | 神经生物学记忆 | 多跳推理、关系理解 |

---

## 🛠️ 实践案例：基于Tablestore的Agent Memory实现

作为存储团队的实践者，我们基于Tablestore设计并发布了新的Agent Memory SDK，具有以下独特优势：

### 1. Serverless架构
- 支持计算与存储自动弹性扩展
- 按量计费
- 支持从0到PB级的弹性扩展
- 满足海量小租户和超大规模大租户两个极端场景

### 2. 高可用保障
- 支持多AZ容灾
- 提供跨区域容灾能力
- 多AZ容灾是默认能力，无需额外开通成本

### 3. 混合检索
- 提供Json、标签、全文、向量等索引
- 支持单表多向量联合检索
- 支持标量和向量混合检索

### 4. 生态集成
- 集成LangChain/LlamaIndex/LangEngine/SpringAI Alibaba等开发框架
- 接入Mem0提供的MCP服务OpenMemory
- 可托管在FC上开箱即用

---

## 📝 结语

Agent Memory领域正在快速发展，从早期的对话记忆，到现在覆盖任务执行、决策支持、个性化服务等多个场景。技术实现也从单一的向量检索，演进到知识图谱、时序建模、多模态记忆等多种方案。

无论你是算法工程师还是开发工程师，希望这篇文章能帮助你建立对Agent Memory的完整认知，找到适合自己的研究或实践方向。

---

## 📚 参考文献

1. 《Memory in the Age of AI Agents: A Survey Forms, Functions and Dynamics》(2025) ⭐ 最新
2. 《The Rise and Potential of Large Language Model Based Agents: A Survey》
3. 《From Human Memory to AI Memory: A Survey on Memory Mechanisms in the Era of LLMs》
4. 《MemoryBank: Enhancing Large Language Models with Long-Term Memory》
5. 《MemGPT: Towards LLMs as Operating Systems》
6. 《ZEP: A TEMPORAL KNOWLEDGE GRAPH ARCHITECTURE FOR AGENT MEMORY》
7. 《A-MEM: Agentic Memory for LLM Agents》
8. 《Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory》
9. 《MemOS: A Memory OS for AI System》
10. 《MIRIX: Multi-Agent Memory System for LLM-Based Agents》
11. 《Memorizing Transformers》
12. 《MemoryLLM: Towards Self-Updatable Large Language Models》
13. 《Generative Agents: Interactive Simulacra of Human Behavior》
14. 《Voyager: An Open-Ended Embodied Agent with Large Language Models》
15. 《Reflexion: Language Agents with Verbal Reinforcement Learning》
16. 《Cognitive Architectures for Language Agents (CoALA)》
17. 《HippoRAG: Neurobiologically Inspired Long-Term Memory》
18. 《Memory³: Language Modeling with Explicit Memory》
19. 《WISE: Rethinking the Knowledge Memory for Lifelong Model Editing》
20. 《Titans: Learning to Memorize at Test Time》
21. 《MemLong: Memory-Augmented Retrieval for Long Context Modeling》



