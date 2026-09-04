---
type: 教程
status: 已发布
level: 进阶
topic:
  - 上下文工程
  - 项目实战
---

# 《上下文工程：从提示到系统的AI应用架构指南》

> **教程定位**：系统化的上下文工程学习路径，从理论认知到生产实践的完整知识体系
> 
> **核心理念**：上下文工程是将不断变化的信息宇宙中最相关内容，精心筛选并放入有限上下文窗口的艺术与科学——本质是"熵减"过程

---

## 📖 目录结构

- [第一部分：认知篇 - 理解上下文工程](#第一部分认知篇---理解上下文工程)
- [第二部分：基础篇 - 结构化上下文](#第二部分基础篇---结构化上下文)
- [第三部分：策略篇 - 动态上下文管理](#第三部分策略篇---动态上下文管理)
- [第四部分：架构篇 - Agent与工具系统](#第四部分架构篇---agent与工具系统)
- [第五部分：进阶篇 - 编程化与自动化](#第五部分进阶篇---编程化与自动化)
- [第六部分：实战篇 - 企业案例与项目](#第六部分实战篇---企业案例与项目)
- [附录：资源索引](#附录资源索引)

---

## 第一部分：认知篇 - 理解上下文工程

### 第1章：什么是上下文工程？

**学习目标**：
- 理解从 Prompt Engineering 到 Context Engineering 的演进
- 掌握上下文窗口的本质与限制
- 建立"正确信息 + 正确工具 + 正确时机 + 正确格式 = 有效Agent"的核心公式

**核心内容**：
1. 上下文工程的定义与边界
2. 六层上下文模型：指令/提示、短期记忆、长期记忆、RAG检索、工具定义、结构化输出
3. 简单Demo vs 魔法Agent的本质区别
4. 上下文工程作为"熵减"过程的理论框架

**参考资源**：

| 资源名称 | 类型 | 链接 | 核心价值 |
|---------|------|------|----------|
| Phil Schmid - The New Skill in AI is Not Prompting | 博客 | https://www.philschmid.de/context-engineering | ★★★★★ 入门必读，首次系统定义 |
| A Survey of Context Engineering for LLMs | 论文 | https://arxiv.org/abs/2507.13334 | ★★★★★ 分析1400+论文的最权威综述 |
| Anthropic - Effective Context Engineering | 官方指南 | https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents | ★★★★★ Claude团队实践经验 |
| 知乎《上下文工程：将工程规范引入提示》 | 中文博客 | https://zhuanlan.zhihu.com/p/1928378624261731252 | ★★★★☆ 中文里写得最透彻 |
| addyo.substack - Context Engineering带来的工程规范 | 博客 | https://addyo.substack.com/p/context-engineering-bringing-engineering | ★★★★☆ 工程规范视角 |

---

### 第2章：核心原理 - 为什么Context有效？

**学习目标**：
- 理解Transformer的Self-Attention机制
- 掌握In-Context Learning（上下文学习）的工作原理
- 认识"Lost in the Middle"现象及其工程影响

**核心内容**：
1. Attention机制：上下文窗口的物理基础
2. Few-Shot学习：模型如何在不更新权重的情况下"学会"新任务
3. 示例的作用：格式 vs 内容的反直觉结论
4. 长文本陷阱：中间迷失现象与Re-ranking策略

**参考资源**：

| 资源名称 | 类型 | 链接 | 核心价值 |
|---------|------|------|----------|
| Attention Is All You Need | 论文 | https://arxiv.org/abs/1706.03762 | 理解Attention机制的基石 |
| Language Models are Few-Shot Learners (GPT-3) | 论文 | https://arxiv.org/abs/2005.14165 | 定义Few-Shot，理解上下文学习 |
| Rethinking the Role of Demonstrations | 论文 | https://arxiv.org/abs/2202.12837 | 反直觉结论：格式比标签正确性更重要 |
| Lost in the Middle: How LLMs Use Long Contexts | 论文 | https://arxiv.org/abs/2307.03172 | ★★★★★ 必读，指导RAG中的Re-ranking |
| Context Engineering 2.0 | 论文 | https://arxiv.org/abs/2510.26493 | 熵减理论框架，历史演进追溯 |
| GAIR-NLP/Context-Engineering-2.0 | GitHub | https://github.com/GAIR-NLP/Context-Engineering-2.0 | 配套代码与资源 |

---

### 第3章：上下文失败模式与诊断

**学习目标**：
- 识别四种常见的上下文失败模式
- 掌握针对性的修复策略
- 建立上下文质量评估的基本框架

**核心内容**：
1. **Context Poisoning（中毒）**：幻觉错误被反复引用
2. **Context Distraction（分心）**：过长上下文导致重复历史动作
3. **Context Confusion（混淆）**：无关工具/信息干扰响应
4. **Context Clash（冲突）**：多轮对话中信息矛盾
5. 修复策略：动态工具加载、上下文隔离、总结压缩

**参考资源**：

| 资源名称 | 类型 | 链接 | 核心价值 |
|---------|------|------|----------|
| How Contexts Fail—and How to Fix Them | 博客 | https://www.dbreunig.com/2025/06/22/how-contexts-fail-and-how-to-fix-them.html | ★★★★★ 反面教材+对策 |
| Practical Tips on Building LLM Agents | 博客 | https://letters.lossfunk.com/p/practical-tips-on-building-llm-agents | ★★★★☆ 一线经验+成本考量 |
| DataCamp - Context Engineering | 博客 | https://www.datacamp.com/blog/context-engineering | 常见失败案例及缓解 |

---

## 第二部分：基础篇 - 结构化上下文

### 第4章：提示技巧（Prompting Techniques）

**学习目标**：
- 掌握经典提示技术：CoT、Few-Shot
- 学习高级策略：ToT、ReAct
- 理解Token效率优化技巧

**核心内容**：
1. Chain of Thought (CoT)：让模型"展示工作"
2. Few-Shot Prompting：示例的选择与排列
3. Tree of Thoughts (ToT)：探索多条推理路径
4. ReAct：推理与行动的交替循环
5. 结合CoT与Few-shot的最佳实践

**参考资源**：

| 资源名称 | 类型 | 链接 | 核心价值 |
|---------|------|------|----------|
| Anthropic User Guides - Prompt Engineering | 官方文档 | https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview | XML Tags、Prefill预填、Quote Citations |
| OpenAI Cookbook | GitHub | https://github.com/openai/openai-cookbook | Token counting、Function Calling |
| Lilian Weng - Prompt Engineering / LLM Agents | 博客 | https://lilianweng.github.io/posts/2023-06-23-agent/ | CoT、ReAct分类框架 |
| PromptingGuide.ai - Context Engineering Guide | 指南 | https://www.promptingguide.ai/guides/context-engineering-guide | 完整代码示例 |

---

### 第5章：查询增强与意图识别（Query Augmentation & Intent Recognition）

**学习目标**：
- 理解查询处理在上下文工程中的重要性
- 掌握查询重写、扩展、分解三大技术
- 学习意图识别作为上下文构建的前置条件
- 掌握意图驱动的上下文策略选择

**核心内容**：
1. 查询重写（Query Rewriting）：将模糊输入转换为精确术语
2. 查询扩展（Query Expansion）：生成多个相关查询
3. 查询分解（Query Decomposition）：将复杂问题分解为子查询
4. 查询Agent：智能处理整个查询管道
5. **意图识别与路由**：
   - 意图分类（Intent Classification）：识别用户真实需求
   - 槽位填充（Slot Filling）：提取关键参数
   - 多意图识别与消歧：处理复合请求
   - 意图驱动的上下文策略：不同意图 → 不同工具/知识库/检索策略

**意图识别在上下文工程中的作用**：

```
用户输入 → 【意图识别】→ 决定上下文构建策略
                ↓
        ┌───────┴───────┐
        ↓               ↓
   事实性问题      开放性讨论
        ↓               ↓
   RAG精确检索     长上下文对话
```

| 意图类型 | 上下文策略 | 示例 |
|---------|-----------|------|
| 信息查询 | RAG检索 + 精确匹配 | "北京今天天气如何？" |
| 任务执行 | 工具调用 + 参数提取 | "帮我订一张去上海的机票" |
| 闲聊对话 | 短期记忆 + 人格设定 | "你觉得今天过得怎么样？" |
| 复杂推理 | CoT + 长上下文 | "分析这份报告的关键趋势" |
| 多意图 | 并行处理 + 结果合并 | "查天气顺便帮我订机票" |

**参考资源**：

| 资源名称 | 类型 | 链接 | 核心价值 |
|---------|------|------|----------|
| LangChain - The Rise of Context Engineering | 博客 | https://blog.langchain.com/the-rise-of-context-engineering/ | 从"提示"到"上下文"的转变 |
| LangChain - Context Engineering for Agents | 博客 | https://blog.langchain.com/context-engineering-for-agents/ | ★★★★★ 写入、选择、压缩、隔离四大策略 |
| langchain-ai/how_to_fix_your_context | GitHub | https://github.com/langchain-ai/how_to_fix_your_context | 配套代码示例 |
| Weaviate - The Context Engineering Guide (电子书) | 电子书 | https://weaviate.io/ebooks/the-context-engineering-guide | 架构模式指南 |
| Semantic Router | GitHub | https://github.com/aurelio-labs/semantic-router | ★★★★☆ 基于语义的快速意图路由 |
| LangChain Router Chains | 文档 | https://python.langchain.com/docs/how_to/routing/ | 意图路由实践 |
| Rasa NLU 意图识别 | 开源框架 | https://rasa.com/docs/rasa/nlu-training-data/ | 经典意图分类实现 |

---

## 第三部分：策略篇 - 动态上下文管理

### 第6章：检索系统（Retrieval/RAG）

**学习目标**：
- 理解RAG的核心架构
- 掌握分块（Chunking）策略的设计原则
- 学习检索精度与上下文丰富性的平衡

**核心内容**：
1. RAG架构：检索增强生成的完整流程
2. 简单分块技术：固定大小、递归、文档结构
3. 高级分块技术：语义分块、LLM分块、Agentic分块、层次分块、延迟分块
4. 预分块 vs 后分块的权衡
5. 混合检索：关键词 + 语义 + 图检索

**参考资源**：

| 资源名称                                   | 类型     | 链接                                                             | 核心价值                                   |
| -------------------------------------- | ------ | -------------------------------------------------------------- | -------------------------------------- |
| LlamaIndex High-Level Concepts         | 官方文档   | https://docs.llamaindex.ai/en/stable/getting_started/concepts/ | Sentence Window、Auto-Merging Retrieval |
| Weaviate Chunking Strategies Blog      | 博客     | https://weaviate.io/blog/chunking-strategies-for-rag           | 分块策略详解                                 |
| Codecademy - Context Engineering in AI | 指南     | https://www.codecademy.com/article/context-engineering-in-ai   | RAG和记忆系统的完整实现                          |
| phodal/build-agent-context-engineering | GitHub | https://github.com/phodal/build-agent-context-engineering      | ★★★★★ 中文最系统，关键词+语义+图检索                 |

---

### 第7章：记忆系统（Memory）

**学习目标**：
- 理解短期记忆与长期记忆的区别
- 掌握混合记忆架构的设计
- 学习记忆管理的关键原则

**核心内容**：
1. 短期记忆：即时工作空间，上下文学习
2. 长期记忆：情节记忆 vs 语义记忆
3. 混合记忆设置：工作记忆、程序记忆
4. 记忆管理原则：修剪和细化、选择性存储、检索优化
5. "脑洗"Agent：主动清理内存，丢弃噪音

**参考资源**：

| 资源名称 | 类型 | 链接 | 核心价值 |
|---------|------|------|----------|
| Generative Agents (斯坦福小镇) | 论文 | https://arxiv.org/abs/2304.03442 | Memory Stream → Retrieval → Reflection |
| MemGPT: Towards LLMs as Operating Systems | 论文 | https://arxiv.org/abs/2310.08560 | 虚拟内存概念，Main vs External Context |
| Brainwash Your Agent | 博客 | https://www.camel-ai.org/blogs/brainwash-your-agent-how-we-keep-the-memory-clean | ★★★★☆ 内存管理技巧 |
| LangChain LangMem | 文档 | - | 短期/长期记忆实现 |

---

## 第四部分：架构篇 - Agent与工具系统

### 第8章：Agent架构设计

**学习目标**：
- 理解AI Agent的核心特性
- 掌握单Agent与多Agent架构的选择
- 学习Agent的核心策略与任务管理

**核心内容**：
1. Agent的四大能力：动态决策、维护状态、自适应工具使用、基于结果修改方法
2. 单Agent vs 多Agent架构
3. 上下文窗口的挑战与管理策略
4. Agent核心策略：总结、验证、修剪、自适应检索、卸载、动态工具选择

**参考资源**：

| 资源名称 | 类型 | 链接 | 核心价值 |
|---------|------|------|----------|
| Agentic Context Engineering: Evolving Contexts | 论文 | https://arxiv.org/abs/2510.04618 | ACE框架，上下文作为演化"剧本" |
| Context Engineering for Multi-Agent Code Assistants | 论文 | https://arxiv.org/abs/2508.08322 | 多Agent代码助手工作流 |
| Context Engineering for AI Agents in OSS | 论文 | https://arxiv.org/abs/2510.21413 | 466个OSS项目的AGENTS.md分析 |
| ginobefun/agentic-design-patterns-cn | GitHub | https://github.com/ginobefun/agentic-design-patterns-cn | ★★★★☆ 设计模式中英对照 |

---

### 第9章：工具集成（Tools）

**学习目标**：
- 理解从提示到行动的演变
- 掌握函数调用与工具链设计
- 学习思考-行动-观察循环

**核心内容**：
1. 函数调用（Function Calling）/ 工具调用（Tool Calling）
2. 工具链设计：多工具编排
3. 编排挑战：发现、选择、参数制定、反思
4. 思考-行动-观察循环（Thought-Action-Observation Cycle）
5. MCP（Model Context Protocol）标准

**参考资源**：

| 资源名称 | 类型 | 链接 | 核心价值 |
|---------|------|------|----------|
| Model Context Protocol (MCP) 介绍 | 博客 | https://humanloop.com/blog/mcp | 工具集成的新标准 |
| Elysia Agentic RAG框架 | 博客 | https://weaviate.io/blog/elysia-agentic-rag | 编排框架实战案例 |
| AWS中国 - Context Engineering | 博客 | https://aws.amazon.com/cn/blogs/china/agentic-ai-infrastructure-practice-series-nine-context-engineering/ | 企业级基础设施视角 |

---

### 第10章：12-Factor Agents 工程化原则

**学习目标**：
- 理解生产级Agent的工程化原则
- 掌握"拥有你的上下文窗口"的核心理念
- 学习成本优化与状态管理

**核心内容**：
1. 12条核心原则（类比12-Factor App）
2. Factor 3：拥有你的上下文窗口（最关键）
3. 80%确定性代码 + 20% LLM调用
4. 成本呈二次增长，必须优化
5. 状态管理、暂停/恢复、人机协同

**参考资源**：

| 资源名称 | 类型 | 链接 | 核心价值 |
|---------|------|------|----------|
| 12-Factor Agents | GitHub | https://github.com/humanlayer/12-factor-agents | ★★★★★ 生产级方法论 |
| AI Engineer World's Fair Talk | YouTube | https://www.youtube.com/watch?v=8kMaTybvDUw | 现场讲解视频 |

---

## 第五部分：进阶篇 - 编程化与自动化

### 第11章：自动化与编译式上下文

**学习目标**：
- 理解从"提示"到"编译"的范式转变
- 掌握DSPy的核心理念
- 学习Many-Shot In-Context Learning

**核心内容**：
1. DSPy：将上下文视为可优化的"参数"
2. 自动提示优化（APO）
3. Many-Shot ICL：百万级上下文模型的海量示例效果
4. 上下文工程从"玄学"走向"科学"

**参考资源**：

| 资源名称 | 类型 | 链接 | 核心价值 |
|---------|------|------|----------|
| DSPy: Compiling Declarative LM Calls | 论文/库 | https://arxiv.org/abs/2310.03714 / https://github.com/stanfordnlp/dspy | ★★★★★ 编译式上下文 |
| Many-Shot In-Context Learning | 论文 | https://arxiv.org/abs/2404.11018 | 海量示例效果超过微调 |
| Context Engineering with DSPy Tutorial | 教程 | https://towardsdatascience.com/context-engineering-a-comprehensive-hands-on-tutorial-with-dspy/ | DSPy实战+1小时20分视频 |

---

### 第12章：安全与攻防

**学习目标**：
- 理解Prompt Injection攻击原理
- 掌握上下文隔离策略
- 学习安全的上下文架构设计

**核心内容**：
1. Prompt Injection（提示注入）攻击
2. 上下文隔离策略
3. 分隔符（Delimiters）防御
4. 安全的上下文架构设计

**参考资源**：

| 资源名称 | 类型 | 链接 | 核心价值 |
|---------|------|------|----------|
| Simon Willison - Prompt Injection | 博客 | https://simonwillison.net/2022/Sep/12/prompt-injection/ | 提示注入攻击原理与防御 |
| Gartner - Context Engineering | 报告 | https://www.gartner.com/en/articles/context-engineering | 商业落地与安全考量 |

---

## 第六部分：实战篇 - 企业案例与项目

### 第13章：企业级实战案例

**学习目标**：
- 学习Manus的生产级经验
- 掌握KV缓存优化策略
- 理解长任务Agent的优化技巧

**核心内容**：
1. **Manus案例**：
   - KV缓存优化：只追加不修改，命中率提升10倍
   - 工具遮蔽：动态显示/隐藏工具
   - 文件系统作为上下文
   - 复述操控注意力
   - 保留错误信息
2. **Anthropic Claude Code案例**：
   - 上下文压缩、笔记系统、子Agent隔离
3. **成本优化**：
   - 多轮成本呈二次增长（50轮=$2.5，100轮=$100）
   - KV缓存命中可降低90%成本

**参考资源**：

| 资源名称 | 类型 | 链接 | 核心价值 |
|---------|------|------|----------|
| Manus - Context Engineering for AI Agents | 博客 | https://manus.im/zh-cn/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus | ★★★★★ 企业级坑点 |
| Anthropic - Effective Context Engineering | 官方指南 | https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents | Claude团队实践 |
| CSDN《上下文工程才是核心竞争力》 | 中文博客 | https://blog.csdn.net/Baihai_IDP/article/details/149437955 | 中文实战案例 |

---

### 第14章：完整项目实战

**学习目标**：
- 从零构建完整的上下文工程系统
- 掌握结构化提示词、RAG、工具系统、多Agent的完整链路
- 学习GitHub Copilot等产品的上下文策略

**核心内容**：
1. 结构化提示词工程：输入/输出结构化、链式设计、路由分发
2. 上下文工程与RAG：查询改写、HyDE、重排序
3. 工具系统设计：语义清晰、无状态、原子性、最小权限、MCP协议
4. Agent规划与多Agent：预先分解 vs 交错分解、记忆系统、自我完善
5. 经典案例分析：GitHub Copilot上下文优先级排序、Cursor Rule设计

**参考资源**：

| 资源名称 | 类型 | 链接 | 核心价值 |
|---------|------|------|----------|
| phodal/build-agent-context-engineering | GitHub | https://github.com/phodal/build-agent-context-engineering | ★★★★★ 中文最系统实战 |
| davidkimai/Context-Engineering | GitHub | https://github.com/davidkimai/Context-Engineering | ★★★★★ 最佳学习路径，4.1K+ Stars |
| WakeUp-Jin/Practical-Guide-to-Context-Engineering | GitHub | https://github.com/WakeUp-Jin/Practical-Guide-to-Context-Engineering | 七类上下文分解 |
| Awesome-Context-Engineering | GitHub | https://github.com/Meirtz/Awesome-Context-Engineering | 数百篇论文、框架和实现指南 |

---

## 附录：资源索引

### A. 完整资源清单

#### 学术论文

| 论文名称 | arXiv编号 | 核心价值 |
|---------|----------|----------|
| Attention Is All You Need | 1706.03762 | Transformer基础 |
| Language Models are Few-Shot Learners (GPT-3) | 2005.14165 | 上下文学习定义 |
| Rethinking the Role of Demonstrations | 2202.12837 | 示例作用研究 |
| Lost in the Middle | 2307.03172 | 长上下文陷阱 |
| Generative Agents (斯坦福小镇) | 2304.03442 | 记忆系统架构 |
| MemGPT | 2310.08560 | 虚拟内存概念 |
| DSPy | 2310.03714 | 编译式上下文 |
| Many-Shot In-Context Learning | 2404.11018 | 海量示例效果 |
| A Survey of Context Engineering for LLMs | 2507.13334 | ★★★★★ 1400+论文综述 |
| Context Engineering 2.0 | 2510.26493 | 熵减理论框架 |
| Agentic Context Engineering | 2510.04618 | ACE框架 |
| Context Engineering for Multi-Agent | 2508.08322 | 多Agent代码助手 |
| Context Engineering for AI Agents in OSS | 2510.21413 | OSS项目分析 |

#### 官方文档与指南

| 资源名称 | 链接 |
|---------|------|
| Anthropic - Effective Context Engineering | https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents |
| Anthropic User Guides - Prompt Engineering | https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview |
| OpenAI Cookbook | https://github.com/openai/openai-cookbook |
| LlamaIndex High-Level Concepts | https://docs.llamaindex.ai/en/stable/getting_started/concepts/ |
| PromptingGuide.ai | https://www.promptingguide.ai/guides/context-engineering-guide |

#### 博客与文章

| 资源名称 | 作者/机构 | 链接 |
|---------|----------|------|
| The New Skill in AI is Not Prompting | Phil Schmid | https://www.philschmid.de/context-engineering |
| How Contexts Fail—and How to Fix Them | Drew Breunig | https://www.dbreunig.com/2025/06/22/how-contexts-fail-and-how-to-fix-them.html |
| Practical Tips on Building LLM Agents | Paras Chopra | https://letters.lossfunk.com/p/practical-tips-on-building-llm-agents |
| The Rise of Context Engineering | LangChain | https://blog.langchain.com/the-rise-of-context-engineering/ |
| Context Engineering for Agents | LangChain | https://blog.langchain.com/context-engineering-for-agents/ |
| Lilian Weng - LLM Agents | Lilian Weng | https://lilianweng.github.io/posts/2023-06-23-agent/ |
| Brainwash Your Agent | CAMEL-AI | https://www.camel-ai.org/blogs/brainwash-your-agent-how-we-keep-the-memory-clean |
| Manus Context Engineering | Manus | https://manus.im/zh-cn/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus |
| Prompt Injection | Simon Willison | https://simonwillison.net/2022/Sep/12/prompt-injection/ |

#### GitHub仓库

| 仓库名称 | Stars | 链接 |
|---------|-------|------|
| davidkimai/Context-Engineering | 4.1K+ | https://github.com/davidkimai/Context-Engineering |
| Meirtz/Awesome-Context-Engineering | - | https://github.com/Meirtz/Awesome-Context-Engineering |
| phodal/build-agent-context-engineering | - | https://github.com/phodal/build-agent-context-engineering |
| humanlayer/12-factor-agents | - | https://github.com/humanlayer/12-factor-agents |
| langchain-ai/how_to_fix_your_context | - | https://github.com/langchain-ai/how_to_fix_your_context |
| stanfordnlp/dspy | - | https://github.com/stanfordnlp/dspy |
| WakeUp-Jin/Practical-Guide-to-Context-Engineering | - | https://github.com/WakeUp-Jin/Practical-Guide-to-Context-Engineering |
| ginobefun/agentic-design-patterns-cn | - | https://github.com/ginobefun/agentic-design-patterns-cn |
| GAIR-NLP/Context-Engineering-2.0 | - | https://github.com/GAIR-NLP/Context-Engineering-2.0 |
| aurelio-labs/semantic-router | - | https://github.com/aurelio-labs/semantic-router |

#### 意图识别与路由资源

| 资源名称 | 类型 | 链接 |
|---------|------|------|
| Semantic Router | GitHub | https://github.com/aurelio-labs/semantic-router |
| LangChain Router Chains | 文档 | https://python.langchain.com/docs/how_to/routing/ |
| Rasa NLU 意图识别 | 开源框架 | https://rasa.com/docs/rasa/nlu-training-data/ |
| DialogFlow Intent Detection | 云服务 | https://cloud.google.com/dialogflow/docs |

#### 中文资源

| 资源名称 | 链接 |
|---------|------|
| AWS中国 - Agentic AI Context Engineering | https://aws.amazon.com/cn/blogs/china/agentic-ai-infrastructure-practice-series-nine-context-engineering/ |
| 知乎《上下文工程：将工程规范引入提示》 | https://zhuanlan.zhihu.com/p/1928378624261731252 |
| CSDN《上下文工程才是核心竞争力》 | https://blog.csdn.net/Baihai_IDP/article/details/149437955 |
| 微信文章 | https://mp.weixin.qq.com/s/KbviOJ6q-K4ik_wzsUs2dw |

#### 电子书与视频

| 资源名称 | 类型 | 链接 |
|---------|------|------|
| Weaviate - The Context Engineering Guide | 电子书 | https://weaviate.io/ebooks/the-context-engineering-guide |
| Context Engineering: The Outer Loop | 视频 | YouTube - Hammad Bashir (Chroma CTO) |
| 12-Factor Agents Talk | 视频 | https://www.youtube.com/watch?v=8kMaTybvDUw |
| DSPy Tutorial | 视频 | https://towardsdatascience.com/context-engineering-a-comprehensive-hands-on-tutorial-with-dspy/ |

---

### B. 学习路径推荐

#### 入门路径（1-2周）

```
1. Phil Schmid 概念文章 → 建立框架
2. Drew Breunig 失败模式 → 避坑
3. LangChain 四大策略 → 建立方法论
```

#### 进阶路径（2-4周）

```
4. phodal 中文实战仓库 → 系统实践
5. 12-Factor Agents → 工程化原则
6. Manus/Anthropic 企业案例 → 生产经验
```

#### 专家路径（1-3个月）

```
7. 1400+论文综述 → 全景视野
8. davidkimai 完整学习路径 → 深度优化
9. 贡献开源项目 → 社区参与
```

---

### C. 核心技术要点速查

#### 上下文工程四大支柱

1. **意图理解**：意图分类、槽位填充、多意图消歧、意图路由
2. **信息检索**：RAG、混合检索、Agentic检索
3. **记忆管理**：短期/长期记忆、反思机制
4. **工具编排**：MCP协议、工具路由、并行调用

#### 生产环境最佳实践

- ✅ 任务原子化（10-15分钟粒度）
- ✅ 使用KV缓存（只追加不修改）
- ✅ 建立验证系统（每步明确成功/失败）
- ✅ 工具单一职责（原子性+语义清晰）
- ✅ 上下文压缩与隔离（总结+剪枝+沙盒）
- ❌ 避免上下文中毒和冲突
- ❌ 防止过度依赖长上下文

#### 成本优化关键指标

- 多轮Agent成本呈**二次增长**（50轮=$2.5，100轮=$100）
- KV缓存命中可降低**90%成本**
- 上下文总结可减少**60-80%令牌**

---

### D. 资源价值矩阵

| 资源 | 理论深度 | 实践性 | 代码质量 | 适合人群 |
|------|---------|--------|---------|---------|
| Phil Schmid文章 | ⭐⭐⭐ | ⭐⭐⭐⭐ | - | 所有人 |
| LangChain博客 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 开发者 |
| 12-Factor Agents | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 工程师 |
| phodal仓库 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 中文开发者 |
| Drew Breunig文章 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | - | 调试者 |
| 1400+论文综述 | ⭐⭐⭐⭐⭐ | ⭐⭐ | - | 研究者 |
| davidkimai仓库 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 深度学习者 |
| Manus案例 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 企业团队 |

---

**文档版本**：v1.1  
**最后更新**：2025年12月  
**资源统计**：论文13篇 + 官方文档5项 + 博客9篇 + GitHub仓库10个 + 中文资源4项 + 电子书/视频4项 + 意图识别资源4项 = **49项资源**

---

> 📌 **使用建议**：本大纲可作为教程的骨架，每章节可根据目标读者的水平进行扩展。建议配合代码示例和实战项目，让读者在学习理论的同时获得动手实践的机会。

