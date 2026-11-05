---
type: 入口页
status: 已发布
level: 通用
topic:
  - Agent
  - 框架工具
---

# Awesome Agent - AI Agent 开发核心资源

> **精选** AI Agent 开发的**核心资源**（只推荐最重要的！）

## 📌 定位说明

- ✅ 只收录 Agent 开发**直接相关**的资源
- ✅ 只推荐**面试会考**、**项目会用**的
- ✅ 每个资源都标注**适合场景**和**学习难度**

**🔗 需要更全面的资源？**  
查看作者的另一个项目：[Awesome-Awesome-LLM](https://github.com/adongwanai/Awesome-Awesome-LLM)  
（涵盖 LLM 全栈的 200+ Awesome 系列资源）

---

**⭐ 本文档持续更新，欢迎贡献！**

---

## 📑 目录

- [Agent 开发框架](#-agent-开发框架)
- [工具调用](#-工具调用-tool-use)
- [记忆模块](#-记忆模块-memory)
- [GUI Agent / Web Agent](#️-gui-agent--web-agent)
- [评估 Benchmark](#-评估-benchmark)
- [评估工具 (Evaluation Harness)](#-评估工具-evaluation-harness)
- [必读论文](#-必读论文)
- [学习资源](#-学习资源)
- [相关文档](#-相关文档)

---

## 🏗️ Agent 开发框架

> **详细框架对比见**：[Agent 开发框架推荐](./frameworks.md)

> 核对时间 2026-11。AutoGen 已进入维护模式（新项目用 Microsoft Agent Framework），OpenAI Swarm 已被 Agents SDK 取代，LangChain 1.x 改用 `create_agent`。

### 编排与 Agent 框架

| 框架 | 链接 | 定位 |
|:---|:---|:---|
| LangGraph | [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) | 有状态图编排，checkpoint、中断恢复、人工介入 |
| LangChain | [langchain-ai/langchain](https://github.com/langchain-ai/langchain) | `create_agent` + middleware，模型与工具集成最全 |
| <a id="microsoft-agent-framework"></a>Microsoft Agent Framework | [microsoft/agent-framework](https://github.com/microsoft/agent-framework) | 合并 AutoGen 与 Semantic Kernel，Python / .NET；适合企业工具集成、审批、遥测与持久状态。二开可接一个企业工具或工作流，验证循环终止、权限边界和状态恢复 |
| Pydantic AI | [pydantic/pydantic-ai](https://github.com/pydantic/pydantic-ai) | 类型化、可测试的 Python Agent 框架 |
| CrewAI | [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI) | 角色协作（Crew）+ 事件驱动流程（Flow） |
| AgentScope | [agentscope-ai/agentscope](https://github.com/agentscope-ai/agentscope) | 阿里通义，消息驱动的多 Agent 框架 |
| Parlant | [emcie-co/parlant](https://github.com/emcie-co/parlant) | 基于行为准则的合规对话 Agent，见 [深度解析](../../docs/02-tech-stack/22-parlant-agent-compliance-deep-dive.md) |

### 厂商 Agent SDK

| SDK | 链接 | 定位 |
|:---|:---|:---|
| OpenAI Agents SDK | [openai/openai-agents-python](https://github.com/openai/openai-agents-python) | Agent、handoff、guardrail、tracing；Swarm 的后继 |
| Claude Agent SDK | [anthropics/claude-agent-sdk-python](https://github.com/anthropics/claude-agent-sdk-python) | Claude Code 背后的 Agent 循环与工具集 |
| Google ADK | [google/adk-python](https://github.com/google/adk-python) | Google 的 Agent Development Kit |

### 多 Agent 研究项目

| 项目 | 链接 | 说明 |
|:---|:---|:---|
| MetaGPT | [FoundationAgents/MetaGPT](https://github.com/FoundationAgents/MetaGPT) | 用标准作业流程（SOP）模拟软件团队的多角色协作 |
| OWL / CAMEL | [camel-ai/owl](https://github.com/camel-ai/owl) | 基于 CAMEL 的多 Agent 协作，面向通用任务自动化 |

### 历史项目（了解即可）

- **AutoGen**（[microsoft/autogen](https://github.com/microsoft/autogen)）：维护模式，仅接受修复类贡献。
- **Swarm**（[openai/swarm](https://github.com/openai/swarm)）：已被 Agents SDK 取代，源码很短，可用来理解 handoff 概念。
- **AutoGPT**（[Significant-Gravitas/AutoGPT](https://github.com/Significant-Gravitas/AutoGPT)）：2023 年“自主 Agent”热潮的代表，现已转型为 Agent 平台；适合作为 Agent 发展史阅读。

---

## 🔧 工具调用 (Tool Use)

### 1. ToolBench ⭐⭐⭐⭐

- **链接**：https://github.com/OpenBMB/ToolBench
- **简介**：工具学习基准测试
- **包含**：16000+ 真实 API、工具调用数据集
- **适合场景**：工具调用评估和测试

### 2. Gorilla ⭐⭐⭐⭐

- **链接**：https://github.com/ShishirPatil/gorilla
- **简介**：大模型 API 调用优化
- **特点**：专注于提升工具调用准确性
- **适合场景**：提升 Agent 工具调用能力

### 3. ToolLLM ⭐⭐⭐⭐

- **链接**：https://github.com/OpenBMB/ToolLLM
- **简介**：工具学习的 LLM 训练
- **适合场景**：训练专门的工具调用模型

---

## 💾 记忆模块 (Memory)

> **详细记忆系统设计见**：[Agent 记忆系统](./memory.md)

### 1. Mem0 ⭐⭐⭐⭐⭐

- **链接**：https://github.com/mem0ai/mem0
- **Stars**：20k+
- **简介**：轻量级 Agent 记忆模块
- **特点**：简单易用、支持多种后端
- **适合场景**：快速集成记忆功能
- **注意**：社区反馈有稳定性问题，使用前需验证

### 2. MemGPT ⭐⭐⭐⭐

- **链接**：https://github.com/cpacker/MemGPT
- **简介**：长期记忆管理系统
- **特点**：虚拟内存机制
- **适合场景**：需要长期记忆的 Agent

### 3. Zep ⭐⭐⭐⭐

- **链接**：https://github.com/getzep/zep
- **简介**：长期记忆存储
- **特点**：企业级、可扩展
- **适合场景**：生产环境的记忆管理

### 4. LangChain Memory

- **文档**：https://python.langchain.com/docs/modules/memory/
- **简介**：LangChain 内置记忆模块
- **特点**：多种记忆类型
- **适合场景**：LangChain 生态内使用

---

## 🕷️ GUI Agent / Web Agent

### 1. AppAgent ⭐⭐⭐⭐

- **链接**：https://github.com/mnotgod96/AppAgent
- **简介**：移动应用 Agent
- **特点**：自主操作手机应用
- **适合场景**：移动端自动化

### 2. SeeAct ⭐⭐⭐⭐

- **链接**：https://github.com/OSU-NLP-Group/SeeAct
- **简介**：视觉理解 + 网页操作
- **特点**：原论文以 GPT-4V 为基座，重点研究多模态模型对网页元素的定位（grounding）问题
- **适合场景**：多模态 Web Agent

### 3. WebShop ⭐⭐⭐⭐

- **链接**：https://github.com/princeton-nlp/WebShop
- **简介**：在线购物 Agent 基准测试
- **适合场景**：电商场景的 Agent 评估

### 4. Mind2Web ⭐⭐⭐⭐

- **链接**：https://github.com/OSU-NLP-Group/Mind2Web
- **简介**：网页任务理解数据集
- **适合场景**：Web Agent 训练和评估

### 5. WebArena ⭐⭐⭐⭐

- **链接**：https://webarena.dev/
- **简介**：Web Agent 评估基准
- **适合场景**：综合评估 Web Agent 能力

---

## 📊 评估 Benchmark

### Agent 综合评估

| Benchmark | 简介 | 链接 |
|-----------|------|------|
| **GAIA** | 通用 AI Agent 评估，测试真实任务完成能力 | [论文](https://arxiv.org/abs/2311.12983) |
| **AgentBench** | Agent 综合能力评估，涵盖多种任务类型 | [GitHub](https://github.com/THUDM/AgentBench) |
| **WebArena** | Web Agent 评估，真实网站交互测试 | [官网](https://webarena.dev/) |
| **ToolBench** | 工具使用评估，测试 API 调用能力 | [GitHub](https://github.com/OpenBMB/ToolBench) |

---

## 🛠️ 评估工具 (Evaluation Harness)

> **详细工具对比见**：[Evaluation Harness 工具选型指南](./evaluation-harness.md)

### 快速选型

| 场景 | 推荐工具 | 说明 |
|------|---------|------|
| 学术研究 | lm-evaluation-harness / OpenCompass | 支持 200+ 标准任务 |
| 工程开发 | Promptfoo / DeepEval | 轻量、易集成 CI/CD |
| Agent 评估 | AgentBench / WebArena | 真实环境交互测试 |
| RAG 评估 | RAGAs / TruLens | 专注检索增强生成质量 |

---

## 📚 必读论文

### 核心论文

#### 1. ReAct ⭐⭐⭐⭐⭐

- **论文**：[Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- **必读原因**：Agent 架构基础，推理与行动的结合
- **核心思想**：将推理（Reasoning）和行动（Acting）交织进行
- **发表会议**：ICLR 2023

#### 2. Reflexion ⭐⭐⭐⭐⭐

- **论文**：[Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)
- **必读原因**：自我反思机制，让 Agent 从错误中学习
- **核心思想**：通过语言反馈进行自我改进
- **发表会议**：NeurIPS 2023

#### 3. Toolformer ⭐⭐⭐⭐

- **论文**：[Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761)
- **必读原因**：工具调用的开创性工作
- **核心思想**：LLM 自学使用外部工具

#### 4. HuggingGPT ⭐⭐⭐⭐

- **论文**：[HuggingGPT: Solving AI Tasks with ChatGPT and its Friends](https://arxiv.org/abs/2303.17580)
- **必读原因**：大模型作为控制器调度其他模型
- **核心思想**：LLM 作为任务规划器和模型调度器

#### 5. ARQs（Attentive Reasoning Queries）⭐⭐⭐⭐ 🔥 新

- **论文**：[Attentive Reasoning Queries: A Systematic Method for Optimizing Instruction-Following in Large Language Models](https://arxiv.org/pdf/2503.03669v1)
- **代码**：https://github.com/emcie-co/parlant/tree/arqs-a-systematic-method-for-optimizing-instruction-following-in-llms
- **必读原因**：系统性方法确保 LLM 指令遵循
- **核心思想**：
  - 通过领域专用推理步骤引导 LLM
  - 使用结构化 JSON 模式的目标查询
  - 三阶段流程：引导 ARQ → 响应生成 → 响应验证
- **实验效果**：在 87 个测试场景中成功率达 90.2%（高于 CoT 的 86.05%）
- **适用场景**：有明确领域指南、失效模式清晰的场景（客服、医疗）

---

## 🎓 学习资源

### 实战经验

#### 1. AI Agent 生产环境实践：挑战与解决方案 ⭐⭐⭐⭐⭐ 强烈推荐

- **文档**：[Agent 生产环境实践](../../docs/03-practice/06-agent-production-challenges.md)
- **核心内容**：基于 12+ 个生产级 Agent 的实战经验
- **关键点**：误差累积、成本问题、工具工程、系统集成
- **适合人群**：想要构建生产级 Agent 的开发者
- **必读原因**：揭示 Agent 的数学局限性和工程最佳实践

### 官方指南

#### 1. Anthropic - Building Effective Agents ⭐⭐⭐⭐⭐

- **链接**：https://www.anthropic.com/engineering/building-effective-agents
- **核心观点**：何时该用 Agent，何时不该用
- **重点内容**：Agent 设计模式、工作流编排
- **必读原因**：Claude 团队的实战经验总结

#### 2. OpenAI - A Practical Guide to Building Agents ⭐⭐⭐⭐

- **链接**：https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf
- **核心内容**：实用的 Agent 构建指南
- **必读原因**：OpenAI 官方最佳实践

#### 3. Agentic Design Patterns（中文版）⭐⭐⭐⭐

- **链接**：https://github.com/ginobefun/agentic-design-patterns-cn
- **核心内容**：Agent 设计模式详解
- **必读原因**：系统性介绍各种 Agent 架构模式

### 开源教程

#### 1. RAG from Scratch ⭐⭐⭐⭐

- **链接**：https://github.com/langchain-ai/rag-from-scratch
- **简介**：从零开始学习 RAG
- **适合人群**：想要深入理解 RAG 的开发者

#### 2. LangChain 中文教程 ⭐⭐⭐⭐

- **链接**：https://github.com/liaokongVFX/LangChain-Chinese-Getting-Started-Guide
- **简介**：LangChain 中文入门指南
- **适合人群**：中文开发者快速上手

#### 3. Build a Large Language Model (From Scratch) ⭐⭐⭐⭐⭐

- **链接**：https://github.com/rasbt/LLMs-from-scratch
- **简介**：从零构建大语言模型
- **适合人群**：想要深入理解 LLM 原理的开发者

---

## 🔗 相关文档

### 本项目其他资源

- [Harness 项目选型与二开](../project-catalogs/agent-harness-projects.md) - 按岗位目标选择项目并建立评测证据
- [Agent 开发框架对比](./frameworks.md) - 详细的框架选型指南
- [Agent 记忆系统设计](./memory.md) - 记忆模块实现方案
- [RAG 资源大全](../rag/README.md) - RAG 技术栈完整资源
- [AI 开发工具箱](../tools.md) - 开发工具和平台推荐

### 学习路线

- [Agent 开发学习路线](../../docs/05-roadmaps/learning-roadmap-development.md)
- [算法面试准备路线](../../docs/05-roadmaps/learning-roadmap-algorithm.md)

---

## 📝 贡献指南

**🙏 欢迎贡献**：如果你发现好的 Agent 相关资源，欢迎提 PR！

### 贡献要求

1. 资源必须与 Agent 开发**直接相关**
2. 提供清晰的**简介**和**使用场景**
3. 标注**学习难度**（⭐~⭐⭐⭐⭐⭐）
4. 优先推荐**面试会考**、**项目会用**的资源

---

## 📄 License

本文档采用 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) 协议。

---

**💡 提示**：本文档专注于 Agent 开发的核心资源。如需更全面的 LLM 资源，请访问 [Awesome-Awesome-LLM](https://github.com/adongwanai/Awesome-Awesome-LLM)。

**⭐ Star 支持**：如果这个资源对你有帮助，欢迎给项目点个 Star！

---

<!-- AUTO-GENERATED-CONTENT:START -->
## 完整内容索引

<!-- 下表由元数据生成，请勿手工编辑。 -->

| 文档 | 类型 | 状态 | 难度 | 主题 |
|:---|:---|:---|:---|:---|
| [🧪 Evaluation Harness - AI 评估工具箱](./evaluation-harness.md) | 资源清单 | 已发布 | 通用 | Agent、评测、框架工具 |
| [Agent 开发框架对比](./frameworks.md) | 资源清单 | 已发布 | 通用 | Agent、多智能体、框架工具 |
| [Agent Memory 资源精选](./memory.md) | 资源清单 | 已发布 | 通用 | Agent、记忆 |
| [AI Agent 权威学习指南](./official-guides.md) | 资源清单 | 已发布 | 通用 | Agent、框架工具 |
| [Agent Memory 核心论文解读](<./papers/agent_memory/Agent Memory 核心论文汇总.md>) | 论文清单 | 已发布 | 进阶 | Agent、科研、记忆 |
| [Agent Memory Papers](./papers/agent_memory/README.md) | 入口页 | 已发布 | 通用 | Agent、科研、记忆 |
| [一口气读完Agent Memory的21篇论文：从理论到实践的完整指南](<./papers/agent_memory/一口气读完agent memory的21篇核心论文.md>) | 论文清单 | 已发布 | 进阶 | Agent、科研、记忆 |
| [Agent RL 开源项目汇总](./papers/agent_rl/Agent➕RL开源项目汇总.md) | 论文清单 | 已发布 | 进阶 | Agent、科研、项目实战 |
| [Agent RL Papers](./papers/agent_rl/README.md) | 入口页 | 已发布 | 通用 | Agent、科研 |
| [Agent Papers](./papers/README.md) | 入口页 | 已发布 | 通用 | Agent、科研 |
<!-- AUTO-GENERATED-CONTENT:END -->
