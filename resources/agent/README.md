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

### 1. LangGraph ⭐⭐⭐⭐⭐ 最推荐

- **链接**：https://github.com/langchain-ai/langgraph
- **Stars**：15k+
- **简介**：LangChain 团队出品，用图结构构建弹性 Agent
- **特点**：状态管理、可视化工作流、灵活编排
- **适合场景**：复杂 Agent 工作流、Multi-Agent 系统
- **学习难度**：⭐⭐⭐⭐
- **AgentGuide 教程**：[LangGraph 核心概念](../../docs/02-tech-stack/04-langchain-guide.md)

### 2. AutoGen ⭐⭐⭐⭐⭐

- **链接**：https://github.com/microsoft/autogen
- **Stars**：30k+
- **简介**：微软开源的多智能体对话框架
- **特点**：Multi-Agent 协作、可视化 Studio、代码执行
- **适合场景**：多智能体系统、角色扮演型 Agent
- **学习难度**：⭐⭐⭐
- **官网**：https://autogen-studio.com/

### 3. CrewAI ⭐⭐⭐⭐

- **链接**：https://github.com/joaomdmoura/crewAI
- **Stars**：20k+
- **简介**：角色扮演型自主 AI Agent 框架
- **特点**：角色定义、任务分工、简单易用
- **适合场景**：明确角色分工的协作系统
- **学习难度**：⭐⭐⭐

### 4. MetaGPT ⭐⭐⭐⭐

- **链接**：https://github.com/geekan/MetaGPT
- **Stars**：40k+
- **简介**：软件公司多角色协作框架
- **特点**：模拟软件公司流程（PM、工程师、测试）
- **适合场景**：软件开发流程自动化
- **学习难度**：⭐⭐⭐

### 5. Swarm ⭐⭐⭐

- **链接**：https://github.com/openai/swarm
- **简介**：OpenAI 官方的轻量级 Multi-Agent 框架
- **特点**：极简设计，适合学习
- **适合场景**：学习 Agent 基础概念
- **学习难度**：⭐⭐

### 6. AutoGPT ⭐⭐⭐

- **链接**：https://github.com/Significant-Gravitas/AutoGPT
- **Stars**：160k+
- **简介**：早期的自主 Agent 框架
- **特点**：完全自主，目标驱动
- **学习难度**：⭐⭐⭐

### 7. Microsoft Agent Framework ⭐⭐⭐

- **链接**：https://github.com/microsoft/agent-framework
- **简介**：微软官方框架，支持 Python 和 .NET
- **特点**：企业级、跨语言
- **适合场景**：企业级应用
- **学习难度**：⭐⭐⭐⭐

### 8. OWL ⭐⭐⭐

- **链接**：https://github.com/camel-ai/owl
- **简介**：优化工作流学习框架
- **特点**：任务自动化、工作流优化
- **学习难度**：⭐⭐⭐

### 9. Parlant ⭐⭐⭐⭐ 🔥 新框架

- **链接**：https://github.com/emcie-co/parlant
- **简介**：确保 Agent 指令遵循的框架，不再依赖 LLM"可能"遵循，而是"确保"必定遵循
- **核心技术**：ARQs（Attentive Reasoning Queries - 注意力推理查询）
- **特点**：
  - 通过预定义推理步骤确保指令遵循
  - 在 87 个测试场景中成功率达 90.2%（高于 CoT 的 86.05%）
  - 针对性预防领域常见失效模式
- **适合场景**：有明确领域指南的场景（客户服务、医疗咨询）
- **学习难度**：⭐⭐⭐
- **关键资源**：
  - [Parlant vs LangGraph](https://www.parlant.io/blog/parlant-vs-langgraph)
  - [Parlant vs DSPy](https://www.parlant.io/blog/parlant-vs-dspy)
  - [ARQs 论文](https://arxiv.org/pdf/2503.03669v1)

### 10. AgentScope ⭐⭐⭐⭐

- **链接**：https://github.com/agentscope-ai/agentscope
- **简介**：面向 Agent 的模块化编程框架
- **特点**：
  - 模块化设计，易于维护和扩展
  - 灵活的集成能力
  - 企业级支持（阿里云等背书）
  - 详细的文档和示例
- **适合场景**：模块化 Agent 系统、企业级应用
- **学习难度**：⭐⭐⭐

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
- **特点**：GPT-4V 驱动
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

- **文档**：[ai-agent-production-challenges.md](./ai-agent-production-challenges.md)
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
