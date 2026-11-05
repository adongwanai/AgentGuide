---
type: 资源清单
status: 已发布
level: 通用
topic:
  - Agent
  - 多智能体
  - 框架工具
---

# Agent 开发框架对比

> 内容核对时间：2026-11。框架版本和维护状态变化很快，本页只记录定位和选型依据，具体 API 以官方文档为准。Star 数变化快且不反映生产可用性，本页不再列出。

## 状态变化（2025—2026）

先说几件会直接影响学习路线的变化：

- **AutoGen 进入维护模式**。仓库声明不再开发新功能，由社区维护，新项目建议使用 [Microsoft Agent Framework](https://github.com/microsoft/agent-framework)。后者合并了 AutoGen 和 Semantic Kernel，2026 年 4 月发布 1.0，支持 Python 和 .NET。
- **OpenAI Swarm 被 [Agents SDK](https://github.com/openai/openai-agents-python) 取代**。Swarm 仓库明确说明 Agents SDK 是其面向生产的演进版本，建议迁移。
- **LangChain 1.x 重构了 Agent 接口**。统一使用 `create_agent` 和 middleware；`LLMChain`、`ConversationChain` 等旧链和旧 memory 类迁到 `langchain-classic` 包，短期记忆改由 LangGraph checkpointer 管理。网上大量基于 `initialize_agent`、`ConversationBufferMemory` 的教程已不适用。
- **模型厂商推出了自己的 Agent SDK**。OpenAI Agents SDK、[Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python)、Google [ADK](https://github.com/google/adk-python) 都把“模型 + 工具循环 + 上下文管理”封装成了开箱可用的 harness。

## 对比总表

| 框架 | 维护方 | 核心抽象 | 适合 | 状态 |
|:---|:---|:---|:---|:---|
| [LangGraph](https://github.com/langchain-ai/langgraph) | LangChain | 有状态图、checkpointer、中断恢复 | 需要精确控制流程、持久化和人工介入的工作流 | 活跃，1.x |
| [LangChain](https://github.com/langchain-ai/langchain) | LangChain | `create_agent` + middleware，构建在 LangGraph 上 | 快速搭建标准工具调用 Agent，集成大量模型和工具 | 活跃，1.x |
| [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) | OpenAI | Agent、handoff、guardrail、session、tracing | 轻量的多 Agent 交接和内置追踪 | 活跃 |
| [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python) | Anthropic | Claude Code 的 Agent 循环、工具、上下文压缩、子 Agent | 文件 / 终端类长任务，需要成熟 harness 的场景 | 活跃 |
| [Google ADK](https://github.com/google/adk-python) | Google | Agent、工具、多 Agent 层级、评测 | Google Cloud / Gemini 生态 | 活跃 |
| [Microsoft Agent Framework](https://github.com/microsoft/agent-framework) | Microsoft | Agent + 工作流图，Python / .NET | 微软技术栈、企业应用；AutoGen / Semantic Kernel 的迁移目标 | 活跃，1.x |
| [Pydantic AI](https://github.com/pydantic/pydantic-ai) | Pydantic | 类型化依赖注入和结构化输出 | 重视类型安全和测试的 Python 工程 | 活跃 |
| [CrewAI](https://github.com/crewAIInc/crewAI) | CrewAI | Crew（角色协作）+ Flow（事件驱动流程） | 角色分工明确的协作原型 | 活跃 |
| [LlamaIndex](https://github.com/run-llama/llama_index) | LlamaIndex | 数据连接、索引、Workflows | 以检索和文档处理为主的 Agent | 活跃 |
| [AgentScope](https://github.com/agentscope-ai/agentscope) | 阿里通义 | 消息驱动的多 Agent、ReAct Agent | 多 Agent 仿真与研究、国内模型生态 | 活跃 |
| [Parlant](https://github.com/emcie-co/parlant) | Emcie | 行为准则（guideline）匹配与执行 | 客服等需要严格遵守业务规则的对话 Agent | 活跃 |
| [AutoGen](https://github.com/microsoft/autogen) | Microsoft | 多 Agent 对话 | 阅读已有项目；新项目不建议 | 维护模式 |
| [Swarm](https://github.com/openai/swarm) | OpenAI | Agent + handoff 的最小实现 | 读源码理解 handoff 概念 | 已被 Agents SDK 取代 |

## 分类说明

### 编排框架：LangGraph、Microsoft Agent Framework

这类框架把 Agent 表示成显式的状态图或工作流：节点是模型调用、工具调用或人工审批，边是条件转移。优点是流程可控、状态可持久化（中断后从 checkpoint 恢复）、方便插入人工审核；代价是需要自己设计状态结构，简单场景下代码偏多。

适用信号：流程有固定的分支和回退、需要跨会话恢复、需要审计每一步。

### Agent 框架：LangChain 1.x、Pydantic AI、CrewAI、AgentScope

在“模型 + 工具循环”之上提供更高层的抽象：LangChain 用 middleware 处理摘要、人工介入、动态提示等横切逻辑；Pydantic AI 强调依赖注入和类型；CrewAI 用角色和任务描述多 Agent 协作；AgentScope 以消息传递组织多 Agent。

适用信号：主要逻辑就是“模型决定调用哪个工具”，希望少写样板代码。

### 厂商 SDK：OpenAI Agents SDK、Claude Agent SDK、Google ADK

厂商把自家产品中验证过的 Agent 循环开放出来，通常带有追踪、guardrail、上下文管理等配套能力，与自家模型的工具调用格式配合最好。大多也支持接入其他厂商的模型，但部分特性（如服务端工具、提示缓存）依赖原厂模型。

适用信号：已经确定主力模型；希望直接复用成熟的 harness，而不是自己实现上下文压缩、子 Agent 调度等细节。

### 领域框架：Parlant、LlamaIndex

Parlant 面向“必须遵守规则”的对话场景，把业务规则写成 guideline，由框架负责匹配和执行，详见 [Parlant 深度解析](../../docs/02-tech-stack/22-parlant-agent-compliance-deep-dive.md)。LlamaIndex 的优势在数据接入和检索，适合以 RAG 为核心的 Agent。

## 选型建议

| 需求 | 优先考虑 | 理由 |
|:---|:---|:---|
| 理解 Agent 原理 | 不用框架，先手写一个 ReAct 循环 | 框架会隐藏消息格式、工具调用解析、停止条件等关键细节，见 [最小 Agent Loop](../../examples/minimal-agent-loop.md) |
| 快速做标准工具调用 Agent | LangChain 1.x、Pydantic AI、厂商 SDK | 样板代码少 |
| 复杂流程、需要持久化与人工介入 | LangGraph、Microsoft Agent Framework | 显式状态与 checkpoint |
| 文件、终端、代码类长任务 | Claude Agent SDK、OpenAI Agents SDK | 现成的上下文管理和工具集 |
| 多 Agent 角色协作原型 | CrewAI、AgentScope | 角色抽象直接 |
| 强规则约束的客服对话 | Parlant | guideline 机制 |
| 微软 / .NET 技术栈，或从 AutoGen 迁移 | Microsoft Agent Framework | 官方迁移路径 |

学习顺序建议：

1. 手写一个不依赖框架的工具调用循环（半天），弄清消息格式和停止条件；
2. 选一个 Agent 框架或厂商 SDK 完成一个小项目；
3. 流程复杂后再学 LangGraph 这类编排框架；
4. 其他框架按项目需要学习。重点是能说清楚不同框架的抽象差异，而不是掌握 API 数量。

## 官方指南

- **Anthropic《Building Effective Agents》**：[链接](https://www.anthropic.com/engineering/building-effective-agents)。核心观点是先用简单的工作流（提示链、路由、并行化等），确实需要时再引入自主 Agent。
- **OpenAI《A Practical Guide to Building Agents》**：[PDF](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf)。覆盖单 Agent 与多 Agent 模式、guardrail 设计。
- **Agentic Design Patterns 中英对照**：[ginobefun/agentic-design-patterns-cn](https://github.com/ginobefun/agentic-design-patterns-cn)。按设计模式系统整理 Agent 构建方法。
- **Anthropic 提示工程文档**：[链接](https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/overview)，以及[交互式教程](https://github.com/anthropics/prompt-eng-interactive-tutorial)。
- **Google Gemini 提示指南**：[PDF](https://services.google.com/fh/files/misc/gemini-for-google-workspace-prompting-guide-101.pdf)。

## 面试中如何说明选型

面试官问“为什么选 XX 框架”，想听的是你对需求和抽象层次的判断，而不是框架介绍。一个完整的回答通常包括：

1. **需求**：流程是否固定、是否需要持久化和人工审批、对延迟和成本的要求；
2. **备选与取舍**：考虑过哪些方案，为什么放弃（例如“流程分支多且要中断恢复，CrewAI 的角色抽象表达不了，所以用 LangGraph”）；
3. **框架之外的部分**：哪些能力自己实现（评测、重试、观测），框架升级或替换时的成本；
4. **事后看法**：如果重做会不会换，理由是什么。

如果项目用过 AutoGen 或 Swarm，可以如实说明，并补充现在的迁移选择，这能体现你在跟进生态变化。

## 相关文档

- [Agent 资源总览](./README.md)
- [Agent Memory 资源](./memory.md)
- [AI Agent 生产环境实践](../../docs/03-practice/06-agent-production-challenges.md)

---

**返回**：[AgentGuide README](../../README.md)

## 文档维护

- 2026-11：标注 AutoGen 维护模式与 Swarm 被取代；加入 OpenAI Agents SDK、Claude Agent SDK、Google ADK、Microsoft Agent Framework、Pydantic AI、LlamaIndex；按 LangChain 1.x 更新描述；去掉 star 数。
- 2025-11：新增 Parlant 和 AgentScope，整合官方学习指南。
