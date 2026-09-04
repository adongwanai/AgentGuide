---
type: 教程
status: 已发布
level: 入门
topic:
  - Agent
---

# Agent Learning Map：Agent 学习地图

> Agent 不是一个库名，而是一类系统设计：模型在受控环境里观察状态、选择动作、调用工具、接收反馈，并在必要时请求人类介入。

## 先分清 5 个概念

| 概念 | 核心特征 | 典型例子 | 什么时候用 |
|:---|:---|:---|:---|
| Chatbot | 单轮或多轮对话，主要输出文本 | FAQ、客服问答 | 任务只需要回答，不需要行动 |
| Workflow | 固定流程，LLM 是某些步骤的组件 | 分类 -> 检索 -> 生成 -> 审核 | 步骤清晰、成功标准明确 |
| Agent | 模型自主选择下一步动作 | 研究助手、代码修复助手 | 任务开放，需要多步探索 |
| Multi-Agent | 多个角色协作，带协调器或消息协议 | 研究员 + 编写者 + 审稿人 | 单个角色职责过宽，需要分工 |
| Computer-Use Agent | 操作浏览器、桌面、终端或文件系统 | WebArena、OSWorld、coding agent | 任务必须进入真实软件环境 |

Anthropic 的实践建议很朴素：先从最简单的方案开始，只有当固定 workflow 不足以覆盖任务分支时，再引入更自主的 agent loop。这个判断很关键，因为 Agent 的能力来自工具、环境和评测，而不只是更复杂的框架。

## Agent 的 7 个核心模块

| 模块 | 你要能回答的问题 | 常见坑 |
|:---|:---|:---|
| Goal | 任务成功是什么样？失败边界是什么？ | 只写“帮我完成任务”，没有验收标准 |
| State | 当前任务状态、历史、工作区文件放哪里？ | 只依赖聊天历史，长任务一压缩就丢线索 |
| Context | 哪些信息进入模型？顺序和格式是什么？ | 把所有资料塞进上下文，导致干扰和成本暴涨 |
| Tools | 工具有哪些？schema、权限、错误如何描述？ | 工具名含糊、返回值太长、没有分页和重试 |
| Loop | 最大步数、停止条件、反思、恢复怎么做？ | 无限循环或过早停止 |
| Guardrails | 哪些动作需要确认？哪些输入不可信？ | 让模型自己决定安全边界 |
| Eval | 怎么证明它真的完成任务？ | 只有演示视频，没有可重复评测集 |

## 三条学习路线

### 路线 A：工程落地方向

目标是做出能交付的 Agent 应用。重点学：

- LangGraph 的状态、持久执行和 human-in-the-loop
- OpenAI Agents SDK 的 handoff、guardrail、tracing
- Pydantic AI 的类型约束、结构化输出和依赖注入
- Promptfoo、DeepEval、Inspect 这类 eval harness
- OWASP LLM Top 10、MCP Top 10、最小权限和审计日志

适合项目：旅行规划 Agent、企业知识库 Agent、客服流程 Agent。

### 路线 B：研究与算法方向

目标是理解 Agent 为什么成功或失败。重点学：

- ReAct、Plan-and-Solve、Reflection、Tree Search
- WebArena、OSWorld、GAIA、SWE-bench 等 benchmark
- Tool-use 数据集、轨迹评分、LLM-as-judge 校准
- RAG / multimodal RAG / memory 的消融实验
- Agent RL、过程奖励、失败轨迹复盘

适合项目：论文研读 Agent、Web Agent 评测、RAG 检索策略对比。

### 路线 C：求职作品方向

目标是做出能写进简历、能在面试里讲清楚的项目。重点不是“用了什么框架”，而是：

- 用户是谁，任务为什么需要 Agent
- 工具如何设计，权限如何分级
- 上下文如何管理，成本如何控制
- 评测集怎么构造，失败原因是什么
- 项目如何部署，别人能否 clone 运行

适合项目：`projects/01-paper-agent`、`projects/02-travel-agent`、`projects/03-web-agent`。

## 框架怎么选

| 场景 | 首选 | 理由 |
|:---|:---|:---|
| 学 agent loop 基础 | 原生 API + 50 行工具循环 | 能看清 observe -> act -> observe |
| 复杂状态流 | LangGraph | 状态图、持久执行、恢复、人类审核都比较成熟 |
| OpenAI 生态 | OpenAI Agents SDK | handoff、guardrails、tracing 是一等概念 |
| 类型安全 Python 应用 | Pydantic AI | Pydantic 风格，结构化输出和依赖注入舒服 |
| 多角色协作 demo | AutoGen / CrewAI | 快速搭建 planner、executor、reviewer 分工 |
| 数据密集 RAG | LlamaIndex | 数据连接器、索引、检索和 workflow 生态较完整 |

## 一个判断标准

只要你的系统会让模型调用工具，你就要同步设计评测和安全边界。没有 eval 的 Agent 是 demo；没有权限边界的 Agent 是风险源；没有 trace 的 Agent 很难调试。

## 延伸阅读

- [Anthropic: Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)
- [OpenAI: A practical guide to building agents](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf)
- [LangGraph GitHub](https://github.com/langchain-ai/langgraph)
- [OpenAI Agents SDK](https://platform.openai.com/docs/guides/agents-sdk/)
- [Pydantic AI GitHub](https://github.com/pydantic/pydantic-ai)
- [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications)
