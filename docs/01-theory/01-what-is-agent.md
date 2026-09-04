---
type: 教程
status: 已发布
level: 入门
topic:
  - Agent
  - 基础模型
---

# 什么是 AI Agent？

> AI Agent 是一个能在给定目标和约束下，观察环境、选择动作、调用工具、接收反馈，并持续推进任务的软件系统。

一句话区分：LLM 负责“推理和生成”，Agent 系统负责“把模型放进可行动、可观察、可评测的环境里”。

## Agent 不等于 Chatbot

| 类型 | 主要能力 | 是否行动 | 典型例子 |
|:---|:---|:---:|:---|
| Chatbot | 对话和回答 | 否 | FAQ、知识问答 |
| Workflow | 按固定步骤处理任务 | 有，但流程固定 | 分类 -> 检索 -> 生成 -> 审核 |
| Agent | 模型根据状态选择下一步动作 | 有，路径可变 | 研究助手、Web Agent、Coding Agent |
| Multi-Agent | 多个 Agent 通过角色或协议协作 | 有 | planner + executor + reviewer |

Anthropic 在 Agent 工程实践中强调：很多任务用 workflow 更稳，只有当任务路径难以提前写死、需要模型动态决策时，才需要更自主的 Agent。

## Agent 的最小闭环

```text
Goal
  -> Observe state
  -> Think / plan
  -> Act with tools
  -> Observe result
  -> Stop / continue / ask human
```

这就是常见的 ReAct 思路：Reasoning + Acting。真正工程化时，最好不要暴露完整 chain-of-thought，而是记录可审计的 `reason_summary`、工具调用、观察结果和最终判断。

## Agent 的 8 个组成部分

| 模块 | 作用 | 面试追问 |
|:---|:---|:---|
| Goal | 定义任务和成功标准 | 怎么判断任务完成？ |
| Policy | 系统规则、安全边界、权限 | 哪些动作必须人工确认？ |
| State | 当前进度、历史、临时产物 | 长任务如何恢复？ |
| Memory | 可复用经验和用户偏好 | 什么值得存，什么时候忘？ |
| Context Builder | 组装模型输入 | 如何避免上下文污染？ |
| Tool Registry | 声明可调用工具 | schema、错误、权限怎么设计？ |
| Loop Controller | 决定下一步和停止条件 | 如何防止无限循环？ |
| Eval / Trace | 记录和评估行为 | 如何证明 Agent 有效？ |

## 能力分级

| 级别 | 名称 | 特征 | 示例 |
|:---:|:---|:---|:---|
| L0 | Prompt App | 无工具，只生成文本 | 简单总结器 |
| L1 | Tool-Using App | 单轮工具调用 | 查天气后回答 |
| L2 | Workflow Agent | 多步骤但流程固定 | RAG pipeline |
| L3 | Autonomous Agent | 模型在有限步内选动作 | 资料研究 Agent |
| L4 | Long-running Agent | 有持久状态、恢复、人工介入 | Coding Agent、Web Agent |
| L5 | Multi-Agent System | 多角色协作和协调 | 研究员 + 写作者 + 审稿人 |

求职项目做到 L3-L4 就已经有含金量。L5 很酷，但复杂度和评测成本也高，不建议新手一上来做“全自动公司”。

## 什么时候不该用 Agent

- 任务流程固定，用普通 workflow 更稳定。
- 成功标准无法定义，也没有人工审核。
- 工具风险高，但没有权限、日志和回滚。
- 只是一次文本生成，不需要环境反馈。
- 评测成本太高，无法证明改动有效。

## 面试怎么回答

一个合格回答：

> Agent 是把 LLM 放进一个有状态、有工具、有反馈、有约束的执行循环中。它和普通 chatbot 的区别在于会对环境采取动作，并根据观察结果调整下一步。工程上我会重点设计 tool schema、context builder、loop controller、权限边界、trace 和 eval，而不是只写 prompt。

## 项目落地清单

做任何 Agent 项目前先回答：

1. 用户目标是什么？
2. 为什么 workflow 不够？
3. 工具有哪些，哪些有风险？
4. 状态和产物存哪里？
5. 最大步数和停止条件是什么？
6. 出错时怎么恢复？
7. 用什么 eval case 证明有效？

## 延伸阅读

- [Anthropic: Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)
- [OpenAI: A practical guide to building agents](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [OpenAI Agents SDK](https://platform.openai.com/docs/guides/agents-sdk/)
