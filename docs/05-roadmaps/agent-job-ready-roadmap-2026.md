---
type: 路线图
status: 已发布
level: 通用
topic:
  - 面试求职
  - Agent
  - 项目实战
---

# 2026 Agent 求职通关路线

> 这是一条面向求职和项目落地的 Agent 学习路线。目标不是收藏更多链接，而是一步步做出能验证、能演示、能写进简历的作品。

## 适合谁

| 你的状态 | 建议入口 |
|:---|:---|
| 零基础 | 从 Stage 0 开始，先建立 Agent / Workflow / RAG / Multi-Agent 的坐标系 |
| 做过 LLM 应用 | 从 Stage 1 或 Stage 2 开始，补 Agent Loop、工具调用、评测和上下文工程 |
| 想做简历项目 | 直接看 Stage 7-8，把项目做成可运行、可评测、可复盘的系统 |
| 准备面试 | 对照每个阶段的产出物，练习用架构、业务、结果三维表达项目 |

---

## 当前优先级

Agent 方向变化很快，当前更值得投入的是能落地、能验证、能讲清楚取舍的能力。

| 优先级 | 方向 | 为什么重要 |
|:---:|:---|:---|
| 1 | Claude Code / Codex-style Coding Agents | 真实代码库、shell、文件编辑、测试、权限、上下文压缩，是理解 Agent 工程的最佳样本 |
| 2 | Agent Harness Engineering | Agent 能力很大一部分来自 harness：工具协议、权限、状态、反馈、回放、CI、评测 |
| 3 | Context Engineering | Agent 的核心不是提示词，而是控制信息如何进入模型 |
| 4 | Skills / MCP / A2A / ACP | Skills 负责能力复用，MCP 连接工具，A2A 连接 Agent，ACP 连接宿主应用 |
| 5 | Browser / Computer-Use Agents | 浏览器和桌面操作是 Agent 从 demo 走向真实任务的重要边界 |
| 6 | Evaluation / Observability / Safety | 没有 eval、trace、权限边界的 Agent，只能算 demo |

---

## Stage 0：理解 Agent 是什么

**目标**：区分 chatbot、workflow、agent、multi-agent，知道什么时候不该用 Agent。

需要掌握：

- Agent 基本循环：observe -> think -> act -> observe
- Workflow 和 Agent 的边界
- Agent 失败的常见原因：上下文污染、工具设计差、状态不可恢复、没有评测
- Harness 的意义：模型是推理核心，harness 提供工具、权限、记忆、反馈和运行环境

推荐阅读：

- [Anthropic: Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)
- [OpenAI: A practical guide to building agents](https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/)
- [上下文工程实践](../02-tech-stack/11-context-engineering-practices.md)

**产出物**：写一页笔记回答三个问题：

1. 我的场景为什么需要 Agent，而不是普通 workflow？
2. Context Engineering 和 Prompt Engineering 的区别是什么？
3. Harness 的各层分别解决什么问题？

## Stage 1：写出最小 Agent Loop

**目标**：自己写出一个能选择工具、执行工具、返回最终答案的最小 Agent。

需要掌握：

- 普通 LLM API 调用
- 结构化 JSON 输出
- 工具函数定义和 schema 设计
- tool call / function call 解析
- 最大步数、超时、错误处理

核心循环：

```python
while True:
    response = call_model(messages, tools)
    if response.stop_reason == "end_turn":
        break
    if response.stop_reason == "tool_use":
        result = run_tool(response.tool_call)
        messages.append(result)
        continue
```

推荐阅读：

- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [Claude Tool Use](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview)
- [手撕 ReAct 框架](../01-theory/04-react-framework.md)

**产出物**：一个 50-150 行的最小 Agent，可以调用 calculator、search、read_file 等工具。

## Stage 2：Tool Use、RAG 和 Memory

**目标**：让 Agent 从“会聊天”变成“能查资料、用工具、有记忆”。

需要掌握：

- RAG 全流程：chunk -> embed -> retrieve -> answer with citations
- 工具注册：优先使用 dispatch 表，而不是 if-elif 长链
- 短期上下文、会话记忆、长期记忆的区别
- 工具失败、空结果、重复调用、幻觉引用处理
- 回答中给出来源或证据

推荐阅读：

- [Agent Memory 系统完整指南](../02-tech-stack/15-agent-memory.md)
- [MCP 协议详解](../02-tech-stack/10-mcp-protocol.md)

**产出物**：一个资料研究助手，输入主题后自动搜索、筛选、总结并输出引用链接。

## Stage 3：研究一个现代 Agent Harness

**目标**：从真实系统里学习 Agent 如何组织工具、上下文、权限、状态、日志、子任务和反馈。

推荐研究对象：

| 系统 | 适合学习什么 |
|:---|:---|
| Claude Code | CLI、工具、权限、hooks、subagents、MCP |
| learn-claude-code | 从零复刻 coding agent harness |
| OpenClaw / claw0 | 长运行、本地优先、消息入口、memory、heartbeat |
| OpenHands / SWE-agent | 代码库编辑、shell、测试、sandbox |
| nanobot | 轻量 agent base、multi-provider、memory、skills |

重点读：

- agent loop 在哪里
- tool registry 怎么设计
- permission gate 怎么拦截风险动作
- session store 怎么保存状态
- context compaction 怎么触发
- trace 怎么记录和回放

**产出物**：一个可调试的 harness demo，包含 README、运行步骤、示例输入输出和失败记录。

## Stage 4：把 Multi-Agent 当协调问题

**目标**：理解多 Agent 不是“角色扮演越多越强”，而是任务分解、上下文隔离和结果汇总。

需要掌握：

- planner / executor / reviewer / critic / router 的职责边界
- supervisor 或 graph 如何管理多 Agent
- 每个 Agent 的输入输出 schema 和停止条件
- 循环、争论、任务漂移、上下文膨胀怎么处理
- 什么时候单 Agent 更好

判断标准：

> 如果子任务之间需要频繁交换中间状态，优先单 Agent；如果子任务可以独立完成、最后只汇总结果，才考虑 multi-agent。

推荐阅读：

- [Claude Code Subagents](https://code.claude.com/docs/en/sub-agents)
- [Anthropic: How we built our multi-agent research system](https://www.anthropic.com/engineering/built-multi-agent-research-system)

**产出物**：一个 research -> write -> review -> revise 的小型多 Agent 系统。

## Stage 5：学习 Skills、协议和能力封装

**目标**：理解 Tool、Skill、MCP、A2A、ACP 分别解决什么问题。

需要掌握：

- Tool：可调用接口，解决“能不能做”
- Skill：可复用流程知识，解决“怎么高质量地做”
- MCP：连接外部工具和数据源
- A2A：Agent 之间发现、通信和协作
- ACP：宿主应用与 Agent 的统一接口

一个好的 Skill 应该包含：

- name / description
- 何时使用
- 操作步骤
- 需要渐进加载的脚本或模板
- 验收标准
- smoke test

推荐阅读：

- [Claude Code Skills](https://code.claude.com/docs/en/skills)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [Agent2Agent Protocol](https://google-a2a.github.io/A2A/specification/)

**产出物**：一个可复用 Skill，例如 code-review、research-report、pdf-extraction 或 release-note-writer。

## Stage 6：Browser 和 Computer-Use Agents

**目标**：让 Agent 能操作公开网页，并具备可复盘的安全边界。

需要掌握：

- Browser Agent 和普通 API Tool 的区别
- Playwright / browser-use 的基本操作
- 页面变化、弹窗、加载失败、元素定位失败处理
- 截图、DOM、动作日志记录
- 对登录、支付、发布、删除等动作设置安全限制

推荐阅读：

- [Claude Computer Use](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/computer-use-tool)
- [browser-use](https://github.com/browser-use/browser-use)
- [WebArena](https://arxiv.org/abs/2307.13854)

**产出物**：一个只操作公开网页的 Browser Agent，能打开页面、提取信息、生成摘要。

## Stage 7：Evaluation、Observability 和 Safety

**目标**：用评测和 trace 把 Agent 从“感觉变好”推进到“可以定位、可以回归、可以优化”。

需要掌握三层评测：

| 层级 | 测什么 | 怎么做 |
|:---|:---|:---|
| Component eval | 单个 prompt / tool / retriever 是否可靠 | fixture -> 期望输出，断言关键属性 |
| Trajectory eval | 整段 agent loop 是否走对路径 | 看步数、工具选择、是否绕路，结合 rubric 打分 |
| End-to-end eval | 用户任务是否完成 | 真实任务 + 明确成功标准 |

还需要记录：

- 成功率
- 失败原因
- 工具调用次数
- 成本
- 延迟
- 权限确认记录

推荐阅读：

- [Agent 评估完全指南](../02-tech-stack/agent-evaluation-complete-guide.md)
- [Evaluation Harness 完全指南](../02-tech-stack/26-agent-evaluation-harness-guide.md)
- [OpenAI Evals](https://platform.openai.com/docs/guides/evals)

**产出物**：一个 eval 表格，至少 20 个任务、期望结果、实际结果、失败分类。

## Stage 8：Ship 一个真实 Agent

**目标**：完成一个别人能 clone 下来跑的 Agent 项目。

最低要求：

- 有明确用户、明确任务、明确成功标准
- 有日志、trace、错误重试、超时、成本上限
- 有权限边界和人工确认机制
- 有部署方式：CLI、Web app、Slack bot、GitHub Action 或后台任务
- 有 README：怎么运行、怎么配置、怎么扩展、有哪些限制
- 有 eval：至少 20 条任务，记录通过率和失败归因

**最终产出物**：一个可运行 Agent 项目 + 技术博客 + 简历项目描述。

---

## 简历表达模板

不要只写“基于大模型实现智能问答”。一个 Agent 项目应该从三维表达：

| 维度 | 怎么写 |
|:---|:---|
| 架构表达 | agent loop、工具注册、会话状态、上下文裁剪、权限确认、记忆和错误恢复 |
| 业务表达 | 业务场景、关键工具、数据源、用户路径和约束条件 |
| 结果表达 | 评测集规模、成功率、失败类型、成本优化和消融实验结论 |

示例：

> 基于轻量 Agent Harness 构建垂直场景助手。ReAct loop + dispatch 表注册 5 个业务工具，四层分级 context 管理 system / long-term / short-term / turn 信息；高风险操作接入三级权限确认。20 条评测 case 端到端通过率 82%，通过上下文压缩和模型路由将 token 成本降低 60%。
