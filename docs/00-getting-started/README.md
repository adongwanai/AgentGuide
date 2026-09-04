---
type: 入口页
status: 已发布
level: 通用
topic:
  - Agent
  - 项目实战
---

# AgentGuide 快速开始

> 目标：用最短路径建立 AI Agent 的全局认知，并把学习推进到“能做项目、能评测、能写进简历”。

很多人学 Agent 会卡在两个地方：一是把 Agent 当成“更长的 prompt”，二是看了一堆框架教程却没有可验证产出。这个目录负责解决第一公里问题：先建立地图，再动手做最小项目，最后用评测和复盘把项目变成作品。

## 入口顺序

| 顺序 | 文档 | 你会得到什么 |
|:---:|:---|:---|
| 1 | [Agent 学习地图](./01-agent-map.md) | 区分 chatbot、workflow、agent、multi-agent、coding agent、computer-use agent |
| 2 | [前 7 天学习计划](./02-first-7-days.md) | 每天学什么、做什么、交付什么 |
| 3 | [高质量资源筛选清单](./03-resource-quality-checklist.md) | 判断一个 GitHub 项目/教程/论文是否值得投入时间 |

## 一张图看学习路径

![Agent 学习地图](./agent-learning-map.svg)

## 最小闭环

新手不要一上来追求“全自动通用 Agent”。更稳的闭环是：

1. **先做 workflow**：固定步骤，少量工具，明确输入输出。
2. **再加 agent loop**：让模型在有限步数内决定下一步工具调用。
3. **补 trace 和 eval**：记录每一步，并用 20 条 case 评估成功率。
4. **最后做项目包装**：写 README、架构图、失败分析和简历表达。

## 必读外部资料

- [Anthropic: Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)：非常适合建立 workflow 和 agent 的边界感。
- [Anthropic: Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents)：工具设计比“多接几个 API”重要得多。
- [OpenAI Agents SDK](https://platform.openai.com/docs/guides/agents-sdk/)：学习 handoff、guardrail、tracing 等生产化概念。
- [LangGraph](https://github.com/langchain-ai/langgraph)：学习持久执行、状态图、人类审核和长任务恢复。
- [Model Context Protocol](https://github.com/modelcontextprotocol/modelcontextprotocol)：理解 Agent 如何标准化连接外部工具和数据源。

## 本目录适合谁

- 完全新手：先读 `01-agent-map.md`，不要急着装框架。
- 已会 API 调用：直接读 `02-first-7-days.md`，把 toy demo 推到有 eval 的项目。
- 准备求职：重点读 `03-resource-quality-checklist.md` 和 `projects/` 下三个项目蓝图。
- 仓库维护者：查看 [内容 Backlog](../../BACKLOG.md)，按原路径继续补空缺内容。

---

<!-- AUTO-GENERATED-CONTENT:START -->
## 完整内容索引

<!-- 下表由元数据生成，请勿手工编辑。 -->

| 文档 | 类型 | 状态 | 难度 | 主题 |
|:---|:---|:---|:---|:---|
| [Agent Learning Map：Agent 学习地图](./01-agent-map.md) | 教程 | 已发布 | 入门 | Agent |
| [First 7 Days：前 7 天学习计划](./02-first-7-days.md) | 教程 | 已发布 | 入门 | Agent |
| [Resource Quality Checklist：高质量资源筛选清单](./03-resource-quality-checklist.md) | 教程 | 已发布 | 入门 | Agent |
<!-- AUTO-GENERATED-CONTENT:END -->
