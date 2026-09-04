---
type: 教程
status: 已发布
level: 入门
topic:
  - Agent
---

# First 7 Days：前 7 天学习计划

> 这不是“收藏 100 个链接”的计划，而是一周内做出最小 Agent 项目骨架的执行清单。

## Day 1：建立边界

**学习目标**：区分 chatbot、workflow、agent、multi-agent。

**阅读**：

- [Agent 学习地图](./01-agent-map.md)
- [Anthropic: Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)

**产出**：

- 写一页笔记：你的目标项目为什么需要 Agent，而不是普通 workflow。
- 画出最小流程：输入、工具、输出、失败边界。

## Day 2：手写最小 Agent Loop

**学习目标**：理解 observe -> think -> act -> observe。

**任务**：

- 只接 2 个工具：`search_notes(query)`、`write_summary(text)`。
- 限制最大 5 步。
- 每一步写入 JSONL trace：`step`、`thought_summary`、`tool`、`args`、`observation`、`cost_estimate`。

**验收**：

- 能处理 5 条固定任务。
- 出错时不崩溃，返回可读失败原因。

## Day 3：工具设计

**学习目标**：让工具变得“模型友好”。

**阅读**：

- [Anthropic: Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents)

**任务**：

- 给每个工具补：名称、使用场景、参数 schema、返回结构、错误码、示例。
- 增加分页、截断、重试和 timeout。
- 把高风险工具标成 `requires_confirmation`。

**验收**：

- 工具返回不超过模型真正需要的内容。
- 错误信息能指导下一步动作。

## Day 4：上下文工程

**学习目标**：控制什么进入模型。

**任务**：

- 把 context 分成 5 层：system、task、memory、retrieved evidence、recent trace。
- 做一个 context builder，把每层内容结构化输出。
- 对长工具结果只保留摘要和可追溯引用。

**验收**：

- 同一任务重复运行时，prompt 结构稳定。
- 不把完整日志、完整网页、完整 PDF 直接塞进模型。

## Day 5：评测集

**学习目标**：从“感觉能用”变成“可测量”。

**任务**：

- 写 20 条 eval case：10 条正常任务、5 条边界任务、5 条安全/失败任务。
- 每条包含：输入、期望行为、禁止行为、评分方式。
- 记录通过率、平均步数、失败原因、人工修复建议。

**可选工具**：

- [Promptfoo](https://github.com/promptfoo/promptfoo)：适合 prompt、RAG、agent 回归测试和红队测试。
- [DeepEval](https://github.com/confident-ai/deepeval)：适合 pytest 风格的 LLM/Agent 单元测试。
- [Inspect](https://inspect.aisi.org.uk/)：适合更严肃的模型能力与安全评测。

## Day 6：项目化包装

**学习目标**：让别人能 clone、运行、理解。

**任务**：

- 写 `README.md`：项目目标、架构、安装、运行、测试、限制。
- 写 `eval_report.md`：评测数据、失败类型、改进计划。
- 写 `demo_script.md`：面试或路演时怎么演示。

**验收**：

- 新机器按 README 能跑通最小 demo。
- 面试官能从 README 看出你做的不只是 API wrapper。

## Day 7：复盘与简历表达

**学习目标**：把项目变成可讲述的工程经验。

**任务**：

- 做一次失败归因：工具失败、检索失败、模型误判、权限不足、上下文污染分别占多少。
- 写一段简历 bullet，包含架构、场景、指标。

**模板**：

> 构建面向 X 场景的 Agent 系统，采用 ReAct loop + 工具注册表 + 分层 context builder，接入 N 个外部工具并对高风险动作设置 human-in-the-loop；设计 20 条端到端 eval case，任务成功率达到 X%，通过工具结果截断和模型路由将平均成本降低 X%。

## 一周结束后继续做什么

- 做论文方向：进入 [Paper Agent 项目蓝图](../../projects/01-paper-agent/README.md)。
- 做生活/业务方向：进入 [Travel Agent 项目蓝图](../../projects/02-travel-agent/README.md)。
- 做浏览器方向：进入 [Web Agent 项目蓝图](../../projects/03-web-agent/README.md)。
- 做资料型项目：进入 [多模态 RAG 资源](../../resources/multimodal/README.md)。
