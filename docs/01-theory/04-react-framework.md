---
type: 教程
status: 已发布
level: 进阶
topic:
  - Agent
  - 框架工具
---

# 手撕 ReAct 框架

> ReAct = Reasoning + Acting。它让模型在推理和行动之间交替：先判断下一步，再调用工具，再根据观察结果继续。

ReAct 是理解 Agent loop 的入门框架，但工程里不要迷信“让模型一直想”。你真正要设计的是工具、状态、停止条件、错误恢复和 trace。

## 基本模式

```text
Question: ...
Thought: I need to search for evidence.
Action: search(query="...")
Observation: ...
Thought: The evidence is enough.
Final Answer: ...
```

真实系统建议不要存完整 `Thought`，而是存可审计摘要：

```json
{
  "step": 2,
  "action": "search",
  "args": {"query": "agentic rag evaluation"},
  "reason_summary": "Need external evidence before comparing methods.",
  "observation_summary": "Found 5 papers from 2024-2025."
}
```

## 最小实现

```python
def react(task, model, tools, max_steps=6):
    trace = []
    observations = []

    for step in range(max_steps):
        prompt = build_prompt(task, tools, observations)
        decision = model.generate_json(prompt)

        if decision["type"] == "final":
            return {
                "answer": decision["answer"],
                "trace": trace,
                "status": "completed",
            }

        tool_name = decision["tool"]
        if tool_name not in tools:
            obs = {"ok": False, "error": "unknown_tool"}
        else:
            args = decision.get("args", {})
            tool = tools[tool_name]
            obs = tool(**args)

        observations.append(obs)
        trace.append({
            "step": step + 1,
            "tool": tool_name,
            "args": decision.get("args", {}),
            "observation": summarize(obs),
        })

    return {
        "answer": None,
        "trace": trace,
        "status": "max_steps_exceeded",
    }
```

## Prompt 结构

```text
You are an agent that solves the task by calling tools.

Rules:
- Use only the listed tools.
- Stop when enough evidence is collected.
- If evidence is missing, say what is missing.
- Never fabricate tool results.

Task:
{task}

Tools:
{tool_cards}

Previous observations:
{observations}

Return JSON:
{
  "type": "tool_call | final",
  "tool": "tool name or null",
  "args": {},
  "answer": "final answer or null",
  "reason_summary": "short auditable reason"
}
```

## Tool Card 很重要

ReAct 失败常常不是模型不会推理，而是工具描述太差。每个工具至少写清楚：

- 什么时候用。
- 什么时候不要用。
- 输入 schema。
- 输出 schema。
- 错误类型。
- 权限等级。
- 示例。

可参考：[Tool Card 模板](../../examples/tool-card-template.md)。

## 常见失败模式

| 失败 | 现象 | 修复 |
|:---|:---|:---|
| 无限循环 | 一直搜索或重复点击 | 最大步数、重复动作检测 |
| 工具选择错 | 本该查数据库却搜网页 | 工具描述加 Use When / Do Not Use |
| 观察过长 | 工具返回淹没关键信息 | 工具层分页、摘要、截断 |
| 过早 final | 证据不足就回答 | evidence gate、引用检查 |
| 幻觉工具结果 | 没调用工具却声称查到了 | trace 强制引用 observation |
| 高风险动作 | 自动付款、删除、提交 | permission tier + human-in-the-loop |

## ReAct vs Plan-and-Execute

| 模式 | 适合 | 风险 |
|:---|:---|:---|
| ReAct | 短任务、信息查找、工具反馈快 | 容易局部贪心和循环 |
| Plan-and-Execute | 长任务、步骤明确但需要调整 | 初始计划可能过时 |
| Reflection | 需要复盘和修正的任务 | 反思也可能幻觉 |
| Graph workflow | 状态分支明确、需要恢复 | 设计成本更高 |

## 评测 ReAct Agent

至少记录：

- 任务成功率。
- 平均步数。
- 工具调用成功率。
- 重复动作次数。
- 证据引用准确率。
- 高风险动作拦截率。
- 成本和延迟。

## 面试表达

> 我会把 ReAct 看成 Agent loop 的基础模式，但不会只依赖 prompt。工程实现里会加工具注册表、结构化输出、最大步数、重复动作检测、权限确认和 trace。对需要稳定性的业务流程，我会优先用 graph/workflow 承载确定步骤，只在开放决策点使用 ReAct。

## 延伸阅读

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [Anthropic: Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [OpenAI Swarm](https://github.com/openai/swarm)
