# 最小 Agent Loop 模板

> 目标：不用任何复杂框架，先理解 Agent 的骨架。真实项目可以把这里的模块替换成 LangGraph、OpenAI Agents SDK、Pydantic AI 或自研 harness。

## 核心循环

```python
def run_agent(task, tools, model, max_steps=6):
    state = {
        "task": task,
        "steps": [],
        "artifacts": {},
        "final": None,
    }

    for step_id in range(1, max_steps + 1):
        context = build_context(task=task, state=state, tools=tools)
        action = model.decide(context)

        if action.type == "final_answer":
            state["final"] = action.content
            trace(step_id, action, observation=None, state=state)
            return state

        if action.tool_name not in tools:
            observation = {"ok": False, "error": "unknown_tool"}
        else:
            tool = tools[action.tool_name]
            if tool.requires_confirmation and not ask_human(action):
                observation = {"ok": False, "error": "human_rejected"}
            else:
                try:
                    observation = tool.call(**action.args)
                except TimeoutError:
                    observation = {"ok": False, "error": "timeout"}
                except Exception as exc:
                    observation = {"ok": False, "error": "tool_exception", "message": str(exc)}

        state["steps"].append({"action": action, "observation": observation})
        trace(step_id, action, observation, state)

    state["final"] = {
        "ok": False,
        "reason": "max_steps_exceeded",
        "summary": summarize_state(state),
    }
    return state
```

## 4 个必须有的边界

### 1. 最大步数

没有最大步数，Agent 可能在工具错误、检索噪声或不确定任务里无限循环。入门项目建议 `max_steps=5-8`，复杂任务再根据 eval 数据调整。

### 2. 工具白名单

模型只能调用注册表里的工具。不要让模型直接拼 shell、SQL 或 HTTP 请求，除非有 sandbox、权限和审计。

### 3. 可读错误

成功、未知工具、人工拒绝和工具异常都必须写入 `state["steps"]`，再由下一轮 `build_context` 带回模型。只写 trace 不会自动更新模型上下文；如果失败后直接 `continue`，模型可能看不到错误并重复同一个动作。

工具失败时不要只返回 stack trace。推荐返回结构：

```json
{
  "ok": false,
  "error": "rate_limited",
  "retryable": true,
  "hint": "Wait 30 seconds or reduce page_size.",
  "partial_result": []
}
```

### 4. Trace

每一步都记录：任务、上下文摘要、工具、参数、观察、耗时、成本、是否成功。没有 trace，失败后只能猜。

## Context Builder 推荐结构

```text
System:
  - 角色、边界、输出格式

Task:
  - 用户目标
  - 成功标准
  - 禁止行为

Tools:
  - 名称
  - 何时使用
  - 参数 schema
  - 返回 schema
  - 错误处理

State:
  - 已完成步骤
  - 当前证据
  - 已知失败
  - 预算和剩余步数
```

## 什么时候升级到框架

| 触发条件 | 可以考虑 |
|:---|:---|
| 状态分支变多，需要可视化和恢复 | LangGraph |
| 需要 handoff、guardrail、tracing 一体化 | OpenAI Agents SDK |
| 强依赖结构化输出和类型校验 | Pydantic AI |
| 需要多个角色协作 | AutoGen / CrewAI |
| 需要接很多外部工具 | MCP server/client |
