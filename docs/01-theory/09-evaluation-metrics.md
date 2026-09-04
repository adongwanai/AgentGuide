---
type: 教程
status: 已发布
level: 高阶
topic:
  - Agent
  - 评测
---

# Agent 评估指标体系

> 没有评测的 Agent 只能算 demo。Agent 评估不仅看最终答案，还要看路径、工具、证据、安全、成本和可恢复性。

## 三层评估

| 层级 | 评什么 | 示例 |
|:---|:---|:---|
| Component Eval | 单个模块是否正确 | 检索相关性、工具 schema、PDF 解析 |
| Trajectory Eval | 执行路径是否合理 | 工具顺序、重复动作、错误恢复 |
| End-to-End Eval | 最终任务是否完成 | 旅行方案是否满足预算，论文综述是否有证据 |

只看 end-to-end 成功率很危险，因为你不知道失败来自模型、工具、检索、上下文还是安全拦截。

## 核心指标

### 任务完成

| 指标 | 含义 |
|:---|:---|
| `task_success_rate` | 任务最终是否完成 |
| `partial_success_rate` | 部分目标是否完成 |
| `constraint_satisfaction` | 是否满足预算、时间、格式、安全等约束 |
| `user_acceptance_rate` | 人工审核是否接受输出 |

### 工具与轨迹

| 指标 | 含义 |
|:---|:---|
| `tool_call_accuracy` | 工具选择和参数是否正确 |
| `tool_error_rate` | 工具调用失败比例 |
| `avg_steps` | 平均行动步数 |
| `repeated_action_rate` | 重复无效动作比例 |
| `recovery_rate` | 出错后是否能恢复 |

### 证据与事实性

| 指标 | 含义 |
|:---|:---|
| `citation_precision` | 引用是否支持结论 |
| `evidence_recall` | 是否找到了必要证据 |
| `hallucination_rate` | 无证据断言占比 |
| `abstention_quality` | 证据不足时是否拒答 |

### 成本与性能

| 指标 | 含义 |
|:---|:---|
| `latency_p50/p95` | 响应耗时 |
| `token_cost` | token 成本 |
| `tool_cost` | 外部 API 或浏览器成本 |
| `cache_hit_rate` | 缓存命中率 |

### 安全与治理

| 指标 | 含义 |
|:---|:---|
| `unsafe_action_block_rate` | 高风险动作是否被拦截 |
| `secret_leak_rate` | 是否泄漏敏感信息 |
| `policy_violation_rate` | 是否违反系统规则 |
| `human_escalation_rate` | 需要人工确认的比例 |

## Eval Case 格式

```json
{
  "id": "web-001",
  "category": "normal",
  "task": "打开官方文档并找到安装命令",
  "expected_behavior": [
    "访问页面",
    "提取命令",
    "给出来源 URL"
  ],
  "forbidden_behavior": [
    "不访问页面直接猜",
    "编造命令"
  ],
  "scoring": {
    "task_success": 0.5,
    "evidence": 0.3,
    "format": 0.2
  }
}
```

可参考：[examples/eval-cases.jsonl](../../examples/eval-cases.jsonl)。

## 评分方式

| 方式 | 优点 | 缺点 |
|:---|:---|:---|
| Exact match | 稳定、便宜 | 只适合确定答案 |
| Rule-based scorer | 可解释 | 规则维护成本高 |
| LLM-as-judge | 灵活 | 需要校准，会偏 |
| Human review | 质量最高 | 成本高 |
| Hybrid | 平衡稳定和覆盖 | 实现复杂 |

建议：关键指标用规则或人工抽检，开放质量用 LLM judge 辅助。

## Trace 是评测前提

每次运行至少保存：

- run_id、case_id。
- 每步工具调用和参数。
- 工具返回摘要。
- 截图或文档证据路径。
- token、成本、耗时。
- 最终答案和评分。

示例 schema：[trace-schema.json](../../examples/trace-schema.json)。

## 最小评测集

新项目建议从 20 条开始：

- 10 条正常任务。
- 5 条边界任务。
- 3 条工具失败任务。
- 2 条安全/拒绝任务。

上线或写进简历前扩展到 50 条以上，并加入回归测试。

## 常见评测陷阱

- 只测 happy path。
- eval case 和 prompt 互相泄漏。
- 用同一个模型生成答案又当 judge。
- 不保存 trace，无法定位失败。
- 不统计成本，只追成功率。
- 忽视安全拒绝，把“拒绝危险请求”误判成失败。

## 推荐工具

- [Promptfoo](https://github.com/promptfoo/promptfoo)：prompt、RAG、Agent 回归测试和红队。
- [DeepEval](https://github.com/confident-ai/deepeval)：pytest 风格 LLM eval。
- [Inspect](https://inspect.aisi.org.uk/)：更规范的模型能力和安全评测。
- [RAGAS](https://github.com/explodinggradients/ragas)：RAG 评测。
- [BrowserGym](https://github.com/ServiceNow/BrowserGym)：Web Agent 评测环境。

## 简历表达

> 为 Agent 系统设计 50 条 eval case，覆盖正常任务、边界约束、工具失败和安全拒绝；记录每步 tool call、observation、token、latency 和 artifacts，按任务成功率、引用准确率、恢复率、平均成本和高风险动作拦截率做回归评测。
