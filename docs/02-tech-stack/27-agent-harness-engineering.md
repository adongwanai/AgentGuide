---
type: 教程
status: 已发布
level: 高阶
topic:
  - Agent
  - 上下文工程
  - MCP
---

# Agent Harness Engineering：把裸模型变成能干活的系统

> 模型决定推理下限，Harness 决定产品上限。真正的 Agent 工程，不是只调用 LLM API，而是给模型配好工具、权限、状态、记忆、通道、反馈和评测。

## 什么是 Agent Harness

可以把 Agent 拆成两部分：

```text
Agent = Model + Harness
```

- **Model**：负责理解、推理、生成、选择动作。
- **Harness**：负责把模型放进真实工作环境，提供工具、上下文、权限、状态、日志、回放、评测和人类反馈。

同一个模型，放在聊天窗口里只是问答助手；放进好的 harness 里，才可能成为能长期执行任务的 Agent。

## 为什么 Harness 重要

很多 Agent demo 失败，不是模型不够强，而是 harness 太弱：

- 工具描述含糊，模型经常选错工具
- 工具返回太长，污染上下文
- 没有权限分级，危险动作不可控
- 没有 session store，任务中断后无法恢复
- 没有 trace，失败后无法复盘
- 没有 eval，改动后不知道能力是否退化
- 没有成本上限，循环调用工具烧 token

Harness Engineering 的核心问题是：

> 如何设计一个运行环境，让模型能稳定、可控、可观测地完成真实任务？

---

## 七层模型

```text
L7  自驱层（heartbeat / cron）      让 Agent 自己找活干
L6  通道层（CLI / Web / IM / Email）让 Agent 进入真实入口
L5  人格层（SOUL / policy）         让 Agent 跨模型不漂移
L4  记忆层（JSONL / SQLite / vector）让 Agent 记得上下文和偏好
L3  工具层（tools / skills / MCP）  让 Agent 真的能干活
L2  循环层（agent loop）            让 Agent 能持续观察和行动
L1  模型层（provider abstraction）  让 Agent 不绑死一家模型
```

初学者不要一开始就搭七层。第一周只需要跑通 L1 + L2 + L3：

1. 模型能调用
2. agent loop 能持续执行
3. 工具能注册、执行、返回结果

后面的 memory、channel、heartbeat 都应该在真实需求出现后再加。

## L1：模型层

目标：换模型不用改业务代码。

关键做法：

- 所有模型调用统一走 `call_model(messages, tools, config)`
- provider 配置外置，支持 OpenAI / Anthropic / Gemini / 本地模型切换
- 输出统一成内部事件格式，例如 `message`, `tool_call`, `final`, `error`
- 记录 token、延迟、模型名、请求 id

面试常问：

> 为什么不能直接在业务代码里到处调 SDK？

回答重点：

- provider 行为不同，必须做统一抽象
- 便于模型路由和 fallback
- 便于统计成本和排查问题

## L2：循环层

目标：从一次问答变成持续工作。

最小 agent loop：

```python
for step in range(max_steps):
    response = call_model(messages, tools)
    if response.type == "final":
        return response.content
    if response.type == "tool_call":
        result = run_tool(response.tool_call)
        messages.append(format_tool_result(result))
        continue
raise MaxStepsExceeded()
```

必须有：

- 最大步数
- 超时
- 工具错误回传
- stop condition
- interrupt / resume 设计
- trace id

进阶设计：

- 一个 turn 可以包含多个 tool call
- 支持用户中途 steer
- 支持 context compaction
- 支持任务暂停和恢复

## L3：工具层

目标：让 Agent 不只聊天，还能读取、搜索、计算、写入和调用外部系统。

工具设计原则：

| 原则 | 说明 |
|:---|:---|
| 名称语义互斥 | 不要让 `search_docs` 和 `find_docs` 同时出现 |
| description 面向模型 | 让模型知道什么时候用、什么时候不用 |
| schema 严格 | JSON Schema / enum / required 字段要清楚 |
| 返回可控 | 大结果分页、截断、摘要、附来源 |
| 最小权限 | 工具只暴露完成任务所需的最小能力 |
| 可观测 | 每次调用记录参数、结果、耗时、错误 |

推荐结构：

```python
TOOLS = {
    "search": search_tool,
    "read_file": read_file_tool,
    "write_file": write_file_tool,
}
```

新增工具应该是加一项注册，而不是改一坨 if-elif。

## L4：记忆层

目标：让 Agent 在任务、会话和长期偏好之间保持连续性。

三类记忆：

| 类型 | 存什么 | 常见实现 |
|:---|:---|:---|
| Working memory | 当前任务的短期状态 | messages / scratchpad |
| Episodic memory | 过去发生过什么 | JSONL / SQLite |
| Semantic memory | 用户偏好、事实、知识 | 向量库 / hybrid search |

不要过早上向量库。很多个人项目用 JSONL + SQLite 就能跑很久。

关键问题：

- 谁决定什么值得存？
- 写入门槛是什么？
- 什么时候合并、遗忘、压缩？
- 召回结果如何防止污染当前上下文？

## L5：人格层

目标：让 Agent 的行为风格和边界跨模型保持稳定。

可以沉淀：

- 项目规则：`AGENTS.md` / `CLAUDE.md`
- 行为原则：`SOUL.md`
- 安全策略：policy / permission config
- 输出风格：templates

人格层不是玄学，它解决的是：

- 换模型后风格漂移
- 多入口回复不一致
- 风险场景没有统一边界
- 项目协作规则反复遗忘

## L6：通道层

目标：让 Agent 不只躲在终端里，而是进入用户真实入口。

常见通道：

- CLI
- Web app
- Slack / 飞书 / Discord / Telegram
- Email
- GitHub Action
- 定时后台任务

设计原则：

- agent core 和 channel adapter 分离
- 不同通道共用 session / permission / trace
- 先把一个通道用透，再加第二个通道

## L7：自驱层

目标：让 Agent 从“踹一下动一下”变成能主动检查任务。

常见方式：

- heartbeat
- cron
- queue worker
- webhook
- monitor

谨慎使用：

- 自驱层会放大成本
- 会放大错误动作
- 需要更强权限边界
- 需要更完整 trace 和告警

第一个月建议先关掉自驱层，等 eval 和权限成熟后再加。

---

## 可靠性六件套

| 机制 | 做什么 | 不做的后果 |
|:---|:---|:---|
| Idempotency | 工具调用带幂等 key | 重启后重复发邮件、重复写入、重复扣款 |
| Timeout + Circuit Breaker | 每个工具有超时，连续失败后熔断 | trajectory 被慢工具拖垮 |
| Rate Limit Aware Retry | 指数退避、jitter、多 provider fallback | 限流后整个系统挂掉 |
| Cost Guard | 每个 trajectory 设置 token / 金额上限 | bug 无限循环烧成本 |
| Permission Tier | 读自动、写 dry-run + 人审、不可逆显式确认 | Agent 执行危险动作 |
| Observability | 记录 prompt、tool call、result、token、延迟 | 出问题无法回放和归因 |

## Evaluation Harness vs Agent Harness

这两个概念容易混：

| 概念 | 解决什么 |
|:---|:---|
| Agent Harness | 让模型能在真实环境里工作 |
| Evaluation Harness | 评估模型或 Agent 是否完成任务 |

二者关系：

- Agent Harness 产生 trajectory
- Evaluation Harness 读取 trajectory、输出分数和失败归因
- 生产级系统里，评测和治理都依赖 trace

相关文档：

- [Evaluation Harness 完全指南](./26-agent-evaluation-harness-guide.md)
- [Agent 评估完全指南](./agent-evaluation-complete-guide.md)

## 面试速查

| 问题 | 回答框架 |
|:---|:---|
| 怎么设计一个 Agent？ | workflow vs agent -> loop -> tools -> context -> memory -> permission -> eval |
| 上下文太长怎么办？ | 分层、压缩、检索、工具结果后处理、cache-friendly 结构 |
| 工具选错怎么办？ | 改 description、拆/合工具、加 router、few-shot、看 trajectory |
| 什么时候用 multi-agent？ | 子任务独立性强、上下文隔离有收益、最后能汇总 |
| 成本怎么控制？ | 模型路由、prompt 瘦身、context 压缩、prefix cache、cost guard |
| HITL 怎么设计？ | pre-turn、mid-turn、post-turn；读自动、写审查、不可逆确认 |

## 最小实现 checklist

- [ ] `call_model()` provider 抽象
- [ ] agent loop 有最大步数和超时
- [ ] tools 用注册表管理
- [ ] 工具 schema 严格，返回可控
- [ ] 每步记录 trace id、tool call、result、latency、token
- [ ] 高风险工具有 permission tier
- [ ] session 可以保存和恢复
- [ ] 至少 20 条 eval case
- [ ] README 写清运行、配置、限制和扩展方式

## 学习顺序

1. 先手写一个最小 agent loop
2. 加 3 个工具，跑通 dispatch 表
3. 加 trace，能复盘每一步
4. 加 context compaction 和 cost guard
5. 加 permission tier
6. 加 memory
7. 加 eval
8. 最后再考虑 multi-agent、browser、channel 和 heartbeat
