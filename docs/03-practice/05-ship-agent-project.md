---
type: 实践指南
status: 已发布
level: 进阶
topic:
  - Agent
  - 项目实战
  - 面试求职
---

# 如何落地一个可写进简历的 Agent 项目

> 不是装一个框架跑 hello world，而是从理解、规划、实现、评测到复盘，做出一个能演示、能解释、能量化的 Agent 项目。

## 最终标准

一个简历级 Agent 项目至少应该做到：

- 有明确用户和任务，不是泛泛的“智能助手”
- 有可运行入口，别人 clone 后能按 README 跑起来
- 有清晰 agent loop、工具注册、上下文管理、权限边界
- 有日志或 trace，失败后能定位问题
- 有 eval case，能说明通过率、失败类型、成本和延迟
- 有复盘：为什么这样设计，替代方案是什么，哪些机制真的带来提升

---

## Step 1：建立全局认知

不要一上来逐文件读源码。先回答 5 个问题：

1. Agent loop 在哪里？也就是模型调用、工具调用、观察结果回写的主循环在哪里。
2. 工具注册机制是什么？新增工具是加配置、加函数，还是改核心逻辑？
3. Context 怎么组装？system、history、memory、retrieved context、tool result 的顺序是什么？
4. Memory 怎么存？JSONL、SQLite、向量库、外部服务，分别存什么？
5. 通道层怎么抽象？CLI、Web、IM、定时任务是否共用同一套 agent core？

建议做法：

- 先看 README、架构图、入口文件、测试文件
- 用 `rg` 搜索 `tool`, `agent`, `loop`, `memory`, `session`, `trace`
- 把核心文件压缩到 3-5 个，逐行读这些文件
- 配角文件只扫职责，不要陷进每个实现细节

**产出物**：一页项目理解笔记，包含模块图和核心调用链。

## Step 2：准备 AI 编程环境

核心不是装 IDE，而是建立：

> Plan -> 执行 -> 验证 -> 复盘

建议在项目根目录沉淀这些文件：

```text
项目根/
├── AGENTS.md 或 CLAUDE.md    # 给 coding agent 的项目规则
└── docs/
    ├── spec.md              # 需求规格
    ├── prompt_plan.md       # 实现计划
    └── eval_plan.md         # 评测设计
```

`AGENTS.md` 或 `CLAUDE.md` 建议写四块：

| 模块 | 内容 |
|:---|:---|
| WHAT | 项目一句话定位 |
| WHY | 不可变约束，例如安全边界、数据权限、成本上限 |
| HOW | 如何安装、运行、测试、调试 |
| TEST | 改代码必须同步验证什么，外部依赖如何 mock |

**产出物**：能让 AI coding 工具稳定协作的项目规则文件。

## Step 3：建立项目理解 Skill

当项目变大后，只靠一次性 prompt 很容易丢上下文。更好的方式是把项目理解沉淀成可复用 Skill 或项目指南。

可以写成：

```text
custom_skills/
└── understand_project.md
```

内容建议：

- 项目一句话定位
- 目录结构
- 核心流程
- Agent loop 位置
- 工具注册方式
- Context / Memory / Eval 相关文件
- 常用命令
- 修改代码时的注意事项
- 验收标准

这样之后你可以让 coding agent “参考项目理解 Skill”，不需要每次重新读一遍源码。

**产出物**：一份项目理解 Skill 或等价的项目上下文文档。

## Step 4：先写 Spec，再写代码

脑子里的需求通常不完整。先把需求写成 spec，再拆成可测试的小步骤。

Spec 至少回答：

| 问题 | 示例 |
|:---|:---|
| 目标用户是谁 | 研究生、客服、运营、开发者 |
| 触达渠道是什么 | CLI、Web、飞书、Slack、浏览器 |
| 核心任务是什么 | 搜索资料、生成报告、审查代码、整理会议 |
| 工具有哪些 | search、read_file、write_file、browser、database |
| 每个工具失败怎么办 | 空结果、超时、权限不足、参数错误 |
| 哪些动作需要确认 | 发邮件、删文件、付款、发布内容 |
| 成功标准是什么 | 完成率、引用正确率、延迟、成本 |
| 成本约束是什么 | 每个 trajectory 的 token / 金额上限 |

实现计划建议拆成 5-10 个小步骤，每步都能独立验证。

**产出物**：`docs/spec.md` 和 `docs/prompt_plan.md`。

## Step 5：评测、归因和消融实验

没有 eval 的 Agent 修改就是“感觉变好了”。项目要能写进简历，必须有数据。

准备 10-20 个评测 case，覆盖：

- 正常任务
- 空输入
- 工具失败
- 检索不到结果
- prompt injection
- 高风险动作
- 长上下文
- 多轮追问

每次跑完记录：

| 字段           | 说明                        |
| :----------- | :------------------------ |
| task_id      | 任务编号                      |
| input        | 用户输入                      |
| expected     | 期望行为                      |
| actual       | 实际输出                      |
| pass         | 是否通过                      |
| failure_type | 工具选错、上下文污染、模型能力、业务规则、权限问题 |
| steps        | agent loop 步数             |
| tool_calls   | 工具调用次数                    |
| cost         | token 或金额                 |
| latency      | 延迟                        |

消融实验可以这样做：

- 去掉 context 压缩，看成功率和成本变化
- 去掉 reranker，看引用正确率变化
- 换小模型做 gate，看成本和误判率变化
- 合并或拆分工具，看工具选择准确率变化
- 关闭 memory，看多轮任务成功率变化

**产出物**：一张 eval 表格 + 一段项目复盘。

---

## 可靠性六件套

| 机制 | 做什么 | 不做的后果 |
|:---|:---|:---|
| Idempotency | 工具调用带幂等 key | 重启后重复发邮件、重复扣款、重复写入 |
| Timeout + Circuit Breaker | 每个工具有超时，连续失败后熔断 | trajectory 被慢工具拖垮 |
| Rate Limit Aware Retry | 指数退避、jitter、多 provider fallback | 限流后整个系统挂掉 |
| Cost Guard | 每个 trajectory 设 token / 金额上限 | bug 无限循环烧成本 |
| Permission Tier | 读自动、写 dry-run + 人审、不可逆显式确认 | Agent 执行危险动作 |
| Observability | 记录 prompt、tool call、result、token、延迟 | 出问题无法回放和归因 |

## 简历写法

不要写：

> 基于大模型实现智能问答系统。

可以写：

> 基于轻量 Agent Harness 构建资料研究助手。ReAct loop + dispatch 表注册 search / fetch / summarize / cite 4 类工具，四层 context 管理 system、memory、retrieval、turn 信息；高风险写入动作接入 dry-run 和人工确认。构建 20 条 eval case，端到端通过率 82%，消融实验显示 context 压缩使成功率提升 12%，模型路由将 token 成本降低 60%。

## 常见误区

| 误区 | 更好的做法 |
|:---|:---|
| 先选最强模型 | 先搭好 harness，模型应该可以快速切换 |
| 先想清架构再动手 | Agent 是反馈密集型，先跑通最小闭环 |
| 框架越大越好 | 轻量基座 + 清晰边界通常更适合个人项目 |
| 只看最终输出 | 看 trajectory，失败常发生在工具选择和上下文管理 |
| 没有测试就加功能 | 先有 eval，再扩工具、加 memory、上 multi-agent |
