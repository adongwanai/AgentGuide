---
type: 求职指南
status: 已发布
level: 高阶
topic:
  - 面试求职
  - Agent
---

# Agent Harness 彻底备战手册（补充模块对应版）
> **本手册定位**：面向 AI Agent 求职与面试准备的专题复习资料，用于补充 AgentGuide 面试题库与项目讲述材料。
> 内容以公开资料、项目复盘和工程实践方法论为基础做二次整理，建议结合自己的真实项目经历进行取舍与改写。

---
## 目标：把 Harness Engineering 从概念到机制到工程实践全部打透，面试时能讲得像做过生产级 Agent 系统的人

> 使用建议：这份文档对应《面试备战手册_补充模块.md》的第七节 Harness 和第九节 Claude Code 源码。五份核心材料全部吸收：
> 1. `Harness_101_ReAct_Loop`: 从单轮 ReAct 到 Skill 的六层阶梯推演
> 2. `深入浅出 Harness 核心模式`: Claude Code 的持久化指令、分层记忆、工作流编排、隔离子智能体
> 3. `一文讲透 Harness = 控制论`: 可能性空间压缩、控制力累积、共轭控制、传感器
> 4. `Harness Engineering 90% AI Coding`: 四根支柱、十阶段流水线、质量门禁必须可程序化验证
> 5. `Harness 不是目的，知识才是护城河`: 五层存储 × 五种类型 × 三级成熟度、三级渐进索引
>
> 特别说明：KTClaw 手册已经覆盖了你做的具体工程底座（三进程、多入口、分身、Gateway）。这份文档拔高一层，让你能用 Harness Engineering 的理论框架去解释和提升你做过的工程决策。

---

# 一、先总览：面试官在 Harness 这条线上会怎么打

JD 要求"熟悉主流 agent harness 框架"。面试官不会只让你背名词，而是会从三个层次来考：

1. **概念层**：Harness Engineering 是什么？为什么 Prompt Engineering → Context Engineering → Harness Engineering 是三次范式跃迁？
2. **机制层**：ReAct Loop 怎么跑？Plan-then-Act 什么时候加？Nudge 什么时候打？Offloading 和 Compression 的边界在哪？Skill 怎么做到能力数量和 token 成本解耦？
3. **工程层**：质量门禁怎么设计才能被机械化执行？为什么要把执行的 Agent 和评判的 Agent 分开？知识怎么沉淀成团队资产而不是散落在 prompt 里？

你能把这三层讲清楚，你就是面试官心目中"能做 Agent 系统架构"的人，而不只是"调过 Agent API"的人。

---

# 二、核心知识框架

## 1. Harness Engineering 是什么

### 1.1 一句话定义
> "Harness Engineering 是围绕 AI Agent 设计和构建**约束机制（Constraints）、反馈回路（Feedback Loops）、工作流控制（Workflow Orchestration）与持续改进循环（Continuous Improvement）**的系统工程实践。"

### 1.2 三次范式跃迁

| 阶段 | 时期 | 核心关注 | 隐喻 |
|------|------|---------|------|
| **Prompt Engineering** | 2022-2024 | 单次交互优化——Few-shot、CoT、角色设定 | "写好一封邮件" |
| **Context Engineering** | 2025 | 给 Agent 看什么——动态上下文窗口 | "给邮件附上正确附件" |
| **Harness Engineering** | 2026 | 跨会话、跨角色的完整系统架构 | "设计整个邮局系统" |

### 1.3 最核心的一句话
> "Agents aren't hard; the Harness is hard." —— OpenAI 工程师 Ryan Lopopolo

### 1.4 Mitchell Hashimoto 的操作性定义
> "Every time you discover an agent has made a mistake, you take the time to engineer a solution so that it can never make that mistake again."

Harness 不是一次性的 prompt 优化，而是一个**持续演进的系统工程闭环**。

### 1.5 为什么不能只靠模型本身 —— Agent 的四种典型失败模式

Anthropic 系统总结了 Agent 在复杂项目中的四类失败：

1. **One-shot Syndrome（试图一步到位）**：Agent 拿到复杂需求后在一个上下文窗口内完成全部工作，窗口过半后输出质量急剧下降。Sweet spot 在上下文填充率 40% 以下
2. **Premature Victory Declaration（过早宣布胜利）**：核心功能未实现就说"编码完成"，实际上编译都过不了
3. **Premature Feature Completion（过早标记完成）**：认为功能已实现但未做端到端验证
4. **Cold Start Problem（冷启动）**：多次会话间缺乏持久化记忆，每次新会话花大量 Token 重新理解项目

这四类失败的共同根源：**Agent 缺乏外部的结构化约束和反馈机制。** Anthropic 进一步指出一个根本性限制：**"Agents are incapable of accurately evaluating their own work"**——Agent 无法准确评估自身产出的质量。

### 必背点
- Prompt → Context → Harness 三次跃迁
- "Agents aren't hard; the Harness is hard"
- Mitchell Hashimoto 定义：每个错误都工程化消除它再次发生的可能
- 四种失败模式：One-shot / Premature Victory / Premature Feature / Cold Start
- 上下文 Sweet Spot 在 40% 填充率以下
- Agent 无法准确评估自身产出

---

## 2. Harness Engineering = 控制论

这是理解 Harness 最深层的理论视角。如果你能在面试里用控制论解释 Harness，面试官会对你刮目相看。

### 2.1 控制论的核心公式

```
[ 设定目标 ] → [ 感知偏差 ] → [ 施加干预 ] → [ 消除偏差 ]
```

对应到 Agent 场景：
- **目标** = 需求（需要 Agent 完成什么事）
- **控制器** = AI 模型
- **被控系统** = 软件代码
- **传感器** = QA 校验工具

### 2.2 三个核心概念

**概念一：降低可能性空间**
AI 写代码有无限种可能。每加一条约束（用什么语言、什么框架、什么命名规范），可能性空间就缩小一步。好的 Harness 就是用规则和架构把 Agent 生成结果的可能性空间从 M 压到 m。

**概念二：控制力累积**
Agent 一次满足不了所有要求。正确的做法是像老鹰俯冲抓兔子一样，把大任务拆成小步骤，每步通过反馈矫正偏差，多步控制累积最终达到目标。

**概念三：共轭控制**
曹冲称象的原理：把无法直接做的事（称象），转换成可以做的事（称石头），再转换回去。在 Agent 场景里：各种 MCP 工具、Skill 脚本就是"感受器"和"效应器"，让 AI 能用工具完成原来不能直接完成的事。

### 2.3 传感器：Harness 的"神经系统"

**业务无关的传感器**（不需要理解业务就能判断错误）：
- 编译是否通过
- 代码规范检查结果
- 安全扫描结果
- 部署失败日志

**业务相关的传感器**（必须对照业务规则才能判断）：
- 接口的测试用例是否全部通过
- 业务规则的边界条件是否覆盖
- 端到端流程是否走通

面试金句：
> "一切不可被机器验证的约束，在 Agent 执行中都是无效约束。如果它不能被机械化地执行，Agent 就会偏离。"

---

## 3. Harness 的六层机制阶梯

这是 Harness 101 材料里最精华的部分。从最朴素的 ReAct Loop 出发，每一层都是对上一层暴露问题的局部回应。

### 第 1 层：单轮 ReAct Loop

最朴素的核心循环：

```python
while True:
    response = llm.call(messages, tools=TOOLS)
    if not response.tool_calls or iterations > MAX:
        break
    results = run_tools_parallel(response.tool_calls)
    messages.extend(results)
```

**关键设计决策**：
- **谁决定停机？** Harness 不决定——只看 assistant turn 里有没有 tool_call。全部决策权在模型手里
- **一次 assistant turn 可以并行 emit 多个 tool_call**——Harness 必须支持并行执行
- Harness 不做决策，只负责搬运

### 第 2 层：多轮 Loop → Deep Research 雏形

Loop 不变，只把工具从 `get_weather` 换成 `web_search + web_fetch`。行为空间爆炸：search → 挑链接 → fetch → 信息不够 → 换个关键词 search → 再 fetch。走多少步、怎么拐，全靠模型自己判断。

### 第 3 层：Plan-then-Act

纯 ReAct 的毛病：想到哪走到哪，容易漏维度。解法是加一个 `write_todos` 工具：

```python
# Agent 第一步先输出 plan
write_todos([
  { "content": "了解整体规模", "status": "in_progress" },
  { "content": "梳理主要玩家", "status": "pending" },
  ...
])
# 每完成一项更新 status → completed
```

**补丁 A**：Plan 之前允许一次 `web_search`（one-time + before planning），让 Agent 对陌生话题建立基本认知但不抢 plan 的主场。

**补丁 B**：Nudge——每次 tool_result 之后，如果 todo list 非空，硬编码塞一条 `<system-reminder>` 提醒更新进度。Nudge 不来自用户，不占用户 turn。

### 第 4 层：Coding Agent

工具从 `web_search + web_fetch` 换成 `read_file + write_file + bash`。**Loop 不变，工具选型决定了 Agent 的形态。**

### 第 5 层：Offloading

Context Window 不是无限的。当 tool_result 超大时，Harness 把原始内容写到本地磁盘，Context 里只留一个路径占位符。Agent 需要时再用 `read_file` 拉回来。

**Offloading vs Compression 的关键区别**：
- **Offloading**：无损可还原，触发在 tool_result 诞生那一刻
- **Compression**：有损不可恢复，触发在 Context 快爆时

### 第 6 层：Skill

Skill 是在 Offloading 之后更进一层的答案——不解决"一条 tool_result 太大"，而是解决"**能力太多，System Prompt 装不下**"。

**关键设计**：
- System Prompt 只放 Skill 索引（name + description，~20 tokens/skill）
- Agent 用 `read_file` 主动按需加载 SKILL.md 全文
- 不加新工具，复用已有工具链
- 100 个 Skill 的索引开销 ≈ 2000 tokens

### 六层的共同主题

每一层都**只做一件事**，每一层都**假设下面几层已经稳定**。好的 Harness 不是一次把六层画完，而是**允许每一层被单独调整和替换**。改了 Offloading 的阈值策略，不会伤到 Skill 的加载机制。改了 Nudge 的触发规则，不会动到 ReAct 的基本节拍。

### 面试时可以这样讲
> "Harness 的六层机制——ReAct Loop → 多轮 Loop → Plan-then-Act → Coding Agent → Offloading → Skill——本质上是一层一层往上摞的。每一层只解决上一层暴露出来的一个问题：Loop 太盲目就加 Plan，Plan 执行中容易忘就加 Nudge，tool_result 太大就加 Offloading，System Prompt 装不下能力就加 Skill。这个'逐层响应问题'的设计哲学，比具体某一层的实现更重要。"

---

## 4. Harness 的四根支柱

综合 Anthropic 和 OpenAI 的生产级 Agent 实践，Harness Engineering 可以归纳为四根支柱：

### 支柱一：上下文架构（Context Architecture）
Agent 应该恰好获得当前任务所需的上下文——不多不少。
- **错误做法**：AGENTS.md 写成百科全书 → "所有内容都重要 = 没有内容重要"
- **正确做法**：AGENTS.md 控制在 ~100 行，作为索引和地图，指向更深的 Design Docs 和 Quality Criteria
- **三层分层加载**：L1 会话常驻层（~400 行 Agent 定义 + Rules）→ L2 阶段触发层（按需加载 Skill）→ L3 按需查询层（Wiki 知识库）

### 支柱二：Agent 专业化（Agent Specialization）
拥有受限工具集的专用 Agent，优于拥有全部权限的通用 Agent。
- Anthropic 明确分离三种角色：Planner（规划）、Generator（实现）、Evaluator（验证）
- **核心杠杆**："将做事的 Agent 和评判的 Agent 分开"

### 支柱三：持久化记忆（Persistent Memory）
进度持久化在文件系统上，而非上下文窗口中。
- Anthropic 的标准化启动序列：检查当前工作目录 → 读取 Git Log 和进度文件 → 定位未完成任务 → 开始工作
- 使跨数十个会话的长时间任务成为可能

### 支柱四：结构化执行（Structured Execution）
永远不让 Agent 在未经审查和批准书面计划之前写代码。
- 理想的执行流：理解 → 规划 → 执行 → 验证，每阶段之间有明确的质量门禁
- **关键原则**：质量门禁必须可程序化验证。用 Custom Linter + Structure Tests + Taste Invariants 构建机械化约束，完全替代文档层面的"建议"和"最佳实践"

---

## 5. 知识沉淀：Harness 的终极护城河

### 5.1 核心论点
> "工作流只是管道，知识才是流过管道的活水。将来的技术护城河不在模型，而在垂直领域知识的沉淀。"

工作流是可以替换的——今天用 16 阶段状态机，明天可能用 DAG 编排。但团队积累的知识——"广告预算扣减在高并发下会超扣，需用 Redis+Lua 保证原子性"——这条知识不管工作流怎么变，都是有价值的。

### 5.2 知识的三维体系

| 维度 | 回答的问题 | 分层 |
|------|-----------|------|
| **存储层**（在哪） | 知识存在哪里？ | Layer 0-P 个人 → 0-T 团队 → 1 技术 → 2 业务 → 3 项目 |
| **知识类型**（是什么） | 知识描述的是什么？ | model / decision / guideline / pitfall / process |
| **成熟度**（多可信） | 知识经过多少验证？ | draft → verified → proven |

### 5.3 三级渐进式索引

借鉴 Karpathy 的 LLM Wiki Pattern：

- **Layer A**：全景目录（~50 行）→ "知识库有什么？" → 零成本
- **Layer B**：分类清单（~100-300 行）→ "这个分类有哪些条目？" → 低成本
- **Layer C**：完整条目（~50-200 行/条）→ "这条知识说了什么？" → 按需

**Agent 用 ~50 行的成本了解知识库全貌，用 ~300 行的成本定位相关条目，只在真正需要时才读取完整内容。**

### 5.4 自动衰减 + 引用追踪闭环

成熟度自动升降：
- proven 条目 12 个月未被引用 → 降级为 verified
- verified 条目 6 个月未被引用 → 降级为 draft
- draft 条目持续未引用 → 归档

被引用的知识 maturity 自动提升，长期未引用的自动衰减。**知识不是写完就完了，它有生命周期。**

---

# 三、高频面试题 + 高强度答案

## Q1. 什么是 Harness Engineering？它和 Prompt Engineering、Context Engineering 的区别是什么？

### 主答模板
Prompt Engineering 关注的是单次交互——怎么写 prompt 让模型给出更好的回答。Context Engineering 进了一步——关注会话窗口里该放什么文档、什么历史、什么工具定义。Harness Engineering 再往前一步——不再是"一次对话"或"一次上下文窗口"，而是设计跨越多个会话、多个 Agent 角色、多个执行阶段的完整系统架构。

核心跃迁在于：Prompt Engineering 的隐喻是"写好一封邮件"，Context Engineering 是"给邮件附上正确的附件"，而 Harness Engineering 是"设计整个邮局系统"——包括什么时候谁来处理、质量怎么检查、出错怎么回退、知识怎么沉淀。

### 面试官追问：最重要的区别是什么？

**推荐回答**：从"优化模型单次输出"到"设计让模型持续正确运行的系统"。Harness 不试图让模型更聪明，而是让模型的错误变得可控、可发现、可修复。

---

## Q2. Agent 为什么需要一个 Harness？直接调 API 不行吗？

### 主答模板
直接调 API 在简单场景下可以，但一旦进入复杂长任务就会出现 Anthropic 总结的四类失败模式：
1. One-shot Syndrome：上下文窗口过半后输出质量断崖式下降
2. Premature Victory：核心功能没做就说"完成了"
3. Premature Feature：不做端到端验证就标记完成
4. Cold Start：每次新会话重新理解项目，大量 Token 被浪费

更根本的问题是 Anthropic 指出的：**Agent 无法准确评估自身产出的质量**。它需要一个外部的约束和反馈系统来弥补这个缺陷。Harness 就是这个外部系统。

---

## Q3. 你说"一切不可被机器验证的约束都是无效约束"，能举一个具体例子吗？

### 主答模板
"检查 CI 是否通过"是一句自然语言描述，Agent 可能认为状态码 SUCCESS 即通过，却忽略了测试用例数为 0 的异常。正确的做法是把质量门禁写成可程序化验证的条件：`status == SUCCESS && total_tests > 0 && passed == total`。同样，"生成评审报告"不够——必须校验"目标路径下文件存在且包含必填章节"。

核心原则：**If it can't be mechanically enforced, the agent will drift.**

---

## Q4. 为什么要把"做事的 Agent"和"评判的 Agent"分开？

### 主答模板
这是 Anthropic 反复强调的一个核心杠杆。评审 Agent 不需要更聪明，它只需要用一套不同于编码 Agent 的检查视角来审视产出物。这种 Agent-to-Agent Review 的本质是将传统的 Code Review 自动化，将质量发现前移到 Human Review 之前。实际效果是：编码 Agent 遗漏的逻辑、跳过的阶段、偷工减料的部分，评审 Agent 能以不同的视角捕捉到。

更深一层的原因是 Anthropic 的发现：**Agent 无法准确评估自身产出**——让同一个 Agent 既写代码又审自己的代码，相当于让考生自己批改自己的试卷。分开执行者和评判者，是最廉价的可靠性提升。

---

## Q5. 你在 KTClaw 里做的三进程架构、多入口抽象、分层记忆——这和 Harness Engineering 有什么关系？

### 主答模板
KTClaw 的工程底座本质上就是一个领域定制的 Agent Harness：
- 三进程架构（Renderer/Main/Gateway）对应的是 Harness 的结构化执行和 Agent 专业化——不同进程承担不同角色
- 多入口统一抽象（ChannelAdapter + InboundContext）对应的是上下文架构——不管来自哪个平台，Agent 都消费统一的上下文对象
- 分层记忆（索引→事实→流程→归档）对应的是持久化记忆支柱——进度和数据不只在内存里
- 分身隔离和主 Agent+专业 Agent 对应的是执行与评判分离的思想——不同的上下文、不同的职责

面试时可以这样讲：
> "我在联想做的三进程多 Agent 架构，回头看就是一套面向企业知识问答场景的 Harness。和 Claude Code 的通用 Harness 相比，我的设计更聚焦特定业务域，但在架构层面——工具隔离、权限管控、上下文管理、持久化状态——是同一套设计哲学。"

---

## Q6. 你对"98% 的代码是 Harness，只有 2% 是 AI 逻辑"这个发现怎么看？

### 主答模板
Claude Code 源码分析揭示了一个核心事实：512,000 行 TypeScript 中仅 1.6% 是 AI 决策逻辑，98.4% 是 Harness 基础设施。这个数字说明了两个问题：
1. **AI 本身的"智力"已经是 commodity，真正的工程复杂度在怎么驾驭它**。权限系统、上下文压缩、工具注册、错误恢复、状态持久化——这些才是决定一个 Agent 系统能不能在生产环境稳定运行的关键。
2. **未来工程师的核心竞争力正在从"写代码"转向"设计 Agent 的工作环境"**。这和 OpenAI 团队的实践结论完全一致：程序员不再是代码的生产者，而是环境的架构师、反馈回路的设计者、质量标准的编码者。

---

## Q7. 如果让你为小红书 C 端 Agent 设计一套 Harness，核心原则是什么？

### 主答模板
五条核心原则：

1. **上下文分层加载**：用户画像和长期记忆不进 System Prompt 常驻——只放索引，正文按需读取。借鉴 Claude Code 的 Progressive Disclosure
2. **执行与评判分离**：生成推荐内容的 Agent 和验证推荐质量的 Agent 必须分开。评判 Agent 检查事实准确性、时效性、是否有幻觉
3. **质量门禁必须可程序化验证**：不是说"检查推荐是否合理"，而是"营业状态必须经过外部 API 交叉验证"、"推荐理由必须引用至少一条用户笔记"
4. **知识要沉淀成团队资产**：每次用户反馈（"推荐的餐厅已关门"、"这个不适合带小孩"）不是修一个 badcase 就完事，而是回写到 Harness 的规则里——"餐厅推荐必须验证最近 30 天内的营业状态"、"亲子推荐必须包含儿童设施字段"
5. **冷启动不靠模型能力，靠约束设计**：新用户没有画像时，Agent 的行为靠规则约束——宁可保守（多问一个澄清问题），不要激进（猜一个用户偏好）

---

## Q8. Skill 和 Tool 在 Harness 视角下的区别是什么？

### 主答模板
Tool 是 Harness 提供给模型的原子能力——它能调用什么。Skill 是 Harness 组织能力的方式——它怎么把多个 Tool 调用编织成稳定的工作流。

从 Harness 设计角度看：
- Tool 对应的是 Harness 的工具注册层——负责 schema 定义、权限检查、执行路由、结果格式化
- Skill 对应的是 Harness 的上下文架构层——负责决定什么时候该让 Agent 看到哪条 Skill、怎么渐进式加载、怎么条件激活

Skill 的加载机制本身不需要新工具——Claude Code 用 `read_file` 加载 SKILL.md，Hermes 用 `skill_view` 加载。这意味着 Skill 层对 Harness 的工具注册层是**零增量**的——它只改变上下文的内容，不改变工具链。

---

# 四、高频总追问清单

## A. 概念与框架
1. Harness Engineering 的三次范式跃迁分别是什么？
2. 为什么说 Agent 无法准确评估自身产出？
3. Mitchell Hashimoto 对 Harness 的定义是什么？
4. OpenAI 3 人 5 个月 100 万行代码的核心经验有哪些？
5. "98% 的代码是 Harness"这个发现说明了什么？

## B. 机制与模式
6. ReAct Loop 谁决定停机？为什么不在 Harness 层做判断？
7. Plan-then-Act 和纯 ReAct 的本质区别是什么？
8. Nudge 是 user message 还是 system message？为什么要在 harness 层打？
9. Offloading 和 Compression 的本质区别是什么？
10. Skill 为什么不引入专门的 `load_skill` 新工具？

## C. 工程实践
11. 为什么质量门禁必须可程序化验证？
12. 分离执行 Agent 和评判 Agent 有什么实际收益？
13. 上下文窗口 Sweet Spot 在 40% 以下填充率——为什么？
14. 十阶段流水线为什么连一个 6 行代码的小需求也要完整走？
15. 为什么说"等待成本高于纠错成本"？

## D. 知识沉淀
16. 为什么说 Harness 不是目的，知识才是护城河？
17. 五层存储 × 五种类型 × 三级成熟度的三维体系怎么设计？
18. 三级渐进式索引如何让 Agent 用 ~50 行的成本了解知识库全貌？
19. 知识的自动衰减和引用追踪闭环怎么实现？

## E. 和你的项目对接
20. KTClaw 的三进程架构在 Harness 框架下属于哪一层？
21. GoAfar 的 OR-Tools 作为 verifier 能不能理解为 Harness 中的"业务传感器"？
22. TinyClaw 的 RAG-to-Skill 在 Harness 框架下属于哪个机制？
23. 如果你现在重新做 KTClaw，在 Harness 层面会做什么不同？

---

# 五、串联叙事：面试时最强的 Harness 主线

> "我对 Harness Engineering 的理解，不是从某一个框架开始的，而是从自己做的 Agent 系统里感受到的痛点出发的。在 KTClaw 里，我做了三进程分层架构、多入口统一抽象、分层记忆——回头看，这些本质上就是在给 Agent 搭 Harness：上下文架构、角色分离、持久化状态。

> 后来我深入研究了 Anthropic 和 OpenAI 的 Harness 工程实践，有几个启发特别深。第一是 Mitchell Hashimoto 的定义——每发现一个错误，就工程化地消除它再次发生的可能性。这和我在 KTClaw 里修小模型工具调用死循环、修 Gateway 生命周期管理的思路完全一致。第二是 Anthropic 的核心洞察——Agent 无法准确评估自身产出，所以必须把执行的 Agent 和评判的 Agent 分开。我在 KTClaw 里做的主 Agent+专业 Agent 分工，本质上也是这个思想。第三是 OpenAI 的经验——质量门禁必须可程序化验证，如果它不能被机械化地执行，Agent 就会偏离。这让我反思自己之前在评测体系上的一些设计，是不是太依赖自然语言描述了。

> 从更宏观的视角看，我认为 Harness Engineering 背后是控制论的思想——设定目标、感知偏差、施加干预、消除偏差。Agent 不是程序员，是被控系统；而 Harness 是让这个系统能持续稳定运行的控制回路。技术护城河不在模型，而在 Harness 里积累的约束规则、反馈回路和领域知识——这些是模型换代也带不走的。"

---

# 六、最后：Harness 准备到什么程度，才算真准备好了？

## 你至少要做到 7 件事
1. **能讲清楚 Prompt → Context → Harness 三次跃迁**
2. **能画出 ReAct Loop → 多轮 → Plan-then-Act → Coding Agent → Offloading → Skill 的六层阶梯**
3. **能说出 Anthropic 总结的四种 Agent 失败模式和根本原因**
4. **能用控制论的语言解释 Harness：可能性空间、控制力累积、共轭控制、传感器**
5. **能解释为什么质量门禁必须可程序化验证，并举出具体例子**
6. **能讲清楚知识沉淀为什么比工作流本身更重要**
7. **能把 KTClaw/GoAfar/TinyClaw 的工程决策放进 Harness 框架里解释**

## 如果你想让面试官"折服"，你要主动讲出来的 4 句话
1. **"Harness Engineering 的本质不是让 Agent 更聪明，而是让 Agent 的错误变得可控、可发现、可修复。"**
2. **"做事的 Agent 和评判的 Agent 必须分开——因为 Agent 无法准确评估自身产出，这是 Anthropic 的核心发现。"**
3. **"如果一条规则不能被机械化地验证，它在 Agent 执行中就是无效约束。质量门禁不能是自然语言描述，必须是可执行的检查。"**
4. **"技术护城河不在模型，而在 Harness 里积累的约束规则、反馈回路和领域知识——模型会换代，工具链会重构，但领域知识是永恒的。"**
