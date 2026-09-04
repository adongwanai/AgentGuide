---
type: 求职指南
status: 已发布
level: 高阶
topic:
  - 面试求职
  - Coding Agent
  - Agent
---

# Claude Code 源码分析 彻底备战手册
> **本手册定位**：面向 AI Agent 求职与面试准备的专题复习资料，用于补充 AgentGuide 面试题库与项目讲述材料。
> 内容以公开资料、项目复盘和工程实践方法论为基础做二次整理，建议结合自己的真实项目经历进行取舍与改写。

---
## 目标：把 Claude Code 的架构设计讲得像真正读过 512K 行源码的人

> 使用建议：Claude Code 是 2026 年最成熟的生产级 AI Coding Agent。和 OpenClaw 的个人助理定位不同，Claude Code 解决的是"如何让 Agent 在复杂软件工程任务中稳定长时运行"。面试时引用它的设计决策，远比说"可以用 LangChain"有说服力。
>
> 综合吸收三份材料：
> 1. `ClaudeCode源码深度解读笔记`：19 节完整覆盖，最系统的拆解（含启动层、REPL、Bridge、特征门控、KAIROS/Undercover/反蒸馏等隐藏特性）
> 2. `深度解析_Claude_Code_Prompt_Context_Harness`：Prompt/Context/Harness 三维分析（含 6 步 Prompt 组装、三层渐进压缩、Memdir、System Reminder 注入）
> 3. `Claude_Code_源码拆解_从启动到多_Agent_扩展层`：7 个核心模块的复杂度分层视角（含启动三段式、REPL 控制面、Tool Runtime 收归横切关注点）

---

# 一、核心事实与定位

## 源码泄露事件

- **2026 年 3 月 31 日**：v2.1.88 随包附带 59.8 MB source map，512,000+ 行 TypeScript，约 1,900 个文件
- 根本原因：Bun 默认生成 source map，`.npmignore` 忘了排除 `*.map`
- 负责人 Boris Cherny："100% of my contributions to Claude Code were written by Claude Code."
- **代码可见 ≠ 开源**：Anthropic 的 license 不允许重新分发或修改

## 最核心的数字

| 指标 | 数值 |
|------|------|
| 代码量 | 512,000+ 行 TypeScript |
| AI 决策逻辑占比 | **仅 1.6%** |
| Harness 基础设施占比 | **98.4%** |
| 文件数 | ~1,900 |
| 最大单文件 | `query.ts` 约 785 KB |
| `QueryEngine.ts` | ~46,000 行 |
| `permissions.ts` | ~61 KB |

## 核心定位

> "Claude Code serves as the **agentic harness around Claude**: it provides the tools, context management, and execution environment that turn a language model into a capable coding agent."
>
> **模型从不直接触碰文件系统**。Harness 决定：这次读取是否被允许、读取结果怎样被处理、多少内容能装进下一轮 prompt。

---

# 二、启动与入口层：别急着拉起全世界

这是源码拆解中最容易被忽略但最决定架构寿命的一层。很多 Agent 系统的入口是一个越来越胖的 `main`，恨不得一上来就把全世界拉起来。Claude Code 很克制——先分流，再装配，最后才进入会话。

## 三段式启动

**第一段：入口分流（Entry Routing）**——先判断这次启动是什么模式：本地交互 / headless（无界面）/ SDK / remote / 后台 session / fast path。很多路径动态加载——从一开始就在控制启动成本。

**第二段：进程级初始化（Process-level Init）**——处理运行环境：配置、telemetry、远程设置、清理回调。**刻意不碰当前会话语义。** 回答的是"进程能不能跑"，不是"这一轮 Agent 该怎么跑"。

**第三段：会话级准备（Session-level Setup）**——确定 cwd、sessionId、工具面、权限模式、扩展能力、系统约束、恢复方式。决定由交互式宿主承载还是由无界面引擎承载。

## 关键判断：进程状态 vs 交互状态

Claude Code 把两类状态显式分开：
- **进程状态**：cwd、projectRoot、sessionId、telemetry、token/cost 计数 → 沉在底层全局状态
- **交互状态**：tasks、MCP clients、plugin 状态、permission context、UI 选择 → 进入 AppState

这个分层让 React state 不会误当成整套 runtime 的唯一真相，也不会把所有状态做成不可控的全局变量。

## 面试话术
> "启动层的质量平时不显山不露水，但系统一旦开始长模式、长宿主、长入口，它就是最先决定架构寿命的那一层。凡是会影响执行边界的东西——权限、工具面、扩展能力——都应该在第一轮请求前定型，而不是让主循环一边跑一边猜。"

---

# 三、12 层渐进式 Harness（架构骨架）

这是理解 Claude Code 工程复杂度最直接的方式。每一层都建立在前一层之上，核心循环**本身从不改变**。

| 层 | 机制 | 关键洞察 |
|----|------|---------|
| s01 | **Agent Loop** | while-true + tool_use 检查 + append |
| s02 | **Tool Dispatch** | `buildTool()` 工厂 + 统一注册表 |
| s03 | **Planning** | `EnterPlanModeTool` / `TodoWriteTool`——**完成率翻倍** |
| s04 | **Sub-Agents** | 子代理获得全新 messages[]，保持主对话干净 |
| s05 | **Knowledge On Demand** | Skill 通过 tool_result 注入，不污染 system prompt |
| s06 | **Context Compression** | 三层策略：autoCompact + snipCompact + contextCollapse |
| s07 | **Persistent Tasks** | 基于文件的任务图，带状态追踪和依赖关系 |
| s08 | **Background Tasks** | 守护线程运行命令，完成时注入通知 |
| s09 | **Agent Teams** | 持久化队友 + 异步邮箱 |
| s10 | **Team Protocols** | 统一的请求-响应模式驱动所有代理间协商 |
| s11 | **Autonomous Agents** | 空闲周期 + 自动认领，无需 lead 逐个分配 |
| s12 | **Worktree Isolation** | 任务管理目标，worktree 管理目录，通过 ID 绑定 |

### 为什么是 12 层而不是 1 层？

核心循环本身从不改变。添加新功能 = 添加新工具 或 新的包装层。这是**开闭原则**在 Agent 系统中的经典实践。

### s03 Planning 的"免费午餐"

从源码注释看，仅仅添加"先列步骤再执行"的机制就使任务完成率翻倍。对任何 Agent 系统来说，Planning 是**投入最小、收益最大**的改进。

---

# 四、三条主干链路

Claude Code 最深刻的架构洞察：整个系统不是七层堆叠，而是收敛成**三条主干链路**：

### ① 控制链：怎么想、怎么续跑
启动层定义边界 → REPL 汇总能力面 → Query Loop 推进连续运行

### ② 执行链：怎么动、怎么受约束
工具不直接调用 → 进入 Tool Runtime → 穿过 Permission 和 sandbox → 触达文件/命令/网络

### ③ 任务链：怎么并发、怎么持续、怎么回流
多 Agent 不塞回 Query Loop，而是进入独立的 Task Runtime 管理生命周期

**扩展层不是第四条孤立链路，而是给这三条主链持续注入能力。**

---

# 五、Query Loop：从函数封装到状态机

这是 Claude Code 和普通 Agent 实现差异最大的地方。

## 不是"模型调用封装"，而是 runtime

```typescript
state = {
  messages,
  toolUseContext,
  maxOutputTokensOverride,
  autoCompactTracking,
  maxOutputTokensRecoveryCount,
  hasAttemptedReactiveCompact,
  turnCount,
  pendingToolUseSummary,
  transition,
}
```

一个普通 orchestrator 不会长期维护 `autoCompactTracking`、`maxOutputTokensRecoveryCount`、`pendingToolUseSummary`。这些状态进入主循环，意味着系统承认：**一次 agent turn 会被压缩、恢复、工具回灌、预算和中断反复改写。**

## 核心骨架

```python
while True:
    prefetchMemoryAndSkills()
    messagesForQuery = applyBudget(messages)
    messagesForQuery = snipAndCompact(messagesForQuery)
    assistant = streamModel(messagesForQuery)
    if not assistant.hasToolUse: return finishTurn(assistant)
    toolResult = runToolUse(assistant.toolUse, toolUseContext)
    state.messages = writeBack(messages, assistant, toolResult)
```

**真正高明的点不是 while loop 本身，而是把 `prefetch`、`budget`、`compact`、`tool result write-back` 全部拉回主循环正中央。**

## 工具结果回灌的协议化

```python
messagesForQuery = getMessagesAfterCompactBoundary(messages)
assistantMessages = streamModel(normalize(messagesForQuery))
toolUseBlocks = collectToolUses(assistantMessages)
toolResults = runTools(toolUseBlocks)
state.messages = [...messages, ...assistantMessages, ...toolResults]
```

工具结果被重新标准化成 user message，再回灌到主消息流。**很多 Agent 系统在这里偷懒，工具执行链和会话链是两张皮。**

## "坏路径也是主路径"

- `prompt-too-long` → reactive compact
- `max_output_tokens` → recovery count / escalation
- API 失败 → fallback model
- 连续 3 次 compact 失败 → 本会话禁用压缩（**熔断保护**）

源码注释揭示：全球每天约 25 万次 API 调用被失败的压缩循环浪费——这正是熔断保护引入的真实原因。

---

# 六、Tool Runtime：把野生工具变成系统调用

## 普通团队的 Tool 抽象

```typescript
type Tool = (input: unknown) => Promise<string>
```

## Claude Code 的 Tool 抽象

```typescript
interface Tool {
  name: string
  inputSchema: Schema
  canRunInParallel: boolean
  validate(input): ValidationResult
  execute(input, context): AsyncIterable<ToolEvent>
  toModelResult(output): StructuredResult
}
```

**真正的差别不在写法，而在系统观。**前者只是"模型能调一个函数"，后者才是"runtime 知道这个动作该怎么被约束、观测、并发和回灌"。

## fail-closed 默认值

```typescript
const TOOL_DEFAULTS = {
  isConcurrencySafe: () => false,    // 默认不安全 → 串行
  isReadOnly: () => false,            // 默认有写入 → 需要权限
  checkPermissions: () => 'allow',
  toAutoClassifierInput: () => '',    // 默认跳过安全分类器
}
```

新增工具时忘记声明并发安全性？→ **自动串行执行**。忘记声明只读？→ **自动触发权限检查**。

## 权限拒绝也被纳入统一协议

```python
if permissionDecision.behavior !== 'allow':
    return rejectAsToolResult()  # 拒绝也被包装成标准 tool_result
```

**Claude Code 连"被拒绝"这件事都纳入了统一协议。**主循环不需要知道这次是"执行成功"还是"权限拒绝"——它只需要知道"我收到了一份结构化结果"。

## Tool Runtime 收敛的横切关注点

参数校验、权限检查、并发治理、进度上报、错误归一化、结果回填——**六件事全部统一，新工具作者不需要重新发明。**

---

# 七、Permission System：四层防御纵深

## 不是弹窗系统，是完整决策链

```
① 规则层（Rule Layer）：匹配允许/拒绝/待确认，保留来源和理由
② 运行时判定层：classifier、hooks、coordinator 自动决策
③ 交互层：真的需要用户参与时才走确认
④ 执行隔离层（Sandbox）：把逻辑权限映射成真实的文件/网络/命令边界
```

## 最关键的判断：逻辑授权和执行隔离必须打通

很多 Agent 只做前者（权限像提示框）；有些只做后者（沙箱像硬隔离）。**Claude Code 的成熟点：两层是打通的。**

## 权限决策不是布尔值

```typescript
type PermissionDecision =
  | { behavior: 'allow'; updatedInput?; decisionReason? }
  | { behavior: 'ask'; message; suggestions?; blockedPath?; pendingClassifierCheck? }
  | { behavior: 'deny'; message; decisionReason }
```

**`decisionReason`、`suggestions`、`blockedPath`、`pendingClassifierCheck` 都是正式字段。**权限不再只是"过不过"，而是"为什么过、卡在哪、下一步怎么处理"。

## 实际效果
- ML 分类器在 Auto Mode 下实现 **93% 批准率**
- Sandbox 相比纯权限提示减少 **84% 权限弹窗**

---

# 八、上下文管理：三重压缩 + 自愈

## 三层压缩体系

| 策略 | 触发 | 机制 |
|------|------|------|
| **MicroCompact** | 工具结果过期 | 纯规则驱动，无 LLM 调用，最低成本 |
| **autoCompact** | token 超过阈值 | 调用 Claude API 生成结构化摘要（9 段模板） |
| **contextCollapse** | 上下文结构性低效 | 重构上下文组织方式 |

**按成本由低到高执行：便宜的本地操作优先，昂贵的 API 调用仅在必要时。**

## 熔断保护

连续 3 次 autoCompact 失败 → 本会话禁用压缩。源码注释揭示全球每天约 25 万次 API 调用被失败压缩循环浪费。

## Memdir 结构化记忆

四种核心类型：User（用户偏好）、Feedback（反馈修正）、Project（技术决策）、Reference（文档片段）。LLM-in-the-loop 检索：每次最多 5 条最相关记忆。

---

# 九、多 Agent 系统：先统一任务抽象

## 普通"多 Agent"实现

```typescript
spawnAgent(prompt): Promise<string>
```

## Claude Code 的任务抽象

```typescript
interface Task {
  id: string
  status: 'pending' | 'running' | 'blocked' | 'done' | 'failed'
  progress: ProgressState
  output: StructuredOutput[]
  notifications: Notification[]
  cancel(): void
  resume(): void
}
```

**前者只是多开了一个智能体，后者才是把"持续执行"变成系统里的正式对象。**

## 四种 Spawn 模式

| 模式 | 隔离级别 | 适用场景 |
|------|---------|---------|
| default | messages[] 共享 | 简单委派 |
| fork | 全新 messages[] | 研究性任务 |
| worktree | 独立目录 + 全新消息 | 并行修改不同功能 |
| remote | Bridge to Container | 完全隔离 |

## 最精妙的设计：Fork Agent 的工具输出不污染主对话

> "creates a fork, which runs in the background and keeps its tool output out of your context — so you can keep chatting with the user while it works"

---

# 十、六大内置 Agent 角色

| Agent | 模型 | 权限 | 一句话 |
|-------|------|------|--------|
| **General-Purpose** | 默认 | 全部工具 | "万能打工人" |
| **Explore** | Haiku | 严格只读 | "代码库侦察兵"，快+便宜 |
| **Plan** | 继承主模型 | 严格只读 | "软件架构师" |
| **Verification** | 继承主模型 | 只读+可写/tmp | "红蓝对抗质检官" |
| **Claude Code Guide** | Haiku | dontAsk | "自我说明书" |
| **Statusline Setup** | Sonnet | Read+Edit | "终端美化师" |

## Verification Agent 最值得讲（面试时最出彩）

- **不是确认代码能跑，而是想办法把它搞崩**
- 反偷懒话术："代码看起来是对的" — 看起来不是验证，运行它。"这大概没问题" — "大概"不是"验证过了"
- 为十几种变更类型定义了专用验证策略（前端/后端/CLI/基础设施/Bug修复/数据库迁移/重构/移动端）
- 严格的权限控制：只能看不能改，例外是 `/tmp` 写测试脚本

---

# 十一、System Prompt 动态组装：六步流水线

## 为什么这层重要

很多人以为 Prompt Engineering 就是"写一个漂亮的 System Prompt"。但 Claude Code 真正的工程含量在于——System Prompt 不是写出来的，是**组装出来的**。它由多个文件协同工作，最终拼装成字符串数组发送给 API。

## 六步组装流程

**Step 1：QueryEngine.ask() 发起请求**

**Step 2：并行获取三大组件**
- `defaultSystemPrompt`：调用 `getSystemPrompt()` 构建默认 prompt
- `systemContext`：获取 Git 状态快照
- `userContext`：获取 CLAUDE.md 内容 + 当前日期

**Step 3：getSystemPrompt() 核心组装**——把 prompt 分成静态部分和动态部分。静态部分每个用户都一样（身份介绍、工具使用规则、安全守则），动态部分每个用户/会话可能不同。

**Step 4：优先级决策**——`buildEffectiveSystemPrompt()` 按优先级选择最终使用的 prompt。优先级从高到低：用户自定义 → CLAUDE.md → 默认系统 prompt。

**Step 5：注入上下文信息**
- `appendSystemContext()`：把 Git 状态追加到 System Prompt 末尾
- `prependUserContext()`：把 CLAUDE.md 内容和当前日期作为 `<system-reminder>` 消息插入到用户消息列表最前面

**Step 6：缓存分块**——`splitSysPromptPrefix()` 把 System Prompt 拆分成缓存友好的块，明确标注哪些是 Prefix（可走 KV Cache），哪些不需要缓存。

## CLAUDE.md 的四层路径

CLAUDE.md 是 Claude Code 的"项目说明书"，支持四种路径：

| 路径 | 用途 | Git |
|------|------|-----|
| `~/.claude/CLAUDE.md` | 个人全局人设（"始终用中文回复"） | 不提交 |
| 项目根 `CLAUDE.md` | 团队共享规范（架构、编码规范、构建命令） | 必须提交 |
| `CLAUDE.local.md` | 个人私有指令（测试账号、敏感配置） | 不提交 |
| `.claude/rules/*.md` | 按文件类型分类的规则（前端/后端/测试） | 提交 |

## 和 OpenClaw 的对比

Claude Code 用 CLAUDE.md 就够了——它是 AI Coding Agent，需要的是"项目要求"。OpenClaw 需要 AGENTS.md + SOUL.md + IDENTITY.md + USER.md + TOOLS.md + HEARTBEAT.md + MEMORY.md——它是个人 AI 助理，需要更细粒度的身份和行为定义。**两者都是 Markdown 文件驱动，但文件类型的差异直接反映了产品定位的差异。**

---

# 十二、REPL / UI Orchestration：UI 不是传话筒

很多 Agent 团队把聊天 UI 理解成"显示消息的壳"。Claude Code 明显不是——它的 REPL 本质上是一个**运行时控制台**。

## REPL 的两大职责

**职责一：汇总当前能力面**

REPL 在用户提交输入的瞬间，把以下全部汇在一起生成 turn-scoped 执行上下文：
- 本地 tools + MCP tools + plugin commands
- 动态 skills + 任务状态
- 权限确认队列 + MCP 连接状态
- remote session 信息

REPL 先把"这一轮在什么制度下运行"准备好，再把控制权交给推理循环。

**职责二：归并当前事件流**

REPL 消费的不是纯文本，而是一串带语义的事件流：
- assistant message → tool progress → compact boundary → pending permission → task notification → API error

所有这些事件在 REPL 里重新归并成用户能理解的会话视图。REPL 既是 query 的入口，也是整套运行时事件的落点。

## 设计价值

> "UI 不再只是消费文本，而是在消费统一事件协议。query、permission、tool runtime、task system 都可以通过结构化事件和 REPL 协作，而不是各自偷偷改 UI 状态。"

对于需要权限确认、长时执行、工具调用的 Agent，这种可控感往往比多一点模型智商更重要。用户看到的不只是一句回复，而是系统正在执行什么、为什么停下来、当前有哪些能力、后台有哪些任务。

---

# 十三、Memdir 结构化记忆 + System Reminder 动态注入

## Memdir：四种记忆类型

Claude Code 的记忆系统叫 Memdir，把记忆拆解为四种结构化类型：

| 类型 | 内容 | 作用 |
|------|------|------|
| **User** | 用户偏好、操作习惯、指令风格 | "越用越懂你" |
| **Feedback** | 行为修正记录、历史纠错案例 | "避坑指南" |
| **Project** | 技术选型、架构决策、约束条件 | 多轮对话技术立场一致 |
| **Reference** | 通用文档片段、代码模式 | 高频调用的知识底座 |

## LLM-in-the-loop 检索

当记忆库规模扩大时，Claude Code 不使用简单的关键词匹配或固定相似度阈值。它用 **Sonnet 模型** 做语义相关性判断——让大模型充当"图书管理员"，约束每次最多返回 5 条最相关记忆。

> "用 LLM 的推理能力解决传统检索在复杂语义下的失效问题，同时通过数量限制严格控制 Token 消耗和延迟。"

## System Reminder 动态注入

`wrapInSystemReminder()` 是 Claude Code 所有系统元信息的统一包裹机制。它把 CLI 配置、日期、工具执行结果、Hook 反馈、待办任务提醒、Skill List、Agent List 等全部用 `<system-reminder>...</system-reminder>` 标签包裹，明确告诉模型"这是系统注入的元信息，不是用户输入"。

这个机制被深度集成到 `normalizeMessagesForAPI` 中——**上下文的组装不是手动拼接字符串，而是一条标准化的工程流水线。**

---

# 十四、Bridge 系统 + 编译时特征门控 + 隐藏特性

## Bridge：终端到 IDE 的桥梁

Claude Code 不只是 CLI 工具，它通过 Bridge 系统将操作环境从纯终端扩展到 IDE：

- VS Code / JetBrains 双向通信
- REPL 会话桥接——在 IDE 中恢复终端会话
- 将 Agent 的执行能力注入到开发者已有的 IDE 工作流中

## 编译时特征门控（Feature Flags + DCE）

Claude Code 利用 **Bun 的 `bun:bundle` 特性标志**实现编译时死代码消除：

- `feature("KAIROS")` → 未启用时整个 KAIROS 子系统在 bundle 中不存在
- 通过 **GrowthBook** 做 A/B 测试与渐进式发布
- 编译时消除（不是运行时判断）意味着：实验性功能对生产 bundle 的大小和性能零影响

## 隐藏特性（未发布但已近乎完成）

**KAIROS（全自主代理模式）**：源码中被引用 150+ 次的最大的未发布特性。名字源于希腊语"对的时机"。把 Claude Code 从一个"工具"变成一个"守护进程"：
- 主动监控 GitHub Issues / Slack 消息
- 自动认领任务、在后台工作
- **Auto-Dream**：四阶段记忆整固系统（摄入→关联→梦境→巩固）
- 独占工具：推送通知、文件发送、自主模式专用工具

**Undercover Mode**：隐藏 AI 身份——专门防止 Anthropic 内部信息泄露到公共 git 提交。讽刺的是，这个防止泄露的子系统，因为 source map 泄露被全世界看到了。

**Anti-Distillation（反蒸馏）**：防止竞争模型通过 API 流量训练——假工具注入、挫败感正则匹配。背景是多家 AI 公司被指控通过 API 调用抓取 Claude 输出来训练竞争模型。反蒸馏是对这类窃取的直接回应。

---

# 十五、设计模式总结：Claude Code 教我们的 8 个原则

1. **开闭原则在 Agent 中的应用**：核心循环从不改变。新功能 = 新工具 或 新包装层。这比每加一个功能就改主循环的做法高明一个数量级。

2. **复杂度前置**：把模式判断、权限边界、宿主差异压到启动层解决，运行时主链路反而更纯。

3. **胖核心 + 薄扩展**：`query.ts` 约 785KB 不是混乱，是刻意让流式处理、工具调度、压缩、子代理管理在同一个文件中协调，保持核心循环的原子性。

4. **状态机而非函数调用**：Query Loop 维护跨迭代的压缩状态、恢复计数、预算管理。承认一次 agent turn 会被压缩、恢复、工具回灌、预算和中断反复改写。

5. **fail-closed 默认值**：新增工具默认串行、默认需要权限、默认跳过安全分类器——安全的默认比显式的权限检查更重要。

6. **统一协议收归横切关注点**：参数校验、权限检查、并发治理、进度上报、错误归一化、结果回填——六件事全部统一，新工具作者不需要重新发明。

7. **坏路径也是主路径**：prompt-too-long → reactive compact；API 失败 → fallback model；连续 3 次 compact 失败 → 熔断。全球每天约 25 万次 API 调用被失败的压缩循环浪费——正因如此才有熔断保护。

8. **编译时优于运行时**：通过 Bun feature flags + DCE 让实验性功能对生产 bundle 零影响。不是用 `if` 判断，而是编译时就消除。

---

# 十六、和 KTClaw 的关系（面试最该讲）

面试时可以这样串联：

> "Claude Code 的三个核心设计最让我印象深刻——也是我在 KTClaw 里尝试践行的。第一是三条主干链路的分离：控制链管推理节奏、执行链管行动治理、任务链管并发和生命周期。KTClaw 的三进程架构本质上也是把这三件事分开——Renderer 管控制面、Main 管调度、Gateway 管执行。第二是 fail-closed 默认值：新增工具默认串行、默认需要权限——安全的默认比显式的权限检查更重要。KTClaw 的安全模型也是这个思路。第三是 Fork Agent 的上下文隔离——工具输出不污染主对话。这恰好是我在 KTClaw 里做多 Agent 架构时最核心的设计动机。"

---

# 十七、高频面试题

## Q1. Claude Code 的 512K 行代码里只有 1.6% 是 AI 逻辑——这说明什么？

### 主答模板
说明生产级 Agent 的核心竞争力不在模型，而在 Harness 基础设施。权限系统、上下文压缩、工具注册、错误恢复、状态持久化、任务调度——这些才是决定一个 Agent 能不能稳定运行的关键。模型决定了上限，Harness 决定了下限。大多数 Agent 系统卡住不是模型不够聪明，而是 Harness 接不住真实世界的复杂度。

## Q2. Claude Code 的 Query Loop 和普通"调一次 API"有什么区别？

### 主答模板
普通实现是一次函数调用：拿历史消息→调 API→拿到结果→结束。Claude Code 的 Query Loop 是一个状态机：维护跨迭代的压缩状态、恢复计数、预算管理、失败回退。它承认一次 agent turn 会被压缩、恢复、工具回灌、预算和中断反复改写。它还把 prefetch、budget、compact、tool write-back 全部拉进主循环正中央，而不是散落在调用点。最关键的工程态度是"坏路径也是主路径"。

## Q3. Claude Code 的 Verification Agent 为什么重要？

### 主答模板
因为它直接回应了 Anthropic 自己的核心发现：Agent 无法准确评估自身产出。Verification Agent 不是一个 smarter model，而是一个被制度约束的独立评判者——它只能读不能改、它默认不信任实现者（因为实现者也是 AI）、它被明确训练成"想办法把代码搞崩"而不是"确认代码能跑"。这是"把做事的和评判的分开"在 Agent 系统中的最高级实践。

---

## Q4. Claude Code 的启动层为什么要拆成三段？单段 main 不行吗？

### 主答模板
单段 main 的问题是入口一多就会裂。一个成熟 Agent 要同时支持本地交互、headless、SDK、remote、后台 session、会话恢复。如果启动层不先把模式、边界、权限和上下文装配清楚，每个宿主都会偷偷长出自己的运行语义，最终系统裂成几套互不兼容的实现。

Claude Code 的三段启动——入口分流→进程初始化→会话准备——把几个最容易搅在一起的问题拆开了：启动模式是什么、运行环境是否就绪、当前会话制度是什么、最后由谁承载会话。

最关键的设计是**先装配共享 session/runtime 语义，再选择交互式或 headless 宿主**。这让无界面运行、交互式运行、远程运行、后台运行可以共享同一套核心 runtime，而不是各自长一套逻辑。

另外，把进程状态（cwd、sessionId、telemetry）和交互状态（tasks、MCP、permission context）显式分开，也是一个容易被忽略但对架构非常关键的判断。

---

## Q5. Claude Code 的 System Prompt 组装和普通"写一个 prompt"有什么本质区别？

### 主答模板
普通做法是把 prompt 写死在代码里或配置文件里，改一次要发版。Claude Code 的做法是**六步动态组装流水线**：并行获取三大组件（default + systemContext + userContext）→ 静态/动态分离 → 优先级决策 → Git 状态注入 → CLAUDE.md 注入 → 缓存分块。

三个关键工程决策：
1. **静态/动态分离**：静态部分每个用户都一样，可走 KV Cache；动态部分每个会话不同，单独注入
2. **缓存友好分块**：`splitSysPromptPrefix()` 明确标注哪些是 Prefix——显式告诉 API 缓存边界，提高命中率
3. **CLAUDE.md 四层路径**：全局/项目/本地/规则——不同内容放在不同层级，Git 提交策略也不同

这背后是一种思路转变——从"Prompt 是写出来的"升级到"Prompt 是组装出来的"。这和 OpenClaw、Hermes 的 System Prompt 拼装逻辑是同一个方向。

---

## Q6. Claude Code 有哪些未发布但你认为最值得关注的能力？

### 主答模板
三个最值得关注的未发布特性：

**KAIROS（全自主代理模式）**：源码中引用 150+ 次的最大的未发布特性。把 Claude Code 从"工具"变成"守护进程"——主动监控 GitHub Issues/Slack、自动认领任务、后台工作。最迷人的部分是 Auto-Dream——四阶段记忆整固系统（摄入→关联→梦境→巩固），类似 OpenClaw Dreaming 但集成在自主代理循环中。

**Undercover Mode**：隐藏 AI 身份的系统——专门防止 Anthropic 内部信息泄露到公共 git 提交。讽刺的是，这个防止泄露的子系统因为 source map 泄露被全世界看到了。

**Anti-Distillation（反蒸馏）**：防止竞争模型通过 API 调用抓取 Claude 输出来训练。通过假工具注入和正则匹配识别自动化抓取行为。反映了头部 AI 公司对模型被盗的深度焦虑。

这三个特性揭示了一个趋势：下一代 Agent 不只是"更好的工具"，而是"自主运行 + 安全可控 + 知识产权保护"的三位一体。

---

# 十八、面试金句

1. **"512K 行代码，1.6% 是 AI 逻辑，98.4% 是 Harness。这个比例本身就在定义 Agent 工程的核心竞争力。"**
2. **"Claude Code 没有试图用一个'大一统 Agent 核心'吃掉所有问题，而是让每种复杂度只在一个地方爆炸。"**
3. **"Tool Runtime 不是'给模型挂几个函数'，而是把一次外部行动放进统一协议，让主循环继续活下去。"**
4. **"权限系统的成熟标志不是它拦得有多凶，而是能不能把风险控制、自动化和可解释性同时放进一套机制里。"**
5. **"Fork Agent 的工具输出不污染主对话——这正是我在 KTClaw 做多 Agent 架构时的核心设计动机。"**
6. **"启动层的质量平时不显山不露水，但系统一旦开始长模式、长宿主、长入口，它就是最先决定架构寿命的那一层。"**
7. **"Prompt 不是写出来的，是组装出来的。静态/动态分离 + 缓存分块 + 优先级决策 = 生产级 Prompt Engineering。"**
8. **"REPL 不是聊天 UI，是运行时控制台——它汇总能力面、归并事件流，让用户看到的不是'模型回了什么'而是'系统在做什么'。"**
9. **"KAIROS 把 Claude Code 从工具变成守护进程——这才是 Agent 的终极形态：不是你在用工具，而是工具在替你主动工作。"**
10. **"坏路径也是主路径——A production agent doesn't just succeed well; it fails well."**
