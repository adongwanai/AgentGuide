---
type: 求职指南
status: 已发布
level: 高阶
topic:
  - 面试求职
  - Coding Agent
  - Agent
---

# OpenClaw 源码分析 彻底备战手册
> **本手册定位**：面向 AI Agent 求职与面试准备的专题复习资料，用于补充 AgentGuide 面试题库与项目讲述材料。
> 内容以公开资料、项目复盘和工程实践方法论为基础做二次整理，建议结合自己的真实项目经历进行取舍与改写。

---
## 目标：把 OpenClaw 的架构、设计哲学和关键实现讲得像真正读过源码、做过对比的人

> 使用建议：OpenClaw 是 2026 年最重要的开源 Agent 系统。你的 KTClaw 项目底层就是基于 OpenClaw SDK，面试官如果问"你对 OpenClaw 有什么了解"或者"KTClaw 和 OpenClaw 有什么关系"，这份手册让你能讲出源码级的理解。
>
> 综合吸收了三份材料：
> 1. `openclaw (1).md`：源码级架构拆解，Gateway/Pi/Session/Channels/Security 全覆盖
> 2. `深入理解OpenClaw技术架构与实现原理（上）.md`：系统模块详解
> 3. `深度解析_OpenClaw_Prompt_Context_Harness.md`：Prompt/Context/Harness 三维分析
>
> 已在前几份手册覆盖、不再重复的：Memory/Dreaming（Agent Memory 手册）、Skills（Skills 手册）、Harness 工程（Harness 手册）

---

# 一、30 秒 + 2 分钟版本

## 30 秒版本
OpenClaw 是全世界增长最快的开源 Agent 项目，由 Peter Steinberger 在 2025 年 11 月启动，GitHub 330K+ stars。它不是又一个 LangChain wrapper，而是一个 **local-first 的个人 AI 操作系统**——把 LLM 变成 24/7 在所有聊天 App 里活着的代理。核心技术栈：Hub-and-Spoke 架构 + 单 Gateway 控制平面 + Pi Agent Runtime + 文件系统即状态机的 Session 管理 + 25+ 渠道统一适配。

## 2 分钟版本
OpenClaw 从架构上可以拆成五层：
1. **Gateway 控制平面**：WebSocket v3 协议、100+ RPC 方法、RBAC 四层权限模型。整个系统**每台主机只允许一个 Gateway**——因为 WhatsApp Baileys 协议严格单设备
2. **Pi Agent Runtime**：~200 行 agent loop + 4 个工具 + <1000 tokens system prompt。Mario 的设计哲学是"Trust the model, minimize the scaffolding"
3. **Session 管理**：append-only JSONL + id/parentId 树结构——这是 **git 的数据结构映射到 LLM 会话**。分支零成本（新 id + 指向父），compaction 不删除历史
4. **Channels 抽象**：Dock（轻量元数据层）/ Plugin（重量实现层）两层分离，25+ 渠道统一适配
5. **安全模型**：三道独立闸门（入站认证 + 执行审批 + Docker 沙箱），假设模型终将被 jailbreak

关键是它的不做清单——no MCP、no sub-agents、no permission popups、no plan mode、no built-in to-dos。这是小团队打败大 framework 的秘诀。

---

# 二、核心架构：Hub-and-Spoke

## 架构全景图

```
                     ┌──────────────┐
                     │   Channels   │  25+ 平台 (WhatsApp/Telegram/Discord/...)
                     └──────┬───────┘
                            │ WebSocket v3
                     ┌──────▼───────┐
                     │   Gateway    │  单进程 Node.js，默认端口 18789
                     │              │  路由 · 权限 · 事件 · 状态 · RPC
                     └──────┬───────┘
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                  ▼
   ┌──────────┐    ┌──────────────┐    ┌──────────┐
   │ Pi Agent │    │   Tools      │    │  Memory  │
   │ Runtime  │    │ exec/browser │    │ JSONL    │
   │ ~200行   │    │ /canvas/nodes│    │ files    │
   └──────────┘    └──────────────┘    └──────────┘
```

## 关键设计决策

### 单 Gateway 的硬约束
> **"Exactly one Gateway controls a single Baileys session per host."**

WhatsApp Web 这类 IM 协议通常会限制同一账号/设备会话的并发登录。Baileys 会话是**有状态**的加密 session key，必须持久驻留。所以 Gateway 是 stateful 的，不能做 stateless replica。这是**外部约束驱动内部架构**的经典案例。

### OpenClaw vs Pi 的分工
OpenClaw 和 Pi 的分工非常优雅：
- **Pi 负责"让 LLM 能行动"**：最小可行的 agent loop
- **OpenClaw 负责"让 LLM 持久存在"**：Gateway + Channels + Memory + Cron + Identity

这是 **mechanism（机制）vs policy（策略）的经典分层**，Unix 哲学在 AI Agent 时代的转世。

### 不做清单（最重要的架构决策）
- No MCP（用 mcporter CLI 桥接）
- No sub-agents（用 tmux 或 extension）
- No permission popups（用容器或 extension）
- No plan mode（用 Markdown 文件）
- No built-in to-dos（"they confuse models"——原话）
- No background bash（用 tmux）

**元原则**："If I don't need it, it won't be built." 这**不是懒，是克制**。

---

# 三、Gateway：WebSocket v3 协议

## 三种 Frame 类型

```typescript
type GatewayFrame =
  | { type: "req"; id: string; method: string; params: unknown }     // 请求
  | { type: "res"; id: string; ok: boolean; payload?: unknown }      // 响应
  | { type: "event"; event: string; payload: unknown; seq?: number } // 事件推送
```

## 连接握手

1. 服务端先发 `connect.challenge` 事件（含 nonce）
2. 客户端发 `connect` 请求，device 字段含签名后的 nonce
3. 服务端回 `hello-ok`，含协商版本 + state 快照 + device token

## RBAC 四层权限

```
Role 层（operator vs node）
  → Scope 层（operator.read / write / admin / pairing）
    → Method 层（某些 method 有额外检查）
      → Command 层（slash 命令还要过 command check）
```

## 关键约束

- Events 断开重连后**不重放**——客户端检测到 seq gap 必须刷新整个状态
- 副作用方法必须带 **idempotency key**
- Schema 跨语言生成：TypeBox → JSON Schema → Swift/Kotlin struct

---

# 四、Pi Agent Runtime

## AgentLoop 伪代码（200 行核心）

```
while true:
  1. REASON: 调 LLM（pi-ai 做 provider 归一化）
  2. 判断: 没 tool_call → 结束
  3. PREFLIGHT: beforeToolCall hook（可 block）
  4. ACT: 并行或串行执行工具
  5. POSTFLIGHT: afterToolCall hook（可 terminate）
  6. OBSERVE: tool results 注入 context
  7. STEERING: 检查用户是否插队（mid-run 打断）
  8. 循环继续
```

## 关键设计

- **没有 maxSteps**——loop 直到 agent 说 done。Mario 原话："I never found a use case for that, so why add it?"
- **Steering vs Follow-up**：Steering = Enter 打断当前执行，Follow-up = Alt+Enter 等待自然完成
- **Tool 默认并行执行**，per-tool 可覆盖为 sequential

## Cross-provider Context Handoff

pi-ai 层做 provider 归一化，可以从 Anthropic 切到 OpenAI 再切回来。Anthropic 的 thinking traces 在切到 OpenAI 时被包在 `<thinking>` 标签里。**但 Mario 诚实地说这只是一种 best-effort 妥协**——不同 provider 的 thinking trace 并不真正代表模型内部的推理过程。

---

# 五、Session 管理：文件系统即状态机

## JSONL 树结构

```jsonl
{"type":"session","id":"uuid-1","cwd":"...","timestamp":...,"parentSession":null}
{"type":"message","id":"e1","parentId":null,...}
{"type":"message","id":"e2","parentId":"e1",...}
{"type":"message","id":"e3","parentId":"e2",...}
{"type":"compaction","id":"e4","parentId":"e2","firstKeptEntryId":"e2","summary":"..."}
```

**五种 entry 类型**：message / custom_message / custom / compaction / branch_summary

## 核心洞察

> "这种 append-only JSONL + id/parentId tree 是 **git 的数据结构映射到 LLM 会话**。"
>
> - 每次 write 都是 append（ACID 简单）
> - 分支是零成本（新 id + 指向父，不拷贝数据）
> - compaction 不删除历史（被替换的 entries 留在文件里，只是不被 active path 访问）
> - 这给了三个免费礼物：**可回滚、可对比分支、可审计历史**

## Session Key 解析

**整个 OpenClaw 最热的代码路径**。每条入站消息都会过：

```
DM scope = main → 所有 DM 合并到同一 session（最 persistent identity）
DM scope = per-channel-peer → 每个人每个渠道独立 session（多用户安全推荐）
Group chat → 总是独立 key
Telegram topic → 在 group key 基础上追加 -topic-{threadId}
```

---

# 六、Channels：两层抽象

## Dock vs Plugin

| 层 | ChannelDock（轻量） | ChannelPlugin（重量） |
|---|---|---|
| **包含** | capabilities 元数据、commands、outbound、streaming | 完整实现 + lifecycle hooks + monitor |
| **依赖** | 无重量级 SDK | grammY / Baileys / discord.js 等 |
| **用途** | 让上层代码不 import 几十 MB 的 Baileys | 实际协议适配 |

**为什么分层**：共享代码路径（reply flow、command auth、sandbox explain）不能 import 整个 Baileys（几十 MB）。Dock 用 capabilities 元数据告诉上层"这个渠道支持 polls 吗？mention 模式是什么？"——agent 据此决定能做什么。

## 25+ 渠道适配对比

| Channel | SDK | Auth | 特殊性 |
|---------|-----|------|--------|
| WhatsApp | Baileys（逆向协议） | QR code linked device | 必须 Gateway 直连，strict single device |
| Telegram | grammY | Bot token | Sequentialize middleware 防并发破坏 context |
| Signal | signal-cli（外部 Java daemon） | QR / phone | JSON-RPC 2.0 over HTTP + SSE |
| iMessage | AppleScript + BlueBubbles | macOS 权限 | 必须跑 macOS |
| Discord | discord.js / Carbon | Bot token | 支持 thread 和 forum channel |

---

# 七、Context 管理

## 上下文压缩触发

**两种触发模式**：
1. **Error-driven**：LLM 返回 "request_too_large" / "context length exceeded" → 触发压缩 → 重试
2. **Threshold-driven**：`contextTokens > contextWindow - 20000 - softThreshold` → 触发压缩

## Multi-stage Compaction Pipeline

```
Stage 1: 分块（tool_call 必须和 matching tool_result 配对在同一块）
Stage 2: 尾部保护（pending tool block 不压缩）
Stage 3: 错误 block 不 hold（aborted/error tool-call 可被切）
Stage 4: 逐块 summarize（可配置专用便宜模型）
Stage 5: Replace（原 entries 留在 JSONL 做 audit）
```

## Compaction vs Pruning

| | Compaction | Pruning |
|---|---|---|
| 作用范围 | 整个历史 | 仅 toolResult |
| 持久化 | ✓ 写入 JSONL | ✗ 仅内存 |
| 信息保留 | 生成摘要 | 直接删减（有损） |
| 成本 | 调用 LLM | 规则裁剪（低成本） |

---

# 八、安全模型：三道独立闸门

## 安全哲学

> "System prompt guardrails are soft guidance only; hard enforcement comes from tool policy, exec approvals, sandboxing, and channel allowlists."

假设 LLM **终将被 jailbreak**，所以三道闸门独立运作、任一道都能阻止：

1. **入站层**：DM pairing、allowlist、auth token。未知发送者必须通过配对码验证
2. **执行层**：exec approval pipeline、browser 沙箱、Docker 隔离。敏感操作必须显式批准
3. **审批层**：高风险命令（如 `rm -rf`、`sudo`）需要 human-in-the-loop 显式确认

---

# 九、和 KTClaw 的关联（面试时最该讲的）

## KTClaw 基于 OpenClaw SDK

你的 KTClaw 项目底部跑的就是 OpenClaw Agent SDK / Pi Agent Runtime。面试时可以这样讲：

> "KTClaw 的底层 Agent 引擎是 OpenClaw SDK，我们基于它做了企业场景的定制化 Harness——包括三进程分层架构（把 Gateway 的安全边界进一步强化）、多 IM 渠道的适配器模式（借鉴了 ChannelDock 的思想但做了企业级的消息归一化）、以及针对企业知识问答场景的记忆和检索优化。"

## KTClaw 在哪些地方超越了简单的 OpenClaw 部署

1. **三进程架构**：OpenClaw 的 Gateway 是单进程，KTClaw 拆成 Renderer/Main/Gateway 三层，把安全边界从"同一进程内的权限检查"升级成了"进程级别的物理隔离"
2. **企业级渠道适配**：OpenClaw 的 ChannelDock 解决的是渠道能力差异，KTClaw 的 ChannelAdapter 解决的是企业 IM 消息格式的深度归一化
3. **分层记忆 + 三路召回**：OpenClaw 的 Memory 是 MEMORY.md + 每日笔记，KTClaw 升级成了四层记忆架构 + 元数据/语义/流程关系三路召回
4. **评测体系**：OpenClaw 没有内置的系统化 Agent 评测，KTClaw 加上了稳定性、命中率、复用率等工程评测

---

# 十、高频面试题 + 答案

## Q1. OpenClaw 的架构核心是什么？

### 主答模板
Hub-and-Spoke 架构：中心一个 Gateway 作为控制平面，25+ 渠道、Pi Agent Runtime、工具、记忆系统都挂在这个 Hub 上。所有组件通过 WebSocket v3 协议通信。整个系统每台主机只允许一个 Gateway——因为 WhatsApp Baileys 协议严格单设备，这是外部约束驱动内部架构的经典案例。

---

## Q2. OpenClaw 和 LangChain 的本质区别是什么？

### 主答模板
LangChain 解决的是"怎么编排 LLM 调用"——它是 framework。OpenClaw 解决的是"怎么让 LLM 有身份、有记忆、有日程、能被所有人在所有渠道找到、并真正操作你的电脑"——它是 **local-first 的个人 AI 操作系统**。核心差异：LangChain 让 LLM 跑一次任务，OpenClaw 让 LLM 成为持久存在的计算实体。

---

## Q3. 你怎么看 OpenClaw 的 JSONL Session 树结构设计？

### 主答模板
这是 OpenClaw 架构里我最欣赏的设计。它把 git 的数据结构映射到 LLM 会话上：append-only JSONL + id/parentId 树。每条新消息只是追加一行，分支零成本（新 id + 指向父），compaction 不删除历史（被替换的 entries 留在文件里只是不被 active path 访问）。这给了三个免费礼物：可回滚、可对比分支、可审计历史。工程上的简洁性令人敬佩。

---

## Q4. OpenClaw 的"不做清单"对你有什么启发？

### 主答模板
Mario 的原话是"If I don't need it, it won't be built." 这看起来像懒，实际上是最难的架构决策——**刻意不做**。no MCP、no sub-agents、no plan mode、no built-in to-dos。这些功能不是不重要，而是"可以用已有机制（extension、tmux、markdown 文件）组合出来"。OpenClaw 的核心只有 ~200 行 agent loop + 4 个工具——极简内核让任何人都能一个下午读完并贡献。这是小团队打败大 framework 的秘诀。

---

## Q5. KTClaw 和 OpenClaw 是什么关系？你做了什么 OpenClaw 没有的事？

### 主答模板
KTClaw 的底层 Agent 引擎就是 OpenClaw SDK，但我们不是简单部署了一个 OpenClaw 实例。我们在四个维度做了深度定制：
1. **三进程架构**：把安全边界从权限检查升级到进程级物理隔离
2. **企业级渠道归一化**：ChannelAdapter 做深度消息格式统一
3. **分层记忆 + 三路召回**：比 MEMORY.md 的范式更结构化
4. **系统化 Agent 评测**：这是 OpenClaw 原生不具备的能力

---

# 十一、最后：OpenClaw 准备到什么程度才算真准备好了？

## 你至少要做到 6 件事
1. **能讲出 Hub-and-Spoke 架构和单 Gateway 约束的原因**
2. **能讲出 AgentLoop 的 8 步核心循环**
3. **能解释 JSONL Session 树为什么是"git 的数据结构映射"**
4. **能讲出 Channels 的 Dock/Plugin 两层抽象和为什么**
5. **能讲出安全模型的三道独立闸门**
6. **能把 KTClaw 和 OpenClaw 的关系讲清楚：什么是继承的，什么是超越的**

## 面试金句
1. **"OpenClaw 不是 framework，是 local-first 的个人 AI 操作系统。"**
2. **"Pi 负责让 LLM 能行动，OpenClaw 负责让 LLM 持久存在。这是 mechanism vs policy 的经典分层。"**
3. **"JSONL 树结构是 git 的数据模型映射到 LLM 会话——append-only + 零成本分支 + 不删除历史。这是我在架构层面学到的最有价值的东西。"**
4. **"OpenClaw 的不做清单不是懒，是刻意不做。'If I don't need it, it won't be built.'——这是最难的设计决策。"**
