# 上下文工程：业界最佳实践精华

> **核心观点**：上下文工程是 Agent 开发的本质，成功的 Agent 系统关键不在于调用什么模型，而在于如何设计系统来控制信息流向 LLM。

---

## 📖 目录

- [引言](#引言)
- [一、核心概念](#一核心概念)
  - [什么是上下文工程](#什么是上下文工程)
  - [上下文工程的组成部分](#上下文工程的组成部分)
  - [提示工程 vs 上下文工程](#提示工程-vs-上下文工程)
  - [为什么需要上下文工程](#为什么需要上下文工程)
- [二、长上下文问题与解决方案](#二长上下文问题与解决方案)
  - [Context-Rot 问题](#context-rot-问题)
  - [LangChain 的四类方法](#langchain-的四类方法)
- [三、业界工程实践](#三业界工程实践)
  - [Claude Code 的工程实践](#claude-code-的工程实践)
  - [Manus 的优化实践](#manus-的优化实践)
  - [Spec-Driven Development](#spec-driven-development)
- [四、未来展望](#四未来展望)
- [总结](#总结)

---

## 引言

随着 AI Agent 的快速发展，一个新的名词「**上下文工程**」进入大家的视野。很多人会好奇它与「提示词工程」有什么区别，是又在造新的概念吗？

本文将围绕三个核心主题展开：

1. **概念定义**：介绍上下文工程的基本概念和核心组成部分
2. **业界工程实践**：深入分析业界知名产品在上下文工程方面的具体实践
3. **未来展望**：探讨上下文工程后续可能的演进方向

### 核心问题

- 为什么需要上下文工程？
- 为什么 Claude Code 效果这么好？
- Manus 在优化 Agent 上做了哪些尝试？
- 为什么 Spec Driven + Context Engineering 会代替 Vibe Coding + Prompt Engineering？

---

## 一、核心概念

### 什么是上下文工程

**上下文工程是指构建动态系统，以合适的格式为大语言模型（LLM）提供正确的信息和工具，从而让 LLM 能够合理地完成任务。**

上下文不仅指你发送给 LLM 的单一提示词（prompt），更应该被视为模型在生成响应前所看到的一切信息。上下文工程就是如何将合适的信息填充到有限的上下文里的**艺术和科学**。

#### 核心特点

- ✅ **动态构建系统**
- ✅ **提供正确信息和工具**
- ✅ **合适的格式化**
- ✅ **让 LLM 合理完成任务**

![上下文工程概念图](https://mmbiz.qpic.cn/mmbiz_png/Z6bicxIx5naIYUbHzZuplVicOIQMq1erpbW8rWuY2KZnILQPQT4VpDeUNs1vu4SzSGvm6NFl7oHtMxiaaiaibCmG4Aw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=1)

<sub>图片来源于网络</sub>

---

### 上下文工程的组成部分

一个完整的上下文工程系统包含以下 **7 个核心组成部分**：

| 序号 | 组件 | 说明 |
|:---:|:---|:---|
| 1 | **指令/系统提示词** | 定义模型整体行为的初始指令，可以（也应该）包含示例、规则等 |
| 2 | **用户提示词** | 用户提出的即时任务或问题 |
| 3 | **短期记忆** | 用户和模型此前的对话内容，展现当前交流的背景 |
| 4 | **长期记忆** | 跨多次对话积累的持久性知识库，比如用户喜好、历史项目摘要、记住的特定事实 |
| 5 | **检索的信息（RAG）** | 外部的、最新的信息，包括从文档、数据库或 API 获取的相关内容 |
| 6 | **可用工具** | 模型可以调用的所有函数或内置工具定义（如检查库存、发送邮件等） |
| 7 | **结构化输出** | 明确定义模型输出的格式，例如 JSON 格式的对象 |

---

### 提示工程 vs 上下文工程

![提示工程与上下文工程对比](https://mmbiz.qpic.cn/mmbiz_png/Z6bicxIx5naJZu6aVojlqCkAIbPXxdIhFYEmag4Uq8yhkvWHKX7CdPAKKAicCdweL2lQtaethibqp7DMuibW38X7Hw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=2)

Context Engineering 代表着从传统 Prompt Engineering 到新范式的转变。

![范式转变](https://mmbiz.qpic.cn/mmbiz_jpg/Z6bicxIx5naIYUbHzZuplVicOIQMq1erpbHHZGHEPHtFjvcZBjEeicic2S8rLicqZTma81fIaf46o9UXEic0BrDVUQGA/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=3)

#### 对比表格

| 对比维度 | Prompt Engineering | Context Engineering |
|:---|:---|:---|
| **关注点** | 词句技巧和表达方式 | 提供全面上下文 |
| **作用范围** | 只限于任务描述的表达 | 包含文档、示例、规则、模式和验证 |
| **类比** | 就像贴一张便签 | 就像写一部详细剧本 |

---

### 为什么需要上下文工程

#### 简单演示对比

让我们通过一个简单的演示来理解上下文工程的价值：

**❌ 上下文贫乏的情况：**

> **用户**："Hey, just checking if you're around for a quick sync tomorrow. 嘿，想问一问明天方不方便，我们快速碰个头？"
>
> **AI**："Thank you for your message. Tomorrow works for me. May I ask what time you had in mind? 感谢来信！明天我有空。你想约在几点？"

**✅ 上下文丰富的"神奇"产品：**

> **上下文**：你的日历信息（显示你日程已满）、你与此人的过往邮件（用于确定合适的非正式语气）、你的联系人列表（用于识别 ta 为关键的合作伙伴）、send_invite 或 send_email 工具。
>
> **AI**："Hey Jim! Tomorrow's packed on my end, back-to-back all day. Thursday AM free if that works for you? Sent an invite, lmk if it works. 嗨 Jim！明天我这边日程全排满了，从早到晚连轴转。周四上午有空，你看行不？邀请已发，确认下是否合适~"

#### 上下文工程的价值

上下文工程能够带来以下重要价值：

1. **降低 AI 失败率**：大多数 Agent 失败不是模型问题，而是上下文不全
2. **保证一致性**：AI 能遵循你的项目模式和规范
3. **支持复杂特性**：有了完整上下文，AI 能处理多步骤实现
4. **自我修正**：验证循环让 AI 能自动修正错误

---

## 二、长上下文问题与解决方案

### Context-Rot 问题

随着上下文长度的增加，模型的注意力机制可能会出现"**腐蚀**"现象，导致对关键信息的关注度下降。

![Context-Rot 问题示意图](https://mmbiz.qpic.cn/mmbiz_png/Z6bicxIx5naIYUbHzZuplVicOIQMq1erpbaDfZiaOfEnyvux33dlc4900pUPXDcZOCPcib3UrYWFcAVKlKYvhUdduw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=4)

<sub>图片来源于网络</sub>

#### 问题表现

- ❌ 产生幻觉后，会被持续带偏
- ❌ 模糊性导致信息冲突，模型的行为会变得不可预测
- ❌ 关键信息被稀释，随着上下文的增长，模型的注意力会被分散
- ❌ 大量重复文本导致的"行动瘫痪"

#### 影响因素

- 上下文长度超过训练时的常见长度
- 模型能力的限制
- 信息密度不均匀分布
- 自然语言的模糊性

#### 解决方案

为了解决长上下文带来的问题，业界提出了系统性的上下文工程方法论：

- **Offload**：通过引用减少上下文长度
- **Retrieve**：RAG 技术动态检索相关信息
- **Reduce**：压缩裁剪冗余信息
- **Isolate**：分而治之，通过 SubAgent 处理子任务

---

### LangChain 的四类方法

在上下文工程领域，有三个产品代表了不同的实践方向：

1. **LangChain**：代表 Agent 框架和工具集合，早期的 Agent 框架，提供了各种 Agent 开发的基础设施，提出了一套上下文管理的方法论
2. **Claude Code**：代表 Code Agent 能力上限，编码 Agent 的能力标杆，在长短记忆、分层多 Agent 协作等方面有独到实践
3. **Manus**：重新展现 Agent 能力，让 Agent 回到大众视野，带动 MCP 发展，在工具使用、缓存设计等方面有独到实践

LangChain 提出了四类上下文管理的基本方法：

![LangChain 四类方法](https://mmbiz.qpic.cn/mmbiz_png/Z6bicxIx5naIYUbHzZuplVicOIQMq1erpbicI4b1ibkhjyorpfcw7LcMI07sMibazVZAbAibSTsuU3qAPCvEjYdfWia5g/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=5)

<sub>图片来源于网络</sub>

---

#### 1. 写入（Offload）上下文

![Offload 示意图 1](https://mmbiz.qpic.cn/mmbiz_png/Z6bicxIx5naIYUbHzZuplVicOIQMq1erpbOUbiaBMH7icEic06wREx5NNYD6nng2BKyRR5QBQHuB5HgO122feyMjictQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=6)

![Offload 示意图 2](https://mmbiz.qpic.cn/mmbiz_png/Z6bicxIx5naIYUbHzZuplVicOIQMq1erpb0ib4GYLcE1icRNPVWbWxiaDRUkQHCmQY9kDcLvy4SOtuyyGiaNOIQg7g7w/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=7)

<sub>图片来源于网络</sub>

**核心思想**：将信息保存在上下文窗口之外，以帮助 Agent 完成任务。不要将工具返回的全部原始信息都直接喂给 LLM。相反，应将其"卸载"到外部存储（如文件系统、数据库或一个专门的代理状态对象中），然后只将一个轻量级的"指针"返回给模型。

**核心组件**：
- File System - 文件系统
- Memories - 长期记忆系统（zep, mem0）
- DataBase - 数据库存储

**应用场景**：
- 长期记忆构建（Claude）
- 任务计划保存（Manus）
- 用户偏好记录
- 知识库管理

---

#### 2. 选择（Retrieve）上下文

**核心思想**：简单来说就是我们所熟悉的 RAG，通过检索和过滤相关信息，来控制进入 Context 的内容的数量和质量。

**核心组件**：
- **更高级的检索（Agentic Search）**：从向量检索出发，逐渐往更复杂的搜索体系演化。例如混合召回、结合图谱的 GraphRAG、Rerank 等等
- **返璞归真的文本检索**：仅仅使用 llms.txt + grep/find 之类的工具，通过 Agent 的多轮工具调用来获取相关信息。这也是 Claude Code 的实现方式

**应用场景**：
- 代码索引（DeepWiki）
- TodoList 召回
- 过多工具的召回（LangGraph-BigTool，Manus 不推荐，可能导致缓存失效）
- 知识库

---

#### 3. 压缩（Compress）上下文

通过各种手段来裁剪上下文的内容，只保留完成任务所需的 tokens。

![Compress 示意图](https://mmbiz.qpic.cn/mmbiz_png/Z6bicxIx5naJZu6aVojlqCkAIbPXxdIhFj2tN9eKqG3BibYwyaQ8n5mhEHfyarKOLMgNFtqYy0l3oicicd9hUicoGdw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=8)

<sub>图片来源于网络</sub>

**核心组件**：
- **摘要生成**：提取核心信息
- **Rerank**：移除不太相关的信息，RAG 场景中常用
- **语义总结、压缩**：保持含义精简表达。如果总结得不好，一样会出现关键信息丢失，甚至引入幻觉等问题

**应用场景**：
- 网络搜索
- RAG 系统
- 大量工具使用
- 多轮聊天

---

#### 4. 隔离（Isolate）上下文

非常类似 Workflow 时代的"分而治之"思想。如果一个任务的 Context 压力太过巨大，我们就拆分一下，分配给不同的 SubAgent 来完成各个子任务。这样每个 Agent 的 Context 内容都是独立的，会更加清晰和聚焦。

![Isolate 示意图](https://mmbiz.qpic.cn/mmbiz_png/Z6bicxIx5naIYUbHzZuplVicOIQMq1erpbGOfmXibgygA23myJtb8CgyZChqO9fmia3tgTBcFMBULLrFN4KpsYIRIg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=9)

<sub>图片来源于网络</sub>

**核心组件**：
- **环境隔离**：环境/沙盒隔离，让部分内容在 LLM 环境外运行，如代码执行场景，非常类似"卸载"
- **多 Agent 分离**：不同角色独立上下文，容易产生冲突的工作尤其要注意。"只读"类的工作最合适

**应用场景**：
- 智能体中涉及代码执行或数据分析
- 智能体工具调用
- 复杂的多智能体系统（如 Manus）

---

## 三、业界工程实践

### Claude Code 的工程实践

Claude Code 作为编码 Agent 的标杆，在上下文工程方面有很多独到的实践：

- ✅ **三层记忆架构**：实现从实时访问到持久化存储的完整覆盖
- ✅ **实时 Steering 机制**：流式输出提供持续交互反馈
- ✅ **分层多 Agent 协作**：主 Agent 协调 + SubAgent 执行的分层架构
- ✅ **动态上下文注入**：自动识别和注入相关文件内容

---

#### 1. 三层记忆架构

**挑战**：在长对话中，上下文管理面临 Token 限制导致信息丢失、传统压缩方法破坏上下文连续性、无法支持复杂多轮协作任务等挑战。

**解决方案**：Claude Code 构建了三层记忆系统：

- **短期记忆**（当前对话）
- **中期记忆**（智能压缩）
- **长期记忆**（CLAUDE.md 项目知识库）

实现从实时访问到持久化存储的完整覆盖。

**关键要点**：
- 92% 阈值自动触发智能压缩
- 8 段式结构化保存核心信息
- 跨会话恢复项目背景和用户偏好

![三层记忆架构](https://mmbiz.qpic.cn/mmbiz_png/Z6bicxIx5naIYUbHzZuplVicOIQMq1erpbkeUc6N2z2woPEo7nsJQVB60RIJvkPm1Vd0vETibH63eC1z9WicVickHNg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=10)

---

#### 2. 实时 Steering 机制

**挑战**：传统 Agent 无法中断，用户必须等待完整执行结束才能调整方向，导致资源浪费和用户体验差，无法应对动态变化的需求。

**解决方案**：Claude Code 采用异步消息队列 + 主循环的双引擎设计，支持实时中断和恢复，用户可以随时调整任务方向，系统自动保存状态并无缝切换。

**关键要点**：
- 异步消息队列支持实时中断
- 主循环自适应流程控制
- 流式输出提供持续交互反馈

![实时 Steering 机制](https://mmbiz.qpic.cn/mmbiz_png/Z6bicxIx5naIYUbHzZuplVicOIQMq1erpb5uM5uucQv57iaxc3tNlauvK4F2UibkzJp53AsAocIvMYyLY2U4vH4jCA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=11)

---

#### 3. 分层多 Agent 协作

**挑战**：复杂任务需要并发处理多个子任务，单 Agent 模式容易出现上下文污染、资源竞争和故障传播，影响整体执行效率和稳定性。

**解决方案**：Claude Code 采用主 Agent 负责任务协调，SubAgent 执行专项任务，实现隔离执行环境，调度器控制最多 10 个工具并发，确保任务隔离和资源优化。

**关键要点**：
- 主 Agent 协调 + SubAgent 执行的分层架构
- 独立执行环境避免上下文污染
- 智能调度器实现 10 工具并发控制

![分层多 Agent 协作](https://mmbiz.qpic.cn/mmbiz_png/Z6bicxIx5naIYUbHzZuplVicOIQMq1erpbMibQIJd6X3uPPnWlPBd6GqqFB7qBV4EGF8CxcoRQCsic7rabrsYOQD4g/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=12)

---

#### 4. 动态上下文注入

**挑战**：用户在对话中提及文件或概念时，系统无法自动关联相关信息，导致模型缺乏必要的上下文背景，影响响应质量和准确性。

**解决方案**：Claude Code 智能检测用户意图中的文件引用，自动读取相关内容并注入上下文，基于依赖关系推荐相关文件，提供语法高亮和格式化显示，最大 20 文件 8K Token 限制。

**关键要点**：
- 自动识别和注入相关文件内容
- 智能推荐基于依赖关系分析
- 容量控制和格式优化提升体验

![动态上下文注入](https://mmbiz.qpic.cn/mmbiz_png/Z6bicxIx5naIYUbHzZuplVicOIQMq1erpbsCqH4XEHYbHcKqmoRhDFRgA1WGBWquW5GXIttjiblEpB2ThSsq8v9cw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=13)

---

### Manus 的优化实践

Manus 在上下文工程方面有诸多独特的优化实践：

- ✅ **KV 缓存优化**：围绕 KV 缓存设计，大幅降低成本和延迟
- ✅ **工具遮蔽**：遮蔽而非移除工具，保持上下文稳定性
- ✅ **文件系统记忆**：使用文件系统作为终极上下文
- ✅ **注意力操控**：通过复述操控注意力，保持目标一致
- ✅ **错误保留**：保留错误内容，让模型从失败中学习
- ✅ **多样性增强**：避免少样本示例陷阱，增加结构化变化

---

#### 1. 围绕 KV 缓存进行设计

**背景**：随着每一步的推进，上下文不断增长，而输出保持相对简短。这使得 Agent 相比聊天机器人的预填充和解码比例高度倾斜。在 Manus 中，平均输入与输出的 token 比例约为 **100:1**。

**价值**：具有相同前缀的上下文可以利用 KV 缓存，大大减少首个 token 生成的时间和推理成本。使用 Claude Sonnet 时：
- 输入 token 缓存：**0.3 美元/百万 token**
- 未缓存：**3 美元/百万 token**
- **十倍成本差异！**

**关键要点**：
- 保持前缀稳定，时间戳会使 KV 缓存失效
- 使上下文只追加，确保 JSON 序列化是确定性的，键顺序不稳定会破坏缓存
- 在需要时明确标记缓存断点

![KV 缓存设计](https://mmbiz.qpic.cn/mmbiz_png/Z6bicxIx5naJZu6aVojlqCkAIbPXxdIhFLyC6sncrs9ibnCLD060CbiaeWrfflmQOjzPIAQxTU8JvldaIxicy7IvaA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=14)

---

#### 2. 遮蔽，而非移除工具

**挑战**：工具数量爆炸式增长，模型更可能选择错误的行动或采取低效的路径，但是要避免在迭代过程中动态添加或移除工具：
- 动态更改会导致 KV 缓存失效
- 模型会对不再定义的工具感到困惑

**解决方案**：Manus 使用上下文感知的状态机来管理工具可用性，在解码过程中掩蔽 token 的 logits，基于当前上下文阻止或强制选择某些工具。

**关键要点**：
- 在实践中，大多数模型提供商和推理框架都支持某种形式的响应预填充（Response Prefill），这允许你在不修改工具定义的情况下约束动作空间

![工具遮蔽](https://mmbiz.qpic.cn/mmbiz_png/Z6bicxIx5naJZu6aVojlqCkAIbPXxdIhFW9jQa2k1MCT21hb8oictdibWzZxCFOlZoZZnH1yiaGFx1ibHE17picazYYA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=15)

---

#### 3. 使用文件系统作为上下文

**痛点**：当前上下文窗口限制带来三个常见的痛点：
- 观察结果可能非常庞大，容易超出上下文限制
- 超过一定的上下文长度后，模型性能往往会下降
- 即使使用 KV 缓存，长输入成本依然高昂

**解决方案**：Manus 将文件系统视为**终极上下文**：大小不受限制，天然持久化，并且 Agent 可以直接操作。模型学会按需写入和读取文件——不仅将文件系统用作存储，还用作结构化的外部记忆。

**关键要点**：
- 只要保留 URL，网页内容就可以从上下文中移除
- 如果沙盒中仍然保留文档路径，则可以省略文档内容
- 这使得 Manus 能够缩短上下文长度，而不会永久丢失信息

![文件系统作为上下文](https://mmbiz.qpic.cn/mmbiz_png/Z6bicxIx5naJZu6aVojlqCkAIbPXxdIhFNMgyibbAxFyUBmZkfw7U5Hoia4Fzro2xVdA6erNRZcuaJbUgjk5L934A/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=16)

---

#### 4. 通过复述操控注意力

**挑战**：Manus 中的一个典型任务平均需要大约 **50 次工具调用**。这是一个很长的循环——由于 Manus 依赖 LLM 进行决策，它很容易偏离主题或忘记早期目标，尤其是在长上下文或复杂任务中。

**解决方案**：Manus 通过不断重写待办事项列表，将目标复述到上下文的末尾。这将全局计划推入模型的近期注意力范围内，避免了"丢失在中间"的问题。

**关键要点**：
- 避免"丢失在中间"问题
- 保持目标一致性
- 提升长任务执行能力
- 无需架构变更

![复述操控注意力](https://mmbiz.qpic.cn/mmbiz_png/Z6bicxIx5naJZu6aVojlqCkAIbPXxdIhFTd2bISe7YRbPdnmplhfIyHl9cgAichBUBnspQ1A6cFfpsGxTW4XXH8Q/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=17)

---

#### 5. 保留错误的内容

**问题**：在多步骤任务中，失败不是例外；它是循环的一部分。然而，一个常见的冲动是隐藏这些错误，这是有代价的：**擦除失败会移除证据。没有证据，模型就无法适应。**

**解决方案**：Manus 将错误的尝试保留在上下文中。当模型看到一个失败的行动——以及由此产生的观察结果或堆栈跟踪——它会隐式地更新其内部信念。这会改变其先验，降低重复相同错误的可能性。

**关键要点**：
- 模型从错误中学习
- 降低重复错误概率
- 提供负面样本训练
- 增强适应能力

![保留错误内容](https://mmbiz.qpic.cn/mmbiz_png/Z6bicxIx5naIYUbHzZuplVicOIQMq1erpborQvdaysqK8amZw0QAsIzpD9Q0h2pJyEZjX1uaiaCXZG9sWjjicicIoaA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=18)

---

#### 6. 不要被少样本示例所困

**问题**：Few-Shot 是提高 LLM 输出的常用技术。但在 Agent 系统中，它可能会以微妙的方式适得其反。LLM 倾向于模仿上下文中的行为模式，容易导致偏离、过度泛化，或产生幻觉。

**解决方法**：增加多样性。Manus 在行动和观察中引入少量的结构化变化——不同的序列化模板、替代性措辞、顺序或格式上的微小噪音。

**关键要点**：
- 不同的序列化模板
- 替代性措辞表达
- 顺序上的微小变化
- 格式上的受控噪音

![避免少样本陷阱](https://mmbiz.qpic.cn/mmbiz_png/Z6bicxIx5naIYUbHzZuplVicOIQMq1erpbxaXLQdqMvQlmcTcLMXnFOuzEb9KUdgUz1LazWq3xXGlNUxLPXdqeJg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=19)

---

### Spec-Driven Development

从提示词到规范驱动的开发范式转变。

#### Vibe Coding 的局限性

传统的 Vibe Coding 模式（Prompt → Code）存在明显局限性：

1. **不现实的期望**：指望用户写出高质量的提示词是不现实的
2. **技术债务**：快速迭代生成的代码缺乏充分的文档、单元测试或架构约束，易引入技术债务
3. **维护困难**：开发者可能不完全理解生成的代码，当需要调试、修改或扩展功能时面临巨大困难

---

#### Spec-Driven 的理念

**解决方案**：采用 Spec-Driven Development

- ❌ **Vibe Coding**：`Prompt → Code`
- ✅ **Spec-Driven Development**：`Prompt → Requirements → Design → Tasks → Code`

**关键优势**：
- 优先定义需求文档、系统设计和任务清单，确保逻辑清晰，与业务目标对齐
- 标准化利于针对性训练和调优大模型返回
- 标准化利于构建完善的上下文，包含数据、实体、交互等
- 对于大项目的维护和多人协作更有帮助

---

#### Kiro 的实现方式

在具体实现上，Kiro 项目采用了规范驱动的开发方式：

![Kiro 实现方式](https://mmbiz.qpic.cn/mmbiz_png/Z6bicxIx5naIYUbHzZuplVicOIQMq1erpb6s2ztG492JaAMIdM42d4MZuBicwpfU9ZyVb5ibBXUTno1hsG6ydgqVtQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=20)

**1. requirement.md**

定义了软件使用的 Story，这些 Story 定义遵循一种叫做 EARS (Easy Approach to Requirements Syntax) 的格式：

```
WHEN [condition/event] THE SYSTEM SHALL [expected behavior]
```

由 LLM 根据需求生成，可以二次确认和修改。

**2. design.md**

详细列举设计的各种技术细节：
- 架构设计与模块拆分
- 接口与流程
- 数据库表结构
- 前端实现

**3. tasks.md**

- 把开发过程分解成任务
- 每一个任务都被定义成一个 TODO List
- 可以点击一个按钮启动一个 Task 的开发过程
- 实时更新、回溯状态

---

## 四、未来展望

### 从上下文工程到环境工程

> LLM 现在就像在一间封闭的屋子中，我们通过发短信和它交流，未来它需要更完善的五感。
>
> **上下文工程仍是中间态，环境工程是终极目标。**

#### 三个阶段对比

| 阶段 | 主要内容 | 特点与局限性 |
|:---|:---|:---|
| **提示词工程** | 设计单条高质量 Prompt，指导模型输出 | 静态、一次性、依赖人类编写，缺乏动态适应性 |
| **上下文工程** | 动态收集、组织和注入多模态、多维度上下文信息 | 关注"模型输入"，提升智能体表现，但仍以模型为中心 |
| **环境工程** | 构建一个持续演化、可感知、可交互的智能环境 | 关注"模型所处的世界"，AI 不仅感知环境，还能主动塑造环境 |

---

### 为什么环境工程是终极目标？

1. **更广阔的范围**：环境不仅包含上下文，还包括动态变化的世界状态、规则、交互历史、反馈机制等

2. **主动性**：AI Agent 不再只是"被动"接受上下文，而是"主动"感知、探索、影响环境

3. **双向作用**：环境工程强调 AI 与环境的双向作用，支持持续学习、自适应、协作等更复杂的智能行为

4. **真实世界交互**：在环境工程中，AI 的输入输出不再局限于文本或结构化数据，而是包括真实世界的感知、动作和长期影响

---

## 总结

通过对 **Claude Code、Manus 和 Kiro** 等产品的分析，我们可以看到上下文工程在现代 AI 系统中的关键作用。它不仅解决了传统提示词工程的局限性，还为构建更智能、更可靠的 AI Agent 提供了系统化的方法论。

### 核心要点

✅ **LangChain 的四类方法**：Offload、Retrieve、Reduce、Isolate，构成了上下文管理的基础框架

✅ **Claude Code 的实践**：三层记忆架构、实时 Steering 机制、分层多 Agent 协作、动态上下文注入

✅ **Manus 的优化**：KV 缓存优化、工具遮蔽、文件系统记忆、注意力操控、错误保留

✅ **Kiro 的范式**：从 Vibe Coding 到 Spec-Driven Development，规范驱动的开发模式

### 未来方向

未来，随着**环境工程**概念的成熟，我们将看到 AI 系统从被动接受上下文走向主动感知和塑造环境，实现更高级别的智能交互。

---

## 📚 相关资源

- [上下文工程完全指南](./18-context-engineering-guide.md) - 7大核心组件详解
- [长上下文陷阱与修复技巧](./14-context-engineering.md) - 6大修复方法
- [Agent Memory 完整教程](./15-agent-memory.md) - Memory 系统设计
- [12-Factor Agent 架构](./12-factor-agent-architecture.md) - 工程化实践
- [Claude Code 最佳实践](./17-claude-code-best-practices.md) - 顶级产品经验

---

<div align="center">
<sub>本文总结了业界在上下文工程方面的最佳实践，希望对你构建 Agent 系统有所帮助。</sub>
</div>
