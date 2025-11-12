# 技术栈：以上下文工程为核心的 Agent 开发

## 核心理念

> **Agent 开发的本质 = 上下文工程**
>
> 在深入研究业界主流 Agent 产品（Claude Code、Manus、Cursor 等）后，我们发现：**Agent 开发的核心不是调用 LLM API，而是如何设计系统来控制信息流向 LLM**。
>
> 这就是上下文工程（Context Engineering）的核心——在正确的时间，以正确的格式，向 LLM 提供正确的信息。

---

## 📚 目录结构

### 🎯 核心：上下文工程体系

上下文工程包含 7 个核心组成部分，我们的所有文档都围绕这个体系组织：

```
上下文工程 (Context Engineering)
├── 1. 指令/系统提示词 (System Prompt)
│   └── 18-context-engineering-guide.md - 提示技巧详解
│
├── 2. 用户提示词 (User Prompt)
│   └── [集成在各个实战文档中]
│
├── 3. 短期记忆 (Short-term Memory)
│   └── 15-agent-memory.md - Memory 系统完整指南
│
├── 4. 长期记忆 (Long-term Memory)
│   ├── 15-agent-memory.md - Memory 持久化
│   └── 12-factor-agent-architecture.md - 架构设计
│
├── 5. 检索信息 (RAG)
│   ├── 14-context-engineering.md - RAG 与上下文
│   └── 18-context-engineering-guide.md - 检索系统设计
│
├── 6. 工具调用 (Tools)
│   ├── 17-claude-code-best-practices.md - 工具使用最佳实践
│   └── 12-factor-agent-architecture.md - 工具编排
│
└── 7. 结构化输出 (Structured Output)
    └── 18-context-engineering-guide.md - 输出格式控制
```

---

## 📖 学习路径

### 🚀 快速入门路径（1-2天）

**目标**：理解上下文工程的核心概念

1. **[context-engineering-practices.md](./context-engineering-practices.md)** ⭐ **必读**
   - 600字快速了解上下文工程
   - 业界实践（Claude Code、Manus、Kiro）
   - 适合：快速建立全局认知

2. **[18-context-engineering-guide.md](./18-context-engineering-guide.md)** ⭐ **必读**
   - 完整的上下文工程指南
   - 6大核心组件详解
   - 适合：系统学习理论基础

3. **[14-context-engineering.md](./14-context-engineering.md)**
   - 长上下文的陷阱与修复
   - 6大修复技巧
   - 适合：解决实际问题

---

### 🎯 深度学习路径（1-2周）

**目标**：掌握上下文工程的各个组成部分

#### 第一步：理解 Agent 架构
- **[12-factor-agent-architecture.md](./12-factor-agent-architecture.md)** ⭐ 核心
  - 从简单循环到工程化架构
  - 12个关键因子
  - 适合：构建生产级 Agent

#### 第二步：掌握 Memory 系统
- **[15-agent-memory.md](./15-agent-memory.md)** ⭐ 核心
  - Memory 的原理与实战
  - 短期/长期记忆设计
  - 适合：实现有记忆的 Agent

#### 第三步：学习工具最佳实践
- **[17-claude-code-best-practices.md](./17-claude-code-best-practices.md)** ⭐ 核心
  - Claude Code 的工程实践
  - 工具设计、上下文注入
  - 适合：提升 Agent 能力上限

#### 第四步：模型微调（算法岗必备）
- **[16-sft-finetuning.md](./16-sft-finetuning.md)**
  - SFT 监督微调完全指南
  - Function Call 微调
  - 适合：算法工程师优化模型

#### 第五步：总结失败经验
- **[lessons-learned.md](./lessons-learned.md)**
  - 真实项目的坑与教训
  - 如何避免常见错误
  - 适合：少走弯路

---

### 💼 实战应用路径（2-4周）

**目标**：将上下文工程应用到实际项目

根据你的方向选择：

**🔬 算法岗方向**：
1. 上下文工程完全指南 → 理解理论基础
2. Agent Memory → 研究记忆压缩算法
3. SFT 微调 → 优化模型的 Context 处理能力
4. 失败经验 → 设计对比实验

**🛠️ 开发岗方向**：
1. 12-Factor Agent 架构 → 构建系统框架
2. Claude Code 最佳实践 → 学习工程化方法
3. Context Engineering 修复技巧 → 解决实际问题
4. Agent Memory → 实现持久化存储

**🎯 全栈方向**（推荐）：
- 按顺序完整学习所有文档
- 同时准备算法和工程项目
- 机会翻倍！

---

## 🎓 核心文档详解

### ⭐ 必读文档（5篇）

#### 1. [context-engineering-practices.md](./context-engineering-practices.md)
**一句话总结**：业界主流产品的上下文工程实践精华

**核心内容**：
- 什么是上下文工程？与提示词工程的区别
- LangChain 的 4 类方法：Offload、Retrieve、Reduce、Isolate
- Claude Code 的工程实践：三层记忆、实时 Steering、分层 Agent
- Manus 的优化技巧：KV 缓存、工具遮蔽、文件系统记忆
- Spec-Driven Development：从 Vibe Coding 到规范驱动

**适合场景**：
- ✅ 快速建立全局认知（600字精华）
- ✅ 面试前准备（了解业界实践）
- ✅ 技术分享、写作参考

**学习时间**：10-15 分钟

---

#### 2. [18-context-engineering-guide.md](./18-context-engineering-guide.md)
**一句话总结**：上下文工程的完整理论体系

**核心内容**：
- 上下文工程的 6 大核心组件
- Agents、Query Augmentation、Retrieval
- Prompting Techniques、Memory、Tools
- 每个组件的设计原则和实现方法

**适合场景**：
- ✅ 系统学习理论基础
- ✅ 设计 Agent 系统架构
- ✅ 准备算法岗面试

**学习时间**：1-2 小时

---

#### 3. [14-context-engineering.md](./14-context-engineering.md)
**一句话总结**：长上下文的 4 种失效模式与 6 大修复技巧

**核心内容**：
- 上下文失效的 4 种模式：中毒、干扰、混乱、冲突
- 6 大修复技巧：精简、缓存、总结、过滤、结构化、分层
- LangChain 的实践方案
- 实际案例分析

**适合场景**：
- ✅ 解决 Agent 失败率高的问题
- ✅ 优化长对话的性能
- ✅ 降低成本和延迟

**学习时间**：1 小时

---

#### 4. [12-factor-agent-architecture.md](./12-factor-agent-architecture.md)
**一句话总结**：从简单循环到生产级 Agent 的完整架构指南

**核心内容**：
- Agent 的本质：Reasoning + Acting 循环
- 12 个关键因子：记忆、规划、工具、反思等
- 工程化实践：错误处理、可观测性、安全性
- 架构设计模式

**适合场景**：
- ✅ 构建生产级 Agent 系统
- ✅ 面试系统设计题
- ✅ 理解 Agent 的完整生命周期

**学习时间**：2-3 小时

---

#### 5. [17-claude-code-best-practices.md](./17-claude-code-best-practices.md)
**一句话总结**：Claude Code 的工程实践与最佳实践

**核心内容**：
- Claude Code 的核心能力
- 工具设计最佳实践
- 上下文注入机制
- 提示词工程技巧
- 错误处理与重试

**适合场景**：
- ✅ 学习顶级产品的工程实践
- ✅ 提升 Agent 的能力上限
- ✅ 了解 Code Agent 的特殊性

**学习时间**：2-3 小时

---

### 📖 进阶文档（3篇）

#### 6. [15-agent-memory.md](./15-agent-memory.md)
**一句话总结**：Agent Memory 的完整教程

**核心内容**：
- Memory 的原理与分类
- 短期记忆 vs 长期记忆
- Memory 系统设计
- 主流 Memory 框架（Mem0、Zep）

**适合场景**：
- ✅ 实现有记忆的对话 Agent
- ✅ 个性化 Agent 系统
- ✅ 算法岗：Memory 压缩优化

**学习时间**：1-2 小时

---

#### 7. [16-sft-finetuning.md](./16-sft-finetuning.md)
**一句话总结**：SFT 监督微调完全指南

**核心内容**：
- SFT 的原理与流程
- Function Call 微调
- LlaMA-Factory 实战
- 数据准备与评估

**适合场景**：
- ✅ 算法岗必备技能
- ✅ 优化模型的 Context 处理能力
- ✅ 垂直领域模型定制

**学习时间**：2-4 小时

---

#### 8. [lessons-learned.md](./lessons-learned.md)
**一句话总结**：真实项目的坑与教训

**核心内容**：
- 常见的失败模式
- 如何避免踩坑
- 调试技巧
- 最佳实践总结

**适合场景**：
- ✅ 快速避坑
- ✅ 提升工程能力
- ✅ 面试准备（讲失败经验）

**学习时间**：30 分钟

---

## 🎯 不同角色的学习建议

### 🔬 算法工程师
**核心目标**：理论深度 + 算法创新

**必读文档**：
1. 18-context-engineering-guide.md - 理论基础
2. 14-context-engineering.md - 优化方向
3. 15-agent-memory.md - Memory 算法
4. 16-sft-finetuning.md - 模型微调
5. context-engineering-practices.md - 业界实践

**项目方向**：
- Agent Memory 压缩算法（存储 -60%）
- Agentic RAG 策略优化（准确率 +20%）
- Context 感知的 Reranker 训练

---

### 🛠️ 开发工程师
**核心目标**：系统设计 + 工程落地

**必读文档**：
1. 12-factor-agent-architecture.md - 架构设计
2. 17-claude-code-best-practices.md - 工程实践
3. 14-context-engineering.md - 问题修复
4. 15-agent-memory.md - Memory 实现
5. lessons-learned.md - 避坑指南

**项目方向**：
- 企业级 RAG 系统（服务 1000+ 用户）
- Multi-Agent 协作平台
- Agent 自动化 RPA 系统

---

### 🎯 全栈 Agent 工程师（推荐）
**核心目标**：算法 + 工程双修

**学习顺序**：
1. context-engineering-practices.md - 快速建立认知
2. 18-context-engineering-guide.md - 理论体系
3. 12-factor-agent-architecture.md - 架构设计
4. 14-context-engineering.md - 实战优化
5. 15-agent-memory.md - Memory 系统
6. 17-claude-code-best-practices.md - 最佳实践
7. 16-sft-finetuning.md - 模型微调
8. lessons-learned.md - 经验总结

**项目策略**：
- 同一个项目，准备算法版和开发版
- 简历上既有创新点，又有业务价值
- 面试机会翻倍！

---

## 💡 学习建议

### ✅ 推荐做法

1. **先快速过一遍"context-engineering-practices.md"**
   - 10 分钟建立全局认知
   - 了解业界在做什么

2. **深入学习理论体系**
   - 18-context-engineering-guide.md
   - 理解 6 大核心组件

3. **动手实践**
   - 选择一个实战项目
   - 边做边学其他文档

4. **总结复盘**
   - 记录踩过的坑
   - 整理面试话术

### ❌ 不推荐做法

1. ❌ 不要死记硬背理论
   - 理论要结合实践理解

2. ❌ 不要跳过基础直接做项目
   - 容易陷入调包侠陷阱

3. ❌ 不要只学一个方向
   - 算法岗也要懂工程
   - 开发岗也要懂原理

4. ❌ 不要忽视失败经验
   - 别人踩过的坑要记住

---

## 🔗 相关资源

### ⭐ 上下文工程资源合集（必看！）

**[📖 全网最全最优质的上下文工程资源合集](./context-engineering-resources.md)** 🔥

涵盖 18+ 篇核心资源：
- ✅ **核心概念**：Philipp Schmid、上下文工程 2.0
- ✅ **实战方法**：LangChain 四大策略、12-Factor Agents、phodal 实战
- ✅ **避坑指南**：Drew Breunig 失败模式、实战经验
- ✅ **企业案例**：Manus、Anthropic、CAMEL-AI
- ✅ **学术综述**：1400+ 论文分析、前沿研究
- ✅ **开源工具**：4.1K+ stars 仓库、完整代码
- ✅ **学习路径**：入门（1-2周）→ 进阶（2-4周）→ 专家（1-3月）

### 📚 理论学习
- [LangChain 官方文档](https://python.langchain.com/docs/get_started/introduction)
- [Anthropic Context Engineering Guide](https://docs.anthropic.com/claude/docs/guide-to-anthropics-prompt-engineering-resources)

### 🛠️ 实战框架
- [LangChain](https://github.com/langchain-ai/langchain)
- [LlamaIndex](https://github.com/run-llama/llama_index)
- [AutoGen](https://github.com/microsoft/autogen)
- [CrewAI](https://github.com/joaomdmoura/crewAI)

### 📖 论文阅读
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)
- [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601)

---

## 🎯 总结

**上下文工程是 Agent 开发的核心**，理解了上下文工程，就理解了：
- ✅ 为什么 Claude Code 效果这么好？（三层记忆、实时 Steering）
- ✅ 为什么 Manus 成本这么低？（KV 缓存优化、工具遮蔽）
- ✅ 为什么你的 Agent 总是失败？（上下文中毒、干扰、混乱）
- ✅ 如何构建生产级 Agent？（12-Factor 架构）

**从现在开始，用上下文工程的视角重新理解 Agent 开发！**

---

## 📬 反馈与贡献

如果你有任何问题或建议，欢迎：
- 提交 Issue
- 贡献 PR
- 加入学习社群讨论

**让我们一起把 AgentGuide 做得更好！** 🚀
