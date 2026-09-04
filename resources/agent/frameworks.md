---
type: 资源清单
status: 已发布
level: 通用
topic:
  - Agent
  - 多智能体
  - 框架工具
---

# Agent 开发框架对比

> **只推荐最核心的 5-7 个框架**（面试会问、项目会用的）

---

## 🔥 核心框架对比表

| 框架             | Stars |  难度  | 适合场景        | 优点           | 缺点     |
| :------------- | :---: | :--: | :---------- | :----------- | :----- |
| **LangGraph**  | 15k+  | ⭐⭐⭐⭐ | 复杂工作流       | 灵活、可视化       | 学习曲线陡  |
| **AutoGen**    | 30k+  | ⭐⭐⭐  | Multi-Agent | 易用、可视化Studio | 定制化难   |
| **CrewAI**     | 20k+  |  ⭐⭐  | 角色协作        | 简单、快速上手      | 功能相对简单 |
| **LangChain**  | 90k+  | ⭐⭐⭐  | 快速原型        | 生态完善         | 抽象过度   |
| **Swarm**      |  5k+  |  ⭐   | 学习用途        | 极简、清晰        | 功能基础   |
| **Parlant**    |   -   | ⭐⭐⭐  | 指令遵循        | 高可控、ARQs     | 场景受限   |
| **AgentScope** |  3k+  | ⭐⭐⭐  | 模块化开发       | 易维护、企业级      | 相对新    |

---

## 📖 详细对比

### 1. LangGraph ⭐⭐⭐⭐⭐ 最推荐生产环境

**官网**：https://github.com/langchain-ai/langgraph

**核心特点**：
- ✅ 状态机驱动，精确控制流程
- ✅ 可视化工作流，易于调试
- ✅ 支持复杂条件分支和循环
- ✅ 与 LangChain 深度集成

**适合场景**：
- 复杂的 Agent 工作流（10+ 步骤）
- 需要状态管理的应用
- Multi-Agent 协作系统

**学习成本**：⭐⭐⭐⭐（需要 3-5 天深入学习）

**面试加分点**：
```
"我使用 LangGraph 设计了状态机驱动的 Agent 工作流，
通过可视化编排，实现了复杂的条件分支和异常处理"
```

**快速上手**：
```python
from langgraph.graph import StateGraph

# 30 行代码搭建 Agent 工作流
```

---

### 2. AutoGen ⭐⭐⭐⭐⭐ Multi-Agent 首选

**官网**：https://github.com/microsoft/autogen

**核心特点**：
- ✅ 天然支持 Multi-Agent
- ✅ AutoGen Studio 可视化编排
- ✅ 支持代码执行环境
- ✅ 微软官方支持

**适合场景**：
- 多智能体协作（3+ Agents）
- 需要可视化编排
- 企业级应用

**学习成本**：⭐⭐⭐（1-2 天快速上手）

**面试加分点**：
```
"使用 AutoGen 构建多智能体协作系统，
实现了分类Agent、查询Agent、执行Agent的协同工作"
```

---

### 3. CrewAI ⭐⭐⭐⭐ 快速原型首选

**官网**：https://github.com/joaomdmoura/crewAI

**核心特点**：
- ✅ 角色定义清晰
- ✅ API 简单易用
- ✅ 快速开发（1小时搭建原型）

**适合场景**：
- 快速验证想法
- 角色明确的协作场景
- 小规模应用

**学习成本**：⭐⭐（半天上手）

---

### 4. LangChain ⭐⭐⭐⭐⭐ 必学基础

**官网**：https://github.com/langchain-ai/langchain

**核心特点**：
- ✅ 生态最完善
- ✅ 文档最详细
- ✅ 社区最活跃

**适合场景**：
- 快速原型开发
- 学习 Agent 基础概念
- 与其他工具集成

**学习成本**：⭐⭐⭐（2-3 天基础掌握）

**注意事项**：
- ❌ 抽象过度，调试困难
- ⚠️ 建议：理解原理，必要时魔改

---

### 5. Swarm ⭐⭐⭐ 学习用途

**官网**：https://github.com/openai/swarm

**核心特点**：
- ✅ OpenAI 官方出品
- ✅ 代码极简（<500行）
- ✅ 适合理解 Agent 原理

**适合场景**：
- 学习 Agent 设计思想
- 理解多智能体协作
- 不建议用于生产

**学习成本**：⭐（1-2小时读完源码）

---

### 6. Parlant ⭐⭐⭐⭐ 指令遵循专家

**官网**：https://github.com/emcie-co/parlant

**核心特点**：
- ✅ **确保指令遵循**：不再依赖 LLM"可能"遵循指令，而是"确保"必定遵循
- ✅ **ARQs 技术**：Attentive Reasoning Queries（注意力推理查询）
- ✅ **领域指导优势**：通过预定义推理步骤，针对性预防常见失效模式
- ✅ **高成功率**：在 87 个测试场景中，总成功率达 90.2%（高于 CoT 的 86.05%）

**核心技术：ARQs（Attentive Reasoning Queries）**

ARQs 是一种结构化推理方法，包含三个阶段：

1. **引导 ARQ 阶段**
   - LLM 处理预定义引导问题
   - 恢复关键指令
   - 恢复提示中的重要上下文信息
   - 促进分步推理与中间计算

2. **响应生成阶段**
   - 基于引导 ARQ 阶段的推理结果
   - 生成最终响应

3. **响应验证阶段**（可选）
   - LLM 通过预定义验证问题评估响应
   - 不满足则重新生成并验证
   - 直至符合标准

**适合场景**：
- ✅ 有明确领域指南的场景（客户服务、医疗咨询）
- ✅ 失效模式清晰的场景
- ✅ 需要高度可控的 Agent 系统
- ✅ 高风险客户应用

**学习成本**：⭐⭐⭐（2-3 天理解核心概念）

**面试加分点**：
```
"使用 Parlant 框架确保 Agent 的指令遵循，
通过 ARQs 技术实现了领域专用推理步骤，
在客户服务场景中达到 90%+ 的成功率"
```

**关键资源**：
- [Parlant vs LangGraph](https://www.parlant.io/blog/parlant-vs-langgraph)
- [Parlant vs DSPy](https://www.parlant.io/blog/parlant-vs-dspy)
- [How Parlant Ensures Compliance](https://www.parlant.io/blog/how-parlant-guarantees-compliance)
- [ARQs 论文](https://arxiv.org/pdf/2503.03669v1)
- [ARQs 代码](https://github.com/emcie-co/parlant/tree/arqs-a-systematic-method-for-optimizing-instruction-following-in-llms)

**vs LangGraph**：
- Parlant 强调**确定性控制**，LangGraph 强调**灵活编排**
- Parlant 适合**规则明确**的场景，LangGraph 适合**复杂工作流**

**vs DSPy**：
- Parlant 通过**预定义推理步骤**确保遵循，DSPy 通过**优化提示**提升性能
- Parlant 更适合**生产环境**的可控性需求

---

### 7. AgentScope ⭐⭐⭐⭐ 模块化开发框架

**官网**：https://github.com/agentscope-ai/agentscope

**核心特点**：
- ✅ **模块化设计**：将 Agent 功能模块化，易于维护和扩展
- ✅ **灵活集成**：与各种外部工具和服务无缝集成
- ✅ **开发者友好**：详细的文档和示例
- ✅ **企业级支持**：阿里云等机构背书

**适合场景**：
- 模块化 Agent 系统开发
- 需要灵活集成的应用
- 企业级 Agent 应用
- 快速原型到生产的迁移

**学习成本**：⭐⭐⭐（2-3 天掌握基础）

**面试加分点**：
```
"使用 AgentScope 构建模块化 Agent 系统，
通过清晰的模块划分和灵活的集成能力，
实现了从原型到生产的快速迁移"
```

---

## 🎯 选择建议

### 按场景选择

| 你的需求 | 推荐框架 | 原因 |
|:---|:---|:---|
| **学习 Agent 原理** | Swarm | 代码简单，易于理解 |
| **快速做 Demo** | CrewAI | 上手快，API 简单 |
| **做复杂工作流** | LangGraph | 状态管理，灵活控制 |
| **做 Multi-Agent** | AutoGen | 天然支持，可视化 |
| **通用开发** | LangChain | 生态完善，工具多 |
| **高可控场景** | Parlant | 确保指令遵循，适合客服/医疗 |
| **模块化系统** | AgentScope | 易维护，企业级支持 |

### 学习顺序建议

```
Step 1: Swarm (理解原理，1天)
  ↓
Step 2: LangChain (掌握基础，3天)
  ↓
Step 3: LangGraph 或 AutoGen (深入一个，5天)
  ↓
Step 4: 其他框架按需学习
```

---

## 📚 官方学习指南（必读）

### 1. Anthropic 构建有效智能体指南 ⭐⭐⭐⭐⭐ 强烈推荐

**链接**：https://www.anthropic.com/engineering/building-effective-agents

**内容**：Anthropic 团队总结的 Agent 构建最佳实践

**核心观点**：
- Agent 不是万能的，要知道什么时候用、什么时候不用
- **Workflow > Agents**：能用工作流解决就不要用 Agent
- 简单的 Prompt 优化往往比复杂的 Agent 更有效

**必读原因**：Claude 团队的实战经验，帮助你避免过度设计

---

### 2. OpenAI 实用智能体构建指南 ⭐⭐⭐⭐

**链接**：https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf

**内容**：OpenAI 官方的 Agent 构建指南

**适合人群**：想要了解 OpenAI 视角下的 Agent 设计

---

### 3. Agentic Design Patterns（中英对照）⭐⭐⭐⭐⭐ 强烈推荐

**链接**：https://github.com/ginobefun/agentic-design-patterns-cn

**内容**：《Agentic Design Patterns: A Hands-On Guide to Building Intelligent Systems》中英文对照译本

**适合人群**：
- AI 工程师
- 软件架构师
- 产品经理
- 对智能系统设计感兴趣的研究者

**核心内容**：
- Agent 设计模式
- 实践流程
- 系统架构

---

### 4. Anthropic 提示工程指南 ⭐⭐⭐⭐

**链接**：https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview

**内容**：系统化的 Prompt Engineering 教程

**交互式教程**：https://github.com/anthropics/prompt-eng-interactive-tutorial

---

### 5. Google Gemini 提示指南 ⭐⭐⭐

**链接**：https://services.google.com/fh/files/misc/gemini-for-google-workspace-prompting-guide-101.pdf

**内容**：Google 官方的 Prompt Engineering 指南

---

## 🎯 学习建议

### 1. 先学框架原理，再学框架使用

- 理解 Agent 的本质（规划 + 工具 + 记忆）
- 手撕简单的 ReAct Agent
- 再去学习框架的封装

### 2. 不要过度依赖框架

- 框架是工具，不是答案
- 遇到问题要能调试源码
- 必要时可以魔改或自己实现

### 3. 多框架对比学习

- 同一个功能，用不同框架实现
- 理解每个框架的设计哲学
- 面试时能说出框架选择的理由

---

## 💬 面试中的框架选择问题

### 常见问题："为什么选择 XX 框架？"

**回答模板**：
```
我选择 LangGraph 是因为：
1. 【场景需求】项目需要复杂的状态管理和条件分支
2. 【技术优势】LangGraph 的可视化工作流便于调试
3. 【生态支持】与 LangChain 深度集成，工具链完善
4. 【对比分析】相比 AutoGen，LangGraph 在状态控制上更灵活
```

**关键点**：
- ✅ 从场景需求出发
- ✅ 说明技术优势
- ✅ 对比其他方案
- ✅ 体现深入思考

---

## 📝 相关文档

### 本目录其他资源

- [Agent 资源总览](./README.md)
- [Agent Memory 资源](./memory.md)
- [AI Agent 生产环境实践](../../docs/03-practice/06-agent-production-challenges.md)

### AgentGuide 其他章节


---

**👉 返回主文档**：[AgentGuide README](../../README.md)

---

## 📌 文档维护

**最后更新**：2025-11
**贡献者**：欢迎提交 PR 补充新框架

**更新日志**：
- 2025-11：新增 Parlant（指令遵循）和 AgentScope（模块化）框架
- 2025-11：整合官方学习指南
- 2025-11：优化框架对比表和选择建议





