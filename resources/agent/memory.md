---
type: 资源清单
status: 已发布
level: 通用
topic:
  - Agent
  - 记忆
---

# Agent Memory 资源精选

> Agent 记忆模块相关的核心资源

---

## 🧠 Memory 开源工具对比

### 完整对比表

| 工具 | Stars | 开源/闭源 | 存储类型 | 特点 | 推荐度 | 使用难度 |
|:---|:---:|:---:|:---:|:---|:---:|:---:|
| **Mem0** | 20k+ | 开源+托管 | 图+向量 | 简单易用、自动提取 | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **MemGPT** | 10k+ | 开源+托管 | 图+向量 | 虚拟内存机制 | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Zep** | 5k+ | 开源+托管 | 图+向量 | 时间知识图谱 | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Cognee** | 2k+ | 开源+托管 | 图+向量 | 知识图谱增强 | ⭐⭐⭐ | ⭐⭐⭐ |
| **Memary** | 1k+ | 开源 | 图 | 轻量级方案 | ⭐⭐⭐ | ⭐⭐ |
| **LangChain Memory** | - | 开源 | 向量 | 框架内置 | ⭐⭐⭐⭐ | ⭐ |

**🔗 更多 Memory 工具**：查看 [完整列表](#更多-memory-工具)

---

## 📚 详细介绍

### 1. Mem0 ⭐⭐⭐⭐⭐
- **链接**：https://github.com/mem0ai/mem0
- **Stars**：20k+
- **简介**：轻量级 Agent 记忆模块
- **特点**：
  - ✅ 简单易用，10 行代码集成
  - ✅ 支持多种后端（Redis、Qdrant、PostgreSQL）
  - ✅ 自动提取和存储重要信息
- **适合场景**：
  - 对话 Agent（客服、助手）
  - 个性化推荐
  - 上下文持久化
- **⚠️ 注意**：社区反馈有稳定性问题，使用前需充分测试
- **学习成本**：⭐⭐（1天上手）

**快速示例**：
```python
from mem0 import Memory

memory = Memory()
memory.add("用户喜欢红色", user_id="user123")
result = memory.search("用户喜欢什么颜色?", user_id="user123")
```

---

### 2. MemGPT ⭐⭐⭐⭐
- **链接**：https://github.com/cpacker/MemGPT
- **Stars**：10k+
- **简介**：长期记忆管理系统
- **特点**：
  - ✅ 虚拟内存机制（类似操作系统）
  - ✅ 分层存储（工作记忆+长期记忆）
  - ✅ 自动内存管理
- **适合场景**：
  - 长期对话（跨会话记忆）
  - 复杂任务追踪
- **学习成本**：⭐⭐⭐（2-3天理解机制）

---

### 3. Zep ⭐⭐⭐⭐
- **链接**：https://github.com/getzep/zep
- **简介**：企业级长期记忆存储
- **特点**：
  - ✅ 生产级稳定性
  - ✅ 可扩展性强
  - ✅ 丰富的检索能力
- **适合场景**：
  - 企业级应用
  - 大规模用户
- **学习成本**：⭐⭐⭐

---

### 4. LangChain Memory (内置)
- **文档**：https://python.langchain.com/docs/modules/memory/
- **简介**：LangChain 内置记忆模块
- **类型**：
  - ConversationBufferMemory（缓冲记忆）
  - ConversationSummaryMemory（总结记忆）
  - ConversationKGMemory（知识图谱记忆）
- **适合场景**：
  - LangChain 项目
  - 快速集成
- **学习成本**：⭐（30分钟）

---

## 📚 Memory 核心论文

### 必读论文

1. **MemGPT论文**
   - 标题：MemGPT: Towards LLMs as Operating Systems
   - 核心思想：虚拟内存机制

2. **Retrieval-Augmented Memory**
   - 核心思想：检索增强的记忆系统

---

## 💡 Memory 设计要点

### 核心问题

**1. 什么时候存储？**
- ❓ 每句话都存 vs 只存重要信息？
- 💡 解决方案：重要性评分机制

**2. 如何存储？**
- ❓ 原文存储 vs 总结存储？
- 💡 解决方案：分层存储（短期+长期）

**3. 如何检索？**
- ❓ 向量检索 vs 关键词检索？
- 💡 解决方案：混合检索 + 时间衰减

**4. 何时遗忘？**
- ❓ 一直累积 vs 定期清理？
- 💡 解决方案：滑动窗口 + 重要性保留

---

## 🎯 面试高频问题

**Q1: 如何设计 Agent 的长期记忆机制？**

**标准答案**：
```
我会采用分层记忆架构：

1. 【工作记忆】（短期）
   - 当前对话上下文（最近 10 轮）
   - 存储：内存（快速访问）
   
2. 【情节记忆】（中期）
   - 本次会话的关键信息
   - 存储：向量数据库
   
3. 【语义记忆】（长期）
   - 用户偏好、历史重要信息
   - 存储：知识图谱 + 向量库

检索策略：
- 先查工作记忆（O(1)）
- 再查情节记忆（向量检索）
- 最后查语义记忆（混合检索）

遗忘机制：
- 时间衰减（越久越不重要）
- 重要性保留（关键信息不删除）
- 定期压缩（总结+去重）
```

---

**Q2: Mem0 和 MemGPT 有什么区别？**

| 维度 | Mem0 | MemGPT |
|:---|:---|:---|
| **核心思想** | 自动提取+向量存储 | 虚拟内存机制 |
| **易用性** | 简单（10行代码） | 复杂（需理解机制） |
| **适合场景** | 对话Agent、个性化 | 长期任务、复杂记忆 |
| **稳定性** | ⚠️ 有待验证 | ✅ 相对稳定 |

---

---

## 🗂️ 更多 Memory 工具

### 图+向量混合存储

| 工具 | URL | GitHub | 特点 |
|:---|:---|:---|:---|
| **Cognee** | https://www.cognee.ai/ | [GitHub](https://github.com/topoteretes/cognee) | 知识图谱增强 |
| **Memonto** | - | [GitHub](https://github.com/shihanwan/memonto) | 纯图存储 |
| **Memary** | https://finetune.dev/ | [GitHub](https://github.com/kingjulio8238/Memary) | 轻量级 |
| **GraphRAG** | https://microsoft.github.io/graphrag/ | [GitHub](https://github.com/microsoft/graphrag) | 微软出品 |

### 纯向量存储

| 工具 | URL | GitHub | 特点 |
|:---|:---|:---|:---|
| **BaseAI** | https://langbase.com/docs/memory | [GitHub](https://github.com/LangbaseInc/baseai) | Langbase 出品 |
| **BondAI** | https://bondai.dev/ | [GitHub](https://github.com/krohling/bondai) | 简单易用 |
| **Vanna.AI** | https://vanna.ai/ | [GitHub](https://github.com/vanna-ai/vanna) | SQL 场景 |

---

## 📚 Memory 核心论文（必读）

### 应用驱动派（推荐先看）

1. **MemGPT** ⭐⭐⭐⭐⭐
   - 论文：[MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560)
   - 核心思想：虚拟上下文管理
   - 适合：理解 Memory 系统设计

2. **Mem0** ⭐⭐⭐⭐⭐
   - 论文：[Mem0: Building Production-Ready AI Agents](https://arxiv.org/html/2504.19413v1)
   - 核心思想：图增强记忆框架
   - 适合：生产环境设计

3. **Zep - Graphiti** ⭐⭐⭐⭐
   - 论文：[Temporal Knowledge Graph](https://arxiv.org/pdf/2501.13956)
   - 核心思想：时序知识图谱
   - 适合：复杂关系记忆

4. **HippoRAG** ⭐⭐⭐⭐
   - 论文：[Neurobiologically Inspired LTM](https://arxiv.org/abs/2405.14831)
   - 核心思想：模拟海马体
   - 适合：长期记忆设计

### 模型驱动派（算法岗推荐）

5. **Memorizing Transformers** ⭐⭐⭐⭐
   - 论文：https://arxiv.org/abs/2203.08913
   - 核心思想：外部记忆 + KNN
   - 适合：模型底层改造

6. **MemoryLLM** ⭐⭐⭐⭐
   - 论文：https://arxiv.org/abs/2402.04624
   - 核心思想：可更新记忆
   - 适合：终身学习

7. **Memory³** ⭐⭐⭐⭐
   - 论文：https://arxiv.org/abs/2407.01178
   - 核心思想：分层记忆
   - 适合：记忆分类研究

8. **WISE** ⭐⭐⭐⭐
   - 论文：https://arxiv.org/abs/2405.14768
   - 核心思想：主记忆+侧记忆
   - 适合：模型编辑

9. **Titans** ⭐⭐⭐⭐
   - 论文：https://arxiv.org/abs/2501.00663
   - 核心思想：学习何时存储/遗忘
   - 适合：超长上下文

10. **MemAgent** ⭐⭐⭐⭐
    - 论文：https://arxiv.org/abs/2507.02259
    - 核心思想：强化学习记忆聚合
    - 适合：Agent RL 研究

### 综述论文

11. **Agent Memory Survey** ⭐⭐⭐⭐⭐
    - 论文：https://arxiv.org/pdf/2404.13501.pdf
    - GitHub：https://github.com/nuster1128/LLM_Agent_Memory_Survey
    - 核心：系统总结记忆机制

12. **Multimodal Memory Survey** ⭐⭐⭐⭐
    - GitHub：https://github.com/patrick-tssn/Awesome-Multimodal-Memory
    - 核心：400+ 多模态记忆论文

---

## 🎯 两大流派对比

| 维度 | 模型驱动（改模型） | 应用驱动（加外挂） |
|:---|:---|:---|
| **代表工作** | Memorizing Transformers、MemoryLLM | MemGPT、Mem0、Zep |
| **核心思路** | 改造模型内部结构 | 应用层记忆管理 |
| **优势** | 性能上限高、读取效率高 | 落地快、易扩展 |
| **劣势** | 研发成本高、周期长 | 依赖底层模型 |
| **适合** | 算法岗（做研究） | 开发岗（做应用） |

---

## 📝 相关文档

- [Agent Memory 论文精选](./papers/agent_memory/Agent%20Memory%20核心论文汇总.md) - 10篇必读论文详解
- [Agent 框架对比](./frameworks.md)
- [Tool Use 资源](../tools.md)
- [返回 Agent 资源总览](./README.md)

**🔗 完整论文列表**：查看 [Awesome-Awesome-LLM](https://github.com/adongwanai/Awesome-Awesome-LLM)


