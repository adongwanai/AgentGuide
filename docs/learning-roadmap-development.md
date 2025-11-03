# AI Agent 开发工程师学习路线图

> **目标岗位**：AI Agent 开发工程师（应用型、工程型）  
> **学习时长**：4-6 周（每天 2-4 小时）  
> **最终产出**：2-3 个系统落地型项目 + 完整的技术栈掌握

---

## 🎯 开发岗核心要求

### 你需要具备的能力

<table>
<tr>
<td width="33%">

**系统设计**
- 高可用架构
- 性能优化
- 监控告警
- 异常处理

</td>
<td width="33%">

**工程实现**
- 框架熟练使用
- 代码质量高
- 快速开发能力
- 调试能力强

</td>
<td width="33%">

**业务理解**
- 用户需求分析
- 场景适配能力
- 数据驱动优化
- ROI 评估

</td>
</tr>
</table>

### 开发岗简历必备

✅ **至少2个完整系统项目**：端到端可运行  
✅ **业务指标提升**：QPS+50%、成本-40% 等数据  
✅ **技术栈丰富**：LangChain + 向量DB + 缓存 + 监控  
✅ **生产化经验**：部署、监控、异常处理

---

## 📅 6周详细学习计划

### 🗓️ Week 1：快速上手 LangChain + RAG

**学习目标**：
- 快速掌握 LangChain 框架
- 理解 RAG 的基本流程
- 能够搭建一个基础 RAG 系统

#### Day 1-2：LangChain 快速入门

**学习内容**：
- [ ] LangChain 核心概念：LLM、Prompt、Chain
- [ ] LangChain 基础组件：Document Loaders、Text Splitters、Vector Stores
- [ ] LCEL（LangChain Expression Language）

**实战任务**：
```python
# 任务1：10分钟搭建第一个 LLM 应用
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# 实现一个简单的问答系统

# 任务2：30分钟搭建基础 RAG
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# 实现文档检索+生成
```

**学习资源**：
- LangChain 官方文档：https://python.langchain.com/
- LangChain 中文教程：https://github.com/liaokongVFX/LangChain-Chinese-Getting-Started-Guide
- AgentGuide：[LangChain核心概念](./02-tech-stack/04-langchain-guide.md)

---

#### Day 3-4：向量数据库实战

**学习内容**：
- [ ] Embedding 原理：文本 → 向量
- [ ] 向量数据库选型：Milvus vs Chroma vs FAISS
- [ ] 向量检索：相似度计算、Top-K

**实战任务**：
```python
# 任务1：本地 FAISS 快速实现
import faiss
import numpy as np

# 实现向量索引和检索

# 任务2：Milvus 部署与使用
# Docker 部署 Milvus
# 实现插入、检索、删除操作
```

**学习资源**：
- Milvus 官方文档：https://milvus.io/docs
- FAISS 教程：https://github.com/facebookresearch/faiss/wiki

---

#### Day 5-7：完整 RAG 系统搭建

**学习内容**：
- [ ] 文档解析：PDF、Word、Markdown
- [ ] 文本分块策略
- [ ] 检索优化：Hybrid Search、Rerank
- [ ] 生成优化：Prompt Engineering

**实战任务**：
```python
# 项目：智能文档问答系统
# 功能：
# 1. 上传文档（PDF/Word）
# 2. 自动解析和索引
# 3. 问答接口
# 4. 历史对话管理

# 技术栈：
# - 文档解析：PyPDF2 / Unstructured
# - 向量化：Sentence-Transformers
# - 向量库：Chroma（本地）
# - LLM：OpenAI API
# - 后端：FastAPI
```

**产出目标**：
- ✅ 一个可运行的 RAG Demo
- ✅ 理解 RAG 的完整流程
- ✅ 能在 30 分钟内从零搭建

---

### 🗓️ Week 2：Agent 开发实战

**学习目标**：
- 理解 Agent 的工作机制
- 掌握 Tool Calling 和 Function Calling
- 能够开发一个实用的 Agent 应用

#### Day 8-10：Agent 核心概念

**学习内容**：
- [ ] Agent 的核心组件：Planning、Tools、Memory
- [ ] ReAct 框架理解（概念层面，不需要推导）
- [ ] Tool Calling 机制

**实战任务**：
```python
# 任务1：手撕简单 ReAct Agent（30行代码）
def react_loop(query):
    thought = llm.think(query)
    action = llm.decide_action(thought)
    observation = execute_tool(action)
    answer = llm.generate(thought, observation)
    return answer

# 任务2：LangChain Agent 实现
from langchain.agents import create_react_agent
from langchain.tools import Tool

# 创建自定义工具
# 实现 Agent 循环
```

**学习资源**：
- LangChain Agent 文档：https://python.langchain.com/docs/modules/agents/
- ReAct 原理解读（不需要看论文）

---

#### Day 11-14：Tool Use & Function Calling

**学习内容**：
- [ ] Function Calling 原理
- [ ] 如何定义工具（Tool Schema）
- [ ] 工具调用的错误处理
- [ ] 多工具编排

**实战任务**：
```python
# 任务1：实现 3 个自定义工具
# Tool 1：天气查询
# Tool 2：数据库查询
# Tool 3：API 调用

# 任务2：工具链式调用
# Agent 根据用户问题，自主选择和调用工具
# 处理工具调用失败的情况

# 任务3：Function Calling 实战
# 使用 OpenAI Function Calling
# 实现结构化输出
```

**产出目标**：
- ✅ 理解 Tool Use 的完整流程
- ✅ 能开发自定义工具
- ✅ 能处理工具调用异常

---

### 🗓️ Week 3：系统优化与生产化

**学习目标**：
- 掌握系统性能优化方法
- 学习生产环境部署
- 建立监控和告警体系

#### Day 15-17：性能优化

**学习内容**：
- [ ] 缓存策略：Redis 缓存 LLM 响应
- [ ] 批处理优化：Batch Processing
- [ ] 异步处理：async/await
- [ ] 并发控制：线程池、协程

**实战任务**：
```python
# 优化 RAG 系统性能
# 目标：响应时间从 2s → 300ms

# 优化点1：Redis 缓存
# - 缓存热点查询的检索结果
# - 缓存 LLM 生成结果

# 优化点2：批处理
# - Embedding 批量生成
# - Reranker 批量推理

# 优化点3：异步处理
# - 多个工具并行调用
# - 非关键路径异步执行
```

**性能目标**：
- P99 延迟 < 500ms
- QPS > 100
- 缓存命中率 > 70%

---

#### Day 18-21：监控与可观测性

**学习内容**：
- [ ] LangSmith：Agent 链路追踪
- [ ] Prometheus：指标监控
- [ ] Grafana：可视化
- [ ] 日志系统：ELK Stack

**实战任务**：
```python
# 任务1：集成 LangSmith
from langsmith import traceable

@traceable
def agent_call(query):
    # 追踪每一步 Agent 执行
    pass

# 任务2：Prometheus 指标暴露
from prometheus_client import Counter, Histogram

# 监控：请求数、延迟、错误率
```

**监控指标**：
- 请求量（QPS）
- 响应延迟（P50、P99）
- 错误率
- LLM Token 消耗
- 缓存命中率

---

### 🗓️ Week 4：Multi-Agent 系统开发

**学习目标**：
- 掌握 Multi-Agent 框架（AutoGen/CrewAI）
- 实现多智能体协作
- 设计复杂工作流

#### Day 22-24：AutoGen 框架

**学习内容**：
- [ ] AutoGen 核心概念：ConversableAgent、GroupChat
- [ ] Agent 角色定义
- [ ] 通信机制

**实战任务**：
```python
# 任务：客服 Multi-Agent 系统
# Agent 1：分类Agent（判断问题类型）
# Agent 2：查询Agent（查询知识库）
# Agent 3：回复Agent（生成回答）

from autogen import ConversableAgent, GroupChat

# 实现 3 个 Agent 的协作
# Supervisor 模式编排
```

---

#### Day 25-28：CrewAI 框架

**学习内容**：
- [ ] CrewAI 的角色（Role）和任务（Task）
- [ ] 工作流设计
- [ ] Agent 间通信

**实战任务**：
```python
# 项目：内容创作 Multi-Agent
# Agent 1：研究员（搜集资料）
# Agent 2：撰写者（写初稿）
# Agent 3：编辑者（审核修改）

from crewai import Agent, Task, Crew

# 实现完整的内容创作流水线
```

**产出目标**：
- ✅ 完成一个 Multi-Agent 应用
- ✅ 理解不同框架的设计理念
- ✅ 能根据场景选择合适框架

---

### 🗓️ Week 5-6：系统落地型项目实战

**核心目标**：完成 2 个可写进简历的完整系统

#### 项目选择建议

<table>
<tr>
<td width="50%">

**项目方向1：企业级 RAG 系统**

**业务场景**：
- 企业知识库问答
- 客服智能助手
- 文档智能检索

**技术要求**：
- 文档解析（多格式支持）
- 混合检索（BM25 + 向量）
- Reranker 重排序
- 缓存优化
- API 服务化

**系统指标**：
- 准确率 > 85%
- P99延迟 < 500ms
- QPS > 100
- 用户满意度 > 90%

</td>
<td width="50%">

**项目方向2：Agent 自动化系统**

**业务场景**：
- RPA 自动化
- 数据分析 Agent
- 研究助手

**技术要求**：
- 多工具集成（20+）
- Multi-Agent 协作
- 异常处理机制
- 任务调度

**系统指标**：
- 自动化率 > 80%
- 成功率 > 95%
- 效率提升 > 3倍
- 成本节省 > 50%

</td>
</tr>
</table>

#### Day 29-35：项目一开发

**Day 29-30：需求分析与系统设计**
- [ ] 明确业务场景和用户需求
- [ ] 设计系统架构图
- [ ] 技术选型（框架、数据库、部署方案）
- [ ] API 接口设计

**Day 31-33：核心功能开发**
- [ ] 文档解析模块
- [ ] 向量索引模块
- [ ] 检索模块（BM25 + 向量 + Reranker）
- [ ] 生成模块（LLM 调用 + Prompt 优化）

**Day 34-35：优化与部署**
- [ ] 性能优化（缓存、批处理）
- [ ] 监控集成（LangSmith）
- [ ] 异常处理（重试、降级）
- [ ] Docker 部署

---

#### Day 36-42：项目二开发

**按照类似流程开发第二个项目**

**重点关注**：
- [ ] 业务价值的体现（省钱、提效）
- [ ] 系统稳定性（异常处理）
- [ ] 可扩展性（QPS 增长10倍怎么办？）

---

### 🗓️ Week 7：面试准备（可选，如果时间紧）

#### Day 43-45：简历撰写（开发岗版本）

**项目描述模板**：
```markdown
【项目名称】（系统落地型）

- **背景**：XX业务场景，XX痛点（用户量、处理量）
- **技术**：XX框架 + XX数据库 + XX工具
  - 核心模块：XX、XX、XX
  - 技术亮点：XX、XX
- **优化**：XX指标从XX→XX
  - 性能优化：XX
  - 成本优化：XX
- **成果**：服务XX用户，日均XX请求
  - 准确率XX%
  - 用户满意度XX%
  - （如果有）节省成本XX万/年
- **技能**：系统设计、性能优化、XX框架
```

**关键点**：
- ✅ 强调业务价值
- ✅ 强调系统指标
- ✅ 强调技术栈丰富
- ❌ 不要过分强调算法创新

---

#### Day 46-49：系统设计题准备

**高频系统设计题**：

**题目1：设计一个高并发的 RAG 系统**

**回答框架**：
```
1. 【需求clarify】
   - QPS 多少？（假设 1000）
   - 延迟要求？（P99 < 500ms）
   - 数据规模？（10万文档）

2. 【系统架构】
   ┌─────────┐
   │ Load    │
   │Balancer │
   └────┬────┘
        │
   ┌────┴─────┐
   │  API     │ (多实例)
   │ Gateway  │
   └────┬─────┘
        │
   ┌────┴──────┬────────┐
   │           │        │
┌──┴──┐   ┌───┴──┐  ┌──┴──┐
│Redis│   │Vector│  │ LLM │
│Cache│   │  DB  │  │ API │
└─────┘   └──────┘  └─────┘

3. 【关键技术点】
   - 缓存：Redis 缓存热点查询（命中率70%）
   - 批处理：Embedding 批量生成
   - 异步：非关键路径异步执行
   - 监控：Prometheus + Grafana

4. 【扩展性】
   - 横向扩展：增加 API 实例
   - 向量库分片：Milvus 集群
   - 缓存分布式：Redis Cluster
```

**题目2：设计一个 Agent 自动化系统**

**回答框架**（同样强调架构、性能、监控）

---

## 🏗️ 开发岗项目Portfolio建议

### 理想的简历项目组合

```
项目1：RAG 系统（必须！）
  → 企业知识库问答系统
  → 智能客服系统
  → 关键词：完整系统、业务指标、上线

项目2：Agent 应用（必须！）
  → RPA 自动化系统
  → 数据分析 Agent
  → 关键词：多工具、自动化、效率提升

项目3：微调/部署（加分项）
  → Function Call 微调
  → vLLM 推理部署
  → 关键词：训练、部署、优化
```

---

## 🛠️ 开发岗必备技术栈

### 后端开发
- **Python**：必须精通
- **FastAPI** ⭐：RESTful API 开发
- **异步编程**：async/await、协程
- **数据库**：PostgreSQL、MongoDB

### Agent 开发
- **LangChain** ⭐⭐⭐：必须熟练
- **LlamaIndex**：RAG 开发
- **AutoGen/CrewAI**：Multi-Agent

### 向量数据库
- **Milvus** ⭐：分布式向量库
- **Chroma**：轻量级向量库
- **FAISS**：本地向量检索

### 部署运维
- **Docker** ⭐：容器化部署
- **Redis**：缓存
- **Nginx**：反向代理
- **Prometheus + Grafana**：监控

### 文档解析
- **MinerU** ⭐：PDF 解析
- **Unstructured**：多格式解析
- **PaddleOCR**：OCR 识别

---

## 🎤 开发岗面试话术

### 项目介绍（3-5分钟）

```
【开场】
我做的是XX系统，核心目标是解决XX业务痛点。

【背景】
在XX业务场景下，存在XX问题：
- 用户每天需要处理XX量级的XX
- 原有方案效率低，准确率仅XX%
- 人力成本高达XX万/年

【技术方案】
我设计并实现了一套XX系统，架构如下：
（简要描述系统架构，3-5个核心模块）

核心技术栈：
- 后端：FastAPI + Redis + PostgreSQL
- AI：LangChain + Milvus + OpenAI
- 部署：Docker + Nginx
- 监控：LangSmith + Prometheus

【技术亮点】
1. 混合检索策略：BM25 + 向量 + Reranker，准确率提升15%
2. 缓存优化：Redis 缓存热点查询，命中率70%，延迟降低60%
3. 批处理优化：Embedding 批量生成，吞吐量提升5倍
4. 可观测性：LangSmith 追踪每一步，问题定位效率提升5倍

【业务成果】
- 服务XX用户，日均XX请求
- 准确率从XX%提升至XX%
- 响应时间从XXs降至XXms
- 节省人力成本XX万/年
- 用户满意度XX%

【技术成长】
通过这个项目，我深入理解了XXX，掌握了XXX技术。
```

### 深度追问应对

**Q: 系统遇到过什么性能瓶颈？怎么解决的？**
```
A: 遇到过XX瓶颈，具体表现是：
1. 【问题定位】通过火焰图分析，发现瓶颈在XX
2. 【数据验证】XX操作耗时XXms，占总时间的XX%
3. 【解决方案】
   - 方案1：XX优化，效果提升XX%
   - 方案2：XX优化，效果提升XX%
   - 最终选择方案X，因为XXX
4. 【效果验证】延迟从XXms→XXms，满足业务要求
5. 【监控告警】设置XX指标告警，避免再次发生
```

**Q: 如果QPS增长10倍，系统怎么扩展？**
```
A: 我会从以下几个方面考虑：
1. 【API层】水平扩展，增加实例数（目前单机，可扩展到10+实例）
2. 【缓存层】Redis Cluster，分布式缓存
3. 【向量库】Milvus 集群模式，数据分片
4. 【LLM层】批处理优化，或部署本地模型
5. 【监控】实时监控QPS、延迟，自动扩缩容
6. 【成本】预估XX QPS下，成本约XX元/天
```

**Q: 线上出现过什么问题？怎么排查的？**
```
A: 出现过XX问题，表现是：
1. 【现象】用户反馈XX，监控显示XX指标异常
2. 【定位】查看日志，发现是XX模块报错
3. 【原因】根因是XXX（具体技术细节）
4. 【修复】临时方案：XX，永久方案：XX
5. 【防范】增加XX监控，添加XX测试用例
6. 【复盘】写了故障复盘文档，团队分享经验
```

---

## 💼 开发岗简历示例

### 示例1：RAG 系统

```markdown
【企业级智能知识库问答系统】

- **背景**：公司10万+内部文档需智能检索，原方案准确率60%，响应慢
- **技术**：LangChain + Milvus + Neo4j + FastAPI
  - 混合检索：BM25（文本匹配）+ 向量检索（语义）+ GraphRAG（关系）
  - 三路召回融合，Reranker 重排序
  - Redis 缓存热点查询
- **优化**：
  - 准确率从60%→85%（A/B测试验证）
  - P99延迟从2s→300ms（Redis命中率70%）
  - QPS从50→200（批处理 + 异步优化）
- **成果**：服务1000+员工，日均2000+查询，用户满意度90%
- **技能**：系统设计、性能优化、向量检索、分布式缓存、LangChain
```

### 示例2：Agent 自动化系统

```markdown
【Agent 驱动的客服 RPA 系统】

- **背景**：客服部门日均5000+重复工单，人力成本200万/年
- **技术**：LangChain + AutoGen + Mem0 + WebShaper
  - 多Agent协同：分类Agent、查询Agent、执行Agent
  - 集成20+工具（数据库、API、浏览器操作、邮件发送）
  - Mem0 记忆模块，支持上下文对话
- **优化**：
  - 异常重试机制，成功率从70%→95%
  - 并发处理，吞吐量提升5倍
  - LangSmith 链路追踪，问题定位时间从2h→20min
- **成果**：自动化率80%，效率提升3倍，节省成本150万/年
  - 获部门最佳项目奖
- **技能**：Agent开发、工具集成、异常处理、系统监控、AutoGen
```

---

## 📊 系统设计面试题库

### 高频题型

**1. 设计一个高并发的 RAG 系统**
- 需求：QPS 1000+，P99 < 500ms
- 考察：架构设计、性能优化、扩展性

**2. 设计一个 Agent 自动化系统**
- 需求：多工具调用、异常处理、监控
- 考察：工作流编排、容错设计、可观测性

**3. 设计一个 Multi-Agent 协作系统**
- 需求：3+ Agent 协作，任务分解
- 考察：通信机制、任务调度、一致性

**4. 如何优化 LLM 调用成本？**
- 考察：缓存策略、小模型替代、Prompt 优化

**5. 如何保证 Agent 系统的稳定性？**
- 考察：异常处理、监控告警、降级策略

详见：[系统设计面试题总结](./04-interview/02-system-design-questions.md)

---

## 🔧 开发岗避坑指南

### ❌ 常见错误

**1. 过度依赖框架**
- 问题：LangChain 封装太多，出问题不会调试
- 解决：理解框架原理，看源码，必要时魔改

**2. 忽视生产化**
- 问题：只关注功能，不考虑性能、成本、监控
- 解决：从一开始就考虑缓存、异步、监控

**3. 缺少评估**
- 问题：Demo 效果好，但没有量化指标
- 解决：建立评估体系，A/B 测试，数据驱动

**4. 业务理解浅**
- 问题：只会实现功能，不理解为什么
- 解决：多思考业务场景，多问为什么

---

## 📚 开发岗学习资源

### 🎥 推荐课程
- **LangChain 官方教程**：https://python.langchain.com/docs/get_started/
- **FastAPI 教程**：https://fastapi.tiangolo.com/zh/
- **RAG from Scratch**：https://github.com/langchain-ai/rag-from-scratch

### 📖 推荐博客
- LangChain Blog
- Pinecone Blog
- AgentGuide 资源：[优质博客](../resources/blogs.md)

### 🔧 推荐工具
- **DeepWiki**：理解开源项目源码
- **Cursor**：AI 辅助编程
- **LangSmith**：Agent 调试

---

## 🎯 开发岗职业发展路径

```
初级开发工程师（应届-1年）
  ↓
  核心能力：
  - 熟练使用 LangChain/LlamaIndex
  - 能搭建基础 RAG/Agent 系统
  - 能解决常见技术问题
  ↓
中级开发工程师（1-3年）
  ↓
  核心能力：
  - 能设计复杂系统架构
  - 能优化系统性能（10x提升）
  - 能主导项目开发
  ↓
高级开发工程师（3-5年）
  ↓
  核心能力：
  - 能设计平台级系统
  - 能解决疑难技术问题
  - 能指导初级工程师
  ↓
技术专家/架构师（5年+）
  ↓
  核心能力：
  - 定义技术架构标准
  - 解决跨团队技术难题
  - 技术战略规划
```

---

## ✅ 学习检查清单

在完成学习后，检查自己是否达标：

### 技术能力
- [ ] 能在 30 分钟内从零搭建一个 RAG 系统
- [ ] 能熟练使用 LangChain/LlamaIndex
- [ ] 能集成至少 5 种工具到 Agent
- [ ] 能部署一个完整的 API 服务
- [ ] 能设计高并发系统架构

### 项目产出
- [ ] 至少 2 个完整的系统项目
- [ ] 每个项目都有业务指标数据
- [ ] 代码托管在 GitHub，有 README
- [ ] 有系统架构图和技术文档

### 面试准备
- [ ] 能流利讲解项目的业务价值
- [ ] 能回答系统设计问题
- [ ] 能手撕常见代码题（LeetCode Hot 100）
- [ ] 能回答性能优化问题

---

**👉 返回主文档**：[README.md](../README.md)  
**👉 查看算法岗路线**：[算法岗学习路线](./learning-roadmap-algorithm.md)

