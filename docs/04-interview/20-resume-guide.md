---
type: 求职指南
status: 已发布
level: 入门
topic:
  - 面试求职
  - Agent
---

# AI Agent 简历编写指南

> 简历不是经历流水账，而是让面试官相信你能从 0 到 1 交付 Agent 系统的证据链。

---

## 📌 Section Goals（本节目标）

- 学会把 AI Agent / RAG / 大模型项目写成可被面试追问的简历亮点。
- 区分算法岗与开发岗的表达重点，避免“一份简历投所有岗位”。
- 用 L1-L4 能力模型组织项目证据：基础认知、开发实现、高级优化、生产工程。
- 掌握一页简历结构、项目 bullet 公式、量化指标写法和常见避坑清单。
- 使用可运行脚本快速检查项目描述是否过于空泛、缺少指标或缺少技术动作。

---

## 💡 Core Concepts（核心概念）

### 1. 简历的本质：面试入口，而不是个人传记

AI Agent 岗位的简历只需要回答三个问题：

1. **你做过什么真实问题？**
2. **你用了什么 Agent / RAG / LLM 技术解决？**
3. **结果是否能被量化、复现、追问？**

如果一句项目描述不能被追问出架构、数据、指标、取舍，它就不是有效亮点。

### 2. 面向三类读者写简历

| 读者 | 他们关心什么 | 简历应该怎么写 |
|:---|:---|:---|
| ATS / AI 筛选 | 关键词匹配 | 明确写出 LangGraph、RAG、MCP、Milvus、FastAPI、评估指标等关键词 |
| HR / 招聘 | 岗位匹配、稳定性 | 标题、技能、项目方向要贴 JD，避免过多陌生缩写 |
| 技术面试官 | 技术深度、贡献边界 | 每个项目写清楚你的动作、指标、难点和取舍 |

### 3. 1-2-5 框架在简历中的落地

| 框架 | 简历落地方式 |
|:---|:---|
| 1 个原则 | 从“我学过什么”改成“我做成了什么” |
| 2 条轨道 | 算法岗强调实验、模型、指标；开发岗强调系统、稳定性、业务价值 |
| 5 步链路 | 简历、投递、模拟面试、Vibe Coding、成果展示要互相支撑 |

### 4. L1-L4 能力模型映射

| 层级 | 简历证据 | 常见写法 |
|:---|:---|:---|
| L1 基础认知 | 理解 Agent / RAG / LLM 基础 | “掌握 ReAct、Tool Use、向量检索、Function Calling 基础原理” |
| L2 开发实现 | 能做出可运行系统 | “基于 LangGraph + FastAPI 实现多工具 Agent 服务” |
| L3 高级优化 | 能定位瓶颈并优化 | “通过混合检索 + rerank 将 Recall@5 从 71% 提升到 84%” |
| L4 生产工程 | 能上线、监控、降本、容错 | “接入日志追踪、失败重试、成本预算和评估流水线” |

---

## 🔍 Deep Understanding（深入理解）

### 一页简历推荐结构

```text
姓名 / 电话 / 邮箱 / GitHub / 个人主页
目标岗位：AI Agent 开发工程师 / 大模型算法工程师

教育背景：学校、专业、时间、核心课程或排名

技术栈：按方向分组，不要堆满所有工具
  Agent: LangGraph / AutoGen / ReAct / Tool Calling / Memory
  RAG: Milvus / FAISS / BGE / Reranker / GraphRAG
  LLM Engineering: vLLM / FastAPI / Docker / LangSmith / Promptfoo

项目经历：2-3 个强项目，每个 4-5 条 bullet

实习 / 科研 / 开源：优先放与目标岗位最贴近的经历

论文 / 博客 / 奖项：只放能增强岗位可信度的内容
```

### 项目 bullet 万能公式

```text
基于【技术栈】在【业务/研究场景】中设计并实现【系统/模块】，
解决【具体问题】，通过【优化动作】将【指标】从 A 提升到 B，
并沉淀【可复用产物/工程能力】。
```

### 不同岗位的写法差异

#### 算法岗版本

算法岗强调“问题定义、实验设计、指标提升、创新点”。

```text
设计面向多跳问答的 Agentic RAG 检索策略，引入 query decomposition + reranker
重排机制，在 300 条自建金融 QA 测试集上将 Recall@5 从 68.3% 提升至 82.7%，
并完成 BM25、Dense Retrieval、GraphRAG 三组 baseline 对比与消融实验。
```

#### 开发岗版本

开发岗强调“系统落地、稳定性、性能、业务闭环”。

```text
基于 LangGraph + FastAPI + Milvus 搭建企业知识库 Agent 服务，支持文档解析、
混合检索、工具调用和流式回答；通过缓存、异步队列和失败重试将 P95 响应
时间从 4.8s 降至 1.6s，日均稳定处理 2000+ 次查询。
```

### AI Agent 项目应该写出的 7 类证据

| 证据 | 说明 | 示例 |
|:---|:---|:---|
| 场景 | 为什么要做 | “企业制度文档检索慢，人工客服重复回答” |
| 架构 | 系统如何组织 | “Planner、Retriever、Tool Executor、Evaluator 四层架构” |
| 数据 | 用什么验证 | “自建 300 条 QA 集，覆盖规章、流程、异常案例” |
| 指标 | 怎么衡量成功 | “Recall@5、Faithfulness、P95 延迟、工具调用成功率” |
| 优化 | 做了哪些动作 | “混合检索、rerank、缓存、fallback、prompt 压缩” |
| 取舍 | 为什么这样设计 | “牺牲部分召回换取低延迟，保障在线体验” |
| 复盘 | 还能怎么改 | “下一步接入在线反馈和持续评估集” |

### 常见低质量写法

| 低质量写法 | 问题 | 改法 |
|:---|:---|:---|
| “熟悉 LangChain，做过 RAG 项目” | 没有场景、动作、指标 | 写出系统、数据集、指标和你的贡献 |
| “使用大模型完成智能问答” | 过于泛化 | 写清楚检索、生成、评估、部署链路 |
| “提升了系统效果” | 无法验证 | 写 Recall、准确率、延迟、成本、满意度 |
| “参与项目开发” | 贡献边界模糊 | 写“负责文档解析模块 / 评估模块 / Agent 状态机” |
| “阅读多篇论文” | 不是产出 | 写“复现某方法，并在自建数据集上完成对比实验” |

### 技术栈不要堆，要分组

不推荐：

```text
Python、Java、C++、LangChain、LangGraph、AutoGen、CrewAI、PyTorch、TensorFlow、
Milvus、Redis、MySQL、Docker、Kubernetes、Vue、React、Linux...
```

推荐：

```text
Agent 开发：LangGraph、ReAct、Function Calling、Tool Registry、Memory
RAG 系统：Milvus、FAISS、BGE Embedding、Reranker、GraphRAG
工程部署：FastAPI、Docker、Redis、异步任务、日志监控
模型训练：PyTorch、LoRA、SFT、DPO、评估集构建
```

### 三个可直接套用的项目模板

#### 模板 1：RAG / 知识库 Agent

```text
项目：企业级知识库 RAG Agent
- 基于 FastAPI + Milvus + BGE + LangGraph 实现企业知识库问答系统，覆盖文档解析、
  混合检索、rerank、引用溯源和多轮追问。
- 构建 300 条业务 QA 评估集，使用 Recall@5、Faithfulness、Answer Relevance
  评估效果，将正确引用率从 72% 提升至 88%。
- 设计检索失败 fallback、敏感词拦截和日志追踪机制，降低幻觉回答比例并提升可排查性。
- 将项目沉淀为 Docker Compose 一键启动 Demo，并编写算法岗/开发岗两版项目讲解稿。
```

#### 模板 2：Multi-Agent 协作系统

```text
项目：旅行规划 Multi-Agent 系统
- 基于 LangGraph 设计 Planner、Researcher、Budgeter、Critic 四类 Agent，
  支持用户偏好解析、景点检索、预算约束和行程反思。
- 引入状态机和任务队列控制 Agent 协作顺序，解决循环调用和上下文膨胀问题。
- 使用 50 个真实旅行需求构建评估集，从可执行性、预算一致性、用户偏好覆盖率
  三个维度评估规划质量。
- 接入工具调用日志和失败重试机制，将工具调用成功率提升到 94%。
```

#### 模板 3：Agent 评估 / Harness

```text
项目：Agent 自动化评估 Harness
- 设计面向工具调用 Agent 的离线评估流水线，支持测试集管理、LLM-as-Judge、
  规则评分和回归对比。
- 将任务完成率、工具调用正确率、平均成本和 P95 延迟纳入统一报告，支持 PR 前自动检查。
- 对比 Promptfoo、LangSmith、Braintrust 的适用场景，最终选择轻量自研脚本满足课程项目验证。
- 通过持续评估发现 3 类高频失败：工具参数错误、上下文污染、拒答边界不稳定。
```

---

## 💻 Code Examples（代码示例）

### 环境说明

- Python 版本：Python 3.10+
- 依赖：仅使用标准库
- `requirements.txt`：见 `docs/04-interview/examples/requirements.txt`

### 运行方式

```bash
cd AgentGuide
python docs/04-interview/examples/resume_storytelling_check.py --mode resume
python docs/04-interview/examples/resume_storytelling_check.py --mode story
```

这个脚本不会替你写简历，但可以帮你发现三类问题：

- 项目 bullet 是否缺少量化指标。
- 是否只写“负责、参与、熟悉”等弱动作。
- 项目讲述是否缺少背景、动作、结果、复盘。

### 最小输入示例

```text
基于 LangGraph + FastAPI + Milvus 搭建企业知识库 Agent 服务，
通过混合检索和 rerank 将 Recall@5 从 68% 提升到 84%，
并接入日志追踪、失败重试和评估集回归。
```

### 预期输出示例

```text
score: 100
warnings: []
suggestions:
- 保留当前写法，面试时准备架构图、评估集构造和失败案例。
```

---

## 🎯 Interview Questions（面试中如何考）

### Q1：你这个项目到底是不是自己做的？

回答思路：

1. 先用 20 秒讲清楚你负责的模块。
2. 再说一个你亲手解决的技术难点。
3. 最后给出指标变化或具体产物。

示例：

```text
我主要负责检索和评估模块。最开始系统直接用向量检索，FAQ 类问题还可以，
但流程类问题经常召回不全。我后来加入 BM25 + dense retrieval 的混合检索，
再用 reranker 做重排，并构建了 300 条业务 QA 测试集，Recall@5 从 68%
提升到 84%。这个结果也暴露出长文档切分粒度的问题，后面我又做了章节级切分。
```

### Q2：为什么不用现成平台，比如 Dify / Coze？

回答思路：

- 平台适合快速验证业务流程。
- 自研适合展示底层能力、可控性和可评估性。
- 如果岗位偏开发，可以强调平台集成能力；如果岗位偏算法，可以强调可控实验。

### Q3：你的指标怎么来的？可信么？

回答思路：

1. 说明测试集来源。
2. 说明指标定义。
3. 说明人工抽检或失败案例分析。

```text
测试集来自 50 份真实制度文档和 300 条人工改写问题，覆盖定义类、流程类、
异常处理类问题。指标上我没有只看主观满意度，而是同时看 Recall@5、
引用准确率、答案相关性和人工抽检结果。
```

### Q4：简历中最容易被追问的点有哪些？

- 指标提升是否可复现。
- 你的贡献边界是否清晰。
- 为什么选择这个架构或框架。
- 有没有失败案例和迭代记录。
- 如果数据量扩大 10 倍，系统怎么改。

---

## 📚 Extended Reading（扩展阅读）

- [秋招完整攻略](./08-job-hunting-guide.md)
- [HR 面试完全攻略](./10-hr-interview.md)
- [Agent 系统面试题](./03-agent-questions.md)
- [RAG 全流程面试题](./02-rag-questions.md)
- [Agent 评估完全指南](../02-tech-stack/agent-evaluation-complete-guide.md)
- [Agent Evaluation Harness 完全指南](../02-tech-stack/26-agent-evaluation-harness-guide.md)
- [项目讲述技巧](./21-storytelling.md)

---

## ✅ 最终自检清单

提交简历前逐项检查：

- 每个项目至少有 1 个量化指标。
- 每个项目都能讲清楚“背景、动作、结果、复盘”。
- 技术栈按方向分组，没有无意义堆关键词。
- 算法岗版本突出实验和指标，开发岗版本突出系统和落地。
- 简历中的每个缩写都能解释清楚。
- GitHub / Demo / 文章链接可以正常访问。
- 一页简历优先，不把空间浪费在无关课程和低相关经历上。
