# AI Agent 开发工程师学习路线图（工程落地版）

> **目标岗位**：AI Agent 开发工程师（应用型、工程型）  
> **学习时长**：8 周（全职投入）  
> **最终产出**：2-3 个生产级、可部署的 Agent 系统 + 完整的全栈技术能力

---

## 一、你能获得什么

> **用8周，打造从原型到生产的完整工程能力**
>
> ✅ **8周系统课程**：从 LangChain 基础到生产级 Agent 系统架构
>
> ✅ **每周代码实战**：手撕 RAG、Agent、多智能体，将想法变为高可用服务
>
> ✅ **2个工业级项目**：完成从需求分析、技术选型、开发部署到监控优化的全流程
>
> ✅ **生产级技术栈**：掌握 FastAPI, Docker, Redis, Prometheus 等后端必备技能
>
> ✅ **顶级面试能力**：搞定系统设计、性能优化、故障排查等高频面试题

---

## 二、开发岗核心要求

### 你需要具备的能力

<table>
<tr>
<td width="33%">

**系统设计**
- 高可用、高并发架构
- 性能与成本优化
- 监控、告警与日志
- 故障排查与容错

</td>
<td width="33%">

**工程实现**
- 熟练使用 Agent 框架
- 高质量、可维护的代码
- 快速开发与迭代能力
- 强大的调试与问题定位

</td>
<td width="33%">

**业务理解**
- 用户需求转化为技术方案
- 场景适配与方案选型
- 数据驱动的系统优化
- 评估技术方案的 ROI

</td>
</tr>
</table>

### 开发岗简历必备

✅ **至少2个完整系统项目**：端到端可运行，有线上部署经验  
✅ **量化的业务指标提升**：如 QPS+100%、P99延迟-80%、成本-50% 等数据  
✅ **丰富的生产级技术栈**：LangChain + FastAPI + Milvus + Redis + Docker + Prometheus  
✅ **生产化经验**：有部署、监控、性能优化、异常处理的实战经历

---

## 三、8周详细学习计划

### **第 1 周：大模型应用开发基础 + 手撕 Naive RAG**

> **学习内容:**
> - **后端基础**: FastAPI 路由、异步 I/O、Pydantic 数据校验
> - **LangChain 核心**: LLM, Prompt Templates, Output Parsers, LCEL
> - **Naive RAG 流程**: Document Loaders, Text Splitters, Embeddings, Vector Stores
> - **向量数据库**: FAISS/ChromaDB 本地化使用
>
> **手撕系列:**
> - [ ] FastAPI 搭建 "Hello, World" API 服务
> - [ ] LangChain LCEL 编写第一个 LLM Chain
> - [ ] 30分钟手撕一个完整的 Naive RAG 应用
>
> **解锁技能:**
> - 熟练使用 FastAPI 搭建 API
> - 掌握 LangChain 核心组件与 LCEL 表达式语言
> - 能够从零开始，快速构建一个基于文档问答的 RAG Demo

**🌟 每日学习计划**

| **天数** | **学习主题** | **资源链接** | **目标** |
|---|---|---|---|
| 1 | FastAPI 快速入门 | 教程: [FastAPI Official Tutorial](https://fastapi.tiangolo.com/tutorial/) | 掌握 FastAPI 基础，能够创建路由、处理请求 |
| 2 | LangChain 核心概念 | 文档: [LangChain Quickstart](https://python.langchain.com/v0.1/docs/get_started/quickstart/) | 理解 LangChain 六大核心模块，熟练使用 LCEL |
| 3 | RAG Part 1: 加载与分割 | 文档: [LlamaIndex Loaders](https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/root.html) | 掌握不同格式文档 (PDF, MD) 的加载和文本分块策略 |
| 4 | RAG Part 2: 向量化与存储 | 教程: [FAISS Intro](https://github.com/facebookresearch/faiss/wiki/Getting-started) | 理解 Embedding 原理，使用 FAISS/Chroma 构建本地向量索引 |
| 5-6 | 手撕 Naive RAG 系统 | 教程: [RAG from Scratch](https://github.com/langchain-ai/rag-from-scratch) | 整合 FastAPI + LangChain，完成一个端到端的文档问答 API |
| 7 | 周度总结与项目部署 | | 将本周的 RAG 项目用 Docker 打包，并成功运行 |

---

### **第 2 周：Advanced RAG 与生产级向量数据库**

> **学习内容:**
> - **Advanced RAG 技术**: Query Transformation, Re-ranking, Hybrid Search
> - **RAG 评估**: 使用 RAGAs, TruLens 进行自动化评估
> - **生产级向量数据库**: Milvus/Zilliz Cloud 部署与使用
> - **数据处理**: Unstructured.io 解析复杂文档
>
> **手撕系列:**
> - [ ] 实现 BM25 + 向量的混合检索
> - [ ] 引入 Cohere Rerank 模型提升检索精度
> - [ ] 使用 RAGAs 评估 RAG 系统的 Faithfulness 和 Answer Relevancy
> - [ ] Docker 部署 Milvus 并进行增删改查操作
>
> **解锁技能:**
> - 掌握 10+ 种 RAG 优化策略
> - 能够建立 RAG 系统的自动化评估流水线
> - 熟练使用生产级的分布式向量数据库 Milvus
> - 具备处理复杂、非结构化文档的能力

**🌟 每日学习计划**

| **天数** | **学习主题** | **资源链接** | **目标** |
|---|---|---|---|
| 8 | Query Transformation | 教程: [LlamaIndex Query Transforms](https://docs.llamaindex.ai/en/stable/module_guides/querying/query_transforms/root.html) | 实现 HyDE, Multi-Query 等查询改写策略 |
| 9 | 混合检索与重排 (Rerank) | 教程: [LlamaIndex Reranking](https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/CohereRerank.html) | 实现 BM25 + Embedding 混合检索，并集成 Reranker |
| 10-11 | RAG 评估体系 | 文档: [RAGAs 评估框架](https://docs.ragas.io/en/latest/index.html) | 学习 RAG 核心评估指标，并用 RAGAs 评估优化前后的系统性能 |
| 12 | 生产级向量数据库 (Milvus) | 文档: [Milvus Quick Start](https://milvus.io/docs/install_standalone-docker.md) | 使用 Docker 部署 Milvus，并掌握其 Python SDK |
| 13 | 高级数据处理 | 文档: [Unstructured.io](https://unstructured-io.github.io/unstructured/) | 使用 Unstructured 解析包含表格、图片的复杂 PDF |
| 14 | 周度总结与系统升级 | | 将第一周的 RAG 系统升级，集成混合检索、Reranker 和 Milvus |

---

### **第 3 周：Agent 开发与 Tool Calling**

> **学习内容:**
> - **Agent 核心**: ReAct 框架, Planning, Tool Use, Memory
> - **Tool Calling**: OpenAI Function Calling, Tool Schema 定义
> - **工具开发**: 如何将 API, 数据库查询等封装为 Agent 可用的工具
> - **错误处理**: 工具调用失败的重试、降级策略
>
> **手撕系列:**
> - [ ] 实现 3个 自定义工具 (天气查询, SQL数据库查询, API调用)
> - [ ] 基于 LangChain 构建一个可以链式调用工具的 Agent
> - [ ] 使用 OpenAI Function Calling 实现结构化数据提取
>
> **解锁技能:**
> - 深刻理解 Agent 的"思考-行动"工作流
> - 能够开发、测试、维护自定义工具集
> - 掌握 Function Calling 的原理与应用
> - 具备构建能处理真实世界任务的 Agent 的能力

**🌟 每日学习计划**

| **天数** | **学习主题** | **资源链接** | **目标** |
|---|---|---|---|
| 15 | Agent 核心概念 | 文档: [LangChain Agents](https://python.langchain.com/v0.1/docs/modules/agents/) | 理解 ReAct 框架，并运行一个 LangChain 官方的 Agent 示例 |
| 16 | 自定义工具开发 | 教程: [LangChain Custom Tools](https://python.langchain.com/v0.1/docs/modules/tools/custom_tools/) | 编写一个查询天气的自定义工具，并集成到 Agent 中 |
| 17 | SQL & 数据库工具 | 教程: [LangChain SQL Agent](https://python.langchain.com/v0.1/docs/use_cases/sql/) | 构建一个能根据自然语言查询数据库的 SQL Agent |
| 18 | Function Calling 实战 | 文档: [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling) | 使用 OpenAI API 实现一个能根据用户问题调用函数的 Agent |
| 19 | Agent Memory | 文档: [LangChain Memory](https://python.langchain.com/v0.1/docs/modules/memory/) | 为 Agent 添加对话历史记忆 (ConversationBufferMemory) |
| 20 | Agent 错误处理 | | 为工具调用添加重试机制 (`tenacity` 库) 和降级策略 |
| 21 | 周度总结与项目构建 | | 构建一个集成 RAG 和 Web 搜索工具的 "研究助手" Agent |

---

### **第 4 周：系统性能优化**

> **学习内容:**
> - **缓存策略**: Redis 缓存 LLM 响应和 Embedding 结果
> - **异步处理**: `asyncio`, `aiohttp` 实现高并发
> - **批处理优化**: Embedding 和 LLM 调用的批处理
> - **推理加速**: vLLM, TensorRT-LLM 部署与使用
>
> **手撕系列:**
> - [ ] 为 RAG 系统引入 Redis 缓存，对比优化前后性能
> - [ ] 将 FastAPI 的同步接口改造为异步接口
> - [ ] 部署 vLLM 并通过 API 进行推理
>
> **解锁技能:**
> - 掌握 LLM 应用的核心性能优化手段
> - 能够将系统的 QPS 提升 10 倍以上
> - 熟练使用 Redis 进行缓存设计
> - 具备部署和使用高性能推理引擎的能力

**🌟 每日学习计划**

| **天数** | **学习主题** | **资源链接** | **目标** |
|---|---|---|---|
| 22 | 性能瓶颈分析 | | 学习使用 `cProfile`, `py-spy` 等工具分析现有 Agent 系统的性能瓶颈 |
| 23 | 缓存优化 (Redis) | 教程: [FastAPI with Redis](https://testdriven.io/blog/fastapi-redis/) | 为 Agent 系统添加 Redis 缓存，缓存 LLM 响应 |
| 24-25 | 异步处理 (Async) | 教程: [FastAPI Async](https://fastapi.tiangolo.com/async/) | 将系统中 I/O 密集型操作 (如 API 调用) 改造为异步 |
| 26 | 批处理优化 (Batching) | | 实现 Embedding 和 Reranker 的批处理，提升吞吐量 |
| 27 | 高性能推理 (vLLM) | 文档: [vLLM Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html) | 使用 vLLM 部署一个开源模型 (如 Llama 3)，并测试其吞吐量 |
| 28 | 周度总结与性能压测 | | 使用 `locust` 或 `jmeter` 对优化前后的系统进行压测，并记录 QPS, P99 等指标 |

---

### **第 5 周：监控、可观测性与部署**

> **学习内容:**
> - **Agent 链路追踪**: LangSmith, OpenTelemetry
> - **指标监控**: Prometheus 监控业务和系统指标
> - **可视化**: Grafana 创建监控大盘
> - **日志系统**: ELK Stack (Elasticsearch, Logstash, Kibana)
> - **容器化部署**: Docker, Docker Compose
>
> **手撕系列:**
> - [ ] 为 Agent 应用集成 LangSmith，追踪每一步的调用和延迟
> - [ ] 使用 Prometheus 暴露自定义指标 (如 Token 消耗, 缓存命中率)
> - [ ] 使用 Docker Compose 将 FastAPI + Milvus + Redis 整套系统一键部署
>
> **解锁技能:**
> - 具备构建完整 LLM 应用可观测性体系的能力
> - 能够快速定位和诊断线上问题
> - 掌握基于 Docker 的容器化部署和编排
> - 拥有完整的 DevOps for LLM Apps 经验

**🌟 每日学习计划**

| **天数** | **学习主题** | **资源链接** | **目标** |
|---|---|---|---|
| 29 | 链路追踪 (LangSmith) | 文档: [LangSmith](https://docs.smith.langchain.com/) | 将 LangSmith 集成到现有 Agent 应用中，分析调用链路 |
| 30 | 指标监控 (Prometheus) | 教程: [Prometheus Python Client](https://github.com/prometheus/client_python) | 暴露 API 的 QPS, 延迟, 错误率等核心指标 |
| 31 | 可视化 (Grafana) | | 安装 Grafana，并创建一个简单的监控大盘来展示 Prometheus 指标 |
| 32 | 容器化 (Docker) | 教程: [Docker for FastAPI](https://fastapi.tiangolo.com/deployment/docker/) | 为 FastAPI 应用编写 Dockerfile 并成功构建镜像 |
| 33 | 服务编排 (Docker Compose) | | 编写 `docker-compose.yml` 文件，一键启动整个应用栈 |
| 34 | 日志系统 | | 配置应用将日志输出为 JSON 格式，为接入 ELK 做准备 |
| 35 | 周度总结与生产环境模拟 | | 模拟一次线上故障，并使用本周学习的工具链进行问题定位 |

---

### **第 6 周：Multi-Agent 系统开发**

> **学习内容:**
> - **Multi-Agent 框架**: AutoGen vs. CrewAI
> - **Agent 角色定义**: 如何设计具有不同职责和能力的 Agent
> - **通信机制与工作流**: GroupChat, Sequential/Hierarchical flow
> - **状态管理**: 如何在多个 Agent 之间共享和传递状态
>
> **手撕系列:**
> - [ ] 使用 AutoGen 构建一个“研究员-程序员-测试员”协作的软件开发团队
> - [ ] 使用 CrewAI 构建一个“旅行规划师-本地向导-预订专员”的旅行 Agent
>
> **解锁技能:**
> - 掌握至少两种主流的 Multi-Agent 开发框架
> - 能够根据复杂业务需求，设计和实现多智能体协作系统
> - 理解不同协作模式 (如层级式 vs. 对话式) 的优缺点

**🌟 每日学习计划**

| **天数** | **学习主题** | **资源链接** | **目标** |
|---|---|---|---|
| 36-37 | AutoGen 核心概念 | 文档: [AutoGen Tutorial](https://microsoft.github.io/autogen/docs/getting-started/) | 学习 `ConversableAgent`, `GroupChat` 等核心概念，并运行官方示例 |
| 38 | AutoGen 实战 | | 实现一个“研究员-程序员-测试员”的 Multi-Agent 系统 |
| 39-40 | CrewAI 核心概念 | 文档: [CrewAI Docs](https://docs.crewai.com/) | 学习 Agent, Task, Crew, Process 的概念，并运行官方示例 |
| 41 | CrewAI 实战 | | 实现一个“旅行规划师-本地向导-预订专员”的 Multi-Agent 系统 |
| 42 | 框架对比与总结 | | 对比 AutoGen 和 CrewAI 的设计哲学、优缺点和适用场景 |

---

### **第 7-8 周：工业级项目实战与面试准备**

> **核心目标**：完成 1-2 个可写进简历的完整系统，并准备面试。

#### **项目1：企业级智能客服 RAG 系统**

> **业务场景**: 为某电商公司构建智能客服系统，自动回答 80% 的重复性用户问题 (订单状态、物流、退款等)。
>
> **技术要求**:
> -   **数据源**: 对接 FAQ 文档、商品信息数据库 (PostgreSQL)。
> -   **核心**: 实现一个混合检索 RAG，优先从数据库精确查询，无法命中再从文档模糊检索。
> -   **性能**: 系统 QPS > 200, P99 延迟 < 500ms。
> -   **监控**: 完整的 LangSmith + Prometheus + Grafana 监控体系。
> -   **部署**: 使用 Docker Compose 部署。
>
> **简历亮点**: 高并发、低延迟、生产级监控、节省XX人力成本。

#### **项目2：Agent 驱动的自动化投研系统**

> **业务场景**: 为投资分析师构建自动化报告生成 Agent，输入公司名，自动完成信息搜集、分析和报告撰写。
>
> **技术要求**:
> -   **Multi-Agent**: 使用 CrewAI 构建，包含 `信息搜集Agent` (调用搜索引擎、API)、`财报分析Agent` (解析PDF财报、计算关键指标)、`报告撰写Agent`。
> -   **工具集**: 集成 Google Search, SEC API, 文件读写等至少 5 个工具。
> -   **稳定性**: 强大的异常处理和重试机制，任务成功率 > 95%。
> -   **工作流**: 设计一个顺序工作流，并记录每一步的中间产出。
>
> **简历亮点**: Multi-Agent 协作、复杂工作流自动化、为分析师提升XX%工作效率。

**🌟 学习计划 (2周)**

| **天数** | **学习主题** | **目标** |
|---|---|---|
| 43-47 | 项目一：智能客服 RAG | 完成需求分析、架构设计、核心功能开发 |
| 48-51 | 项目一：优化与部署 | 完成性能优化、监控集成和 Docker 部署，撰写项目文档 |
| 52-56 | 项目二：自动化投研 Agent | 完成需求分析、Agent 设计、工具开发和工作流实现 |
| 57-58 | 简历撰写与项目总结 | 按照开发岗模板，将两个项目经历量化地写入简历 |
| 59-60 | 系统设计与面试 Mock | 准备高频系统设计题，并进行 1v1 模拟面试 |

---

## 👉 返回主文档：[README.md](../README.md)

