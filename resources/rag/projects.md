---
type: 资源清单
status: 已发布
level: 通用
topic:
  - RAG
  - 项目实战
---

# 🔥 2025年最全RAG知识库项目汇总 | 从入门到精通的150+开源项目清单

## 💡 引言

在大模型时代，**RAG（检索增强生成）**已成为构建智能问答系统、知识库助手和AI应用的核心技术。无论你是：

- 🎓 **学生/转行者**：想通过实战项目提升简历竞争力
- 💼 **企业开发者**：需要快速搭建私域知识库或智能客服
- 🔬 **技术研究者**：探索GraphRAG、多模态检索等前沿方向
- 🚀 **创业者**：寻找可落地的AI产品解决方案

这份汇总都能为你提供明确的方向！

本文精心整理了 **150+ 个优质RAG开源项目**，涵盖：
- ✅ 开箱即用的企业级平台（如 Dify、FastGPT）
- ✅ 核心开发框架（LangChain、LlamaIndex）
- ✅ 前沿技术方向（GraphRAG、多智能体协作）
- ✅ 完整评估体系（ragas、AutoRAG）
- ✅ 实习/求职落地项目推荐

---

## 📋 目录


- [🚀 开箱即用平台](#-开箱即用平台)
    
- [🔧 核心开发框架](#-核心开发框架)
    
- [🔬 GraphRAG 图增强检索](#-graphrag-图增强检索)
    
- [⚙️ RAG 引擎与服务](#️-rag-引擎与服务)
    
- [📊 评估与优化框架](#-评估与优化框架)
    
- [📄 数据处理框架](#-数据处理框架)
    
- [🎯 实际应用项目](#-实际应用项目)
    
- [📚 学习资源与社区](#-学习资源与社区)
    
- [🏆 模型排行榜](#-模型排行榜)
    
      
    

---

  

## 🚀 开箱即用平台

  

### 企业级RAG平台

- **[Dify](https://github.com/langgenius/dify)** - 开源LLM应用开发平台，集成AI工作流、RAG管道、Agent能力
    
- **[AnythingLLM](https://github.com/Mintplex-Labs/anything-llm)** - 全能AI应用，支持任意LLM和完整RAG功能
    
- **[RAGFlow](https://github.com/infiniflow/ragflow)** - 基于深度文档理解的开源RAG引擎
    
- **[RAGapp](https://github.com/ragapp/ragapp)** - 企业级Agent RAG最简方案
    
      
    

### 中文知识库系统

- **[MaxKB](https://github.com/1Panel-dev/MaxKB)** - 基于LLM的知识库问答系统，开箱即用，支持快速嵌入第三方业务系统
    
- **[FastGPT](https://github.com/labring/FastGPT)** - 知识库平台，支持可视化流程编排
    
- **[Langchain-Chatchat](https://github.com/chatchat-space/Langchain-Chatchat)** - 基于Langchain与ChatGLM的本地知识库问答
    
      
    

### 个人文档助手

- **[Quivr](https://github.com/QuivrHQ/quivr)** - 个人生产力助手，支持PDF、CSV等文档对话
    
- **[kotaemon](https://github.com/Cinnamon/kotaemon)** - 简洁可定制的RAG聊天界面
    
- **[QAnything](https://github.com/netease-youdao/QAnything)** - 网易有道出品，支持任意内容问答
    
- **[Verba](https://github.com/weaviate/Verba)** - Weaviate驱动的检索增强生成聊天机器人
    
      
    

---

  

## 🔧 核心开发框架

  

### 主流开发框架

- **[LangChain](https://github.com/langchain-ai/langchain)** - 使用LLM构建应用程序的框架，生态最完整
    
- **[LlamaIndex](https://github.com/run-llama/llama_index)** - 构建数据驱动LLM应用程序的框架
    
- **[Haystack](https://github.com/deepset-ai/haystack)** - 使用神经网络构建搜索引擎的框架
    
- **[BentoML](https://github.com/bentoml/BentoML)** - 构建推理API、LLM应用、多模型链和RAG
    
- **[Dcup](https://github.com/Dcup-dev/dcup)** - 开源RAG即服务，自托管RAG管道
    
- **[Scout](https://github.com/scoutos)** - 使用LLM/向量数据库/网页抓取构建应用
    
      
    

### 专业化工具框架

- **[FlashRAG](https://github.com/RUC-NLPIR/FlashRAG)** - 高效RAG研究工具包
    
- **[RAGLite](https://github.com/superlinear-ai/raglite)** - 基于PostgreSQL/SQLite的轻量RAG工具
    
- **[MiniRAG](https://github.com/HKUDS/MiniRAG)** - 让小模型也能有好RAG性能的极简框架
    
- **[Chonkie](https://github.com/bhavnicksm/chonkie)** - 轻量快速的文本分块库
    
- **[Rankify](https://github.com/DataScienceUIBK/rankify)** - 检索、重排序和RAG的综合工具包
    
- **[TurboRAG](https://github.com/MooreThreads/TurboRAG)** - 通过预计算KV缓存加速RAG
    
      
    

### 新兴框架

- **[langflow](https://github.com/langflow-ai/langflow)** - 构建、扩展和部署RAG及多智能体AI应用
    
- **[dspy](https://github.com/stanfordnlp/dspy)** - 使用模块化编程构建语言模型应用
    
- **[mem0](https://github.com/mem0ai/mem0)** - AI应用的内存层
    
- **[cognee](https://github.com/topoteretes/cognee)** - 构建GraphRAG应用的内存框架
    
- **[ragbits](https://github.com/deepsense-ai/ragbits)** - 快速开发GenAI应用的构建模块
    
- **[Langroid](https://github.com/langroid/langroid)** - 轻松构建LLM驱动应用的Python框架
    
- **[Swiftide](https://github.com/bosun-ai/swiftide)** - 使用LLM构建快速流式应用的Rust库
    
- **[semantic-router](https://github.com/aurelio-labs/semantic-router)** - 使用语义向量路由LLM请求的框架
    
- **[Swarm by OpenAI](https://github.com/openai/swarm)** - 轻量级多智能体编排的教育框架
    
- **[NeMo-Guardrails](https://github.com/NVIDIA/NeMo-Guardrails)** - 为基于LLM的应用添加可编程护栏
    
      
    

### 云服务与托管平台

- **[AWS Bedrock Knowledge Bases](https://aws.amazon.com/bedrock/knowledge-bases/)** - 构建、扩展和部署RAG应用的托管服务
    
- **[Vertex AI Knowledge Engine](https://cloud.google.com/vertex-ai/generative-ai/docs/rag-overview)** - Google的上下文增强LLM应用数据框架
    
- **[Contextual AI](https://contextual.ai)** - 端到端RAG解决方案，包含文档理解、检索、生成和评估
    
      
    

---

  

## 🔬 GraphRAG 图增强检索

  

### 图RAG核心系统

- **[GraphRAG](https://github.com/microsoft/graphrag)** - 微软出品的模块化图RAG系统
    
- **[LightRAG (HKUDS)](https://github.com/HKUDS/LightRAG)** - 简单快速的检索增强生成
    
- **[LightRAG (SylphAI)](https://github.com/SylphAI-Inc/LightRAG)** - 帮助开发者构建和优化检索-智能体-生成管道
    
- **[Fast-GraphRAG](https://github.com/circlemind-ai/fast-graphrag)** - 智能适应您用例、数据和查询的RAG
    
- **[nano-GraphRAG](https://github.com/gusye1234/nano-graphrag)** - 简单易修改的GraphRAG实现
    
- **[Tiny-GraphRAG](https://github.com/limafang/tiny-graphrag)** - 极简图RAG实现
    
- **[DB-GPT GraphRAG](https://github.com/eosphoros-ai/DB-GPT/tree/main/dbgpt/storage/knowledge_graph)** - 仅需50%token的高效图RAG
    
      
    

### 图RAG扩展工具

- **[GraphRAG-Ollama-UI](https://github.com/severian42/GraphRAG-Ollama-UI)** - 带Gradio界面的Ollama图RAG
    
      
    

---

  

## ⚙️ RAG 引擎与服务

  

### 企业级RAG引擎

- **[R2R](https://github.com/SciPhi-AI/R2R)** - RAG界的Elasticsearch，快速构建可扩展RAG解决方案
    
- **[Agentset](https://github.com/agentset-ai/agentset)** - 开源智能体RAG平台
    
- **[TrustGraph](https://github.com/trustgraph-ai/trustgraph)** - LLM无关的智能体开发平台
    
- **[Engramic](https://github.com/engramic/engramic)** - 专注长期记忆和高级上下文管理的RAG引擎
    
- **[Vectara](https://www.vectara.com/)** - 快速构建AI助手和智能体的可信RAG平台
    
      
    

### 数据库集成RAG

- **[PostgresML](https://github.com/postgresml/postgresml)** - Postgres + GPU，提供分块、嵌入、转换和排序功能
    
- **[Korvus](https://github.com/postgresml/korvus)** - 在单个数据库查询中完成整个RAG管道
    
- **[pgai](https://github.com/timescale/pgai)** - 在PostgreSQL中开发RAG、语义搜索等AI应用的工具套件
    
- **[txtai](https://github.com/neuml/txtai)** - 用于语义搜索、LLM编排和RAG工作流的一体化嵌入数据库
    
      
    

### 轻量化RAG工具

- **[Embedchain](https://github.com/mem0ai/mem0/tree/main/embedchain)** - 10行代码内个性化LLM响应的开源框架
    
- **[dsRAG](https://github.com/D-Star-AI/dsRAG)** - 非结构化数据的高性能检索引擎
    
- **[Flash-Rank](https://github.com/PrithivirajDamodaran/FlashRank)** - 使用成对或列表重排序器提高搜索精度
    
- **[rag-citation](https://github.com/rahulanand1103/rag-citation)** - 结合RAG和自动引文生成提高内容可信度
    
- **[mode](https://github.com/rahulanand1103/mode)** - 专家模型、智能聚类和小数据集高效检索的RAG框架
    
- **[haiku.rag](https://github.com/ggozad/haiku.rag)** - 基于SQLite的开源RAG框架，支持监控、CLI、搜索、问答
    
      
    

### 对话与界面工具

- **[chainlit](https://github.com/Chainlit/chainlit)** - 几分钟内构建生产就绪的对话AI应用
    
- **[pathway](https://github.com/pathwaycom/pathway)** - 用于流处理、实时分析、LLM管道和RAG的Python ETL框架
    
- **[cognita](https://github.com/truefoundry/cognita)** - 模块化开源生产应用的RAG框架
    
      
    

### API与服务

- **[Liquid Index](https://liquidindex.dev)** - 统一RAG平台，一个API搞定所有工具
    
- **[Graphlit](https://github.com/graphlit)** - 构建知识驱动AI应用和智能体的API优先平台
    
- **[Interchange](https://github.com/getinterchange)** - 从文档上传到搜索的端到端RAG API
    
- **[ZeroEntropy AI](https://github.com/ZeroEntropy-AI)** - 开源重排序器、嵌入和端到端检索API
    
- **[memori](https://github.com/gibsonai/memori)** - 为AI智能体提供类人记忆的多智能体记忆引擎
    
      
    

### 特殊用途RAG

- **[KAG](https://github.com/OpenSPG/KAG)** - 基于OpenSPG引擎的知识增强生成框架
    
- **[CAG](https://github.com/hhhuang/CAG)** - 利用现代LLM扩展上下文窗口的缓存增强生成
    
- **[RAG-GPT](https://github.com/open-kf/rag-gpt)** - 利用LLM和RAG技术学习用户定制知识库
    
- **[TEN](https://github.com/TEN-framework/ten_framework)** - 下一代AI智能体框架，世界首个真正实时多模态AI智能体框架
    
- **[RAG-Anything](https://github.com/HKUDS/RAG-Anything)** - 全能RAG系统
    
      
    

---

  

## 📊 评估与优化框架

  

### 核心评估工具

- **[ragas](https://github.com/explodinggradients/ragas)** - 评估和量化RAG管道性能的框架
    
- **[AutoRAG](https://github.com/Marker-Inc-Korea/AutoRAG)** - 端到端RAG优化：解析、分块、评估数据集创建和管道部署
    
- **[XRAG](https://github.com/DocAILab/XRAG)** - 评估高级RAG系统基础组件的基准框架
    
      
    

### 监控与可观测性

- **[Trulens](https://github.com/truera/trulens)** - 通过反馈函数测量和增强LLM应用质量
    
- **[Phoenix](https://github.com/Arize-ai/phoenix)** - 为实验、评估和故障排除设计的AI可观测性平台
    
- **[langfuse](https://github.com/langfuse/langfuse)** - 追踪、评估、提示管理和指标，调试和改进LLM应用
    
- **[StepsTrack](https://github.com/lokwkin/steps-track)** - 跟踪、检查和可视化管道每个步骤的可观测性工具
    
      
    

### 优化与调试工具

- **[TextGrad](https://github.com/zou-group/textgrad)** - 基于LLM的文本优化框架，专注减少幻觉和改进提示
    
- **[Deepchecks](https://github.com/deepchecks/deepchecks)** - AI和ML模型的持续验证，检测数据漂移和模型问题
    
- **[evalmy.ai](https://github.com/evalmy-ai/evalmyai-python)** - 微调轻量级RAG评估服务 + Python客户端库
    
- **[syftr](https://github.com/datarobot/syftr)** - 多目标端到端智能体RAG优化
    
- **[zbench](https://github.com/ZeroEntropy-AI/zbench)** - 检索和重排序的标注评估框架
    
      
    

### 专业评估模型

- **[Vectara HHEM](https://huggingface.co/vectara/hallucination_evaluation_model)** - RAG幻觉评估模型
    
- **[LMUnit](https://docs.contextual.ai/api-reference/lmunit/lmunit)** - 为评估自然语言单元测试优化的语言模型
    
      
    

---

  

## 📄 数据处理框架

  

### ETL与索引框架

- **[CocoIndex](https://github.com/cocoindex-io/cocoindex)** - 构建新鲜索引的ETL框架
    
- **[Gitana.io](https://gitana.io/)** - 编辑审批和定时部署训练数据集到RAG向量数据库的内容平台
    
- **[Chonkie](https://github.com/chonkie-inc/chonkie)** - 无废话、轻量级快速的RAG分块库
    
      
    

---

  

## 🎯 实际应用项目

  

### 文档解析与处理

- **[LlamaParse](https://github.com/run-llama/llama_parse)** - GenAI原生文档解析平台
    
- **[Unstructured.io](https://github.com/Unstructured-IO/unstructured)** - 为标注、训练或生产ML构建自定义预处理管道
    
- **[Unstract](https://github.com/Zipstack/unstract)** - 启动API和ETL管道以结构化非结构化文档的无代码平台
    
- **[Reducto](https://reducto.ai/)** - 解析复杂文档并创建LLM就绪输入
    
- **[Chunkr](https://github.com/lumina-ai-inc/chunkr)** - 基于视觉模型的PDF分块和OCR，优化大数据集快速处理
    
- **[extractous](https://github.com/yobix-ai/extractous)** - AI应用的极快数据提取
    
- **[zchunk](https://github.com/ZeroEntropy-AI/zchunk)** - 开源高效的基于LLM的分块工具
    
- **[ChatDOC PDF Parser](https://pdfparser.io/?src=github)** - 将文档转换为RAG系统完美结构化数据的精确PDF解析
    
      
    

### 专业应用平台

- **[Langchain-extract](https://github.com/langchain-ai/langchain-extract)** - 使用LLM从文本和文件中提取信息的Web服务器
    
- **[Needle](https://github.com/oeken/needle-python)** - 开箱即用的生产就绪RAG管道
    
- **[Raggenie](https://github.com/sirocco-ventures/raggenie)** - 构建自定义基于RAG的AI应用的低代码平台
    
- **[tldw](https://github.com/rmusser01/tldw)** - 类似NotebookLM的开源项目
    
- **[SWIRL](https://github.com/swirlai/swirl-search)** - 工作场所AI搜索和RAG，立即获得公司知识AI洞察
    
- **[Gurubase](https://github.com/Gurubase/gurubase)** - 通过索引网站、PDF文档、YouTube视频和GitHub代码库创建AI驱动的问答助手
    
- **[Archive Agent](https://github.com/shredEngineer/Archive-Agent)** - 带OCR + AI搜索的开源语义文件跟踪器
    
- **[hydrot](https://github.com/rjxby/hydrot)** - 为企业文档设计的生产就绪RAG系统，支持markdown内容
    
      
    

### 商业服务平台

- **[Humata.ai](https://www.humata.ai/)** - 跨所有文档文件提问
    
- **[Ragie.ai](https://www.ragie.ai/)** - 为开发者提供的完全托管RAG即服务
    
- **[Midship](https://midship.ai/)** - 直接提取文档数据到电子表格/ERP/CRM
    
- **[DocuPanda](https://www.docupanda.io/)** - 将文档转换为结构化标准字段和值集合
    
      
    

### 特殊工具与集成

- **[contextual-doc-retrieval-opneai-reranker](https://github.com/lesteroliver911/contextual-doc-retrieval-opneai-reranker)** - 使用GPT-4和Cohere进行查询扩展和BM25重排序
    
- **[MidrasAI](https://github.com/ajac-zero/midrasai)** - Colpali多模态检索模型的简单API
    
- **[EmbeddingBridge](https://github.com/ProgramComputer/EmbeddingBridge)** - 嵌入的版本控制和迁移工具
    
- **[Stream-Rag-Agent](https://github.com/onurbaran/stream-rag-agent)** - Kafka流式RAG智能体
    
- **[Cerbos](https://github.com/cerbos/cerbos)** - RAG和LLM的访问控制
    
      
    

---

  

## 📚 学习资源与社区

  

### 技术资源与论文

- **[Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)** - Anthropic推出的上下文检索技术
    
- **[Open-RAG](https://arxiv.org/abs/2410.01782)** - 开源大语言模型增强检索推理
    
- **[ColPali](https://arxiv.org/abs/2407.01449)** - 使用视觉语言模型的高效文档检索
    
- **[RAG Techniques](https://github.com/NirDiamant/RAG_Techniques)** - 展示RAG系统各种高级技术
    
- **[GenAI_Agents](https://github.com/NirDiamant/GenAI_Agents)** - 各种AI智能体技术的教程和实现
    
      
    

### 社区与协作

- **[r/RAG](https://www.reddit.com/r/Rag/)** - RAG社区Reddit讨论区
    
- **[Discord](https://discord.gg/nn92wC5QmN)** - RAG项目Discord服务器
    
- https://datawhalechina.github.io/all-in-rag/#/
    
      
    

---

  

## 🏆 模型排行榜

  

### 性能评估榜单

- **[Artificial Analysis](https://artificialanalysis.ai)** - LLM对比分析
    
- **[MTEB嵌入模型排行榜](https://huggingface.co/spaces/mteb/leaderboard)** - HuggingFace嵌入模型性能榜单
    
- **[Vectara幻觉排行榜](https://github.com/vectara/hallucination-leaderboard)** - LLM幻觉检测评估排行榜
    
      
    

---

## 实习/转行落地项目推荐：

#### 1. graph-rag-agent (https://github.com/1517005260/graph-rag-agent)

- **完整描述**：这是一个聚焦于GraphRAG与私域Deep Search结合的项目，实现可解释、可推理的智能问答系统。融合GraphRAG、LightRAG、Neo4j-llm-graph-builder进行知识图谱构建和搜索；整合DeepSearch技术实现私域RAG推理；自制针对GraphRAG的评估框架，支持多Agent协作和知识图谱增强，构建完整的RAG智能交互解决方案。支持多格式文档处理（TXT、PDF、MD、DOCX、DOC、CSV、JSON、YAML/YML），包括实体提取、增量更新、社区检测和一致性验证。
    
- **Star数**：约1.2k。
    
- **业务背景/用例**：适用于私域知识管理、智能问答和复杂问题解决场景，如企业内部文档查询、政策解读或学生奖学金申请条件分析。通过多Agent协作处理复杂任务，适合教育、金融或法律等需要推理的业务。
    
- **关键特征**（包含RAG+Agent）：是的，使用GraphRAG进行检索增强生成，支持多级检索策略（本地、全局、混合搜索）和图谱增强上下文；Agent类型包括NaiveRagAgent、GraphAgent、HybridAgent、DeepResearchAgent、FusionGraphRAGAgent，支持多步思考-搜索-推理、证据链跟踪和可视化。评估框架有20+指标，RESTful API支持部署。
    
- **编程语言**：Python（基于FastAPI和相关LLM库）。
    
- **最后更新日期**：未明确，但仓库活跃，包含近期提交如添加OpenAI temporal graph agent示例。
    
- **Demo/示例**：提供终端测试输出（如查询“优秀学生的申请条件是什么？”的性能指标和结果）；Web Demo截图展示非调试/调试模式下的问答、轨迹跟踪、知识图谱和文档来源；快速启动文档（https://github.com/1517005260/graph-rag-agent/blob/master/assets/start.md）。
    
      
    

#### 2. chatwiki (https://github.com/zhimaAi/chatwiki)

- **完整描述**：这是一个开箱即用的基于企业私有知识库的LLM大语言模型智能客服机器人问答系统，支持私有化部署。由芝麻小客服官方推出，基于RAG和GraphRAG知识图谱构建，提供数据处理、模型调用，支持docx、Excel、PDF、OFD、PPT、Markdown等格式。支持20+大模型如DeepSeek、Qwen、OpenAI、Claude等。
    
- **Star数**：未找到明确数据（可能低于100，仓库相对新或 niche）。
    
- **业务背景/用例**：针对企业、大学和政府部门构建私有知识库AI问答系统，如客服支持、帮助中心、FAQ管理。多终端集成（网站、桌面客户端、微信、抖音、快手、API），处理通用用户查询，支持人工客服转介。适用于多步任务编排和业务系统集成，提高效率和隐私保护。
    
- **关键特征**（包含RAG+Agent）：是的，核心基于RAG检索增强生成，支持知识图谱和向量混合检索（可视化）；文本/图像回复、实时对话监控、问题优化和“猜你想问”推荐；细粒度权限管理、本地数据存储（加密、审计日志）；自动数据预处理、分段和向量化。Agent方面虽未明确，但支持工作流配置实现多步任务代理。
    
- **编程语言**：前端Vue.js，后端Golang + Python。
    
- **最后更新日期**：2025/09/26（包括工作流AI对话节点优化和性能提升）。
    
- **Demo/示例**：在线云版本（https://chatwiki.com）；WebApp Demo（http://h5.wikichat.com.cn/#/chat?robot_key=lTo85wvGSj）；客户端Demo下载（https://chatwiki.oss-cn-hangzhou.aliyuncs.com/demo/chatwiki.zip）；演示视频（https://xkf-upload-oss.xiaokefu.com.cn/chatwiki/Introductory.mp4）；界面截图展示知识库管理、对话和检索。
    

  

  

|   |   |   |   |   |
|---|---|---|---|---|
|项目名称|GitHub 链接|简要描述|业务背景|Star数|
|Controllable-RAG-Agent|https://github.com/NirDiamant/Controllable-RAG-Agent|高级RAG解决方案，使用确定性图谱作为Agent“大脑”，支持多步推理和幻觉预防。|文档分析和复杂查询，如法律审查或研究工具。|1.4k|
|LLMs-in-Finance|https://github.com/hananedupouy/LLMs-in-Finance|使用AutoGen等构建AI Agents，支持RAG检索和金融文档解析。|金融分析，如市场趋势和交易策略。|659|
|finllm-apps|https://github.com/tinztwins/finllm-apps|金融LLM应用集合，使用RAG和AI Agents，本地运行开源模型。|金融数据处理，适合隐私敏感场景。|134|
|MedRAG|https://github.com/Teddy-XiongGZ/MedRAG|RAG工具包，支持多代理框架，用于医疗QA的多步推理。|医疗问答，如临床决策和文献分析。|429|
|Multi-Agent-Medical-Assistant|https://github.com/souvikmajumder26/Multi-Agent-Medical-Assistant|多代理医疗助手，使用RAG和代理协作，支持图像分析和实时检索。|医疗诊断和研究，如脑瘤检测。|564|

  

这些筛选后的项目Star较高，社区活跃，便于落地。如果需要更多细节或额外搜索特定场景的项目，请告知！

  

## 🎯 选择建议

  

### 新手入门推荐

1. **个人学习** → Quivr / kotaemon
    
2. **企业试点** → MaxKB / FastGPT
    
3. **开发实验** → Dify / AnythingLLM
    
      
    

### 进阶应用推荐

1. **图谱增强** → LightRAG / GraphRAG
    
2. **性能优化** → FlashRAG / AutoRAG
    
3. **生产部署** → RAGFlow / R2R
    
      
    

### 技术研究推荐

1. **评估测试** → ragas / XRAG
    
2. **算法优化** → RAG Techniques
    
3. **前沿探索** → ColPali / TEN