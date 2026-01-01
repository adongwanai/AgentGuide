


> 本文系统梳理Agentic RAG领域的优质开源项目和前沿论文，包含GitHub链接、论文地址和核心创新点。

---

## 目录

1. [什么是Agentic RAG](#什么是agentic-rag)
2. [开源项目推荐](#开源项目推荐)
3. [论文全景](#论文全景)
4. [学习路径](#学习路径)
5. [技术选型建议](#技术选型建议)


---

## 什么是Agentic RAG?

传统RAG虽然能检索知识生成回答，但在处理复杂多步推理、动态适应等任务时力不从心。**Agentic RAG**通过引入自主AI Agent，结合以下核心模式：

- **反思 (Reflection)**: 自我评估和迭代优化
- **规划 (Planning)**: 多步任务分解和编排
- **工具使用 (Tool Use)**: 动态调用外部API和知识库
- **多智能体协作 (Multi-Agent)**: 分布式任务处理

让系统能够动态适应任务需求，在医疗诊断、金融分析、法律合规等领域表现出色。

---

## 开源项目推荐

### 🌟 1. 基础单智能体RAG

#### Athina AI RAG Cookbooks ⭐⭐⭐⭐⭐
- **GitHub**: https://github.com/athina-ai/rag-cookbooks
- **技术栈**: LangChain + FAISS + Athina AI
- **推荐理由**: 完整的Agentic RAG技术实现，包含Basic、Corrective、Adaptive、ReAct、Self RAG，配套Colab Notebook可直接运行
- **适合人群**: 初学者系统学习

#### IBM Granite Agentic RAG
- **GitHub**: https://github.com/ibm-granite-community/granite-snack-cookbook
- **技术栈**: IBM Granite-3-8B-Instruct + Watsonx.ai + Chroma DB
- **特色**: 展示国产化模型实现Agentic RAG

#### NVIDIA Agentic RAG
- **GitHub**: https://github.com/NVIDIA/workbench-example-agentic-rag
- **技术栈**: LangGraph + Chroma + NVIDIA NIMs + Tavily Search
- **特色**: 路由架构智能判断RAG/WebSearch pipeline，利用NVIDIA推理加速

---

### 🤝 2. 多智能体协作RAG

#### Azure GPT-RAG Agentic Orchestrator
- **GitHub**: https://github.com/Azure/gpt-rag-agentic
- **技术栈**: AutoGen + SQL + AI Search
- **特色**: 工厂模式+预定义策略（classic_rag、nl2sql），灵活性极高

#### Hierarchical Multi-Agent RAG
- **GitHub**: https://github.com/lorenzejay/agentic-rag-practical-example
- **技术栈**: Weaviate + ExaSearch + Groq + CrewAI
- **特色**: 分层智能体架构，Manager Agent协调专门工具智能体

---

### 🔧 3. 自适应与纠错RAG

#### Hugging Face Agentic RAG
- **GitHub**: https://github.com/aymericroucher/agentic-rag-query-reformulation
- **技术栈**: SmolAgents + HuggingFace + HyDE + Self-Query
- **特色**: 查询重构+自查询策略+自评分机制，符合Corrective RAG理念

---

### 🏢 4. 企业级文档工作流

#### LlamaCloud Demo系列
- **GitHub**: https://github.com/run-llama/llamacloud-demo
- **应用场景**:
  - 患者病例摘要: `/examples/document_workflows/patient_case_summary/`
  - 合同审查: `/examples/document_workflows/contract_review/`
  - 保险理赔: `/examples/document_workflows/auto_insurance_claims/`
  - 研究报告生成: `/examples/report_generation/research_paper_report_generation.ipynb`
- **特色**: Agentic Document Workflows(ADW)完整实现，企业级应用典范

---

### ⚡ 5. 性能优化RAG

#### Redis Agentic RAG
- **GitHub**: https://github.com/redis-developer/agentic-rag
- **技术栈**: LlamaIndex + Redis + Amazon Bedrock + SemanticCache
- **特色**: ReAct agent架构+Redis向量存储+语义缓存，大幅降低LLM调用成本

---

### 📚 6. 垂直领域应用

#### LawGlance (法律研究)
- **Colab**: https://colab.research.google.com/drive/1yrS2Kp-kprYWot_sEu7JeWMIRAei_vov
- **技术栈**: Crew AI + LangChain + Chroma
- **特色**: 多智能体协作检索法律文档，提供精准法律见解

---

## 论文全景

### 📊 分类概览

| 分类 | 数量 | 代表工作 |
|------|------|----------|
| **综述** | 1篇 | Agentic RAG Survey |
| **经典基础** | 3篇 | RAG, ReAct, Self-RAG |
| **强化学习驱动** | 40+篇 | Search-R1系列 |
| **图增强** | 2篇 | GeAR, Agent-G |
| **自适应** | 4篇 | Adaptive-RAG系列 |
| **纠错型** | 2篇 | CRAG系列 |

---

### 📋 1. 必读综述

#### Agentic RAG Survey (2025.01) ⭐⭐⭐⭐⭐
- **论文**: https://arxiv.org/abs/2501.09136
- **GitHub**: https://github.com/asinghcsu/AgenticRAG-Survey
- **核心贡献**: 最全面的Agentic RAG综述，系统梳理分类体系、设计模式、应用场景和未来方向
- **推荐理由**: 入门必读，提供完整技术路线图

---

### 🎯 2. 经典基础论文

#### RAG开山之作 (NeurIPS 2020) ⭐⭐⭐⭐⭐
- **论文**: https://arxiv.org/abs/2005.11401
- **GitHub**: https://github.com/huggingface/transformers
- **标题**: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
- **核心贡献**: Facebook AI提出的检索增强生成范式，奠定整个RAG研究基础

#### ReAct (ICLR 2023) ⭐⭐⭐⭐⭐
- **论文**: https://arxiv.org/abs/2210.03629
- **GitHub**: https://github.com/ysymyth/ReAct
- **标题**: Synergizing Reasoning and Acting in Language Models
- **核心贡献**: Reasoning + Acting范式，LLM交替生成推理轨迹和执行动作，是Agentic RAG理论基石

#### Self-RAG (ICLR 2024 Oral) ⭐⭐⭐⭐⭐
- **论文**: https://arxiv.org/abs/2310.11511
- **GitHub**: https://github.com/AkariAsai/self-rag
- **标题**: Learning to Retrieve, Generate, and Critique through Self-Reflection
- **核心贡献**: 训练LLM生成反思标记来自我评估检索必要性和响应质量，自反思机制里程碑工作

#### Reflexion (NeurIPS 2023)
- **论文**: https://arxiv.org/abs/2303.11366
- **GitHub**: https://github.com/noahshinn024/reflexion
- **标题**: Language Agents with Verbal Reinforcement Learning
- **核心贡献**: 基于语言反馈的强化学习框架，支撑反思型Agent设计

#### Toolformer (2023.02)
- **论文**: https://arxiv.org/abs/2302.04761
- **标题**: Language Models Can Teach Themselves to Use Tools
- **核心贡献**: Meta AI展示LLM自主学习使用外部工具能力

#### REPLUG (2023.01)
- **论文**: https://arxiv.org/abs/2301.12652
- **GitHub**: https://github.com/swj0419/REPLUG
- **标题**: Retrieval-Augmented Black-Box Language Models
- **核心贡献**: 黑盒LLM检索增强，无需模型微调

---

### 🚀 3. 强化学习驱动的搜索增强RAG (2025年最前沿)

> 这是2025年最热门的研究方向，通过RL训练智能体自主决策何时检索、如何检索，实现真正的Agentic RAG。

#### 🔥 核心开创性工作

**Search-o1** (2025.01) ⭐⭐⭐⭐⭐
- 论文: https://arxiv.org/abs/2501.05366
- GitHub: https://github.com/sunnynexus/Search-o1
- 核心: 增强O1推理模式的大模型自主检索能力

**Search-R1** (COLM 2025) ⭐⭐⭐⭐⭐
- 论文: https://arxiv.org/abs/2503.09516
- GitHub: https://github.com/PeterGriffinJin/Search-R1
- 核心: 将RL扩展到RAG场景，开创RL for RAG研究范式
- 影响: 后续40+篇论文的基础工作

**R1-Searcher** (2025.03) ⭐⭐⭐⭐
- 论文: https://arxiv.org/abs/2503.05592
- GitHub: https://github.com/RUCAIBox/R1-Searcher
- 核心: 两阶段RL框架，仅依赖最终奖励

**R1-Searcher++** (2025.05)
- 论文: https://arxiv.org/abs/2505.17005
- GitHub: https://github.com/RUCAIBox/R1-Searcher-plus
- 核心: 增强版，引入内部知识利用奖励+记忆机制

#### 📍 检索优化方向

**DeepRetrieval** (2025.03)
- 论文: https://arxiv.org/abs/2503.00223
- GitHub: https://github.com/pat-jj/DeepRetrieval
- 核心: 以检索指标为奖励，查询增强补充语义

**O1 Embedder** (2025.02)
- 论文: https://arxiv.org/abs/2502.07555
- GitHub: https://github.com/RuiranYan/o1embedder
- 核心: Embedder检索前生成推理thoughts

**ZeroSearch** (2025.05)
- 论文: https://arxiv.org/abs/2505.04588
- GitHub: https://github.com/Alibaba-NLP/ZeroSearch
- 核心: LLM模拟搜索引擎，降低88%成本

**s3** (2025.05)
- 论文: https://arxiv.org/abs/2505.14146
- GitHub: https://github.com/pat-jj/s3
- 核心: 解耦搜索器和生成器，仅用2.4k样本，提出Gain Beyond RAG奖励

#### 🎨 奖励机制创新

**ReSearch** (2025.03)
- 论文: https://arxiv.org/abs/2503.19470
- GitHub: https://github.com/Agent-RL/ReSearch
- 核心: 格式奖励+F1 score

**R-Search** (2025.06)
- 论文: https://arxiv.org/abs/2506.04185
- GitHub: https://github.com/QingFei1/R-Search
- 核心: 多阶段混合奖励（答案质量+证据质量+格式）

**StepSearch** (2025.05)
- 论文: https://arxiv.org/abs/2505.15107
- GitHub: https://github.com/Zillwang/StepSearch
- 核心: Token级步骤奖励（StePPO），信息增益+冗余惩罚

**LeTS** (2025.05)
- 论文: https://arxiv.org/abs/2505.17447
- GitHub: https://github.com/Cheungki/LeTS
- 核心: 过程-结果奖励混合，惩罚重复检索

**Search Wisely (β-GRPO)** (2025.05)
- 论文: https://arxiv.org/abs/2505.17281
- GitHub: https://github.com/mianzhang/Search-R1
- 核心: 减少不确定性，缓解次优搜索

**HiPRAG** (2025.10)
- 论文: https://arxiv.org/abs/2510.07794
- GitHub: https://github.com/qualidea1217/HiPRAG
- 核心: 分层过程奖励，实时检测冗余/缺失搜索

**E-GRPO** (2025.10)
- 论文: https://arxiv.org/abs/2510.24694
- 核心: 合成数据实体信息作为细粒度奖励

**LLDS & MA-GRPO** (2025.12) 🌟
- 论文: https://arxiv.org/abs/2512.04220
- 核心: 解决GRPO训练崩溃，提出懒惰似然位移理论+似然保持正则化

#### 🧠 知识利用策略

**IKEA** (2025.05)
- 论文: https://arxiv.org/abs/2505.07596
- GitHub: https://github.com/hzy312/knowledge-r1
- 核心: 内外部知识协同，优先使用内部知识

**AutoRefine** (2025.05)
- 论文: https://arxiv.org/abs/2505.11277
- GitHub: https://github.com/syr-cn/AutoRefine
- 核心: 边检索边精炼，检索过程自我进化

#### 🌐 真实环境与长视野

**DeepResearcher** (2025.04)
- 论文: https://arxiv.org/abs/2504.03160
- GitHub: https://github.com/GAIR-NLP/DeepResearcher
- 核心: 真实网络环境深度研究，Search+Browse

**ASearcher** (2025.08)
- 论文: https://arxiv.org/abs/2508.07976
- GitHub: https://github.com/inclusionAI/ASearcher
- 核心: 大规模异步RL，支持10+轮长视野搜索

#### 🎭 多智能体与模块化

**ReAgent** (2025.03)
- 论文: https://arxiv.org/abs/2503.06951
- GitHub: https://github.com/astridesa/ReAgent
- 核心: 可逆多智能体推理，解决错误积累

**QAgent** (2025.10)
- 论文: https://arxiv.org/abs/2510.08383
- GitHub: https://github.com/LivingFutureLab/QAgent
- 核心: 模块化搜索代理，交互式查询理解

#### 🎯 决策与执行分离

**DeSA** (2025.10)
- 论文: https://arxiv.org/abs/2510.04695
- GitHub: https://github.com/yiding-w/DeSA
- 核心: 质疑结果奖励假设，解耦搜索和回答

**DecEx-RAG** (2025.10)
- 论文: https://arxiv.org/abs/2510.05691
- GitHub: https://github.com/sdsxdxl/DecEx-RAG
- 核心: 显式解耦决策与执行，搜索树过程监督，SFT+DPO

#### 📊 多跳推理专项

**GlobalRAG** (2025.10)
- 论文: https://arxiv.org/abs/2510.20548
- GitHub: https://github.com/CarnegieBin/GlobalRAG
- 核心: 全局规划+忠实执行，仅用8k训练数据

**EKA** (2025.12)
- 论文: https://arxiv.org/abs/2512.20144
- GitHub: https://github.com/yxzwang/EarlyKnowledgeAlignment
- 核心: 早期知识对齐，规划前执行首次检索

#### 🔀 混合检索架构

**RouteRAG** (2025.12)
- 论文: https://arxiv.org/abs/2512.09487
- GitHub: https://github.com/YucanGuo/RouteRAG
- 核心: 文本+图混合检索，灵活选择检索方式

**MARAG-R1** (2025.10)
- 论文: https://arxiv.org/abs/2510.27569
- 核心: 多工具架构（语义+关键词+过滤+聚合）

**Interact-RAG** (2025.10)
- 论文: https://arxiv.org/abs/2510.27566
- 核心: 细粒度检索控制（多面检索+锚定匹配+上下文塑造）

#### ⚡ 效率优化

**TeaRAG** (2025.11)
- 论文: https://arxiv.org/abs/2511.05385
- GitHub: https://github.com/Applied-Machine-Learning-Lab/TeaRAG
- 核心: Token效率优化，检索压缩+推理压缩（IP-DPO）

**Bi-RAR** (2025.11)
- 论文: https://arxiv.org/abs/2511.09109
- 核心: 双向信息距离，多目标RL避免冗长推理链

#### 🎓 自我进化

**EvolveSearch** (2025.05)
- 论文: https://arxiv.org/abs/2505.22501
- 核心: 迭代自进化，RL+SFT协同，无需人工标注

**Search Self-play** (2025.10)
- 论文: https://arxiv.org/abs/2510.18821
- GitHub: https://github.com/Alibaba-Quark/SSP
- 核心: 搜索自我博弈，LLM交替提问解决

**SSRL** (2025.08)
- 论文: https://arxiv.org/abs/2508.10874
- GitHub: https://github.com/TsinghuaC3I/SSRL
- 核心: LLM直接作为搜索引擎

**InfoFlow** (2025.10)
- 论文: https://arxiv.org/abs/2510.26575
- 核心: 奖励密度优化，子问题分解+双代理精炼

---

### 🕸️ 4. 图增强RAG

**GeAR** (2024.12)
- 论文: https://arxiv.org/abs/2412.18431
- 核心: 图扩展技术增强多跳推理

**Agent-G** (Under Review)
- 论文: https://openreview.net/forum?id=g2C947jjjQ
- 核心: 图知识库+非结构化数据融合，包含Critic模块

---

### 🎯 5. 自适应RAG

**Adaptive-RAG** (2024.03) ⭐⭐⭐⭐
- 论文: https://arxiv.org/abs/2403.14403
- 核心: 根据问题复杂度动态选择检索策略

**MBA-RAG** (2024.12)
- 论文: https://arxiv.org/abs/2412.01572
- 核心: Multi-Armed Bandit算法优化检索策略

**CtrlA** (2024.05)
- 论文: https://arxiv.org/abs/2405.18727
- 核心: 置信度评估+动态路由，无需额外训练

**AT-RAG** (2024.10)
- 论文: https://arxiv.org/abs/2410.12886
- 核心: 主题过滤+迭代推理，适合金融等垂直领域

---

### 🔄 6. 纠错型RAG

**CRAG** (2024.01) ⭐⭐⭐⭐⭐
- 论文: https://arxiv.org/abs/2401.15884
- LangGraph教程: https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_crag/
- GitHub: https://github.com/athina-ai/rag-cookbooks
- 核心: 评估检索文档相关性，过滤无关内容或触发网络搜索

**Enterprise Troubleshooting** (2024.12)
- 论文: https://arxiv.org/abs/2412.12006
- 核心: Corrective RAG应用于企业系统故障排查

---

### 🏥 7. 垂直领域应用

**Medical Reasoning** (2024.01)
- 论文: https://arxiv.org/abs/2401.15269
- 核心: 自反思RAG应用于医疗推理，MedQA达到SOTA

**Time Series Analysis** (2024.08)
- 论文: https://arxiv.org/abs/2408.14484
- 核心: 多智能体RAG应用于时间序列分析

**Golden-Retriever** (2024.08)
- 论文: https://arxiv.org/abs/2408.00798
- 核心: 工业知识库高保真RAG，自反思机制

---

## 学习路径

### 🎯 新手入门路径 (5步)

1. **RAG基础** → 阅读 NeurIPS 2020 RAG论文
2. **Agent理论** → 阅读 ReAct (ICLR 2023)
3. **全景理解** → 阅读 Agentic RAG Survey
4. **核心技术** → 阅读 CRAG + Self-RAG
5. **实战练习** → 跑 Athina AI Cookbooks

### 🚀 进阶研究路径

**自适应方向** (优化检索策略):
- Adaptive-RAG → MBA-RAG → CtrlA → AT-RAG

**自反思方向** (提升生成质量):
- Self-RAG → Medical Reasoning → Golden-Retriever

**图增强方向** (知识图谱应用):
- GeAR → Agent-G

**多智能体方向** (复杂任务编排):
- Time Series → MetaGPT → Reflexion

**RL驱动方向** (最前沿):
- Search-R1 → R1-Searcher → StepSearch → LLDS

### 💼 实战应用路径

1. 选定场景（医疗/金融/法律/客服）
2. 阅读垂直领域论文
3. 参考开源项目实现
4. 结合实际数据调优

---

## 技术选型建议

### 🎯 按应用场景选择

**简单QA场景**:
- 开源项目: Athina AI Cookbooks
- 论文参考: Search-R1, R1-Searcher
- 理由: 实现简单，效果稳定

**复杂多跳推理**:
- 开源项目: Hierarchical Multi-Agent RAG
- 论文参考: GlobalRAG, EKA, StepSearch
- 理由: 过程监督+全局规划

**成本敏感场景**:
- 开源项目: Redis Agentic RAG
- 论文参考: ZeroSearch, s3, TeaRAG
- 理由: 降低API成本，提升token效率

**企业级部署**:
- 开源项目: LlamaCloud Demo, Azure GPT-RAG
- 论文参考: QAgent, DeSA, DecEx-RAG
- 理由: 模块化设计，易于集成

**研究创新**:
- 论文参考: LLDS, Interact-RAG, MARAG-R1
- 理由: 前沿技术，高影响力

### 🔬 按研究方向选择

**奖励设计研究**:
- StepSearch → LeTS → HiPRAG → LLDS

**检索策略研究**:
- DeepRetrieval → s3 → MARAG-R1 → RouteRAG

**知识利用研究**:
- IKEA → AutoRefine → R1-Searcher++

**自我进化研究**:
- EvolveSearch → Search Self-play → SSRL

---

## RL-Based RAG研究脉络

```
基础范式层（2025.01-03）
├─ Search-o1: O1推理+搜索
├─ Search-R1: RL for RAG开创性工作 ⭐
└─ R1-Searcher: 两阶段RL框架

奖励机制演进（2025.03-12）
├─ 结果奖励: Search-R1, ReSearch
├─ 过程奖励: StepSearch, LeTS, HiPRAG
├─ 混合奖励: R-Search, AutoRefine
├─ 细粒度奖励: E-GRPO, β-GRPO
└─ 训练稳定性: LLDS 🌟

知识利用策略（2025.05）
├─ 内外协同: IKEA
├─ 精炼策略: AutoRefine
└─ 零搜索: ZeroSearch

决策优化（2025.10）
├─ 解耦架构: DeSA, DecEx-RAG
├─ 模块化: QAgent
└─ 全局规划: GlobalRAG, EKA

检索架构创新（2025.10-12）
├─ 混合检索: RouteRAG, MARAG-R1
├─ 交互式: Interact-RAG
└─ 效率优化: TeaRAG, s3

自我进化（2025.05-10）
├─ 迭代进化: EvolveSearch
├─ 自我博弈: Search Self-play
└─ 内部搜索: SSRL

长视野与真实环境（2025.04-08）
├─ 深度研究: DeepResearcher
└─ 长视野: ASearcher
```

---

## 学习资源

### 📚 课程推荐

**Andrew Ng系列** (DeepLearning.AI):
- Building Agentic RAG with LlamaIndex: https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex/
- AI Agentic Design Patterns with AutoGen: https://www.deeplearning.ai/short-courses/ai-agentic-design-patterns-with-autogen/

**博客系列** (Andrew Ng):
1. How Agents Can Improve LLM Performance
2. Agentic Design Patterns Part 2: Reflection
3. Agentic Design Patterns Part 3: Tool Use
4. Agentic Design Patterns Part 4: Planning
5. Agentic Design Patterns Part 5: Multi-Agent Collaboration

### 📖 教程推荐

- LangGraph CRAG: https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_crag/
- LangGraph Adaptive RAG: https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag/
- LlamaIndex Agentic RAG: https://www.llamaindex.ai/blog/agentic-rag-with-llamaindex-2721b8a49ff6
- Hugging Face Agentic RAG: https://huggingface.co/learn/cookbook/en/agent_rag

---

## 引用格式

### Agentic RAG Survey
```bibtex
@misc{singh2025agenticrag,
  title={Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG},
  author={Aditi Singh and Abul Ehtesham and Saket Kumar and Tala Talaei Khoei},
  year={2025},
  eprint={2501.09136},
  archivePrefix={arXiv},
  primaryClass={cs.AI}
}
```

### Self-RAG
```bibtex
@inproceedings{asai2024selfrag,
  title={Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection},
  author={Akari Asai and Zeqiu Wu and Yizhong Wang and Avirup Sil and Hannaneh Hajishirzi},
  booktitle={ICLR},
  year={2024}
}
```

### ReAct
```bibtex
@inproceedings{yao2023react,
  title={ReAct: Synergizing Reasoning and Acting in Language Models},
  author={Shunyu Yao and Jeffrey Zhao and Dian Yu and Nan Du and Izhak Shafran and Karthik Narasimhan and Yuan Cao},
  booktitle={ICLR},
  year={2023}
}
```

### Search-R1
```bibtex
@article{jin2025searchr1,
  title={Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning},
  author={Jin, Peter Griffin and others},
  journal={COLM},
  year={2025}
}
```

---

## 总结

Agentic RAG正在从学术研究走向工业落地，本文梳理的**6大开源项目**和**60+篇论文**覆盖了：

- ✅ 从基础单智能体到复杂多智能体协作
- ✅ 从通用框架到垂直领域应用
- ✅ 从经典RAG到最前沿的RL驱动搜索
- ✅ 从理论研究到企业级工程实践

### 📊 统计数据

- **开源项目**: 6个（均可直接运行）
- **经典论文**: 7篇（RAG, ReAct, Self-RAG等）
- **RL-Based论文**: 40+篇（2025最前沿）
- **其他前沿论文**: 13篇（图增强、自适应、纠错等）
- **总计**: **60+篇论文 + 6个项目**

### 🎯 建议

- **入门学习**: 从Athina AI Cookbooks开始
- **企业应用**: 参考LlamaCloud文档工作流
- **性能优化**: 看Redis和NVIDIA加速方案
- **垂直领域**: 学习LawGlance等专业应用
- **前沿研究**: 关注Search-R1系列和LLDS

---

**持续更新提醒**: Agentic RAG是2024-2025年最热门的研究方向

**关注渠道**:
- arXiv: cs.AI, cs.CL分类
- 顶会: NeurIPS, ICLR, ACL, EMNLP
- GitHub: LangChain, LlamaIndex相关项目

**更新时间**: 2025年12月

---

> 作者：大模型算法工程师 | 大模型知识分享博主
>
> 如有帮助，欢迎Star和分享！