---
type: 求职指南
status: 已发布
level: 入门
topic:
  - 面试求职
---


## 一、秋招核心考察点
秋招更注重**综合能力**，相比实习，“抽奖”机会减少。以下是核心考察点：
1. **优先级**：
   - **一作论文**：对研究岗尤为重要，证明学术能力。
   - **垂直领域实习**：与目标岗位相关的实习经历最有说服力。
   - **项目经历**：次要但需体现技术深度和业务理解。
2. **工程能力**是重点：
   - 除顶尖算法大佬（如ACM金牌）、顶会作者外，秋招更看重工程实践能力，而非纯算法竞赛水平。
   - 建议围绕热门方向（如大模型训练、RAG、Agent等）准备，展现对业务场景的理解和落地能力。

---
## 二、项目经历准备
项目经历是简历和面试的核心，需根据是否有实习经历分别准备：
#### 1. 无实习经历
- **简历内容**：
  - 聚焦**实验室横向项目**，优先包含**RAG**（Retrieval-Augmented Generation）或**Training相关**项目（如LoRA、SFT、RLHF）。
  - 项目覆盖热门方向：**智能客服**、**文档解析**、**代码生成**等，体现特定领域数据结构处理能力。
  - 建议：确保项目经历全面，突出Training经验，增加“抽奖”机会（被选中面试）。
- **注意事项**：
  - 避免泛泛而谈，项目需体现技术深度，如数据合成处理、模型微调的具体实现。
  - 项目应尽量贴合目标岗位的业务场景。
----

#### 2. 有实习经历（业务岗）
- **简历内容**：
  - 项目和实习经历应与**热门方向**相关。
  - **避免重复**：如果实习和项目都涉及RAG，只写其一，突出最具代表性的经历。
  - **专有领域数据处理**：避免写烂大街的开源数据集（如GSM8K），而是突出对特定业务数据的处理经验，体现业务理解。
- **论文处理**：
  - 论文仅在简历“学校”一栏单列一行（如“在投”），避免占用过多空间。
  - 业务算法岗更看重工程能力，研究经历描述应精简。
- **总结**：
  - 项目和实习经历需**覆盖面广**（如Post-training、Agent）。
----

  - 突出**热门方向**（如RL、Agent Memory等）。
  - 强调**专有领域数据处理**和**Training经验**。
  - 保持简历**一页**，精简重复内容，突出核心亮点。

####  3. 推荐的开源项目工具
以下开源项目在业务岗中应用广泛，可根据目标公司选择性使用：

#### 基础框架与平台
- **大模型框架**：Transformers、vLLM、TGI、LMDeploy、SGLang
- **训练框架**：DeepSpeed、FairScale、Megatron-LM、ColossalAI
- **推理优化**：TensorRT-LLM、OpenVINO、ONNX Runtime、BitsAndBytes

#### 模型微调与训练
- **微调工具**：Llama-Factory、Swift、Axolotl、Unsloth、PEFT
- **数据处理**：DataTrove、FineWeb、RedPajama、Dolma
- **评估框架**：OpenCompass、LM-Eval-Harness、HELM、BigBench

#### RAG相关技术
----

- **RAG框架**：LangChain、LlamaIndex、Haystack、RAGFlow、Dify
- **向量数据库**：Milvus、Weaviate、Pinecone、Chroma、Qdrant、FAISS
- **Embedding模型**：BGE、E5、Sentence-Transformers、Instructor
- **文档解析**：MinerU、Unstructured、LlamaParse、PyPDF2、Docling

#### Agent相关技术
- **Agent框架**：AutoGPT、LangGraph、CrewAI、MetaGPT、AutoGen、Swarm
- **工具调用**：ToolBench、Gorilla、ToolLLM、ReAct、Function Calling
- **记忆模块**：Mem0、MemGPT、LangChain Memory、Zep
- **GUI Agent**：AppAgent、SeeAct、WebShop、Mind2Web
----

#### 多模态技术
- **视觉理解**：CLIP、BLIP、LLaVA、Qwen-VL、InternVL、MiniGPT-4
- **图像生成**：Stable Diffusion、DALL-E、Midjourney API、ComfyUI
- **视频理解**：VideoChatGPT、Video-LLaVA、LLaMA-VID、VideoChat
- **OCR技术**：PaddleOCR、EasyOCR、TrOCR、GOT-OCR、Rapid_OCR


#### 部署与监控
- **模型服务**：FastAPI、Flask、Gradio、Streamlit、BentoML
- **容器化**：Docker、Kubernetes、Helm Charts
- **监控工具**：MLflow、Weights & Biases、TensorBoard、Neptune
- **API管理**：OpenAI API、Anthropic API、Together API、Replicate

#### 数据处理与存储
- **数据库**：PostgreSQL、Redis、MongoDB、ElasticSearch
- **数据处理**：Pandas、Polars、Dask、Apache Spark
- **特征工程**：Scikit-learn、Feature-engine、Category Encoders

---

## 三、热门方向准备
为应对秋招，以下5个方向需重点关注：

1. **多模态方向**：
   - 结合图像、文本等多模态数据处理技术
   - 关注多模态理解、生成、推理等核心技术

2. **RAG方向**：
   - **多模态RAG**、**GraphRAG**、**AI搜索**等
   - 检索增强生成技术的各种变体和优化

3. **Reasoning方向**：
   - **Deep Research RL**（工具调用）、**Long COT Reasoning**（如M2-Reasoning）
   - 注意：Reasoning方向实习需求多，但秋招可能因落地难度较大，邀面机会较少，需谨慎选择
---

4. **Agent方向**：
   - **Agent Memory**：如Mem0（注意复现问题，谨慎使用）
   - **Agent RL**：强化学习在智能体中的应用
   - 关注**DeepResearch**相关技术（如MiroMind Agent、CognitiveKernel-Pro、WebShaper、Owl、AgentOrchestra）
   - 评估Benchmark：**GAIA**、**Human Last Exam**
   - **GUI Agent**：参考GUI TARs等论文

5. **上下文工程方向**：
   - 长上下文处理、上下文压缩、上下文优化等技术
   - 提示工程、上下文学习等相关技术

**建议**：优先选择**RAG方向**和**Agent方向**，因岗位需求多且较易落地。

---

## 四、LeetCode与算法准备
1. **笔试**：
   - 难度因公司而异（如美团难、PDD和小红书较简单），需靠运气和扎实准备。
   - 建议刷**LeetCode Hot 100** + **Interview 150**，覆盖常见题型，足以应对大部分笔试。
2. **面试手撕代码**：
   - 业务岗通常要求掌握Hot 100，追求高目标（如SSP offer）需更深入准备。
   - **高质量刷题建议**：
     - **看评论区**：通过后研究大佬解法，学习优雅代码和数学证明。
     - **不死磕**：10-15分钟无思路即查缺补漏，快速提升知识点。
     - **看讲解视频**：推荐B站“灵茶山艾府”，通过典型例题巩固专题知识。


---

## 五、八股文准备
八股文（技术面常见问题）是面试重点，需根据实习经历差异化准备：
1. **有实习经历**：
   - 重点描述**实习项目故事**，结构化回答：
     - **问题点**：项目中遇到的挑战。
     - **你的贡献**：具体优化措施。
     - **优化成果**：指标从XX提升到XX。
   - 避免废话，突出技术深度和业务价值。
2. **无实习经历**：
   - 聚焦**Training相关八股**，如SFT、RLHF经验，回答中突出：
     - 参数设置的理由和优化动机。
     - 实战经验（如通信优化、训练稳定性）。
   - 通过八股展现工程能力和技术理解，弥补实习经历不足。
---

## 六、其他建议
1. **论文**：
   - 写一篇论文并投稿（如OpenReview），门槛较低且可记录在简历（标“在投”）。
   - 非研究岗无需过多描述论文，简历空间留给项目和实习。
2. **简历优化**：
   - **一页原则**：删除重复或次要经历，突出核心亮点。
   - 结构清晰，分模块展示教育背景、实习经历、项目经历、技术技能。
3. **时间规划**：
   - 提前准备（1-2个月刷题，3-4个月积累项目）。
   - 持续关注热门方向（如R1蒸馏、Agent），确保简历与时俱进。

---

## 七、总结
- **核心目标**：展现**工程能力**和**业务理解**，通过项目和实习经历体现对热门方向（如RAG、Agent）的掌握。
- **重点准备**：
  - **项目**：覆盖RAG、Training，突出专有领域数据处理。
  - **算法**：刷Hot 100+Interview 150，注重高质量刷题。
  - **八股**：有实习讲项目故事，无实习突出Training经验。
- **心态**：秋招竞争激烈，保持积极心态，多投简历，机会总会来。

祝大家秋招顺利，拿到心仪offer！

---

### 🔥 系列预告：大模型细分方向深度解析

**新开系列：《大模型XX到底值不值得做？怎么做？》**

#### 🗓️ 系列规划：
1. **AI Infra方向** - 大模型基础设施，赛道深度剖析
2. **多模态方向** - 视觉+语言融合，未来趋势抢先布局  
3. **RAG方向** - 检索增强生成，落地性最强的技术方向
4. **Agent方向** - 智能体技术，最具想象空间的创新领域
5. **Reasoning方向** - 推理能力突破，AGI路径核心技术
6. **上下文工程** - 长文本处理，大模型应用关键技术


- **📢 关注作者，第一时间获取系列更新！**

- #互联网大厂 #大模型 #算法 #秋招 #校招 #RAG #Agent #多模态 #AI基础设施

