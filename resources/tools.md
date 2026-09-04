---
type: 资源清单
status: 已发布
level: 通用
topic:
  - Agent
  - 框架工具
---

# AI Agent 开发者工具箱

> **工欲善其事，必先利其器！精选 Agent 开发必备工具**

---

## 🤖 AI 工具配置（核心）

### 💻 代码开发
- **Cursor** ⭐⭐⭐⭐⭐ 最推荐
  - 链接：https://cursor.sh/
  - 用途：AI 辅助编程，写 Agent 代码必备
  - 特点：Claude 集成，代码补全，AI 问答

- **Claude Code**
  - 链接：https://claude.ai/
  - 用途：复杂代码逻辑设计，架构咨询
  - 特点：长上下文，代码理解能力强

### 📚 论文阅读
- **腾讯元宝** ⭐⭐⭐⭐⭐ 最推荐
  - 用途：读论文、总结论文、理解算法原理
  - 特点：中文友好，免费

- **豆包**
  - 用途：快速理解论文核心思想
  - 特点：响应快，免费

- **论文翻译工具**
  - 链接：https://hjfy.top/
  - 用途：整篇论文翻译，保留格式

### 📖 开源项目阅读
- **DeepWiki** ⭐⭐⭐⭐⭐
  - 链接：https://opendeep.wiki/
  - 用途：理解复杂开源项目（如 LangChain 源码）
  - 特点：AI 辅助代码理解

- **ZRead.ai**
  - 链接：https://zread.ai/
  - 用途：快速理解 GitHub 项目

### 💡 Idea 生成与验证
- **GPT-4** / **GPT-o3-mini**
  - 用途：头脑风暴，技术方案设计
  - 配合高质量 Prompt 使用

### 🎨 可视化工具
- **Excalidraw** ⭐⭐⭐⭐⭐
  - 链接：https://excalidraw.com/
  - 用途：绘制系统架构图、Agent 工作流图
  - 特点：免费，简洁，适合技术图

- **Draw.io**
  - 链接：https://draw.io/
  - 用途：复杂系统架构图

### ✍️ 文档润色
- **Grok** (X.AI)
  - 用途：论文润色、降重
  - 特点：生成质量高

- **Claude 4.5**
  - 用途：Paper Review，技术文档审查
  - 特点：逻辑严谨

---

## 📊 数据资源

### 🔍 找数据集
- **GitHub**：搜索 `awesome-{topic}` 或 `{topic}-dataset`
- **Hugging Face Datasets**：https://huggingface.co/datasets
- **ModelScope**：https://modelscope.cn/datasets
- **OpenLabData**：开放科研数据
- **OpenKG**：中文知识图谱数据

### 📄 找论文
- **Hugging Face Papers（趋势榜）**：https://huggingface.co/papers/trending
- **ArXiv**：https://arxiv.org/
- **OpenReview**：https://openreview.net/

### 📖 找综述
- **GitHub搜索技巧**：`Awesome + 关键词`
  - 示例：`Awesome RAG`, `Awesome Agent`, `Awesome LLM`

### 🌍 找行业趋势
- **国外AI趋势**：https://www.theaivalley.com/
- **Bing搜索技巧**：关键词 + `pdf` 找研究报告

---

## 🛠️ Agent 开发技术栈

### 基础框架
- **Transformers**：https://github.com/huggingface/transformers
- **vLLM**：https://github.com/vllm-project/vllm
- **TGI (Text Generation Inference)**：https://github.com/huggingface/text-generation-inference
- **LMDeploy**：https://github.com/InternLM/lmdeploy
- **SGLang**：https://github.com/sgl-project/sglang

### 训练框架
- **DeepSpeed**：https://github.com/microsoft/DeepSpeed
- **FairScale**：https://github.com/facebookresearch/fairscale
- **Megatron-LM**：https://github.com/NVIDIA/Megatron-LM
- **ColossalAI**：https://github.com/hpcaitech/ColossalAI

### 微调工具
- **Llama-Factory** ⭐⭐⭐⭐⭐：https://github.com/hiyouga/LLaMA-Factory
- **Swift**：https://github.com/modelscope/swift
- **Axolotl**：https://github.com/OpenAccess-AI-Collective/axolotl
- **Unsloth**：https://github.com/unslothai/unsloth
- **PEFT**：https://github.com/huggingface/peft

### RAG 技术栈

**RAG 框架**：
- **LangChain** ⭐：https://github.com/langchain-ai/langchain
- **LlamaIndex** ⭐：https://github.com/run-llama/llama_index
- **Haystack**：https://github.com/deepset-ai/haystack
- **RAGFlow**：https://github.com/infiniflow/ragflow
- **Dify**：https://github.com/langgenius/dify

**向量数据库**：
- **Milvus** ⭐：https://github.com/milvus-io/milvus
- **Weaviate**：https://github.com/weaviate/weaviate
- **Pinecone**：https://www.pinecone.io/
- **Chroma**：https://github.com/chroma-core/chroma
- **Qdrant**：https://github.com/qdrant/qdrant
- **FAISS**：https://github.com/facebookresearch/faiss

**Embedding 模型**：
- **BGE** ⭐：https://github.com/FlagOpen/FlagEmbedding
- **E5**：https://github.com/microsoft/unilm/tree/master/e5
- **Sentence-Transformers**：https://github.com/UKPLab/sentence-transformers
- **Instructor**：https://github.com/xlang-ai/instructor-embedding

**文档解析**：
- **MinerU** ⭐：https://github.com/opendatalab/MinerU
- **Unstructured**：https://github.com/Unstructured-IO/unstructured
- **LlamaParse**：https://github.com/run-llama/llama_parse
- **Docling**：https://github.com/DS4SD/docling
- **PyPDF2**：https://github.com/py-pdf/pypdf2

### Agent 技术栈

**Agent 框架**：
- **AutoGPT**：https://github.com/Significant-Gravitas/AutoGPT
- **LangGraph** ⭐：https://github.com/langchain-ai/langgraph
- **CrewAI**：https://github.com/joaomdmoura/crewAI
- **MetaGPT**：https://github.com/geekan/MetaGPT
- **AutoGen** ⭐：https://github.com/microsoft/autogen
- **Swarm**：https://github.com/openai/swarm

**工具调用**：
- **ToolBench**：https://github.com/OpenBMB/ToolBench
- **Gorilla**：https://github.com/ShishirPatil/gorilla
- **ToolLLM**：https://github.com/OpenBMB/ToolLLM

**记忆模块**：
- **Mem0** ⭐：https://github.com/mem0ai/mem0
- **MemGPT**：https://github.com/cpacker/MemGPT
- **LangChain Memory**：内置在 LangChain 中
- **Zep**：https://github.com/getzep/zep

**GUI Agent**：
- **AppAgent**：https://github.com/mnotgod96/AppAgent
- **SeeAct**：https://github.com/OSU-NLP-Group/SeeAct
- **WebShop**：https://github.com/princeton-nlp/WebShop
- **Mind2Web**：https://github.com/OSU-NLP-Group/Mind2Web

### 多模态技术

**视觉理解**：
- **CLIP**：https://github.com/openai/CLIP
- **BLIP**：https://github.com/salesforce/BLIP
- **LLaVA**：https://github.com/haotian-liu/LLaVA
- **Qwen-VL**：https://github.com/QwenLM/Qwen-VL
- **InternVL**：https://github.com/OpenGVLab/InternVL

**OCR 技术**：
- **PaddleOCR** ⭐：https://github.com/PaddlePaddle/PaddleOCR
- **EasyOCR**：https://github.com/JaidedAI/EasyOCR
- **TrOCR**：https://github.com/microsoft/unilm/tree/master/trocr
- **GOT-OCR**：https://github.com/Ucas-HaoranWei/GOT-OCR2.0
- **Rapid_OCR**：https://github.com/RapidAI/RapidOCR

### 部署与监控

**模型服务**：
- **FastAPI** ⭐：https://github.com/tiangolo/fastapi
- **Gradio**：https://github.com/gradio-app/gradio
- **Streamlit**：https://github.com/streamlit/streamlit
- **BentoML**：https://github.com/bentoml/BentoML

**监控工具**：
- **LangSmith** ⭐：https://smith.langchain.com/
- **Weights & Biases**：https://wandb.ai/
- **MLflow**：https://github.com/mlflow/mlflow
- **TensorBoard**：https://github.com/tensorflow/tensorboard

---

## 📚 学习资源

### 🎓 入门教程
- **阿东的大模型实验室**：https://ccn7vpu5l5y8.feishu.cn/wiki/Nh8NwmGpsi0cAykX6OWcHmS3nyc
  - Agent 学习资料汇总
  - 实战项目案例
  - 面试题库

- **WayToAGI 小白入门**：https://waytoagi.feishu.cn/wiki/QPe5w5g7UisbEkkow8XcDmOpn8e
  - 零基础AI入门
  - 系统化学习路径

- **吴恩达 LLM 课程（中文版）**：https://github.com/datawhalechina/llm-cookbook
  - 大模型基础入门
  - Datawhale 出品

### 🎨 宝藏资源库
- **宝藏AI产品知识库**：https://zw73xyquvv.feishu.cn/wiki/V7JxwkK1ti8hjwkqxwAcf0cEngR
- **AI开源产品解决方案库**：https://d.aigclink.ai/

### 📰 技术公众号推荐
- 腾讯技术工程
- 阿里云开发者
- 阿里技术
- 大淘宝技术

---

## 💡 高效学习工作流

### 📖 学习 Agent 的最佳实践

```
Step 1: 读论文理解原理
  工具：腾讯元宝 / 豆包
  目标：理解 ReAct、Reflexion 等架构
  
Step 2: 阅读开源项目源码
  工具：DeepWiki / ZRead.ai
  目标：理解 LangChain、AutoGen 的实现
  
Step 3: 动手实现项目
  工具：Cursor + Claude Code
  目标：完成 3 个简历级项目
  
Step 4: 绘制架构图
  工具：Excalidraw / Draw.io
  目标：能清晰讲解系统设计
  
Step 5: 准备面试
  工具：AgentGuide 面试题库
  目标：掌握高频面试题
```

---

## 🎯 不同场景的工具组合

### 场景1：开发 RAG Agent

```
数据准备：MinerU（文档解析）
向量化：BGE-Embedding
存储：Milvus
框架：LangChain
监控：LangSmith
部署：FastAPI
```

### 场景2：开发 Multi-Agent 系统

```
框架：AutoGen / CrewAI
编排：LangGraph
记忆：Mem0
工具：各类API
监控：LangSmith
部署：FastAPI + Redis
```

### 场景3：学习与研究

```
读论文：腾讯元宝
读源码：DeepWiki
写代码：Cursor
画图：Excalidraw
做笔记：Feishu / Notion
```

---

## 📝 数据获取技巧

### GitHub 搜索技巧
1. **找数据集**：`{topic} dataset`
2. **找综述**：`Awesome {topic}`
3. **找实现**：`{论文名} implementation`
4. **找教程**：`{topic} tutorial Chinese`

### Hugging Face 技巧
1. **找热门论文**：https://huggingface.co/papers/trending
2. **找数据集**：https://huggingface.co/datasets
3. **找模型**：https://huggingface.co/models

---

## 🚀 效率提升建议

### 养成的好习惯
1. ✅ **做笔记**：每学一个知识点，立即记录
2. ✅ **写博客**：学完就写，加深理解
3. ✅ **写代码**：理论学完立即实践
4. ✅ **问AI**：遇到问题先问元宝/ChatGPT

### 知识平权时代的三大能力
1. **搜索能力**：快速找到高质量资源
2. **动手能力**：快速实现和验证想法
3. **思考能力**：不盲从，有自己的判断

> **核心理念**：选择大于努力！选对工具，效率翻倍！

---

## 🔗 相关资源

- [Agent 开发框架推荐](./agent/frameworks.md)
- [必读论文清单](./31篇LLM的经典论文速读版清单.md)
- [优质博客与教程](./learning-resources.md)
- [开源项目推荐](../projects/README.md)

