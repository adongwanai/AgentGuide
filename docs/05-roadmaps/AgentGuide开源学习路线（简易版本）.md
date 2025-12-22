
# 🚀 大模型 / Agent 全栈学习路线

<img src="https://youke1.picui.cn/s1/2025/12/03/692f10fac4dcf.png" height="700" alt="从零到Offer的完整路径">


![1764693346990.png](https://youke1.picui.cn/s1/2025/12/03/692f15359ede7.png)
![1764693461348.png](https://youke1.picui.cn/s1/2025/12/03/692f15a7e331a.png)


**核心逻辑**：先明确目标岗位→按岗位路线分阶学习→项目实战→简历包装→面试冲刺，全程以「拿 Offer」为导向，避免盲目学习。


### **阶段 1：定位与路线规划（1 周）**

**优先资源**：🎯 **AgentGuide 求职宝典**🌐 GitHub：adongwanai/AgentGuide《Agent 求职通关秘籍》：
![1764692344870.png](https://youke1.picui.cn/s1/2025/12/03/692f114d004bc.png)

- **岗位分轨**：明确「算法岗（10-15 周）」vs「开发岗（8-12 周）」核心差异（算法岗重策略优化 / RL，开发岗重工程落地 / RAG）。
- **路线拆解**：
    - 算法岗：基础（Python/ML）→ LLM 原理（Transformer）→ Agent 范式（ReAct/CoT）→ RAG 进阶→ RLHF / 对齐→ 论文复现。
    - 开发岗：基础（Python / 数据库）→ LLM 部署（API / 本地）→ RAG 全流程→ Agent 框架（Dify/Coze）→ 系统集成→ 项目上线。
- **求职工具**：简历模板（算法 / 开发岗适配）、面试高频考点（如「Agent 与传统软件的区别」「RAG 性能优化」）、谈薪 / HR 攻略，提前规避求职坑。
    
    **产出**：确定目标岗位，制定周度学习计划，标注「面试必考点」。

### **阶段 2：基础能力夯实（2-4 周）**

**算法岗核心**：

1. **Python 与 ML 基础**：
    - 推荐：《Python 编程：从入门到实践》+ LeetCode 中等题（数组 / 字符串 / 动态规划，面试基础）。
2. **LLM 底层原理**：
    - 推荐：
        
        🧠 **nanoGPT**（karpathy/nanoGPT）：300 行代码吃透 Transformer 注意力机制、词嵌入、生成逻辑，理解「大模型黑箱」。
    ![1764692441549.png](https://youke1.picui.cn/s1/2025/12/03/692f11ab92b4c.png)
        
        📚 **吴恩达大模型系列课程中文版**（datawhalechina/llm-cookbook）：AI 教父课程本地化，覆盖预训练 / 微调核心概念。
![1764692379509.png](https://youke1.picui.cn/s1/2025/12/03/692f1173c2e7b.png)        
        **开发岗核心**：
3. **Python 与工程基础**：
    - 推荐：FastAPI/Flask（API 开发）、Docker（环境封装）、Git（版本控制），掌握「代码可复用性」。
4. **LLM 应用入门**：
    - 推荐：
        
        🎓 **AI-Guide-and-Demos-zh_CN**（Hoper-J）：从 OpenAI API 调用→ 本地部署（Llama 3/Qwen3）→ 基础微调，提供 Colab 在线环境，零基础可上手。
        ![1764692512488.png](https://youke1.picui.cn/s1/2025/12/03/692f11f2c02f2.png)
        💻 **动手学大模型**（Lordog/dive-into-llms）：上交大开源教程，含 GUI Agent、数学推理等实战，适合快速练手。![1764692568091.png](https://youke1.picui.cn/s1/2025/12/03/692f122ea5721.png)
    
        **产出**：算法岗能复现简单 Transformer 模块；开发岗能独立调用 LLM API 并部署本地模型。

### **阶段 3：核心技术突破（算法岗 6-8 周 / 开发岗 4-6 周）**

#### **模块 A：Agent 核心（算法 / 开发岗均需）**

- **优先资源**：
    
    🔫 **Hello-Agents**（datawhalechina/hello-agents）
    
    《AI 原生 Agent 从 0 到 1 构建指南》：
    - 算法岗重点：拆解 ReAct、Plan-and-Solve 等经典范式，实现单智能体决策逻辑，进阶多智能体通信（MCP 协议）与 Agentic RL 训练。
    - 开发岗重点：用 Coze/Dify 低代码平台快速搭建「智能旅行助手」，再尝试自研轻量 Agent 框架（如任务调度 + 工具调用）。
    - ![1764692676925.png](https://youke1.picui.cn/s1/2025/12/03/692f1299dfa34.png)
- **实战项目**：
    - 算法岗：复现「论文检索 Agent」（结合 RAG 实现学术文献自动摘要与引用）。
    - 开发岗：搭建「赛博小镇」多智能体系统（模拟居民互动，涉及 Agent 协作与记忆机制）。

#### **模块 B：RAG 技术（Agent 知识底座）**

- **优先资源**：
    
    🔍 **All-in-RAG**（datawhalechina/all-in-rag）
    ![1764692753750.png](https://youke1.picui.cn/s1/2025/12/03/692f12e4f3228.png)
    《RAG 全栈通关手册》：
    - 算法岗重点：向量索引优化（如 IVF-PQ 量化）、混合检索策略（BM25 + 向量检索）、检索评估指标设计（如 MRR、Hit@k）。
    - 开发岗重点：数据加载→文本分块→Milvus 部署→检索接口开发，完成「智能美食推荐 RAG 系统」（300 + 食谱匹配）。
- **产出**：算法岗能优化 RAG 检索精度；开发岗能独立搭建生产级 RAG 问答系统。

#### **模块 C：微调技术（算法岗核心 / 开发岗可选）**

- **算法岗优先**：
    
    ⚡ **Unsloth**（unslothai/unsloth）：2 倍训练速度 + 70% 显存节省，用 14GB GPU 训练 20B 模型，适配 GPT-OSS/Qwen3，原生支持 GRPO 强化学习（提升 Agent 工具调用精度）。
    
    🚀 **LLaMA-Factory**（hiyouga/LLaMA-Factory）：100 + 模型统一微调框架，覆盖 SFT/DPO/GRPO，可实现多模态微调（如 Qwen3-VL 图文检索）。
    ![1764692883236.png](https://youke1.picui.cn/s1/2025/12/03/692f1365caa80.png)
- **开发岗可选**：用 LLaMA-Factory Web UI 完成 LoRA 微调，适配特定场景（如客服问答）。
- **数据支撑**：
    
    📊 **Easy-Dataset**（ConardLi/easy-dataset）：自动处理 PDF/Markdown 文档，生成带 CoT 的问答数据，导出 Alpaca 格式直接对接微调框架，告别手动标注。
    
    **产出**：算法岗能完成 LLM 策略微调；开发岗能基于开源数据快速适配模型。

### **阶段 4：项目实战与简历包装（2-3 周）**

1. **简历项目打磨**：
    - 参考 AgentGuide 的「项目杀器」：
        - 算法岗：突出「论文检索 Agent」（技术点：ReAct 范式 + Graph RAG+GRPO 强化学习）、「大模型对齐实验」（DPO 训练）。
        - 开发岗：突出「旅行规划 Multi-Agent」（技术点：Dify 低代码 + Milvus 向量库 + Docker 部署）、「企业知识库 RAG 系统」（技术点：混合检索 + 权限控制）。
    - 每个项目标注「技术栈 + 解决问题 + 量化指标」（如「检索准确率提升 25%」「部署成本降低 40%」）。
2. **实战项目落地**：
    - 算法岗：复现顶会 Agent 论文（如 AutoGPT、MetaGPT）核心模块，在 GitHub 开源并撰写技术博客。
    - 开发岗：将 RAG+Agent 系统部署到云服务器（如阿里云 ECS），提供公开 Demo 链接。
        
        **产出**：适配目标岗位的简历 + 2-3 个可演示的实战项目。

### **阶段 5：面试冲刺（1-2 周）**

1. **高频考点突击**：
    - 算法岗：Agent 范式对比（ReAct vs Plan-and-Solve）、RAG 性能调优（分块策略 / 向量量化）、RLHF 原理（DPO/PPO 区别）。
    - 开发岗：LLM 部署方案（vLLM/Ollama）、向量数据库选型（Milvus vs Pinecone）、系统高可用设计（缓存 / 降级）。
2. **模拟面试**：
    - 用 AgentGuide 的「面试题库」自测，重点准备「项目复盘」（如「你在 RAG 系统中遇到的最大问题是什么？如何解决？」）。
3. **谈薪与 HR 攻略**：
    - 参考 AgentGuide 的「谈薪技巧」，明确行业薪资范围，突出项目价值（如「搭建的 RAG 系统为公司节省 50 万标注成本」）。
        
        **产出**：面试通过率提升 30%+，拿到目标 Offer。

### **学习建议**

1. **按岗位聚焦**：算法岗深耕原理与策略，开发岗侧重工程与落地，避免「全而不精」。
2. **边学边输出**：每完成一个模块，在 GitHub 提交代码、写技术笔记（如 CSDN / 知乎），形成个人品牌。
3. **紧跟社区**：关注项目更新（如 Unsloth 新增 TTS 支持、LLaMA-Factory 适配 Llama 4），面试时体现技术敏锐度。

**总结**：以 AgentGuide 为路线图，先定岗位→再夯基础→攻核心技术→做实战项目→冲面试，全程目标明确，效率拉满！按此路线执行，8-15 周即可具备大模型 / Agent 领域求职硬实力～