✴️ 全网最全最优质的上下文工程资源合集

> **上下文工程（Context Engineering）**：一门将不断变化的信息宇宙中最相关内容，精心筛选并放入有限上下文窗口的艺术与科学。本质是"熵减"过程——将高熵的原始上下文转化为机器可理解的低熵表示。

---

## 🎯 **一、核心概念入门（建立框架）**

### 1. **The New Skill in AI is Not Prompting, It's Context Engineering**

- **作者**：Philipp Schmid（Google DeepMind 高级AI工程师）
- **链接**：https://www.philschmid.de/context-engineering
- **核心价值**：★★★★★（入门必读）
- **内容要点**：
    - **首次系统定义**：上下文工程 ≠ 提示工程，是动态系统而非静态字符串
    - **六层上下文模型**：指令/提示、短期记忆、长期记忆、RAG检索、工具定义、结构化输出
    - **经典案例对比**：简单Demo vs 魔法Agent的区别在于上下文质量
    - **核心公式**：正确信息 + 正确工具 + 正确时机 + 正确格式 = 有效Agent
- **适用人群**：所有开发者，快速建立术语体系和框架认知

### 2. **上下文工程技术总结（中文叙事版）**

- **包含资源**：
    - 论文：https://arxiv.org/pdf/2510.26493
    - 仓库：https://github.com/GAIR-NLP/Context-Engineering-2.0
    - 微信文章：https://mp.weixin.qq.com/s/KbviOJ6q-K4ik_wzsUs2dw
- **核心价值**：★★★★★（理论深度）
- **内容要点**：
    - **历史视角**：上下文工程可追溯至20多年前，非LLM时代新创
    - **熵减理论**：RAG（文档→片段）、CoT（模糊推理→逻辑链）、多模态（高熵数据→共享向量空间）
    - **2.0核心策略**：
        - 文本处理：时间戳标记、功能/语义标签、QA对压缩、层级笔记
        - 选择策略：语义相关性、逻辑依赖、时效性、信息去重、用户偏好
- **适用人群**：研究者，需要理论深度和演进脉络

---

## 🛠️ **二、实战方法论（可落地的流程）**

### 3. **Context Engineering for Agents（LangChain官方）**

- **作者**：Harrison Chase（LangChain联合创始人兼CEO）
- **链接**：https://blog.langchain.com/context-engineering-for-agents/
- **配套仓库**：https://github.com/langchain-ai/how_to_fix_your_context
- **核心价值**：★★★★★（实践框架）
- **四大策略框架**：
    1. **Write Context（写入）**：Scratchpads、短期/长期记忆、LangMem
    2. **Select Context（选择）**：动态检索、工具筛选（BigTool）、RAG优化
    3. **Compress Context（压缩）**：总结（递归/层级）、剪枝（时间/令牌）
    4. **Isolate Context（隔离）**：多Agent、沙盒环境、状态对象
- **技术栈支持**：LangGraph + LangSmith 闭环测试
- **适用人群**：使用LangChain生态的开发者，有完整代码示例

### 4. **12-Factor Agents（工程化原则）**

- **作者**：Dex Horthy（HumanLayer创始人）
- **链接**：https://github.com/humanlayer/12-factor-agents
- **核心价值**：★★★★★（生产级方法论）
- **12条核心原则**（类比12-Factor App）：
    - Factor 1：自然语言→工具调用
    - Factor 2：拥有你的提示词
    - **Factor 3：拥有你的上下文窗口**（最关键）
    - Factor 4：工具即结构化输出
    - Factor 5-12：状态管理、暂停/恢复、人机协同、控制流、错误处理、小型聚焦、多渠道、无状态Reducer
- **核心观点**：
    - 80%的Agent是确定性代码 + 20% LLM调用
    - 不要盲目信任框架，要拥有核心组件
    - 成本呈二次增长，必须优化
- **适用人群**：需要构建生产级Agent的工程师

### 5. **phodal/build-agent-context-engineering（中文最全实战）**

- **作者**：phodal（ThoughtWorks资深架构师）
- **链接**：https://github.com/phodal/build-agent-context-engineering
- **核心价值**：★★★★★（中文最系统）
- **四大学习路径**：
    1. **结构化提示词工程**：输入/输出结构化、链式设计、路由分发
    2. **上下文工程与RAG**：关键词+语义+图检索、查询改写、HyDE、重排序
    3. **工具系统设计**：语义清晰、无状态、原子性、最小权限、MCP协议
    4. **Agent规划与多Agent**：预先分解 vs 交错分解、记忆系统、自我完善
- **经典案例**：GitHub Copilot上下文优先级排序、Cursor Rule设计
- **适用人群**：中文开发者，从零到一构建完整Agent系统

---

## 🔬 **三、问题诊断与修复（避坑指南）**

### 6. **How Contexts Fail—and How to Fix Them**

- **作者**：Drew Breunig（独立研究者）
- **链接**：https://www.dbreunig.com/2025/06/22/how-contexts-fail-and-how-to-fix-them.html
- **核心价值**：★★★★★（反面教材+对策）
- **四种上下文失败模式**：
    1. **Context Poisoning（中毒）**：幻觉错误被反复引用
        - 案例：Gemini玩Pokémon时目标被污染，陷入死循环
    2. **Context Distraction（分心）**：过长上下文导致重复历史动作
        - 数据：Llama 3.1 405B在32K时正确率开始下降
    3. **Context Confusion（混淆）**：无关工具/信息干扰响应
        - 案例：Berkeley测试显示所有模型在多工具场景下准确率下降
    4. **Context Clash（冲突）**：多轮对话中信息矛盾
        - 数据：分片提示导致o3性能从98.1%跌至64.1%
- **修复策略**：动态工具加载、上下文隔离、总结压缩
- **适用人群**：调试Agent系统的开发者

### 7. **Practical Tips on Building LLM Agents**

- **作者**：Paras Chopra（Lossfunk创始人）
- **链接**：https://letters.lossfunk.com/p/practical-tips-on-building-llm-agents
- **核心价值**：★★★★☆（一线经验+成本考量）
- **6大实战洞察**：
    1. **任务切分**：每个原子任务≤10-15分钟（METR数据：成功率~90%）
    2. **单一LLM+长上下文**：避免RAG碎片化，完整文件入上下文
    3. **验证系统**：每步明确成功/失败标准，防止错误累积
    4. **重复关键信息**：定期重复待办列表，对抗"健忘症"
    5. **工具读写**：让LLM自主构建上下文，而非预填充
    6. **缓存策略**：成本呈二次增长（50轮=$2.5，100轮=$100），必须命中KV缓存降至1/10
- **真实指标**：Cline错误率从20%降至<5%（用LLM生成diff取代fast-diff模型）
- **适用人群**：产品经理、创业者、需要成本优化的开发者

---

## 🏢 **四、企业实战案例（生产环境经验）**

### 8. **Context Engineering for AI Agents: Lessons from Building Manus**

- **作者**：季逸超（Manus联合创始人）
- **链接**：https://manus.im/zh-cn/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus
- **核心价值**：★★★★★（企业级坑点）
- **核心经验**：
    - **KV缓存优化**：只追加不修改，命中率提升10倍，成本降至1/10
    - **工具遮蔽**：动态显示/隐藏工具，避免上下文混淆
    - **文件系统作为上下文**：让Agent读写Memory文件，持久化信息
    - **复述操控注意力**：重复关键信息到上下文末尾，提升模型关注度
    - **保留错误信息**：不要急于清理错误，LLM需要错误上下文来自我纠正
    - **避免少样本陷阱**：Few-shot过多会偏离当前任务
- **适用人群**：长任务Agent优化、需要控制延迟和成本的团队

### 9. **Effective Context Engineering for AI Agents（Anthropic官方）**

- **链接**：https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
- **核心价值**：★★★★★（Claude团队实践）
- **官方定义**："上下文工程是将不断变化的信息宇宙中最相关内容，精心筛选并放入有限上下文窗口的艺术与科学"
- **三大维度**：
    - **有效上下文**：提示词、工具定义、Few-shot示例
    - **即时检索**：动态工具调用、RAG增强
    - **长任务管理**：上下文压缩、笔记系统、子Agent隔离
- **最佳实践**：管理有限注意力预算、避免上下文腐烂、数百轮对话管理
- **适用人群**：使用Claude系列模型的开发者

### 10. **Brainwash Your Agent: How We Keep the Memory Clean**

- **作者**：CAMEL-AI团队
- **链接**：https://www.camel-ai.org/blogs/brainwash-your-agent-how-we-keep-the-memory-clean
- **核心价值**：★★★★☆（内存管理技巧）
- **核心概念**："脑洗"Agent = 主动清理内存，丢弃噪音
- **三大技术**：
    1. **上下文总结**：提取关键组件、自动触发压缩
    2. **工作流内存**：记录过去策略以复用
    3. **工具输出缓存**：外部存储大输出，避免上下文膨胀
- **收益**：长期任务保持焦点、减少令牌消耗、避免侧任务干扰
- **挑战**：信息丢失风险（需要权衡）
- **适用人群**：长工作流Agent开发者、开源贡献者

---

## 📚 **五、学术综述与前沿研究**

### 11. **A Survey of Context Engineering for Large Language Models**

- **论文**：https://arxiv.org/pdf/2507.13334
- **仓库**：https://github.com/Meirtz/Awesome-Context-Engineering
- **核心价值**：★★★★★（最全综述）
- **研究规模**：分析1400+篇论文，160+页
- **内容框架**：
    - **核心机制**：检索增强、上下文处理、记忆管理
    - **优势分析**：性能提升、资源优化、减少幻觉
    - **改进方向**：效率优化、评估标准、长上下文挑战
- **分类体系**：基础概念、内存系统、RAG、长上下文、结构化数据、自生成上下文、Agent通信、评估
- **适用人群**：研究者、需要全景视野的架构师

### 12. **Context Engineering 2.0: The Context of Context Engineering**

- **论文**：https://arxiv.org/pdf/2510.26493
- **仓库**：https://github.com/GAIR-NLP/Context-Engineering-2.0
- **核心价值**：★★★★☆（概念演进）
- **核心贡献**：
    - 提出**上下文工程2.0**概念（Agent中心智能）
    - 将CE定位为**熵减过程**的理论框架
    - 策略分类：文本处理（时间戳/标签/QA/层级）+ 选择策略（语义/逻辑/时效/去重/偏好）
- **引用网络**：连接众多相关工作，建立知识图谱
- **适用人群**：理论研究者、需要引文脉络的学术人员

---

## 💻 **六、开源工具与代码仓库**

### 13. **davidkimai/Context-Engineering**

- **链接**：https://github.com/davidkimai/Context-Engineering
- **Stars**：4.1K+（社区广泛认可）
- **核心价值**：★★★★★（最佳学习路径）
- **内容模块**：
    - 基础学习路径：从原子概念到元递归
    - 可操作模板：数学模型、RAG架构、记忆设计
    - 进阶示例：工具集成、上下文修剪、认知工具
    - 上下文协议：标准化规范
    - Agent系统：完整架构参考
- **引用研究**：ICML、NeurIPS前沿成果
- **适用人群**：需要深入符号机制和优化的开发者

### 14. **WakeUp-Jin/Practical-Guide-to-Context-Engineering**

- **链接**：https://github.com/WakeUp-Jin/Practical-Guide-to-Context-Engineering
- **核心价值**：★★★★☆（实践指南）
- **七类上下文分解**：系统提示、用户输入、短期记忆、长期记忆、RAG、工具输出、结构化输出
- **内容覆盖**：基础技术、管理策略、Agent架构、评估方法
- **特色**：鼓励社区贡献、项目实践案例
- **适用人群**：需要结构化导航的开发者

### 15. **ginobefun/agentic-design-patterns-cn**

- **链接**：https://github.com/ginobefun/agentic-design-patterns-cn/tree/main
- **核心价值**：★★★★☆（设计模式）
- **内容**：《Agentic Design Patterns》中英对照译本
- **模式库**：提示链、反思、工具使用、多Agent、记忆管理、协议设计、异常处理、RAG集成、生产模式
- **上下文工程关联**：通过模式支持连续性和上下文管理
- **适用人群**：AI工程师、架构师、产品经理

---

## 🎥 **七、视频与多媒体资源**

### 16. **Context Engineering: The Outer Loop**

- **作者**：Hammad Bashir（Chroma CTO）
- **平台**：YouTube
- **时长**：30分钟
- **核心价值**：★★★★☆（可视化演示）
- **内容**：如何用向量数据库构建外循环动态上下文
- **技术栈**：Chroma + 向量检索 + 动态上下文构建
- **适用人群**：视觉学习者、需要实操演示的开发者

### 17. **AI Engineer World's Fair Talk: 12-Factor Agents**

- **链接**：https://www.youtube.com/watch?v=8kMaTybvDUw
- **演讲者**：Dex Horthy
- **核心价值**：★★★★☆（现场分享）
- **内容**：12-Factor Agents方法论现场讲解
- **适用人群**：参会者、需要系统理解的学习者

---

## 📖 **八、电子书与完整指南**

### 18. **The Context Engineering Guide**

- **出版方**：Weaviate
- **链接**：https://weaviate.io/ebooks/the-context-engineering-guide
- **核心价值**：★★★★☆（完整eBook）
- **章节覆盖**：
    - Agent架构设计
    - 查询增强技术
    - 检索策略优化
    - 内存系统构建
    - 工具集成方案
    - 生产级模式
- **特色**：实用建筑模式、可靠性保障
- **适用人群**：需要系统化学习的开发者

---

## 🎯 **学习路径推荐**

### **入门路径（1-2周）**

1. 阅读 Philipp Schmid 的概念文章（建立框架）
2. 学习 Drew Breunig 的失败模式（避坑）
3. 浏览 LangChain 的四大策略（建立方法论）

### **进阶路径（2-4周）**

4. 深入 phodal 的中文实战仓库（系统实践）
5. 研读 12-Factor Agents（工程化原则）
6. 学习 Manus/Anthropic 的企业案例（生产经验）

### **专家路径（1-3个月）**

7. 研读 1400+论文综述（全景视野）
8. 实践 davidkimai 的完整学习路径（深度优化）
9. 贡献开源项目（社区参与）

---

## 📊 **核心技术要点速查**

### **上下文工程三大支柱**

1. **信息检索**：RAG、混合检索、Agentic检索
2. **记忆管理**：短期/长期记忆、反思机制
3. **工具编排**：MCP协议、工具路由、并行调用

### **生产环境最佳实践**

- ✅ 任务原子化（10-15分钟粒度）
- ✅ 使用KV缓存（只追加不修改）
- ✅ 建立验证系统（每步明确成功/失败）
- ✅ 工具单一职责（原子性+语义清晰）
- ✅ 上下文压缩与隔离（总结+剪枝+沙盒）
- ❌ 避免上下文中毒和冲突
- ❌ 防止过度依赖长上下文

### **成本优化关键指标**

- 多轮Agent成本呈**二次增长**（50轮=$2.5，100轮=$100）
- KV缓存命中可降低**90%成本**
- 上下文总结可减少**60-80%令牌**

---

## 🌟 **资源价值矩阵**

|资源|理论深度|实践性|代码质量|适合人群|
|---|---|---|---|---|
|Philipp Schmid文章|⭐⭐⭐|⭐⭐⭐⭐|-|所有人|
|LangChain博客|⭐⭐⭐|⭐⭐⭐⭐⭐|⭐⭐⭐⭐⭐|开发者|
|12-Factor Agents|⭐⭐⭐⭐|⭐⭐⭐⭐⭐|⭐⭐⭐⭐|工程师|
|phodal仓库|⭐⭐⭐⭐|⭐⭐⭐⭐⭐|⭐⭐⭐⭐|中文开发者|
|Drew Breunig文章|⭐⭐⭐⭐|⭐⭐⭐⭐⭐|-|调试者|
|1400+论文综述|⭐⭐⭐⭐⭐|⭐⭐|-|研究者|
|davidkimai仓库|⭐⭐⭐⭐|⭐⭐⭐⭐|⭐⭐⭐⭐|深度学习者|
|Manus案例|⭐⭐⭐|⭐⭐⭐⭐⭐|⭐⭐⭐|企业团队|

---

**这份合集代表了上下文工程领域（截至2025年）的最新进展，涵盖从理论基础到生产实践的完整知识体系！** 🚀