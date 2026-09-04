---
type: 求职指南
status: 已发布
level: 高阶
topic:
  - 面试求职
  - RAG
  - Agent
---

# Agentic RAG / Deep Research 彻底备战手册（补充模块对应版）
> **本手册定位**：面向 AI Agent 求职与面试准备的专题复习资料，用于补充 AgentGuide 面试题库与项目讲述材料。
> 内容以公开资料、项目复盘和工程实践方法论为基础做二次整理，建议结合自己的真实项目经历进行取舍与改写。

---
## 目标：把 RAG 演进、Agentic RAG、Deep Research 从概念到实现全部打透

> 使用建议：这份文档对应《面试备战手册_补充模块.md》的第八节 Agentic RAG / Deep Research。综合吸收：
> 1. 通义 DeepResearch 技术报告：端到端 Agentic 中训练+后训练框架，30.5B 参数
> 2. AgenticRAGTracer 学习笔记：hop-aware benchmark，六步数据构建流水线
> 3. Harness 101 多轮 Loop 章节：Deep Research 作为多轮 ReAct 的自然延伸
> 4. 补充模块第八节：RAG 四代演进路径

---

# 一、先总览：面试官在 Agentic RAG 这条线上会怎么打

Agentic RAG 是 JD 的核心要求（Agentic Search），也是你简历上相对薄弱的显式经验项。面试官会从三个角度切入：

1. **概念层**：Agentic RAG 和传统 RAG 的本质区别是什么？一口说出 Naive → Advanced → Modular → Agentic 四代演进
2. **机制层**：Self-RAG 的自我反思怎么触发？CRAG 的纠错检索怎么工作？通义 DeepResearch 的端到端训练框架怎么做？
3. **场景层**：小红书场景下怎么设计 Agentic Search？和 Perplexity 这类通用搜索有什么本质差异？

---

# 二、核心知识框架

## 1. RAG 四代演进

面试时必须能一口气讲出这条线。

```
Naive RAG (2023) → Advanced RAG (2024) → Modular RAG (2025) → Agentic RAG (2025-2026)
```

### 第一代：Naive RAG
- **流程**：query → embed → retrieve top-k → stuff into context → generate
- **特点**：单轮、无推理、无判断。把检索当成一步做完就完事
- **典型问题**：
  - 检索质量差时，生成跟着差
  - 不需要检索的简单问题也硬搜（浪费成本）
  - 检索结果不充分时没有二次检索

### 第二代：Advanced RAG
- **新增**：query rewriting（改写查询提升召回）、reranking（重排序提升精度）、hybrid search（混合检索）
- **本质**：仍然是线性 pipeline，但每一步做得更好
- **局限**：组件是死的——先改写再检索再重排，不会根据中间结果调整策略

### 第三代：Modular RAG
- **新增**：每个组件可独立替换（retriever / reranker / generator / query planner）。你可以换 embedding 模型、换向量库、换 reranker
- **本质**：工程模块化，但逻辑流仍是预定义的

### 第四代：Agentic RAG
- **核心变化**：Agent 自主决定**是否检索、检索什么、什么时候停止、是否需要二次检索、检索结果是否充分**
- **三个关键词**：**自主规划**（plan the search）、**自主反思**（reflect on results）、**自主纠错**（correct bad retrievals）
- **和前三代的本质差异**：前三代是 pipeline，这一代是 **agent 在驾驶检索过程**

### 一句话讲法（面试必背）
> "Naive RAG 是'搜一次就答'，Advanced RAG 是'搜得更准再答'，Agentic RAG 是'搜不搜、搜几次、搜什么、搜够了没——全部由 Agent 自己判断'。"

---

## 2. Naive RAG → Traditional Search → Agentic Search 三者区别

面试官可能问"Agentic Search 和传统搜索有什么区别"，你需要能把三者区分清楚。

| 维度 | 传统搜索 | RAG | Agentic Search |
|------|---------|-----|---------------|
| **执行模式** | query → ranking → results | query → retrieve → generate | query → 理解意图 → 规划搜索策略 → 多步搜索 → 信息整合 → 推理验证 → 回答 |
| **轮次** | 单轮 | 单轮检索+生成 | 多轮、有规划、有纠错 |
| **谁做决策** | 搜索引擎（TF-IDF/BM25/learning-to-rank） | 固定的 retrieve-then-generate 逻辑 | **Agent 自主做全部搜索决策** |
| **纠错能力** | 无 | 无 | 有——发现检索不够会换关键词/策略重新搜 |
| **典型产品** | Google、百度 | 早期的 ChatGPT + retrieval | Perplexity、通义 DeepResearch、Gemini Deep Research |

---

## 3. Agentic RAG 的五大核心设计模式

| 模式 | 核心机制 | 一句话 | 和你的关联 |
|------|---------|--------|-----------|
| **Adaptive Retrieval** | Agent 判断是否需要检索（简单问题直接答） | "不需要搜的时候别硬搜" | TinyClaw 的 Skill 命中就跳过 RAG |
| **Self-RAG** | 模型自我评估检索质量和答案质量，决定是否重新检索 | "搜完了自己判断够不够" | 类似你的不确定性门控思路 |
| **Corrective RAG (CRAG)** | 检索结果质量差时自动触发修正检索 | "搜错了就重新搜" | 和 GoAfar 里的纠错轨迹逻辑一致 |
| **Multi-Step Retrieval** | 分步检索，后一步基于前一步结果 | "搜了 A 才知道下一步该搜 B" | GoAfar 的多步工具调用 |
| **Graph RAG** | 基于知识图谱做关联检索 | "不只是搜文档，而是搜关系" | AgentGuide 的知识图谱 |

### Self-RAG 详细机制（必背）
Self-RAG 在每个生成步骤中插入特殊的 reflection tokens：
- `[Retrieve]`：需要检索，∴触发检索
- `[NoRetrieve]`：不需要检索，直接生成
- `[Relevant]`：检索结果是相关的
- `[Irrelevant]`：检索结果不相关，∴需要重新检索
- `[Supported]` / `[PartiallySupported]` / `[Unsupported]`：生成内容被检索结果支持的程度

这本质上是在做**生成过程中的实时自我质量评估**。

### CRAG 详细机制（必背）
1. 检索 → 用 retrieval evaluator 评估质量
2. 如果质量好（≥ 置信阈值）→ 直接用
3. 如果质量差（< 置信阈值）→ 触发 web search 补充
4. 如果仍不够 → 用"知识精炼"从混合结果中提取最相关的内容

---

## 4. 通义 DeepResearch：端到端 Agentic 深度研究训练框架

这是 2025-2026 年 Deep Research 方向最系统的技术报告之一，面试中引用来展示你对"怎么训练一个会做深度研究的 Agent"的理解。

### 4.1 核心定位

通义 DeepResearch 不是"给通用 LLM 加一个搜索工具"，而是专为**长周期、深度信息检索研究任务**设计的**Agentic 语言模型**。

### 4.2 训练框架：Agentic 中训练 + 后训练

| 阶段 | 做什么 | 关键设计 |
|------|--------|---------|
| **Agentic Mid-Training** | 在大量 agent interaction 数据上继续预训练 | 让模型学会 agent 的基本行为模式 |
| **Agentic Post-Training** | SFT + RL（GRPO） | 让模型学会在深度研究场景下规划、搜索、整合 |

### 4.3 数据合成：完全自动化，无需人工标注

这是通义 DeepResearch 最值得引用的地方——**它的训练数据合成是完全自动化的**：
- 高度可扩展的数据合成流水线
- 覆盖所有训练阶段的数据需求
- 为每个训练阶段构建定制化的环境（确保交互过程稳定一致）

### 4.4 模型架构与性能

- 305 亿总参数，每 token 仅激活 33 亿（MoE 架构）
- 在 BrowseComp、GAIA、FRAMES、WebWalkerQA、xbench-DeepSearch 等多个 Agentic 深度研究 benchmark 上表现卓越
- 证明了"专门为深度研究设计的 Agentic 模型"可以超过"通用大模型+搜索工具"的组合

### 面试时怎么用
> "通义 DeepResearch 的核心证明了一件事：深度研究能力不是靠给通用 LLM 加一个搜索工具就能获得的，而是需要通过 Agentic 中训练和后训练，让模型自己学会'什么时候搜、搜什么、搜到多少算够、搜到的信息怎么整合'。这和我在 GoAfar 里的经验是一致的——搜索不是能力原语，而是需要被训练的行为。"

---

## 5. Deep Research 的产品形态

### 核心流程
```
用户提问 → Agent 理解意图 + 澄清 → 制定研究计划 → 多轮搜索 →
信息交叉验证 → 结构化整合 → 生成研究报告
```

### 和 Agentic RAG 的关系
- Agentic RAG 强调的是**检索过程中的智能决策**：自适应、多策略、有纠错
- Deep Research 强调的是**研究深度**：多轮、多源、交叉验证、长报告
- 两者是包含关系：Deep Research 是 Agentic RAG 在"深度研究"这个场景下的极致应用

### Perplexity vs 小红书 Agentic Search

| 维度 | Perplexity | 小红书 Agentic Search |
|------|-----------|---------------------|
| **信息源** | 全网网页 | 小红书 UGC 笔记 + 评论 + 外部事实 API |
| **内容特点** | 结构化新闻/百科/文档 | 高度主观、多模态、生活方式密集 |
| **核心挑战** | 信息量太大、时效性 | 主观性 vs 客观事实分离、UGC 质量参差 |
| **差异化优势** | 覆盖面广 | 生活方式细节 + 社区语义 + 个性化 |

---

## 6. Deep Research 相关的关键技术要点

### 6.1 搜索策略规划（Search Planning）

Agent 不能想到哪搜到哪。Deep Research 的一个关键技术是**搜索策略的显式规划**：
- 把用户复杂问题分解成多个搜索子目标
- 确定每个子目标的搜索关键词和预期信息类型
- 区分"必须精确匹配"和"可以模糊探索"的维度

### 6.2 信息交叉验证（Cross-Verification）

单源信息不可靠。Deep Research 需要在多个独立来源之间做交叉验证：
- 对于事实性信息（价格、营业时间、地点），用多个来源互相印证
- 对于不一致的信息，降低置信度或标记为"不确定"
- 对于无法验证的信息，明确告知用户"以下信息来自单一来源，未经验证"

### 6.3 长上下文的信息组织

Deep Research 一次任务可能产生几十轮搜索、十几篇文档的上下文。信息组织是关键：
- 不是把所有检索结果塞进上下文——只保留关键证据、关键矛盾、关键结论
- 中间搜索结果做结构化摘要
- 最终输出是对用户有价值的整合结论，不是搜索过程日志

---

# 三、高频面试题 + 高强度答案

## Q1. Naive RAG、Advanced RAG、Agentic RAG 的核心区别是什么？

### 主答模板
Naive RAG 是搜一次就答。它把所有检索当一步做，检索质量不好生成就跟着不好。Advanced RAG 加上了 query rewriting、reranking、hybrid search 让检索更准，但仍然是固定的线性 pipeline——先做什么后做什么是写死的。Agentic RAG 的核心跃迁在于**把检索的控制权从 pipeline 交给 Agent 自己**：Agent 自己判断要不要搜、搜什么关键词、搜到多少算够、搜到的信息足不足、需不需要换个方向重新搜。前三代是"告诉模型怎么搜"，第四代是"让模型学会自己搜"。

---

## Q2. Self-RAG 和 CRAG 分别是什么？有什么区别？

### 主答模板
Self-RAG 的核心是在生成过程中插入 reflection tokens——`[Retrieve]`/`[NoRetrieve]`/`[Relevant]`/`[Irrelevant]`/`[Supported]`——让模型在每个生成步骤实时评估是否需要检索、检索结果是否相关、生成内容是否有据可依。

CRAG 的核心是 retrieval evaluator + 纠错回路：检索完之后先用一个轻量评估器判断检索质量，质量不够就自动触发 web search 补充或"知识精炼"。

两者最大的区别：Self-RAG 是"生成过程中边做边反思"，CRAG 是"检索完之后统一评估并纠错"。Self-RAG 更细粒度但推理成本更高，CRAG 更轻量但评估只发生在检索阶段。

---

## Q3. 通义 DeepResearch 的端到端训练框架和"通用 LLM + 搜索 API"有什么本质差别？

### 主答模板
"通用 LLM + 搜索 API"的做法是：给一个不会搜索的模型，外挂一个搜索工具，然后在 inference 时让它试着用。这种做法有两个根本问题：第一，模型没有被训练"搜索是一种解决问题的手段"，它可能该搜不搜、不该搜乱搜；第二，工具使用行为完全靠 prompt 驱动，不稳定。

通义 DeepResearch 的做法是**端到端的 Agentic 训练**：Agentic Mid-Training 让模型在大量 agent interaction 数据上学会基本的搜索-推理循环行为，Agentic Post-Training 用 SFT + GRPO 训练模型在深度研究场景下规划搜索、执行搜索、整合信息。核心差异：**搜索行为不是外挂的，是训练进权重的。**

---

## Q4. Agentic RAG 和 Deep Research 是什么关系？

### 主答模板
Agentic RAG 是一个更大的伞概念——任何让 Agent 自主决定检索策略的 RAG 系统都可以叫 Agentic RAG。Deep Research 是 Agentic RAG 在"深度研究"场景下的极致应用——它不只是让 Agent 决定怎么搜，还要求多轮、多源、交叉验证、长报告。简单来说：Deep Research 一定是 Agentic RAG，但 Agentic RAG 不一定是 Deep Research。比如一个简单的"判断要不要检索然后回答问题"的系统是 Agentic RAG，但不是 Deep Research。

---

## Q5. 如果让你设计小红书的 Agentic Search——用户问"上海有什么好吃的湖南菜"——请描述核心流程。

### 主答模板
传统搜索会直接返回一堆湖南菜笔记的排名列表。Agentic Search 会：

1. **意图理解与拆解**：不是"搜湖南菜"，而是拆成——用户在上海、要湖南菜（正宗度是核心维度）、隐含需求是"好吃"（需要真实评价佐证）、可能还关心辣度、价格、排队情况

2. **多步搜索规划**：第一轮搜候选餐厅，第二轮搜每个餐厅的真实 UGC 评价（不只评分，还要看具体描述），第三轮搜营业状态/排队/交通信息，第四轮交叉验证推荐一致性

3. **信息结构化**：把非结构化的笔记和评论变成结构化槽位——餐厅名、人均、辣度、排队时间、最近评价里是否提到"正宗"、"好吃"的频率、是否有最近差评提示

4. **事实 vs 观点分离**："有剁椒鱼头"是事实（可以交叉验证），"全上海最好吃"是观点（不能当事实用）。Agent 必须显式区分这两类信息

5. **个性化排序**：结合用户的历史偏好——如果用户收藏过"重辣"标签的笔记，排序时辣度权重更高

6. **最终呈现**：不是扔一个链接列表，而是给出"你的推荐 + 为什么推荐 + 每条推荐的证据来源 + 置信度"

---

## Q6. 小红书的 UGC 内容高度主观、质量参差不齐——这对 Agentic Search 设计有什么特殊挑战？

### 主答模板
三个核心挑战：

1. **事实和观点高度混合**：一篇笔记可能说"这家店超级好吃（观点），有剁椒鱼头（事实），人均 80（半事实半主观）"。Agent 必须把这三类信息分开处理——事实可以交叉验证，观点只能作为参考，半事实需要多源确认

2. **信息可信度没有统一权威来源**：不像新闻有出处、百科有编辑规范，UGC 的可信度完全靠内容层面的信号：内容一致性、用户历史行为、是否多篇笔记互相印证、时效性。Agent 需要自己构建可信度打分

3. **热点偏差**：小红书上的推荐往往集中在少数爆款店铺，真正好吃但没人写笔记的店可能被系统性忽略。Agentic Search 需要有意识地引入探索性维度——不是只推最火的，也推有潜力但曝光少的

---

## Q7. Deep Research 场景下，Agent 怎么判断"搜够了"？

### 主答模板
"搜够了"的判断是 Agentic RAG 里最核心也最难的决策之一。不能靠固定条数，也不能无限搜下去。合理的判断标准有几个：

1. **信息饱和（Information Saturation）**：最近几轮搜索的新增有效信息越来越少，新文档基本在重复已有结论
2. **答案置信度达标**：对于可以验证的事实（营业时间、价格、地点），达到了交叉验证标准（至少 2 个独立来源一致）
3. **预算耗尽**：硬性限制——最大搜索轮数、最大 token 消耗、最大时间。这是兜底，不是主要策略
4. **Agent 主动判断**：在 prompt 中训练 Agent 在每个搜索步骤后自我提问"现在的信息足够给用户一个有用且准确的回答了吗？还缺什么？"

---

# 四、高频总追问清单

## A. RAG 演进
1. 四代 RAG 各自的核心变化是什么？
2. 为什么说从 Modular RAG 到 Agentic RAG 是一次质变？
3. Agentic RAG 和传统搜索引擎的本质区别在哪？

## B. Self-RAG / CRAG
4. Self-RAG 的 reflection tokens 具体有哪些？各自触发什么行为？
5. CRAG 的 retrieval evaluator 怎么工作？
6. Self-RAG 和 CRAG 各自更适合什么场景？

## C. Deep Research
7. Deep Research 和 Agentic RAG 是什么关系？
8. 通义 DeepResearch 的 Agentic Mid-Training + Post-Training 框架怎么做？
9. Deep Research 场景下怎么判断"搜够了"？
10. Deep Research 的多源信息交叉验证怎么做？

## D. 小红书场景
11. 小红书 UGC 场景下 Agentic Search 最大的三个特殊挑战是什么？
12. 怎么处理 UGC 中事实和观点高度混合的问题？
13. 和 Perplexity 相比，小红书 Agentic Search 的最大差异化在哪？
14. 如果你的 Agentic Search 推荐的餐厅用户去了发现不好吃——你怎么闭环？

## E. 和你的项目对接
15. GoAfar 的三路召回和 Agentic RAG 的 multi-source retrieval 有什么异同？
16. TinyClaw 的 RAG-to-Skill 和 Adaptive Retrieval（命中 Skill 就跳过检索）是什么关系？
17. Omni-Aware RAG 的 omni-routing 和 Agentic RAG 的 adaptive retrieval 有什么关系？
18. AgenticRAGTracer 的 hop-aware benchmark 对 Agentic RAG 评测有什么价值？

---

# 五、串联叙事：面试时最强的 Agentic RAG 主线

> "我对 Agentic RAG 的理解是从 RAG 的四代演进切入的。Naive RAG 是最朴素的搜一次就答，Advanced RAG 把检索质量做上去了但流程仍是固定的，Agentic RAG 的核心变化是把检索的控制权从 pipeline 交给 Agent 自己——Agent 自己判断要不要搜、搜什么、搜到什么时候算够。这和我在 TinyClaw 做的 RAG-to-Skill 在思路上是相通的——TinyClaw 的本质就是让 Agent 从'每次重搜'逐步过渡到'优先复用已验证的搜索和调用流程'。

> Deep Research 是 Agentic RAG 在深度研究场景下的极致版本。通义 DeepResearch 的端到端训练框架给我的启发最大——它不是给通用 LLM 加搜索工具，而是通过 Agentic 中训练和后训练，让模型自己学会搜索-推理循环。这和我在 GoAfar 里的体会一致：搜索不是外挂能力，而是需要被训练的行为。

> 如果落地到小红书场景，Agentic Search 最大的挑战不是技术层面的'怎么搜'，而是信息质量层面的'怎么判断搜到的东西可不可信'。UGC 的事实和观点高度混合，Agent 必须显式地做事实-观点分离和信息交叉验证。这恰恰是 Omni-Aware RAG 里不确定性估计和 EAB 这类机制最有价值的场景。"

---

# 六、最后：Agentic RAG 准备到什么程度，才算真准备好了？

## 你至少要做到 6 件事
1. **能一口气讲出 RAG 四代演进，每代一句话说清核心变化**
2. **能讲清楚 Self-RAG 和 CRAG 的核心机制和差异**
3. **能讲出通义 DeepResearch 的 Agentic Mid-Training + Post-Training 框架**
4. **能设计一个完整的小红书 Agentic Search 流程（6 步）**
5. **能讲出 UGC 场景下 Agentic Search 的三个特殊挑战和应对方案**
6. **能把 GoAfar / TinyClaw / Omni-Aware RAG 放进 Agentic RAG 的框架里解释**

## 如果你想让面试官"折服"，你要主动讲出来的 4 句话
1. **"Agentic RAG 和传统 RAG 的本质区别不在检索质量，而在检索决策——前三代是'告诉模型怎么搜'，第四代是'让模型学会自己决定怎么搜'。"**
2. **"Deep Research 给我的最大启发是通义 DeepResearch 的训练哲学——搜索不是外挂工具，而是需要被端到端训练的行为。这和我在 GoAfar 里做 GRPO 训练路线规划时的体会完全一致。"**
3. **"小红书 Agentic Search 的核心挑战不是技术层面的，而是信息质量层面的——UGC 的事实和观点高度混合，Agent 必须做事实-观点分离。这不是一个检索问题，是一个信息可信度建模问题。"**
4. **"判断'搜够了'是 Agentic RAG 里最被低估的难题。不能靠固定条数，要结合信息饱和、答案置信度和预算约束——这才是真正的 Agent 判断力。"**
