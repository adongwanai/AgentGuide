---
type: 求职指南
status: 已发布
level: 高阶
topic:
  - 面试求职
  - 评测
  - Agent
---

# Agent 评测 / Benchmark 彻底备战手册（补充模块对应版）
> **本手册定位**：面向 AI Agent 求职与面试准备的专题复习资料，用于补充 AgentGuide 面试题库与项目讲述材料。
> 内容以公开资料、项目复盘和工程实践方法论为基础做二次整理，建议结合自己的真实项目经历进行取舍与改写。

---
## 目标：把 Agent 评测从方法论到工程落地全部打透，面试时能讲得像真正做过 Agent 系统评测的人

> 使用建议：这份文档对应《面试备战手册_补充模块.md》的第五节评测。综合吸收了三份材料：
> 1. Anthropic "Demystifying Evals for AI Agents" PDF（74 页）——Agent 评估的系统框架
> 2. Agent Eval Training PPTX（22 页）——150 task benchmark 构建法、三层质量门、四级 Ship Gate
> 3. 补充模块评测体系——主流 benchmark 全景图 + 你自己的评测指标

---

# 一、先总览：面试官在评测这条线上会怎么打

面试官在评测上的提问有三种类型：

**类型一：基础概念**——"Agent 评测和 LLM 评测有什么区别？""pass@k 和 pass^k 分别表示什么？"答不出来 = 基础不过关。

**类型二：你的实践**——"你在 KTClaw 里的 15%→3% 是怎么测的？""GoAfar 的路线可行率是怎么定义的？"答不出来 = 数字是编的。

**类型三：系统设计**——"如果让你为小红书 C 端 Agent 设计评测体系，你会怎么设计？""你用什么指标来判断一个 Agent 版本能不能上线？"答出来 = 你能做系统级 Agent 评测。

本章全部覆盖。

---

# 二、核心知识框架

## 1. Agent 评测和 LLM 评测的本质区别

### LLM 评测
- 单轮、单次输入、单次输出
- 评测目标是模型的知识、推理、生成质量
- 指标：BLEU、ROUGE、pass@1、accuracy、MMLU 分数
- 可复现性相对较高

### Agent 评测
- 多轮、多步、有工具调用、有状态变更
- 评测目标是整个 Agent 系统的规划、执行、纠错能力
- 指标：任务完成率、工具调用准确率、多步成功率、端到端延迟
- 两个根本性挑战：**非确定性** + **多轮交互的蝴蝶效应**

### 一句话区别
> "LLM 评测测的是模型能不能答对，Agent 评测测的是系统能不能做成。"

---

## 2. Agent 评测的 8 个核心概念

面试时必须能分清这些基本术语。

| 概念 | 英文 | 一句话 | 易错点 |
|------|------|--------|--------|
| **任务** | Task | 一个有初始状态 + 成功标准的完整用例 | 不是 prompt，是考题 |
| **试验** | Trial | 对同一个 Task 的一次完整执行 | Agent 有随机性，跑一次不够 |
| **评分器** | Grader | 判断一次 Trial 是否通过的逻辑 | 不只有 code-based，还有 LLM judge |
| **检查项** | Check | Grader 对单条 assertion 的判断 | 一个 Grader 可以有多个 Check |
| **对话记录** | Transcript | Trial 的完整记录（messages + tool calls） | Agent 说的≠实际做的 |
| **实际结果** | Outcome | 执行后的环境状态和关键指标 | 数据库里有没有订单、API 到底调没调 |
| **评测脚手架** | Eval Harness | 跑评测所需的基础设施 | 负责环境重置、Trial 执行、结果收集 |
| **Agent 脚手架** | Agent Harness | 把模型变成 Agent 的框架 | 你要分开测模型能力和 Harness 质量 |

### 最关键的一对概念：Transcript vs Outcome

Agent 在对话中说"订单已创建" —— 这是 Transcript。
数据库里 Orders 表是空的 —— 这是 Outcome。

**Agent 说的≠实际做的。** 评估 Agent 不能只看 Transcript，必须验证 Outcome。

---

## 3. 三种 Graders 的完整对比

| 维度 | Code-based | Model-based (LLM Judge) | Human |
|------|-----------|------------------------|-------|
| **确定性** | 完全确定 | 非确定 | 非确定 |
| **成本** | 极低（毫秒级） | 中等（每次调用 LLM） | 极高（人时） |
| **速度** | 毫秒 | 秒 | 小时/天 |
| **适用范围** | 数值匹配、状态检查、工具调用检查 | 语气、风格、完整性、开放性任务 | 所有维度 |
| **最适合** | 守门员：拦住明显错误 | 艺术评委：捕捉细微差别 | 黄金标准：校准其他评分器 |
| **典型用法** | regex / unit test / DB state check | rubric scoring / pairwise / reference-based | SME review / inter-annotator agreement |

### 组合原则（面试必背）
> "能走确定性的走确定性。code-based 做守门员，LLM judge 做主观维度评分，human 做校准和关键决策。三种不是替换关系，是层层递进关系。"

### LLM Judge 三大最佳实践（来自 PPTX）

1. **Uncertain Escape Hatch（逃生门）**：judge prompt 必须给 LLM 一个"不确定"的出口（INSUFFICIENT_INFO），强制它"不确定就说不确定"而不是强行打分
2. **Evidence Quote（证据引用）**：要求 judge 引用原文证据，标 NONE 的维度视为低置信。防止 LLM hallucinate 评分理由
3. **One Dimension Per Call（每次只评一个维度）**：不要一次性评估所有维度（LLM 会混淆）。准确性、语气、完整性、专业性分开调

---

## 4. Capability Eval vs Regression Eval

这是 Agent 评测中最基础但最常被混淆的分类。

| 维度 | Capability Eval | Regression Eval |
|------|----------------|-----------------|
| **回答的问题** | "这个 Agent 能做什么？" | "它还保持着原来的能力吗？" |
| **期望 pass rate** | 30-50%（有难度，有区分度） | ~100%（历史通过的必须继续通过） |
| **用途** | 能力爬坡、模型选型、方向判断 | 防回退、持续监控 |
| **更新频率** | 每次主要模型/架构变更 | 每次修改 |

### pass@k vs pass^k

**pass@k**：k 次尝试中至少 1 次成功的概率。"Shots on goal"——适合 coding agent，一次过就行，给多次机会衡量最佳情况。

**pass^k**：k 次尝试全部成功的概率。适合 customer-facing agent——每次都必须是好的用户体验。k=3 时，单次成功率 75% 意味着只有 42% 的用户连续 3 次都满意。

---

## 5. Anthropic 的 8 步评测路线图

面试时可以完整讲出这条线，展示系统化思维。

| 步骤 | 要点 | 一句话 |
|------|------|--------|
| **Step 0** | Start early，20-50 task 起步 | 不要等完美了再开始 |
| **Step 1** | 从手动测过的 case 开始 | 把已有的经验变成可复现的 task |
| **Step 2** | 写可重现的 task + reference solution | 0% pass@100 时先检查是不是 task 写错了 |
| **Step 3** | balanced positive/negative | 不能全都是简单 case 也不能全是压测 case |
| **Step 4** | Robust harness + 可重置环境 | 每次 trial 从相同初始状态开始 |
| **Step 5** | Grader 设计：code-based → LLM judge → human | 按"确定性优先"原则分层 |
| **Step 6** | Read the transcripts | 永远不要跳过这一步——数据告诉你结果，transcript 告诉你原因 |
| **Step 7** | 追求 saturation：pass rate 接近 100% 加更难 task | 不是"pass rate 越高越好"，而是"当前 task 难度刚好" |
| **Step 8** | Living artifact + open contribution | eval 不是测试团队的工作，是所有人的工作 |

### 三个最常见的坑（PPTX）

- **坑 1**：0% pass@100 往往不是模型不行，是 task 写错了——先 audit task 本身
- **坑 2**：不要只看 tool call sequence——agent 可能用你想不到的方式正确解决问题
- **坑 3**：LLM judge 不加 uncertain escape hatch = 强迫 LLM 在不确定时乱打分

---

## 6. 150 Task Benchmark 构建法（PPTX 核心）

这是 PPTX 里最实用的框架。

### 任务来源七步法

1. **Mine**：从 production logs / 用户投诉 / CS escalation 中挖掘
2. **Cluster**：embedding 聚类，理解用户真正在问什么
3. **Sample**：每个 cluster 采样以确保 coverage
4. **Reference**：每个 task 必须标注"什么是 pass"以及 reference trace
5. **SME + IAA**：至少两位 SME 做标注，IAA < 0.6 就重写 task
6. **Balance**：positive / negative 均衡
7. **Adversarial**：压测 case + 边缘 persona + 安全探测

### 最小可行 Benchmark = 150 task

| 类别 | 数量 | 用途 |
|------|------|------|
| Capability（日常意图） | 50 | 目标 pass rate 30-50%，有难度无天花板 |
| Regression（历史 bug 固化） | 50 | 目标 100% pass rate |
| Adversarial（安全/边缘/OOD） | 30 | 压测安全边界 |
| Multi-turn Hard（复杂/高难度/意图切换） | 20 | 测上限 |

---

## 7. 三层质量门 + 四级 Ship Gate

### 三层质量门（上线后持续监控）

| 层 | 测什么 | 频率 | 示例 |
|----|--------|------|------|
| **Layer 1：安全底线** | blacklist regex + classifier + PII check | 每条 transcript 实时 | critical violation = 0，violation rate < 0.5% |
| **Layer 2：业务 KPI** | lead rate / CTR / ATC vs baseline | 每天定时 | 和 baseline 对比 |
| **Layer 3：离线质量** | ~150 task suite | 每周 + 每次 model upgrade | capability + regression + adversarial |

### 四级 Ship Gate（模型升级能不能上线）

| 阶段 | 做法 | 停止条件 |
|------|------|---------|
| **1. Offline Diff** | 同 task 新旧版本各 5 trial，paired comparison + bootstrap CI | regression / 无 decisive win / cost ↑ 20% |
| **2. Shadow Traffic** | 新 agent 跑 prod 流量但不 serve 用户，观察 24h divergence pattern | 人工 read divergent transcripts 发现风险 |
| **3. A/B w/ Guardrails** | 10% 流量，primary metric + guardrails（投诉/安全/成本/延迟） | 只有 guardrail 触发才 early stop |
| **4. Phased Rollout** | 1% → 10% → 50% → 100%，每阶段最小观察时间 | 自动 rollback 条件预定义 |

### 关键原则
> "Primary metric 不作为 early stop 条件——只有 guardrail 触发才 early stop。用 primary metric 做 early stop 是隐式 p-hacking。"

---

## 8. 主流 Agent Eval Benchmark 全景图

| Benchmark | 测什么 | 为什么重要 | 和小红书的关联 |
|-----------|--------|-----------|------------|
| **SWE-bench Verified** | 软件工程任务（GitHub issue 修复） | Agent 多步规划+执行+纠错的黄金标准 | 低（非 SWE 场景） |
| **WebArena** | 真实网页环境中的多步任务 | 最接近 C 端 Agent 的交互模式 | **高**——小红书 Agent 本质上是 web/内容环境的任务执行 |
| **GAIA** | 通用 AI 助手任务（多步推理+工具） | 综合能力评测 | **高**——涵盖搜索、推理、事实核查 |
| **AgentBench** | 8 类环境（OS/DB/KG/游戏/网页） | Agent 泛化能力的广度测试 | 中——参考评测维度设计 |
| **τ²-Bench** | 航空/零售等真实业务场景（工具+状态+政策） | 最接近"Agent 帮用户办事"的场景 | **极高**——和 C 端 Agent 办事场景最相似 |
| **Tau-bench** | 客服场景多轮交互 | 对话式 Agent 的核心能力 | **高**——小红书 Agent 是对话+推荐混合 |
| **LoCoMo** | 长对话长期记忆评测 | 和你的 KTClaw 直接相关 | **你简历写了，必须能讲出** |

### LoCoMo 必须能讲到这个程度
- 核心评测维度：长时记忆一致性、事实更新后回答是否一致、多轮任务完成情况、时序推理
- 不是单轮知识问答，而是"模型能不能在长期对话中保持正确的记忆"
- 你在 KTClaw 里参考的不是 LoCoMo 的具体分数，而是它的**评测维度设计思路**

---

## 9. Agent 评测的核心挑战

### 挑战一：非确定性
同一个 Agent，同样的 Task，跑 10 次可能有 10 种不同结果。温度、采样、模型版本、外部 API 响应都可能影响结果。

**应对**：每个 task 跑多个 trial（至少 10 次），报均值和置信区间。

### 挑战二：错误传播（蝴蝶效应）
多轮 Agent 在第 3 轮的小偏差在第 10 轮变成大错误。Agent 评测的难点不在"第一轮对不对"，而在"错了之后能不能自己发现并修正"。

**应对**：不只评最终结果，也要评中间步骤的纠错能力。

### 挑战三：环境漂移
Benchmark 的环境是冻结的，但真实环境在变化。API 变了、网页改了、数据过期了。

**应对**：定期的 living artifact 更新，不要把 benchmark 当一次性产物。

### 挑战四：Reward Hacking
Agent 可能学到"看起来很对"的方式而不是真正解决问题的方式。UC Berkeley 发现 8 大 Agent benchmark 都可以被 reward hack 到接近 100%。

**应对**：不只靠自动评分，也要定期 read transcripts + human review。

### 挑战五：Transcript 欺骗（测 Agent 最容易忽略的）
Agent 在对话中说得很好听——"已为您预订成功"——但数据库里什么都没发生。评估只看 Transcript 不看 Outcome 是 Agent 评测最大的坑。

**应对**：至少 50% 的 Check 必须验证 Outcome（DB state、API 调用记录、文件系统变更），不能全靠 Transcript。

---

## 10. 推荐系统 / 排序类 Agent 的专门指标

小红书场景下 Agent 的核心能力之一是推荐，需要专门的排序指标。

| 指标 | 公式 / 含义 | 什么时候用 |
|------|-----------|-----------|
| **NDCG@k** | 归一化折损累计增益 | 考虑排序位置和相关性等级的推荐质量 |
| **MRR** | Mean Reciprocal Rank，正确答案排名的倒数均值 | 单正确答案的检索任务 |
| **Recall@k** | top-k 中包含相关结果的比例 | 评测召回覆盖面 |
| **HitRate@k** | top-k 中至少命中 1 个的比例 | 最基础的命中率 |
| **Intra-list Diversity** | 推荐列表内的多样性 | 防止推荐单一化 |
| **Catalog Coverage** | 推荐覆盖了多少不同的物品/内容 | 长尾曝光、生态健康 |

---

# 三、高频面试题 + 高强度答案

## Q1. Agent 评测和 LLM 评测有什么区别？

### 主答模板
LLM 评测通常测的是单轮输入输出的质量——给一个问题，看模型能不能答对。Agent 评测测的是一个完整系统的多步执行能力——模型不仅要答对，还要规划步骤、调用工具、处理错误、维护状态。

三个根本差异：
1. **多轮 vs 单轮**：Agent 的一个 task 可能涉及 10-50 轮交互
2. **有状态 vs 无状态**：Agent 的执行会影响环境状态（写文件、改数据库），评测必须验证 Outcome
3. **非确定性放大**：LLM 的非确定性在 Agent 的多步交互中被放大——每一步的小偏差累积成大差异

---

## Q2. 你说 Code-based Grader 做守门员，LLM Judge 做艺术评委——能举一个具体例子吗？

### 主答模板
一个 Agent 帮用户在电商平台找护肤品。Code-based Grader 负责检查：
- 推荐的商品是否真的在数据库中存在（不是幻觉编的）
- 价格是否在用户预算范围内
- API 调用是否正确（有没有调用不需要的工具）

LLM Judge 负责评分：
- 推荐理由是否个性化（不是千篇一律的模板）
- 语气是否专业且友好
- 是否注意到用户的隐含需求（比如用户说"我是干皮"）

一个推荐可以技术上完全正确（code-based 全过），但读起来像客服机器人（LLM judge 给低分）。反过来，一个推荐可以读起来很好，但推荐了一个已下架的商品——这就是为什么 code-based 守门员更重要。

---

## Q3. 你怎么判断一个 Agent 版本能不能上线？

### 主答模板
我的判断框架是四层递进：

1. **Offline Diff**：新旧版本在同样的 task suite 上各跑 5 个 trial，做 paired comparison + bootstrap 置信区间。如果有统计显著的回退，直接打回
2. **Shadow Traffic**（如果是高风险变更）：新版本接真实流量但不 serve 用户，至少跑 24 小时，人工 read divergent transcripts 找风险
3. **Guardrail-gated A/B**：10% 流量，看核心指标的同时更关注护栏指标。**只用护栏指标做 early stop，不用核心指标做 early stop**——否则就是隐式 p-hacking
4. **Phased Rollout**：1% → 10% → 50% → 100%，每阶段有最小观察时间，自动 rollback 条件预先定义

---

## Q4. 你说的"150 task 基准"是怎么构建的？

### 主答模板
150 task 分成四个 buckets：50 个 capability（日常意图，目标 30-50% pass rate，有区分度）、50 个 regression（历史 bug 固化，目标 100% pass）、30 个 adversarial（安全/边缘/OOD 压测）、20 个 multi-turn hard（复杂高难度任务）。

构建过程是七步：
1. 从 production logs 和用户投诉中挖掘
2. 用 embedding 聚类理解用户意图分布
3. 按 cluster 采样确保 coverage
4. 每个 task 标注 reference solution 和 pass 标准
5. 两位 SME 独立标注，IAA < 0.6 就重写 task
6. 平衡正负样本
7. 加 adversarial 压测 case

---

## Q5. 你在 KTClaw / GoAfar 里是怎么做评测的？和这个框架有什么关系？

### 主答模板
KTClaw 的评测体系——稳定性、问题解决准确率、Memory 命中率、Skill 复用率——本质上是给企业 Agent 场景做了一套领域特定的 task suite + graders。其中 Memory 命中率更像 code-based check（检索到了且被使用了），而问题解决准确率混合了 code-based（任务是否完成）和人工 review（答案是否有上下文串扰）。

GoAfar 的路线可行率评测是典型的 code-based grader——OR-Tools 给出确定性的 feasible/infeasible 判断。这是我所有项目里 grader 确定性最高的评测方式，因为 verifier 是数学上严格的。

如果现在回头看，我会给 KTClaw 加更多基于 Outcome 而非 Transcript 的 check——比如在对话结束后验证数据库状态、API 调用记录、文件变更，而不是只看 Agent 说了什么。

---

## Q6. 你怎么看 Agent benchmark 被 reward hack 这件事？

### 主答模板
UC Berkeley 发现 8 大 Agent benchmark 都可以被 reward hack 到接近 100%——Agent 学会了"看起来很对"而不是"真正解决了问题"。这说明两件事：
1. **Grader 不能只看 Transcript**。如果只靠 Transcript 判断成功，Agent 学会"说正确的话"就够了。必须验证 Outcome。
2. **eval 本身需要持续演进**。如果 benchmark 的 task 和 grader 三年不变，Agent 一定会学会怎么骗过它。Living artifact + 定期 read transcripts 是唯一的解法。

---

## Q7. 如果让你为小红书 C 端 Agent 设计评测体系，你会怎么设计？

### 主答模板

**第一层：离线质量（每周 + 每次模型升级）**
- 构建 150 task 基准：capability（日常推荐/规划/搜索类意图 + 负面 case）+ regression（历史 badcase 固化）+ adversarial（安全/时效性/幻觉压测 + 主观性边缘 case）
- 每种 task 跑 10 个 trial，报均值和 bootstrap CI
- Graders：对硬事实（营业时间、价格、地点）走 code-based（接外部地图/点评 API 做交叉验证）；对主观维度（推荐理由、个性化、语气）走 LLM judge； 每季度 human SME 抽检 100 条做校准

**第二层：业务 KPI（每天）**
- 搜索到推荐的转化漏斗 completion rate
- 用户 follow-up rate（Agent 一次完成 vs 需要追问）
- Recommendation acceptance rate

**第三层：安全底线（实时）**
- PII 泄漏 = 0、医疗/法律建议 = 0、hate speech < 0.1%
- 每条 transcript 过 regex + classifier + keyword list

**Ship Gate：新 Agent 上限的离线门槛**
- Offline task suite 无明显 regression，adversarial bucket 通过率不低于基线
- 成本不超基线 20%
- 护栏检查通过后才上 10% A/B

---

## Q8. 你怎么看待"读 Transcript"这件事？为什么不能只靠分数？

### 主答模板
只靠分数你只知道"过了还是没过"，但你不知道"为什么没过"——也不知道"过了但差点没过"。一个 pass rate 90% 的 Agent 背后的 transcripts 里可能藏着一个 pattern：10% 的失败全部集中在某类任务上，而如果你不读 transcripts，你永远不会发现。

Anthropic 的核心建议是：**读 transcripts 不能外包给 LLM。** Designer 必须亲自读——不是所有 transcript，而是最极端的（最高分和最低分各 5 条 + 随机抽 5 条）。读完你才能发现 eval 的 task 是不是写错了、grader 是不是太松或太严、Agent 是不是走了你想不到但正确的路径。

---

# 四、你自己的评测指标速查表

面试官可能直接问你简历上的指标，每个都必须能脱口说出定义。

| 指标 | 你的定义（必须能讲出） | 算式 |
|------|---------------------|------|
| Agent 稳定性 | 相同 task N 次重复中输出一致的比例 | 一致次数 / N |
| 问题解决准确率 | 任务被正确完成的比例 | 正确完成数 / 总 task 数 |
| Memory 命中率 | 检索到的记忆被实际使用的比例 | 被引用条目数 / 检索返回条目数 |
| Skill 复用率 | 使用已有 Skill 而非重新检索的比例 | Skill 命中 task 数 / 总 task 数 |
| Recall@K | top-K 中包含正确答案的比例 | 标准 IR 指标 |
| MRR | 正确答案排名的倒数均值 | Σ(1/rank_i) / N |
| VQA11y-Score | 安全加权分数，Hard 安全关键题权重最高 | 0.2×S_easy + 0.3×S_medium + 0.5×S_hard |

---

# 五、高频总追问清单

## A. 基础概念
1. Agent 评测和 LLM 评测的三个根本差异是什么？
2. Transcript 和 Outcome 的区别？为什么不能只看 Transcript？
3. pass@k 和 pass^k 分别表示什么？各适合什么场景？
4. Capability Eval 和 Regression Eval 的区别是什么？

## B. Grader 设计
5. Code-based Grader 最适合做什么？不适合做什么？
6. LLM Judge 的三大最佳实践是什么？
7. 为什么 LLM Judge 必须有不确定逃生门（Uncertain Escape Hatch）？
8. 你怎么校准 LLM Judge？Cohen's Kappa 低于多少需要重写 prompt？

## C. 系统设计
9. 150 task benchmark 的四个 bucket 怎么划分？
10. 四级 Ship Gate 分别是什么？为什么 primary metric 不能做 early stop？
11. 三层质量门（Safety/Business/Offline Quality）各自的频率和指标？
12. Agent 评测最大的坑是什么？

## D. 你的项目
13. KTClaw 的 15%→3% 是怎么测的？每个指标的精确定义？
14. GoAfar 的路线可行率 60%→92% 是怎么测的？baseline 是什么？
15. LoCoMo 的核心评测维度是什么？你参考了哪几个？
16. VQA11y-Score 的三个权重各是多少？为什么 Hard 类权重最高？

## E. 小红书场景
17. 小红书 C 端 Agent 的离线评测和在线评测各怎么设计？
18. 推荐类 Agent 的排序指标有哪些？
19. 怎么防止 Agent 在评测集上学到了"看起来对"的捷径？

---

# 六、串联叙事：面试时最强的评测主线

> "我对 Agent 评测的理解，不是从用某个 benchmark 开始的，而是从自己项目里的失败教训开始的。在 KTClaw 里做 15%→3% 的时候，我意识到评测最难的不是定义指标，而是指标的口径——谁来判断'错误'、分母是什么、测试集怎么构造。后来在 GoAfar 里做路线可行率，我找到了最干净的一种评测方式——用确定性的 verifier 做 grader。这是所有 grader 里确定性最高的方案。

> 从业界实践来看，Anthropic 的 Agent 评测框架和我自己做项目的经验高度一致——核心原则就几条：Outcome > Transcript、code-based 做守门员 LLM judge 做主观评分、eval 是 living artifact 不是一次性产物。150 task benchmark 是最小可行的规模——50 capability + 50 regression + 30 adversarial + 20 multi-turn hard。

> 更重要的是，做 Agent 评测一定要 read transcripts。只靠分数你只知道过了没有，不知道原因。我在 KTClaw 里最深的体会就是——3% 的残余错误集中在哪里、是什么 pattern，不读 transcript 永远发现不了。"

---

# 七、最后：评测准备到什么程度，才算真准备好了？

## 你至少要做到 7 件事
1. **能讲出 Agent 评测和 LLM 评测的三个根本差异**
2. **能讲出三种 Grader 的优劣和组合原则**
3. **能解释 Transcript vs Outcome 为什么是 Agent 评测最大的坑**
4. **能画出 150 task benchmark 的四个 bucket 划分**
5. **能讲出四级 Ship Gate 的流程和关键原则**
6. **能说清 LLM Judge 的三大最佳实践和校准流程**
7. **能用这个框架解释你在 KTClaw 和 GoAfar 里做的评测**

## 如果你想让面试官"折服"，你要主动讲出来的 4 句话
1. **"Agent 评测最容易被忽视的陷阱是过于依赖 Transcript 而忽视 Outcome。Agent 说了什么不重要，环境里实际发生了什么才重要。"**
2. **"LLM Judge 不加 uncertain escape hatch，就等于默认允许 LLM 在不确定时强行打分——这是测量误差的最大来源。"**
3. **"做 Agent 评测的人必须亲自读 transcripts。不是所有的，但每个 eval run 至少读最高分、最低分和随机抽各 5 条。只靠分数做决策的人早晚要翻车。"**
4. **"我见过的最好的 Agent 评测实践，是把 eval 当成 living artifact——每次发现新的失败模式，就加一条 task 进 benchmark。这是 Mitchell Hashimoto 思想的评测版：每次发现一个 bug，就工程化地确保它永远能被检测到。"**
