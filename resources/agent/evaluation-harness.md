---
type: 资源清单
status: 已发布
level: 通用
topic:
  - Agent
  - 评测
  - 框架工具
---

# 🧪 Evaluation Harness - AI 评估工具箱

> **精选** LLM / Agent / RAG 评估框架与工具，帮你快速选型、高效评测

---

## 📌 简介

**Evaluation Harness** 是用于系统化评估 AI 模型和 Agent 能力的测试框架，涵盖从标准 Benchmark 跑分到自定义场景测试的完整流程。本文档汇总了主流的评估工具，帮助你根据场景快速选型。

---

## 📑 目录

- [LLM 评估框架](#-llm-评估框架)
- [Agent 评估框架](#-agent-评估框架)
- [Agent 执行与轨迹评测](#agent-evaluation-runtime)
- [标准 Agent 基准](#agent-benchmarks)
- [RAG 评估框架](#-rag-评估框架)
- [RAG 检索与生成分面诊断](#rag-diagnostics)
- [自建 Harness 工具包](#-自建-harness-工具包)
- [选型建议](#-选型建议)
- [相关文档链接](#-相关文档链接)

---

## 🤖 LLM 评估框架

> 用于评估大语言模型的基础能力（推理、知识、代码等）

| 工具 | Stars | 语言 | 简介 | 适用场景 | 难度 |
|------|-------|------|------|----------|------|
| **[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)** | 10k+ | Python | EleutherAI 出品，支持 300+ 标准任务 | 学术研究、模型对比 | ⭐⭐⭐ |
| **[OpenCompass](https://github.com/open-compass/opencompass)** | 8k+ | Python | 上海 AI Lab 出品，国产模型评测首选 | 国产模型评测、学术研究 | ⭐⭐⭐ |
| **[Lighteval](https://github.com/huggingface/lighteval)** | 2k+ | Python | HuggingFace 出品，轻量高效 | HF 生态集成、快速评测 | ⭐⭐ |
| **[Promptfoo](https://github.com/promptfoo/promptfoo)** | — | Node/Python | Prompt / Agent / RAG 测试与红队检查 | 对抗鲁棒性、回归与 CI 门禁 | ⭐⭐ |
| **[DeepEval](https://github.com/confident-ai/deepeval)** | — | Python | pytest 风格评测，组合任务完成、工具调用、RAG 和自定义指标 | 轨迹诊断、LLM-as-judge 与 CI/CD 集成 | ⭐⭐ |
| **[evals](https://github.com/openai/evals)** | 16k+ | Python | OpenAI 官方评测框架 | OpenAI 模型评测 | ⭐⭐⭐ |

---

## 🕵️ Agent 评估框架

> 用于评估 Agent 在真实任务中的规划、工具调用和执行能力

| 工具 | Stars | 语言 | 简介 | 适用场景 | 难度 |
|------|-------|------|------|----------|------|
| **[AgentBench](https://github.com/THUDM/AgentBench)** | 5k+ | Python | 清华出品，多维度 Agent 能力评测 | Agent 综合能力评估 | ⭐⭐⭐ |
| **[WebArena](https://github.com/web-arena-x/webarena)** | 3k+ | Python | 真实网站交互评测环境 | Web Agent 评估 | ⭐⭐⭐⭐ |
| **[GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA)** | — | 数据集 | 真实世界多步工具与信息处理任务 | 通用 Agent 评估；区分 text-only 子集与全量任务 | ⭐⭐⭐ |
| **[ToolBench](https://github.com/OpenBMB/ToolBench)** | 8k+ | Python | 工具调用评测，16000+ 真实 API | 工具调用能力评估 | ⭐⭐⭐ |
| **[OSWorld](https://github.com/xlang-ai/OSWorld)** | 3k+ | Python | 多模态真实环境 Agent 评测 | 跨平台 Agent 评估 | ⭐⭐⭐⭐ |

<a id="agent-evaluation-runtime"></a>

### Agent 执行与轨迹评测

以下内容于 2026-11-05 合并自 [Harness 项目选型指南](../project-catalogs/agent-harness-projects.md)。上面的 AgentBench、WebArena、GAIA 等定义任务/环境；下面的工具侧重执行、评分和诊断，不能把“支持一个框架”理解为自己的 Agent 无需适配就能跑全部基准。

| 工具 / 入口 | 用途 | 接入时需要确认 |
| --- | --- | --- |
| [Youtu-Agent](../project-catalogs/agent-harness-projects.md#5-垂直应用与-rag-深水区) | 复用已有项目条目中的 Agent 运行、评测与经验学习管线 | 任务格式、工具接口、评分器和模型/预算配置 |
| [AgentCompass](https://github.com/open-compass/AgentCompass) | 解耦 Model、Benchmark、Harness、Environment；统一组合不同任务、Agent 与执行后端 | Harness adapter、沙箱环境、失败分类与轨迹记录；[论文](https://arxiv.org/abs/2607.13705) |
| [Exgentic](https://github.com/Exgentic/exgentic) / [Open Agent Leaderboard](https://huggingface.co/spaces/open-agent-leaderboard/leaderboard) | 面向完整 Agent 系统的评测与公开结果比较 | 使用相同任务、模型/工具配置，结合质量、成本与运行记录解释结果 |
| [Harbor](https://github.com/harbor-framework/harbor) | 用标准任务与隔离环境执行长程 Agent 评测 | 任务版本、容器镜像、资源限制、超时与 verifier |
| [agentevals](https://github.com/langchain-ai/agentevals) | Agent trajectory 与工具调用序列评测 | 精确/宽松路径匹配、工具参数和可接受的等价轨迹 |
| [openevals](https://github.com/langchain-ai/openevals) | 可组合的输出评分器，包括 LLM-as-judge | rubric、judge 版本、人工校准和误判样本 |
| [Phoenix / Langfuse](../rag/projects.md#监控与可观测性) | 复用既有观测平台条目，采集执行链、查看轨迹并关联评测 | trace 与 task/run ID 的关联、敏感数据处理、用量记录 |

DeepEval 和 Promptfoo 在上方原条目维护，不重复新增项目卡片。过程评测至少观察：工具与参数是否正确、是否空转或反复回退、任务何时终止、失败能否归因。最终答案正确与执行过程高效应分别报告。

<a id="agent-benchmarks"></a>

### 标准 Agent 基准

将原文嵌入表格中的五组基准展开如下；GAIA 复用上方条目。选与目标任务相近的一组，不需要全部跑。

| 基准 | 主要测什么 | 报告结果时的边界 |
| --- | --- | --- |
| [τ-bench](https://github.com/sierra-research/tau-bench) / [τ²-bench](https://github.com/sierra-research/tau2-bench) | 多轮用户-工具交互、策略遵守与任务状态；τ² 包含双边交互设定 | 明确版本、领域、用户模拟器和重复试验，不混用不同数据版本成绩 |
| [SWE-bench](https://github.com/SWE-bench/SWE-bench) / [Terminal-Bench](https://www.tbench.ai/) | 真实仓库 issue 修复 / 终端环境任务，以执行式 verifier 验证 | 明确子集、环境、超时和 verifier；Terminal-Bench 的任务版本要与 runner 匹配 |
| GAIA（见上） | 多步工具、网页与文件信息处理 | 明确 split、level、是否含多模态，不能用文本验证子集代替全量 |
| [WebWalkerQA](https://huggingface.co/datasets/callanwu/WebWalkerQA) | 多步网页信息搜索、导航与问答 | 记录网页/检索状态和工具配置，区分环境故障与任务失败 |
| [BFCL](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) | 函数调用、参数与工具交互能力 | 明确 benchmark 版本和子任务；工具调用分数不等同于端到端 Agent 完成率 |

比较 Harness 时固定模型、数据划分、工具权限和预算；比较模型时固定 Harness。保存逐题结果、失败原因、有效分母、随机性/重复次数、token、费用与延迟。不要只抄 leaderboard 的最高分或跨协议比较不同项目的 README 数字。

---

## 📄 RAG 评估框架

> 用于评估 RAG 系统的检索质量、生成质量和端到端效果

| 工具 | Stars | 语言 | 简介 | 适用场景 | 难度 |
|------|-------|------|------|----------|------|
| **[RAGAS](https://github.com/vibrantlabsai/ragas)** | — | Python | 分别检查答案与检索上下文，支持参考答案/参考证据驱动的评测 | RAG 系统质量与失败面诊断 | ⭐⭐ |
| **[FlashRAG](https://github.com/RUC-NLPIR/FlashRAG)** | 4k+ | Python | 人大出品，RAG 训练与评测一体化 | RAG 研究与快速实验 | ⭐⭐⭐ |
| **[TruLens](https://github.com/truera/trulens)** | 3k+ | Python | RAG 可观测性与评测平台 | 生产环境 RAG 监控 | ⭐⭐⭐ |
| **[ARES](https://github.com/stanford-oval/ares)** | 1k+ | Python | 斯坦福出品，自动化 RAG 评测 | RAG 评测自动化 | ⭐⭐⭐ |

<a id="rag-diagnostics"></a>

### RAG 检索与生成分面诊断

| 维度 | 要回答的问题 | 数据与解释边界 |
| --- | --- | --- |
| Faithfulness | 答案中的主张是否由检索上下文支持 | 支持性高不代表必要证据齐全，也不等于外部事实正确 |
| Answer Relevancy | 答案是否回应用户问题 | 与正确率、完整性分开看 |
| Context Precision | 检索内容是否相关，相关内容是否排在前面 | 指标实现可能使用排序和 judge，注明版本及参考数据 |
| Context Recall | 回答所需证据是否被检索到 | 通常需要参考答案、gold evidence 或标注信息；多跳任务关注证据链完整性 |

例如法律问答漏检第二条关键法条，生成器仍可能忠实复述已取回的第一条，导致 faithfulness 高而召回不足。这是失败机制示例，不是本仓库实测案例。GraphRAG 与向量 RAG 的对照要同时报告证据召回和最终答案质量，并计入建图、索引与查询成本。

阈值应按领域风险、人工标注和验证集校准，不把 `0.75`、`0.8` 等示例值当作通用合格线。可以用 RAGAS 做分面分析、用 DeepEval 等纳入回归，但接入后仍需验证评分器、失败判定和成本。

---

## 🛠️ 自建 Harness 工具包

> 当现成框架无法满足需求时，可以用这些工具快速搭建自定义评测流程

| 工具 | 用途 | 说明 |
|------|------|------|
| **pytest + LLM Mock** | 🧪 单元测试 | 用 mock 替代 LLM 调用，测试 Agent 逻辑分支 |
| **Docker Sandbox** | 🐳 安全执行 | 沙箱环境执行 Agent 生成的代码，防止副作用 |
| **Playwright** | 🌐 Web Agent 测试 | 自动化浏览器操作，测试 Web Agent 的页面交互能力 |

### 快速搭建示例

```python
# pytest + LLM Mock 示例
from unittest.mock import patch

def test_agent_tool_call():
    with patch("your_agent.llm_call") as mock_llm:
        mock_llm.return_value = '{"action": "search", "query": "test"}'
        result = agent.run("帮我搜索 test")
        assert result["action"] == "search"
```

---

## 🎯 选型建议

| 场景 | 推荐工具 | 理由 |
|------|----------|------|
| 🎓 **学术研究** | lm-evaluation-harness / OpenCompass | 任务覆盖广、社区认可度高、论文引用多 |
| 💻 **工程开发** | Promptfoo / DeepEval | 易集成、支持 CI/CD、自定义指标灵活 |
| 🕵️ **Agent 评估** | AgentBench / WebArena | 真实任务场景、评测维度全面 |
| 📄 **RAG 评估** | RAGAs / TruLens | 指标丰富、生产环境友好 |

> **💡 提示**：如果不确定选哪个，先从 **Promptfoo**（工程向）或 **lm-evaluation-harness**（学术向）入手，再根据需求扩展。

---

## 🔗 相关文档链接

### 本项目文档

- [Agent 评估完全指南](./README.md#-评估-benchmark) - 评估 Benchmark 汇总
- [Agent 开发框架](./frameworks.md) - 框架选型指南
- [Agent 记忆系统](./memory.md) - 记忆模块设计
- [AI Agent 生产实践](../../docs/03-practice/06-agent-production-challenges.md) - 生产环境经验

### 学习路线

- [Agent 开发学习路线](../../docs/05-roadmaps/learning-roadmap-development.md)
- [算法面试准备路线](../../docs/05-roadmaps/learning-roadmap-algorithm.md)

---

## 📝 贡献指南

**🙏 欢迎贡献**：发现好的评估工具或框架？欢迎提 PR！

### 贡献要求

1. 工具必须与 AI 评估**直接相关**
2. 提供清晰的**简介**和**适用场景**
3. 标注**学习难度**（⭐~⭐⭐⭐⭐⭐）
4. 优先推荐**实际使用过**、**社区活跃**的工具

---

## 📄 License

本文档采用 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) 协议。
