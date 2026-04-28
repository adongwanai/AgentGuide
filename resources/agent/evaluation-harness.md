# 🧪 Evaluation Harness - AI 评估工具箱

> **精选** LLM / Agent / RAG 评估框架与工具，帮你快速选型、高效评测

---

## 📌 简介

**Evaluation Harness** 是用于系统化评估 AI 模型和 Agent 能力的测试框架，涵盖从标准 Benchmark 跑分到自定义场景测试的完整流程。本文档汇总了主流的评估工具，帮助你根据场景快速选型。

---

## 📑 目录

- [LLM 评估框架](#-llm-评估框架)
- [Agent 评估框架](#-agent-评估框架)
- [RAG 评估框架](#-rag-评估框架)
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
| **[Promptfoo](https://github.com/promptfoo/promptfoo)** | 5k+ | Node/Python | Prompt 评测与红队测试 | Prompt 工程、生产环境评测 | ⭐⭐ |
| **[DeepEval](https://github.com/confident-ai/deepeval)** | 3k+ | Python | LLM 评测框架，支持自定义指标 | 工程开发、CI/CD 集成 | ⭐⭐ |
| **[evals](https://github.com/openai/evals)** | 16k+ | Python | OpenAI 官方评测框架 | OpenAI 模型评测 | ⭐⭐⭐ |

---

## 🕵️ Agent 评估框架

> 用于评估 Agent 在真实任务中的规划、工具调用和执行能力

| 工具 | Stars | 语言 | 简介 | 适用场景 | 难度 |
|------|-------|------|------|----------|------|
| **[AgentBench](https://github.com/THUDM/AgentBench)** | 5k+ | Python | 清华出品，多维度 Agent 能力评测 | Agent 综合能力评估 | ⭐⭐⭐ |
| **[WebArena](https://github.com/web-arena-x/webarena)** | 3k+ | Python | 真实网站交互评测环境 | Web Agent 评估 | ⭐⭐⭐⭐ |
| **[GAIA](https://github.com/facebookresearch/gaia)** | 2k+ | Python | Meta 出品，通用 AI Agent 基准 | 真实任务完成能力评测 | ⭐⭐⭐ |
| **[ToolBench](https://github.com/OpenBMB/ToolBench)** | 8k+ | Python | 工具调用评测，16000+ 真实 API | 工具调用能力评估 | ⭐⭐⭐ |
| **[OSWorld](https://github.com/xlang-ai/OSWorld)** | 3k+ | Python | 多模态真实环境 Agent 评测 | 跨平台 Agent 评估 | ⭐⭐⭐⭐ |

---

## 📄 RAG 评估框架

> 用于评估 RAG 系统的检索质量、生成质量和端到端效果

| 工具 | Stars | 语言 | 简介 | 适用场景 | 难度 |
|------|-------|------|------|----------|------|
| **[RAGAs](https://github.com/explodinggradients/ragas)** | 5k+ | Python | RAG 评测框架，支持多维度指标 | RAG 系统质量评估 | ⭐⭐ |
| **[FlashRAG](https://github.com/RUC-NLPIR/FlashRAG)** | 4k+ | Python | 人大出品，RAG 训练与评测一体化 | RAG 研究与快速实验 | ⭐⭐⭐ |
| **[TruLens](https://github.com/truera/trulens)** | 3k+ | Python | RAG 可观测性与评测平台 | 生产环境 RAG 监控 | ⭐⭐⭐ |
| **[ARES](https://github.com/stanford-oval/ares)** | 1k+ | Python | 斯坦福出品，自动化 RAG 评测 | RAG 评测自动化 | ⭐⭐⭐ |

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
- [AI Agent 生产实践](./ai-agent-production-challenges.md) - 生产环境经验

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
