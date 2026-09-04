---
type: 资源清单
status: 已发布
level: 通用
topic:
  - 项目实战
  - Agent
---

# 优质 End-to-End Agent 开源项目

> 选择标准：能运行、有真实任务、有清晰架构、有持续维护价值，最好还能学习 trace、eval、工具权限或多 Agent 协作。

不要只看 stars。Agent 项目真正值得学习的地方通常在：工具边界、状态管理、失败恢复、评测、部署和安全策略。

## 学习顺序建议

| 阶段 | 目标 | 推荐项目 |
|:---:|:---|:---|
| 1 | 看懂单 Agent 如何完成研究任务 | GPT Researcher |
| 2 | 看懂 coding agent 如何操作真实代码库 | OpenHands、SWE-agent |
| 3 | 看懂 multi-agent 如何协作 | AutoGen、Magentic-One、CrewAI examples |
| 4 | 看懂 workflow / graph 如何承载长任务 | LangGraph examples |
| 5 | 看懂评测和 benchmark | BrowserGym、WebArena、SWE-bench 生态 |

## 研究与报告类 Agent

### [GPT Researcher](https://github.com/assafelovic/gpt-researcher)

**定位**：开源深度研究 Agent，围绕 query 做资料搜索、信息聚合和报告生成。

**适合学习**：

- research pipeline 如何拆解。
- 如何管理资料来源和引用。
- 如何从“搜索结果”走到结构化报告。

**可以借鉴到**：

- [Paper Agent](../../projects/01-paper-agent/README.md)
- 竞品分析 Agent
- 行业调研 Agent

### [STORM / OpenScholar](https://github.com/stanford-oval/storm)

**定位**：面向长文写作和学术综述的研究系统。

**适合学习**：

- 如何组织多来源材料。
- 如何在生成前做提纲和信息覆盖。
- 如何把研究任务拆成可审查阶段。

## Coding Agent

### [OpenHands](https://github.com/All-Hands-AI/OpenHands)

**定位**：面向软件开发任务的开源 Agent 平台，前身为 OpenDevin。

**适合学习**：

- coding agent 的 workspace、shell、浏览器、文件编辑和状态管理。
- 用户交互、权限边界、长任务执行。
- 真实代码库中的失败恢复。

**重点看**：

- Agent 如何观察仓库。
- 工具调用如何被记录和展示。
- 如何处理用户确认和任务中断。

### [SWE-agent](https://github.com/SWE-agent/SWE-agent)

**定位**：针对 GitHub issue 自动修复的软件工程 Agent。

**适合学习**：

- 如何把 GitHub issue 转成修复任务。
- 如何设计 shell/file 工具给模型使用。
- 如何在 SWE-bench 这类 benchmark 上评估。

**适合改造成**：

- 课程作业自动修复助手。
- 内部代码库 issue triage / patch assistant。

### [Agentless](https://github.com/OpenAutoCoder/Agentless)

**定位**：SWE-bench 方向的轻量修复方法，强调不依赖复杂自主 Agent。

**适合学习**：

- 为什么有些任务不需要复杂 agent loop。
- 问题定位、补丁生成、补丁验证可以如何拆解。
- “更简单的 pipeline”在某些任务上为什么更稳定。

## Multi-Agent 与通用框架

### [Microsoft AutoGen](https://github.com/microsoft/autogen)

**定位**：微软开源多 Agent 框架，支持多角色会话、工具和应用构建。

**适合学习**：

- 多 Agent 对话模式。
- planner / executor / reviewer 分工。
- Agent Studio 和工程化集成。

### [Magentic-One](https://github.com/microsoft/autogen/tree/main/python/packages/autogen-magentic-one)

**定位**：微软在 AutoGen 生态中的通用 multi-agent 系统。

**适合学习**：

- Web、文件、代码和终端工具如何组合。
- 通用任务 Agent 的协调方式。
- 多 Agent 系统的复杂度和安全边界。

### [CrewAI](https://github.com/crewAIInc/crewAI)

**定位**：以角色、任务、流程为核心的 multi-agent 框架。

**适合学习**：

- 角色驱动的协作任务。
- crew、agent、task、process 的抽象。
- 快速搭建内容、研究、运营类 workflow。

### [OpenAI Swarm](https://github.com/openai/swarm)

**定位**：OpenAI 发布的 educational framework，强调 routines 和 handoffs。

**适合学习**：

- handoff 思想。
- 极简 multi-agent 抽象。
- 为什么轻框架更适合学习概念。

注意：Swarm 定位偏教育，不建议直接当生产框架。

## Graph / Workflow 型 Agent

### [LangGraph](https://github.com/langchain-ai/langgraph)

**定位**：用图结构构建有状态、可恢复、可人工介入的 Agent/workflow。

**适合学习**：

- 状态图如何表达复杂流程。
- durable execution、persistence、human-in-the-loop。
- 多 Agent 和长任务如何做恢复。

### [LlamaIndex Workflows](https://github.com/run-llama/llama_index)

**定位**：围绕数据连接、索引、检索和 workflow 的 LLM 应用框架。

**适合学习**：

- 数据密集型 Agent。
- RAG 与 tool use 如何结合。
- 文档/数据库/知识库连接器生态。

## Web / Computer-Use Agent

### [BrowserGym](https://github.com/ServiceNow/BrowserGym)

**定位**：Web Agent 训练和评测环境，提供 browser-based tasks。

**适合学习**：

- Web Agent 任务如何标准化。
- 观察、动作、奖励/评分如何定义。
- 如何复现网页任务。

### [WebArena](https://webarena.dev/)

**定位**：真实网站环境中的 Web Agent benchmark。

**适合学习**：

- 真实网页任务的复杂性。
- 账号、状态、站点环境如何控制。
- Web Agent 评测为什么难。

### [OSWorld](https://github.com/xlang-ai/OSWorld)

**定位**：桌面操作系统环境中的 multimodal computer-use benchmark。

**适合学习**：

- GUI/OS Agent 的观察与动作空间。
- 截图、鼠标、键盘和应用状态如何进入评测。
- 真实桌面任务的失败模式。

## 文档与多模态 Agent

### [Docling](https://github.com/docling-project/docling)

**定位**：文档解析和转换工具，适合进入 RAG/Agent 系统。

**适合学习**：

- PDF、Office、HTML、图片如何转结构化文档。
- 表格、OCR、版面和导出格式。
- 文档智能 Agent 的输入层。

### [MinerU](https://github.com/opendatalab/MinerU)

**定位**：面向 LLM/RAG/Agent 的高质量文档解析工具。

**适合学习**：

- 复杂 PDF 和扫描件解析。
- Markdown/JSON 输出。
- 中文和多语言文档处理。

### [RAG-Anything](https://github.com/HKUDS/RAG-Anything)

**定位**：多模态 All-in-One RAG 框架。

**适合学习**：

- 文本、图片、表格、公式如何统一进入 RAG。
- 多模态知识图谱和混合检索。
- 如何把文档解析和检索连接起来。

## Eval / Observability

### [Promptfoo](https://github.com/promptfoo/promptfoo)

**定位**：LLM 应用测试、回归评测和红队工具。

**适合学习**：

- 用 YAML/CLI 管理 eval case。
- prompt、RAG、Agent 输出回归测试。
- CI 里如何做质量门禁。

### [DeepEval](https://github.com/confident-ai/deepeval)

**定位**：pytest 风格的 LLM evaluation 框架。

**适合学习**：

- 把 LLM/Agent 行为写成自动化测试。
- RAG、tool use、faithfulness 等指标。
- 开发阶段快速回归。

### [Inspect](https://inspect.aisi.org.uk/)

**定位**：UK AI Security Institute 开源的模型评测框架。

**适合学习**：

- 更规范的评测任务、solver、scorer 抽象。
- 安全和能力评测如何组织。
- 复杂 eval pipeline 如何复现。

## 怎么从开源项目里学

建议按这个顺序读：

1. `README`：项目目标、安装、demo。
2. `examples/`：最小可运行样例。
3. `tests/`：作者认为重要的行为边界。
4. `docs/architecture` 或核心入口文件：agent loop、tool registry、state。
5. issue / discussions：真实用户遇到的失败模式。

输出一页项目笔记：

```markdown
# 项目名

## 它解决什么问题
## Agent loop 在哪里
## 工具有哪些
## 状态和上下文怎么管理
## 如何评测
## 有哪些失败模式
## 我可以借鉴到自己的项目什么
```

## 贡献建议

新增项目时请说明：

- 项目定位。
- 是否能本地运行。
- 学习价值。
- 适合借鉴到哪个 AgentGuide 项目。
- 是否有评测、trace、安全边界。
