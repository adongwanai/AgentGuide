---
type: 资源清单
status: 已发布
level: 通用
topic:
  - Agent
  - 框架工具
---

# 外部资源核验记录

这里记录外部学习资源最近一次实际核验的日期、范围和适用限制。日期不是上游最后更新时间，也不是本仓库文件的修改日期。同一地址在多处出现时共用本页记录；没有记录的历史资源视为**未核验**。

## 怎样理解核验结果

- **链接**：访问了资源并检查最终目标，重定向后仍需确认目标内容。
- **内容/API**：阅读上游目录、README 或官方 API 文档，核对本仓库的推荐用途和版本口径。
- **运行**：实际安装依赖并运行示例，需要另附版本、命令与结果。下面的记录均未声称完成运行核验。

首次覆盖 README“新增优质资源”、综合课程入口及工程路线中更新的 LangChain 文档。其他清单保留未核验状态，不能用本页日期代表全仓库资源均已检查。维护时先看未核验的资源，再看框架版本已变化的教程；更新规则见[贡献指南](../CONTRIBUTING.md#外部资源核验)。

## 首页精选资源

| 资源 / 上游依据 | 最近核验 | 范围与适用限制 |
| --- | --- | --- |
| [learn-workbuddy](https://github.com/adongwanai/learn-workbuddy) | 2026-11-05 | 链接、README 与课程定位；桌面 Agent Harness 教学，不是 WorkBuddy 产品源码；未运行课程 |
| [Superpowers](https://github.com/obra/superpowers) | 2026-11-05 | 链接与 README；可组合 Skills 开发工作流；未在各支持工具中安装验证 |
| [learn-claude-code](https://github.com/shareAI-lab/learn-claude-code) | 2026-11-05 | 链接与课程目录；当前根目录主线 s01-s17，旧版课程需单独识别；未运行示例 |
| [claw0](https://github.com/shareAI-lab/claw0) | 2026-11-05 | 链接与 README；10 个渐进式 Python 章节，覆盖 loop、工具、会话和 gateway；未运行示例 |
| [hello-agents](https://github.com/datawhalechina/hello-agents) | 2026-11-05 | 链接与 README；智能体原理、框架、记忆、协议与评估课程；依赖按各章节核对，未运行示例 |
| [OpenClaw](https://github.com/openclaw/openclaw) | 2026-11-05 | 链接与 README；本地个人助手及消息渠道；未验证各渠道部署与生产可靠性 |
| [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) | 2026-11-05 | 原 engineering 地址重定向至 research；已阅读文章定位及更新提示；保留设计原则，2024 年的工具示例需参考当前文档 |
| [Vibe Coding 专栏](https://adongwanai.github.io/vibecoding/) | 2026-11-05 | 链接、首页与课程目录；免费基础课和 Premium 项目分开呈现；未核验付费内容 |

## 综合教程与课程

以下课程继续作为相应主题的学习参考；目录与定位核验不表示旧版 Notebook 能直接在最新依赖下运行。

| 资源 / 上游依据 | 最近核验 | 范围与适用限制 |
| --- | --- | --- |
| [LLM Universe](https://github.com/datawhalechina/llm-universe) / [阅读入口](https://datawhalechina.github.io/llm-universe/) | 2026-11-05 | 链接与 README；面向入门知识库/RAG 流程，运行需按该项目依赖；未作为当前 LangChain API 的依据 |
| [LLM Cookbook](https://github.com/datawhalechina/llm-cookbook) | 2026-11-05 | 链接与 README；吴恩达课程翻译/复现，保留概念入门用途；已从工程落地路线移除，旧 API 对照迁移指南 |
| [self-llm](https://github.com/datawhalechina/self-llm) | 2026-11-05 | 链接与 README；Linux 下的模型部署与微调教程，按模型目录核对环境；未运行示例 |
| [Dive into LLMs](https://github.com/Lordog/dive-into-llms) | 2026-11-05 | 链接与课程目录；微调、推理、对齐等编程实践；未运行实验 |
| [Happy-LLM](https://github.com/datawhalechina/happy-llm) | 2026-11-05 | 链接与目录；LLM 原理、架构和训练入门；未运行训练代码 |
| [AI-Guide-and-Demos](https://github.com/Hoper-J/AI-Guide-and-Demos-zh_CN) | 2026-11-05 | 链接与 README；从 API 调用进入模型应用和微调；未运行示例 |
| [Maxime Labonne Articles](https://mlabonne.github.io/blog/) | 2026-11-05 | 链接与文章目录；LLM 技术文章及部分 Notebook，版本和硬件需求按单篇核对；未运行 Notebook |
| [LLM-Action](https://github.com/liguodongiot/llm-action) | 2026-11-05 | 链接与目录；训练、推理及工程实践资料；未运行训练或部署 |

## 工程路线的 LangChain 文档

本次直接读取官方文档正文并核对以下概念与接口；对应[工程落地路线](../docs/05-roadmaps/learning-roadmap-development.md)。选择自己实际安装的版本，记录 Python、LangChain、provider 包和模型配置。没有执行需要 API Key 的模型调用。

| 官方文档 | 最近核验 | 已核对内容 |
| --- | --- | --- |
| [Quickstart](https://docs.langchain.com/oss/python/langchain/quickstart) | 2026-11-05 | `create_agent`、工具、消息输入与当前安装说明 |
| [Models](https://docs.langchain.com/oss/python/langchain/models) | 2026-11-05 | 模型初始化、invoke/stream/batch、并发限制与结构化输出 |
| [Agents](https://docs.langchain.com/oss/python/langchain/agents) | 2026-11-05 | Agent Loop、工具执行、middleware 与状态 |
| [Tools](https://docs.langchain.com/oss/python/langchain/tools) | 2026-11-05 | 工具定义、参数 schema 与运行上下文 |
| [Semantic Search / Knowledge Base](https://docs.langchain.com/oss/python/langchain/knowledge-base) | 2026-11-05 | 文档、分块、embedding、向量库与 retriever |
| [Retrieval](https://docs.langchain.com/oss/python/deepagents/retrieval) | 2026-11-05 | 原 LangChain retrieval 地址重定向到此页；2-step、Agentic 与 Hybrid RAG 的区别 |
| [SQL Agent](https://docs.langchain.com/oss/python/langchain/sql-agent) | 2026-11-05 | SQL 工具、`create_agent`、执行错误反馈；示例不等于生产权限隔离 |
| [Short-term Memory](https://docs.langchain.com/oss/python/langchain/short-term-memory) | 2026-11-05 | 消息状态、checkpointer、thread 与持久化；替换旧 ConversationBufferMemory 学习目标 |
| [Custom Middleware](https://docs.langchain.com/oss/python/langchain/middleware/custom) | 2026-11-05 | 模型/工具调用 hooks 与自定义失败处理 |
| [LangChain v1 Migration](https://docs.langchain.com/oss/python/migrate/langchain-v1) | 2026-11-05 | `create_agent` 与 middleware；旧组件迁入 `langchain-classic` 的范围 |
| [LangSmith Observability](https://docs.langchain.com/langsmith/observability) | 2026-11-05 | 执行轨迹、调试和可观测性入口 |

## 维护记录

- 2026-11-05：建立首批逐项核验日期，范围包括链接、资源定位及上表官方 API 文档。没有把构建通过、HTTP 成功或上游自报成绩记作示例运行通过。
- 2026-11-05：工程路线移除 LLM Cookbook 的两处推荐，替换旧版 LangChain quickstart、Agent、工具、SQL、memory、错误处理及异步/批量调用入口，更新对应学习目标。
