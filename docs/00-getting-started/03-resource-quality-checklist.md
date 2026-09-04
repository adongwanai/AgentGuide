---
type: 教程
status: 已发布
level: 入门
topic:
  - Agent
---

# Resource Quality Checklist：高质量资源筛选清单

> Agent 资料很多，但真正值得花时间的资源通常有共同特征：可运行、可验证、可追溯、能解释失败。

## GitHub 项目筛选

| 维度 | 高质量信号 | 低质量信号 |
|:---|:---|:---|
| 可运行性 | 有明确安装、环境变量、demo、测试命令 | 只有截图和口号 |
| 维护状态 | 最近仍有 commit / issue / release | 长期无人维护，依赖版本锁死 |
| 架构透明 | README 解释 agent loop、工具、状态、eval | 只写“multi-agent powered by GPT” |
| 工具边界 | 工具 schema 清晰，权限和错误处理明确 | 工具直接执行任意 shell/API |
| 评测 | 有 benchmark、eval case、失败分析 | 只有单次 demo 成功 |
| 安全 | 有 sandbox、human approval、secret 管理 | 要求把高权限 key 放进 prompt 或前端 |
| 可复用 | 模块化，能替换模型和工具 | 所有逻辑堆在一个 notebook |

## 教程筛选

优先读这些类型：

- 官方文档：OpenAI Agents SDK、LangGraph、Pydantic AI、MCP。
- 工程博客：解释为什么这样设计，而不是只贴代码。
- 可运行 cookbook：有完整依赖、输入、输出和测试方式。
- 论文配套代码：能复现实验或至少提供数据/脚本。

谨慎对待这些类型：

- “10 分钟实现 AutoGPT”但没有失败处理。
- “生产级 Agent”但没有 trace、eval、权限分级。
- “全自动赚钱/投递/下单”但没有法律和安全边界。
- 只堆框架名，不解释 trade-off。

## 论文筛选

| 问题 | 为什么重要 |
|:---|:---|
| 任务定义是否清楚？ | Agent 论文很容易把开放任务写得漂亮但不可复现 |
| 环境是否可复现？ | Web、OS、工具 API 会变化，benchmark 必须控制漂移 |
| 指标是否只看最终成功率？ | 还要看步数、成本、延迟、错误类型、人工介入次数 |
| 是否有消融实验？ | 没有消融就很难知道是模型强，还是框架设计有效 |
| 是否报告失败案例？ | Agent 的失败模式比成功样例更有学习价值 |
| 是否开源代码/数据？ | 不能复现就只能当思路参考 |

## 项目是否值得写进简历

满足下面 8 条中的 6 条，就值得继续打磨：

- 有明确用户和业务场景。
- 有至少 3 个真实工具，不是 mock 全流程。
- 有可追踪 trace 或日志。
- 有 20 条以上 eval case。
- 有失败类型统计。
- 有权限/安全设计。
- 有部署或一键运行说明。
- 有对比实验，比如有无 RAG、有无 rerank、有无 memory。

## 推荐优先资源

### Agent 架构

- [Anthropic: Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)
- [OpenAI Agents SDK](https://platform.openai.com/docs/guides/agents-sdk/)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Pydantic AI](https://github.com/pydantic/pydantic-ai)
- [Microsoft AutoGen](https://github.com/microsoft/autogen)

### 工具与协议

- [Model Context Protocol](https://github.com/modelcontextprotocol/modelcontextprotocol)
- [OpenAI Swarm](https://github.com/openai/swarm)
- [Anthropic: Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents)

### 评测与安全

- [Promptfoo](https://github.com/promptfoo/promptfoo)
- [DeepEval](https://github.com/confident-ai/deepeval)
- [Inspect](https://inspect.aisi.org.uk/)
- [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications)
- [OWASP MCP Top 10](https://owasp.org/www-project-mcp-top-10/)

### 多模态与文档处理

- [Docling](https://github.com/docling-project/docling)
- [MinerU](https://github.com/opendatalab/MinerU)
- [RAG-Anything](https://github.com/HKUDS/RAG-Anything)
- [ColPali paper](https://arxiv.org/abs/2407.01449)

## 最后一句判断

好资源会让你更快回答“为什么这么设计、失败时怎么查、指标怎么证明”；差资源只会让你复制更多样板代码。
