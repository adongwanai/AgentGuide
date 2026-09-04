---
type: 教程
status: 已归档
level: 入门
topic:
  - Agent
---

# Repo Gap Map：仓库补洞地图

> 本文记录对当前仓库空目录和占位内容的盘点，并给出后续补充优先级。

## 本次发现的空目录

| 目录 | 状态 | 建议补充 | 本次处理 |
|:---|:---|:---|:---|
| `docs/00-getting-started` | 空 | 新手入口、学习地图、7 天计划、资源筛选清单 | 已补 |
| `examples` | 空 | 最小 agent loop、eval 模板、trace 样例 | 已补 |
| `projects/01-paper-agent` | 空 | 论文研读 Agent 项目蓝图 | 已补 |
| `projects/02-travel-agent` | 空 | 旅行规划 Agent 项目蓝图 | 已补 |
| `projects/03-web-agent` | 空 | 浏览器/Web Agent 项目蓝图 | 已补 |
| `resources/multimodal` | 空 | 多模态 RAG、文档解析、视觉检索资源 | 已补 |
| `.trae/documents` | 工具隐藏目录 | IDE/工具私有文档，不建议放公开正文 | 暂不处理 |
| `.vercel` | 部署隐藏目录 | Vercel 本地链接信息，不建议手动填内容 | 暂不处理 |

## 仍值得继续补的占位文档

这些文件不在空目录里，但内容仍是 placeholder，后续优先级较高：

| 优先级 | 文件 | 推荐方向 |
|:---:|:---|:---|
| P0 | `docs/01-theory/01-what-is-agent.md` | Agent 定义、能力分级、与 workflow 的区别 |
| P0 | `docs/01-theory/04-react-framework.md` | ReAct 原理、最小实现、失败模式 |
| P0 | `docs/01-theory/05-cot-and-planning.md` | CoT、Plan-and-Solve、树搜索、反思 |
| P0 | `docs/01-theory/09-evaluation-metrics.md` | 任务成功率、轨迹评分、成本、延迟、安全指标 |
| P1 | `docs/02-tech-stack/04-langchain-guide.md` | LangChain 与 LangGraph 的边界，现代用法 |
| P1 | `docs/02-tech-stack/06-multi-agent-frameworks.md` | AutoGen、CrewAI、Swarm、Magentic-One 对比 |
| P1 | `docs/02-tech-stack/10-mcp-protocol.md` | MCP 架构、server/client、权限和安全 |
| P1 | `docs/02-tech-stack/20-rag-full-pipeline.md` | 文档解析、chunk、embedding、rerank、eval |
| P1 | `docs/02-tech-stack/build-your-agent-framework.md` | 从零手写 agent loop、tool registry、trace |
| P2 | `docs/03-practice/02-high-availability-rag.md` | 生产级 RAG 可用性、缓存、降级、监控 |
| P2 | `docs/03-practice/03-agent-security.md` | Prompt injection、权限、sandbox、secret 管理 |
| P2 | `docs/03-practice/04-graduation-project.md` | 毕设选题、范围控制、评测和论文写法 |

## 后续补内容的原则

1. **先入口，再深水区**：新手先能跑通 1 个项目，再读大量框架细节。
2. **先项目，再资源堆叠**：每个资源最好能服务一个具体项目。
3. **先一手来源**：官方文档、论文、GitHub README 优先，二手博客只做补充。
4. **每篇文档都要有产出物**：笔记、代码、评测集、架构图、简历 bullet 至少一个。
5. **避免只写概念**：AgentGuide 的定位是求职和实战导向，文档必须能指导动手。

## 推荐下一轮补充

最值得下一轮完成的是：

1. `docs/01-theory/01-what-is-agent.md`
2. `docs/01-theory/04-react-framework.md`
3. `docs/02-tech-stack/10-mcp-protocol.md`
4. `docs/03-practice/03-agent-security.md`

这四篇补完后，仓库会形成“概念 -> loop -> 协议 -> 安全 -> 项目”的更完整闭环。
