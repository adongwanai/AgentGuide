---
type: 资源清单
status: 已发布
level: 进阶
topic:
  - Agent
  - 项目实战
  - 评测
---

# Agent Harness 开源项目：选型、二开与面试证据

按学习目标选择一个可运行的 Harness，完成一项明确的二次开发，再用基线、轨迹和评测结果证明改动有效。这里覆盖入门教程、轻量运行时、统一控制层、企业平台、垂直应用与评测研究，适合准备 Agent 开发、算法和 RAG 岗位。

更新时间：2026-11-05。来源为[飞书项目推荐文档](https://ccn7vpu5l5y8.feishu.cn/wiki/VjYuwmkEIi8bMzkkHzGcbAeWnyf)，已结合上游仓库核对并与 AgentGuide 既有条目合并。项目功能会随版本变化，开始实践时记录 commit、依赖和配置；本文不使用静态 Star 数作为排序依据。

## 先确定要改哪一层

Harness 负责让模型持续执行任务：组织 Agent Loop、工具、状态、上下文、权限、执行环境和观测。Framework 提供开发这些能力的抽象；二者可以组合，例如 Deep Agents 建在 LangChain / LangGraph 之上。理解模块边界比争论名称更有用，机制详见 [Agent Harness 工程指南](../../docs/02-tech-stack/27-agent-harness-engineering.md)。

| 目标 | 建议起点 | 最小可验收产物 |
| --- | --- | --- |
| 学清 Agent Loop、上下文与隔离 | learn-claude-code → nanobot / Pi | 一张调用链图、一组失败复现与回归用例 |
| 自研轻量 Harness | OpenHarness / cc-mini | 一个工具或 Skill，以及权限和失败测试 |
| 统一多种 Coding Agent | lite-harness | 同任务、同预算的多后端适配与对照记录 |
| 学全栈插件架构 | DeepSeek Harness | 一个插件、配置说明与加载/恢复证据 |
| 做研发任务自动化 | Deep Agents / Symphony | 从任务输入到补丁、验证和人工接收的闭环 |
| 企业平台或数字员工 | DeerFlow / Microsoft Agent Framework / StaffDeck | 权限边界、状态恢复和审计日志 |
| 前端 / 全栈切入 | PilotDeck / oh-my-openagent | 工作区隔离、可追踪的任务流程 |
| Skills 自进化 | PenguinHarness / Hive | 固定留出集上的改进对照和成本记录 |
| 国内 IM 助理 | nanobot / CowAgent | 一个渠道或领域 Skill，验证跨会话隔离 |
| 垂直学习或知识产品 | DeepTutor / WeKnora / Yuxi | 领域数据集、引用溯源和检索/生成分面评测 |
| GraphRAG 或 Agent 算法 | Youtu-GraphRAG / Youtu-Agent | 明确的 baseline、消融实验和失败归因 |

首次做项目，选一个基础系统，再选一个评测入口即可；需要的软件开发、浏览器和多智能体参考，继续看 [End-to-End Agent 项目](./end-to-end-agent-projects.md)。

## 1. 入门与轻量实现

### learn-claude-code：先把内核跑通

沿用 [README 已有教程入口](../../README.md#新增优质资源)。截至核对日，上游当前主线是根目录 `s01-s17`，旧版课程另行保留；不要把不同版本的章节编号混用。

建议先跑 `s01` Agent Loop，再看当前主线的 `s08` Context Compact、`s13` Task-bound Worktrees，最后用 `s15` Integrated Harness 串起机制。对照输入、工具结果和状态变化画出执行过程，再为压缩前后的任务成功率、token 消耗及任务隔离写回归用例。

### 轻量项目对照

| 项目 | 适合精读的机制 | 二开切入点 | 验收证据 |
| --- | --- | --- | --- |
| [HKUDS/OpenHarness](https://github.com/HKUDS/OpenHarness) | 工具、Skills / Plugins、记忆、多 Agent、权限、MCP；内置个人 Agent ohmo | 增加领域工具、provider 或配置预检 | 工具授权测试、配置诊断、失败 trace |
| [HKUDS/nanobot](https://github.com/HKUDS/nanobot) | 个人 Agent 的 loop、记忆、Skills、MCP、定时任务、多渠道接入和 Python SDK | 接一个业务渠道或领域 Skill；接入现有应用 | 渠道隔离、任务完成率、调用成本与可恢复失败 |
| [Pi](https://github.com/earendil-works/pi) | 统一模型 API、Agent Loop、TUI 与 Coding Agent CLI | 对照少量核心工具与按需扩展的取舍 | 同任务工具调用数、上下文开销、任务成功率 |
| [oh-my-pi](https://github.com/can1357/oh-my-pi) | Pi 衍生的 Coding Agent，扩展编辑与开发工具体验 | 对照上游研究新增工具和编辑工作流 | 补丁正确率、诊断信息、测试通过记录 |
| [e10nMa2k/cc-mini](https://github.com/e10nMa2k/cc-mini) | 轻量 Claude Code 风格 Harness 脚手架 | 从最小 loop 增加一个必要机制 | 能运行的最小示例与新增机制回归测试 |

OpenHarness 的 `oh --dry-run` 可作为配置可观测性的源码阅读点：检查 resolved settings、认证状态、工具、Skills 与 MCP，给出 ready / warning / blocked。核对时该能力位于上游 README 的 **Unreleased** 区，使用前确认安装版本支持。Provider 预设、插件和 Docker 后端也应按目标版本验证。

Pi 的旧仓库地址 `badlogic/pi-mono` 已跳转到上表地址，按同一项目处理；oh-my-pi 是独立衍生项目，保留单独入口。nanobot 的学习路线见[求职路线](../../docs/05-roadmaps/agent-job-ready-roadmap-2026.md)，此处只补源码阅读与二开方向。

## 2. 统一控制层与插件架构

| 项目 | 定位与阅读重点 | 二开切入点 |
| --- | --- | --- |
| [LiteLLM-Labs/lite-harness](https://github.com/LiteLLM-Labs/lite-harness) | 以统一 API 调用 Claude Code、Codex；关注后端差异如何映射到同一接口，其他后端按当前支持清单核对 | 增加 adapter，记录取消、超时、会话和成本语义的差异 |
| [deepseek-ai/deepseek-harness](https://github.com/deepseek-ai/deepseek-harness) | “Everything is a Plugin”；阅读 Cordis 插件依赖，以及模型、工具、session、存储、沙箱、调度与 UI 的边界 | 增加 provider / 工具 / 存储插件，验证加载、卸载与恢复行为 |

两者解决的问题不同：统一接口强调多 Agent 后端兼容，插件化强调系统内部组件可替换。对照时报告接入范围、状态语义和异常行为，不能仅以“支持更多模型”概括工程贡献。DeepSeek Harness 的前沿研究入口已在[研究方向专题](../../docs/06-research-frontiers/01-ai-research-directions-expanded.md)中收录。

## 3. 官方生态与企业平台

| 项目 | 主要能力与适用场景 | 二开切入点 |
| --- | --- | --- |
| [LangChain Deep Agents](https://github.com/langchain-ai/deepagents) | 将 planning、文件系统、子代理、上下文管理与 Skills 组合成可直接使用的 Harness | 接领域工具/存储后端，用固定任务集比较委派和上下文策略 |
| [Deep Agents JS](https://github.com/langchain-ai/deepagentsjs) | 同生态的 TypeScript 实现，适合 JS / TS 项目 | 比较两种实现的接口和支持范围；不把 Python 功能自动视为 JS 已支持 |
| [Microsoft Agent Framework](../agent/README.md#microsoft-agent-framework) | 复用已有框架条目；Python / .NET 的 Agent 与多 Agent 工作流 | 接企业工具、审批、遥测和持久状态；验证终止与恢复策略 |
| [OpenAI Symphony](https://github.com/openai/symphony) | 按任务分配隔离工作区、调度 Coding Agent；参考实现以 Linear 工作项驱动研发流程 | 接自己的任务源，补验证结果与人工接收流程；对照 `SPEC.md` 阅读 |
| [bytedance/deer-flow](https://github.com/bytedance/deer-flow) | 长程 Super Agent Harness，组合子代理、记忆、沙箱、Skills 与消息入口 | 研究上下文分层、任务恢复、可观测性及 Docker / K8s 部署；适合平台岗 |

Symphony 官方定位为在可信环境测试的 engineering preview，不能据此承诺生产可用。DeerFlow 的模块较多，适合在跑通轻量项目后对照架构；模型和部署接入应查当前配置文档，不将它描述为只能使用某一家模型。

## 4. 工作区、数字员工与自进化

| 项目 | 阅读重点 | 二开切入点与证据 |
| --- | --- | --- |
| [OpenBMB/PilotDeck](https://github.com/OpenBMB/PilotDeck) | 围绕工作区组织任务和多项目生产力流程 | 改一个任务工作流，验证工作区隔离、状态保留与交互可用性 |
| [OpenBMB/StaffDeck](https://github.com/OpenBMB/StaffDeck) | 企业数字员工、能力档案、流程型 Skills 和 SOP 状态管理 | 把一个明确业务 SOP 变为可审计流程，测试非法状态转移、权限与版本变更 |
| [PenguinHarness](https://github.com/Prism-Shadow/penguin-harness) | Agent 应用构建与改进闭环；阅读 Skills、状态/轨迹、记忆及模型适配 | 增加垂直 Skill 或独立 benchmark，记录初始版本、候选版本、留出集与预算 |
| [CowAgent](https://github.com/zhayujie/CowAgent) | 原 chatgpt-on-wechat；个人助理、任务规划、记忆、知识库、Skills、MCP 和多渠道接入 | 做国内 IM 业务助手，验证记忆召回、渠道权限和跨用户隔离 |
| [oh-my-openagent](https://github.com/code-yeongyu/oh-my-openagent) | 原 oh-my-opencode；Coding Agent 的多模型与任务编排扩展 | 改一个编排或工具策略，保存任务图、执行轨迹与同预算对照 |
| [aden-hive/hive](https://github.com/aden-hive/hive) | 面向生产任务的多 Agent Harness；关注图执行、状态与改进反馈 | 增加检查点或成本观测，比较恢复成功率和端到端任务完成率 |

选型前查看各仓库当前许可证。PilotDeck / StaffDeck 为 AGPL-3.0；“能看到源码”不能直接替代商用、修改后分发或服务部署的许可证判断。

PenguinHarness README 中的低成本/高准确率对比属于作者在特定数据与配置下的报告。它可以作为复现问题，不能直接转换为自己的项目成绩；面试中应区分上游报告、自己的复现和自己新增的贡献。

## 5. 垂直应用与 RAG 深水区

| 项目 | 定位与可复用能力 | 二开切入点 | 建议验证 |
| --- | --- | --- | --- |
| [HKUDS/DeepTutor](https://github.com/HKUDS/DeepTutor) | 学习工作区，结合问答、测验、研究、可视化、知识库、记忆与 Skills | 迁移到一个明确学科或领域，做领域 Skill、知识源或学习流程 | 领域问题集、引用正确性、学习任务完成率 |
| [Tencent/WeKnora](https://github.com/Tencent/WeKnora) | 文档知识平台：RAG 问答、Agent 推理、自维护 Wiki；多源接入、权限与 trace | 新增数据源 connector，或做真实场景私有知识库 | 检索召回、引用覆盖、访问权限、Wiki 更新一致性 |
| [xerrors/Yuxi（语析）](https://github.com/xerrors/Yuxi) | 可私有部署的知识 Agent，组合 RAG、知识图谱、多 Agent、MCP / Skills 与权限 | 实现领域知识问答及图谱可视化 | 检索/图谱消融、多租户隔离、答案证据链 |
| [Youtu-Agent](https://github.com/TencentCloudADP/youtu-agent) | Agent 构建、运行与评测；自动生成 Agent、经验学习及训练研究 | 接自己的 Agent 或任务集，研究工具失败与经验复用 | 接口适配、固定测试集、任务成功率、成本、失败分类 |
| [Youtu-GraphRAG](https://github.com/TencentCloudADP/youtu-graphrag) | 以 graph schema 连接建图、检索和复杂推理；仓库关联 ICLR 2026 论文 | 修改领域 schema，对照普通向量 RAG 做迁移与多跳检索实验 | 多跳证据召回、端到端正确率、索引与查询成本 |

Youtu-Agent 的现成评测代码是接入起点，自己的 Agent 仍需适配任务、工具和评分接口。上游 GAIA `72.8% pass@1` 明确对应 **text-only validation subset** 与 `DeepSeek-V3-0324` 等配置，不能表述为全量 GAIA 成绩，更不能代替二开后的实测。

DeepTutor 偏领域产品，WeKnora 偏知识平台工程，Yuxi 偏知识与多 Agent 的应用集成，Youtu-GraphRAG 偏图检索方法。按要证明的能力选择，RAG 通用资源继续在 [RAG 项目清单](../rag/projects.md)维护。

## 6. 给项目接上评测闭环

评测工具统一在 [Evaluation Harness 工具箱](../agent/evaluation-harness.md)维护，本页不再复制工具介绍。按项目选一套流程：

1. **定义任务和基线**：固定模型、数据划分、工具权限、环境和预算；保留未改动版本。
2. **跑通执行链**：用 Youtu-Agent 的评测管线、AgentCompass 或 Harbor 适配任务，保存逐步 trace、终止原因和原始结果。
3. **补过程检查**：用 DeepEval 或 agentevals 检查工具选择、参数、路径和完成条件；Promptfoo 补安全/对抗回归，Phoenix / Langfuse 帮助观测与归因。
4. **RAG 分面诊断**：分别测检索与生成。答案忠于已检索上下文，不代表必要证据已全部检索到。指标和基准入口见工具箱中的 [RAG 诊断](../agent/evaluation-harness.md#rag-diagnostics)与[标准基准](../agent/evaluation-harness.md#agent-benchmarks)。
5. **比较与回归**：记录共同任务集、失败数、重复试验、token/费用和延迟；改完后复跑同一协议，再决定是否启用改动。

框架跑通、离线指标、线上收益是不同证据。不能仅凭最终答案正确就断言轨迹高效，也不能把换模型和换 Harness 的收益混为一谈；完整方法见 [Agent 评测指南](../../docs/02-tech-stack/agent-evaluation-complete-guide.md)。

## 7. 二开怎样形成简历证据

已有通用方法见 [AI Agent 简历编写指南](../../docs/04-interview/20-resume-guide.md)。本类项目至少要说清五件事：

| 要回答的问题 | 应留存的材料 |
| --- | --- |
| 为什么改这一层 | 场景、失败样本、目标约束和替代方案 |
| 为什么选这个项目 | 候选比较、接口能力、部署成本和许可证 |
| 你具体做了什么 | 本人负责的模块、diff、设计取舍和回归测试 |
| 怎样知道改进有效 | 固定数据/模型/预算下的 baseline 与改进版结果、分母和失败分类 |
| 能迁移到哪里 | 换任务/模型/环境的结果与未验证边界 |

两种表述模板，括号内均需替换为本人证据：

> 基于 OpenHarness 为【团队任务】实现【工具权限 / Skill / 子代理隔离】；在【N 个固定任务、模型和预算】下，与未修改版本比较【完成率、耗时、成本】，结果为【实测值】；个人贡献与验证见【代码 / trace / 报告链接】。

> 基于 nanobot / WeKnora 实现【领域助手】，将【任务集】接入【评测管线】；通过轨迹分析定位【失败原因】，修改【具体模块】，在留出集上观察到【指标及统计口径】，同时记录【成本与剩余失败】。

有上游贡献时附具体 PR 链接和状态，区分本地修改、已提交、已合并。没有实测结果时写清当前进度，先补证据，不填示例数字。

## 来源与合并说明

- 飞书正文版本：`NGQzdiZMooMS22x13OAcdHX5nGb`，revision `74`；读取日期 2026-11-05。
- 嵌入表格已展开：`lTsSpF!A1:D4` 的 3 个官方项目并入第 3 节；`df3P6S!A1:C6` 的 5 组基准并入现有评测工具箱。
- 去重以 GitHub 仓库当前地址为主，结合名称/旧名判断。已有 learn-claude-code、Microsoft Agent Framework、DeepEval、Promptfoo、RAGAS、Phoenix、Langfuse 复用原条目或补充原条目；正文中的重复 Youtu-Agent 引用合并。
- 修正 Pi 的旧地址，并将原文误放在 Yuxi 后的 deepagentsjs 链接归入 Deep Agents；保留 Pi / oh-my-pi、Deep Agents Python / JS 等真实独立实现。
- 未将动态 Star 数、无法核对的活跃度排名、厂商成本比或简历占位数字写作本人实测；原文的通用指标阈值改为按任务集校准。
