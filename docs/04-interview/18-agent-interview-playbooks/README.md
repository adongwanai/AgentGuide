# AI Agent 面试备战手册合集

> 这组资料用于把 AgentGuide 的面试题库、项目讲述和技术教程串成可复习、可表达、可追问的专题手册。

---

## 📌 Section Goals（本节目标）

- 按专题补齐 AI Agent 面试中的高频深挖方向。
- 帮助候选人把项目经历讲成“问题定义 -> 技术取舍 -> 评估验证 -> 业务价值”的闭环。
- 同时服务算法岗与开发岗：既能讲论文、指标和消融，也能讲系统、工程和生产风险。
- 为简历、模拟面试、项目复盘提供可直接复习的材料入口。

---

## 💡 Core Concepts（核心概念）

这组手册不是孤立知识点，而是围绕四层能力模型组织：

| 能力层级 | 对应手册 | 面试价值 |
|:---|:---|:---|
| L1 基础认知 | Agentic RAG、Agent Memory、Skills | 讲清概念边界和技术演进 |
| L2 开发实现 | Harness、OpenClaw、Claude Code | 讲清系统架构和工程实现 |
| L3 高级优化 | Evaluation、Data Synthesis、Hermes | 讲清评估、数据和自进化闭环 |
| L4 生产工程 | STAR 项目讲述、Harness、源码分析 | 讲清稳定性、风险控制和业务价值 |

---

## 🔍 Deep Understanding（深入理解）

### 推荐复习顺序

1. **先建立项目表达框架**
   - [STAR 法则 / 面试话术备战手册](./star-storytelling-playbook.md)

2. **再补齐 Agent 核心能力**
   - [Agent Memory 备战手册](./agent-memory-playbook.md)
   - [Agent Skills / 程序性记忆备战手册](./agent-skills-procedural-memory-playbook.md)
   - [Agentic RAG / Deep Research 备战手册](./agentic-rag-deep-research-playbook.md)

3. **然后学习生产级 Agent 工程**
   - [Agent Harness 备战手册](./agent-harness-playbook.md)
   - [Claude Code 源码分析备战手册](./claude-code-source-playbook.md)
   - [OpenClaw 源码分析备战手册](./openclaw-source-playbook.md)

4. **最后补齐评估、数据和自进化闭环**
   - [Agent 评测 / Benchmark 备战手册](./agent-evaluation-playbook.md)
   - [数据合成 / Data Pipeline 备战手册](./data-synthesis-playbook.md)
   - [Hermes Agent 自进化备战手册](./hermes-self-evolution-playbook.md)

### 按岗位选择重点

| 岗位 | 必读 | 加分 |
|:---|:---|:---|
| Agent 开发工程师 | Harness、OpenClaw、Claude Code、STAR | Evaluation、Skills |
| RAG / 搜索工程师 | Agentic RAG、Evaluation、Data Synthesis | Memory、Harness |
| 大模型算法工程师 | Data Synthesis、Evaluation、Hermes、Agentic RAG | Memory、Skills |
| 全栈 Agent 工程师 | 全部阅读，重点准备 2-3 个能讲深的项目故事 | 将手册内容改写成自己的项目复盘 |

---

## 💻 Code Examples（代码示例）

本目录主要是面试复习资料，不包含独立代码工程。可运行示例请参考：

- [简历与项目讲述自检脚本](../examples/resume_storytelling_check.py)
- [示例依赖说明](../examples/requirements.txt)

运行方式：

```bash
python docs/04-interview/examples/resume_storytelling_check.py --mode resume
python docs/04-interview/examples/resume_storytelling_check.py --mode story
```

---

## 🎯 Interview Questions（面试中如何考）

使用这组手册时，建议每个专题都准备三类回答：

1. **30 秒概念版**：一句话讲清这个技术解决什么问题。
2. **2 分钟项目版**：结合自己的项目讲背景、取舍、指标、复盘。
3. **15 分钟深挖版**：展开架构图、数据流、失败案例、替代方案和生产风险。

通用追问：

- 你为什么选择这个方案，而不是更简单的 baseline？
- 你的指标怎么定义，是否能复现？
- 失败样本有哪些，下一轮会怎么改？
- 如果从 Demo 走向生产，还缺哪些工程能力？
- 这项能力对算法岗和开发岗分别怎么体现？

---

## 📚 Extended Reading（扩展阅读）

- [2026 AI 前沿面试专题](../23-frontier-interview-guides/) - 自进化 Agent、Agentic RL、AI Infra、Coding Agent、世界模型与多模态生成
- [完整面试与求职导航](../README.md) - 按岗位、时间和能力层级选择复习内容
- [AI Agent 简历编写指南](../20-resume-guide.md)
- [AI Agent 项目讲述技巧](../21-storytelling.md)
- [Agent 系统题](../03-agent-questions.md)
- [算法岗专项题](../05-algorithm-specialized.md)
- [开发岗专项题](../06-development-specialized.md)
- [大厂真实面经](../12-company-interview-cases.md)
- [Agent Evaluation Harness 完全指南](../../02-tech-stack/26-agent-evaluation-harness-guide.md)
- [Agent Memory 完整指南](../../02-tech-stack/15-agent-memory.md)
