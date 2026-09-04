---
type: 教程
status: 已发布
level: 高阶
topic:
  - Coding Agent
  - 安全
  - Agent
---

#  OpenClaw彻底带火了沙箱，桌面Agent落地必看

> 阿东最近做桌面 Agent 项目时，几个信号让我重新审视了架构设计
---

## 写在前面

2026 年 1 月，有个叫 OpenClaw 的项目两周刷了 18 万 GitHub Star。作者是奥地利开发者 Peter Steinberger，用语音跟本地 AI 聊天，一个月提交了 6600 次代码。

同一个月，月之暗面发布了 Kimi K2.5，支持同时跑 100 个子 Agent。

Manus、E2B、LangChain、阿里 OpenSandbox...一整波沙箱技术在 2025 年集体爆发。

这些事情看似不相关，但指向同一个问题：**当 AI 有权限访问你的系统时，你怎么保证它不会搞砸一切？**

这篇文章想做三件事：
1. 为什么 2025-2026 是"沙箱元年"
2. 业界有哪些沙箱方案，各自的坑在哪里
3. 如果你也要做本地 Agent，该怎么选型

踩坑经验为主，技术细节为辅。如果我有说错的地方，欢迎指正。

---

## OpenClaw：本地 AI Agent 的范式转移

先说说最火的这个。

![image.png](https://raw.githubusercontent.com/adongwanai/Awesome-Awesome-LLMs/main/20260213000153416.png)

### 两周 18 万星

Peter 的 OpenClaw 2026 年 1 月发布，核心就三个特点：

- 完全本地运行，数据不上传
- 用语音跟 AI 聊天，AI 控制你的电脑
- 1 个月 6600 次代码提交（Peter 自己的开发数据）

他接受采访的时候说了一句话，我记了很久：

> "在云端，AI 能做一些事。但在你的电脑上，AI 能做任何事。"

这不是技术问题，这是权力问题。

### 云端 vs 本地

我列个对比你就明白了：

**云端 Agent 的局限：**
- 不能读取你本地文件（除非你先上传）
- 不能控制你电脑上的东西（特斯拉、台灯、床的温度）
- 你的对话全部存在别人服务器上
- 被云端的 API 限制死了

**本地 Agent 的解放：**
- 直接读取硬盘里的音频文件
- 控制特斯拉、台灯、智能家居
- 回复邮件、整理文件
- 所有数据就是你硬盘上的一堆 Markdown 文件

Peter 说得很直接："如果最终用户有访问权限，我就可以访问这些数据。"

在数据就是权力的时代，**本地优先等于数字主权**。

### 但这里有个问题

OpenClaw 给我们的启示不是"本地真好"，而是**本地 AI Agent 需要沙箱**。

区别在于：
- 云端沙箱解决的是"不要搞崩别人的服务器"
- 本地沙箱解决的是"不要搞崩我自己的电脑"

当你给 Agent 完整的系统访问权限，一个失控的 Agent 足以让你的开发环境一夜回到解放前。**沙箱成了本地 AI Agent 的安全基础设施**。

---

## 为什么 2025-2026 是"沙箱元年"

### 1. Manus 的沙箱机制重回社区视野
![image.png](https://raw.githubusercontent.com/adongwanai/Awesome-Awesome-LLMs/main/20260213000543532.png)


[Manus](https://manus.im/) 做了个完全隔离的沙箱，基于 VM 级隔离，里面是完整的 Ubuntu 文件系统。

核心特点三个：
- 每个任务会话完全独立（Zero Trust）
- 持久化存储，24/7 执行
- 网络和文件系统隔离，防止逃逸

说实话，这套方案在安全性和成本之间做了取舍。如果你的 Agent 需要执行不可信代码，VM 级隔离是绕不过去的坎。

### 2. 阿里开源 OpenSandbox

2025 年阿里开源了 [OpenSandbox](https://github.com/alibaba/OpenSandbox)，定位是"面向 AI 应用场景的通用沙箱平台"。
![image.png](https://raw.githubusercontent.com/adongwanai/Awesome-Awesome-LLMs/main/20260213000619537.png)

四个特点值得关注：
- 统一协议，标准化沙箱接口
- 多语言 SDK，降低集成门槛
- Docker + K8s 支持
- 文档有中文，Issue 回复快

国内团队选型时，OpenSandbox 的本土支持是个加分项。

### 3. LangChain 发布 Deep Agents 沙箱集成

2025 年 11 月，[LangChain 官方博客](https://www.blog.langchain.com/execute-code-with-sandboxes-for-deepagents/)宣布 Deep Agents 全面支持沙箱执行：
![image.png](https://raw.githubusercontent.com/adongwanai/Awesome-Awesome-LLMs/main/20260213000646644.png)


> "Sandboxes create isolation between the agent's execution environment and your host system"

关键特性：
- 远程执行代码和命令
- 创建文件、安装依赖
- 不触碰本地机器

LangChain 这波算是给了大家一个标准答案：Agent 框架应该怎么接沙箱。

### 4. 桌面 Agent 爆发

2025 年扎堆发布了一堆桌面 Agent：
- Anthropic Cowork：面向普通用户的桌面自动化
- MiniMax、阶跃星辰、Qoder：国内厂商纷纷入局
- 阶跃 AI 桌面伙伴：2025 年 9 月发布，比 Cowork 更早

**核心问题来了**：当 AI 有权限删除文件、修改系统时，**安全成了第一优先级**。一个失控的 Agent 足以让开发环境一夜回到解放前。

---

## Kimi K2.5：100 个 Agent 怎么隔离

[月之暗面](https://www.moonshot.cn/) 在 2026 年 1 月发布的 Kimi K2.5 引入了多 Agent 集群能力。
![image.png](https://raw.githubusercontent.com/adongwanai/Awesome-Awesome-LLMs/main/20260213000847834.png)

### 核心特性

1. 动态生成子 Agent，最多同时跑 100 个
2. 串行 + 并行执行，灵活的任务编排
3. 实时生成，不是预设 Agent，而是根据任务动态创建
4. 自动分解问题，将复杂任务拆解并分配

### 沙箱机制（推测）

官方文档还没完全公开技术细节，但从架构推测：

- 任务级隔离：每个子 Agent 在独立环境中执行
- 资源限制：防止单个 Agent 消耗过多资源
- 状态管理：多 Agent 间的状态同步与通信

100 个 Agent 同时跑，隔离和资源调度是核心挑战。

---

## 业界沙箱方案对比

这部分是我踩坑最多的地方，给大家一个参考框架。

### 1. E2B / (Manus 的底层)
**GitHub**: https://github.com/e2b-dev/E2B
- **定位**: 基于 Firecracker 的 microVM 平台
- **优点**: 冷启动快，近 VM 级隔离强
- **缺点**: 管理微VM 的复杂度和成本高于纯容器
- **适用**: 生产环境想兼顾隔离和启动延迟
![image.png](https://raw.githubusercontent.com/adongwanai/Awesome-Awesome-LLMs/main/20260213000943969.png)

**我的体验**：E2B 的冷启动确实做得好，但微VM 的管理成本不容忽视。如果你的 QPS 不高，单实例成本会比容器方案高 2-3 倍。

---

### 2. Anthropic — sandbox-runtime

- **GitHub**: [https://github.com/anthropics/anthropic-quickstarts/tree/main/sandbox-runtime](https://github.com/anthropic-experimental/sandbox-runtime)
- **定位**: OS 原生隔离（Linux 用 bubblewrap，macOS 用 sandbox-exec）
- **优点**: 本地运行延迟极低、开源、易集成
- **缺点**: 隔离强度不如 microVM，默认配置网络权限太宽松
- **适用**: 本地开发、低延迟场景
![image.png](https://raw.githubusercontent.com/adongwanai/Awesome-Awesome-LLMs/main/20260213001301876.png)

**我的体验**：本地开发体验一流，但生产环境必须自己加 egress 规则。

---

### 3. LangChain — Sandboxes for Deep Agents

- **官方文档**: https://docs.langchain.com/oss/python/deepagents/sandboxes
- **定位**: 为 Deep Agents 提供远程沙箱执行环境
- **特点**: 支持多种沙箱后端（Docker、E2B、自定义）
- **适用**: 需要灵活切换沙箱后端的场景
![image.png](https://raw.githubusercontent.com/adongwanai/Awesome-Awesome-LLMs/main/20260213001353375.png)

```python
from langchain_deepagents import SandboxedAgent

agent = SandboxedAgent(
    sandbox_type="e2b",
    setup_script="pip install numpy pandas"
)

result = agent.run("分析这个 CSV 文件")
```

---

### 4. Alibaba — OpenSandbox

- **GitHub**: https://github.com/alibaba/OpenSandbox
- **定位**: 通用 AI 沙箱平台，专注 LLM 和 AI Agent 安全执行
- **特点**: 多语言 SDK、Docker 和 K8s 支持、统一协议
- **适用**: 企业内部部署、生产环境沙箱调度、多租户隔离

---

### 5. Kubernetes SIG — Agent Sandbox
![image.png](https://raw.githubusercontent.com/adongwanai/Awesome-Awesome-LLMs/main/20260213001447081.png)

- **GitHub**: https://github.com/kubernetes-sigs/agent-sandbox
- **定位**: K8s 原生沙箱管理方案
- **优点**: 支持 Warm Pool 预热、gVisor/Kata 多种隔离后端
- **缺点**: 运维与资源成本高
- **适用**: 大规模、多租户企业场景

**我的体验**：如果你的团队已经有成熟的 K8s 运维能力，这个方案是最稳的。如果没有，运维成本会让你怀疑人生。

---

### 6. 字节 AIO Sandbox / All-in-One

- **GitHub**: https://github.com/agent-infra/sandbox
- **定位**: 集成浏览器、VSCode Server、VNC、终端的一体化沙箱
- **优点**: 快速上手、演示/PoC 与开发体验完整
- **缺点**: 隔离仅容器级，生产场景需再套 VM
- **适用**: 演示、本地开发

![](https://github.com/agent-infra/sandbox/raw/main/website/docs/public/images/aio-index.png)
**我的体验**：做 Demo 神器，但别直接上生产。容器隔离对生产环境来说不够硬。

---

### 7. Bolt / StackBlitz（WebContainer）

- **GitHub**: https://github.com/stackblitz/webcontainer-core
- **定位**: 浏览器内运行，Node.js 开发环境
- **优点**: 几乎零延迟、免运维、交互式开发体验出色
- **缺点**: 受限于浏览器 runtime
- **适用**: 教育、交互式 demo

---

### 8. SmolAgents（Hugging Face）
**GitHub**: https://github.com/huggingface/smolagents
- **定位**: 轻量化 agents 库，包含多种代码执行器
- **执行器**: LocalPythonExecutor、BlaxelExecutor、E2BExecutor、DockerExecutor、ModalExecutor、WasmExecutor
- **适用**: 研究、原型与受控内部使用

---

## 选型决策树

根据我的实战经验，给个决策框架：

![image.png](https://raw.githubusercontent.com/adongwanai/Awesome-Awesome-LLMs/main/20260213000004728.png)

```
你的场景是什么？

├─ 本地开发 / 低延迟优先
│  └─ Anthropic sandbox-runtime / WebContainers
│     （延迟 <100ms，隔离强度中等）
│
├─ 中小规模生产环境（QPS < 100）
│  ├─ 需要强隔离 → E2B / Docker + 网络隔离
│  └─ 快速迭代 → LangChain Sandboxes + OpenSandbox
│
└─ 大规模 / 多租户 / 企业级（QPS > 100）
   └─ Kubernetes Agent Sandbox + Kata / gVisor
```

---

## 几个核心结论

### 1. 沙箱不是"可选"

2026 年前，沙箱是锦上添花。2025 年后，沙箱是必须的基础设施。

Agent 从 demo 走向生产的过程中，安全逃逸事件会越来越多。没有沙箱的 Agent，就是裸奔的定时炸弹。

### 2. 隔离等级要根据场景选择

不是所有场景都需要 microVM：

- 本地开发：bubblewrap/sandbox-exec 够用，延迟最低
- 内部工具：Docker + 网络隔离，成本可控
- 对外服务：microVM（E2B/Kata/Firecracker），硬隔离必须的

### 3. 统一协议是趋势

OpenSandbox、LangChain 都在推动标准化接口。这意味着：

- 换沙箱后端不需要重写 Agent 代码
- 社区会有更多工具和最佳实践
- 但还没到"完全兼容"的程度，迁移成本依然存在

### 4. 多 Agent 需要更强隔离

当 100 个 Agent 同时运行时：

- 资源调度成为核心问题（谁获得 CPU/内存？）
- 状态同步不能变成瓶颈（Agent 间如何通信？）
- 故障隔离必须做好（一个 Agent 挂了不能影响其他）

---

## 我的实践建议

### 从小开始，逐步迭代

别一开始就上 K8s + Kata + 分布式调度。我的建议：

1. **PoC 阶段**：用 Anthropic sandbox-runtime 或 Docker，验证核心功能
2. **小规模测试**：接入 E2B 或 OpenSandbox，跑真实流量
3. **生产环境**：根据 QPS 和隔离要求，选方案 A/B/C

### 别忘了成本控制

沙箱是资源大户：

- **E2B**：$0.10/小时（微VM）
- **OpenSandbox**：自托管，成本取决于 K8s 集群规模
- **Docker**：几乎免费，但隔离弱

如果你的 Agent QPS 是 100，用 E2B 每月成本约 $7,200。用 Docker 可能只要 $500。

### 监控和审计必须跟上

沙箱不是万能的。你需要：

- **Egress 控制**：Agent 能访问哪些网络？
- **命令审计**：Agent 执行了哪些命令？
- **资源限制**：CPU/内存/磁盘上限是多少？

没有这些，沙箱只是推迟了风险，而不是消除了风险。

---

## 参考资源

- [Manus Sandbox 详解](https://manus.im/blog/manus-sandbox)
- [LangChain Sandboxes 官方文档](https://docs.langchain.com/oss/python/deepagents/sandboxes)
- [OpenSandbox GitHub](https://github.com/alibaba/OpenSandbox)
- [E2B 官网](https://e2b.dev/)
- [Kubernetes Agent Sandbox](https://github.com/kubernetes-sigs/agent-sandbox)

---

## 写在最后

沙箱技术演进很快，今天的最优解可能明天就过时。但核心原则不变——**根据场景选择合适的隔离等级，而不是盲目追求最强隔离**。

如果你在选型时有具体问题，欢迎交流。
