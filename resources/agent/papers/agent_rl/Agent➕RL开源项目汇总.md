---
type: 论文清单
status: 已发布
level: 进阶
topic:
  - Agent
  - 科研
  - 项目实战
---

# Agent RL 开源项目汇总

> 内容核对时间：2026-11。框架类项目迭代很快，功能描述以各仓库当前 README 为准。算法原理和工程要点见 [Agent 强化学习教程](../../../../docs/02-tech-stack/21-agent-reinforcement-learning.md)。

本页按用途分为五类：学习与复现、训练框架、Agent 训练接入层、代表性研究工作、环境与基准。最后单列了常被误归为“Agent RL”的相关工作。

## 1. 学习与复现

### agentic-rl-lab（Datawhale）

- **仓库**：[datawhalechina/agentic-rl-lab](https://github.com/datawhalechina/agentic-rl-lab) ｜ [在线阅读](https://datawhalechina.github.io/agentic-rl-lab/)
- **内容**：16 个实验，覆盖 loss 函数、GRPO、Vision GRPO、DAPO、GSPO、OPD / OPSD、Search-R1、ReTool、ALFWorld、TEMPO、AgentOPSD、Spec-o3、Tournament-GRPO 等。每个实验包含算法讲解、运行说明、完整代码、SwanLab 训练记录和实际花费。
- **特点**：训练与采样通过 PyTRIO 训练 API 远程执行，本地不需要 GPU；默认基座 Qwen3.5-4B + LoRA。实验规模小，适合理解机制和做对照实验，结果不作 benchmark 结论。
- **建议用法**：按 [教程中的学习顺序](../../../../docs/02-tech-stack/21-agent-reinforcement-learning.md#动手实验agentic-rl-lab) 从 GRPO 读到多轮工具 Agent，再选一个实验只改一个变量做对照。

### Tinker Cookbook（Thinking Machines）

- **仓库**：[thinking-machines-lab/tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) ｜ [Tinker 文档](https://tinker-docs.thinkingmachines.ai/)
- **内容**：基于 Tinker 训练 API 的 SFT、RL、蒸馏示例。和 agentic-rl-lab 的思路相同：本地写训练循环，远端执行 forward/backward、optimizer step 和采样。Tinker 只支持 LoRA 微调。

## 2. 训练框架

| 项目 | 维护方 | 定位 |
|:---|:---|:---|
| [verl](https://github.com/volcengine/verl) | 字节跳动 | 使用最广的开源 LLM RL 框架之一，FSDP / Megatron 训练后端，vLLM / SGLang 推理，支持多轮工具 rollout |
| [slime](https://github.com/THUDM/slime) | 清华 THUDM / 智谱 | Megatron + SGLang，面向大规模 RL 扩展，自定义 rollout 接口便于接 Agent 环境 |
| [AReaL](https://github.com/inclusionAI/AReaL) | 蚂蚁 inclusionAI | 以全异步 RL 为核心设计，生成与训练解耦以提高吞吐 |
| [ROLL](https://github.com/alibaba/ROLL) | 阿里巴巴 | 大规模 RL 训练框架，提供 Agentic 多轮交互 pipeline |
| [SkyRL](https://github.com/NovaSky-AI/SkyRL) | UC Berkeley NovaSky | 面向长程多轮 Agent，含环境库 SkyRL-Gym；相关论文 [SkyRL-Agent](https://arxiv.org/abs/2511.16108) |
| [NeMo RL](https://github.com/NVIDIA-NeMo/RL) | NVIDIA | NeMo 生态中的后训练与 RL 库，提供环境接口 |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | 社区 | Ray + vLLM + DeepSpeed，RLHF / RLVR 的常用实现 |
| [TRL](https://github.com/huggingface/trl) | Hugging Face | 与 Transformers 生态集成，`GRPOTrainer` 等，适合单机实验入门 |

选型思路：只验证算法用训练 API 或 TRL；单节点单轮 RLVR 用 TRL / OpenRLHF；多机、长程多轮 Agent、需要控制吞吐时再看 verl、slime、AReaL、ROLL、SkyRL。

## 3. Agent 训练接入层

这类项目不一定自带完整训练后端，作用是把已有的 Agent 代码、工具或环境接到 RL 训练上。

| 项目 | 说明 |
|:---|:---|
| [rLLM](https://github.com/rllm-org/rllm) | Agentica 团队的 Agent RL 框架，把 Agent 与环境抽象后接入训练后端 |
| [Agent Lightning](https://github.com/microsoft/agent-lightning) | 微软，Agent 执行与训练解耦，可接入用不同框架写的已有 Agent |
| [verl-tool](https://github.com/TIGER-AI-Lab/verl-tool) | 基于 verl 的工具调用 Agent RL 扩展，论文 [VerlTool](https://arxiv.org/abs/2509.01055) |
| [Agent-R1](https://github.com/0russwest0/Agent-R1) | 端到端训练工具调用 Agent 的模块化框架，论文 [Agent-R1](https://arxiv.org/abs/2511.14460) |
| [RAGEN](https://github.com/RAGEN-AI/RAGEN) | 多轮 Agent RL 系统，提出 StarPO 轨迹级优化，并分析了训练中的 “Echo Trap” 不稳定现象；论文 [RAGEN](https://arxiv.org/abs/2504.20073) |
| [ART](https://github.com/OpenPipe/ART) | OpenPipe 的 Agent Reinforcement Trainer，用 GRPO 训练多步 Agent |
| [verifiers](https://github.com/PrimeIntellect-ai/verifiers) | 用于定义多轮环境与奖励的库，可对接多种训练后端 |

## 4. 代表性研究工作

### 4.1 搜索与深度研究

| 工作 | 论文 | 代码 | 要点 |
|:---|:---|:---|:---|
| Search-R1 | [2503.09516](https://arxiv.org/abs/2503.09516) | [PeterGriffinJin/Search-R1](https://github.com/PeterGriffinJin/Search-R1) | 推理过程中自主生成搜索查询；检索 token 掩码；只用结果奖励。agentic-rl-lab 有复现 |
| ReSearch | [2503.19470](https://arxiv.org/abs/2503.19470) | [Agent-RL/ReCall](https://github.com/Agent-RL/ReCall) | 把搜索操作作为推理链的一部分，不使用推理步骤的监督数据；后续扩展为通用工具调用的 ReCall |
| R1-Searcher | [2503.05592](https://arxiv.org/abs/2503.05592) | [RUCAIBox/R1-Searcher](https://github.com/RUCAIBox/R1-Searcher) | 两阶段基于结果的 RL，先学会调用检索，再学会用好检索结果 |
| ZeroSearch | [2505.04588](https://arxiv.org/abs/2505.04588) | [Alibaba-NLP/ZeroSearch](https://github.com/Alibaba-NLP/ZeroSearch) | 用 LLM 模拟搜索引擎作为训练环境，按课程逐步降低检索结果质量，避免真实搜索 API 的成本 |
| StepSearch | [2505.15107](https://arxiv.org/abs/2505.15107) | [Zillwang/StepSearch](https://github.com/Zillwang/StepSearch) | 逐步 PPO，引入基于信息增益的步级搜索奖励 |
| DeepResearcher | [2504.03160](https://arxiv.org/abs/2504.03160) | [GAIR-NLP/DeepResearcher](https://github.com/GAIR-NLP/DeepResearcher) | 在真实网页环境中端到端 RL 训练深度研究 Agent |

### 4.2 工具调用与代码执行

| 工作 | 论文 | 代码 | 要点 |
|:---|:---|:---|:---|
| ReTool | [2504.11536](https://arxiv.org/abs/2504.11536) | [ReTool-RL/ReTool](https://github.com/ReTool-RL/ReTool) | 长推理中交错执行代码，用 RL 学习何时、如何调用代码解释器。agentic-rl-lab 有复现 |
| Multi-Turn-RL-Agent | [2505.11821](https://arxiv.org/abs/2505.11821) | [SiliangZeng/Multi-Turn-RL-Agent](https://github.com/SiliangZeng/Multi-Turn-RL-Agent) | 多轮工具使用中的 turn 级 advantage 估计 |

### 4.3 信用分配与蒸馏

| 工作 | 来源 | 要点 |
|:---|:---|:---|
| AgentOPSD | [2608.05987](https://arxiv.org/abs/2608.05987) | 同一模型带技能说明复评每个 turn，得到 turn 级信用，在轨迹 advantage 上做有界调整 |
| TEMPO | [Dots 团队博客](https://studio.dots.ai/dots/tempo-blog.html) | macro-step 优化 + 生成式 critic，面向超长程交互；目前无论文 |
| OPSD | [2601.18734](https://arxiv.org/abs/2601.18734) | 冻结的初始模型在看到参考解后作为教师，对学生自采样轨迹逐 token 蒸馏 |
| Tournament-GRPO | [2605.26958](https://arxiv.org/abs/2605.26958) | 开放式长文本生成中，用组内两两比较的淘汰赛胜场作为奖励 |

以上四项在 agentic-rl-lab 中均有复现。

## 5. 环境与基准

| 项目 | 说明 |
|:---|:---|
| [ALFWorld](https://github.com/alfworld/alfworld) | 文本化的家务任务环境，六类任务，常用于多轮 Agent RL 实验 |
| [WebShop](https://github.com/princeton-nlp/WebShop) | 模拟电商网站环境，[论文](https://arxiv.org/abs/2207.01206) 中用模仿学习 + RL 训练网页交互 Agent |
| [AgentBench](https://github.com/THUDM/AgentBench) | 多环境 Agent 评测基准（操作系统、数据库、网页等），用于评测，本身不提供 RL 训练 |

## 6. 常被混入、但不属于 Agent RL 的工作

这几项经常出现在 Agent RL 资料里，原理上需要区分清楚：

- **ReAct**（[论文](https://arxiv.org/abs/2210.03629) ｜ [代码](https://github.com/ysymyth/ReAct)）：推理与行动交替的提示范式，论文用 few-shot 提示和少量微调，没有使用 RL。它是多数 Agent RL 工作的轨迹格式来源。
- **Toolformer**（[论文](https://arxiv.org/abs/2302.04761)）：用自监督方式筛选“插入 API 调用后能降低后续 token 损失”的样本再微调，没有使用 RL。
- **RL4LMs**（[论文](https://arxiv.org/abs/2210.01241) ｜ [代码](https://github.com/allenai/RL4LMs)）：面向文本生成任务的 RL 库与基准（含 PPO 和 NLPO），出现在 Agent RL 之前，不涉及多轮工具交互，可作为早期背景阅读。

---

**返回**：[Agent RL Papers](./README.md) ｜ [Agent 资源总览](../../README.md)
