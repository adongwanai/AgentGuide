---
type: 路线图
status: 已发布
level: 通用
topic:
  - 面试求职
  - 模型训练
  - 科研
---

# 前沿算法岗位完整学习指南

> 基于北上杭深真实算法岗位的技术职责、硬要求和加分项生成。生成时间：2026-07-15T16:13:26+08:00。

## 使用方法

这份指南把岗位中的技术职责、硬要求和加分能力归并到阶段能力模块中。模块用于明确学习边界，实践任务、交付物、验收标准和技术索引按阶段统一组织。

建议维护一个贯穿全程的实验仓库，统一保存环境、数据、训练、评测、部署、失败案例和技术决策。不要把每个阶段做成互不关联的玩具项目。

### 完整性口径

- 6635 条技术要求均映射到一个阶段能力模块；主文档不再逐句重复。
- 357 项算法主关键词与补充技术词共同进入各阶段技术索引；每个词放在最适合学习的主阶段，避免重复堆砌。
- 基础技术词表共 535 项，并从技术要求中补充缩写、框架和工具名。
- 职责动作由能力闭环、实践任务和验收标准承接；具体技术对象由模块内容和技术索引承接。
- 教育、院校、毕业批次和数字年限已在上游筛选文档中删除，本指南不重新引入。

## 能力全景

| 阶段 | 能力域 | 建议节奏 | 能力模块 | 技术关键词 | 核心产出 |
|---:|---|---:|---:|---:|---|
| 0 | 工程环境、编程与实验规范 | 2-4 周 | 6 | 62 | 训练模板仓库、故障排查手册 |
| 1 | 数学、统计、机器学习与优化基础 | 4-6 周 | 6 | 86 | 数学推导笔记、传统 ML 基线仓库 |
| 2 | 深度学习、Transformer 与生成建模 | 5-8 周 | 6 | 32 | Transformer 实现、生成模型实验 |
| 3 | 数据工程、语料治理与合成数据 | 4-7 周 | 7 | 73 | 版本化数据集、数据质量报告 |
| 4 | 预训练、模型架构、MoE 与长上下文 | 6-10 周 | 7 | 102 | 预训练配方、Scaling 实验 |
| 5 | 后训练、PEFT、强化学习与模型对齐 | 7-12 周 | 7 | 137 | 后训练流水线、偏好与奖励数据 |
| 6 | RAG、Agent、工具调用、记忆与长程规划 | 6-10 周 | 7 | 95 | RAG 系统、长程 Agent |
| 7 | 多模态、视觉语言、语音、视频与具身模型 | 7-12 周 | 7 | 134 | 多模态微调项目、生成模型项目 |
| 8 | 评测、Benchmark、实验工程与因果分析 | 4-7 周并持续进行 | 7 | 50 | 评测框架、Benchmark 数据卡 |
| 9 | AI 安全、红队、鲁棒性、事实性与隐私 | 4-8 周并持续进行 | 7 | 29 | 威胁模型、红队与安全评测集 |
| 10 | 分布式训练、训练平台与 AI Infra | 5-9 周 | 7 | 58 | 多卡训练方案、Profiling 报告 |
| 11 | 推理、Serving、压缩与性能优化 | 5-8 周 | 7 | 55 | 推理服务、性能基准 |
| 12 | 垂直算法分支与业务建模 | 任选 1-2 个方向，各 6-10 周 | 8 | 75 | 领域端到端项目、业务指标树 |
| 13 | 论文复现、研究方法、产品落地与作品集 | 贯穿全程，集中整理 4-6 周 | 7 | 207 | 论文复现仓库、开源贡献 |

## 推荐推进方式

1. 阶段 0-2 是共同基础，应先完成可复现训练模板、数学基线和 Transformer/生成模型实验。
2. 阶段 3-5 打通数据、预训练和后训练，是基础模型算法岗的核心主线。
3. 阶段 6-9 按 Agent、多模态、评测和安全逐步扩展，所有方向都必须建立可重复评测。
4. 阶段 10-11 负责把算法变成可规模化训练和稳定服务的系统。
5. 阶段 12 选择一到两个垂直方向形成深度；阶段 13 从第一天开始持续积累证据。

不要用课程数量衡量进度。每个阶段必须留下代码、数据说明、实验结果、失败分析和验收记录。

## 阶段 0：工程环境、编程与实验规范

> 建议节奏：2-4 周。能力模块：6 个。

### 学习目标

建立可复现、可调试、可扩展的算法研发底座，能独立完成数据处理、训练、评测和服务接口。

### 必学内容

> 能力闭环：原理研究与复现、数据构建与治理、算法建模与实现、训练调优与稳定性、评测、消融与误差分析、工程系统与工具链、部署、性能与运维、场景适配与产品闭环、安全、鲁棒与合规、原理理解与实际应用。

- **模块 0.1**：Python、C/C++、SQL、Shell；补充 Go/Java/Rust 在服务和高性能模块中的使用边界。
- **模块 0.2**：Linux、Git、Docker、Conda/uv、CUDA 环境、依赖锁定、配置管理和随机种子。
- **模块 0.3**：NumPy、Pandas、SciPy、scikit-learn；PyTorch Dataset、DataLoader、autograd 与自定义算子。
- **模块 0.4**：日志、断点续训、指标记录、实验追踪、数据校验、单元测试、回归测试和故障复现。
- **模块 0.5**：数据结构、复杂度、并发、网络、数据库、RPC/REST、缓存、消息队列和微服务基础。
- **模块 0.6**：Profiling、数值异常、梯度异常、OOM、I/O 瓶颈和线上问题定位。

### 实践任务

1. 实现一个可配置的 PyTorch 训练模板，支持 AMP、断点恢复、独立评测和实验追踪。
2. 为数据为空、标签越界、NaN、梯度爆炸、OOM 和训练中断分别编写复现与修复用例。
3. 把模型封装为 REST/RPC 服务，加入超时、重试、幂等、批处理、监控和压测。

### 可交付成果

- 训练模板仓库
- 故障排查手册
- 接口与压测报告

### 验收标准

- 换机器后可按文档复现实验，关键依赖和随机性来源可追踪。
- 能解释训练慢、显存高、结果漂移和服务不稳定的具体原因。
- 代码具备测试、日志、配置、版本和最小可观测性。

### 技术关键词索引

- `AI-Native`、`c++`、`cuda`、`docker`、`FastAPI`、`FForking`、`Full-stack`、`git`
- `GitHub`、`golang`、`go语言`、`GtHub`、`huggingface`、`ICPC`、`java`、`jax`
- `JS`、`k8s`、`keras`、`kubernetes`、`linux`、`MATLAB`、`mindspore`、`MMDetection`
- `ModelScope`、`MySQL`、`numpy`、`OD`、`OOP`、`OpenCode`、`opencv`、`PaddleDetection`
- `paddlepaddle`、`pandas`、`PostgreSQL`、`PR`、`PyQt`、`python`、`pytorch`、`QNN`
- `repo-level`、`RT-DETR`、`rust`、`scala`、`scikit-learn`、`scipy`、`shell`、`SIMULINK`
- `sklearn`、`SPSS`、`sql`、`SWE-bench`、`TCP`、`tensorflow`、`transformers`、`triton`
- `XZ0000580`、`并发编程`、`数据库`、`网络编程`、`计算机基础`、`软件工程`

## 阶段 1：数学、统计、机器学习与优化基础

> 建议节奏：4-6 周。能力模块：6 个。

### 学习目标

能从目标函数、数据分布和统计假设解释算法，而不是只会调用框架。

### 必学内容

> 能力闭环：原理研究与复现、数据构建与治理、算法建模与实现、训练调优与稳定性、评测、消融与误差分析、工程系统与工具链、部署、性能与运维、场景适配与产品闭环、安全、鲁棒与合规、原理理解与实际应用。

- **模块 1.1**：线性代数：向量空间、矩阵分解、特征值、SVD、低秩近似和张量运算。
- **模块 1.2**：概率统计：条件概率、贝叶斯、常见分布、最大似然、假设检验、置信区间和校准。
- **模块 1.3**：优化：梯度下降、动量、AdamW、约束优化、凸优化、拉格朗日、数值稳定性。
- **模块 1.4**：机器学习：线性/逻辑回归、树模型、Boosting、聚类、降维、异常检测和特征工程。
- **模块 1.5**：泛化：偏差方差、正则化、交叉验证、数据泄漏、分布偏移和不确定性。
- **模块 1.6**：实验统计：效应量、显著性、功效、方差缩减、因果推断和 uplift 建模基础。

### 实践任务

1. 不用深度学习框架实现线性模型、MLP 和反向传播，并进行梯度检查。
2. 完成树模型与神经网络的同数据对比，分析数据规模、特征和误差类型。
3. 设计一组有置信区间的对照实验，说明结论何时成立、何时不能外推。

### 可交付成果

- 数学推导笔记
- 传统 ML 基线仓库
- 统计实验报告

### 验收标准

- 能推导常用损失和优化更新，并解释稳定性与收敛问题。
- 能识别泄漏、混杂、过拟合和指标误导。
- 能为新问题建立合理基线，而不是直接上大模型。

### 技术关键词索引

- `AI+`、`ARM`、`BLOOM`、`C#`、`CANN`、`CCPC`、`co-design`、`DALI`
- `DeepSpeedd`、`Diffusion-based`、`DLRover`、`DNN`、`DQN`、`DSA`、`end-to-end`、`ESPnet`
- `fasterTransformer`、`GBDT`、`GNN`、`GO`、`GPGPU`、`hands-on`、`HMM`、`IEG`
- `Inference`、`IP`、`LIO-SAM`、`LLaMA`、`LLamaFactory`、`LM`、`Long-CoT`、`LP`
- `LSTM`、`MACE`、`MapTR`、`MASt3R`、`MCTS`、`MDP`、`MILP`、`MINLP`
- `MIP`、`MLIR`、`MLM`、`mniGuard`、`MXNet`、`NN`、`NPU`、`NVIDIA`
- `Objective-C`、`PhD`、`PHP`、`Python+Pytorch`、`QWen`、`RDMA`、`RESTful`、`Self-Evolving`
- `SFTRLHF`、`Spec-Decoding`、`SwiftUI`、`SysML`、`TCN`、`TensorBoard`、`TF`、`TorchScript`
- `TRT-LLM`、`VAD`、`VGGT`、`VideoMAE`、`ViT`、`WeNet`、`WFST`、`x86`
- `XLA`、`凸优化`、`因果推断`、`数值计算`、`数学模型`、`数据结构`、`数理统计`、`最优化`
- `机器学习`、`概率统计`、`概率论`、`特征工程`、`特征提取`、`线性代数`

## 阶段 2：深度学习、Transformer 与生成建模

> 建议节奏：5-8 周。能力模块：6 个。

### 学习目标

掌握现代基础模型共同的表示学习、序列建模和生成建模原理。

### 必学内容

> 能力闭环：原理研究与复现、算法建模与实现、训练调优与稳定性、评测、消融与误差分析、工程系统与工具链、部署、性能与运维、场景适配与产品闭环、原理理解与实际应用。

- **模块 2.1**：MLP、CNN、RNN/LSTM、残差、归一化、初始化、激活函数、正则化和损失设计。
- **模块 2.2**：Attention、Transformer、位置编码、RoPE、Encoder/Decoder、KV 表示和自回归建模。
- **模块 2.3**：BERT/GPT/ViT、对比学习、自监督学习、掩码建模和表征学习。
- **模块 2.4**：VAE/VQ-VAE、GAN、Diffusion、Flow Matching、能量模型和自回归生成。
- **模块 2.5**：优化器、学习率、warmup、梯度裁剪、混合精度、稳定性和消融实验。
- **模块 2.6**：参数量、计算量、显存、吞吐、上下文长度和数据规模之间的基本关系。

### 实践任务

1. 从头实现小型 Transformer，并在序列任务上验证 mask、位置编码和解码策略。
2. 训练一个视觉或文本编码器，比较监督、对比和自监督目标。
3. 实现小型 Diffusion 或 Flow Matching 项目，并完成采样速度与质量对比。

### 可交付成果

- Transformer 实现
- 生成模型实验
- 消融与失败分析

### 验收标准

- 能解释架构、目标函数和数据如何共同决定模型行为。
- 能阅读并修改主流模型代码，而不是只调用推理 API。
- 每个结论都有基线、消融和失败案例支持。

### 技术关键词索引

- `ACT`、`attention`、`BaiChuan`、`bert`、`bge-m3`、`ChatGPT`、`DLRM`、`gpt`
- `GR`、`HTML`、`LLM-based`、`MindGPT`、`MindStudio`、`MindX`、`MiniCPM`、`Model-Based`
- `PPT`、`PSI`、`Representation Engineering`、`SeedVL`、`TMT`、`transformer`、`Transformer Architecture`、`偏好模型`
- `基础模型`、`大模型`、`大语言模型`、`奖励模型`、`扩散模型`、`深度学习`、`生成模型`、`语言模型`

## 阶段 3：数据工程、语料治理与合成数据

> 建议节奏：4-7 周。能力模块：7 个。

### 学习目标

建立从原始数据到可训练、可评测、可追溯数据资产的完整流水线。

### 必学内容

> 能力闭环：原理研究与复现、数据构建与治理、算法建模与实现、训练调优与稳定性、评测、消融与误差分析、工程系统与工具链、部署、性能与运维、场景适配与产品闭环、安全、鲁棒与合规。

- **模块 3.1**：采集、解析、清洗、去重、过滤、脱敏、标注、质量评分、版本和血缘。
- **模块 3.2**：Web-scale 数据处理、MinHash/LSH、近重复、污染检测、版权与隐私约束。
- **模块 3.3**：Tokenizer：BPE、Byte-level、SentencePiece、Tiktoken、词表训练与压缩率。
- **模块 3.4**：数据配比、Data Mixture、Curriculum、难例挖掘、主动学习和数据价值评估。
- **模块 3.5**：指令、偏好、过程、轨迹、多模态和领域数据的 schema 与质量标准。
- **模块 3.6**：Synthetic Data、Self-Instruct、拒绝采样、蒸馏数据、Self-Play 和数据飞轮。
- **模块 3.7**：Spark/Flink/Hadoop/Ray、对象存储、流批处理、并行预处理和增量更新。

### 实践任务

1. 构建百万级样本清洗去重管线，输出每一步保留率、质量变化和成本。
2. 训练并比较两个 Tokenizer，分析领域词、长文本和多语言的编码效率。
3. 生成一批指令或偏好数据，建立自动过滤与人工抽检协议。
4. 做数据配比消融，把性能变化归因到来源、质量、难度和污染。

### 可交付成果

- 版本化数据集
- 数据质量报告
- 合成数据与配比实验

### 验收标准

- 任一模型结果可追溯到数据版本、规则和样本分布。
- 能量化去重、过滤、配比和合成策略的收益与副作用。
- 训练集、验证集、评测集之间不存在不可解释的污染。

### 技术关键词索引

- `/ Synthetic Environments`、`AB`、`Adaptive Curriculum Learning`、`AUC`、`CC`、`CDK`、`ClickHouse`、`ColossalAl`
- `Cross-functional Collaboration（产品 / 设计 / 数据 / 人文训练师团队）`、`Data Curriculum`、`Data Filtering`、`Data Filtering / Quality Filtering / Deduplication`、`Data Mixture Optimization`、`Data Mixture Optimization / Data Curriculum`、`Data-Centric`、`Deduplication`
- `DFL`、`DFT`、`DSP`、`EasyR1`、`ETA`、`Graph-LLM`、`HDFS`、`High-Quality Data Curation`
- `IMAGE`、`IO`、`IoT`、`IoU`、`KS`、`LaMMA-Factory`、`LGB`、`LR`
- `LTV`、`mAP`、`MapReduce`、`MaxCompute`、`MongoDB`、`MQTT`、`MVS`、`NebulaGraph`
- `Neo4j`、`NLU`、`NoSQL`、`PB`、`PE`、`Pipeline`、`Quality Filtering`、`query-doc`
- `retrieval`、`RTOS`、`S3`、`SfM`、`Synthetic Data Generation`、`Table+`、`TB`、`UMI`
- `user-item`、`Web-scale Data Filtering`、`Web-scale Data Filtering / Quality Filtering / Deduplication`、`Web-scale Data Processing Pipeline`、`XGB`、`偏好数据`、`合成数据`、`向量数据库`
- `数据分析`、`数据去重`、`数据合成`、`数据标注`、`数据治理`、`数据清洗`、`数据管线`、`数据质量`
- `数据配比`

## 阶段 4：预训练、模型架构、MoE 与长上下文

> 建议节奏：6-10 周。能力模块：7 个。

### 学习目标

理解并实践基础模型从数据配方到稳定训练、扩展和架构迭代的核心方法。

### 必学内容

> 能力闭环：原理研究与复现、数据构建与治理、算法建模与实现、训练调优与稳定性、评测、消融与误差分析、工程系统与工具链、部署、性能与运维、场景适配与产品闭环、安全、鲁棒与合规、原理理解与实际应用。

- **模块 4.1**：Next-token prediction、自监督预训练、Continued/Domain-Adaptive Pre-training。
- **模块 4.2**：Scaling Laws、Compute-optimal、Chinchilla、数据/参数/算力预算和训练曲线外推。
- **模块 4.3**：Dense Transformer、MoE、稀疏激活、路由、专家负载均衡和容量规划。
- **模块 4.4**：AdamW/Lion/Muon/Sophia、学习率、warmup、cosine decay、梯度裁剪和稳定性。
- **模块 4.5**：长上下文：RoPE、YaRN、NTK-aware、ALiBi、Ring Attention 和上下文扩展。
- **模块 4.6**：FlashAttention、Activation/Gradient Checkpointing、BF16/FP8 和训练效率。
- **模块 4.7**：数据混合、Tokenizer、checkpoint、loss spike、退化检测和预训练评测。

### 实践任务

1. 预训练一个小型语言模型，完整记录 token、FLOPs、loss、吞吐和验证能力。
2. 完成一次领域继续预训练，与直接 SFT 做公平对比。
3. 实现或复现小型 MoE，分析路由分布、负载不均和通信成本。
4. 扩展上下文窗口，比较位置编码外推、训练成本和长文本真实收益。

### 可交付成果

- 预训练配方
- Scaling 实验
- MoE/长上下文报告

### 验收标准

- 能从数据、优化、数值、并行和模型结构定位训练异常。
- 能用预算约束解释模型与数据规模选择。
- 能证明长上下文或 MoE 改动带来真实任务收益，而非只改善单一指标。

### 技术关键词索引

- `/ Sparse MoE`、`AD`、`AdamW`、`ADK`、`ALiBi`、`Autoregressive Language Modeling`、`Baichuan-Omni`、`BERT4Rec`
- `CFA`、`Chinchilla-optimal`、`CLIP-type`、`Continued Pre-training`、`Continued Pre-training / Domain-Adaptive Pre-training`、`Cosine Decay`、`CosyVoice`、`CP`
- `CPA`、`Cross-Attention`、`Cross-lingual`、`DCN`、`DeepSpeed-MoE`、`DiT`、`Domain-Adaptive Pre-training`、`DP`
- `DualPipe`、`E2E`、`embedding`、`EP`、`EPLB`、`FlashMLA`、`FRM`、`GQA`
- `GraphRAG`、`Hessian-free`、`HSTU`、`K+`、`KV`、`Learning Rate Schedule / Warmup / Cosine Decay`、`Lion`、`Long-context`
- `Long-Context Modeling`、`Long-Context Pre-training`、`LongCat`、`Mid-training`、`MindDiffusion`、`MindFormers`、`MindPet`、`MindSpeed-LLM`
- `Mixture of Experts`、`Mixture of Experts（MoE）`、`MLA`、`MoE`、`MoE Pre-training`、`MoE Routing`、`MTP`、`Multi-task`
- `Muon`、`MuP`、`Next-Token Prediction / Autoregressive Language Modeling`、`NTK-aware Scaling`、`NTP`、`Omni-model`、`Optimizer`、`Optimizer（AdamW / Lion / Muon / Sophia）`
- `PaddleOCR`、`PAI`、`PP`、`Pre-training`、`Pre-training / Self-Supervised Pre-training`、`ReAct`、`ResidualNet`、`Ring Attention`
- `RoPE`、`RoPE / YaRN / NTK-aware Scaling / ALiBi`、`scale-up`、`Scaling Laws`、`Scaling Laws / Compute-Optimal Training / Chinchilla-optimal`、`self-attention`、`self-improve`、`Self-Supervised Pre-training`
- `Sophia`、`SP`、`Sparse Activation`、`Sparse Activation / MoE Routing`、`STEM`、`Tokenizer`、`TP`、`Transformer Architecture / Mixture of Experts（MoE） / Sparse MoE`
- `U2++`、`V-LLM`、`VAE`、`VALLE`、`VITS`、`VQ`、`Warmup`、`Wav2Vec`
- `WLLM`、`XML`、`YaRN`、`自监督预训练`、`长上下文`、`预训练`

## 阶段 5：后训练、PEFT、强化学习与模型对齐

> 建议节奏：7-12 周。能力模块：7 个。

### 学习目标

掌握从 SFT、偏好优化、奖励模型到可验证强化学习和持续对齐的完整链路。

### 必学内容

> 能力闭环：原理研究与复现、数据构建与治理、算法建模与实现、训练调优与稳定性、评测、消融与误差分析、工程系统与工具链、部署、性能与运维、场景适配与产品闭环、安全、鲁棒与合规、原理理解与实际应用。

- **模块 5.1**：SFT、Instruction Tuning、Cold Start、数据配方、模板、packing 和灾难性遗忘。
- **模块 5.2**：LoRA/QLoRA/DoRA/AdaLoRA、Adapter、Prompt/Prefix/P-Tuning 与全参微调。
- **模块 5.3**：Preference Modeling、Reward Model、PRM/ORM、Process/Outcome Supervision。
- **模块 5.4**：RLHF、RLAIF、RLVR、PPO、GRPO、DAPO、RLOO、DPO、SimPO、KTO、ORPO。
- **模块 5.5**：Verifiable Reward、Advantage、credit assignment、reward hacking 和训练稳定性。
- **模块 5.6**：Scalable Oversight、Constitutional AI、Weak-to-Strong、Debate、IDA 与递归改进。
- **模块 5.7**：持续后训练、在线反馈、Self-Play、Self-Improvement、模型退化和安全边界。

### 实践任务

1. 完成 SFT 与 LoRA/QLoRA 对比，记录精度、显存、吞吐和遗忘。
2. 构建偏好数据，训练奖励模型并完成 DPO/GRPO 中至少一种方法。
3. 为数学、代码或工具任务设计可验证奖励，分析作弊和奖励投机。
4. 比较过程监督和结果监督，做 KL、长度、格式和泛化消融。

### 可交付成果

- 后训练流水线
- 偏好与奖励数据
- 对齐/RL 实验报告

### 验收标准

- 能说明不同后训练方法的目标、数据需求、偏差和适用边界。
- 能诊断 reward collapse、长度偏好、KL 漂移、模式坍塌和验证集过拟合。
- 模型提升同时经过能力、安全、风格、事实性和回归评测。

### 技术关键词索引

- `/ OpenRLHF`、`/ Outcome Reward Model`、`/ Preference Modeling`、`AdaLoRA`、`Adapter Tuning`、`Adapter Tuning / Prompt Tuning / Prompt Tuning v2`、`AgenticSearch`、`alignment`
- `AutoML`、`Best-of-N`、`CodeAgent`、`Context Window Extension / Long-Context Fine-tuning`、`Continuous Post-Training`、`CoT`、`DAPO`、`Debate-style Oversight`
- `DecoderOnly`、`DeepResearch`、`DeepSpeed-Chat`、`DJI`、`DocVQA`、`DPO`、`DPO / SimPO / KTO / ORPO / RLOO / RiskPO`、`Dr. GRPO`
- `Dynamic Preference Modeling`、`End-to-End Post-Training`、`fine-tuning`、`FT`、`Full-parameter Fine-tuning vs PEFT 混合策略`、`GPT-4V`、`GRPO`、`GRPO / DAPO / Dr. GRPO`
- `GSPO`、`ICL`、`IM`、`IPO`、`ISP`、`Iterative Post-Training`、`Iterative Post-Training / Continuous Post-Training`、`JD-LLM`
- `KL`、`KTO`、`KV-Cache`、`Large-scale Preference Data Pipeline`、`Latent Space Alignment`、`Latent Space Alignment / Representation Engineering（RepE）`、`LLaMA-2`、`LLaMA-Factory`
- `Long-Context Fine-tuning`、`Long-Term`、`Long-thought`、`LoRA`、`LoRA / QLoRA / DoRA / AdaLoRA / Prefix Tuning / P-Tuning`、`Meta-learning`、`Model`、`MoE Fine-tuning`
- `MoE Post-training`、`MoE Pre-training / MoE Post-training / MoE Fine-tuning`、`MS-Swift`、`NL2SQL`、`On-policy`、`OneTrans`、`OPD`、`OpenRLHF`
- `ORPO`、`Parameter-Efficient Fine-Tuning`、`PEFT`、`PEFT（Parameter-Efficient Fine-Tuning）`、`Post-Training`、`Post-Training Pipeline`、`Post-Training Pipeline / End-to-End Post-Training`、`PostTraining`
- `PPL`、`Preference Collection Pipeline`、`Preference Data Construction`、`Preference Data Construction / Synthetic Preference Data`、`Pretraining`、`Process Reward Model`、`Process Reward Model（PRM） / Outcome Reward Model（ORM）`、`QLoRA`
- `Qwen2.5-Coder`、`RankMixer`、`Reasoning`、`Recursive Reward Modeling`、`Recursive Reward Modeling（RRM）`、`Recursive Self-Improvement`、`Recursive Self-Improvement（RSI）`、`Reinforcement Learning Post-Training`
- `Reinforcement Learning with Verifiable Rewards`、`reward model`、`Reward Modeling`、`Reward Modeling（RM） / Preference Modeling`、`RL`、`RL Stage / Reinforcement Learning Post-Training`、`RLAIF`、`RLHF`
- `RLHF / RLAIF / RLVF`、`RLOO`、`RLVR`、`RLVR（Reinforcement Learning with Verifiable Rewards）`、`ROLL`、`rollout`、`SAM`、`Scalable Oversight`
- `Self-Alignment`、`Self-Improvement loops`、`Self-Play`、`SFT`、`SFT / Supervised Fine-Tuning / Instruction Tuning / Cold Start`、`SimPO`、`SL`、`STR`
- `Supervised Fine-Tuning`、`Synthetic Data Generation / Synthetic Preference Data Pipeline`、`Synthetic Preference Data`、`Synthetic Preference Data Pipeline`、`Tiktoken`、`Tokenizer（BPE / Byte-level / SentencePiece / Tiktoken）`、`ToSQL`、`Training`
- `UGC`、`Value Alignment`、`VeOmni`、`Verifiable Reward`、`Verifiable Reward / Verifiable Feedback`、`verifier-driven`、`verl（ByteDance RL Framework） / OpenRLHF / TRL / Axolotl`、`Vision-Language`
- `Weak-to-Strong Generalization`、`Weak-to-Strong Generalization / Weak-to-Strong Preference Optimization`、`Weak-to-Strong Preference Optimization`、`价值对齐`、`后训练`、`对齐`、`微调`、`指令微调`
- `模型对齐`

## 阶段 6：RAG、Agent、工具调用、记忆与长程规划

> 建议节奏：6-10 周。能力模块：7 个。

### 学习目标

构建能检索、规划、调用工具、维护状态并在长任务中稳定恢复的智能体系统。

### 必学内容

> 能力闭环：原理研究与复现、数据构建与治理、算法建模与实现、训练调优与稳定性、评测、消融与误差分析、工程系统与工具链、部署、性能与运维、场景适配与产品闭环、安全、鲁棒与合规、原理理解与实际应用。

- **模块 6.1**：RAG：切分、Embedding、召回、混合检索、重排、引用、权限和知识更新。
- **模块 6.2**：Function Calling、Tool Use、MCP/API、代码执行、浏览器/数据库工具和沙箱。
- **模块 6.3**：ReAct、Plan-and-Execute、反思、搜索、任务分解、状态机和工作流编排。
- **模块 6.4**：短期/长期/情景/语义记忆、跨会话记忆、检索、压缩、衰减和隐私。
- **模块 6.5**：Multi-Agent、角色协作、通信协议、冲突、共享状态和可观测性。
- **模块 6.6**：Agentic RL、长程 rollout、异步环境、轨迹优化、Agent World Model。
- **模块 6.7**：Harness Engineering、失败恢复、幂等、成本、延迟、安全和长期 Benchmark。

### 实践任务

1. 实现带引用的混合检索 RAG，评估召回、重排、答案事实性和权限泄漏。
2. 构建能调用代码、搜索和数据库的 Agent，加入超时、重试和执行审计。
3. 设计跨会话记忆实验，比较摘要、向量检索、结构化记忆和遗忘策略。
4. 完成一个 20 步以上任务 Benchmark，统计成功率、错误传播、成本和恢复率。

### 可交付成果

- RAG 系统
- 长程 Agent
- 轨迹与失败分类报告

### 验收标准

- 检索、推理、工具、记忆和环境错误可以被独立定位。
- Agent 输出可追踪到证据、工具调用和状态变化。
- 长任务提升基于可重复 Benchmark，而非演示案例。

### 技术关键词索引

- `Agent`、`Agent World Model`、`Agent World Model（AWM） / Synthetic Environments`、`Agent-RL`、`Agent-to-Agent`、`AgentBench`、`Agentic RL`、`Agentic RL / Tool-augmented RL / Long-horizon RL`
- `Agentic-RL`、`AgenticLLM`、`AgentOS`、`Agents`、`AgentScope`、`AI-Agent`、`AIagents`、`AIGuard`
- `APP`、`arXiv`、`AutoGen`、`AutoGPT`、`B2B`、`Chain-of-Thought`、`chatBI`、`Coevolving World Model`
- `Connector`、`Continual Learning`、`CrewAI`、`CRM`、`crop_video`、`Cross-session Memory`、`Cross-session Memory / Behavioral State Decay Mitigation`、`DataEngineer`
- `DE`、`DeepReaserch`、`Dynamic Preference Modeling / Personalization`、`Episodic-Semantic Memory`、`Fact-Checking`、`FRIDAY`、`FunctionCalls`、`GB10`
- `GUI`、`Hierarchical Agentic RL`、`Hierarchical Agentic RL / Multi-Agent RL`、`Hierarchical Long-Term Memory`、`Hierarchical Long-Term Memory / Episodic-Semantic Memory`、`IDE`、`LangChain`、`LangGraph`
- `Lifelong Learning`、`Lifelong Learning / Continual Learning`、`LlamaIndex`、`LLM-as-a-Judge`、`long-horizon`、`Long-horizon Planning`、`Long-horizon Planning / Strategic Tool Utilization`、`Long-horizon RL`
- `Long-term Memory Modeling`、`MaaS`、`MCP`、`Memory`、`Multi-Agent`、`Multi-Agent RL`、`MultiAgent`、`o3`
- `OpenClaw`、`OS`、`Persistent Memory`、`Persistent Memory / Long-term Memory Modeling`、`Personalization`、`planning`、`Pre-trained`、`Proactive Memory Agent`
- `PydanticAI`、`QA`、`R1`、`Scalable Agentic Rollouts`、`Self-Evolution`、`Self-Evolving Agents`、`Self-Instruct`、`Self-Play / Self-Improvement loops / Self-Evolving Agents`
- `SKILL`、`Stateful Agents`、`Stateful Agents / Proactive Memory Agent`、`Strategic Tool Utilization`、`Tool-augmented RL`、`Tool-use`、`ToT`、`UGC+`
- `workflow`、`XZ0000602`、`zoom-in`、`多智能体`、`工具调用`、`智能体`、`长期记忆`

## 阶段 7：多模态、视觉语言、语音、视频与具身模型

> 建议节奏：7-12 周。能力模块：7 个。

### 学习目标

掌握图像、视频、语音、文本和动作的表示、对齐、生成及统一建模。

### 必学内容

> 能力闭环：原理研究与复现、数据构建与治理、算法建模与实现、训练调优与稳定性、评测、消融与误差分析、工程系统与工具链、部署、性能与运维、场景适配与产品闭环、安全、鲁棒与合规、原理理解与实际应用。

- **模块 7.1**：视觉编码器、CLIP、ViT、投影层、Cross-Attention、Q-Former 和多模态 Token。
- **模块 7.2**：图文/视频理解、OCR、检测、分割、Grounding、时空建模和开放词汇感知。
- **模块 7.3**：ASR、TTS、音频编码、语音对话、说话人和音视频同步。
- **模块 7.4**：VLM 指令微调、多模态预训练、数据混合、对齐、幻觉和细粒度评测。
- **模块 7.5**：文生图/视频、Diffusion/Flow、可控生成、编辑、一致性和生成评测。
- **模块 7.6**：VLA、World Model、行为预测、动作生成、机器人学习和自动驾驶多模态。
- **模块 7.7**：多模态部署的输入管线、显存、延迟、帧率、分辨率和端侧约束。

### 实践任务

1. 微调一个开源 VLM，建立视觉遗漏、文本误解、时序错误和幻觉分类。
2. 完成图像/视频/语音至少两种模态的联合任务与单模态基线对比。
3. 实现小型可控生成或编辑项目，评估质量、一致性、可控性和安全。
4. 选择 VLA、自动驾驶或具身方向，完成数据到闭环评测的最小项目。

### 可交付成果

- 多模态微调项目
- 生成模型项目
- 模态错误分析

### 验收标准

- 能解释不同模态的编码、对齐、融合和解码选择。
- 评测覆盖感知、理解、推理、生成、时序和安全。
- 能把多模态失败归因到数据、编码器、对齐、语言模型或解码器。

### 技术关键词索引

- `A2A`、`ActionInsight`、`AdaRound`、`Agent+`、`AI4SE`、`AIGC`、`AIOps`、`API`
- `asr`、`BEV`、`BLIP`、`CapCut`、`Claude3.5Sonnet`、`CLE`、`CLIP`、`CNC`
- `Co-pilot`、`CoTraining`、`CPT`、`DALL-E`、`DataLoader`、`DAU`、`DETR`、`DiffusionModels`
- `DINO`、`DM`、`dots.mocr`、`dots.mocr-svg`、`dots.ocr`、`DVR`、`Few-shot`、`FlexRound`
- `FLUX`、`FM`、`FP`、`GAN`、`GenAI`、`GenFlare`、`GenX`、`GNSS`
- `GOD`、`GPT-4o`、`GPT4V`、`Grounding`、`HDR`、`HyperOS`、`i2i`、`IC`
- `ID`、`Image2Code`、`IMU`、`INT`、`JSON`、`k+star`、`LBS`、`LiDAR`
- `LLaVA`、`LLM+Recsys`、`llm4rec`、`LLMOps`、`LLMs`、`LMM`、`MindSpeed`、`MindSpeed-MM`
- `ML`、`MLLM`、`MLOps`、`MMBench`、`MR`、`Multimodal Pre-training`、`NCNN`、`OmniGuard`
- `OneRec`、`OneSearch`、`OpenCompass`、`OpenVLA`、`OPPO`、`Outcome Supervision`、`PDF`、`PGC`
- `POC`、`POI`、`Post-Train`、`Pre-train`、`PrefixQuant`、`Process Supervision`、`Process Supervision / Outcome Supervision`、`PSIG`
- `Qwen-Audio`、`Qwen-Coder`、`Qwen-Image`、`Qwen-Omni`、`Qwen-VL`、`QwenVL`、`RFT`、`RKNN`
- `ROI`、`ROL`、`RPM`、`RQ-VAE`、`RTK`、`RTL`、`ScalingUp`、`SenseNova`
- `SOP`、`SOTA`、`SpeechLLM`、`SpinQuant`、`SVA`、`TA`、`tts`、`UI`
- `UID`、`Video-LLaMA`、`VL`、`vla`、`vlm`、`VLMs`、`VP`、`VQ-VAE`
- `VQA`、`WAN`、`Wav2Vec2`、`WLM`、`world model`、`XR`、`YOLO`、`YOLOX`
- `Z-Image`、`ZeRO`、`多模态`、`生成式AI`、`视觉语言`、`视频剪辑`

## 阶段 8：评测、Benchmark、实验工程与因果分析

> 建议节奏：4-7 周并持续进行。能力模块：7 个。

### 学习目标

建立可信、可复现、能驱动迭代的离线与在线评测体系。

### 必学内容

> 能力闭环：原理研究与复现、数据构建与治理、算法建模与实现、训练调优与稳定性、评测、消融与误差分析、工程系统与工具链、部署、性能与运维、场景适配与产品闭环、安全、鲁棒与合规、原理理解与实际应用。

- **模块 8.1**：能力矩阵、Benchmark 设计、数据污染、难度分层、切片和黄金集。
- **模块 8.2**：准确率、F1、NDCG、BLEU/ROUGE、困惑度、校准、鲁棒性和任务成功率。
- **模块 8.3**：LLM-as-Judge、pairwise/rubric、位置偏差、自洽、校准和人工复核。
- **模块 8.4**：Human Evaluation、偏好采集、一致性、标注规范和质量控制。
- **模块 8.5**：A/B Testing、在线指标、护栏指标、方差缩减、序贯检验和 uplift。
- **模块 8.6**：实验管理、消融、回归、显著性、错误分类、bad case 和根因分析。
- **模块 8.7**：Agent/长程/多模态/安全评测以及性能、成本、延迟的联合评价。

### 实践任务

1. 为已有项目建立分层评测集、自动评测、人工复核和回归门禁。
2. 比较至少两种 Judge 策略，并用人工标签分析偏差和置信度。
3. 设计一次 A/B 或离线反事实实验，给出效应量、区间和上线决策。
4. 建立 bad case 数据库，把错误归因和修复动作接入下一轮训练。

### 可交付成果

- 评测框架
- Benchmark 数据卡
- 实验与回归报告

### 验收标准

- 指标与真实目标一致，切片结果和不确定性透明。
- 结论能够经重复实验、统计检验和人工审计。
- 每次模型迭代都有能力、安全、性能和成本回归。

### 技术关键词索引

- `a/b`、`A/B Testing`、`A/B Testing / Online Experimentation / Uplift Modeling`、`AED`、`Agentic Benchmark`、`Agentic Benchmark / Long-horizon Benchmark`、`ATH`、`Automated Evaluation Framework`
- `Automated Judging Pipeline`、`BadCase`、`benchmark`、`ChatSVA`、`CNN`、`DMC`、`ECU`、`ELO`
- `Framework`、`GAIA`、`GenBen`、`GSB`、`Human Evaluation`、`Human Evaluation / Preference Collection Pipeline`、`learning-based`、`LLM-as-Judge`
- `LLM-as-Judge / Automated Judging Pipeline`、`Long-Context Evaluation`、`Long-horizon Benchmark`、`MMLU`、`MOS`、`Online Experimentation`、`Online Experimentation & Iteration`、`PRD`
- `Process vs Outcome Evaluation`、`QPS`、`RNN`、`RT`、`RTL-LLM`、`RTLCoder`、`Seed3D`、`Self-Correction`
- `SigLIP`、`TPOT`、`TTFT`、`Uplift Modeling`、`VeriCoder`、`Verifiable Rewards Evaluation`、`VerilogEval`、`VeriRL`
- `消融实验`、`评测`

## 阶段 9：AI 安全、红队、鲁棒性、事实性与隐私

> 建议节奏：4-8 周并持续进行。能力模块：7 个。

### 学习目标

把安全从上线前检查变成数据、训练、评测和服务全链路能力。

### 必学内容

> 能力闭环：原理研究与复现、数据构建与治理、算法建模与实现、训练调优与稳定性、评测、消融与误差分析、工程系统与工具链、部署、性能与运维、场景适配与产品闭环、安全、鲁棒与合规。

- **模块 9.1**：Safety/Value Alignment、政策约束、拒答边界、Constitutional AI 和安全数据。
- **模块 9.2**：Jailbreak、Prompt Injection、越权工具调用、数据外泄和间接注入。
- **模块 9.3**：自动红队、攻击生成、对抗样本、风险分类、严重度和修复验证。
- **模块 9.4**：幻觉、事实性、引用、校准、不确定性、知识时效和 grounded generation。
- **模块 9.5**：Reward Hacking、欺骗、后门、模型窃取、投毒、成员推断和隐私。
- **模块 9.6**：Representation Engineering、Circuit Breaker、鲁棒训练和监控告警。
- **模块 9.7**：内容、金融、风控、医疗、驾驶和 Agent 执行中的领域安全约束。

### 实践任务

1. 为 RAG/Agent 系统建立威胁模型，覆盖注入、越权、泄漏和危险执行。
2. 构建自动红队集，记录攻击成功率、误拒率、修复率和能力损失。
3. 实现事实性与引用评测，比较检索、训练和解码层修复。
4. 为工具调用加入最小权限、确认、沙箱、审计和回滚。

### 可交付成果

- 威胁模型
- 红队与安全评测集
- 修复及回归报告

### 验收标准

- 风险有明确资产、攻击面、严重度、检测和缓解措施。
- 安全提升不会用不可解释的能力损失换取。
- 上线系统具备持续监控、事件复盘和快速回滚。

### 技术关键词索引

- `AI Safety via Debate`、`AI Safety via Debate / Debate-style Oversight`、`Automated Red Teaming`、`Automated Red Teaming / Red-teaming Automation`、`CBG`、`Circuit Breakers`、`Circuit Breakers / Representation Engineering for Safety`、`Constitutional AI`
- `Constitutional AI / Self-Critique / Self-Alignment`、`Constitutional Constraints`、`Deceptive Behavior Mitigation`、`Factuality Alignment`、`Hallucination Mitigation`、`Hallucination Mitigation / Factuality Alignment`、`Jailbreak Defense`、`Jailbreak Defense / Adversarial Robustness`
- `Representation Engineering for Safety`、`Reward Hacking Prevention`、`Reward Hacking Prevention / Deceptive Behavior Mitigation`、`Safety Alignment`、`Safety Alignment / Value Alignment`、`事实性`、`安全对齐`、`对抗鲁棒性`
- `幻觉`、`红队`、`越狱`、`越狱防御`、`鲁棒性`

## 阶段 10：分布式训练、训练平台与 AI Infra

> 建议节奏：5-9 周。能力模块：7 个。

### 学习目标

能把模型稳定扩展到多卡、多机和大规模训练平台，并定位性能与可靠性瓶颈。

### 必学内容

> 能力闭环：原理研究与复现、数据构建与治理、算法建模与实现、训练调优与稳定性、评测、消融与误差分析、工程系统与工具链、部署、性能与运维、场景适配与产品闭环、原理理解与实际应用。

- **模块 10.1**：DDP、ZeRO、FSDP、Tensor/Pipeline/Sequence/Expert Parallel 和 3D 并行。
- **模块 10.2**：DeepSpeed、Megatron-LM、Colossal-AI、verl、OpenRLHF、TRL 和 Ray。
- **模块 10.3**：NCCL、拓扑、通信、重叠、sharding、checkpoint、恢复和容错。
- **模块 10.4**：BF16/FP16/FP8、Gradient/Activation Checkpointing、FlashAttention 和内存估算。
- **模块 10.5**：数据加载、样本 packing、动态 batch、吞吐、MFU、straggler 和长尾任务。
- **模块 10.6**：Kubernetes、队列、资源调度、弹性、优先级、可观测性和成本治理。
- **模块 10.7**：异步 rollout、环境隔离、偏好数据管线和后训练基础设施。

### 实践任务

1. 把单卡训练迁移到 DDP/FSDP/DeepSpeed，逐层记录吞吐、显存和通信。
2. 完成一次故障注入：进程、节点、存储或网络失败后自动恢复。
3. 实现训练任务调度与监控面板，统计 GPU 利用率、排队和失败原因。
4. 搭建小型异步 rollout 或偏好训练链路，分析生产者消费者瓶颈。

### 可交付成果

- 多卡训练方案
- Profiling 报告
- 容错与调度实验

### 验收标准

- 能估算不同模型、序列和并行策略的显存与通信。
- 能定位计算、通信、I/O、数据、调度和 straggler 瓶颈。
- 训练可恢复、可观测、可复现，并有明确成本指标。

### 技术关键词索引

- `3D Parallelism`、`3D Parallelism / Activation Checkpointing`、`3D Parallelism / Pipeline Parallelism / Tensor Parallelism`、`A100`、`Activation Checkpointing`、`Activation Checkpointing / Gradient Checkpointing`、`ADAS`、`AI-infra`
- `Asynchronous Rollout Infrastructure`、`B+`、`BF16`、`Checkpoint`、`Colossal-AI`、`ColossalAI`、`CPUGPU`、`DDP`
- `deepspeed`、`DeepSpeed ZeRO`、`DeepSpeed ZeRO / FSDP / Megatron-DeepSpeed`、`FlashAttention`、`FlashAttention / Ring Attention / FlashAttention-2/3`、`FlashAttention for Long Context`、`FlashAttention-2/3`、`FP8`
- `FSDP`、`GLM`、`GPU`、`GPU+CPU`、`Gradient Checkpointing`、`HPC`、`HugeCTR`、`Kernel`
- `Kubernetes for Distributed Training`、`megatron`、`Megatron-DeepSpeed`、`Megatron-LM`、`Mixed Precision Training（BF16 / FP8）`、`MPI`、`oneDNN`、`PC`
- `Pipeline Parallelism`、`Ray`、`Ray / Kubernetes for Distributed Training`、`Ring Attention / FlashAttention for Long Context`、`SGLang`、`Tensor Parallelism`、`TileLang`、`UCX`
- `XZ0000582`、`ZeRO-1`、`Zero-Shot`、`分布式系统`、`分布式训练`、`异构集群`、`张量并行`、`数据并行`
- `模型并行`、`流水线并行`

## 阶段 11：推理、Serving、压缩与性能优化

> 建议节奏：5-8 周。能力模块：7 个。

### 学习目标

在精度、延迟、吞吐、显存、成本和可靠性之间做可量化取舍。

### 必学内容

> 能力闭环：原理研究与复现、数据构建与治理、算法建模与实现、训练调优与稳定性、评测、消融与误差分析、工程系统与工具链、部署、性能与运维、场景适配与产品闭环、原理理解与实际应用。

- **模块 11.1**：Prefill/Decode、KV Cache、PagedAttention、Continuous Batching 和调度。
- **模块 11.2**：vLLM、TensorRT-LLM、TGI、Triton、ONNX Runtime、llama.cpp 和服务框架。
- **模块 11.3**：INT4/INT8/FP8、GPTQ/AWQ、量化感知、蒸馏、剪枝和稀疏化。
- **模块 11.4**：Speculative Decoding、并行解码、Prefix Cache、长上下文和多 LoRA Serving。
- **模块 11.5**：CUDA/Triton Kernel、算子融合、内存带宽、CPU/GPU 协同和硬件感知优化。
- **模块 11.6**：在线批处理、限流、熔断、降级、灰度、弹性、SLA 和多租户。
- **模块 11.7**：质量回归、监控、漂移、成本、容量规划和端侧部署。

### 实践任务

1. 用 vLLM/TensorRT-LLM 部署模型，测量 TTFT、TPOT、吞吐、显存和成本。
2. 完成两种量化或蒸馏方案，比较分任务精度和硬件收益。
3. 为长上下文、多并发和突发流量设计容量与降级策略。
4. 优化一个 CUDA/Triton 或数据预处理热点，给出 profiler 证据。

### 可交付成果

- 推理服务
- 性能基准
- 压缩与容量规划报告

### 验收标准

- 性能结论覆盖不同 batch、序列、并发和硬件。
- 压缩后的能力、安全和长尾任务经过回归。
- 系统达到明确 SLA，并能解释瓶颈与扩容策略。

### 技术关键词索引

- `AMP`、`AutoResearch`、`AWQ`、`AWS`、`Continuous Batching`、`CPU`、`Distillation`、`FP16`
- `GCP`、`GMV`、`GPTQ`、`Hardware-aware Optimization`、`Inference Optimization`、`Inference Optimization / Serving Infrastructure`、`Iterated Distillation and Amplification`、`Iterated Distillation and Amplification（IDA）`
- `KV Cache Optimization`、`KV Cache Optimization / PagedAttention / Continuous Batching`、`LVM`、`MNN`、`Model Compression`、`PagedAttention`、`Privacy-Preserving`、`PTQ`
- `QAT`、`QNX`、`Quantization`、`Quantization（INT4 / INT8 / FP8 / GPTQ / AWQ）`、`RTC`、`Serving`、`Serving Infrastructure`、`SLA`
- `Speculative Decoding`、`Speculative Decoding / Model Compression / Distillation`、`tensorrt`、`TensorRT-LLM`、`test-time`、`TGI`、`ToB`、`TPU`
- `Triton Inference Server`、`TVM`、`vLLM`、`vLLM / TensorRT-LLM / TGI / Triton Inference Server`、`WebSocket`、`X-Learner`、`剪枝`、`推理优化`
- `推理加速`、`推理引擎`、`模型压缩`、`模型蒸馏`、`模型量化`、`蒸馏`、`量化`

## 阶段 12：垂直算法分支与业务建模

> 建议节奏：任选 1-2 个方向，各 6-10 周。能力模块：8 个。

### 学习目标

把通用模型能力转化为领域问题定义、数据、指标、约束和线上收益。

### 必学内容

> 能力闭环：原理研究与复现、数据构建与治理、算法建模与实现、训练调优与稳定性、评测、消融与误差分析、工程系统与工具链、部署、性能与运维、场景适配与产品闭环、安全、鲁棒与合规、原理理解与实际应用。

- **模块 12.1**：NLP：分类、抽取、检索、问答、翻译、文本生成、知识图谱和多语言。
- **模块 12.2**：CV/AIGC：检测、分割、OCR、三维、图像/视频理解、生成、编辑和质量评价。
- **模块 12.3**：语音：ASR、TTS、声学/语言模型、端到端、多语种、对话和实时流式。
- **模块 12.4**：搜索/推荐/广告：召回、排序、重排、CTR/CVR、序列建模、生成式推荐和因果。
- **模块 12.5**：图学习、时序预测、异常检测、风控反欺诈、金融建模和数据挖掘。
- **模块 12.6**：运筹优化：规划、调度、组合优化、启发式、整数规划和学习增强优化。
- **模块 12.7**：自动驾驶/机器人：感知、预测、规划、控制、VLA、世界模型和闭环安全。
- **模块 12.8**：业务抽象、目标函数、代理指标、约束、冷启动、反馈回路和 Research-to-Production。

### 实践任务

1. 选择一个方向建立传统算法、深度模型和基础模型三层基线。
2. 从业务目标推导训练目标和评测指标，分析代理指标与真实价值偏差。
3. 打通离线数据、训练、评测、在线服务、监控和反馈闭环。
4. 做跨域、冷启动、长尾、分布漂移和成本约束实验。

### 可交付成果

- 领域端到端项目
- 业务指标树
- 线上化与迭代方案

### 验收标准

- 能把模糊需求转成可验证的算法问题和约束。
- 结果同时优于简单基线，并解释线上线下差异。
- 方案包含数据、算法、系统、评测、安全和成本。

### 技术关键词索引

- `AF`、`AIMET`、`APA`、`AR`、`AUDIO`、`AVP`、`catboost`、`CTR`
- `cv`、`cvpr`、`CVPR2023`、`CVR`、`DGS`、`e.g`、`eccv`、`ECCV2022`
- `emnlp`、`ETL`、`G2O`、`GPT4`、`GroundingDINO`、`GTSAM`、`HPA`、`ICBU`
- `iccv`、`ICLR2020`、`ICSE`、`isaac`、`LI-SLAM`、`lightgbm`、`LTR`、`MPC`
- `mujoco`、`NeRF`、`nlp`、`o1`、`OMPL`、`OpenAI`、`OWL-ViT`、`PaLM`
- `peer-reviewed`、`Perception-to-Control`、`re-caption`、`ros`、`ROS2`、`S-curve`、`SDK`、`SE`
- `slam`、`UniAD`、`VI-SLAM`、`VIO`、`VL-BERT`、`xgboost`、`反欺诈`、`召回`
- `广告算法`、`异常检测`、`排序算法`、`推荐系统`、`搜索算法`、`知识图谱`、`粗排`、`精排`
- `自动驾驶`、`自然语言处理`、`计算机视觉`、`语音合成`、`语音识别`、`资源调度`、`轨迹规划`、`运动规划`
- `运筹优化`、`重排`、`风控`

## 阶段 13：论文复现、研究方法、产品落地与作品集

> 建议节奏：贯穿全程，集中整理 4-6 周。能力模块：7 个。

### 学习目标

把学习转化为第三方可验证的研究、工程和业务能力证据。

### 必学内容

> 能力闭环：原理研究与复现、数据构建与治理、算法建模与实现、训练调优与稳定性、评测、消融与误差分析、工程系统与工具链、部署、性能与运维、场景适配与产品闭环、安全、鲁棒与合规、原理理解与实际应用。

- **模块 13.1**：论文检索、问题定义、假设、相关工作、复现、基线、公平比较和消融。
- **模块 13.2**：ICLR、NeurIPS、ICML、ACL、EMNLP、CVPR、KDD、SIGIR 等方向性阅读。
- **模块 13.3**：实验设计、统计、负结果、错误分析、可复现性、代码质量和数据卡。
- **模块 13.4**：开源协作、Issue/PR、代码评审、许可证、模型卡、技术报告和演示。
- **模块 13.5**：系统设计、技术路线、跨团队接口、里程碑、风险、成本和迭代优先级。
- **模块 13.6**：Research-to-Production、用户反馈、在线实验、产品研究协同和技术影响力。
- **模块 13.7**：面试表达：问题、选择、实现、指标、失败、修复、贡献和边界。

### 实践任务

1. 完整复现一篇论文，补齐未报告细节并做至少三个消融。
2. 向真实开源项目提交可合并贡献，留下设计、测试和评审记录。
3. 把前面项目整合成一个旗舰项目，提供训练、评测、部署和演示。
4. 为每个项目写一页技术决策记录和一份失败复盘。

### 可交付成果

- 论文复现仓库
- 开源贡献
- 旗舰项目
- 技术报告与作品集

### 验收标准

- 第三方能按文档复现核心结果并理解你的贡献。
- 能解释为什么选择某技术、替代方案是什么、失败在哪里。
- 作品集同时证明研究深度、工程质量和实际效果。

### 技术关键词索引

- `aaai`、`acl`、`ACLNeUrIPS`、`ACM`、`ACM-ICPC`、`ACMICPC`、`across tasks/domains`、`Advantage Normalization`
- `Advantage Normalization（across tasks/domains）`、`Adversarial Robustness`、`AGI`、`AI`、`ASE`、`ASI`、`ASi8`、`ASPLOS`
- `ASRU`、`Asynchronous RL Training`、`Asynchronous RL Training / Isolated Environment Execution`、`AWM`、`Axolotl`、`Behavioral State Decay Mitigation`、`BPE`、`BU`
- `Byte-level`、`ByteDance RL Framework`、`CCF`、`CCF-A`、`CCS`、`ChatGLM`、`CIKM`、`Cold Start`
- `coling`、`Compute-Optimal Training`、`Context Window Extension`、`CORL`、`Cross-functional Collaboration`、`CT`、`DeepSeek`、`DL`
- `DoRA`、`DROID-SLAM`、`EDA`、`elasticsearch`、`Execution-Grounded Credit Assignment`、`Expert Specialization`、`Expert Specialization / Load Balancing`、`faiss`
- `Fast-LIVO`、`flink`、`FSE`、`Group Relative Advantage Estimation`、`grpc`、`GS-Cacl`、`H5`、`hadoop`
- `HIL`、`Human-AI Hybrid Feedback`、`Human-AI Hybrid Feedback / Real-user Feedback Loop`、`ICASSP`、`ICDE`、`iclr`、`icml`、`ICRA`
- `IDA`、`ijcai`、`IJCV`、`IJRR`、`iLAM`、`Instruction Tuning`、`INT4`、`INT8`
- `interspeech`、`IOI`、`iOS`、`IR`、`IROS`、`Isolated Environment Execution`、`JD`、`kdd`
- `KDDCup`、`Learning Rate Schedule`、`llm`、`Load Balancing`、`LQR`、`LSH`、`MFU`、`MICCAI`
- `milvus`、`Mixed Precision Training`、`MLSys`、`MM`、`MRI`、`Multi-turn Trajectory Optimization`、`naacl`、`nccl`
- `neurips`、`NeurLPS`、`NeurPS`、`Next-Token`、`Next-Token Prediction`、`NFV`、`NIPS`、`NOI`
- `NPC`、`NV`、`OCC`、`ocr`、`On-device`、`onnx`、`onnxruntime`、`OOD`
- `opencl`、`openvino`、`ORB-SLAM`、`ORM`、`P-Tuning`、`PDAF`、`PID`、`PM`
- `PMO`、`ppo`、`Prefix Tuning`、`PRM`、`Product-Research co-design`、`Prompt Tuning`、`Prompt Tuning v2`、`rag`
- `RAL`、`Real-user Feedback Loop`、`RECSYS`、`Red-teaming Automation`、`redis`、`RepE`、`Research-to-Production Pipeline`、`RiskPO`
- `RL Stage`、`RLVF`、`RM`、`RoboBrain`、`RRM`、`RSI`、`SAC`、`SAFe`
- `SDN`、`Self-Critique`、`SentencePiece`、`SIGGRAPH`、`sigir`、`SIGKDD`、`SIL`、`Slam-Former`
- `spark`、`SplatFusion`、`TASLP`、`TIP`、`TMM`、`Token-level MDP`、`Token-level MDP / Execution-Grounded Credit Assignment`、`TOP`
- `TorchTitan`、`TPAMI`、`TPM`、`TRL`、`TRO`、`USENIX`、`Verifiable Feedback`、`verl`
- `vibe coding`、`VINS`、`WSDM`、`www`、`上下文窗口`、`专利`、`专家混合`、`业务建模`
- `个性化`、`人文训练师团队`、`函数调用`、`可解释性`、`可验证奖励`、`后期制作`、`基准测试`、`工作流`
- `开源项目`、`强化学习`、`技术方案`、`持续学习`、`操作系统`、`检索增强`、`混合精度`、`混音`
- `用户行为`、`稀疏化`、`稀疏激活`、`算法竞赛`、`算法落地`、`终身学习`、`编译原理`、`论文`
- `负载均衡`、`长文本`、`长时序`、`音频处理`、`顶会`、`顶刊`、`高性能计算`
