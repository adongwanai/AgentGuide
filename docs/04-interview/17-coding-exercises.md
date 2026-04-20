# 第一阶段：神经网络基础算子

## 目标

掌握深度学习最基础的算子实现，建立「手写 = 真懂」的学习习惯。这一关全部是 🟢 简单题，但是大模型所有高级组件的基石。

## 刷题清单

### 激活函数（必刷）

- Implement ReLU· 简单 🟢 简单 — 最基础，理解梯度截断
- Sigmoid 激活函数· 简单 🟢 简单 — 二分类基础，注意数值稳定性
- GELU Activation· 简单 🟢 简单 — BERT/GPT 标配，理解 erf 近似
- Tanh 激活函数· 简单 🟢 简单 — RNN 时代经典
- LeakyReLU 激活函数· 简单 🟢 简单 — 解决 dying ReLU 问题

### 归一化（必刷）

- Implement LayerNorm· 中等 🟡 中等 — Transformer 核心组件，必须手写
- Implement RMSNorm· 中等 🟡 中等 — LLaMA 系列标配，比 LayerNorm 更轻量
- Implement BatchNorm· 中等 🟡 中等 — 理解 running stats 的训练/推理差异

### 基础层

- Implement Softmax· 简单 🟢 简单 — 数值稳定 trick（减 max）是考点
- Embedding Layer· 简单 🟢 简单 — 词表映射，理解 weight tying
- Kaiming Initialization· 简单 🟢 简单 — 正确初始化是训练稳定的第一步
- Implement Dropout· 简单 🟢 简单 — 训练/推理行为不同，逐 token 实现
- Simple Linear Layer· 中等 🟡 中等 — 手实现 nn.Linear，理解矩阵乘法

## 刷题重点

- 每道题都要先**不看代码**自己推导公式，再实现
- 重点关注**数值稳定性**（Softmax 减 max、LayerNorm 加 eps）
- 理解**训练模式 vs 推理模式**的区别（BatchNorm、Dropout）

# 第二阶段：Attention 机制核心

## 目标

Attention 是大模型的心脏。从最基础的缩放点积注意力开始，逐步实现各种变体，理解每次改进解决了什么问题。

## 刷题清单

### 基础（必刷，按顺序）

- 因果注意力掩码· 简单 🟢 简单 — 理解为什么 GPT 需要 mask
- Softmax Attention· 简单 🟢 简单 — 所有注意力的基础，QKV 计算
- Causal Self-Attention· 中等 🟡 中等 — 在 Softmax Attention 上加 mask
- Multi-Head Attention· 困难 🔴 困难 — 多头拆分与合并，完整实现

### 位置编码（必刷）

- Sinusoidal Position Encoding· 简单 🟢 简单 — 原始 Transformer 位置编码
- Rotary Position Embedding (RoPE)· 中等 🟡 中等 — LLaMA/Qwen 标配，旋转矩阵实现

### 注意力变体（进阶）

- Multi-Head Cross-Attention· 中等 🟡 中等 — Encoder-Decoder 架构基础
- Grouped Query Attention· 困难 🔴 困难 — LLaMA-2/3 优化，减少 KV 参数量
- ALiBi Attention· 中等 🟡 中等 — 位置偏置的另一种思路
- Sliding Window Attention· 中等 🟡 中等 — Mistral 局部注意力，降低复杂度
- Flash Attention (Tiled)· 困难 🔴 困难 — IO 感知的分块计算，必须理解

## 刷题重点

- **causal-mask → attention → causal-attention → mha**这条线必须一步一步手写
- Flash Attention 重点理解**分块(tiling)**思想，不要死记公式
- GQA 的核心：KV head 数 < Q head 数，理解 `repeat_kv` 操作

# 第三阶段：Transformer 完整模块

## 目标

掌握 Transformer 各关键模块，能从零拼装出一个完整 Block，理解各组件之间的依赖关系。

## 刷题清单

### FFN 变体（必刷）

- GLU 门控线性单元· 简单 🟢 简单 — 门控机制基础，sigmoid 作门
- SwiGLU Activation· 中等 🟡 中等 — LLaMA FFN 标配，SiLU 门控
- SwiGLU MLP· 中等 🟡 中等 — 完整 FFN 实现，gate/up/down 三矩阵

### 完整 Block（必刷）

- Adaptive LayerNorm Zero (adaLN-Zero)· 中等 🟡 中等 — DiT 条件归一化，扩散模型必考
- GPT-2 Transformer Block· 困难 🔴 困难 — 完整 GPT Block，理解残差连接顺序

### 编码器-解码器

- Encoder-Decoder 交叉注意力· 中等 🟡 中等 — Seq2Seq 架构，Q 来自 decoder，KV 来自 encoder

### 视觉 Transformer

- ViT Patch Embedding· 中等 🟡 中等 — 图像分块到 token，多模态基础
- ViT Transformer Block· 困难 🔴 困难 — 视觉 Transformer 完整 Block

## 刷题重点

- GPT-2 Block 的残差连接顺序：**Pre-LN vs Post-LN**的区别是关键考点
- SwiGLU 理解为什么用三个投影矩阵（gate、up、down），而不是两个

# 第四阶段：损失函数与评估指标

## 目标

大模型训练和评估的"度量衡"。从基础交叉熵到对比学习损失，理解每个损失函数的设计动机。

## 刷题清单

### 基础损失（必刷）

- Cross-Entropy Loss· 简单 🟢 简单 — 语言模型预训练的核心损失
- 困惑度（Perplexity）· 简单 🟢 简单 — LM 标准评估指标，PPL = exp(CE)
- KL 散度· 中等 🟡 中等 — 知识蒸馏、VAE、RLHF 中都用到
- Label Smoothing Loss· 简单 🟢 简单 — 防过拟合的正则化技巧

### 对比学习损失

- Contrastive Loss (InfoNCE)· 中等 🟡 中等 — 对比学习基础，CLIP 前身
- Triplet Loss· 中等 🟡 中等 — 三元组损失，嵌入模型训练

### 评估与生成

- Token 准确率· 简单 🟢 简单 — 序列预测准确率，注意 ignore_index
- BLEU 评分· 中等 🟡 中等 — 机器翻译标准评估，n-gram 精度
- 知识蒸馏损失· 中等 🟡 中等 — 软标签蒸馏，KL + CE 加权

### MoE 专项

- MoE Load Balancing Loss· 中等 🟡 中等 — 防止 expert collapse 的辅助损失
- Focal Loss· 中等 🟡 中等 — 解决类别不平衡，检测模型常用
- Multi-Token Prediction Loss· 中等 🟡 中等 — Meta LLaMA-3 的 MTP 预训练目标

## 刷题重点

- KL 散度的**非对称性**：$KL(P\|Q) \neq KL(Q\|P)$，蒸馏时 student 逼近 teacher
- InfoNCE 理解为什么分母要遍历所有负样本（in-batch negative）

# 第五阶段：训练优化技术

## 目标

工业级大模型训练的核心工程技术：优化器、学习率调度、梯度处理、内存优化。

## 刷题清单

### 优化器（必刷）

- Adam Optimizer· 中等 🟡 中等 — 现代 LLM 训练标配，一阶+二阶矩
- AdamW 优化器· 中等 🟡 中等 — 解耦权重衰减，比 Adam+L2 更正确

### 学习率调度

- 线性学习率预热· 简单 🟢 简单 — 训练初期防梯度爆炸
- Cosine LR Scheduler with Warmup· 中等 🟡 中等 — LLM 训练标配调度策略
- EMA 指数移动平均· 简单 🟢 简单 — 模型权重平滑，推理性能更稳定

### 梯度处理

- Gradient Norm Clipping· 简单 🟢 简单 — 防梯度爆炸，稳定训练必备
- Gradient Accumulation· 简单 🟢 简单 — 模拟大 batch size，显存受限时必用

### 内存优化（进阶）

- Activation Checkpointing· 中等 🟡 中等 — 用计算换内存，训练大模型必备
- Mixed Precision Training Step· 中等 🟡 中等 — FP16 前向 + FP32 权重，loss scaling

## 刷题重点

- **Adam vs AdamW**：权重衰减应该加在梯度上（L2）还是直接加在参数上（AdamW）
- Gradient Clipping：clip by norm vs clip by value 的区别
- 混合精度：理解 loss scaling 为什么能防止 FP16 梯度下溢

# 第六阶段：参数高效微调

## 目标

用不到 1% 的参数量达到全量微调的效果。LoRA 系列是大模型微调领域最重要的工程技术，面试必考。

## 刷题清单

- LoRA (Low-Rank Adaptation)· 中等 🟡 中等 — 低秩分解微调，最重要的 PEFT 方法
- QLoRA· 困难 🔴 困难 — 量化 + LoRA，4-bit 显存运行大模型
- Prefix Tuning 前缀微调· 中等 🟡 中等 — 软提示词，冻结主干只训 prefix
- INT8 Quantized Linear· 困难 🔴 困难 — 量化基础，理解 scale/zero-point

## 核心知识点

### LoRA 原理

$$W' = W_0 + \Delta W = W_0 + BA$$

- $W_0$：冻结的预训练权重（$d \times k$）
- $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，秩 $r \ll \min(d, k)$
- 只训练 $A$ 和 $B$，参数量从 $O(dk)$ 降到 $O(r(d+k))$
- 推理时可合并：$W' = W_0 + BA$，**无额外延迟**

### QLoRA 额外知识

- NF4（Normal Float 4-bit）量化：比均匀量化更好地保留信息
- Double Quantization：对量化常数再次量化
- Paged Optimizer：GPU OOM 时分页到 CPU

## 刷题重点

- LoRA 实现中，$B$ 初始化为**全零**，$A$ 正态初始化，保证训练开始时 $\Delta W = 0$
- 量化中 scale 和 zero_point 的计算公式要能手写

# 第七阶段：RLHF与对齐

## 目标

让 LLM 与人类偏好对齐的核心算法。从 PPO 到 DPO 到最新的 GRPO，理解每种方法的优劣权衡。

## 刷题清单

### 奖励建模

- Bradley-Terry Reward Model Loss· 中等 🟡 中等 — 偏好对比损失，RLHF 奖励模型训练
- KL 惩罚（RLHF）· 简单 🟢 简单 — 防止策略偏离参考模型

### 策略优化

- PPO Clipped Loss· 中等 🟡 中等 — ratio clip 目标，RLHF 阶段经典算法
- DPO Loss· 中等 🟡 中等 — 无需奖励模型的偏好对齐，简洁优雅
- GRPO Loss· 困难 🔴 困难 — DeepSeek-R1 使用，组相对策略优化

## 技术路线对比

| 方法 | 奖励模型 | 参考模型 | 适用场景                    |
| ---- | -------- | -------- | --------------------------- |
| PPO  | ✅ 需要   | ✅ 需要   | 复杂任务，效果最好          |
| DPO  | ❌        | ✅ 需要   | 简单对话对齐，工程简单      |
| GRPO | ❌        | ✅ 需要   | 推理任务，DeepSeek 验证有效 |

## 核心公式

### DPO 损失

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)\right]$$

### PPO Clip 目标

$$\mathcal{L}_{\text{PPO}} = -\mathbb{E}\left[\min\!\left(r_t A_t,\ \operatorname{clip}(r_t,\, 1-\varepsilon,\, 1+\varepsilon)\, A_t\right)\right]$$

## 刷题重点

- PPO Clip：理解 clip 的意义——限制更新步长，防止策略崩溃
- GRPO：不需要 value function，用组内相对奖励替代 advantage

# 第八阶段：现代LLM架构创新

## 目标

精读 LLaMA/Qwen/DeepSeek 系列引入的架构创新，这些是近 3 年最重要的技术突破。

## 刷题清单

### 位置编码进阶

- NTK-aware RoPE Scaling· 简单 🟢 简单 — 长上下文外推，YaRN/LongRoPE 基础
- 2D 旋转位置编码· 困难 🔴 困难 — ViT/多模态场景的 2D RoPE

### 注意力架构优化

- Differential Attention· 困难 🔴 困难 — Microsoft 提出，双头相减抵消噪声
- Multi-Head Latent Attention (MLA)· 困难 🔴 困难 — DeepSeek-V2 核心，KV Cache 压缩
- Paged Attention· 困难 🔴 困难 — vLLM 核心，分页管理 KV Cache

### MoE 架构

- Mixture of Experts (MoE)· 困难 🔴 困难 — Qwen-3/DeepSeek-V3 基础架构

### 序列模型

- Mamba SSM Step· 困难 🔴 困难 — 线性复杂度序列模型，替代 Attention 的挑战者

## 架构演进时间线

```
GPT-2 Block（标准 Transformer）
    ↓
RoPE（LLaMA-1）+ GQA（LLaMA-2）
    ↓
MLA（DeepSeek-V2，KV Cache 大幅压缩）
    ↓
MoE + FP8 训练（DeepSeek-V3）
    ↓
思考链 + GRPO（DeepSeek-R1）
```

## 刷题重点

- **MLA 核心**：将 KV 压缩到低维潜空间 $c_{KV}$，推理时动态解压，显存减少 5-13×
- **Paged Attention**：类比操作系统的虚拟内存，解决 KV Cache 碎片化
- **MoE 路由**：Top-K 路由 + 负载均衡 Loss，防止 expert collapse

# 第九阶段：采样与推理加速

## 目标

掌握 LLM 推理阶段的核心技术：从解码策略到量化加速，这是工程面试的高频考点。

## 刷题清单

### 解码策略（必刷）

- Temperature Scaling· 简单 🟢 简单 — 控制生成多样性，temperature=0 退化为贪婪
- Top-k / Top-p Sampling· 中等 🟡 中等 — 截断词表的采样策略
- Beam Search Decoding· 中等 🟡 中等 — 保留 k 个最优序列，翻译任务常用

### 推理加速（进阶）

- KV Cache Attention· 困难 🔴 困难 — 自回归推理必备，理解 cache 的 append 逻辑
- Speculative Decoding· 困难 🔴 困难 — 小模型草稿 + 大模型验证，加速推理
- INT8 Quantized Linear· 困难 🔴 困难 — 量化推理，显存减半

### RAG 相关

- 向量检索打分（RAG）· 简单 🟢 简单 — 余弦相似度 Top-K 检索，RAG 基础

### 分词

- Byte-Pair Encoding (BPE)· 困难 🔴 困难 — GPT/LLaMA tokenizer 算法，自底向上合并

## 刷题重点

- **KV Cache**：理解为什么推理时可以复用，训练时不能（因果 mask 让每步 K,V 不变）
- **Speculative Decoding**：接受条件 $u < p(x)/q(x)$，用拒绝采样保证分布正确性
- **Temperature**：temperature $\to 0$ 变贪婪，temperature $\to \infty$ 变均匀分布

# 第十阶段：多模态关键算法

## 目标

掌握视觉-语言模型的核心对齐技术，从 CLIP 对比学习到扩散模型，理解多模态 LLM 的技术基础。

## 刷题清单

### 图文对齐（必刷）

- CLIP 对比学习损失· 中等 🟡 中等 — 多模态对齐的奠基之作，对称 CE 损失
- 图文匹配打分· 简单 🟢 简单 — cosine similarity + sigmoid，最简单的对齐
- 跨模态注意力· 中等 🟡 中等 — 视觉 token 作 K/V，文本 token 作 Q

### 视觉编码

- ViT Patch Embedding· 中等 🟡 中等 — 图像分块为 token，多模态输入基础
- Swin Patch Merging· 简单 🟢 简单 — 层级视觉特征，Swin Transformer 关键操作

### 扩散模型

- Diffusion Noise Schedules· 简单 🟢 简单 — DDPM 噪声调度，linear/cosine schedule
- DDIM Sampling Step· 中等 🟡 中等 — 确定性采样，加速扩散模型推理
- Flow Matching Loss· 简单 🟢 简单 — 比 DDPM 更简单的生成框架，Flux 使用

## 核心算法：CLIP

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\left[\log\frac{e^{s_{ii}/\tau}}{\sum_j e^{s_{ij}/\tau}} + \log\frac{e^{s_{ii}/\tau}}{\sum_j e^{s_{ji}/\tau}}\right]$$

对角线元素 $s_{ii}$ 是图文配对的正样本，其余是负样本。$\tau$ 是可学习温度参数。

## 刷题重点

- CLIP Loss 是**双向对称**的：图到文 + 文到图，两个方向都要算
- ViT Patch Embedding：`patch_size=16` 时 224×224 图片变成 196 个 token
- Flow Matching：条件向量场 $v_t = x_1 - x_0$，比 score matching 更简单直接

# 进阶阶段：分布式与系统优化

## 目标

面向 LLM Infra 方向的进阶题目，考察系统级理解与工程能力。面试大厂 LLM 基础设施岗必备。

## 刷题清单

- Ring Attention· 困难 🔴 困难 — 序列并行，多设备分布式注意力计算
- Tensor Parallel MLP· 困难 🔴 困难 — 列并行 + 行并行，Megatron-LM 核心
- FSDP Training Step· 困难 🔴 困难 — 全参数分片训练，PyTorch FSDP 原理
- Linear Self-Attention· 困难 🔴 困难 — O(N) 复杂度注意力，核函数近似
- MCTS for Reasoning 🔴 困难 — 蒙特卡洛树搜索，o1/R1 推理扩展基础

## 分布式训练体系

```
数据并行（DDP/FSDP）
    + 张量并行（Tensor Parallel）
    + 流水线并行（Pipeline Parallel）
    = 3D 并行（Megatron-LM/DeepSpeed）
```

| 方法            | 分片维度         | 适用场景               |
| --------------- | ---------------- | ---------------------- |
| DDP             | 无（全副本）     | 小模型，多机多卡       |
| FSDP            | 参数+梯度+优化器 | 单机 OOM，大模型       |
| Tensor Parallel | 权重矩阵         | 超大模型，layer 内并行 |
| Ring Attention  | 序列长度         | 超长上下文             |

## 刷题重点

- Ring Attention：每个设备持有部分 Q，轮流 all-gather K/V，通信计算重叠
- Tensor Parallel MLP：列并行切 W1/Wgate，行并行切 W2，两次 all-reduce
- FSDP：前向时 all-gather，后向时 reduce-scatter，参数不在单卡完整存在
