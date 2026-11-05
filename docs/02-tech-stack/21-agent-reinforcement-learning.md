---
type: 教程
status: 已发布
level: 高阶
topic:
  - 模型训练
  - Agent
---

# Agent 强化学习：从 GRPO 到多轮工具 Agent

> 内容核对时间：2026-11。本文讲算法机制和工程实现，配套的动手实验使用 Datawhale 的 [agentic-rl-lab](https://github.com/datawhalechina/agentic-rl-lab)。面试表达见 [Agentic RL 面试指南](../04-interview/23-frontier-interview-guides/agentic-rl.md)，PPO / DPO / GRPO 的推导细节见 [Post-Training 完整指南](./25-post-training-complete-guide.md)。

## 本节目标

- 区分偏好对齐 RL、单轮 RLVR、多轮工具 Agent RL 和长程交互 Agent RL 各自要解决的问题。
- 用 advantage、采样时 logprob、当前 logprob 三个量读懂 IS、PPO、CISPO 等常见 loss。
- 掌握多轮 rollout 的工程要点：工具协议、observation mask、结果奖励、turn 级信用分配、环境成本。
- 理解 on-policy distillation 一类方法在 Agent 训练里的位置。
- 能按算力和目标选择训练基础设施，并独立完成至少一个可复现的 Agentic RL 实验。

## 核心概念

### 1. 四类问题设定

| 设定 | 一次决策是什么 | 奖励来源 | 主要难点 | 代表工作 |
|:---|:---|:---|:---|:---|
| 偏好对齐 RL（RLHF） | 一次完整回复 | 奖励模型 | reward hacking、奖励模型偏差 | InstructGPT |
| 单轮 RLVR | 一次完整推理 + 答案 | 规则校验器（答案匹配、单元测试） | 奖励稀疏、长度膨胀、熵坍塌 | DeepSeekMath（GRPO）、DeepSeek-R1 |
| 多轮工具 Agent RL | 多次“思考 + 工具调用” | 终局结果，必要时加格式项 | 工具返回不可训练、信用分配、环境开销 | Search-R1、ReTool |
| 长程交互 Agent RL | 几十到几百轮环境交互 | 环境终局信号，辅以中间估值 | 反馈周期长、状态恢复、上下文增长 | ALFWorld 上的 GRPO/PPO、TEMPO |

从上到下，变化的不只是轨迹长度。模型生成的 token 与环境产生的 token 交错出现，训练时要分清哪些 token 由策略负责；环境调用（搜索 API、代码沙箱、游戏引擎）成为 rollout 的主要开销；“一条轨迹一个奖励”越来越难告诉模型哪一步做对了。

### 2. 三个变量读懂 loss

大多数策略梯度 loss 都由三个量组合而成：

- `advantage`：这条轨迹（或这个 token）比同组平均好多少；
- `sampling_logprobs`：采样时的策略给这个 token 的对数概率；
- `target_logprobs`：当前正在更新的策略给这个 token 的对数概率。

两者之比 `ratio = exp(target - sampling)` 用来修正“数据由旧策略产生、梯度作用于新策略”的偏差。

| Loss | 目标 | 作用 |
|:---|:---|:---|
| Importance Sampling | `ratio · A` | 直接按 advantage 推高或压低 token 概率 |
| PPO | `min(ratio · A, clip(ratio) · A)` | 限制单步更新幅度；被裁剪的 token 梯度为 0 |
| CISPO | `sg(clip(ratio)) · log π · A` | 裁剪后的 ratio 只作固定权重，梯度仍经过所有 token 的 logprob |

CISPO 出自 [MiniMax-M1](https://arxiv.org/abs/2506.13585)，动机是 PPO 式裁剪会让少数关键 token（例如推理中的转折词）在早期更新后不再获得梯度。agentic-rl-lab 的 [GRPO 实验](https://github.com/datawhalechina/agentic-rl-lab/tree/main/experiments/grpo) 在同一份 GSM8K 代码里切换三种 loss，可以直接对照。

### 3. GRPO 与常见变体

GRPO（[DeepSeekMath](https://arxiv.org/abs/2402.03300)）对同一 prompt 采样 G 条回答，用组内奖励的相对高低作为 advantage，省掉了 PPO 的 value model。之后的变体大多在改三个地方：advantage 怎么归一、ratio 在什么粒度上计算和裁剪、哪些样本进入 batch。

| 方法 | 改动 | 解决的问题 | 需要注意 |
|:---|:---|:---|:---|
| [Dr. GRPO](https://arxiv.org/abs/2503.20783) | 去掉 std 归一化和按回答长度平均 | 消除对难度和长度的系统性偏置 | 奖励尺度变化后学习率要重调 |
| [DAPO](https://arxiv.org/abs/2503.14476) | Clip-Higher、Dynamic Sampling、token 级平均、超长软惩罚 | 熵坍塌、退化组、长回答权重过低、截断突变 | Dynamic Sampling 的补采在同步系统里会形成等待屏障 |
| [GSPO](https://arxiv.org/abs/2507.18071) | ratio 和裁剪提升到序列级 | 长序列逐 token ratio 方差大，MoE 训练不稳 | 序列级 ratio 对格式、长度变化很敏感 |

“退化组”值得单独理解：如果一组回答全对或全错，组内 advantage 全为 0，这组数据不产生梯度，却已经消耗了全部采样成本。题目太易或太难时，有效 batch 会悄悄变小。

### 4. 多轮轨迹里哪些 token 参与训练

一条 Search-R1 风格的轨迹由四类片段交错组成：

```text
[prompt] [assistant: 思考 + <search>query</search>] [tool: 检索结果] [assistant: 继续思考 + 答案]
```

检索结果要留在上下文里，否则模型看不到证据；但它们由环境产生，策略不对其负责，所以 loss mask 置 0。[Search-R1](https://arxiv.org/abs/2503.09516) 把这一做法称为检索 token 掩码，并报告它对训练稳定性有明显影响。代码执行（ReTool）、网页浏览、游戏 observation 同理。

实现时最容易出错的是“训练时拼出的 token 序列”与“采样时模型实际看到的序列”不一致。常见原因包括：

- chat template 在多轮渲染时改写了历史 assistant 内容（例如剥离或重排 reasoning 段）；
- 采样时增量拼接、训练时整体重新渲染，两次 tokenize 的边界不同；
- 工具返回被截断或格式化的方式在两侧不一样。

agentic-rl-lab 的 ReTool 实验记录了一个具体案例：Qwen3.5 的模板会重构含 `</think>` 的 assistant 消息，导致按真实文本定位 assistant 边界失败；解法是用占位内容计算 canonical 前缀，只把工具消息作为增量拼接。

## 深入理解

### 5. 奖励设计

Agent 任务的奖励通常是“结果 + 格式”，过程奖励只在有可靠判定器时才加。

**结果奖励**：答案精确匹配（问答）、单元测试（代码）、环境 `won` 信号（游戏、家务模拟）、规则校验（约束满足）。这类奖励难以作弊，但稀疏。

**格式奖励**：工具调用是否合法、最终答案是否在指定标签里。训练早期很有用，但很容易成为奖励上涨的主要来源。Tournament-GRPO 复现中，前 3 步到后 3 步总奖励从 0.485 涨到 0.849，约 92% 的增量来自格式项，真正衡量报告质量的淘汰赛奖励只从 0.293 涨到 0.323。看训练曲线时要把各奖励分量拆开记录。

**评测指标同样会被行为变化“带偏”**。几个实验里的例子：

- GSPO 复现中，格式合规率从 47.50% 升到 99.44%，平均回答长度缩短约 48%，但 AIME25 的 Average@12 在 step 20 之后持续下降。格式和长度的显著变化不等于推理能力提升。
- Tournament-GRPO 的评测里，RACE Overall 的大部分涨幅来自“交白卷”的题目变少；只看双方都交了报告的 59 题，Overall 是 34.49 → 35.25。
- Medical SFT 把 MedQA-zh 从 72.17% 提到 79.00%，同时 C-Eval 非医疗子集从 81.67% 掉到 69.33%。只看领域指标会漏掉通用能力的损失。

**LLM 裁判**：开放式任务（研究报告、写作）只能依赖裁判模型。两两比较通常比逐份打绝对分稳定，代价是比较次数随组大小平方增长；裁判本身的位置偏差、长度偏好需要通过交换顺序和多裁判检查。

### 6. 信用分配：从轨迹级到 turn 级

最简单的做法是把轨迹 advantage 广播给所有 assistant token（ALFWorld、Search-R1、ReTool 实验都是这样）。轨迹变长后，一个终局奖励很难区分关键步骤和无关步骤。目前有三类改进思路：

| 思路 | 做法 | 代价 | lab 中的实验 |
|:---|:---|:---|:---|
| 训练 critic | PPO value head 估计中间状态价值 | 多一个模型；scalar head 难以估计需要推理的状态 | — |
| 生成式 critic + macro-step | 每 H 轮为一段，段末由同一模型以 critic 角色推理后给出估值，用 TD 目标训练 | 需要环境支持保存/恢复中间状态 | [TEMPO](https://github.com/datawhalechina/agentic-rl-lab/tree/main/experiments/tempo) |
| 带特权信息的自教师 | 同一参数快照在“加了技能说明”的条件下复评每个 turn，用 token 概率差得到 turn 级证据，在轨迹 advantage 上做有界调整 | 每条轨迹多一次教师前向 | [AgentOPSD](https://github.com/datawhalechina/agentic-rl-lab/tree/main/experiments/agentopsd) |

AgentOPSD 的设计有一个值得注意的约束：教师只能把某个 turn 的权重上下调整至多 10%，不能改变 advantage 的符号。最终方向仍由环境结果决定，局部判断只负责分配强度，避免教师的错误判断推翻真实结果。

TEMPO 目前只有 Dots 团队的[技术博客](https://studio.dots.ai/dots/tempo-blog.html)，没有论文和超参数。lab 中的复现是算法级的：在 ALFWorld 上跑通 macro-step、生成式 critic、value warm-up 和 TD 训练，不追求原文数字。

### 7. On-policy Distillation 系列

On-policy distillation（OPD）让学生模型在自己生成的轨迹上接受教师的逐 token 打分，用反向 KL 把学生拉向教师。它结合了 [GKD](https://arxiv.org/abs/2306.13649)（在学生分布上蒸馏）和 [MiniLLM](https://arxiv.org/abs/2306.08543)（生成式模型用反向 KL）两条工作线。和 RL 相比，OPD 每个 token 都有监督信号，样本效率高；和 SFT 相比，训练数据来自学生自己的分布，缓解了训练与推理的分布偏移。

| 方法 | 教师 | 教师额外看到什么 | 适用场景 |
|:---|:---|:---|:---|
| OPD | 更大的模型或领域 SFT 模型 | 无 | 能力迁移、领域增强 |
| [OPSD](https://arxiv.org/abs/2601.18734) | 冻结的初始模型自身 | 题目的参考解答 | 没有更强教师时的自蒸馏 |
| AgentOPSD | 当前参数快照自身 | 任务相关技能说明 | 多轮 Agent 的 turn 级信用分配 |

Medical OPD 实验给出了一个实用流程：先用领域 SFT 得到医疗教师（会损失通用能力），再从原始模型出发做 OPD 吸收领域能力，最后用原始模型作为教师做一轮“锚点恢复”。这次实验里最终 checkpoint 的 MedQA-zh 为 84.33%，C-Eval 非医疗子集为 84.67%，两项都高于原始模型。作者明确说明 SAR-OPD、IDT-OPD 是为这组实验起的名字，不是已有论文中的标准方法。

### 8. 环境与成本

Agentic RL 的账单结构和单轮 RLVR 很不一样。ALFWorld 实验 80 步花费 ¥1613.71，其中 prefill 约 8.4 亿 token，训练约 6200 万 token，采样约 1400 万 token。每执行一次工具，下一轮采样都要重新读入越来越长的历史；轨迹走到二三十步时，早期上下文已经被重复计算了很多次。

设计实验时需要提前考虑：

- **上下文增长**：限制单轨迹总 token、单轮输出长度和最大交互步数；考虑对历史 observation 做截断或摘要（会引入训推不一致，要在采样和训练两侧同样处理）。
- **环境可靠性**：记录工具超时率、错误率、限流比例。Search-R1 实验专门报告了搜索成功率 100%、429 比例为 0，以排除“搜索失败导致分数变化”。ReTool 的沙箱超时率稳定在 2%～3%。
- **状态保存与恢复**：分支 rollout、macro-step 训练都需要从中间状态继续。环境不支持存档读档时，这类方法无法实现。
- **补采与长尾**：同步训练循环中，一个 step 的耗时由最慢的轨迹决定。DAPO 实验跑到第 35 步已用约 4.17 小时，主要原因是 Dynamic Sampling 补采无法与上一轮的长尾生成重叠。

### 9. 训练基础设施选型

| 类型 | 代表 | 适合 | 不适合 |
|:---|:---|:---|:---|
| 训练 API | [Tinker](https://tinker-docs.thinkingmachines.ai/)、[PyTRIO](https://pytrio.com/) | 个人和小团队复现算法；本地只写训练循环、数据和奖励，远端执行 forward/backward、optimizer step 和采样 | 全参数训练（Tinker 文档说明只支持 LoRA）、需要改动推理引擎或并行策略的系统研究 |
| 单机 / 中小规模库 | [TRL](https://github.com/huggingface/trl)、[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | 熟悉 HF 生态，单轮 RLVR，单节点实验 | 长程多轮 Agent、需要异步 rollout 的场景 |
| 大规模 RL 框架 | [verl](https://github.com/volcengine/verl)、[slime](https://github.com/THUDM/slime)、[AReaL](https://github.com/inclusionAI/AReaL)、[ROLL](https://github.com/alibaba/ROLL)、[SkyRL](https://github.com/NovaSky-AI/SkyRL)、[NeMo RL](https://github.com/NVIDIA-NeMo/RL) | 多机多卡、训推分离或共置、需要控制吞吐和权重同步 | 只想验证一个想法、没有多卡资源 |
| Agent 训练接入层 | [rLLM](https://github.com/rllm-org/rllm)、[Agent Lightning](https://github.com/microsoft/agent-lightning)、[verl-tool](https://github.com/TIGER-AI-Lab/verl-tool)、[ART](https://github.com/OpenPipe/ART)、[verifiers](https://github.com/PrimeIntellect-ai/verifiers) | 已有 Agent 代码或环境，需要接到 RL 训练后端 | 不需要多轮环境的任务 |

这些框架迭代很快，异步支持、多轮工具接口、支持的模型和后端以各仓库当前 README 为准。

训练 API 的价值在于把研究者的注意力集中在算法本身：rollout 怎么组织、奖励怎么算、advantage 和 loss 怎么写。代价是看不到也改不了系统层，例如权重同步方式、采样引擎的调度策略、训推重叠程度。做系统方向研究或大规模训练时仍然需要上面第三类框架。

## 代码示例

下面的脚本只依赖 numpy，演示三个最常用的计算单元：组相对 advantage（含 Dr. GRPO 变体和退化组）、多轮轨迹的 loss mask、三种 loss。采样和环境部分请看 agentic-rl-lab 中对应实验的 `rollout.py`。

```python
"""运行环境：Python 3.10+，numpy>=1.24。"""
import numpy as np


def group_relative_advantage(rewards, normalize_std=True, eps=1e-6):
    """同一 prompt 下 G 条轨迹的终局奖励 -> 每条轨迹的 advantage。
    组内奖励全相同时返回全 0：这一组不产生梯度。"""
    r = np.asarray(rewards, dtype=np.float64)
    centered = r - r.mean()
    if not normalize_std:  # Dr. GRPO 去掉了 std 归一化
        return centered
    return centered / (r.std() + eps)


def build_loss_mask(segments):
    """segments: [(role, n_tokens), ...]。只有 assistant token 计入 loss。"""
    mask = []
    for role, n_tokens in segments:
        mask.extend([1.0 if role == "assistant" else 0.0] * n_tokens)
    return np.asarray(mask)


def is_loss(target_lp, sampling_lp, adv, mask):
    ratio = np.exp(target_lp - sampling_lp)
    return -(ratio * adv * mask).sum() / mask.sum()


def ppo_loss(target_lp, sampling_lp, adv, mask, eps_low=0.2, eps_high=0.2):
    ratio = np.exp(target_lp - sampling_lp)
    clipped = np.clip(ratio, 1 - eps_low, 1 + eps_high)
    obj = np.minimum(ratio * adv, clipped * adv)
    return -(obj * mask).sum() / mask.sum()


def cispo_weights(target_lp, sampling_lp, eps_low=0.2, eps_high=0.2):
    """CISPO 中被 stop-gradient 的权重。实际 loss 为
    -(sg(weight) * target_lp * adv * mask).sum()，梯度只经过 target_lp。"""
    ratio = np.exp(target_lp - sampling_lp)
    return np.clip(ratio, 1 - eps_low, 1 + eps_high)


if __name__ == "__main__":
    rewards = [1.0, 0.0, 0.0, 1.0]
    print("GRPO advantage:   ", np.round(group_relative_advantage(rewards), 3))
    print("Dr.GRPO advantage:", group_relative_advantage(rewards, normalize_std=False))
    print("退化组 advantage: ", group_relative_advantage([1.0, 1.0, 1.0, 1.0]))

    # prompt -> 思考+查询 -> 检索结果 -> 作答
    segments = [("prompt", 6), ("assistant", 5), ("tool", 8), ("assistant", 4)]
    mask = build_loss_mask(segments)
    print("参与 loss 的 token 数:", int(mask.sum()), "/", mask.size)

    rng = np.random.default_rng(0)
    sampling_lp = rng.normal(-1.0, 0.3, mask.size)
    target_lp = sampling_lp + rng.normal(0.0, 0.3, mask.size)
    adv = np.full(mask.size, group_relative_advantage(rewards)[0])  # 轨迹级 advantage 广播

    print("IS   loss:", round(is_loss(target_lp, sampling_lp, adv, mask), 4))
    print("PPO  loss:", round(ppo_loss(target_lp, sampling_lp, adv, mask), 4))
    w = cispo_weights(target_lp, sampling_lp)
    print("CISPO 权重范围:", round(float(w[mask > 0].min()), 3), "~", round(float(w[mask > 0].max()), 3))
```

输出：

```text
GRPO advantage:    [ 1. -1. -1.  1.]
Dr.GRPO advantage: [ 0.5 -0.5 -0.5  0.5]
退化组 advantage:  [0. 0. 0. 0.]
参与 loss 的 token 数: 9 / 23
IS   loss: -1.07
PPO  loss: -1.0298
CISPO 权重范围: 0.8 ~ 1.2
```

## 动手实验：agentic-rl-lab

[agentic-rl-lab](https://github.com/datawhalechina/agentic-rl-lab)（Datawhale，Apache-2.0）用较小的代码量复现了 16 个 LLM / Agent RL 实验。每个实验目录包含算法讲解（`readme.md`）、运行说明（`start.md`）和完整代码，训练与采样通过 PyTRIO 训练 API 远程执行，实验记录用 SwanLab。默认基座是 `Qwen/Qwen3.5-4B` + LoRA rank 32，本地不需要 GPU。仓库另有[在线阅读版](https://datawhalechina.github.io/agentic-rl-lab/)。

### 建议学习顺序

| 阶段 | 实验 | 重点看什么 |
|:---|:---|:---|
| 1. 基础 | Loss Functions → GRPO → Vision GRPO | 三种 loss 的差异；rollout → advantage → Datum → `forward_backward` → `optim_step` 的完整链路；图片输入如何进入训练 |
| 2. 算法变体 | DAPO、GSPO | Dynamic Sampling 的真实时间成本；序列级 ratio 下格式与长度的快速变化 |
| 3. 蒸馏 | General OPD → Medical OPD → OPSD | 反向 KL 逐 token advantage 的构造；领域增强与通用能力保持；同一模型做教师 |
| 4. 多轮工具 Agent | Search-R1 → ReTool | 工具声明与 chat template；分支 rollout；observation mask；只看最终答案的奖励 |
| 5. 长程环境与信用分配 | ALFWorld → TEMPO / AgentOPSD | 长轨迹并发、环境成本；macro-step 与生成式 critic；turn 级信用 |
| 6. 综合应用 | Spec-o3、Tournament-GRPO | 冷启动 SFT + GRPO 的组合；图文交错多轮工具；开放式任务的两两比较奖励 |

PyTrio-Jev 是一个决策（分类）模型的训练实验，用一次前向、在候选符号上做受限 softmax 输出决策和置信度，侧重 SFT 与概率校准，可以作为补充阅读。

### 实验结果怎么读

下表摘自各实验文档。规模都很小（单次训练、几十到几百道评测题、没有多随机种子），数字用来确认“算法机制跑通、方向正确”，不能当作 benchmark 结论；不同实验的评测协议也不同，不能横向比较。

| 实验 | 评测集 | Base → 训练后 | 这组数字说明了什么 |
|:---|:---|:---|:---|
| Vision GRPO（100 步） | GeoQA test 100 题 | 准确率 71.0% → 87.0% | 格式正确时的条件准确率几乎不变（94.7% → 95.6%），提升主要来自格式合规的回答变多 |
| ReTool（200 步） | AIME 30 题 | Average@12 23.61% → 47.50%；格式 25.83% → 76.11% | 跳过冷启动 SFT 直接 RL 可行的前提是 base 已经会调用工具（smoke test 工具调用率 87.5%） |
| Search-R1（30 步） | 70 道开发题 | EM 40.00% → 42.86% | 净增 2 题，同时平均搜索次数从 1.76 增加到 2.77；需要更多题目和重复实验确认 |
| ALFWorld（80 步） | 274 局 | 成功率 52.92% → 56.93%；unseen 52.99% → 58.96% | 成本主要花在 prefill 上 |
| AgentOPSD（80 步） | 同上 274 局，评测协议不同 | 成功率 58.8% → 65.7% | unseen 提升（+9.7 pp）大于 seen；平均无效动作 7.07 → 5.68 |
| Spec-o3 | 256 条开发集 | 4B：99 → 138（SFT）→ 175（RL）题；9B RL：191 题 | 冷启动 SFT 主要解决格式（严格格式合规 0.39% → 85.94%），RL 继续提升判断质量 |
| OPSD（100 步） | AIME25 | Average@12 51.67% → 52.78%；Pass@12 80.00% → 86.67% | Average@12 的提升在两次 Base 评测的波动范围内 |

几条通用的读法：

1. 先看格式合规率和回答长度有没有大幅变化，再看准确率。很多“提升”来自格式。
2. 看 Base 重复评测的波动。GSPO 实验里同一个 Base 两次评测的 Pass@12 分别是 56.67% 和 66.67%。
3. 区分 seen / unseen 或领域内 / 领域外，确认没有用一种能力换另一种能力。
4. 把成本当作结果的一部分记录：token 构成、环境调用次数、每步耗时。

### 运行前准备

```bash
git clone https://github.com/datawhalechina/agentic-rl-lab.git
cd agentic-rl-lab
uv sync                    # Python >= 3.13
uv sync --extra alfworld   # 仅 ALFWorld / TEMPO / AgentOPSD 需要
```

还需要 PyTRIO 账号（按 token 计费）和 SwanLab 账号；Search-R1、Tournament-GRPO 需要配置搜索 API。各实验文档都给出了实际花费，例如 OPSD 正式训练 41.81 元，ALFWorld 1613.71 元，跑之前先估算预算。建议每个实验先按 `start.md` 跑 1 步或几步的 smoke test，确认 loss mask、奖励解析和 checkpoint 保存都正常，再跑正式配置。

### 做成项目时可以改什么

照着跑通一遍只能说明会用。要形成自己的项目证据，建议在一个 Agentic 实验上只改一个变量，并做对照：

- **换环境或工具**：把 Search-R1 的搜索后端换成本地 BM25 / 向量检索，比较奖励曲线和搜索行为；
- **换奖励**：在 ReTool 上把纯结果奖励改成“结果 + 工具执行成功”，观察是否出现为拿奖励而调用工具的行为；
- **换信用分配**：在 ALFWorld 上对比轨迹级 advantage 与 AgentOPSD 的 turn 级调整；
- **补实验严谨性**：至少 2～3 个随机种子，报告 Base 重复评测的波动、格式与准确率分开报告、记录成本。

## 面试中如何考

1. **为什么工具返回的 token 不参与 loss，但要留在上下文里？** 留在上下文是因为后续决策依赖它；不参与 loss 是因为它由环境产生，对它做梯度相当于要求模型“预测检索结果”，会把检索内容的分布错误地当作策略目标，还会放大噪声。
2. **GRPO 的退化组是什么，怎么处理？** 组内奖励全同则 advantage 为 0。可以过滤后补采（DAPO），也可以通过课程或难度采样减少其比例；要同时报告补采的额外 token 与时间成本。
3. **PPO、CISPO 在裁剪上的区别？** PPO 裁剪目标函数，ratio 越界的 token 梯度为 0；CISPO 裁剪后把 ratio 当常数权重，所有 token 的 logprob 都还有梯度。
4. **多轮 Agent RL 中，训推不一致可能出在哪？** chat template 改写历史消息、增量拼接与整体渲染的 tokenize 边界不同、observation 截断方式不同、采样与训练使用不同权重版本。
5. **奖励涨了但评测没涨，你会先查什么？** 拆奖励分量（格式项是否主导）；看回答长度与格式合规率变化；看 Base 重复评测的方差；检查评测集是否混入训练分布。
6. **长程任务的信用分配有哪些方案？** value head、生成式 critic + macro-step（TEMPO）、带特权信息的自教师（AgentOPSD）、过程奖励模型、分支 rollout 做反事实对照。说清每种方案依赖什么（环境能否存档、是否有可靠的过程判定）。
7. **OPD 和 RL 的关系？** OPD 用教师逐 token 的反向 KL 作为 advantage，信号密集但上限受教师限制；RL 用环境结果，信号稀疏但能超过教师。两者可以组合：先 OPD 快速吸收能力，再 RL 提升到教师之上，或用 OPD 做能力保持。
8. **你会怎么选训练框架？** 先看目标：复现算法、验证想法用训练 API；单节点单轮 RLVR 用 TRL / OpenRLHF；多机长程 Agent 用 verl、slime、AReaL、SkyRL 等；已有 Agent 代码要接训练时看 rLLM、Agent Lightning 一类接入层。

更系统的面试题和白板题见 [Agentic RL 面试指南](../04-interview/23-frontier-interview-guides/agentic-rl.md)。

## 扩展阅读

**算法**

- PPO：[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- GRPO：[DeepSeekMath](https://arxiv.org/abs/2402.03300)
- Dr. GRPO：[Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783)
- DAPO：[An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)
- GSPO：[Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071)
- CISPO：[MiniMax-M1](https://arxiv.org/abs/2506.13585)
- GKD：[On-Policy Distillation of Language Models](https://arxiv.org/abs/2306.13649)
- MiniLLM：[Knowledge Distillation of Large Language Models](https://arxiv.org/abs/2306.08543)
- OPSD：[Self-Distilled Reasoner](https://arxiv.org/abs/2601.18734)

**Agent 训练**

- Search-R1：[Training LLMs to Reason and Leverage Search Engines with RL](https://arxiv.org/abs/2503.09516)
- ReTool：[Reinforcement Learning for Strategic Tool Use in LLMs](https://arxiv.org/abs/2504.11536)
- RAGEN：[Understanding Self-Evolution in LLM Agents via Multi-Turn RL](https://arxiv.org/abs/2504.20073)
- AgentOPSD：[Fine-Grained Agentic RL via On-Policy Self-Distillation](https://arxiv.org/abs/2608.05987)
- TEMPO：[Dots 团队技术博客](https://studio.dots.ai/dots/tempo-blog.html)
- Tournament-GRPO：[Group-Wise Tournament Rewards for RL in Open-Ended Long-Form Generation](https://arxiv.org/abs/2605.26958)
- Spec-o3：[A Tool-Augmented Vision-Language Agent for Rare Celestial Object Candidate Vetting](https://arxiv.org/abs/2601.06498)

**环境**

- ALFWorld：[论文](https://arxiv.org/abs/2010.03768) / [代码](https://github.com/alfworld/alfworld)

**本仓库相关**

- [Agent RL 开源项目汇总](../../resources/agent/papers/agent_rl/Agent%E2%9E%95RL%E5%BC%80%E6%BA%90%E9%A1%B9%E7%9B%AE%E6%B1%87%E6%80%BB.md)
- [Post-Training 完整指南](./25-post-training-complete-guide.md)
- [Agent Evaluation Harness](./26-agent-evaluation-harness-guide.md)
- [Agent 沙箱指南](./24-agent-sandbox-guide.md)
