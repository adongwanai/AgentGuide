# AI Infra 面试指南

> AI Infra 面试不是 GPU 名词问答。高质量回答要把硬件限制、算子行为、并行策略、调度、服务指标和成本连成一条因果链。

---

## 📌 本节目标

- 建立从芯片、算子到训练 / 推理平台的完整分层。
- 能用带宽、算力、通信、显存和排队解释性能瓶颈。
- 能为训练并行、推理服务和 Agent Runtime 做选型。
- 用 Profiling 与实验回答“瓶颈在哪里”，而不是凭经验猜。

## 💡 核心概念

### 六层技术地图

| 层级 | 关键对象 | 高频问题 |
|:---|:---|:---|
| 硬件 | GPU / 加速器、HBM、互联、网络 | 算力、带宽、拓扑、精度 |
| Kernel | CUDA、Triton、融合算子、Attention | occupancy、访存、同步、数值稳定 |
| Runtime / Compiler | 计算图、编译、内存规划、执行引擎 | graph break、动态 shape、调度 |
| Distributed | 数据 / 张量 / 流水线 / 序列 / 专家并行 | 通信量、切分边界、负载均衡 |
| Serving | batching、KV cache、prefill / decode、路由 | TTFT、TPOT、吞吐、尾延迟 |
| Platform | 调度、容错、可观测性、多租户、成本 | 利用率、SLO、配额、故障域 |

Agent Infra 还要再加一层有状态 Runtime：会话状态、工具执行、沙箱、权限、长任务恢复和 trace / replay。

### 性能诊断的统一框架

先问瓶颈属于哪一类：

```text
计算受限：算术单元忙，增加带宽收益小
带宽受限：大量数据搬运，算术强度低
通信受限：跨卡同步或 all-to-all 占主导
容量受限：模型、激活或 KV cache 放不下
调度受限：排队、气泡、长尾和碎片降低利用率
外部等待：网络、存储、工具或环境响应慢
```

然后用 profiler、时间线、硬件计数器和受控实验验证。只报 GPU utilization 往往不够：它可能掩盖低效 kernel、频繁小算子或等待间隙。

## 🔍 深入理解

### 1. 训练并行怎样选

| 策略 | 切什么 | 主要收益 | 主要代价 |
|:---|:---|:---|:---|
| Data Parallel | batch | 简单、扩展直接 | 参数 / 梯度同步，单卡仍需放下模型 |
| ZeRO / FSDP | 参数、梯度、优化器状态 | 显著降单卡状态占用 | 通信与参数聚合开销 |
| Tensor Parallel | 单层张量 | 支撑单层超大模型 | 频繁卡间通信、拓扑敏感 |
| Pipeline Parallel | 层 | 跨节点扩展 | pipeline bubble、调度复杂 |
| Sequence / Context Parallel | 序列维 | 支撑长上下文 | Attention 通信与边界处理 |
| Expert Parallel | MoE 专家 | 扩大参数量 | all-to-all、路由不均和热点 |

选型不能离开模型结构、序列长度、batch、集群拓扑和故障率。面试时最好给一个具体配置，估算显存和通信，再说明组合策略。

### 2. 推理为什么要拆 prefill 和 decode

- **Prefill**一次处理较长输入，矩阵乘规模大，更偏计算密集。
- **Decode**逐 token 生成，反复读取权重和 KV cache，更容易带宽与调度受限。

常见优化包括 continuous batching、paged KV cache、prefix cache、量化、speculative decoding 和 prefill / decode 分离。每种优化都应回答适用分布：长 prompt 还是短 prompt、吞吐优先还是交互延迟优先、缓存命中如何、质量是否受影响。

核心指标：

- TTFT：从请求进入到首 token。
- TPOT / ITL：后续 token 间延迟。
- E2E latency：完整请求耗时。
- Tokens/s：单请求或集群吞吐。
- P95 / P99：尾延迟，通常比均值更接近用户体验。
- Goodput：满足 SLO 的有效吞吐。
- Cost per successful task：对 Agent 比单纯 token 成本更有意义。

### 3. 容错与可观测性

训练关注 checkpoint、节点失效、网络抖动和长任务恢复；在线服务关注过载保护、熔断、降级、滚动发布和多模型路由；Agent Runtime 还要处理工具副作用、幂等、会话恢复和人工审批。

一条可用的 trace 至少能串起：请求、模型版本、prompt / cache 命中、每次工具调用、重试、token 与成本、错误码、最终业务结果。

## 🎯 面试中如何考

### 高频必答

1. **怎样判断一个算子是 compute-bound 还是 memory-bound？**
   - 回答轴：算术强度、硬件算力 / 带宽上限、profiler 和对输入规模的敏感性。

2. **为什么算子融合能加速？什么时候反而变慢？**
   - 回答轴：减少 launch 与中间访存；但寄存器压力、并行度和编译复杂度可能上升。

3. **DDP、FSDP / ZeRO、TP、PP 如何选？**
   - 回答轴：模型是否单卡可放、通信频率、拓扑、序列与 batch、故障域。

4. **AllReduce 与 All-to-All 分别在哪些场景成为瓶颈？**
   - 回答轴：梯度同步 / 张量并行 vs MoE 路由，数据量、拓扑和负载均衡。

5. **训练显存由哪些部分组成？**
   - 回答轴：参数、梯度、优化器状态、激活、临时 buffer、通信与碎片。

6. **为什么 decode 阶段常比 prefill 更难跑满 GPU？**
   - 回答轴：小批次逐 token、权重 / KV 搬运、动态请求与尾部调度。

7. **Continuous batching 解决什么，又引入什么？**
   - 回答轴：动态合批提升利用率；调度公平、内存管理和尾延迟更复杂。

8. **量化怎样影响吞吐、显存和质量？**
   - 回答轴：硬件支持、dequant 开销、异常值、校准数据和任务切片。

9. **KV cache 为什么容易成为容量瓶颈？**
   - 回答轴：层数、KV heads、head dim、序列、batch、精度与并发的乘积。

10. **GPU utilization 很高是否说明系统高效？**
    - 回答轴：不一定；需要结合 SM 效率、带宽、kernel 时间、吞吐和 goodput。

### 深挖与设计题

11. 一次训练吞吐突然下降 30%，你按什么顺序排查？
12. MoE 的某些 expert 持续过载，如何定位并治理？
13. 如何为长短请求混合的在线服务设计调度器？
14. Prefix cache 命中率高，但 P99 变差，可能是什么原因？
15. 如何在成本不增加的前提下提高满足 SLO 的 goodput？
16. 设计一个支持多租户、配额、抢占和故障恢复的 GPU 平台。
17. Agent 执行大量外部工具时，GPU 为什么会空等？怎样解耦？
18. 如何证明一次优化没有以输出质量或稳定性为代价？

## 💻 估算与白板题

### 显存估算模板

```text
总显存 ≈ 参数 + 梯度 + 优化器状态 + 激活
       + KV cache / 临时张量 + 通信 buffer + 碎片余量
```

面试时应说明 dtype、是否分片、是否 activation checkpointing、序列和 micro-batch。结果不必精确到 MB，但每个量的数量级和随配置变化的方向要正确。

### 性能实验模板

1. 固定模型、硬件、输入分布和随机种子。
2. 先测端到端基线，再用 profiler 定位最大项。
3. 一次只改一个关键变量。
4. 同时报吞吐、P50 / P95 / P99、显存、质量和成本。
5. 至少预热并重复多轮，标注均值、方差与异常点。

## ✅ 项目证据与自检

- [ ] 我能从业务 SLO 倒推 batch、并发、显存和副本数。
- [ ] 我能画出训练或推理的关键通信路径。
- [ ] 我有 profiler 证据，不只靠 GPU utilization 下结论。
- [ ] 我能解释优化前后的输入分布与测试环境。
- [ ] 我同时报告性能、质量、稳定性和成本。
- [ ] 我准备了一个优化无效或产生回归的案例。

## 📚 扩展阅读

- [开发岗专项题](../06-development-specialized.md)
- [算法与 AI 手撕题库](../22-algorithm-ai-coding-question-bank.md)
- [Agent Harness Engineering](../../02-tech-stack/27-agent-harness-engineering.md)
- [生产级 Agent 设计](../../03-practice/05-ship-agent-project.md)
