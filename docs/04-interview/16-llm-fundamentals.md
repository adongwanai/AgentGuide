---
type: 题库
status: 已发布
level: 入门
topic:
  - 面试求职
  - 基础模型
---

# 大模型基础必考题（107题）

## 📚 适用对象
- ✅ 算法工程师
- ✅ LLM研发工程师
- ⏱️ 建议学习时间：10-15天
- 🎯 覆盖面试高频考点

---

## 第一部分：Transformer 架构基础（26题）

**Q1: Transformer前馈神经网络用的是什么激活函数？**
- 难度：⭐⭐
- 标签：#Transformer #激活函数
- 公司：字节、阿里（基础题）

**考点**：
- 原始Transformer使用ReLU激活函数
- 现代LLM（如LLaMA、Qwen）多使用SwiGLU
- 需要说明不同激活函数的优劣

---

**Q18: 简述一下Transformer的基本结构和原理**
- 难度：⭐⭐⭐
- 标签：#Transformer #架构
- 公司：所有大厂（必考）

**考点**：
- Encoder-Decoder结构
- Multi-Head Attention机制
- Position Encoding
- Feed Forward Network
- Residual Connection + Layer Norm

---

**Q19: Transformer为什么使用多头注意力机制？**
- 难度：⭐⭐⭐
- 标签：#多头注意力 #Attention
- 公司：字节、腾讯（高频）

**考点**：
- 捕获不同子空间的语义信息
- 增强模型表达能力
- 类比CNN中的多个卷积核

---

**Q20: Transformer为何让Q(查询)和K(键)使用独立的权重矩阵进行计算，为什么需要Q、K、V(查询、键、值)三个矩阵？**
- 难度：⭐⭐⭐
- 标签：#Attention #QKV
- 公司：阿里、字节（常考）

**考点**：
- Q和K分离：实现查询和被查询的解耦
- V的作用：存储实际要传递的信息
- 类比检索系统：Query-Key匹配，返回Value

---

**Q21: 为什么在attention中要进行scaled(为什么除以√d_k)？**
- 难度：⭐⭐⭐
- 标签：#Attention #Scaled
- 公司：字节、美团（高频）

**考点**：
- 防止点积结果过大导致softmax梯度消失
- d_k越大，点积方差越大，需要归一化
- 数学推导：Var(Q·K^T) = d_k

---

**Q22: Transformer的位置编码作用及局限性？**
- 难度：⭐⭐⭐
- 标签：#位置编码 #PositionEncoding
- 公司：阿里、腾讯（常考）

**考点**：
- 作用：为模型引入序列顺序信息
- 绝对位置编码的局限：外推能力差
- 引出ROPE等相对位置编码

---

**Q23: 为什么Transformer用LayerNorm而不用BatchNorm？**
- 难度：⭐⭐⭐
- 标签：#归一化 #LayerNorm
- 公司：字节、阿里（高频）

**考点**：
- NLP序列长度不一致，BatchNorm统计量不稳定
- LayerNorm在特征维度归一化，不依赖batch
- Transformer中每个样本独立处理

---

**Q24: 解释一下KV Cache的原理**
- 难度：⭐⭐⭐⭐
- 标签：#推理优化 #KVCache
- 公司：字节、阿里、腾讯（必考）

**考点**：
- 自回归生成中避免重复计算历史token的KV
- 空间换时间：缓存所有历史的Key和Value
- 显存占用计算：2 × n_layers × d_model × seq_len

---

**Q25: Transformer相比RNN、LSTM的优势何在？**
- 难度：⭐⭐⭐
- 标签：#Transformer #对比
- 公司：所有大厂（基础题）

**考点**：
- 并行化训练（RNN必须串行）
- 长距离依赖建模能力强
- 计算效率高（矩阵运算优化好）

---

**Q26: Transformer在哪里做了权重共享？**
- 难度：⭐⭐
- 标签：#权重共享 #Transformer
- 公司：美团、字节

**考点**：
- Encoder和Decoder的Embedding层可以共享
- 输出层和Embedding层权重可以共享（tied weights）
- 减少参数量，提高训练效率

---

**Q32: 介绍一下RoPE位置编码**
- 难度：⭐⭐⭐⭐
- 标签：#ROPE #位置编码
- 公司：字节、阿里（高频）

**考点**：
- 旋转位置编码：通过旋转矩阵注入位置信息
- 相对位置建模：attention score只与相对位置有关
- 外推性好：适合长文本场景

---

**Q33: 旋转位置编码为什么比绝对位置编码更好**
- 难度：⭐⭐⭐⭐
- 标签：#ROPE #对比
- 公司：阿里、字节

**考点**：
- 绝对位置编码：外推能力差，训练长度受限
- ROPE优势：自然支持相对位置，外推性好
- 数学证明：Q_m · K_n^T 只与 (m-n) 有关

---

**Q34: RMSNorm正则化有什么好处和优势**
- 难度：⭐⭐⭐
- 标签：#RMSNorm #归一化
- 公司：字节、阿里

**考点**：
- 相比LayerNorm：去掉mean计算，只保留RMS
- 计算效率提升10-20%
- 效果不降：LLaMA系列都用RMSNorm

---

**Q36: SwiGLU激活函数的原理**
- 难度：⭐⭐⭐
- 标签：#激活函数 #SwiGLU
- 公司：字节、阿里

**考点**：
- GLU变体：Swish(xW) ⊙ (xV)
- LLaMA、Qwen等主流模型标配
- 相比ReLU/GELU性能更好

---

**Q37: 为什么大模型用GQA(Group Query Attention)**
- 难度：⭐⭐⭐⭐
- 标签：#GQA #注意力机制
- 公司：字节、阿里（高频）

**考点**：
- MHA→GQA→MQA：从质量到效率的权衡
- GQA：多个Query共享一组KV，减少KV Cache
- LLaMA2、Qwen都在用GQA

---

**Q74: Transformer模型中，最占用参数的是MLP层吗？**
- 难度：⭐⭐⭐
- 标签：#参数分析 #Transformer
- 公司：美团、字节

**考点**：
- MLP层：2 × d_model × d_ff（通常d_ff = 4d_model）
- Attention层：4 × d_model²
- 结论：MLP层参数量通常占60-70%

---

**Q75: Transformer模型中计算量(FLOPs)的分析**
- 难度：⭐⭐⭐⭐
- 标签：#计算量 #FLOPs
- 公司：字节、阿里

**考点**：
- Attention：2n²d + 2nd²
- MLP：16nd²
- 短序列：MLP主导；长序列：Attention主导
- 临界点：n ≈ 8d

---

**Q76: Transformer模型中计算量(FLOPs)和参数的表格分析**
- 难度：⭐⭐⭐⭐
- 标签：#模型分析 #参数
- 公司：字节、阿里

**考点**：
- 制表对比Attention和MLP的参数量、计算量
- 分析不同序列长度下的瓶颈
- 优化方向：短文本优化MLP，长文本优化Attention

---

**Q84: Multi-head Latent Attention(多头隐变量注意力)是什么？**
- 难度：⭐⭐⭐⭐
- 标签：#MLA #Attention
- 公司：字节、阿里

**考点**：
- DeepSeek-V3提出的新架构
- 压缩KV到低维隐空间，减少KV Cache
- 原理：KV = Compress(X) → Decompress

---

**Q85: Linear Attention(线性注意力)是什么？**
- 难度：⭐⭐⭐⭐
- 标签：#LinearAttention #优化
- 公司：字节、快手

**考点**：
- 将O(n²)的Attention降到O(n)
- 核心：用kernel trick改写softmax(QK^T)V
- 问题：效果通常不如标准Attention

---

**Q86: Cross-Attention(交叉注意力)是什么？**
- 难度：⭐⭐⭐
- 标签：#CrossAttention #Attention
- 公司：所有大厂

**考点**：
- Q来自一个序列，K/V来自另一个序列
- 应用：Encoder-Decoder、多模态融合
- 举例：图像tokens作为K/V，文本query作为Q

---

**Q87: Sparse Attention(稀疏注意力)是什么？**
- 难度：⭐⭐⭐⭐
- 标签：#SparseAttention #长文本
- 公司：字节、阿里

**考点**：
- 只计算部分token之间的attention
- 模式：局部窗口、固定stride、全局tokens
- Longformer、BigBird等长文本模型常用

---

**Q88: 不同Attention之间的区别是什么？**
- 难度：⭐⭐⭐
- 标签：#Attention #对比
- 公司：字节、阿里（总结题）

**考点**：
- Self/Cross/Sparse/Linear Attention对比
- MHA/MQA/GQA的权衡
- 不同场景的选择依据

---

**Q89: Group Normalization (GN) 和 Instance Normalization (IN)的区别是什么？**
- 难度：⭐⭐⭐
- 标签：#归一化 #GN #IN
- 公司：美团、字节

**考点**：
- GN：在通道维度分组归一化
- IN：每个样本每个通道独立归一化
- 应用：GN用于小batch，IN用于风格迁移

---

**Q90: 不同的Normalization之间有什么区别？**
- 难度：⭐⭐⭐
- 标签：#归一化 #对比
- 公司：字节、阿里（总结题）

**考点**：
- BatchNorm/LayerNorm/RMSNorm/GroupNorm对比
- 归一化维度的选择
- 不同任务的适用场景

---

## 第二部分：PEFT 参数高效微调（8题）

**Q2: 解释一下LoRA的原理**
- 难度：⭐⭐⭐⭐
- 标签：#LoRA #PEFT
- 公司：所有大厂（必考）

**考点**：
- 低秩分解：ΔW = BA（B: d×r, A: r×k）
- 冻结预训练权重，只训练A和B
- 典型r=8/16，参数量减少到0.1%

---

**Q3: LoRA是怎样进行初始化矩阵的**
- 难度：⭐⭐⭐
- 标签：#LoRA #初始化
- 公司：字节、阿里

**考点**：
- A使用Kaiming初始化（或高斯初始化）
- B初始化为全0，确保训练初期ΔW=0
- 保证训练起点就是预训练模型

---

**Q4: 解释一下AdaLoRA和QLoRA的原理**
- 难度：⭐⭐⭐⭐
- 标签：#AdaLoRA #QLoRA #PEFT
- 公司：字节、阿里（进阶题）

**考点**：
- AdaLoRA：自适应调整每层的rank
- QLoRA：量化预训练模型+LoRA微调
- QLoRA核心：4bit量化+NF4数据类型+分页优化

---

**Q5: 解释一下Adapter**
- 难度：⭐⭐⭐
- 标签：#Adapter #PEFT
- 公司：阿里、腾讯

**考点**：
- 在Transformer层中插入小型瓶颈网络
- 结构：down-projection → 激活 → up-projection
- 冻结主模型，只训练Adapter

---

**Q6: 介绍几种常见的Adapter**
- 难度：⭐⭐⭐
- 标签：#Adapter #PEFT
- 公司：阿里、腾讯

**考点**：
- Series Adapter：串行插入
- Parallel Adapter：并行加入
- AdapterFusion：多任务Adapter融合

---

**Q7: 解释一下prefix-tuning**
- 难度：⭐⭐⭐
- 标签：#PrefixTuning #PEFT
- 公司：字节、阿里

**考点**：
- 在输入前添加可训练的连续向量
- 不修改模型参数，只优化prefix
- 每个任务学习不同的prefix

---

**Q8: 解释一下P-tuning**
- 难度：⭐⭐⭐
- 标签：#PTuning #PEFT
- 公司：字节、阿里

**考点**：
- 在输入中插入可学习的伪token
- 与prefix-tuning类似，但更灵活
- P-tuning v2：在每层都加入可训练参数

---

**Q9: 解释一下Prompt-tuning**
- 难度：⭐⭐⭐
- 标签：#PromptTuning #PEFT
- 公司：阿里、腾讯

**考点**：
- 只在输入层添加soft prompt
- 比prefix-tuning更轻量
- 适合大模型微调

---

## 第三部分：训练技术（15题）

**Q10: 解释一下混合精度的原理**
- 难度：⭐⭐⭐
- 标签：#混合精度 #训练优化
- 公司：字节、阿里（高频）

**考点**：
- FP16计算+FP32主权重
- Loss Scaling防止梯度下溢
- 训练速度提升2-3倍，显存减半

---

**Q11: 训练的时候用float16、bfloat16还是float32，为什么？**
- 难度：⭐⭐⭐
- 标签：#数据类型 #训练
- 公司：字节、阿里（常考）

**考点**：
- FP32：精度高但慢
- FP16：快但容易溢出
- BF16：动态范围大，不易溢出，主流选择

---

**Q12: 怎么解决训练使用float16导致溢出的问题**
- 难度：⭐⭐⭐
- 标签：#FP16 #溢出
- 公司：字节、美团

**考点**：
- Loss Scaling：放大loss防止梯度下溢
- 动态Loss Scaling：自动调整scale因子
- 梯度裁剪：防止梯度爆炸

---

**Q40: DeepSpeed中ZeRO系列的原理**
- 难度：⭐⭐⭐⭐⭐
- 标签：#DeepSpeed #ZeRO #分布式
- 公司：字节、阿里（必考）

**考点**：
- ZeRO-1：切分optimizer states
- ZeRO-2：切分optimizer + gradients
- ZeRO-3：切分optimizer + gradients + parameters
- ZeRO-Offload：CPU卸载

---

**Q68: 训练时数据长度不一致怎么办，以及如何优化训练速度**
- 难度：⭐⭐⭐
- 标签：#数据处理 #训练优化
- 公司：字节、阿里

**考点**：
- Padding + Attention Mask
- Dynamic Batching：按长度分组
- Pack多个短样本到一个序列

---

**Q80: 为什么需要增量预训练？**
- 难度：⭐⭐⭐
- 标签：#增量预训练 #预训练
- 公司：字节、阿里

**考点**：
- 领域适配：通用模型→领域模型
- 知识更新：注入最新数据
- 成本低：比从头训练省90%+

---

**Q81: 增量预训练的过程当中，loss上升正常吗？**
- 难度：⭐⭐⭐
- 标签：#增量预训练 #训练
- 公司：字节、阿里

**考点**：
- 短期上升可能正常：数据分布变化
- 持续上升有问题：学习率过大/数据质量差
- 监控指标：困惑度、下游任务评估

---

**Q82: 在增量预训练过程中如何设置学习率(learning rate, LR)？**
- 难度：⭐⭐⭐
- 标签：#学习率 #训练
- 公司：字节、阿里

**考点**：
- 通常设为原预训练的10%-50%
- 使用cosine decay或线性衰减
- 需要warmup阶段

---

**Q83: 什么是warmup ratio？训练过程中怎么设置？**
- 难度：⭐⭐⭐
- 标签：#Warmup #训练
- 公司：字节、阿里

**考点**：
- Warmup：学习率从0线性增加到peak
- 作用：稳定训练初期，防止梯度爆炸
- 典型设置：warmup_ratio = 0.03-0.1

---

**Q97: 为什么需要混合精度训练？**
- 难度：⭐⭐⭐
- 标签：#混合精度 #训练优化
- 公司：字节、阿里

**考点**：
- 加速：Tensor Core对FP16/BF16有硬件加速
- 省显存：激活值和梯度减半
- 保精度：主权重保持FP32

---

## 第四部分：模型架构（13题）

**Q13: CLM和MLM分别是什么，有什么区别？**
- 难度：⭐⭐⭐
- 标签：#CLM #MLM #预训练
- 公司：字节、阿里（基础题）

**考点**：
- CLM（Causal LM）：GPT，自回归生成
- MLM（Masked LM）：BERT，完形填空
- 区别：CLM单向，MLM双向；CLM生成，MLM理解

---

**Q14: GLM的自回归空白填充方法与BERT中的使用遮蔽语言模型(MLM)有什么不同，为自然语言理解(NLU)任务带来了哪些优势**
- 难度：⭐⭐⭐⭐
- 标签：#GLM #MLM
- 公司：字节、阿里

**考点**：
- GLM：自回归填充span，统一理解和生成
- 优势：可变长生成、适合生成任务
- 与BERT对比：GLM可以直接做生成

---

**Q16: Encoder-only、Decoder-only和Encoder-Decoder的模型分别有什么区别，怎么运用？**
- 难度：⭐⭐⭐
- 标签：#架构 #Transformer
- 公司：所有大厂（基础题）

**考点**：
- Encoder-only（BERT）：双向理解，适合分类
- Decoder-only（GPT）：单向生成，适合对话
- Encoder-Decoder（T5）：序列到序列，适合翻译

---

**Q17: 为什么现在的大语言模型都用Decoder-only**
- 难度：⭐⭐⭐⭐
- 标签：#架构选择 #DecoderOnly
- 公司：字节、阿里（高频）

**考点**：
- 统一范式：所有任务都用生成来解决
- 扩展性好：预训练目标简单，数据规模大
- In-context Learning：Decoder-only更适合ICL

---

**Q35: 介绍一下BERT的结构**
- 难度：⭐⭐⭐
- 标签：#BERT #Encoder
- 公司：所有大厂（经典题）

**考点**：
- Encoder-only架构
- 预训练任务：MLM + NSP
- 双向上下文建模

---

**Q38: 介绍一下LLaMA的结构**
- 难度：⭐⭐⭐⭐
- 标签：#LLaMA #架构
- 公司：字节、阿里（高频）

**考点**：
- RMSNorm代替LayerNorm
- SwiGLU激活函数
- RoPE位置编码
- GQA（LLaMA2）

---

**Q39: LLaMA模型在训练过程中如何处理梯度消失和梯度爆炸的问题**
- 难度：⭐⭐⭐⭐
- 标签：#LLaMA #训练
- 公司：字节、阿里

**考点**：
- 残差连接：缓解梯度消失
- RMSNorm：稳定训练
- 梯度裁剪：防止梯度爆炸
- Warmup：稳定训练初期

---

**Q51: 什么是MoE(混合专家模型)**
- 难度：⭐⭐⭐⭐
- 标签：#MoE #架构
- 公司：字节、阿里（热点）

**考点**：
- 多个专家网络+路由器
- 稀疏激活：每次只激活部分专家
- 扩大模型容量但不增加计算量

---

**Q52: Dense和MoE模型的区别**
- 难度：⭐⭐⭐⭐
- 标签：#MoE #对比
- 公司：字节、阿里

**考点**：
- Dense：所有参数都参与计算
- MoE：稀疏激活，部分专家参与
- MoE优势：参数多但计算量可控

---

**Q53: 介绍一下Qwen通义千问的结构**
- 难度：⭐⭐⭐⭐
- 标签：#Qwen #架构
- 公司：阿里（必考）

**考点**：
- 基于LLaMA改进
- 使用SwiGLU、RoPE、RMSNorm
- 支持32k/100k上下文长度

---

**Q54: Qwen2有哪些提升？**
- 难度：⭐⭐⭐⭐
- 标签：#Qwen2 #改进
- 公司：阿里

**考点**：
- GQA替代MHA
- 更大的词表（151k）
- 更长的训练数据和上下文窗口

---

**Q55: 介绍一下LLaMA3.1的创新**
- 难度：⭐⭐⭐⭐
- 标签：#LLaMA3 #创新
- 公司：字节、阿里

**考点**：
- 128k上下文窗口
- 工具调用能力
- 多语言能力增强
- 405B超大模型

---

**Q92: DeepSeekV3有啥技术特点？**
- 难度：⭐⭐⭐⭐⭐
- 标签：#DeepSeek #MoE
- 公司：字节、阿里（最新）

**考点**：
- MLA（Multi-head Latent Attention）
- 高效MoE架构
- FP8训练
- 辅助损失优化负载均衡

---

## 第五部分：推理优化（9题）

**Q27: 介绍一下vLLM的加速方法**
- 难度：⭐⭐⭐⭐
- 标签：#vLLM #推理优化
- 公司：字节、阿里（高频）

**考点**：
- PagedAttention：分页管理KV Cache
- Continuous Batching：动态批处理
- 高效内存管理：减少碎片

---

**Q28: 介绍一下FlashAttention的原理**
- 难度：⭐⭐⭐⭐⭐
- 标签：#FlashAttention #优化
- 公司：字节、阿里（必考）

**考点**：
- IO-aware算法：减少HBM访问
- Tiling技术：分块计算
- 加速2-4倍，不损失精度

---

**Q67: 如何缓解大模型inference的时候的重复问题**
- 难度：⭐⭐⭐
- 标签：#推理 #重复问题
- 公司：字节、美团

**考点**：
- Repetition Penalty：惩罚已生成token
- Temperature调节：增加随机性
- Top-k/Top-p采样：避免贪心解码

---

**Q69: 什么是Prefill-Decode分离？为什么分离？**
- 难度：⭐⭐⭐⭐
- 标签：#推理优化 #Prefill
- 公司：字节、阿里

**考点**：
- Prefill：并行处理prompt
- Decode：自回归生成
- 分离原因：计算特性不同，分离可优化吞吐

---

**Q93: 为什么要优化KV Cache？**
- 难度：⭐⭐⭐⭐
- 标签：#KVCache #优化
- 公司：字节、阿里

**考点**：
- KV Cache是推理显存瓶颈
- 长文本场景：KV Cache可占80%+显存
- 优化方法：MQA/GQA、量化、MLA

---

**Q94: MLA (Multi-head Latent Attention) 是什么？**
- 难度：⭐⭐⭐⭐⭐
- 标签：#MLA #DeepSeek
- 公司：字节、阿里（最新）

**考点**：
- DeepSeek-V3核心技术
- 压缩KV到低维隐空间
- 大幅减少KV Cache显存占用

---

**Q99: 为什么推理速度受限于显存带宽？**
- 难度：⭐⭐⭐⭐
- 标签：#推理 #显存带宽
- 公司：字节、阿里

**考点**：
- Decode阶段是memory-bound任务
- 每生成1个token需读取全部模型参数
- 计算量小但IO量大

---

## 第六部分：Tokenization（3题）

**Q29: 介绍一下Tokenizer怎么训练的**
- 难度：⭐⭐⭐
- 标签：#Tokenizer #训练
- 公司：字节、阿里

**考点**：
- 统计语料中的字符/子词频率
- 迭代合并高频pair（BPE）
- 构建词表（通常30k-100k）

---

**Q30: WordPiece如何考虑语言模型的概率**
- 难度：⭐⭐⭐
- 标签：#WordPiece #Tokenizer
- 公司：阿里、腾讯

**考点**：
- 选择合并后使语言模型概率最大的pair
- 与BPE的区别：BPE只看频率
- BERT使用WordPiece

---

**Q31: 介绍一下BPE&BBPE这两种Tokenization方法**
- 难度：⭐⭐⭐
- 标签：#BPE #Tokenizer
- 公司：字节、阿里

**考点**：
- BPE：迭代合并高频字符对
- BBPE（Byte-level BPE）：在字节级别操作
- GPT-2/GPT-3使用BBPE

---

## 第七部分：强化学习与对齐（17题）

**Q41: 解释下强化学习**
- 难度：⭐⭐⭐
- 标签：#强化学习 #基础
- 公司：所有大厂（基础题）

**考点**：
- Agent与Environment交互
- 通过Reward学习Policy
- 目标：最大化累积奖励

---

**Q42: 强化学习中策略函数和值函数是什么**
- 难度：⭐⭐⭐
- 标签：#强化学习 #策略函数
- 公司：字节、阿里

**考点**：
- 策略函数π(a|s)：状态到动作的映射
- 值函数V(s)：状态的期望回报
- Q函数Q(s,a)：状态-动作对的期望回报

---

**Q43: DPO的原理**
- 难度：⭐⭐⭐⭐⭐
- 标签：#DPO #对齐
- 公司：所有大厂（必考）

**考点**：
- 直接优化策略，无需reward model
- Bradley-Terry模型建模偏好
- 损失函数：-log σ(β log(π/π_ref))

---

**Q44: PPO的原理**
- 难度：⭐⭐⭐⭐⭐
- 标签：#PPO #强化学习
- 公司：所有大厂（必考）

**考点**：
- Proximal Policy Optimization
- Clipped Surrogate Objective：限制策略更新幅度
- OpenAI RLHF使用PPO

---

**Q45: 分析一下DPO和PPO的区别**
- 难度：⭐⭐⭐⭐⭐
- 标签：#DPO #PPO #对比
- 公司：字节、阿里（高频）

**考点**：
- PPO：需要reward model，训练复杂
- DPO：直接优化偏好，训练简单
- PPO更灵活，DPO更稳定高效

---

**Q46: DPO的正负样本怎么构造**
- 难度：⭐⭐⭐⭐
- 标签：#DPO #数据
- 公司：字节、阿里

**考点**：
- 人工标注：同一prompt的多个回复排序
- 模型生成：SFT模型生成+人工筛选
- 质量要求：正负样本差异要明显

---

**Q47: DPO的loss是什么？**
- 难度：⭐⭐⭐⭐
- 标签：#DPO #损失函数
- 公司：字节、阿里

**考点**：
- L = -E[log σ(β log(π_θ(y_w|x)/π_ref(y_w|x)) - β log(π_θ(y_l|x)/π_ref(y_l|x)))]
- y_w是chosen，y_l是rejected
- β控制偏离参考模型的程度

---

**Q48: OpenAI对齐为什么要用强化学习，别的方法不行吗？**
- 难度：⭐⭐⭐⭐
- 标签：#对齐 #RLHF
- 公司：字节、阿里

**考点**：
- 人类偏好难以用监督学习直接建模
- RL可以处理非可微的奖励信号
- 探索更优回复，超越训练数据

---

**Q49: 预训练和微调任务有什么区别，两者的目的**
- 难度：⭐⭐⭐
- 标签：#预训练 #微调
- 公司：所有大厂（基础题）

**考点**：
- 预训练：无监督学习通用语言知识
- 微调：有监督学习特定任务
- 两阶段：预训练打基础，微调做适配

---

**Q50: SFT后会出现哪些问题**
- 难度：⭐⭐⭐⭐
- 标签：#SFT #问题
- 公司：字节、阿里

**考点**：
- 过拟合：模型只会模仿训练数据
- 能力退化：忘记预训练知识
- 分布偏移：训练数据偏差影响泛化

---

**Q70: 什么是PRM过程监督奖励？OpenAI的秘密武器？**
- 难度：⭐⭐⭐⭐⭐
- 标签：#PRM #强化学习
- 公司：字节、阿里（前沿）

**考点**：
- Process Reward Model：监督推理过程
- 与ORM对比：ORM只看结果，PRM看每步
- 优势：纠正推理错误，提升数学推理能力

---

**Q91: 强化学习算法是怎么分类的？**
- 难度：⭐⭐⭐
- 标签：#强化学习 #分类
- 公司：字节、阿里

**考点**：
- Model-based vs Model-free
- On-policy vs Off-policy
- Value-based vs Policy-based vs Actor-Critic

---

**Q95: 怎么理解DPO的损失函数？**
- 难度：⭐⭐⭐⭐⭐
- 标签：#DPO #损失函数
- 公司：字节、阿里

**考点**：
- 从Bradley-Terry模型推导
- 隐式reward：r(x,y) = β log(π/π_ref)
- 最大化chosen和rejected的对数几率差

---

**Q96: DPO的微调流程**
- 难度：⭐⭐⭐⭐
- 标签：#DPO #流程
- 公司：字节、阿里

**考点**：
1. 准备SFT模型作为π_ref
2. 构造偏好数据对(x, y_w, y_l)
3. 计算DPO损失并优化
4. 评估对齐效果

---

**Q102: DeepSeek-R1-Zero里的Zero含义？**
- 难度：⭐⭐⭐⭐
- 标签：#DeepSeek #R1
- 公司：字节、阿里（最新）

**考点**：
- Zero：不依赖监督数据，纯RL训练
- 从SFT模型出发，仅用RL自我进化
- 证明RL的潜力

---

**Q103: DeepSeek-R1-Zero中的强化学习(RL)的原理？**
- 难度：⭐⭐⭐⭐⭐
- 标签：#DeepSeek #RL
- 公司：字节、阿里

**考点**：
- 使用PPO/GRPO算法
- Reward：任务正确性（如数学题答案）
- 探索推理策略，自动涌现CoT

---

**Q104: DeepSeek-R1-Zero有什么弊端**
- 难度：⭐⭐⭐⭐
- 标签：#DeepSeek #问题
- 公司：字节、阿里

**考点**：
- 训练不稳定：RL难以收敛
- 推理冗余：生成大量无效内容
- 需要冷启动优化

---

**Q105: 什么是冷启动？为什么要冷启动？**
- 难度：⭐⭐⭐
- 标签：#冷启动 #训练
- 公司：字节、阿里

**考点**：
- 冷启动：从强监督模型开始RL训练
- 原因：纯RL难以收敛，需要好的初始策略
- DeepSeek-R1使用SFT模型冷启动

---

**Q106: 在DeepSeek-R1上PRM和MCTS是否有用？**
- 难度：⭐⭐⭐⭐⭐
- 标签：#PRM #MCTS #DeepSeek
- 公司：字节、阿里

**考点**：
- PRM：过程奖励，引导推理方向
- MCTS：蒙特卡洛树搜索，增强推理规划
- 组合使用：显著提升复杂推理能力

---

**Q107: Deepseek中蒸馏R1是什么？是否比从零RL训练更好？**
- 难度：⭐⭐⭐⭐⭐
- 标签：#蒸馏 #R1 #DeepSeek
- 公司：字节、阿里

**考点**：
- 蒸馏R1：用R1生成的推理数据训练小模型
- 优势：训练稳定，成本低，效果接近R1
- 比从零RL更实用：工程化更简单

---

## 第八部分：位置编码与长度外推（4题）

**Q78: 什么是长度外推？**
- 难度：⭐⭐⭐
- 标签：#长度外推 #位置编码
- 公司：字节、阿里

**考点**：
- 模型在训练长度之外的序列上推理
- 挑战：位置编码未见过的位置
- 解决：ROPE、ALiBi、NTK等方法

---

**Q79: ALiBi (Attention with Linear Biases)思路是什么？**
- 难度：⭐⭐⭐⭐
- 标签：#ALiBi #位置编码
- 公司：字节、阿里

**考点**：
- 不用位置编码，直接在attention score上加线性bias
- bias ∝ 距离，距离越远bias越负
- 外推能力强，BloomZ使用ALiBi

---

**Q98: NTK长度外推方法是什么？**
- 难度：⭐⭐⭐⭐
- 标签：#NTK #长度外推
- 公司：字节、阿里

**考点**：
- Neural Tangent Kernel启发的ROPE外推
- 调整ROPE的base频率：base' = base × scale
- 低成本扩展上下文窗口

---

## 第九部分：模型评估（2题）

**Q100: PPL (Perplexity) 及其数学公式**
- 难度：⭐⭐⭐
- 标签：#困惑度 #评估
- 公司：所有大厂

**考点**：
- PPL = exp(-1/N Σ log P(w_i|context))
- 困惑度越低越好
- 衡量模型对测试集的预测能力

---

**Q101: 大海捞针测试和概率探针分别是什么？**
- 难度：⭐⭐⭐⭐
- 标签：#评估 #长文本
- 公司：字节、阿里

**考点**：
- 大海捞针：在长文本中插入事实，测试检索能力
- 概率探针：评估模型对位置的注意力分布
- 评估长上下文能力

---

## 第十部分：工程实践（6题）

**Q15: 了解迁移学习吗？大模型中是怎么运用迁移学习的？**
- 难度：⭐⭐⭐
- 标签：#迁移学习 #微调
- 公司：所有大厂

**考点**：
- 预训练模型作为初始化
- 在下游任务上微调
- 低资源场景的关键技术

---

**Q71: 从huggingface下载模型时有哪些文件？**
- 难度：⭐⭐
- 标签：#HuggingFace #实践
- 公司：字节、阿里

**考点**：
- pytorch_model.bin / model.safetensors：模型权重
- config.json：模型配置
- tokenizer.json / tokenizer_config.json：分词器
- generation_config.json：生成配置

---

**Q72: 到底什么是端到端模型**
- 难度：⭐⭐
- 标签：#端到端 #概念
- 公司：所有大厂

**考点**：
- 输入原始数据，直接输出最终结果
- 不需要手工特征工程
- 一次训练优化整个pipeline

---

**Q73: 什么叫长尾问题？怎么解决长尾分布问题？**
- 难度：⭐⭐⭐
- 标签：#长尾问题 #数据
- 公司：字节、阿里

**考点**：
- 少数类别占据大部分数据，多数类别样本稀少
- 解决：重采样、类别加权、数据增强、Few-shot学习

---

**Q77: 什么是交叉熵损失函数？大模型的哪里有交叉熵损失函数？**
- 难度：⭐⭐⭐
- 标签：#损失函数 #交叉熵
- 公司：所有大厂

**考点**：
- L = -Σ y_i log(p_i)
- 用于分类任务
- 大模型：下一个token预测（词表上的多分类）

---

## 💡 备考建议

### 算法岗重点
1. **Transformer基础**（Q18-Q26）：必须滚瓜烂熟
2. **位置编码**（Q32-Q33）：ROPE原理要能推导
3. **PEFT方法**（Q2-Q9）：LoRA原理必考
4. **强化学习**（Q43-Q50）：DPO/PPO对比要清楚
5. **推理优化**（Q27-Q28）：FlashAttention原理要懂

### 开发岗重点
1. **模型架构**（Q38、Q53-Q55）：主流模型要熟悉
2. **训练技术**（Q10-Q12、Q40）：混合精度、ZeRO
3. **推理优化**（Q27、Q69、Q93）：vLLM、KV Cache
4. **工程实践**（Q71-Q73）：实际操作经验

### 学习路径
- **第1-3天**：Transformer基础（第一部分）
- **第4-5天**：PEFT方法（第二部分）
- **第6-7天**：训练技术（第三部分）
- **第8-9天**：强化学习与对齐（第七部分）
- **第10天**：推理优化和工程实践

### 答题技巧
- **算法岗**：要能推公式、画图、分析复杂度
- **开发岗**：强调工程实现、性能优化、问题排查
- **进阶题**：结合最新论文（DeepSeek、LLaMA3等）
- **对比题**：制表对比，说明适用场景

---

## 📖 参考资源
- 论文：Attention is All You Need, LoRA, DPO, FlashAttention
- 模型：LLaMA3.1, Qwen2, DeepSeek-V3
- 框架：Transformers, DeepSpeed, vLLM
- 博客：LLM推理优化、RLHF实践

---

**最后更新**: 2025-01-15
**题目数量**: 96道理论题 + 11道手撕代码题 = 107题
