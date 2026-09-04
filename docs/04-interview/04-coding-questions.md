---
type: 题库
status: 已发布
level: 进阶
topic:
  - 面试求职
  - 基础模型
  - Agent
---

# AI Agent 面试题库 - 编程实战篇

## 📚 适用对象
- ✅ 算法工程师（手撕核心算法）
- ✅ 开发工程师（实现系统模块）
- ⏱️ 建议学习时间：3-5天

## 第一部分：LLM 基础编程（必备）

**Q1: 手撕 Self-Attention 机制**
- 难度：⭐⭐⭐
- 时间：30分钟
- 语言：Python + PyTorch
- 标签：#Attention #Transformer #手撕代码
- 公司：字节、阿里、腾讯（高频）

**要求**：实现自注意力机制，输入 Q, K, V (batch, seq_len, d_model)，输出 attention output

---

**Q2: 实现 BPE Tokenizer**
- 难度：⭐⭐⭐
- 时间：40分钟
- 语言：Python
- 标签：#Tokenizer #BPE #编码

**要求**：从零实现 BPE 分词算法，包括训练和编码两个阶段

---

**Q3: 实现 Top-K/Top-P 采样**
- 难度：⭐⭐
- 时间：20分钟
- 语言：Python + PyTorch
- 标签：#采样 #解码策略

**要求**：实现 Top-K 和 Nucleus (Top-P) 采样算法

---

**Q4: 实现 Multi-Head Attention**
- 难度：⭐⭐⭐
- 时间：35分钟
- 语言：Python + PyTorch
- 标签：#MHA #Transformer

**要求**：实现多头注意力机制，支持可配置的头数

---

**Q5: 实现 ROPE 位置编码**
- 难度：⭐⭐⭐⭐
- 时间：45分钟
- 语言：Python + PyTorch
- 标签：#ROPE #位置编码

**要求**：实现旋转位置编码（Rotary Position Embedding）

---

## 第二部分：Agent 核心模块

**Q6: 手撕 ReAct Agent**
- 难度：⭐⭐⭐⭐
- 时间：45分钟
- 语言：Python
- 标签：#ReAct #Agent #框架
- 公司：字节、阿里（高频）

**要求**：实现 ReAct 框架，支持 Thought → Action → Observation 循环

---

**Q7: 实现 Tool Registry 工具注册系统**
- 难度：⭐⭐⭐
- 时间：30分钟
- 语言：Python
- 标签：#ToolUse #Agent

**要求**：实现工具注册、查询、调用的完整系统

---

**Q8: 实现 Memory 系统（短期+长期记忆）**
- 难度：⭐⭐⭐⭐
- 时间：50分钟
- 语言：Python
- 标签：#Memory #Agent #存储

**要求**：实现对话历史管理（短期）+ 向量检索（长期记忆）

---

**Q9: 实现 Chain-of-Thought Prompting**
- 难度：⭐⭐
- 时间：20分钟
- 语言：Python
- 标签：#CoT #Prompt #推理

**要求**：实现 CoT 提示工程，引导 LLM 逐步推理

---

**Q10: 实现 Self-Reflection 自我反思机制**
- 难度：⭐⭐⭐⭐
- 时间：40分钟
- 语言：Python
- 标签：#Reflection #Agent #优化

**要求**：实现 Agent 自我评估与修正机制

---

## 第三部分：RAG 系统实现

**Q11: 实现文档切块策略（Chunking）**
- 难度：⭐⭐
- 时间：20分钟
- 语言：Python
- 标签：#RAG #文档处理 #切块
- 公司：阿里、腾讯（常考）

**要求**：实现固定大小切块、重叠切块、语义切块

---

**Q12: 实现混合检索（BM25 + 向量检索）**
- 难度：⭐⭐⭐
- 时间：35分钟
- 语言：Python
- 标签：#RAG #混合检索 #BM25

**要求**：实现 BM25 关键词检索 + 向量语义检索，并融合结果

---

**Q13: 实现 Reranker 重排序模块**
- 难度：⭐⭐⭐
- 时间：30分钟
- 语言：Python
- 标签：#RAG #Reranker #排序

**要求**：实现基于交叉编码器的重排序算法

---

**Q14: 实现 Semantic Cache 语义缓存**
- 难度：⭐⭐⭐
- 时间：30分钟
- 语言：Python
- 标签：#RAG #缓存 #优化

**要求**：实现基于语义相似度的查询缓存系统

---

**Q15: 实现 HyDE（假设性文档嵌入）**
- 难度：⭐⭐⭐⭐
- 时间：35分钟
- 语言：Python
- 标签：#RAG #HyDE #查询优化

**要求**：实现查询重写，生成假设性答案用于检索

---

## 第四部分：模型优化与推理

**Q16: 实现 KV Cache**
- 难度：⭐⭐⭐⭐
- 时间：45分钟
- 语言：Python + PyTorch
- 标签：#推理优化 #KVCache

**要求**：实现 KV Cache 加速自回归生成

---

**Q17: 实现 Beam Search 解码**
- 难度：⭐⭐⭐
- 时间：35分钟
- 语言：Python
- 标签：#解码 #BeamSearch

**要求**：实现 Beam Search 算法，支持长度惩罚

---

**Q18: 实现 LoRA 微调**
- 难度：⭐⭐⭐⭐⭐
- 时间：60分钟
- 语言：Python + PyTorch
- 标签：#LoRA #微调 #PEFT

**要求**：实现 LoRA 低秩适配器

---

## 第五部分：评估与监控

**Q19: 实现 BLEU/ROUGE 评估指标**
- 难度：⭐⭐
- 时间：25分钟
- 语言：Python
- 标签：#评估 #指标

**要求**：从零实现 BLEU-N 和 ROUGE-L 计算

---

**Q20: 实现 LLM-as-a-Judge 评估框架**
- 难度：⭐⭐⭐
- 时间：30分钟
- 语言：Python
- 标签：#评估 #LLMJudge

**要求**：使用 LLM 评估另一个模型的输出质量

---

**Q21: 实现 Agent 任务成功率统计**
- 难度：⭐⭐
- 时间：20分钟
- 语言：Python
- 标签：#评估 #Agent #监控

**要求**：统计 Agent 执行任务的成功率、平均步数、Token 消耗

---

## 第六部分：综合实战

**Q22: 实现一个完整的 Mini RAG 系统**
- 难度：⭐⭐⭐⭐⭐
- 时间：90分钟
- 语言：Python
- 标签：#RAG #综合实战 #系统设计
- 公司：字节、阿里（大题）

**要求**：
- 文档加载与切块
- 向量化与存储
- 检索与生成
- 支持流式输出

---

**Q23: 实现一个支持工具调用的 Agent**
- 难度：⭐⭐⭐⭐⭐
- 时间：90分钟
- 语言：Python
- 标签：#Agent #综合实战 #ToolUse
- 公司：字节、阿里、腾讯（大题）

**要求**：
- 工具定义（天气查询、计算器、搜索）
- ReAct 推理循环
- 异常处理与重试
- 对话历史管理

---

## 💡 备考建议

### 算法岗重点
- 第一部分：必练（Attention、ROPE）
- 第二部分：重点（ReAct、Memory）
- 第四部分：重点（LoRA、KV Cache）

### 开发岗重点
- 第二部分：必练（ReAct、Tool Registry）
- 第三部分：必练（RAG 全部）
- 第六部分：必练（综合实战）

### 时间分配建议
- 简单题（⭐⭐）：先快速实现，保证正确性
- 中等题（⭐⭐⭐）：重点练习，反复优化
- 困难题（⭐⭐⭐⭐+）：理解思路，能讲清楚即可

---

## 第七部分：Transformer 核心组件手撕（11题）

**Q24: 手撕自注意力机制（用PyTorch）**
- 难度：⭐⭐⭐
- 时间：30分钟
- 语言：Python + PyTorch
- 标签：#SelfAttention #Transformer
- 公司：所有大厂（必考）

**要求**：
- 实现标准Self-Attention机制
- 输入shape: (batch, seq_len, d_model)
- 支持Attention Mask
- 包含Scaled Dot-Product Attention

```python
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # 实现Q、K、V的线性变换
        pass

    def forward(self, x, mask=None):
        # 实现注意力计算
        pass
```

---

**Q25: 手撕自注意力机制（不用PyTorch）**
- 难度：⭐⭐⭐⭐
- 时间：40分钟
- 语言：Python + NumPy
- 标签：#SelfAttention #NumPy
- 公司：字节、阿里（算法岗）

**要求**：
- 纯NumPy实现Self-Attention
- 包含矩阵乘法、Softmax、Scaled
- 理解底层数学原理

---

**Q26: 手撕多头注意力机制（用PyTorch）**
- 难度：⭐⭐⭐⭐
- 时间：35分钟
- 语言：Python + PyTorch
- 标签：#MultiHeadAttention #Transformer
- 公司：所有大厂（高频）

**要求**：
- 实现Multi-Head Attention
- 支持可配置的头数num_heads
- 包含多头拼接和输出投影
- 输入输出shape保持一致

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        # 实现多头注意力
        pass

    def forward(self, x, mask=None):
        # 1. 分头
        # 2. 计算注意力
        # 3. 拼接
        # 4. 输出投影
        pass
```

---

**Q27: 手撕多头注意力机制（不用PyTorch）**
- 难度：⭐⭐⭐⭐⭐
- 时间：50分钟
- 语言：Python + NumPy
- 标签：#MultiHeadAttention #NumPy
- 公司：字节、阿里（算法岗难题）

**要求**：
- 纯NumPy实现MHA
- 处理多头的reshape和拼接
- 完整实现包括权重矩阵

---

**Q28: 手撕MQA（Multi-Query Attention）**
- 难度：⭐⭐⭐⭐
- 时间：35分钟
- 语言：Python + PyTorch
- 标签：#MQA #Attention
- 公司：字节、阿里

**要求**：
- 实现MQA：多个Query共享一组KV
- 对比MHA：减少KV Cache大小
- 支持mask和缓存

```python
class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        # MQA: K和V只有1个头，Q有num_heads个头
        pass

    def forward(self, x, kv_cache=None, mask=None):
        # 实现MQA逻辑
        pass
```

---

**Q29: 手撕绝对位置编码**
- 难度：⭐⭐⭐
- 时间：25分钟
- 语言：Python + PyTorch
- 标签：#位置编码 #PositionEncoding
- 公司：所有大厂（常考）

**要求**：
- 实现原始Transformer的sin/cos位置编码
- PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
- PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

```python
def positional_encoding(max_len, d_model):
    """
    生成绝对位置编码
    Args:
        max_len: 最大序列长度
        d_model: 模型维度
    Returns:
        pos_encoding: shape (max_len, d_model)
    """
    pass
```

---

**Q30: 手撕RoPE（旋转位置编码）**
- 难度：⭐⭐⭐⭐⭐
- 时间：50分钟
- 语言：Python + PyTorch
- 标签：#ROPE #位置编码
- 公司：字节、阿里（必考）

**要求**：
- 实现RoPE位置编码
- 通过旋转矩阵注入位置信息
- 支持外推（长度超过训练长度）

```python
class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, max_len=2048, base=10000):
        super().__init__()
        # 计算旋转角度
        pass

    def forward(self, q, k, seq_len):
        """
        对Q和K应用旋转位置编码
        Args:
            q: (batch, num_heads, seq_len, head_dim)
            k: (batch, num_heads, seq_len, head_dim)
        """
        pass
```

---

**Q31: 手撕Transformer中FFN代码**
- 难度：⭐⭐⭐
- 时间：20分钟
- 语言：Python + PyTorch
- 标签：#FFN #Transformer
- 公司：所有大厂

**要求**：
- 实现Position-wise Feed-Forward Network
- 两层线性变换 + 激活函数
- FFN(x) = max(0, xW1 + b1)W2 + b2

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # 通常 d_ff = 4 * d_model
        pass

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        pass
```

---

**Q32: 手撕Layer Norm**
- 难度：⭐⭐⭐
- 时间：25分钟
- 语言：Python + PyTorch
- 标签：#LayerNorm #归一化
- 公司：字节、阿里（常考）

**要求**：
- 实现Layer Normalization
- 在最后一维（特征维度）归一化
- 包含可学习的γ和β参数

```python
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        归一化公式: (x - mean) / sqrt(var + eps) * gamma + beta
        """
        pass
```

---

**Q33: 手撕RMSNorm**
- 难度：⭐⭐⭐
- 时间：20分钟
- 语言：Python + PyTorch
- 标签：#RMSNorm #归一化
- 公司：字节、阿里（LLaMA相关）

**要求**：
- 实现Root Mean Square Normalization
- 相比LayerNorm：去掉mean，只用RMS
- RMSNorm(x) = x / RMS(x) * γ，其中RMS(x) = sqrt(mean(x²) + eps)

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        RMSNorm: x / sqrt(mean(x^2) + eps) * weight
        """
        pass
```

---

**Q34: 手撕FlashAttention（简化版）**
- 难度：⭐⭐⭐⭐⭐
- 时间：60分钟
- 语言：Python + PyTorch
- 标签：#FlashAttention #优化
- 公司：字节、阿里（顶级难题）

**要求**：
- 理解FlashAttention的分块计算思想
- 实现Tiling优化（分块加载到SRAM）
- 减少HBM访问次数

```python
def flash_attention_forward(Q, K, V, block_size=64):
    """
    FlashAttention简化版实现
    核心思想：分块计算attention，减少显存IO

    Args:
        Q, K, V: (batch, num_heads, seq_len, head_dim)
        block_size: 分块大小

    Returns:
        O: (batch, num_heads, seq_len, head_dim)
    """
    # 1. 初始化输出和归一化统计量
    # 2. 分块遍历K和V
    # 3. 对每个块计算局部attention
    # 4. 在线更新全局统计量和输出
    pass
```

**考点**：
- IO复杂度分析：标准Attention vs FlashAttention
- Tiling技术：如何选择block_size
- 在线Softmax：如何增量更新归一化

**提示**：
- 标准Attention：O(N²) HBM读写
- FlashAttention：O(N²/B) HBM读写，B是block_size
- 核心是在线更新Softmax分母和输出

---

## 💡 第七部分学习建议

### 必练题目（面试高频）
1. **Q24: 手撕Self-Attention（PyTorch）** - 100%会考
2. **Q26: 手撕Multi-Head Attention（PyTorch）** - 90%会考
3. **Q30: 手撕RoPE** - 字节/阿里必考
4. **Q32/Q33: Layer Norm / RMSNorm** - 基础必备

### 算法岗加练
- **Q25/Q27**: 不用PyTorch版本，考察数学功底
- **Q34: FlashAttention** - 顶会算法岗会考
- **Q28: MQA** - 理解KV Cache优化

### 开发岗重点
- **Q24/Q26**: 能用PyTorch快速实现即可
- **Q31: FFN** - 简单但必考
- 理解原理比手撕更重要

### 时间规划
- **基础题（Q24/Q29/Q31/Q32）**：2天练熟
- **进阶题（Q26/Q28/Q30/Q33）**：3天掌握
- **顶级题（Q25/Q27/Q34）**：2天理解思路

### 答题技巧
1. **先讲思路**：画图说明计算流程
2. **写伪代码**：确保逻辑正确
3. **边写边测**：举小例子验证
4. **优化空间**：讨论时间/空间复杂度
5. **提及应用**：哪些模型用了这个技术
