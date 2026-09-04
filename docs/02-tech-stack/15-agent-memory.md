---
type: 教程
status: 已发布
level: 进阶
topic:
  - Agent
  - 记忆
---

# Agent Memory - 从原理到实战

> **完整的 Agent 记忆系统开发指南**

---

## 📌 本节目标

学完本章，你将能够：
- ✅ 理解 Agent Memory 的核心概念和设计原则
- ✅ 掌握主流 Memory 工具的使用（Mem0、MemGPT、Zep）
- ✅ 设计并实现自己的 Memory 系统
- ✅ 回答 Memory 相关的面试题

---

## 💡 为什么 Agent 需要 Memory？

### 问题场景

**没有 Memory 的 Agent**：
```
用户（第1轮）："我叫张三，我喜欢红色"
Agent："好的，记住了"

用户（第2轮）："我喜欢什么颜色？"
Agent："抱歉，我不知道" ❌

用户（第3轮）："我叫什么名字？"
Agent："我不知道你的名字" ❌
```

**有 Memory 的 Agent**：
```
用户（第1轮）："我叫张三，我喜欢红色"
Agent："好的，我记住了：你叫张三，喜欢红色"
      [存入Memory：name=张三, favorite_color=红色]

用户（第2轮）："我喜欢什么颜色？"
Agent："你喜欢红色" ✅
      [从Memory检索：favorite_color=红色]

用户（第3轮）："我叫什么名字？"
Agent："你叫张三" ✅
      [从Memory检索：name=张三]
```

### 核心价值

- ✅ **持久化对话**：跨会话记忆用户信息
- ✅ **个性化体验**：基于用户偏好定制服务
- ✅ **任务连续性**：长期任务的状态管理
- ✅ **知识积累**：学习和演进

---

## 🏗️ Memory 系统架构

### 三层记忆架构（推荐）

```
┌─────────────────────────────────────┐
│   Layer 1: 工作记忆 (Working Memory)   │
│   • 当前会话的对话历史                      │
│   • 存储：内存（最快）                      │
│   • 容量：有限（最近 10-20 轮）             │
└─────────────────────────────────────┘
            ↓ (重要信息提取)
┌─────────────────────────────────────┐
│   Layer 2: 情节记忆 (Episodic Memory)  │
│   • 本次会话的关键信息                      │
│   • 存储：向量数据库                       │
│   • 容量：中等（单会话级别）                 │
└─────────────────────────────────────┘
            ↓ (知识提炼)
┌─────────────────────────────────────┐
│   Layer 3: 语义记忆 (Semantic Memory)  │
│   • 用户长期偏好、历史知识                  │
│   • 存储：知识图谱 + 向量库                │
│   • 容量：无限（跨会话持久化）               │
└─────────────────────────────────────┘
```

---

## 🛠️ 实战：3种 Memory 实现方案

### 方案1：Mem0（最简单，10行代码）⭐⭐⭐⭐⭐

**适合场景**：快速原型、对话 Agent、个性化应用

**快速开始**：

```python
from mem0 import Memory

# 1. 初始化 Memory
memory = Memory()

# 2. 添加记忆
memory.add(
    "用户叫张三，喜欢红色，职业是工程师",
    user_id="user_001"
)

# 3. 检索记忆
results = memory.search(
    "用户喜欢什么颜色？",
    user_id="user_001"
)

print(results)
# 输出：用户喜欢红色
```

**配置后端**：

```python
# 使用 Qdrant 作为向量库
config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": 6333
        }
    }
}

memory = Memory.from_config(config)
```

**优势**：
- ✅ 自动实体提取
- ✅ 自动去重
- ✅ 支持多种后端

**⚠️ 注意**：
- 生产环境需测试稳定性
- 注意 Token 消耗（自动提取会调用 LLM）

---

### 方案2：MemGPT（虚拟内存机制）⭐⭐⭐⭐

**适合场景**：长期对话、复杂任务、研究助手

**核心概念**：

```python
from memgpt import MemGPT

# 1. 创建 Agent（带虚拟内存）
agent = MemGPT.create_agent(
    persona="你是一个研究助手",
    human="用户是一名研究员"
)

# 2. 发送消息
response = agent.send_message(
    "帮我研究量子计算的最新进展"
)

# 3. Memory 自动管理
# MemGPT 会自动：
# - 存储重要信息到外部存储
# - 从外部加载相关记忆
# - 清理不重要的历史
```

**虚拟内存工作流**：

```
1. 用户输入
   ↓
2. 检索相关 Memory（从外部存储）
   ↓
3. 加载到主上下文（Swap in）
   ↓
4. LLM 处理
   ↓
5. 提取新信息
   ↓
6. 存储重要信息（Swap out旧的）
```

**优势**：
- ✅ 突破上下文窗口限制
- ✅ 自动内存管理
- ✅ 支持无限长对话

---

### 方案3：自己实现简单 Memory（LangChain）⭐⭐⭐

**适合场景**：学习理解、定制化需求、轻量级应用

**完整实现**（50行代码）：

```python
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

class CustomAgentMemory:
    def __init__(self):
        # 短期记忆（对话历史）
        self.buffer_memory = ConversationBufferMemory()
        
        # 长期记忆（向量存储）
        self.vector_store = Chroma(
            embedding_function=OpenAIEmbeddings()
        )
        
        self.llm = ChatOpenAI(model="gpt-4")
    
    def add_to_long_term(self, text: str, metadata: dict):
        """存储到长期记忆"""
        self.vector_store.add_texts([text], metadatas=[metadata])
    
    def retrieve_from_long_term(self, query: str, k: int = 3):
        """从长期记忆检索"""
        docs = self.vector_store.similarity_search(query, k=k)
        return "\n".join([doc.page_content for doc in docs])
    
    def chat(self, user_input: str):
        """对话接口"""
        # 1. 从长期记忆检索相关信息
        relevant_memory = self.retrieve_from_long_term(user_input)
        
        # 2. 构建完整上下文
        context = f"相关历史信息：\n{relevant_memory}\n\n当前对话："
        
        # 3. LLM 处理
        chain = ConversationChain(
            llm=self.llm,
            memory=self.buffer_memory
        )
        
        response = chain.run(input=context + user_input)
        
        # 4. 判断是否需要存入长期记忆
        if self.is_important(user_input):
            self.add_to_long_term(
                user_input + " -> " + response,
                metadata={"timestamp": "..."}
            )
        
        return response
    
    def is_important(self, text: str) -> bool:
        """判断是否重要（简化版）"""
        # 实际应该用 LLM 判断
        keywords = ["喜欢", "名字", "职业", "偏好"]
        return any(kw in text for kw in keywords)

# 使用示例
memory_agent = CustomAgentMemory()

# 对话1
memory_agent.chat("我叫张三，我是工程师")
# 对话2
memory_agent.chat("我喜欢什么？")
```

---

## 🎯 Memory 设计的5大核心问题

### 问题1：什么时候存储？

**策略选项**：
- **全部存储**：简单，但内存爆炸
- **关键信息存储**：需要判断"重要性"
- **周期性存储**：每 N 轮对话总结一次

**推荐方案**（重要性评分）：

```python
def calculate_importance(text: str) -> float:
    """计算信息重要性（0-1）"""
    score = 0.0
    
    # 1. 语义相关性（与用户画像相关）
    score += semantic_relevance(text) * 0.4
    
    # 2. 实体密度（包含多少关键实体）
    score += entity_density(text) * 0.3
    
    # 3. 时效性（是否是最新信息）
    score += recency(text) * 0.2
    
    # 4. 用户明确要求记住
    if "记住" in text or "别忘了" in text:
        score += 0.5
    
    return min(score, 1.0)

# 只存储重要性 > 0.6 的信息
if calculate_importance(text) > 0.6:
    memory.store(text)
```

---

### 问题2：如何存储？

**存储格式选择**：

| 格式 | 优点 | 缺点 | 适合场景 |
|:---|:---|:---|:---|
| **原文** | 信息完整 | 冗余、占空间 | 重要对话 |
| **摘要** | 节省空间 | 可能丢信息 | 一般对话 |
| **结构化**（实体+关系）| 便于检索、推理 | 提取成本高 | 知识型对话 |

**推荐：混合存储**

```python
class HybridMemoryStorage:
    def store(self, conversation: str):
        # 1. 原文存储（向量检索用）
        self.vector_store.add(conversation)
        
        # 2. 结构化提取（图谱用）
        entities = extract_entities(conversation)
        relations = extract_relations(conversation)
        self.knowledge_graph.add(entities, relations)
        
        # 3. 摘要存储（压缩用）
        summary = summarize(conversation)
        self.summary_store.add(summary)
```

---

### 问题3：如何检索？

**检索策略**：

```python
def retrieve_memory(query: str, user_id: str):
    """混合检索策略"""
    results = []
    
    # 1. 向量检索（语义相似）
    vector_results = vector_db.search(query, top_k=5)
    results.extend(vector_results)
    
    # 2. 图谱检索（关系推理）
    entities = extract_entities(query)
    graph_results = knowledge_graph.query(entities)
    results.extend(graph_results)
    
    # 3. 时间过滤（最近的优先）
    results = filter_by_recency(results, days=30)
    
    # 4. 重排序（综合评分）
    results = rerank(results, query)
    
    return results[:5]
```

---

### 问题4：何时遗忘？

**遗忘策略**：

```python
def should_forget(memory_item) -> bool:
    """判断是否应该遗忘"""
    
    # 1. 时间衰减
    age_days = (now - memory_item.timestamp).days
    if age_days > 90 and memory_item.importance < 0.5:
        return True
    
    # 2. 访问频率
    if memory_item.access_count == 0 and age_days > 30:
        return True
    
    # 3. 空间限制
    if memory_store.size() > MAX_SIZE:
        # 删除最不重要的
        return memory_item.importance < threshold
    
    # 4. 信息冗余
    if has_duplicate(memory_item):
        return True
    
    return False
```

---

### 问题5：如何更新？

**更新策略**：

```python
def update_memory(new_info: str, user_id: str):
    """更新记忆"""
    
    # 1. 检查是否与已有信息冲突
    existing = memory.search(new_info, user_id)
    
    for mem in existing:
        # 2. 如果冲突，判断哪个更新
        if is_conflict(mem, new_info):
            if is_newer(new_info):
                # 更新为新信息
                memory.update(mem.id, new_info)
            else:
                # 保留旧信息
                pass
        
        # 3. 如果是补充，则合并
        elif is_complementary(mem, new_info):
            merged = merge(mem, new_info)
            memory.update(mem.id, merged)
    
    # 4. 如果是全新信息，直接添加
    if not existing:
        memory.add(new_info, user_id)
```

---

## 💻 完整实战：对话 Agent + Memory

### 需求

构建一个客服 Agent，需要：
- 记住用户的基本信息（姓名、偏好）
- 记住历史问题和解决方案
- 个性化推荐

### 实现（基于 LangChain + Mem0）

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, Tool
from mem0 import Memory

class CustomerServiceAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4")
        self.memory = Memory()
        
        # 定义工具
        self.tools = [
            Tool(
                name="search_memory",
                func=self.search_memory,
                description="搜索用户的历史信息和偏好"
            ),
            Tool(
                name="save_to_memory",
                func=self.save_to_memory,
                description="保存重要的用户信息"
            )
        ]
        
        # 创建 Agent
        self.agent = create_react_agent(self.llm, self.tools)
    
    def search_memory(self, query: str, user_id: str) -> str:
        """搜索记忆"""
        results = self.memory.search(query, user_id=user_id)
        if results:
            return "\n".join([r['text'] for r in results])
        return "没有找到相关记忆"
    
    def save_to_memory(self, content: str, user_id: str) -> str:
        """保存记忆"""
        self.memory.add(content, user_id=user_id)
        return f"已保存：{content}"
    
    def chat(self, user_input: str, user_id: str) -> str:
        """对话接口"""
        # 1. 自动检索相关记忆
        relevant_memory = self.search_memory(
            user_input, 
            user_id=user_id
        )
        
        # 2. 构建提示词
        prompt = f"""你是客服 Agent。

相关用户信息：
{relevant_memory}

用户当前问题：
{user_input}

如果发现重要信息（如用户姓名、偏好），使用 save_to_memory 工具保存。
"""
        
        # 3. Agent 处理
        response = self.agent.invoke({
            "input": prompt
        })
        
        return response

# 使用
agent = CustomerServiceAgent()

# 第一轮对话
agent.chat("你好，我叫张三", user_id="user_001")
# Agent 会调用 save_to_memory("用户叫张三")

# 第二轮对话
agent.chat("我上次买了什么？", user_id="user_001")
# Agent 会调用 search_memory，找到"用户叫张三"
# 然后回答：张三你好，让我查查你的历史订单...
```

---

## 🎯 Memory 优化技巧

### 技巧1：分级存储（节省成本）

```python
class TieredMemory:
    def store(self, text: str, importance: float):
        if importance > 0.8:
            # 高重要性：原文 + 向量 + 图谱
            self.vector_store.add(text)
            self.graph.add(extract_entities(text))
        elif importance > 0.5:
            # 中重要性：摘要 + 向量
            summary = summarize(text)
            self.vector_store.add(summary)
        else:
            # 低重要性：仅摘要（可选）
            pass
```

### 技巧2：时间衰减（防止过载）

```python
def search_with_decay(query: str, decay_factor=0.9):
    """检索时考虑时间衰减"""
    results = vector_db.search(query)
    
    for r in results:
        age_days = (now - r.timestamp).days
        decay = decay_factor ** (age_days / 30)
        r.score = r.similarity * decay
    
    return sorted(results, key=lambda x: x.score, reverse=True)
```

### 技巧3：知识图谱增强（复杂推理）

```python
# 结合图谱和向量检索
def graph_enhanced_search(query: str):
    # 1. 向量检索找到候选实体
    entities = vector_search_entities(query)
    
    # 2. 图谱扩展（找到相关节点）
    expanded = knowledge_graph.expand(entities, hops=2)
    
    # 3. 重排序
    return rerank(expanded, query)
```

---

## 🎤 面试高频问题

### Q1: 如何设计 Agent 的长期记忆机制？

**标准答案**（开发岗）：
```
我会采用三层记忆架构：

1. 【工作记忆】
   - 当前对话（最近 10 轮）
   - 存储：内存
   - 检索：O(1) 直接访问

2. 【情节记忆】
   - 本次会话关键信息
   - 存储：向量数据库（Chroma）
   - 检索：向量相似度

3. 【语义记忆】
   - 用户长期偏好
   - 存储：知识图谱（Neo4j）+ 向量库
   - 检索：混合检索（图+向量）

使用 Mem0 作为实现框架，10 行代码集成。
```

**标准答案**（算法岗）：
```
我会参考 MemGPT 的虚拟内存机制：

1. 【核心算法】
   - 重要性评分函数：f(语义相关性, 实体密度, 时效性)
   - Swap策略：LRU + 重要性加权
   - 压缩算法：摘要 + 实体提取

2. 【优化方案】
   - 引入强化学习优化 swap 策略
   - 设计自适应窗口大小
   - 实验验证：内存占用-60%，性能不降

3. 【评估体系】
   - 记忆保留率（重要信息是否被保留）
   - 检索准确率（能否找到相关记忆）
   - 系统开销（时间、空间复杂度）
```

---

### Q2: Memory 系统的性能瓶颈在哪里？

**回答要点**：

**瓶颈1：检索延迟**
- 向量检索：50-200ms
- 图谱查询：100-500ms
- 解决：缓存热点查询

**瓶颈2：存储成本**
- LLM 调用（实体提取）：$$$
- 向量存储：空间成本
- 解决：分级存储、批处理

**瓶颈3：信息膨胀**
- 长对话后记忆爆炸
- 解决：定期压缩、遗忘机制

---

## 📚 推荐学习路径

### 开发岗（2-3天掌握）

```
Day 1: 理解概念 + Mem0 快速上手
  - 阅读本文档
  - 跑通 Mem0 示例
  
Day 2: 实现自己的 Memory
  - 基于 LangChain 实现
  - 集成到你的 Agent 项目
  
Day 3: 优化与调试
  - 性能优化
  - 测试稳定性
```

### 算法岗（1-2周掌握）

```
Week 1: 论文阅读
  - Agent Memory Survey（必读）
  - MemGPT + Mem0（应用驱动）
  - Memorizing Transformers（模型驱动）
  
Week 2: 算法设计与实验
  - 设计自己的 Memory 优化方案
  - 实验验证（对比 baseline）
  - 准备论文/开源
```

---

## 🔗 相关资源

- [Memory 工具对比](../../resources/agent/memory.md) - 工具选型
- [Memory 论文精选](../../resources/agent/papers/agent_memory/Agent%20Memory%20核心论文汇总.md) - 10篇必读
- [上下文工程](./14-context-engineering.md) - Context Offloading 技巧

---

**👉 返回主文档**：[README.md](../../README.md)


