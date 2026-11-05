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
- ✅ 掌握主流 Memory 工具的使用（Mem0、Letta/MemGPT、Zep）
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

# 3. 检索记忆（mem0ai 2.x：用户范围写在 filters 里）
results = memory.search(
    "用户喜欢什么颜色？",
    filters={"user_id": "user_001"},
    top_k=3,
)

for item in results["results"]:
    print(item["memory"], item["score"])
# 输出类似：喜欢红色 0.8x
```

> 版本说明：以上按 mem0ai 2.x 编写。旧版本的 `search(query, user_id=...)` 写法在 2.x 中会报错；`add` 仍然使用 `user_id=` 参数。默认配置调用 OpenAI 的模型做事实抽取和向量化，换成其他模型或向量库用下面的 `Memory.from_config`。

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

MemGPT 借用操作系统虚拟内存的思路：上下文窗口相当于内存，外部存储相当于磁盘，由模型自己调用函数决定把什么换入、换出。项目已更名为 [Letta](https://github.com/letta-ai/letta)，SDK 用法以官方文档为准。下面是机制示意，不对应具体 SDK：

```text
上下文窗口（主上下文）
├── 系统指令
├── core memory：常驻、可被模型编辑的记忆块
│   ├── persona：Agent 自身设定
│   └── human：用户画像（姓名、偏好……）
└── 最近的对话消息（FIFO，满了就摘要后移出）

外部存储
├── recall memory：完整对话历史，可按关键词 / 时间检索
└── archival memory：向量化的长期知识，可语义检索

模型可调用的记忆函数（示意）
core_memory_append / core_memory_replace   # 改写常驻记忆
conversation_search                        # 检索历史对话
archival_memory_insert / archival_memory_search
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

**完整实现**（LangChain 1.x）：

LangChain 1.x 中，短期记忆由 LangGraph 的 checkpointer 按 `thread_id` 保存完整对话状态；旧版的 `ConversationBufferMemory`、`ConversationChain` 已迁到 `langchain-classic`，新代码不再使用。

```python
"""
依赖：langchain>=1.0  langgraph  langchain-chroma，以及所选模型厂商的集成包
模型用 "provider:model" 字符串指定，从环境变量读取，例如
CHAT_MODEL="openai:<model-id>"、EMBEDDING_MODEL="openai:<embedding-model-id>"
"""
import os

from langchain.agents import create_agent
from langchain.embeddings import init_embeddings
from langchain_chroma import Chroma
from langgraph.checkpoint.memory import InMemorySaver


class CustomAgentMemory:
    def __init__(self):
        # 长期记忆：向量库，跨会话保留，按 user_id 隔离
        self.vector_store = Chroma(
            collection_name="long_term_memory",
            embedding_function=init_embeddings(os.environ["EMBEDDING_MODEL"]),
            persist_directory="./memory_db",
        )
        # 短期记忆：checkpointer 按 thread_id 保存对话状态
        # InMemorySaver 进程退出即丢失，生产环境换成 PostgresSaver 等持久化实现
        self.agent = create_agent(
            model=os.environ["CHAT_MODEL"],
            tools=[],
            system_prompt="你是一个会记住用户信息的助手。",
            checkpointer=InMemorySaver(),
        )

    def add_to_long_term(self, text: str, user_id: str):
        self.vector_store.add_texts([text], metadatas=[{"user_id": user_id}])

    def retrieve_from_long_term(self, query: str, user_id: str, k: int = 3) -> str:
        docs = self.vector_store.similarity_search(query, k=k, filter={"user_id": user_id})
        return "\n".join(doc.page_content for doc in docs)

    def chat(self, user_input: str, user_id: str, thread_id: str) -> str:
        # 1. 从长期记忆检索相关信息
        relevant = self.retrieve_from_long_term(user_input, user_id)
        content = f"相关历史信息：\n{relevant}\n\n用户：{user_input}" if relevant else user_input

        # 2. 同一 thread_id 的历史消息由 checkpointer 自动带上
        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": content}]},
            config={"configurable": {"thread_id": thread_id}},
        )
        reply = result["messages"][-1].content

        # 3. 判断是否写入长期记忆
        if self.is_important(user_input):
            self.add_to_long_term(user_input, user_id)
        return reply

    def is_important(self, text: str) -> bool:
        """简化版：实际项目一般让 LLM 抽取事实并打分，见下文“什么时候存储”"""
        keywords = ["喜欢", "名字", "叫", "职业", "偏好"]
        return any(kw in text for kw in keywords)


memory_agent = CustomAgentMemory()
memory_agent.chat("我叫张三，我是工程师", user_id="u001", thread_id="session-1")
# 新会话：短期记忆为空，靠长期记忆找回用户信息
print(memory_agent.chat("我是做什么工作的？", user_id="u001", thread_id="session-2"))
```

这个示例为了直观，把检索结果直接拼进了本轮用户消息，因此会随对话历史一起被 checkpointer 保存。生产中更常见的做法是用 middleware 在每次调用模型前动态注入系统提示，检索结果不写入历史。

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

### 实现（基于 LangChain 1.x + Mem0）

记忆的读写做成工具，由 Agent 自己决定何时检索、何时保存；`user_id` 通过运行时 context 注入工具，不让模型自己填写，避免串用户。

```python
"""
依赖：langchain>=1.0  langgraph  mem0ai>=2.0，以及所选模型厂商的集成包
"""
import os

from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from mem0 import Memory

memory = Memory()  # 默认配置使用 OpenAI；其他模型和向量库用 Memory.from_config


class UserContext(BaseModel):
    user_id: str


@tool
def search_memory(query: str, runtime: ToolRuntime[UserContext]) -> str:
    """搜索当前用户的历史信息、偏好和以往问题。"""
    res = memory.search(query, filters={"user_id": runtime.context.user_id}, top_k=5)
    hits = [item["memory"] for item in res.get("results", [])]
    return "\n".join(hits) if hits else "没有找到相关记忆"


@tool
def save_to_memory(content: str, runtime: ToolRuntime[UserContext]) -> str:
    """保存用户明确提供、且长期有效的信息，例如姓名、偏好、已解决问题的方案。"""
    memory.add(content, user_id=runtime.context.user_id)
    return "已保存"


agent = create_agent(
    model=os.environ["CHAT_MODEL"],
    tools=[search_memory, save_to_memory],
    system_prompt=(
        "你是客服 Agent。回答与用户历史相关的问题前，先用 search_memory 查询；"
        "用户提供姓名、偏好等长期有效的信息时，用 save_to_memory 保存。"
    ),
    context_schema=UserContext,
    checkpointer=InMemorySaver(),
)


def chat(user_input: str, user_id: str, thread_id: str) -> str:
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config={"configurable": {"thread_id": thread_id}},
        context=UserContext(user_id=user_id),
    )
    return result["messages"][-1].content


# 第一轮：Agent 应调用 save_to_memory 保存“用户叫张三”
chat("你好，我叫张三", user_id="user_001", thread_id="t1")

# 换一个会话：短期记忆为空，Agent 调用 search_memory 找回用户信息
chat("你还记得我叫什么吗？", user_id="user_001", thread_id="t2")
```

几个设计点：

- **短期记忆与长期记忆分开**：checkpointer 管单个会话内的上下文，Mem0 管跨会话的用户事实。
- **身份不交给模型**：`user_id` 来自调用方，模型只决定查什么、存什么。
- **写入要克制**：工具描述里限定“长期有效的信息”，否则模型会把寒暄也存进去；上线前要抽查实际写入内容。

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


