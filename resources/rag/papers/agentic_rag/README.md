---
type: 入口页
status: 已发布
level: 通用
topic:
  - RAG
  - 科研
  - Agent
---

# Agentic RAG 论文汇总

> 本文档汇总了 RAG 与推理、智能体结合的前沿研究论文

---

## 📑 目录

- [RAG 与推理（RAG-Reasoning）](#rag-与推理rag-reasoning)
- [RAG 系统性思考](#rag-系统性思考)
- [幻觉缓解技术](#幻觉缓解技术)

---

## RAG 与推理（RAG-Reasoning）

### 综述 1：同济大学（2025.04.24）⭐⭐⭐⭐

**Synergizing RAG and Reasoning: A Systematic Review**

- **论文链接**：https://arxiv.org/pdf/2504.15909
- **代码库**：暂无
- **核心内容**：主要探讨了检索增强生成（RAG）与推理能力结合的研究进展
- **适合场景**：
  - 面试准备：RAG 如何与推理结合？怎么实现？
  - 系统设计：理解 RAG-Reasoning 架构
- **扩展阅读**：[面试官：RAG 如何与推理结合？怎么实现？](https://ccn7vpu5l5y8.feishu.cn/wiki/XfwcwnOLoiAqdfkZUGncYuAnnMc)

---

### 综述 2：清华大学（2025.07.16）⭐⭐⭐⭐⭐ 强烈推荐

**Towards Agentic RAG with Deep Reasoning: A Survey of RAG-Reasoning Systems in LLMs**

- **论文链接**：https://arxiv.org/pdf/2507.09477
- **代码库**：https://github.com/DavidZWZ/Awesome-RAG-Reasoning
- **推荐原因**：最全面的 RAG-Reasoning 综述，配套 Awesome 资源库

#### 核心观点

检索增强生成（RAG）通过注入外部知识来提升大型语言模型（LLMs）的事实准确性，但在需要多步骤推理的问题上表现不足；相反，纯粹以推理为导向的方法往往会编造或错误关联事实。

本综述从统一的**推理-检索视角**综合了这两条研究路径：

#### 三大研究方向

1. **推理增强型 RAG（Reasoning-Enhanced RAG）**
   - 高级推理如何优化 RAG 的各个阶段
   - 提升检索质量和相关性判断

2. **RAG 增强型推理（RAG-Enhanced Reasoning）**
   - 不同类型检索知识如何提供缺失的前提
   - 扩展复杂推理的上下文

3. **协同 RAG-推理框架（Synergistic RAG-Reasoning）** ⭐ 重点
   - 智能体式（Agentic）LLMs 通过迭代交替搜索和推理
   - 在知识密集型基准测试中实现最先进的性能

#### 内容涵盖

- ✅ 方法分类与对比
- ✅ 数据集汇总
- ✅ 开放挑战分析
- ✅ 未来研究方向

#### 研究目标

构建更有效、多模态自适应、可信且以人类为中心的深度 RAG-Reasoning 系统。

---

## RAG 系统性思考

### 综述 3：阿里巴巴 + 新加坡南洋理工（2025.10.10）⭐⭐⭐⭐⭐

**When Retrieval Succeeds and Fails: Rethinking Retrieval-Augmented Generation for LLMs**

- **论文链接**：https://arxiv.org/pdf/2510.09106
- **代码库**：暂无
- **核心内容**：系统性思考当前 RAG 面临的关键挑战和解决思路

#### 四大核心问题

##### 1. 何时触发检索？（When to Retrieve）

**挑战**：LLM 知识边界不明，不知何时该检索

**解决方案**：
- 自适应触发检索
- 通过不确定性评估（语义不确定性、自我不确定性等）判断 LLM 对查询的置信度
- **仅当 LLM 无法自信回答时激活 RAG**

##### 2. 检索什么？（What to Retrieve）

**挑战**：复杂查询检索低效

**解决方案**：
- 发展 **Agentic RAG**
- 通过智能体分解复杂任务为多步骤
- 自适应检索外部知识
- **未来方向**：深化用户意图理解，建立适配多样复杂任务的统一范式

##### 3. 信任什么？（What to Trust）

**挑战**：外部数据源不可靠

**解决方案**：
- 构建高质量知识库
- 做工具增强智能体（Tool-Augmented Agents）
- 建立知识验证机制

##### 4. RAG 如何工作？（How RAG Works）

**挑战**：ICL 机制与 RAG 结合不明

**研究方向**：
- 借助解释性技术（注意力滚动、因果追踪、表示探测等）
- 研究 ICL（In-Context Learning）机制
- 分析查询与检索知识的信息流动
  - 注意力头作用
  - 证据与参数先验竞争等内容

---

## 幻觉缓解技术

### 综述 4：LLM 幻觉缓解技术总结（2025.10）⭐⭐⭐⭐

**Mitigating Hallucination in Large Language Models (LLMs): An Application-Oriented Survey on RAG, Reasoning, and Agentic Systems**

- **论文链接**：https://arxiv.org/pdf/2510.24476
- **核心内容**：从能力增强视角系统分析 LLM 幻觉缓解方案

#### 幻觉分类

1. **知识型幻觉（Knowledge Hallucination）**
   - 表现：事实错误
   - 缓解方案：RAG（检索增强生成）

2. **逻辑型幻觉（Logical Hallucination）**
   - 表现：推理缺陷
   - 缓解方案：推理增强

#### 三级缓解框架

```
RAG（检索增强生成）+ 推理增强 + 智能体系统（Agentic Systems）
```

#### 方案风险分析

##### RAG 风险
- 检索质量依赖查询理解
- 不相关内容可能引入噪声

##### 推理增强风险
- **过度思考（Overthinking）问题**
- 冗长推理链可能引入新幻觉
- 计算成本增加

##### 智能体系统风险
- 模块协作效率低
- 错误易在模块间传播
- 系统复杂度增加

#### 核心启示

虽然 RAG、推理增强和智能体系统三者结合是缓解幻觉的主流方向，但需要：
- ✅ 平衡各方案的收益与风险
- ✅ 建立有效的质量控制机制
- ✅ 优化模块间的协作效率

---

## 🔗 相关资源

### 本项目其他文档

- [RAG 资源汇总](../../README.md)
- [RAG 论文总览](../README.md)

### 外部资源

- [Awesome-RAG-Reasoning](https://github.com/DavidZWZ/Awesome-RAG-Reasoning) - 清华大学维护的 RAG-Reasoning 资源库

---

## 📝 论文阅读建议

### 入门级（必读）

1. **综述 2**（清华）- 最全面的 RAG-Reasoning 综述
2. **综述 3**（阿里巴巴）- 系统性思考 RAG 的四大问题

### 进阶级

1. **综述 1**（同济）- RAG 与推理结合的深入探讨
2. **综述 4** - 从幻觉缓解角度理解 RAG、推理、Agent 三者关系

---

## 🎯 面试重点

### 高频问题

1. **RAG 如何与推理结合？**
   - 推理增强型 RAG：用推理优化检索过程
   - RAG 增强型推理：用检索知识辅助推理
   - 协同框架：迭代交替搜索和推理

2. **什么是 Agentic RAG？**
   - 智能体驱动的 RAG 系统
   - 能够自主分解复杂任务
   - 自适应决策何时检索、检索什么

3. **RAG 面临的主要挑战？**
   - 何时触发检索（知识边界问题）
   - 检索什么内容（复杂查询理解）
   - 信任什么数据（数据质量问题）
   - 如何工作（ICL 机制不明）

4. **如何缓解 LLM 幻觉？**
   - RAG：解决知识型幻觉
   - 推理增强：解决逻辑型幻觉
   - 智能体系统：综合优化
   - 注意各方案的风险权衡

---

## 📌 更新日志

- **2025.10** - 新增幻觉缓解技术综述
- **2025.10** - 新增阿里巴巴 RAG 系统性思考论文
- **2025.07** - 新增清华大学 RAG-Reasoning 综述
- **2025.04** - 新增同济大学 RAG 与推理综述

---

## 🙏 贡献

欢迎补充更多 Agentic RAG 相关论文！

**贡献要求**：
- 论文必须与 RAG-Reasoning 或 Agentic RAG 直接相关
- 提供论文链接和简要概括
- 标注核心观点和适用场景

---

**⭐ 如果这个资源对你有帮助，欢迎 Star 支持！**

---

<!-- AUTO-GENERATED-CONTENT:START -->
## 完整内容索引

<!-- 下表由元数据生成，请勿手工编辑。 -->

| 文档 | 类型 | 状态 | 难度 | 主题 |
|:---|:---|:---|:---|:---|
| [目录](./agentic_rag.md) | 论文清单 | 已发布 | 进阶 | RAG、科研 |
<!-- AUTO-GENERATED-CONTENT:END -->
