---
type: 入口页
status: 已发布
level: 通用
topic:
  - 多模态
  - RAG
---

# Multimodal RAG 与文档智能资源

> 目标：把 PDF、扫描件、图表、表格、公式、图片和网页等混合内容，变成 Agent/RAG 系统可检索、可引用、可评测的知识。

传统 RAG 往往默认“文档 = 文本”。但真实知识库里大量信息藏在版面、表格、图像、公式、截图和扫描件里。多模态 RAG 的核心不是简单“接一个视觉模型”，而是设计一条稳定管线：解析 -> 对齐 -> 索引 -> 检索 -> 生成 -> 引用 -> 评测。

## 推荐先读

| 顺序 | 文档 | 解决什么问题 |
|:---:|:---|:---|
| 1 | [多模态 RAG 管线](../../docs/02-tech-stack/28-multimodal-rag-pipeline.md) | 从文件进入系统到答案生成的完整链路 |
| 2 | [工具与项目清单](./tools-and-projects.md) | Docling、MinerU、ColPali、RAG-Anything 等怎么选 |
| 3 | [评测清单](../../docs/03-practice/07-multimodal-rag-evaluation-checklist.md) | 怎么证明系统真的读懂了图表、表格和版面 |

## 一张图看管线

![Multimodal RAG Pipeline](./multimodal-rag-pipeline.svg)

## 适合做的项目

### 项目 1：论文图表问答

输入一篇论文 PDF，用户问“表 2 说明了什么”“图 3 的方法比 baseline 好在哪里”。系统需要定位图表、抽取 caption、关联正文段落，并给出 page/figure/table 引用。

### 项目 2：财报多模态 RAG

输入年报、财报 PDF 和表格，支持问“营收增长来自哪个业务”“现金流为什么变化”。重点是表格抽取、图表解释、来源引用和数值一致性。

### 项目 3：产品手册 Agent

输入图文混排手册，支持排障问答。重点是步骤图、警告框、零件图和版本差异。

## 技术路线

| 路线 | 适合场景 | 代表工具 |
|:---|:---|:---|
| Text-first | 文档主要是正文，图表不是核心证据 | PyMuPDF、Docling、LlamaIndex |
| Layout-aware | PDF 版面复杂，需要保留阅读顺序和表格 | Docling、MinerU |
| Vision retrieval | 视觉线索重要，直接检索页面图像 | ColPali / ColQwen 系列 |
| All-in-one multimodal | 文本、图片、表格、公式都要统一处理 | RAG-Anything、LightRAG 生态 |
| Agentic document QA | 需要多步定位、验证、引用和报告 | LangGraph / OpenAI Agents SDK + 文档工具 |

## 关键原则

1. **保留坐标和页码**：没有 page、bbox、section，答案很难追溯。
2. **不要过早丢图**：图表、公式和版面本身可能就是证据。
3. **表格要结构化**：表格转纯文本很容易丢列关系和单位。
4. **引用粒度要细**：至少能回到 page；最好能回到 figure/table/section。
5. **评测要覆盖视觉问题**：只测普通文本问答，会高估系统能力。

## 外部高质量资料

- [Docling](https://github.com/docling-project/docling)：文档解析和 PDF 理解工具，支持多格式、版面、表格、OCR 和 GenAI 集成。
- [MinerU](https://github.com/opendatalab/MinerU)：面向 LLM/RAG/Agent 的高精度文档解析引擎，输出 Markdown/JSON。
- [ColPali](https://arxiv.org/abs/2407.01449)：用视觉语言模型直接对文档页面图像做多向量检索。
- [RAG-Anything](https://github.com/HKUDS/RAG-Anything)：一体化多模态文档 RAG 框架，覆盖文本、图片、表格、公式等。
- [ViDoRe Benchmark](https://huggingface.co/vidore)：视觉文档检索评测集。

---

<!-- AUTO-GENERATED-CONTENT:START -->
## 完整内容索引

<!-- 下表由元数据生成，请勿手工编辑。 -->

| 文档 | 类型 | 状态 | 难度 | 主题 |
|:---|:---|:---|:---|:---|
| [多模态 RAG 工具与项目清单](./tools-and-projects.md) | 资源清单 | 已发布 | 通用 | 多模态、RAG、框架工具 |
<!-- AUTO-GENERATED-CONTENT:END -->
