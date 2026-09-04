---
type: 资源清单
status: 已发布
level: 通用
topic:
  - 多模态
  - RAG
  - 框架工具
---

# 多模态 RAG 工具与项目清单

> 选型原则：先看你的核心难点是“解析复杂文档”，还是“检索视觉页面”，还是“统一管理多模态证据”。

## 文档解析

### Docling

- 链接：https://github.com/docling-project/docling
- 适合：PDF、Office、HTML、图片、音频等多格式文档进入 GenAI/RAG 系统。
- 亮点：支持 PDF 版面、阅读顺序、表格结构、OCR、Markdown/HTML/JSON 导出，并提供 LangChain、LlamaIndex、CrewAI、Haystack 等集成。
- 适合项目：论文图表问答、企业知识库、财报解析。

### MinerU

- 链接：https://github.com/opendatalab/MinerU
- 适合：复杂 PDF、Office 文档、扫描件、多语言 OCR。
- 亮点：面向 LLM/RAG/Agent workflow，输出结构化 Markdown/JSON，支持版面可视化和 span 可视化。
- 适合项目：中文文档知识库、扫描 PDF 解析、学术 PDF 批处理。

## 视觉文档检索

### ColPali / ColQwen 系列

- 论文：https://arxiv.org/abs/2407.01449
- 适合：视觉丰富文档的页面级检索，例如论文、财报、海报、表格密集 PDF。
- 亮点：把文档页面作为图像直接 embedding，用 late interaction 做匹配，减少传统 OCR + chunk 管线的脆弱性。
- 注意：通常更适合“找到相关页面”，后续仍需要 VLM 或解析工具生成可引用答案。

### ViDoRe

- 链接：https://huggingface.co/vidore
- 适合：评测视觉文档检索。
- 亮点：覆盖不同领域、语言和页面级任务，可以用来比较 text-first 与 vision-first 检索。

## 一体化多模态 RAG

### RAG-Anything

- 链接：https://github.com/HKUDS/RAG-Anything
- 适合：希望快速搭建文本、图片、表格、公式统一处理的多模态 RAG。
- 亮点：基于 LightRAG 生态，使用 MinerU 做高保真解析，提供多模态知识图谱和混合检索。
- 注意：一体化框架适合快速实验，但生产系统仍要理解每个阶段的中间数据和失败模式。

### LightRAG

- 链接：https://github.com/HKUDS/LightRAG
- 适合：轻量级 RAG、图增强检索、多模态扩展基础。
- 亮点：工程上容易上手，适合作为 RAG-Anything 生态背景阅读。

## Agent 与工作流

### LangGraph

- 链接：https://github.com/langchain-ai/langgraph
- 适合：多阶段文档处理、人工审核、失败恢复。
- 用法：把 parsing、enrichment、indexing、retrieval、answering、review 拆成状态图节点。

### OpenAI Agents SDK

- 链接：https://platform.openai.com/docs/guides/agents-sdk/
- 适合：需要 handoff、guardrails、tracing、sandbox 和 server-managed state 的应用。
- 用法：让文档解析、视觉理解、检索和引用检查成为不同 specialist。

## 选型建议

| 需求 | 推荐组合 |
|:---|:---|
| 论文 PDF 问答 | Docling + text/table index + VLM caption |
| 中文扫描件知识库 | MinerU + OCR + page/bbox 引用 |
| 视觉页面检索 | ColPali/ColQwen + VLM answerer |
| 快速多模态 RAG demo | RAG-Anything |
| 生产级文档 Agent | Docling/MinerU + LangGraph + eval harness |

## 不建议的做法

- 只用 OCR 文本，不保留页码和 bbox。
- 把表格压成普通段落后直接 embedding。
- 让 VLM 看完整 PDF，但不保存中间证据。
- 只测“普通问答”，不测图表、表格和数值。
- 对外输出没有引用，无法回到原文。
