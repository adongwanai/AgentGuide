# Paper Agent：论文研读与综述助手

> 项目目标：构建一个能检索论文、阅读 PDF、抽取贡献、对比方法并生成可追溯综述的 Agent。

这个项目适合想做算法岗、RAG 岗、科研助手或论文写作工具的同学。它的价值不在于“总结论文”四个字，而在于完整覆盖：学术检索、PDF 解析、证据引用、表格对比、长期笔记和评测。

## 为什么值得做

论文任务天然适合 Agent：

- 问题开放，需要多步检索和筛选。
- 资料来源多，包括 arXiv、Semantic Scholar、OpenAlex、Crossref、PDF。
- 输出必须可追溯，不能只靠模型记忆。
- 很容易设计 eval：是否找到正确论文、是否引用原文、是否区分贡献与局限。

## MVP 范围

| 能力 | MVP 做法 | 进阶做法 |
|:---|:---|:---|
| 论文检索 | arXiv API + Semantic Scholar / OpenAlex 元数据 | 多源融合、引用网络扩展 |
| PDF 解析 | PyMuPDF / Docling 抽文本 | 版面结构、表格、公式、图像说明 |
| 摘要生成 | 按贡献、方法、实验、局限输出 | 多论文对比矩阵 |
| 证据引用 | 每条结论附 paper url / page / section | 支持原文 quote 和截图证据 |
| 笔记沉淀 | Markdown artifact | SQLite + 向量索引 |
| 评测 | 20 条固定 query | 引用准确率、覆盖率、幻觉率 |

## 推荐架构

```text
User Query
  -> Planner
  -> Paper Search Tools
      - arXiv
      - Semantic Scholar
      - OpenAlex
  -> Paper Ranker
  -> PDF Reader
  -> Evidence Store
  -> Synthesis Agent
  -> Review Agent
  -> Markdown / BibTeX / Eval Report
```

## 核心工具设计

| 工具 | 输入 | 输出 | 注意点 |
|:---|:---|:---|:---|
| `search_papers` | query、年份、数量 | 论文元数据列表 | 多源结果要去重 |
| `expand_citations` | paper_id | 引用/被引论文 | 防止无限扩展，限制深度 |
| `download_pdf` | pdf_url | 本地路径、hash | 只下载公开可访问 PDF |
| `parse_pdf` | path | section、page、text | 保留页码和 section |
| `extract_claims` | paper_text | 贡献、方法、数据集、指标 | 每条 claim 必须带证据 |
| `compare_papers` | claims[] | 对比表 | 避免把不同任务指标硬比较 |
| `write_review` | evidence | Markdown 综述 | 明确不确定性和缺失证据 |

## 数据源

优先使用一手或结构化数据：

- [arXiv API](https://info.arxiv.org/help/api/index.html)：适合检索预印本和获取 PDF。
- [Semantic Scholar API](https://api.semanticscholar.org/api-docs/)：适合引用、作者、摘要和影响力信息。
- [OpenAlex API](https://docs.openalex.org/)：开放学术图谱，适合元数据和引用关系。
- [Crossref REST API](https://www.crossref.org/documentation/retrieve-metadata/rest-api/)：适合 DOI 和出版元数据。
- [Docling](https://github.com/docling-project/docling)：适合复杂 PDF 转结构化文档。

## 输出格式

建议默认输出一个目录：

```text
outputs/paper-agent/<run-id>/
├── review.md
├── papers.bib
├── evidence.jsonl
├── comparison.csv
├── trace.jsonl
└── eval_summary.md
```

`review.md` 推荐结构：

```markdown
# 主题综述

## 结论先行

## 代表论文

| Paper | Year | Core Idea | Evidence |
|:---|:---:|:---|:---|

## 方法谱系

## 实验设置对比

## 局限与开放问题

## 延伸阅读
```

## Eval 设计

### Case 类型

| 类型 | 示例 | 评分重点 |
|:---|:---|:---|
| 主题检索 | 找 agentic RAG 代表论文 | 覆盖率、相关性 |
| 单篇精读 | 总结某 PDF 的方法和实验 | 引用准确率 |
| 多篇对比 | 比较 ColBERT、ColPali、RAG-Anything | 对比维度是否公平 |
| 失败处理 | 搜不到或 PDF 解析失败 | 是否诚实说明限制 |
| 安全边界 | 用户要求绕过付费墙 | 是否拒绝 |

### 指标

- `paper_relevance@k`：前 k 篇是否相关。
- `citation_precision`：结论引用是否真的支持该说法。
- `coverage_score`：是否覆盖方法、数据、实验、局限。
- `hallucination_rate`：无证据断言占比。
- `avg_cost`、`avg_latency`、`avg_steps`。

## 实现里程碑

### Milestone 1：检索 + 综述

- 接入 arXiv API。
- 根据 query 返回 5 篇论文。
- 生成带链接的简短综述。

### Milestone 2：PDF 解析 + 证据

- 下载公开 PDF。
- 解析 section 和 page。
- 每条结论带页码或 section。

### Milestone 3：多源检索 + 对比

- 接入 Semantic Scholar 或 OpenAlex。
- 做去重和排序。
- 输出对比表。

### Milestone 4：评测 + 复盘

- 写 20 条 eval case。
- 保存 trace。
- 输出失败类型统计。

## 可选技术栈

| 层 | 推荐 |
|:---|:---|
| Agent loop | LangGraph / OpenAI Agents SDK / Pydantic AI |
| PDF 解析 | PyMuPDF / Docling / MinerU |
| 检索 | arXiv API / Semantic Scholar / OpenAlex |
| 存储 | SQLite + JSONL |
| 向量索引 | Qdrant / Chroma / FAISS |
| Eval | Promptfoo / DeepEval / 自定义 pytest |

## 简历表达

> 构建论文研读 Agent，接入 arXiv、Semantic Scholar 和 PDF 解析工具，使用分层 context builder 管理 query、候选论文、证据片段和生成草稿；为每条综述结论保留 paper/page/section 引用，设计 20 条主题检索和单篇精读 eval case，统计相关性、引用准确率、幻觉率和平均成本。

## 社区实现

- [agent-interview](https://github.com/11XuX/agent-interview)：按本蓝图实现的完整项目，同一业务用三种 Agent 形态各写一遍（写死的 workflow / 裸 ReAct / deepagents harness），附三形态对照跑分数据。

  - 架构与本蓝图一一对应：Planner → Paper Search Tools → Paper Ranker → PDF Reader → Evidence Store → Synthesis Agent → Review Agent
  - 数据源用 Europe PMC 的 `fullTextXML` 接口，直接返回带 `<sec><title>` 的结构化全文，**免去 PDF 解析**（PyMuPDF / Docling 那一层可以整块跳过）
  - 21 个 commit 一步一个 LangGraph 概念，可 `git checkout` 逐步走，适合当 LangGraph 入门材料
  - 实测结论：文献密集的题上写死的 workflow 引用密度是 agent 形态的 2 倍以上；交叉领域的题上 workflow 会撞死循环上限并**静默降级**（交出看似正常实则敷衍的短报告），agent 形态则能自适应但 token 成本随轮次二次增长

## 参考项目与资料

- [GPT Researcher](https://github.com/assafelovic/gpt-researcher)：开源深度研究 Agent，适合学习 research report pipeline。
- [Sakana AI: The AI Scientist](https://github.com/SakanaAI/AI-Scientist)：自动化科研流程项目，适合了解研究型 Agent 的边界和风险。
- [OpenScholar](https://github.com/stanford-oval/storm)：学术写作和综述生成方向的代表项目之一。
- [arXiv API](https://info.arxiv.org/help/api/index.html)
- [Semantic Scholar API](https://api.semanticscholar.org/api-docs/)
- [OpenAlex API](https://docs.openalex.org/)
