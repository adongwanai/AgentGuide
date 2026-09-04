---
type: 教程
status: 已发布
level: 进阶
topic:
  - 多模态
  - RAG
---

# 多模态 RAG 管线

> 多模态 RAG 的关键不是“把图片也喂给模型”，而是让每一种证据都有结构、位置、引用和评测。

## 1. Ingestion：输入与文件治理

输入可能包括：

- PDF、DOCX、PPTX、XLSX
- 扫描件和图片
- HTML 网页
- 论文、财报、产品手册、合同、教学课件

入库时先保存：

```json
{
  "doc_id": "annual-report-2025",
  "source_path": "data/raw/report.pdf",
  "source_url": "https://...",
  "sha256": "...",
  "mime_type": "application/pdf",
  "created_at": "2026-05-29T00:00:00Z"
}
```

不要只保存抽取后的文本。原文件、hash、页码和解析版本都要保留，否则后续无法复现。

## 2. Parsing：结构化解析

解析层负责把复杂文档拆成可处理单元：

| 元素 | 需要保留 |
|:---|:---|
| 正文段落 | page、section、reading_order、text |
| 表格 | page、bbox、header、rows、caption、footnote |
| 图片/图表 | page、bbox、caption、surrounding_text、image_path |
| 公式 | LaTeX、page、上下文 |
| 标题层级 | h1/h2/h3、页码、父子关系 |

推荐工具：

- Docling：适合多格式解析、PDF 版面理解、表格、OCR 和 Markdown/JSON 导出。
- MinerU：适合复杂 PDF、Office 文档、OCR、多语言和 Agent/RAG 工作流。

## 3. Enrichment：多模态补充理解

解析出来的图片、图表和表格通常还需要二次处理：

- 图像 caption：用 VLM 生成简短描述。
- 图表转表格：从柱状图、折线图提取类别、数值、趋势。
- 表格摘要：保留列名、单位、关键行和异常值。
- 公式解释：把公式变量和上下文段落对齐。
- 跨模态关系：把图 3、caption、正文引用段落连起来。

注意：VLM 生成的 caption 不是原始事实，应该标注 `generated=true`，并保留模型和时间。

## 4. Indexing：多索引并存

不要强迫所有内容进入同一个向量索引。更稳的设计是多索引：

| 索引 | 存什么 | 用途 |
|:---|:---|:---|
| Text index | 段落、标题、caption、表格摘要 | 普通语义检索 |
| Layout index | page、bbox、section、reading_order | 精确定位和引用 |
| Table index | 结构化表格和列信息 | 数值问答 |
| Image/page index | 页面截图或图像 embedding | 视觉检索 |
| Graph index | 文档、章节、图表、实体关系 | 跨模态关联 |

ColPali 这类路线的价值在于：对视觉丰富文档，可以直接把页面图像作为检索对象，不必完全依赖脆弱的 OCR/版面抽取。

## 5. Retrieval：检索与路由

一个简单但实用的路由：

```text
用户问题
  -> 判断问题类型
     -> 纯文本事实：text index
     -> 表格/数值：table index + text context
     -> 图/版面/截图：image/page index + caption
     -> 跨章节综述：text index + graph expansion
  -> rerank
  -> evidence pack
```

Evidence pack 建议格式：

```json
{
  "question": "图 3 说明了什么？",
  "evidence": [
    {
      "doc_id": "paper-001",
      "page": 5,
      "type": "figure",
      "label": "Figure 3",
      "caption": "...",
      "surrounding_text": "...",
      "bbox": [72, 120, 520, 430],
      "image_path": "artifacts/paper-001/page-5-figure-3.png"
    }
  ]
}
```

## 6. Generation：带证据生成

生成层必须遵守：

- 每个关键结论都有证据编号。
- 数值保留单位和来源。
- 图表解释区分“原文 caption”和“模型推断”。
- 找不到证据时明确说明，不要补脑。
- 输出能跳回 page / figure / table。

## 7. Evaluation：评测

多模态 RAG 至少要测四类问题：

- 文本事实：答案是否被正文支持。
- 表格数值：列、行、单位、计算是否正确。
- 图像/图表：是否正确定位图表并解释。
- 跨模态：是否能把正文引用、caption、图表和公式连起来。

不要只用 “LLM judge 打分”。建议人工抽检一小批 golden cases，尤其是数值和图表问题。

## 常见失败模式

| 失败 | 原因 | 修复 |
|:---|:---|:---|
| 表格答案错列 | 表格转文本丢结构 | 保留 HTML/CSV 和列 schema |
| 引用页码错 | chunk 与 page 映射丢失 | 每个 chunk 都带 page/bbox |
| 图表解释幻觉 | caption 是模型生成且未校验 | 保留原图、caption、周围正文并做交叉验证 |
| 检索不到图 | 只建文本索引 | 加页面图像 embedding 或图表 caption 索引 |
| 复杂 PDF 顺序混乱 | 阅读顺序解析失败 | 用 Docling/MinerU 的 layout 结果人工抽检 |

## 最小项目切法

1. 选 10 篇论文或 10 份财报。
2. 用 Docling 或 MinerU 解析成 Markdown/JSON。
3. 保留 page、section、table、figure 元数据。
4. 建 text index 和 table index。
5. 写 20 条 eval case，其中 8 条必须依赖表格/图表。
6. 输出 `eval_report.md`，统计文本题、表格题、图表题的通过率。
