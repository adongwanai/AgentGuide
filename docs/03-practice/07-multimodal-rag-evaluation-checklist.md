---
type: 实践指南
status: 已发布
level: 高阶
topic:
  - 多模态
  - RAG
  - 评测
---

# 多模态 RAG 评测清单

> 多模态系统最容易“看起来答得像”，但细查发现页码错、表格错列、图表解释幻觉。评测要逼它回到证据。

## Golden Case 设计

至少准备 40 条问题：

| 类型 | 数量 | 示例 |
|:---|:---:|:---|
| 文本事实 | 10 | 论文方法的核心假设是什么？ |
| 表格数值 | 10 | 表 2 中哪个方法在 X 指标最高？ |
| 图表解释 | 10 | 图 3 的趋势说明了什么？ |
| 跨模态推理 | 5 | 正文对图 4 的结论是否被实验表支持？ |
| 失败/安全 | 5 | 文档里没有的信息，系统是否会承认缺失？ |

每条 case 建议写成：

```json
{
  "id": "mmrag-001",
  "question": "表 2 中哪个方法在 F1 指标最高？",
  "required_evidence": [
    {"doc_id": "paper-001", "page": 6, "type": "table", "label": "Table 2"}
  ],
  "answer_key": "Method X",
  "must_include": ["F1", "Table 2", "page 6"],
  "forbidden": ["没有引用", "把 Accuracy 当成 F1"]
}
```

## 指标

| 指标 | 说明 |
|:---|:---|
| `answer_accuracy` | 最终答案是否正确 |
| `evidence_recall` | 是否找到了必要证据 |
| `citation_precision` | 引用是否真的支持结论 |
| `table_cell_accuracy` | 表格行列和单位是否正确 |
| `figure_grounding` | 是否定位到正确图/页 |
| `hallucination_rate` | 无证据断言占比 |
| `abstention_quality` | 证据不足时是否拒绝胡编 |
| `latency` | 文档解析、检索、生成耗时 |
| `cost` | VLM、embedding、rerank 成本 |

## 人工抽检要看什么

1. 答案里的每个数字能否在表格中找到。
2. 引用页码是否正确。
3. 图表解释是否来自图本身、caption 或正文，而不是模型猜测。
4. 是否混淆相似表格或相邻页面。
5. 对扫描件和低质量图片是否说明不确定性。

## 消融实验

建议至少做三组：

| 实验 | 对比 |
|:---|:---|
| Text-only vs Layout-aware | 普通 OCR/chunk 与 Docling/MinerU 结构化解析 |
| Text retrieval vs Vision retrieval | 文本 embedding 与 ColPali 页面检索 |
| No rerank vs Rerank | 只向量召回与二阶段 rerank |
| No citation check vs Citation check | 直接生成与引用一致性检查 |

## 失败归因模板

| Case ID | 失败类型 | 根因 | 修复 |
|:---|:---|:---|:---|
| mmrag-007 | 表格错列 | HTML 表头解析失败 | 保留原表格结构并加表头归一化 |
| mmrag-012 | 图表定位错 | caption 与图像分离 | 建 figure-caption 关系索引 |
| mmrag-021 | 编造答案 | 检索为空但仍生成 | evidence gate：无证据则拒答 |

## 最低通过标准

一个可写进简历的多模态 RAG 项目，建议至少达到：

- 40 条 golden case。
- 文本题准确率 > 80%。
- 表格/图表题准确率 > 65%。
- 引用准确率 > 85%。
- 所有失败 case 都有 trace 和截图/页码证据。
- 能解释至少 3 个失败类型和对应改进。
