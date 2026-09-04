import { describe, expect, it } from 'vitest';
import { filterAiScope } from '../../scripts/lib/filter_ai_scope';
import type { ParsedQuestionRow } from '../../scripts/lib/types';

const inputRows: ParsedQuestionRow[] = [
  {
    id: 'q1',
    question: 'Transformer 结构',
    memberCount: 50,
    categoryName: 'NLP与大模型',
    categoryKey: 'nlp与大模型',
    companies: ['腾讯'],
    variants: [],
    source: 'knowledge',
  },
  {
    id: 'q2',
    question: 'tcp 三次握手',
    memberCount: 57,
    categoryName: 'uncertain',
    categoryKey: 'uncertain',
    companies: ['腾讯'],
    variants: [],
    source: 'knowledge',
  },
  {
    id: 'q3',
    question: 'RAG 系统架构怎么设计',
    memberCount: 12,
    categoryName: 'system_design',
    categoryKey: 'system_design',
    companies: ['阿里巴巴'],
    variants: ['LLM 检索增强架构'],
    source: 'knowledge',
  },
];

const ALLOWED = new Set([
  'nlp与大模型',
  '机器学习基础',
  '深度学习',
  '推荐系统',
  '计算机视觉',
  '机器学习系统',
  '项目与行为面试',
  '编程与算法',
]);

describe('filterAiScope', () => {
  it('keeps AI core categories and AI system design keywords only', () => {
    const output = filterAiScope([...inputRows]);

    expect(output.length).toBe(2);
    expect(output.every((r) => ALLOWED.has(r.categoryKey) || r.aiSystemDesign === true)).toBe(true);
  });
});
