import { describe, expect, it } from 'vitest';

import { applyQuestionFilters } from '../../src/components/filters/FilterBar';

const questions = [
  {
    id: 'q1',
    title: 'Transformer 注意力机制',
    frequency: 10,
    categoryKey: 'nlp',
    companies: ['字节跳动', '美团'],
  },
  {
    id: 'q2',
    title: '推荐系统冷启动',
    frequency: 8,
    categoryKey: 'rec',
    companies: ['美团'],
  },
  {
    id: 'q3',
    title: 'CNN 基础',
    frequency: 6,
    categoryKey: 'cv',
    companies: ['字节跳动'],
  },
];

describe('applyQuestionFilters', () => {
  it('supports combined company and category filtering', () => {
    const result = applyQuestionFilters(questions, { company: '字节跳动', categoryKey: 'nlp' });
    expect(result.map((item) => item.id)).toEqual(['q1']);
  });

  it('keeps keyword and minFrequency behavior', () => {
    const result = applyQuestionFilters(questions, { keyword: '推荐', minFrequency: 7 });
    expect(result.map((item) => item.id)).toEqual(['q2']);
  });
});
