import { describe, expect, it } from 'vitest';

import { fromCategorySlug, toCategorySlug } from '../../src/lib/path';

describe('category slug', () => {
  it('uses english slugs for known categories', () => {
    expect(toCategorySlug('项目与行为面试')).toBe('project-behavior');
    expect(toCategorySlug('nlp与大模型')).toBe('nlp-llm');
    expect(toCategorySlug('机器学习基础')).toBe('ml-fundamentals');
  });

  it('can resolve chinese category names from known slugs', () => {
    expect(fromCategorySlug('project-behavior')).toBe('项目与行为面试');
    expect(fromCategorySlug('nlp-llm')).toBe('nlp与大模型');
  });

  it('always returns ascii-only slugs', () => {
    const slug = toCategorySlug('计算机视觉');
    expect(/^[a-z0-9-]+$/.test(slug)).toBe(true);
  });
});
