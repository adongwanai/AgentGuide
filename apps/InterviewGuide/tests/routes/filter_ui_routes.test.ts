import fs from 'node:fs';
import { describe, expect, it } from 'vitest';

describe('filter UI routes', () => {
  it('has hot page company/category/keyword filter controls', () => {
    const source = fs.readFileSync('src/pages/hot/index.astro', 'utf8');
    expect(source.includes('id="hot-company-filter"')).toBe(true);
    expect(source.includes('id="hot-category-filter"')).toBe(true);
    expect(source.includes('id="hot-keyword-search"')).toBe(true);
  });

  it('has company page category filter control', () => {
    const source = fs.readFileSync('src/pages/companies/[company].astro', 'utf8');
    expect(source.includes('id="company-category-filter"')).toBe(true);
  });
});
