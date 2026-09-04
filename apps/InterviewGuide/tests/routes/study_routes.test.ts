import fs from 'node:fs';
import { describe, expect, it } from 'vitest';

describe('study routes', () => {
  it('has hot and category routes', () => {
    expect(fs.existsSync('src/pages/hot/index.astro')).toBe(true);
    expect(fs.existsSync('src/pages/categories/index.astro')).toBe(true);
    expect(fs.existsSync('src/pages/categories/[category].astro')).toBe(true);
  });
});
