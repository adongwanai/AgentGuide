import fs from 'node:fs';
import { describe, expect, it } from 'vitest';

describe('companies routes', () => {
  it('has companies routes', () => {
    expect(fs.existsSync('src/pages/companies/index.astro')).toBe(true);
    expect(fs.existsSync('src/pages/companies/[company].astro')).toBe(true);
  });
});
