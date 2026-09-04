import fs from 'node:fs';
import { describe, expect, it } from 'vitest';

describe('question detail and progress routes', () => {
  it('has question detail route and progress route', () => {
    expect(fs.existsSync('src/pages/questions/[id].astro')).toBe(true);
    expect(fs.existsSync('src/pages/progress/index.astro')).toBe(true);
  });
});
