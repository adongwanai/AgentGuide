import fs from 'node:fs';
import { describe, expect, it } from 'vitest';

describe('admin routes and templates', () => {
  it('has admin page and issue template', () => {
    expect(fs.existsSync('src/pages/admin/index.astro')).toBe(true);
    expect(fs.existsSync('.github/ISSUE_TEMPLATE/admin-update.yml')).toBe(true);
  });
});
