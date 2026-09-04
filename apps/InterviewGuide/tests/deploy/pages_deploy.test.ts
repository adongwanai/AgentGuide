import fs from 'node:fs';
import { describe, expect, it } from 'vitest';

describe('github pages deployment setup', () => {
  it('has deploy workflow and deployment docs', () => {
    expect(fs.existsSync('.github/workflows/deploy.yml')).toBe(true);
    expect(fs.existsSync('README.md')).toBe(true);
  });

  it('configures astro base/site for github pages builds', () => {
    const config = fs.readFileSync('astro.config.mjs', 'utf-8');

    expect(config).toContain('GITHUB_ACTIONS');
    expect(config).toContain('GITHUB_REPOSITORY');
    expect(config).toContain('base');
    expect(config).toContain('site');
  });

  it('documents github pages deployment steps in README', () => {
    const readme = fs.readFileSync('README.md', 'utf-8');

    expect(readme).toContain('GitHub Pages');
    expect(readme).toContain('deploy.yml');
    expect(readme).toContain('npm run build');
  });
});
