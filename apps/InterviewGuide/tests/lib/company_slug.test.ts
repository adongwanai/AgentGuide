import { describe, expect, it } from 'vitest';

import { toCompanySlug } from '../../src/lib/path';

describe('company slug', () => {
  it('uses english slugs for known company names', () => {
    expect(toCompanySlug('字节跳动')).toBe('bytedance');
    expect(toCompanySlug('美团')).toBe('meituan');
    expect(toCompanySlug('阿里巴巴')).toBe('alibaba');
  });

  it('maps noisy name with 等N家公司 suffix to canonical english slug', () => {
    expect(toCompanySlug('字节跳动 等8家公司')).toBe('bytedance');
  });

  it('returns ascii-only slug', () => {
    const slug = toCompanySlug('小红书');
    expect(/^[a-z0-9-]+$/.test(slug)).toBe(true);
  });
});
