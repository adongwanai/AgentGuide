import { describe, expect, it } from 'vitest';

import { normalizeCompanyList, normalizeCompanyName } from '../../scripts/lib/company_normalizer';

describe('company normalization', () => {
  it('normalizes suffix pattern like 等N家公司 to base company name', () => {
    expect(normalizeCompanyName('字节跳动 等6家公司')).toBe('字节跳动');
    expect(normalizeCompanyName('美团 等12家公司')).toBe('美团');
    expect(normalizeCompanyName('  阿里巴巴 等7家公司  ')).toBe('阿里巴巴');
  });

  it('deduplicates company list after normalization', () => {
    const companies = normalizeCompanyList(['字节跳动', '字节跳动 等6家公司', '美团', '美团 等7家公司']);
    expect(companies).toEqual(['字节跳动', '美团']);
  });
});
