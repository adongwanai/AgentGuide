import { describe, expect, it } from 'vitest';

import { withBaseFrom } from '../../src/lib/path';

describe('withBaseFrom', () => {
  it('appends trailing slash for directory-like routes', () => {
    expect(withBaseFrom('/InterviewGuide/', '/companies')).toBe('/InterviewGuide/companies/');
    expect(withBaseFrom('/InterviewGuide/', '/companies/%E5%AD%97%E8%8A%82%E8%B7%B3%E5%8A%A8')).toBe(
      '/InterviewGuide/companies/%E5%AD%97%E8%8A%82%E8%B7%B3%E5%8A%A8/',
    );
    expect(withBaseFrom('/InterviewGuide/', '/categories/nlp%E4%B8%8E%E5%A4%A7%E6%A8%A1%E5%9E%8B')).toBe(
      '/InterviewGuide/categories/nlp%E4%B8%8E%E5%A4%A7%E6%A8%A1%E5%9E%8B/',
    );
  });

  it('keeps root and asset paths stable', () => {
    expect(withBaseFrom('/InterviewGuide/', '/')).toBe('/InterviewGuide/');
    expect(withBaseFrom('/InterviewGuide/', '/favicon.svg')).toBe('/InterviewGuide/favicon.svg');
  });
});
