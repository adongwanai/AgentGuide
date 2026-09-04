const TRAILING_COMPANY_COUNT_RE = /\s*等\d+家公司\s*$/u;

export function normalizeCompanyName(name: string): string {
  return name.replace(TRAILING_COMPANY_COUNT_RE, '').trim();
}

export function normalizeCompanyList(companies: string[]): string[] {
  const normalized: string[] = [];
  const seen = new Set<string>();

  for (const company of companies) {
    const cleaned = normalizeCompanyName(company);
    if (!cleaned || seen.has(cleaned)) continue;
    seen.add(cleaned);
    normalized.push(cleaned);
  }

  return normalized;
}
