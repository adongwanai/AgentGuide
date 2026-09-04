import fs from 'node:fs/promises';

import { normalizeCompanyName } from './company_normalizer';
import type { ParsedCompanyRow } from './types';

const COMPANY_RE = /^##\s+(.+)$/;
const QUESTION_RE = /^####\s+\d+\.\s+(.+)$/;
const COUNT_RE = /^\*\*出现次数\*\*:\s*(?:\*\*)?(\d+)(?:\*\*)?/;

export async function parseCompany(filePath: string): Promise<ParsedCompanyRow[]> {
  const raw = await fs.readFile(filePath, 'utf-8');
  const lines = raw.split(/\r?\n/);

  let company = '';
  let question = '';
  const rows: ParsedCompanyRow[] = [];

  for (const line of lines) {
    const companyMatch = line.match(COMPANY_RE);
    if (companyMatch) {
      company = normalizeCompanyName(companyMatch[1].trim());
      continue;
    }

    const questionMatch = line.match(QUESTION_RE);
    if (questionMatch) {
      question = questionMatch[1].trim();
      continue;
    }

    const countMatch = line.match(COUNT_RE);
    if (countMatch && company && question) {
      rows.push({
        id: `company-${rows.length + 1}`,
        company,
        question,
        memberCount: Number.parseInt(countMatch[1], 10),
        source: 'company',
      });
      question = '';
    }
  }

  return rows;
}
