import fs from 'node:fs/promises';

import { normalizeCompanyList } from './company_normalizer';
import type { ParsedQuestionRow } from './types';

interface MutableQuestion {
  question: string;
  memberCount: number;
  companies: string[];
  variants: string[];
}

const QUESTION_HEADER_RE = /^###\s+\d+\.\s+(?:🔥+\s*)?(.+)$/;
const CATEGORY_RE = /^##\s+(.+)$/;
const COMPANIES_RE = /^\*\*出现公司\*\*:\s*(.+)$/;
const COUNT_RE = /^\*\*出现次数\*\*:\s*(?:\*\*)?(\d+)(?:\*\*)?/;
const VARIANTS_RE = /^\*\*常见变体\*\*:/;

export async function parseKnowledge(filePath: string): Promise<ParsedQuestionRow[]> {
  const raw = await fs.readFile(filePath, 'utf-8');
  const lines = raw.split(/\r?\n/);

  let categoryName = 'uncertain';
  let categoryKey = 'uncertain';
  let readingVariants = false;
  let current: MutableQuestion | null = null;

  const rows: ParsedQuestionRow[] = [];

  const pushCurrent = () => {
    if (!current) return;
    const id = `knowledge-${rows.length + 1}`;
    rows.push({
      id,
      question: current.question,
      memberCount: current.memberCount,
      categoryName,
      categoryKey,
      companies: [...current.companies],
      variants: [...current.variants],
      source: 'knowledge',
    });
  };

  for (const line of lines) {
    const categoryMatch = line.match(CATEGORY_RE);
    if (categoryMatch && !line.startsWith('###')) {
      pushCurrent();
      current = null;
      readingVariants = false;
      categoryName = categoryMatch[1].trim();
      categoryKey = categoryName.toLowerCase();
      continue;
    }

    const questionMatch = line.match(QUESTION_HEADER_RE);
    if (questionMatch) {
      pushCurrent();
      current = {
        question: questionMatch[1].trim(),
        memberCount: 0,
        companies: [],
        variants: [],
      };
      readingVariants = false;
      continue;
    }

    if (!current) continue;

    const companiesMatch = line.match(COMPANIES_RE);
    if (companiesMatch) {
      current.companies = normalizeCompanyList(
        companiesMatch[1]
          .split(/[，,]/)
          .map((item) => item.trim())
          .filter(Boolean),
      );
      continue;
    }

    const countMatch = line.match(COUNT_RE);
    if (countMatch) {
      current.memberCount = Number.parseInt(countMatch[1], 10);
      continue;
    }

    if (line.match(VARIANTS_RE)) {
      readingVariants = true;
      continue;
    }

    if (readingVariants) {
      if (line.startsWith('- ')) {
        current.variants.push(line.slice(2).trim());
      } else if (line.trim() === '') {
        continue;
      } else {
        readingVariants = false;
      }
    }
  }

  pushCurrent();
  return rows;
}
