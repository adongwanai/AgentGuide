import type { ParsedQuestionRow } from './types';
import {
  AI_CORE_CATEGORY_KEYS,
  AI_SYSTEM_DESIGN_KEYWORDS,
  AI_SYSTEM_DESIGN_SOURCE_KEYS,
} from './keywords';

export interface ScopedQuestionRow extends ParsedQuestionRow {
  aiSystemDesign: boolean;
}

const normalize = (value: string): string => value.trim().toLowerCase();

const matchesAiSystemDesign = (row: ParsedQuestionRow): boolean => {
  if (!AI_SYSTEM_DESIGN_SOURCE_KEYS.has(normalize(row.categoryKey))) {
    return false;
  }

  const corpus = `${row.question} ${row.variants.join(' ')}`.toLowerCase();
  return AI_SYSTEM_DESIGN_KEYWORDS.some((keyword) => corpus.includes(keyword.toLowerCase()));
};

export function filterAiScope(rows: ParsedQuestionRow[]): ScopedQuestionRow[] {
  const output: ScopedQuestionRow[] = [];

  for (const row of rows) {
    const key = normalize(row.categoryKey);

    if (AI_CORE_CATEGORY_KEYS.has(key)) {
      output.push({ ...row, categoryKey: key, aiSystemDesign: false });
      continue;
    }

    if (matchesAiSystemDesign(row)) {
      output.push({
        ...row,
        categoryName: 'AI系统设计',
        categoryKey: 'ai系统设计',
        aiSystemDesign: true,
      });
    }
  }

  return output;
}
