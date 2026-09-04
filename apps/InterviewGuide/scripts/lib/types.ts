export interface ParsedQuestionRow {
  id: string;
  question: string;
  memberCount: number;
  categoryName: string;
  categoryKey: string;
  companies: string[];
  variants: string[];
  source: 'knowledge' | 'company';
}

export interface ParsedCompanyRow {
  id: string;
  company: string;
  question: string;
  memberCount: number;
  source: 'company';
}
