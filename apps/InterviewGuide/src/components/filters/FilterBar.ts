export interface QuestionListItem {
  id: string;
  title: string;
  frequency: number;
  categoryKey: string;
  companies: string[];
}

export interface QuestionFilterOptions {
  keyword?: string;
  minFrequency?: number;
  company?: string;
  categoryKey?: string;
}

export function applyQuestionFilters<T extends QuestionListItem>(
  questions: T[],
  options: QuestionFilterOptions = {},
): T[] {
  const keyword = options.keyword?.trim().toLowerCase() ?? '';
  const minFrequency = options.minFrequency ?? 0;
  const company = options.company?.trim();
  const categoryKey = options.categoryKey?.trim();

  return questions.filter((question) => {
    if (question.frequency < minFrequency) return false;
    if (company && !question.companies.includes(company)) return false;
    if (categoryKey && question.categoryKey !== categoryKey) return false;
    if (!keyword) return true;
    return question.title.toLowerCase().includes(keyword);
  });
}
