import fs from 'node:fs/promises';
import path from 'node:path';

import { filterAiScope } from './lib/filter_ai_scope';
import { parseCompany } from './lib/parse_company';
import { parseKnowledge } from './lib/parse_knowledge';

interface CompanyAgg {
  name: string;
  questionCount: number;
  highFrequencyCount: number;
  questionIds: string[];
}

interface CategoryAgg {
  key: string;
  name: string;
  questionCount: number;
  highFrequencyCount: number;
  questionIds: string[];
}

const HIGH_FREQUENCY_THRESHOLD = 5;

function resolveArg(args: string[], key: string, fallback: string): string {
  const index = args.indexOf(key);
  if (index >= 0 && args[index + 1]) {
    return args[index + 1];
  }
  return fallback;
}

async function main() {
  const root = process.cwd();
  const args = process.argv.slice(2);

  const knowledgePath = resolveArg(
    args,
    '--knowledge',
    path.join(root, '..', 'result', 'output', 'knowledge_summary.md'),
  );

  const companyPath = resolveArg(
    args,
    '--company',
    path.join(root, '..', 'result', 'output', 'company_summary.md'),
  );

  const [knowledgeRows, companyRows] = await Promise.all([
    parseKnowledge(knowledgePath),
    parseCompany(companyPath),
  ]);

  const scopedRows = filterAiScope(knowledgeRows);
  const outputDir = path.join(root, 'src', 'data');
  await fs.mkdir(outputDir, { recursive: true });

  const questions = scopedRows.map((row, index) => ({
    id: `q-${index + 1}`,
    title: row.question,
    frequency: row.memberCount,
    categoryKey: row.categoryKey,
    categoryName: row.categoryName,
    companies: row.companies,
    variants: row.variants,
    isHighFrequency: row.memberCount >= HIGH_FREQUENCY_THRESHOLD,
    aiSystemDesign: row.aiSystemDesign,
    source: row.source,
  }));

  const companyMap = new Map<string, CompanyAgg>();
  for (const question of questions) {
    for (const company of question.companies) {
      if (!companyMap.has(company)) {
        companyMap.set(company, {
          name: company,
          questionCount: 0,
          highFrequencyCount: 0,
          questionIds: [],
        });
      }

      const item = companyMap.get(company)!;
      item.questionCount += 1;
      if (question.isHighFrequency) item.highFrequencyCount += 1;
      item.questionIds.push(question.id);
    }
  }

  const categoryMap = new Map<string, CategoryAgg>();
  for (const question of questions) {
    if (!categoryMap.has(question.categoryKey)) {
      categoryMap.set(question.categoryKey, {
        key: question.categoryKey,
        name: question.categoryName,
        questionCount: 0,
        highFrequencyCount: 0,
        questionIds: [],
      });
    }

    const item = categoryMap.get(question.categoryKey)!;
    item.questionCount += 1;
    if (question.isHighFrequency) item.highFrequencyCount += 1;
    item.questionIds.push(question.id);
  }

  const companies = [...companyMap.values()].sort((a, b) => b.questionCount - a.questionCount);
  const categories = [...categoryMap.values()].sort((a, b) => b.questionCount - a.questionCount);

  const stats = {
    totalQuestions: questions.length,
    totalCompanies: companies.length,
    totalCategories: categories.length,
    highFrequencyQuestions: questions.filter((q) => q.isHighFrequency).length,
    parsedCompanyRows: companyRows.length,
  };

  const searchIndex = questions.map((question) => ({
    id: question.id,
    q: question.title,
    c: question.categoryKey,
    f: question.frequency,
    co: question.companies,
  }));

  await Promise.all([
    fs.writeFile(path.join(outputDir, 'questions.json'), `${JSON.stringify(questions, null, 2)}\n`, 'utf-8'),
    fs.writeFile(path.join(outputDir, 'companies.json'), `${JSON.stringify(companies, null, 2)}\n`, 'utf-8'),
    fs.writeFile(path.join(outputDir, 'categories.json'), `${JSON.stringify(categories, null, 2)}\n`, 'utf-8'),
    fs.writeFile(path.join(outputDir, 'stats.json'), `${JSON.stringify(stats, null, 2)}\n`, 'utf-8'),
    fs.writeFile(path.join(outputDir, 'search-index.json'), `${JSON.stringify(searchIndex, null, 2)}\n`, 'utf-8'),
  ]);

  console.log(`knowledge rows: ${knowledgeRows.length}`);
  console.log(`scoped rows: ${scopedRows.length}`);
  console.log(`company rows: ${companyRows.length}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
