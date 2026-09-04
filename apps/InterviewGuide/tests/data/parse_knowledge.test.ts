import { describe, expect, it } from 'vitest';
import { parseKnowledge } from '../../scripts/lib/parse_knowledge';

describe('parseKnowledge', () => {
  it('parses knowledge markdown into question rows', async () => {
    const rows = await parseKnowledge('tests/fixtures/knowledge_sample.md');

    expect(rows.length).toBeGreaterThan(0);
    expect(rows[0]).toHaveProperty('question');
    expect(rows[0]).toHaveProperty('memberCount');
    expect(rows[0]).toHaveProperty('categoryName');
  });
});
