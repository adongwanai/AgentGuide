import fs from 'node:fs';
import { describe, expect, it } from 'vitest';

describe('task schema', () => {
  it('contains required workflow fields', () => {
    const raw = fs.readFileSync('task.json', 'utf-8');
    const json = JSON.parse(raw) as { tasks: Array<Record<string, unknown>> };

    expect(Array.isArray(json.tasks)).toBe(true);
    expect(json.tasks[0]).toHaveProperty('id');
    expect(json.tasks[0]).toHaveProperty('title');
    expect(json.tasks[0]).toHaveProperty('status');
    expect(json.tasks[0]).toHaveProperty('steps');
  });
});
