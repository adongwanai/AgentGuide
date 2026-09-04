import { describe, expect, it } from 'vitest';
import { buildAdminIssuePayload, toIssueQueryString } from '../../src/lib/admin_payload';

describe('admin payload builder', () => {
  it('builds normalized structured payload', () => {
    const payload = buildAdminIssuePayload({
      title: '  修复 RAG 相关题目频次  ',
      company: '  字节跳动 ',
      category: ' ai系统设计 ',
      questionId: ' q-3 ',
      changeType: 'update',
      priority: 'high',
      reason: '线上反馈频次明显偏低',
      proposedChange: '将频次从 6 调整到 14，补充 2 个变体',
      references: 'https://example.com/evidence-1',
      reporter: 'maintainer',
    });

    expect(payload.title).toBe('[Admin][high] 修复 RAG 相关题目频次');
    expect(payload.labels).toEqual(['admin-update', 'priority:high', 'change:update']);
    expect(payload.metadata).toEqual({
      schemaVersion: 1,
      questionId: 'q-3',
      company: '字节跳动',
      category: 'ai系统设计',
      changeType: 'update',
      priority: 'high',
      reporter: 'maintainer',
    });
    expect(payload.body).toContain('### 背景与原因');
    expect(payload.body).toContain('### 期望变更');
    expect(payload.body).toContain('```json');
    expect(payload.body).toContain('"questionId": "q-3"');
  });

  it('uses fallback values for optional inputs', () => {
    const payload = buildAdminIssuePayload({
      title: '新增题目',
      reason: '补充近三个月高频',
      proposedChange: '新增 5 道多模态题',
    });

    expect(payload.metadata).toEqual({
      schemaVersion: 1,
      questionId: 'unknown',
      company: 'unknown',
      category: 'unknown',
      changeType: 'add',
      priority: 'medium',
      reporter: 'anonymous',
    });
    expect(payload.labels).toEqual(['admin-update', 'priority:medium', 'change:add']);
  });

  it('serializes payload into issue query string', () => {
    const payload = buildAdminIssuePayload({
      title: '修复标签',
      reason: '分类错误',
      proposedChange: '改为机器学习系统',
    });

    const query = toIssueQueryString(payload);
    const params = new URLSearchParams(query);

    expect(params.get('title')).toBe(payload.title);
    expect(params.get('body')).toBe(payload.body);
    expect(params.get('labels')).toBe(payload.labels.join(','));
  });
});
