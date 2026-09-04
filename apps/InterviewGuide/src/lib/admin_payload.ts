export type AdminChangeType = 'add' | 'update' | 'delete' | 'other';
export type AdminPriority = 'low' | 'medium' | 'high';

export interface AdminIssueInput {
  title: string;
  company?: string;
  category?: string;
  questionId?: string;
  changeType?: AdminChangeType;
  priority?: AdminPriority;
  reason: string;
  proposedChange: string;
  references?: string;
  reporter?: string;
}

export interface AdminIssueMetadata {
  schemaVersion: 1;
  questionId: string;
  company: string;
  category: string;
  changeType: AdminChangeType;
  priority: AdminPriority;
  reporter: string;
}

export interface AdminIssuePayload {
  title: string;
  labels: string[];
  body: string;
  metadata: AdminIssueMetadata;
}

const DEFAULT_CHANGE_TYPE: AdminChangeType = 'add';
const DEFAULT_PRIORITY: AdminPriority = 'medium';

function clean(input: string | undefined): string {
  return (input ?? '').trim();
}

function normalizeChangeType(input: string | undefined): AdminChangeType {
  if (input === 'add' || input === 'update' || input === 'delete' || input === 'other') {
    return input;
  }
  return DEFAULT_CHANGE_TYPE;
}

function normalizePriority(input: string | undefined): AdminPriority {
  if (input === 'low' || input === 'medium' || input === 'high') {
    return input;
  }
  return DEFAULT_PRIORITY;
}

export function buildAdminIssuePayload(input: AdminIssueInput): AdminIssuePayload {
  const titleCore = clean(input.title) || '未命名变更';
  const questionId = clean(input.questionId) || 'unknown';
  const company = clean(input.company) || 'unknown';
  const category = clean(input.category) || 'unknown';
  const reason = clean(input.reason) || '未填写';
  const proposedChange = clean(input.proposedChange) || '未填写';
  const references = clean(input.references) || '无';
  const reporter = clean(input.reporter) || 'anonymous';
  const changeType = normalizeChangeType(clean(input.changeType));
  const priority = normalizePriority(clean(input.priority));

  const metadata: AdminIssueMetadata = {
    schemaVersion: 1,
    questionId,
    company,
    category,
    changeType,
    priority,
    reporter,
  };

  const title = `[Admin][${priority}] ${titleCore}`;
  const labels = ['admin-update', `priority:${priority}`, `change:${changeType}`];
  const body = [
    '### 背景与原因',
    reason,
    '',
    '### 期望变更',
    proposedChange,
    '',
    '### 参考资料',
    references,
    '',
    '### 结构化 Payload',
    '```json',
    JSON.stringify(metadata, null, 2),
    '```',
  ].join('\n');

  return {
    title,
    labels,
    body,
    metadata,
  };
}

export function toIssueQueryString(payload: AdminIssuePayload): string {
  const params = new URLSearchParams();
  params.set('title', payload.title);
  params.set('body', payload.body);
  params.set('labels', payload.labels.join(','));
  return params.toString();
}
