# InterviewGuide V1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 交付 InterviewGuide V1（公司定向默认入口 + 高频/知识点学习 + 题目详情 + 进度看板 + 管理更新入口），并落地 `AGENT.MD`/`task.json`/`progress.txt` 持久化开发流程。

**Architecture:** 使用 Astro 静态站 + 构建期数据转换脚本。题库数据来自 `result/output`，通过脚本过滤出 AI 核心范围并生成 JSON 索引。前端用客户端脚本处理筛选与学习进度，本地存储用户状态；管理员入口先用弱门禁并输出结构化更新任务给 GitHub 自动化流程。

**Tech Stack:** Astro 5, TailwindCSS, TypeScript, Vitest, Node.js scripts, GitHub Actions。

---

### Task 1: 建立测试与脚本基础设施

**Files:**
- Modify: `package.json`
- Create: `vitest.config.ts`
- Create: `tests/smoke/basic.test.ts`
- Create: `tests/fixtures/.gitkeep`

**Step 1: 写失败测试（测试环境先不存在）**

```ts
// tests/smoke/basic.test.ts
import { describe, expect, it } from 'vitest';

describe('smoke', () => {
  it('runs test runner', () => {
    expect(1 + 1).toBe(2);
  });
});
```

**Step 2: 运行测试验证失败**

Run: `npm run test`
Expected: FAIL with `Missing script: test`.

**Step 3: 最小实现（测试脚本与配置）**

```json
// package.json (scripts)
{
  "scripts": {
    "test": "vitest run",
    "test:watch": "vitest"
  }
}
```

```ts
// vitest.config.ts
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'node',
    include: ['tests/**/*.test.ts'],
  },
});
```

**Step 4: 安装依赖并验证通过**

Run: `npm i -D vitest @types/node`  
Run: `npm run test`
Expected: PASS with `1 passed`.

**Step 5: Commit**

```bash
git add package.json vitest.config.ts tests/smoke/basic.test.ts tests/fixtures/.gitkeep
git commit -m "chore: add test runner baseline"
```

---

### Task 2: 持久化流程文件（AGENT.MD/task.json/progress.txt）

**Files:**
- Create: `AGENT.MD`
- Create: `task.json`
- Create: `progress.txt`
- Create: `tests/workflow/task_schema.test.ts`

**Step 1: 写失败测试（task.json 结构校验）**

```ts
import { describe, expect, it } from 'vitest';
import fs from 'node:fs';

describe('task schema', () => {
  it('contains required workflow fields', () => {
    const raw = fs.readFileSync('task.json', 'utf-8');
    const json = JSON.parse(raw);
    expect(Array.isArray(json.tasks)).toBe(true);
    expect(json.tasks[0]).toHaveProperty('id');
    expect(json.tasks[0]).toHaveProperty('title');
    expect(json.tasks[0]).toHaveProperty('status');
    expect(json.tasks[0]).toHaveProperty('steps');
  });
});
```

**Step 2: 运行测试验证失败**

Run: `npm run test -- tests/workflow/task_schema.test.ts`
Expected: FAIL with `ENOENT: no such file or directory, open 'task.json'`.

**Step 3: 最小实现（创建流程文件）**

- `AGENT.MD`: 固定流程（一次只做一个任务、阻塞写明、同一任务同一 commit）。
- `task.json`: 初始化 8-12 个 V1 任务，状态用 `pending`。
- `progress.txt`: 初始化日志头与首条记录（设计已确认/计划已落地）。

**Step 4: 运行测试验证通过**

Run: `npm run test -- tests/workflow/task_schema.test.ts`
Expected: PASS.

**Step 5: Commit**

```bash
git add AGENT.MD task.json progress.txt tests/workflow/task_schema.test.ts
git commit -m "chore: add persistent workflow files"
```

---

### Task 3: 数据解析脚本（从 markdown 到结构化问题）

**Files:**
- Create: `scripts/build-data.ts`
- Create: `scripts/lib/parse_knowledge.ts`
- Create: `scripts/lib/parse_company.ts`
- Create: `scripts/lib/types.ts`
- Create: `tests/data/parse_knowledge.test.ts`
- Create: `tests/fixtures/knowledge_sample.md`
- Create: `tests/fixtures/company_sample.md`

**Step 1: 写失败测试（能提取题目与频次）**

```ts
it('parses knowledge markdown into question rows', async () => {
  const rows = await parseKnowledge('tests/fixtures/knowledge_sample.md');
  expect(rows.length).toBeGreaterThan(0);
  expect(rows[0]).toHaveProperty('question');
  expect(rows[0]).toHaveProperty('memberCount');
});
```

**Step 2: 运行测试验证失败**

Run: `npm run test -- tests/data/parse_knowledge.test.ts`
Expected: FAIL with `Cannot find module` or parser not implemented.

**Step 3: 写最小实现（解析器）**

- 按 `## 分类`、`### 编号`、`出现次数`、`出现公司`、`常见变体` 解析。
- 输出统一结构：`id/title/category/memberCount/companies/variants/source`。

**Step 4: 运行测试验证通过**

Run: `npm run test -- tests/data/parse_knowledge.test.ts`
Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/lib/types.ts scripts/lib/parse_knowledge.ts scripts/lib/parse_company.ts scripts/build-data.ts tests/data/parse_knowledge.test.ts tests/fixtures/knowledge_sample.md tests/fixtures/company_sample.md
git commit -m "feat: add markdown parsers for interview data"
```

---

### Task 4: AI 范围过滤与 JSON 产物生成

**Files:**
- Create: `scripts/lib/filter_ai_scope.ts`
- Create: `scripts/lib/keywords.ts`
- Create: `tests/data/filter_ai_scope.test.ts`
- Modify: `scripts/build-data.ts`
- Create: `src/data/questions.json` (generated)
- Create: `src/data/companies.json` (generated)
- Create: `src/data/categories.json` (generated)
- Create: `src/data/stats.json` (generated)
- Create: `src/data/search-index.json` (generated)

**Step 1: 写失败测试（过滤后仅保留 AI 核心）**

```ts
it('keeps AI core categories and AI system design keywords only', () => {
  const output = filterAiScope(inputRows);
  expect(output.every((r) => ALLOWED.has(r.category) || r.aiSystemDesign === true)).toBe(true);
});
```

**Step 2: 运行测试验证失败**

Run: `npm run test -- tests/data/filter_ai_scope.test.ts`
Expected: FAIL.

**Step 3: 写最小实现（分类白名单 + AI 系统设计关键词）**

- 白名单分类按设计文档。
- `system_design/uncertain` 仅命中关键词才保留。
- 生成 5 个前端 JSON 文件。

**Step 4: 构建数据并验证**

Run: `node scripts/build-data.ts`  
Run: `npm run test -- tests/data/filter_ai_scope.test.ts`
Expected: PASS and `src/data/*.json` generated.

**Step 5: Commit**

```bash
git add scripts/lib/filter_ai_scope.ts scripts/lib/keywords.ts scripts/build-data.ts tests/data/filter_ai_scope.test.ts src/data/questions.json src/data/companies.json src/data/categories.json src/data/stats.json src/data/search-index.json
git commit -m "feat: generate AI-scoped data artifacts"
```

---

### Task 5: 公司定向主链路页面（默认入口）

**Files:**
- Modify: `src/pages/index.astro`
- Create: `src/pages/companies/index.astro`
- Create: `src/pages/companies/[company].astro`
- Create: `src/components/questions/QuestionCard.astro`
- Create: `src/components/filters/SearchBox.ts`
- Create: `src/lib/path.ts`

**Step 1: 写失败测试（路由存在）**

```ts
it('has companies routes', () => {
  expect(fs.existsSync('src/pages/companies/index.astro')).toBe(true);
  expect(fs.existsSync('src/pages/companies/[company].astro')).toBe(true);
});
```

**Step 2: 运行测试验证失败**

Run: `npm run test -- tests/routes/companies_routes.test.ts`
Expected: FAIL.

**Step 3: 写最小实现（公司页 + 默认跳转逻辑）**

- 首页 CTA 默认指向 `/companies`。
- 公司列表页展示公司卡片与题量统计。
- 公司详情页展示该公司题目列表，支持标题搜索与频次排序。

**Step 4: 运行构建验证**

Run: `npm run build`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/pages/index.astro src/pages/companies/index.astro src/pages/companies/[company].astro src/components/questions/QuestionCard.astro src/components/filters/SearchBox.ts src/lib/path.ts tests/routes/companies_routes.test.ts
git commit -m "feat: add company-directed learning flow"
```

---

### Task 6: 高频优先 + 知识点学习链路

**Files:**
- Create: `src/pages/hot/index.astro`
- Create: `src/pages/categories/index.astro`
- Create: `src/pages/categories/[category].astro`
- Create: `src/components/filters/FilterBar.ts`
- Create: `tests/routes/study_routes.test.ts`

**Step 1: 写失败测试（新增入口页存在）**

```ts
it('has hot and category routes', () => {
  expect(fs.existsSync('src/pages/hot/index.astro')).toBe(true);
  expect(fs.existsSync('src/pages/categories/index.astro')).toBe(true);
});
```

**Step 2: 运行测试验证失败**

Run: `npm run test -- tests/routes/study_routes.test.ts`
Expected: FAIL.

**Step 3: 写最小实现（列表页 + 筛选）**

- 高频页按 `memberCount` 降序展示。
- 分类页展示知识点导航与分类详情。
- 公共筛选栏支持关键词、公司、频次。

**Step 4: 运行构建验证**

Run: `npm run build`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/pages/hot/index.astro src/pages/categories/index.astro src/pages/categories/[category].astro src/components/filters/FilterBar.ts tests/routes/study_routes.test.ts
git commit -m "feat: add hot and category learning flows"
```

---

### Task 7: 题目详情与学习进度看板

**Files:**
- Create: `src/pages/questions/[id].astro`
- Create: `src/pages/progress.astro`
- Create: `src/components/questions/MarkButton.ts`
- Create: `src/lib/progress.ts`
- Create: `tests/progress/progress_store.test.ts`

**Step 1: 写失败测试（progress 存储行为）**

```ts
it('stores completed and starred ids', () => {
  setCompleted('q1', true);
  setStarred('q1', true);
  expect(getCompleted().has('q1')).toBe(true);
  expect(getStarred().has('q1')).toBe(true);
});
```

**Step 2: 运行测试验证失败**

Run: `npm run test -- tests/progress/progress_store.test.ts`
Expected: FAIL.

**Step 3: 写最小实现（localStorage 封装 + 页面）**

- `progress.ts` 提供读写 API（完成、收藏、最近学习）。
- 详情页接入标记按钮。
- 进度页展示总览和分维度统计。

**Step 4: 运行测试与构建**

Run: `npm run test -- tests/progress/progress_store.test.ts`  
Run: `npm run build`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/pages/questions/[id].astro src/pages/progress.astro src/components/questions/MarkButton.ts src/lib/progress.ts tests/progress/progress_store.test.ts
git commit -m "feat: add question detail and progress dashboard"
```

---

### Task 8: 管理员入口 + 更新任务生成（弱门禁）

**Files:**
- Create: `src/pages/admin.astro`
- Create: `src/components/admin/AdminGate.ts`
- Create: `src/components/admin/SubmissionForm.ts`
- Create: `src/lib/submission.ts`
- Create: `.github/ISSUE_TEMPLATE/interview-submission.yml`
- Create: `tests/admin/submission_payload.test.ts`

**Step 1: 写失败测试（提交 payload 结构）**

```ts
it('builds deterministic submission payload', () => {
  const payload = buildSubmissionPayload({ text: 'xx', sourceUrl: 'https://x.com' });
  expect(payload).toHaveProperty('createdAt');
  expect(payload).toHaveProperty('text');
  expect(payload).toHaveProperty('sourceUrl');
});
```

**Step 2: 运行测试验证失败**

Run: `npm run test -- tests/admin/submission_payload.test.ts`
Expected: FAIL.

**Step 3: 写最小实现（门禁 + 任务导出）**

- 管理员页输入口令后可见表单。
- 表单产出结构化 JSON。
- 生成 GitHub New Issue 预填链接（标题/正文含 JSON）。
- 本地保存最近提交记录（localStorage）。

**Step 4: 运行测试与构建**

Run: `npm run test -- tests/admin/submission_payload.test.ts`  
Run: `npm run build`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/pages/admin.astro src/components/admin/AdminGate.ts src/components/admin/SubmissionForm.ts src/lib/submission.ts .github/ISSUE_TEMPLATE/interview-submission.yml tests/admin/submission_payload.test.ts
git commit -m "feat: add admin submission entry and issue template"
```

---

### Task 9: 独立部署与构建自动化

**Files:**
- Create: `.github/workflows/deploy-pages.yml`
- Modify: `astro.config.mjs`
- Modify: `package.json`
- Create: `README.md`

**Step 1: 写失败检查（本地 build + preview）**

Run: `npm run build`
Expected: 先失败（若 base/site 配置缺失）。

**Step 2: 写最小实现（Pages 配置与工作流）**

- `astro.config.mjs` 支持 `SITE_URL`/`BASE_PATH` 环境变量。
- workflow: checkout -> setup node -> npm ci -> npm run build -> deploy pages。
- README 补部署说明。

**Step 3: 本地验证**

Run: `npm run build && npm run preview`
Expected: PASS.

**Step 4: Commit**

```bash
git add .github/workflows/deploy-pages.yml astro.config.mjs package.json README.md
git commit -m "ci: add standalone pages deployment workflow"
```

---

### Task 10: AgentGuide 子路径集成准备（延后执行）

**Files:**
- Create: `plans/agentguide-integration-checklist.md`
- Create: `scripts/export-dist.sh`

**Step 1: 先写集成检查清单**

- 目标路径：`/interview/`
- 资源路径检查：base path、静态资源、导航链接。
- 构建产物复制策略。

**Step 2: 写最小脚本（导出 dist）**

```bash
#!/usr/bin/env bash
set -euo pipefail
npm run build
echo "dist ready"
```

**Step 3: 验证脚本**

Run: `bash scripts/export-dist.sh`
Expected: PASS.

**Step 4: Commit**

```bash
git add plans/agentguide-integration-checklist.md scripts/export-dist.sh
git commit -m "chore: prepare AgentGuide integration checklist"
```

---

## 全局执行约束

- 相关技能：`@test-driven-development`、`@verification-before-completion`、`@systematic-debugging`。
- 每个任务必须更新 `progress.txt` 和 `task.json` 后再提交。
- 任何阻塞必须将任务标记为 `blocked`，并记录解除条件。
- 不做超出设计范围的功能（YAGNI）。

## 验收命令（最终）

```bash
npm run test
npm run build
```

预期：全部通过，且核心路由可访问（`/companies`、`/hot`、`/categories`、`/questions/:id`、`/progress`、`/admin`）。
