# InterviewGuide

面向 AI 算法岗的结构化面经学习站，基于 Astro + Tailwind 构建。

## 本地开发

```bash
npm ci
npm run dev
```

## 测试与构建

```bash
npm run test
npm run build
```

## GitHub Pages 部署

仓库已提供部署工作流：`.github/workflows/deploy.yml`。

1. 推送到 `main` 分支，或在 Actions 页面手动触发 `workflow_dispatch`。
2. 工作流会执行 `npm ci` 和 `npm run build`，并上传 `dist/` 产物。
3. `actions/deploy-pages` 会将产物发布到 GitHub Pages。

部署前请在仓库设置中确认：

1. `Settings -> Pages -> Build and deployment` 选择 `GitHub Actions`。
2. 仓库公开可访问（或你的账户具备对应 Pages 权限）。
