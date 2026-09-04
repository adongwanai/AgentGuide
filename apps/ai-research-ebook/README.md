# Vibe Research - AI 科研指南

> 从 Idea 到论文投稿的完整 AI 科研工作流指南

## 简介

本项目已升级为 **Astro + Tailwind + MDX** 自定义站点，前端风格参考 `adongwanai.github.io`，并使用科研章节体系组织内容。

## 在线阅读

默认地址：`https://yiweinanzi.github.io/ai-research-ebook`

可通过环境变量覆盖：

```bash
GITHUB_USERNAME=<你的用户名> GITHUB_REPO=ai-research-ebook npm run build
```

## 本地开发

```bash
npm install
npm run dev
npm run build
npm run preview
```

## 部署前配置

`astro.config.mjs` 支持：

- `GITHUB_USERNAME`：默认 `yiweinanzi`
- `GITHUB_REPO`：默认 `ai-research-ebook`
- `SITE_URL`：默认 `https://<username>.github.io`
- `BASE_PATH`：默认 `/<repo>`

## 路由

- `/`：复刻风格首页
- `/docs/...`：章节文档页（含左侧导航、TOC、上下篇）
- `/skills`：技能工具页

示例：

- `/docs/intro/01-overview`
- `/docs/idea/01-research`
- `/docs/code/03-agents`

## 技术栈

- [Astro](https://astro.build)
- [Tailwind CSS](https://tailwindcss.com)
- [MDX](https://docs.astro.build/en/guides/integrations-guide/mdx/)
- GitHub Pages + GitHub Actions

## 内容结构

```text
src/
  content/
    docs/
      intro/
      idea/
      code/
      figures/
      writing/
      review/
      tools/
      library/
  layouts/
  components/
  pages/
```

## 许可证

MIT License
