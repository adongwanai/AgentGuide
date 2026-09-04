// @ts-check
import { defineConfig } from 'astro/config';
import tailwind from '@astrojs/tailwind';
import mdx from '@astrojs/mdx';

const repoName = process.env.GITHUB_REPO ?? 'ai-research-ebook';
const githubUsername = process.env.GITHUB_USERNAME ?? 'yiweinanzi';
const siteUrl = process.env.SITE_URL ?? `https://${githubUsername}.github.io`;
const basePath = process.env.BASE_PATH ?? `/${repoName}`;

export default defineConfig({
  site: siteUrl,
  base: basePath,
  integrations: [tailwind(), mdx()],
  output: 'static',
  build: {
    format: 'directory',
  },
});
