// @ts-check
import { defineConfig } from 'astro/config';
import tailwind from '@astrojs/tailwind';
import mdx from '@astrojs/mdx';

const githubRepository = process.env.GITHUB_REPOSITORY ?? '';
const [repositoryOwner = '', repositoryName = ''] = githubRepository.split('/');

const rawRepoName = process.env.GITHUB_REPO ?? repositoryName;
const rawGithubUsername = process.env.GITHUB_USERNAME ?? repositoryOwner;
const repoName = rawRepoName || 'InterviewGuide';
const githubUsername = rawGithubUsername || 'yiweinanzi';
const siteUrl = process.env.SITE_URL ?? `https://${githubUsername}.github.io`;
const isGitHubActions = process.env.GITHUB_ACTIONS === 'true';
const basePath = process.env.BASE_PATH ?? (isGitHubActions ? `/${repoName}` : '/');

export default defineConfig({
  site: siteUrl,
  base: basePath,
  integrations: [tailwind(), mdx()],
  output: 'static',
  build: {
    format: 'directory',
  },
});
