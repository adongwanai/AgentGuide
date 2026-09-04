(() => {
    'use strict';

    const RESOURCE_BATCH_SIZE = 8;
    const CATEGORY_ORDER = [
        '全部',
        '快速开始',
        '学习路线',
        '技术栈',
        '项目实战',
        '面试求职',
        '理论',
        '资源合集',
        '开源项目',
    ];

    const ROUTES = {
        developer: {
            link: 'https://github.com/adongwanai/AgentGuide/blob/main/docs/05-roadmaps/learning-roadmap-development.md',
            linkLabel: '查看完整开发路线',
            steps: [
                ['理解边界', 'Agent Loop 与 Workflow', '明确状态、动作、观察与停止条件。'],
                ['做出系统', '工具、上下文与记忆', '让工具可治理，让上下文可控制。'],
                ['证明质量', 'Trace、Replay 与 Eval', '用案例集和指标证明改动有效。'],
                ['完成表达', 'Demo、README 与面试话术', '把架构决策、实验与结果讲清楚。'],
            ],
        },
        algorithm: {
            link: 'https://github.com/adongwanai/AgentGuide/blob/main/docs/05-roadmaps/algorithm-complete-learning-guide.md',
            linkLabel: '查看完整算法路线',
            steps: [
                ['打牢基础', '模型、训练与推理原理', '掌握 Transformer、微调、对齐与推理优化。'],
                ['建立方法', 'RAG、Agent 与多模态', '理解检索、决策、记忆与视觉语言建模。'],
                ['完成实验', '指标、基线与消融', '用可复现的实验验证方法与工程选择。'],
                ['面向岗位', '手撕、项目与系统追问', '把原理推导、实验结果与业务价值串起来。'],
            ],
        },
    };

    const searchInput = document.getElementById('resource-search');
    const categoryTabs = document.getElementById('category-tabs');
    const typeFilter = document.getElementById('type-filter');
    const levelFilter = document.getElementById('level-filter');
    const sortFilter = document.getElementById('sort-filter');
    const clearFiltersButton = document.getElementById('clear-filters');
    const resourceGrid = document.getElementById('resource-grid');
    const resultsSummary = document.getElementById('results-summary');
    const emptyState = document.getElementById('empty-state');
    const errorState = document.getElementById('error-state');
    const loadMoreButton = document.getElementById('load-more');

    const state = {
        resources: [],
        filtered: [],
        category: '全部',
        type: '全部',
        level: '全部',
        sort: 'recommended',
        query: '',
        visible: RESOURCE_BATCH_SIZE,
    };

    function escapeHtml(value) {
        return String(value)
            .replaceAll('&', '&amp;')
            .replaceAll('<', '&lt;')
            .replaceAll('>', '&gt;')
            .replaceAll('"', '&quot;')
            .replaceAll("'", '&#039;');
    }

    function normalizeText(value) {
        return String(value || '')
            .normalize('NFKC')
            .toLowerCase()
            .replace(/[，。！？、：；（）【】《》“”‘’·|/\\_-]+/g, ' ')
            .replace(/\s+/g, ' ')
            .trim();
    }

    function queryTerms(value) {
        const normalized = normalizeText(value);
        if (!normalized) return [];

        const aliases = {
            '智能体': ['agent'],
            '求职': ['面试', '简历', 'offer'],
            '找工作': ['面试', '简历', 'offer'],
            '检索增强': ['rag', '检索'],
            '记忆': ['memory'],
            '工具协议': ['mcp'],
            '强化学习': ['rlhf', 'grpo'],
            '微调': ['sft', 'lora'],
            '评估': ['评测', 'eval'],
        };

        const terms = normalized.split(' ').filter(Boolean);
        for (const [key, additions] of Object.entries(aliases)) {
            if (normalized.includes(key)) terms.push(...additions);
        }
        return [...new Set(terms)];
    }

    function searchableText(resource) {
        return normalizeText([
            resource.title,
            resource.description,
            resource.category,
            resource.type,
            resource.level,
            ...(resource.tags || []),
            resource.sourcePath,
        ].join(' '));
    }

    function scoreResource(resource, terms) {
        if (terms.length === 0) return resource.featured ? 60 : 0;

        const title = normalizeText(resource.title);
        const description = normalizeText(resource.description);
        const tags = normalizeText((resource.tags || []).join(' '));
        const category = normalizeText(`${resource.category} ${resource.type} ${resource.level}`);
        const all = resource._searchText;
        let score = 0;
        let matchedTerms = 0;

        for (const term of terms) {
            let matched = false;
            if (title === term) {
                score += 80;
                matched = true;
            } else if (title.startsWith(term)) {
                score += 46;
                matched = true;
            } else if (title.includes(term)) {
                score += 32;
                matched = true;
            }
            if (tags.includes(term)) {
                score += 20;
                matched = true;
            }
            if (category.includes(term)) {
                score += 14;
                matched = true;
            }
            if (description.includes(term)) {
                score += 7;
                matched = true;
            }
            if (!matched && all.includes(term)) {
                score += 3;
                matched = true;
            }
            if (matched) matchedTerms += 1;
        }

        if (matchedTerms === terms.length) score += 24;
        if (resource.featured) score += 8;
        return matchedTerms > 0 ? score : -1;
    }

    function formatDate(value) {
        if (!/^\d{4}-\d{2}-\d{2}$/.test(value || '')) return '';
        const [year, month, day] = value.split('-');
        return `${year}.${month}.${day}`;
    }

    function resourceCard(resource) {
        const tags = (resource.tags || []).slice(0, 3)
            .map((tag) => `<span class="resource-tag">${escapeHtml(tag)}</span>`)
            .join('');
        const reading = Number(resource.readingMinutes) > 0
            ? `<span>约 ${Number(resource.readingMinutes)} 分钟</span>`
            : `<span>${escapeHtml(resource.type)}</span>`;
        const target = resource.url.includes('adongwanai.github.io/AgentGuide/') ? '' : ' target="_blank" rel="noopener noreferrer"';
        const status = resource.status === '建设中' ? '<span>建设中</span>' : '';
        const featured = resource.featured ? '<span class="is-featured">精选</span>' : '';

        return `
            <article class="resource-card">
                <a class="resource-card-link" href="${escapeHtml(resource.url)}"${target} aria-label="打开 ${escapeHtml(resource.title)}"></a>
                <div class="resource-card-top">
                    <div class="resource-card-heading">
                        <div>
                            <span class="resource-category">${escapeHtml(resource.category)}</span>
                            <h3>${escapeHtml(resource.title)}</h3>
                        </div>
                        <span class="resource-arrow" aria-hidden="true">↗</span>
                    </div>
                    <p>${escapeHtml(resource.description)}</p>
                    <div class="tag-list">${tags}</div>
                </div>
                <div class="resource-card-footer">
                    <span>${escapeHtml(resource.level)}</span>
                    <span>${escapeHtml(resource.type)}</span>
                    ${reading}
                    <span>${escapeHtml(formatDate(resource.date))}</span>
                    ${status}
                    ${featured}
                </div>
            </article>
        `;
    }

    function sortResources(items, scores) {
        return [...items].sort((a, b) => {
            if (state.sort === 'updated') {
                return String(b.date).localeCompare(String(a.date)) || a.title.localeCompare(b.title, 'zh-CN');
            }
            if (state.sort === 'shortest') {
                const aMinutes = a.readingMinutes || 999;
                const bMinutes = b.readingMinutes || 999;
                return aMinutes - bMinutes || a.title.localeCompare(b.title, 'zh-CN');
            }

            const scoreDiff = (scores.get(b.id) || 0) - (scores.get(a.id) || 0);
            if (scoreDiff !== 0) return scoreDiff;
            if (a.status !== b.status) return a.status === '已发布' ? -1 : 1;
            return String(b.date).localeCompare(String(a.date)) || a.title.localeCompare(b.title, 'zh-CN');
        });
    }

    function applyFilters({ resetVisible = true } = {}) {
        if (resetVisible) state.visible = RESOURCE_BATCH_SIZE;
        const terms = queryTerms(state.query);
        const scores = new Map();

        const matched = state.resources.filter((resource) => {
            if (state.category !== '全部' && resource.category !== state.category) return false;
            if (state.type !== '全部' && resource.type !== state.type) return false;
            if (state.level !== '全部' && resource.level !== state.level) return false;

            const score = scoreResource(resource, terms);
            scores.set(resource.id, score);
            return score >= 0;
        });

        state.filtered = sortResources(matched, scores);
        renderResults();
        updateUrl();
    }

    function renderResults() {
        const visibleItems = state.filtered.slice(0, state.visible);
        resourceGrid.setAttribute('aria-busy', 'false');
        resourceGrid.innerHTML = visibleItems.map(resourceCard).join('');

        const hasResults = state.filtered.length > 0;
        resourceGrid.hidden = !hasResults;
        emptyState.hidden = hasResults;
        errorState.hidden = true;
        loadMoreButton.hidden = state.visible >= state.filtered.length || !hasResults;

        if (!hasResults) {
            resultsSummary.textContent = '没有匹配的资源';
            return;
        }

        const queryLabel = state.query ? `“${state.query}”` : '当前筛选';
        resultsSummary.textContent = `${queryLabel}找到 ${state.filtered.length} 项，已显示 ${visibleItems.length} 项`;
    }

    function categoryCounts() {
        const counts = new Map([['全部', state.resources.length]]);
        for (const resource of state.resources) {
            counts.set(resource.category, (counts.get(resource.category) || 0) + 1);
        }
        return counts;
    }

    function renderCategories() {
        const counts = categoryCounts();
        const available = CATEGORY_ORDER.filter((category) => counts.has(category));
        categoryTabs.innerHTML = available.map((category) => {
            const active = state.category === category ? ' is-active' : '';
            return `<button type="button" class="category-tab${active}" data-category="${escapeHtml(category)}">${escapeHtml(category)}<span class="tab-count">${counts.get(category)}</span></button>`;
        }).join('');
    }

    function populateTypes() {
        const types = [...new Set(state.resources.map((resource) => resource.type))]
            .sort((a, b) => a.localeCompare(b, 'zh-CN'));
        typeFilter.innerHTML = '<option value="全部">全部类型</option>'
            + types.map((type) => `<option value="${escapeHtml(type)}">${escapeHtml(type)}</option>`).join('');
        typeFilter.value = state.type;
    }

    function updateUrl() {
        const url = new URL(window.location.href);
        const entries = {
            q: state.query,
            category: state.category === '全部' ? '' : state.category,
            type: state.type === '全部' ? '' : state.type,
            level: state.level === '全部' ? '' : state.level,
            sort: state.sort === 'recommended' ? '' : state.sort,
        };

        for (const [key, value] of Object.entries(entries)) {
            if (value) url.searchParams.set(key, value);
            else url.searchParams.delete(key);
        }
        history.replaceState(null, '', `${url.pathname}${url.search}${url.hash}`);
    }

    function loadStateFromUrl() {
        const params = new URLSearchParams(window.location.search);
        state.query = params.get('q') || '';
        state.category = params.get('category') || '全部';
        state.type = params.get('type') || '全部';
        state.level = params.get('level') || '全部';
        state.sort = params.get('sort') || 'recommended';
        searchInput.value = state.query;
        levelFilter.value = state.level;
        sortFilter.value = state.sort;
    }

    function resetFilters() {
        state.query = '';
        state.category = '全部';
        state.type = '全部';
        state.level = '全部';
        state.sort = 'recommended';
        searchInput.value = '';
        typeFilter.value = '全部';
        levelFilter.value = '全部';
        sortFilter.value = 'recommended';
        renderCategories();
        applyFilters();
        searchInput.focus({ preventScroll: true });
    }

    async function initResources() {
        try {
            const response = await fetch('data/resources.json', { cache: 'no-cache' });
            if (!response.ok) throw new Error(`Resource index returned ${response.status}`);
            const resources = await response.json();
            if (!Array.isArray(resources)) throw new Error('Resource index is not an array');

            state.resources = resources.map((resource) => ({
                ...resource,
                _searchText: searchableText(resource),
            }));
            loadStateFromUrl();

            const categories = new Set(state.resources.map((resource) => resource.category));
            if (state.category !== '全部' && !categories.has(state.category)) state.category = '全部';
            const types = new Set(state.resources.map((resource) => resource.type));
            if (state.type !== '全部' && !types.has(state.type)) state.type = '全部';

            populateTypes();
            renderCategories();
            applyFilters();

            document.querySelectorAll('[data-resource-total]').forEach((node) => {
                node.textContent = `${state.resources.length}+`;
            });
        } catch (error) {
            console.error(error);
            resourceGrid.hidden = true;
            emptyState.hidden = true;
            errorState.hidden = false;
            loadMoreButton.hidden = true;
            resultsSummary.textContent = '资源索引加载失败';
        }
    }

    function bindResourceControls() {
        let searchTimer;
        searchInput.addEventListener('input', () => {
            window.clearTimeout(searchTimer);
            searchTimer = window.setTimeout(() => {
                state.query = searchInput.value.trim();
                applyFilters();
            }, 90);
        });

        categoryTabs.addEventListener('click', (event) => {
            const button = event.target.closest('[data-category]');
            if (!(button instanceof HTMLButtonElement)) return;
            state.category = button.dataset.category || '全部';
            renderCategories();
            applyFilters();
        });

        typeFilter.addEventListener('change', () => {
            state.type = typeFilter.value;
            applyFilters();
        });
        levelFilter.addEventListener('change', () => {
            state.level = levelFilter.value;
            applyFilters();
        });
        sortFilter.addEventListener('change', () => {
            state.sort = sortFilter.value;
            applyFilters();
        });
        clearFiltersButton.addEventListener('click', resetFilters);
        document.querySelector('[data-empty-reset]').addEventListener('click', resetFilters);

        loadMoreButton.addEventListener('click', () => {
            state.visible += RESOURCE_BATCH_SIZE;
            renderResults();
        });
    }

    function focusSearch() {
        document.getElementById('resources').scrollIntoView({ behavior: 'smooth', block: 'start' });
        window.setTimeout(() => searchInput.focus({ preventScroll: true }), 320);
    }

    function bindGlobalControls() {
        document.querySelectorAll('[data-theme-toggle]').forEach((button) => {
            button.addEventListener('click', () => {
                const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
                document.documentElement.dataset.theme = next;
                try {
                    localStorage.setItem('agentguide-theme', next);
                } catch {}
            });
        });

        document.querySelectorAll('[data-search-trigger]').forEach((button) => {
            button.addEventListener('click', focusSearch);
        });

        document.addEventListener('keydown', (event) => {
            if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
                event.preventDefault();
                focusSearch();
            }
        });
    }

    function bindRoadmapSwitch() {
        const steps = document.getElementById('roadmap-steps');
        const routeLink = document.getElementById('route-link');
        document.querySelectorAll('[data-route]').forEach((button) => {
            button.addEventListener('click', () => {
                const routeKey = button.dataset.route;
                const route = ROUTES[routeKey];
                if (!route) return;

                document.querySelectorAll('[data-route]').forEach((item) => {
                    item.classList.toggle('is-active', item === button);
                });
                steps.innerHTML = route.steps.map(([label, title, description]) => `
                    <li>
                        <span>${escapeHtml(label)}</span>
                        <div><strong>${escapeHtml(title)}</strong><p>${escapeHtml(description)}</p></div>
                    </li>
                `).join('');
                routeLink.href = route.link;
                routeLink.innerHTML = `${escapeHtml(route.linkLabel)} <span aria-hidden="true">↗</span>`;
            });
        });
    }

    async function loadGithubStats() {
        const cachedKey = 'agentguide-github-stats';
        try {
            const cached = JSON.parse(localStorage.getItem(cachedKey) || 'null');
            if (cached && Date.now() - cached.savedAt < 60 * 60 * 1000) {
                renderGithubStats(cached);
                return;
            }

            const response = await fetch('https://api.github.com/repos/adongwanai/AgentGuide', {
                headers: { Accept: 'application/vnd.github+json' },
            });
            if (!response.ok) throw new Error(`GitHub API returned ${response.status}`);
            const data = await response.json();
            const stats = {
                stars: Number(data.stargazers_count) || 0,
                forks: Number(data.forks_count) || 0,
                savedAt: Date.now(),
            };
            localStorage.setItem(cachedKey, JSON.stringify(stats));
            renderGithubStats(stats);
        } catch (error) {
            console.info('GitHub stats unavailable', error);
            document.querySelectorAll('[data-github-forks]').forEach((node) => {
                node.textContent = '查看';
            });
        }
    }

    function renderGithubStats(stats) {
        const numberFormat = new Intl.NumberFormat('zh-CN');
        document.querySelectorAll('[data-github-stars]').forEach((node) => {
            node.textContent = numberFormat.format(stats.stars);
        });
        document.querySelectorAll('[data-github-forks]').forEach((node) => {
            node.textContent = numberFormat.format(stats.forks);
        });
    }

    function init() {
        document.body.classList.add('is-ready');
        bindGlobalControls();
        bindResourceControls();
        bindRoadmapSwitch();
        initResources();
        loadGithubStats();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init, { once: true });
    } else {
        init();
    }
})();
