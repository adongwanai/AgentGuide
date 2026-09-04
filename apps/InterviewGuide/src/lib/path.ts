const TRAILING_COMPANY_COUNT_RE = /\s*等\d+家公司\s*$/u;

const COMPANY_SLUG_MAP: Record<string, string> = {
  字节跳动: 'bytedance',
  美团: 'meituan',
  腾讯: 'tencent',
  百度: 'baidu',
  阿里巴巴: 'alibaba',
  小红书: 'xiaohongshu',
  未知: 'unknown',
  通用题库: 'general-bank',
  华为: 'huawei',
  京东: 'jd',
  小米: 'xiaomi',
  蚂蚁集团: 'ant-group',
  拼多多: 'pinduoduo',
  OPPO: 'oppo',
  滴滴: 'didi',
  网易: 'netease',
  哔哩哔哩: 'bilibili',
  荣耀: 'honor',
  商汤: 'sensetime',
  联想: 'lenovo',
  VIVO: 'vivo',
  携程: 'trip-com',
  知乎: 'zhihu',
  快手: 'kuaishou',
  '阿里（阿里云 / 达摩院）': 'alibaba-cloud-damo',
  '阿里（阿里妈妈）': 'alimama',
  '虾皮 Shopee': 'shopee',
  'B 站': 'bilibili-b',
};

const CATEGORY_SLUG_MAP: Record<string, string> = {
  项目与行为面试: 'project-behavior',
  'nlp与大模型': 'nlp-llm',
  编程与算法: 'coding-algorithms',
  机器学习基础: 'ml-fundamentals',
  推荐系统: 'recommender-systems',
  深度学习: 'deep-learning',
  机器学习系统: 'ml-systems',
  计算机视觉: 'computer-vision',
  'ai系统设计': 'ai-system-design',
};

const COMPANY_NAME_BY_SLUG = new Map<string, string>(
  Object.entries(COMPANY_SLUG_MAP).map(([name, slug]) => [slug, name]),
);

const CATEGORY_NAME_BY_SLUG = new Map<string, string>(
  Object.entries(CATEGORY_SLUG_MAP).map(([name, slug]) => [slug, name]),
);

function normalizeCompanyRouteName(name: string): string {
  return name.replace(TRAILING_COMPANY_COUNT_RE, '').trim();
}

function toAsciiSlug(name: string, prefix: string): string {
  const ascii = name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
  if (ascii) return ascii;

  const hex = Buffer.from(name, 'utf8').toString('hex').slice(0, 16);
  return `${prefix}-${hex}`;
}

function splitSuffix(path: string): { pathname: string; suffix: string } {
  const match = path.match(/^([^?#]*)(.*)$/);
  return {
    pathname: match?.[1] ?? path,
    suffix: match?.[2] ?? '',
  };
}

function normalizeRoutePath(path: string): string {
  const normalized = path.startsWith('/') ? path : `/${path}`;
  const { pathname, suffix } = splitSuffix(normalized);
  if (pathname === '/') return `/${suffix}`;
  if (pathname.endsWith('/')) return `${pathname}${suffix}`;

  const lastSegment = pathname.split('/').pop() ?? '';
  if (lastSegment.includes('.')) return `${pathname}${suffix}`;

  return `${pathname}/${suffix}`;
}

export function withBaseFrom(base: string, path: string): string {
  const normalizedBase = base === '/' ? '' : base.replace(/\/$/, '');
  return `${normalizedBase}${normalizeRoutePath(path)}`;
}

export function withBase(path: string): string {
  return withBaseFrom(import.meta.env.BASE_URL, path);
}

export function toCompanySlug(name: string): string {
  const normalized = normalizeCompanyRouteName(name);
  const known = COMPANY_SLUG_MAP[normalized];
  if (known) return known;
  return toAsciiSlug(normalized, 'company');
}

export function fromCompanySlug(slug: string): string {
  return COMPANY_NAME_BY_SLUG.get(slug) ?? decodeURIComponent(slug);
}

export function toCategorySlug(key: string): string {
  const normalized = key.trim();
  const known = CATEGORY_SLUG_MAP[normalized];
  if (known) return known;
  return toAsciiSlug(normalized, 'category');
}

export function fromCategorySlug(slug: string): string {
  return CATEGORY_NAME_BY_SLUG.get(slug) ?? decodeURIComponent(slug);
}

export function toQuestionSlug(id: string): string {
  return encodeURIComponent(id);
}

export function fromQuestionSlug(slug: string): string {
  return decodeURIComponent(slug);
}
