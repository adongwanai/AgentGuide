import html
import json
import math
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from content_metadata import (
    PUBLIC_STATUSES,
    has_metadata,
    infer_metadata,
    is_scoped_markdown,
    parse_front_matter,
    validate_metadata,
)


ROOT = Path(__file__).resolve().parent.parent
OWNER_REPO = "adongwanai/AgentGuide"
SITE_ROOT = "https://adongwanai.github.io/AgentGuide"

INCLUDE_DIRS = [ROOT / "docs", ROOT / "resources", ROOT / "projects"]

FEATURED_PATHS = {
    "docs/00-getting-started/README.md",
    "docs/00-getting-started/02-first-7-days.md",
    "docs/02-tech-stack/27-agent-harness-engineering.md",
    "docs/03-practice/05-ship-agent-project.md",
    "docs/04-interview/README.md",
    "docs/04-interview/22-algorithm-ai-coding-question-bank.md",
    "docs/04-interview/23-frontier-interview-guides/README.md",
    "docs/04-interview/19-xiaohongshu-ai-algorithm-interview-bank.md",
    "docs/05-roadmaps/agent-job-ready-roadmap-2026.md",
    "docs/05-roadmaps/algorithm-complete-learning-guide.md",
    "docs/05-roadmaps/embodied-ai-vla-learning-guide.md",
    "docs/06-research-frontiers/README.md",
    "docs/06-research-frontiers/01-ai-research-directions-expanded.md",
}

CURATED_EXTERNAL = [
    {
        "id": "external-learn-workbuddy",
        "title": "learn-workbuddy",
        "description": "从 0 搭建 WorkBuddy-style Desktop Agent Harness，覆盖 Agent Loop、工具调用、上下文工程、长期记忆、Sidecar、权限审计与真实模型评测。",
        "category": "开源项目",
        "tags": ["Agent", "项目实战", "上下文工程"],
        "level": "进阶",
        "type": "项目蓝图",
        "url": "https://github.com/adongwanai/learn-workbuddy",
        "sourcePath": "",
        "date": "2026-07-27",
        "readingMinutes": 0,
        "wordCount": 0,
        "status": "已发布",
        "featured": True,
        "external": True,
    },
    {
        "id": "portal-vibe-research",
        "title": "Vibe Research AI 科研指南",
        "description": "从 Idea 生成、代码实现、论文图表、写作到审稿与 Rebuttal 的完整 AI 科研工作流。",
        "category": "资源合集",
        "tags": ["科研", "项目实战"],
        "level": "进阶",
        "type": "研究专题",
        "url": f"{SITE_ROOT}/research/",
        "sourcePath": "",
        "date": "2026-07-27",
        "readingMinutes": 0,
        "wordCount": 0,
        "status": "已发布",
        "featured": True,
        "external": False,
    },
    {
        "id": "portal-interview-guide",
        "title": "InterviewGuide AI 面经题库",
        "description": "按公司、知识点和频次筛选的 AI 算法与大模型面试题库，支持收藏、进度和高频优先。",
        "category": "面试求职",
        "tags": ["面试求职", "模型训练", "Agent"],
        "level": "进阶",
        "type": "题库",
        "url": f"{SITE_ROOT}/interview/",
        "sourcePath": "",
        "date": "2026-07-27",
        "readingMinutes": 0,
        "wordCount": 0,
        "status": "已发布",
        "featured": True,
        "external": False,
    },
]


def read_text(path):
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def clean_inline(text):
    text = re.sub(r"!\[[^]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[`*_~]", "", text)
    text = text.replace("—", "-").replace("–", "-")
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip(" #>|-")


def first_heading(md, fallback):
    for line in md.splitlines():
        match = re.match(r"^#{1,6}\s+(.+)$", line.strip())
        if match:
            return clean_inline(match.group(1)) or fallback
    return fallback


def first_paragraph(md):
    in_code = False
    paragraphs = []
    current = []

    for raw_line in md.splitlines():
        line = raw_line.strip()
        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        if re.match(r"^#{1,6}\s+", line) or line == "---":
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        if line.startswith("|") or line.startswith("<table") or line.startswith("<img"):
            continue
        if re.match(r"^[-*+]\s+\[[ xX]\]", line):
            continue

        cleaned = clean_inline(re.sub(r"^>+\s*", "", line))
        if not cleaned:
            continue
        if re.search(r"正在编写中|敬请期待", cleaned):
            continue
        current.append(cleaned)
        if len(" ".join(current)) >= 80:
            paragraphs.append(" ".join(current))
            break

    if current and not paragraphs:
        paragraphs.append(" ".join(current))

    description = next((item for item in paragraphs if len(item) >= 18), "")
    if not description:
        return "打开文档查看完整内容、实践步骤与延伸资源。"
    return description[:180].rstrip("，,。 ") + ("。" if not description.endswith("。") else "")


def category_for_path(rel):
    parts = rel.split("/")
    if parts[0] == "docs" and len(parts) > 1:
        section = parts[1]
        if section.startswith("00-getting-started"):
            return "快速开始"
        if section.startswith("01-theory"):
            return "理论"
        if section.startswith("02-tech-stack"):
            return "技术栈"
        if section.startswith("03-practice"):
            return "项目实战"
        if section.startswith("04-interview"):
            return "面试求职"
        if section.startswith("05-roadmaps"):
            return "学习路线"
        if section.startswith("06-research-frontiers"):
            return "研究前沿"
        return "技术栈"
    if parts[0] == "resources":
        return "资源合集"
    if parts[0] == "projects":
        return "开源项目"
    return "其他"


def content_stats(md):
    without_code = re.sub(r"```.*?```", " ", md, flags=re.DOTALL)
    plain = clean_inline(without_code)
    units = re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+", plain)
    word_count = len(units)
    return word_count, max(1, math.ceil(word_count / 400))


_GIT_DATE_CACHE = {}


def _git_date_uncached(path):
    try:
        relative = str(path.relative_to(ROOT))
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", relative],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if dirty.stdout.strip():
            # Pre-commit generation must match the date that git log will expose
            # after the content change is committed (dates are day-granular here).
            return datetime.now().strftime("%Y-%m-%d")
        result = subprocess.run(
            ["git", "log", "-1", "--format=%cs", "--", relative],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        date = result.stdout.strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
            return date
    except (OSError, ValueError):
        pass
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")
    except OSError:
        return datetime.now().strftime("%Y-%m-%d")


def git_date(path):
    # sitemap 现在按内容来源文件取日期，同一文件会被多次查询
    # （例如 12960 条题目页共用 questions.json），缓存避免重复 fork git。
    key = str(path)
    if key not in _GIT_DATE_CACHE:
        _GIT_DATE_CACHE[key] = _git_date_uncached(path)
    return _GIT_DATE_CACHE[key]


def newest_git_date(paths, fallback):
    """取一组文件中最新的提交日期；集合为空时回退到 fallback。"""
    dates = [git_date(path) for path in paths]
    return max(dates) if dates else fallback


def build_url(rel):
    return f"https://github.com/{OWNER_REPO}/blob/main/{quote(rel, safe='/')}"


def collect_resources():
    items = []
    for base in INCLUDE_DIRS:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.md")):
            rel = path.relative_to(ROOT).as_posix()
            if not is_scoped_markdown(rel):
                continue
            raw_md = read_text(path)
            metadata, md = parse_front_matter(raw_md)
            inferred = infer_metadata(rel, md)
            if not has_metadata(metadata):
                print(
                    f"WARNING: {rel}: metadata missing or incomplete; using fallback inference",
                    file=sys.stderr,
                )
            illegal = [
                message
                for message in validate_metadata(metadata)
                if not message.startswith("missing fields:")
            ]
            if illegal:
                raise ValueError(f"{rel}: {'; '.join(illegal)}")
            resolved_metadata = {**inferred, **metadata}
            if resolved_metadata["status"] not in PUBLIC_STATUSES:
                continue

            title = first_heading(md, path.stem)
            word_count, reading_minutes = content_stats(md)
            item = {
                "id": "doc-" + re.sub(r"[^a-z0-9]+", "-", rel.lower()).strip("-"),
                "title": title,
                "description": first_paragraph(md),
                "category": category_for_path(rel),
                "tags": resolved_metadata["topic"],
                "level": resolved_metadata["level"],
                "type": resolved_metadata["type"],
                "url": build_url(rel),
                "sourcePath": rel,
                "date": git_date(path),
                "readingMinutes": reading_minutes,
                "wordCount": word_count,
                "status": resolved_metadata["status"],
                "featured": rel in FEATURED_PATHS,
                "external": False,
            }
            items.append(item)

    items.extend(CURATED_EXTERNAL)
    items.sort(
        key=lambda item: (
            not item["featured"],
            item["status"] != "已发布",
            -int(item["date"].replace("-", "")),
            item["title"].lower(),
        )
    )
    return items


def sitemap_urls():
    """生成 sitemap 条目，lastmod 取自各 URL 对应内容来源文件的最后提交日期。

    此前所有条目统一使用 datetime.now()，导致两个问题：
    - 内容未改动时也会产生 diff（跨天即全量重写 13023 行），bot 因此提交空改动
    - 全站 lastmod 恒为同一天，该字段对搜索引擎失去参考价值
    """
    site_date = git_date(ROOT / "index.html")

    research_content = ROOT / "external/ai-research-ebook/src/content/docs"
    research_pages = sorted(research_content.rglob("*.mdx")) if research_content.exists() else []
    research_date = newest_git_date(research_pages, site_date)

    interview_data = ROOT / "external/InterviewGuide/src/data"
    questions_path = interview_data / "questions.json"
    categories_path = interview_data / "categories.json"
    companies_path = interview_data / "companies.json"

    # 三个 JSON 各自驱动一批 URL，同一批共用来源文件的日期
    questions_date = git_date(questions_path) if questions_path.exists() else site_date
    categories_date = git_date(categories_path) if categories_path.exists() else site_date
    companies_date = git_date(companies_path) if companies_path.exists() else site_date

    urls = [
        (f"{SITE_ROOT}/", site_date, "daily", "1.0"),
        (f"{SITE_ROOT}/research/", research_date, "weekly", "0.9"),
        (f"{SITE_ROOT}/research/docs/", research_date, "weekly", "0.8"),
        (f"{SITE_ROOT}/research/skills/", research_date, "weekly", "0.8"),
        (f"{SITE_ROOT}/interview/", questions_date, "daily", "0.9"),
        (f"{SITE_ROOT}/interview/hot/", questions_date, "daily", "0.9"),
        (f"{SITE_ROOT}/interview/categories/", categories_date, "weekly", "0.8"),
        (f"{SITE_ROOT}/interview/companies/", companies_date, "weekly", "0.8"),
    ]

    for path in research_pages:
        slug = path.relative_to(research_content).with_suffix("").as_posix()
        urls.append((f"{SITE_ROOT}/research/docs/{slug}/", git_date(path), "monthly", "0.7"))

    if questions_path.exists():
        for question in json.loads(read_text(questions_path)):
            urls.append((f"{SITE_ROOT}/interview/questions/{quote(question['id'], safe='')}/", questions_date, "monthly", "0.6"))

    category_slugs = {
        "项目与行为面试": "project-behavior",
        "nlp与大模型": "nlp-llm",
        "编程与算法": "coding-algorithms",
        "机器学习基础": "ml-fundamentals",
        "推荐系统": "recommender-systems",
        "深度学习": "deep-learning",
        "机器学习系统": "ml-systems",
        "计算机视觉": "computer-vision",
        "ai系统设计": "ai-system-design",
    }
    if categories_path.exists():
        for category in json.loads(read_text(categories_path)):
            slug = category_slugs.get(category["key"], quote(category["key"], safe=""))
            urls.append((f"{SITE_ROOT}/interview/categories/{slug}/", categories_date, "weekly", "0.7"))

    company_slugs = {
        "字节跳动": "bytedance", "美团": "meituan", "腾讯": "tencent", "百度": "baidu",
        "阿里巴巴": "alibaba", "小红书": "xiaohongshu", "未知": "unknown", "通用题库": "general-bank",
        "华为": "huawei", "京东": "jd", "小米": "xiaomi", "蚂蚁集团": "ant-group",
        "拼多多": "pinduoduo", "OPPO": "oppo", "滴滴": "didi", "网易": "netease",
        "哔哩哔哩": "bilibili", "荣耀": "honor", "商汤": "sensetime", "联想": "lenovo",
        "VIVO": "vivo", "携程": "trip-com", "知乎": "zhihu", "快手": "kuaishou",
        "阿里（阿里云 / 达摩院）": "alibaba-cloud-damo", "阿里（阿里妈妈）": "alimama",
        "虾皮 Shopee": "shopee", "B 站": "bilibili-b",
    }
    if companies_path.exists():
        for company in json.loads(read_text(companies_path)):
            slug = company_slugs.get(company["name"], quote(company["name"], safe=""))
            urls.append((f"{SITE_ROOT}/interview/companies/{slug}/", companies_date, "weekly", "0.7"))

    return urls


def write_sitemap():
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ]
    for location, lastmod, changefreq, priority in sitemap_urls():
        lines.extend([
            "  <url>",
            f"    <loc>{html.escape(location)}</loc>",
            f"    <lastmod>{lastmod}</lastmod>",
            f"    <changefreq>{changefreq}</changefreq>",
            f"    <priority>{priority}</priority>",
            "  </url>",
        ])
    lines.append("</urlset>")
    (ROOT / "sitemap.xml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def main():
    items = collect_resources()
    output_dir = ROOT / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "resources.json"
    output_path.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_sitemap()
    print(f"Wrote {len(items)} resources to {output_path}")
    print(f"Wrote {len(sitemap_urls())} canonical URLs to {ROOT / 'sitemap.xml'}")


if __name__ == "__main__":
    main()
