import html
import json
import math
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from urllib.parse import quote


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

TAG_RULES = [
    ("Agent", r"\bagent(?:ic)?\b|智能体|langgraph|langchain|autogen|crewai|agent loop|harness"),
    ("RAG", r"\brag\b|retriev|检索|embedding|rerank|向量|graphrag"),
    ("上下文工程", r"context engineering|上下文工程|上下文管理|context window"),
    ("Memory", r"\bmemory\b|记忆系统|长期记忆|短期记忆"),
    ("MCP", r"\bmcp\b|model context protocol"),
    ("评测", r"\beval|evaluation|benchmark|评测|评估|llm-as-judge|test case"),
    ("安全", r"security|safety|sandbox|安全|权限|红队|hitl"),
    ("模型训练", r"sft|lora|qlora|rlhf|dpo|grpo|post-training|微调|强化学习"),
    ("多模态", r"multimodal|多模态|\bvlm\b|\bvla\b|vision language|具身智能"),
    ("面试", r"interview|面试|手撕|题库|简历|求职|offer"),
    ("项目实战", r"project|项目|实战|ship|毕业设计|作品集|portfolio"),
    ("科研", r"research|paper|论文|审稿|rebuttal|实验设计"),
    ("部署", r"vllm|sglang|serving|部署|推理优化|量化"),
    ("框架", r"framework|框架|agentscope|camelai"),
]

CURATED_EXTERNAL = [
    {
        "id": "external-learn-workbuddy",
        "title": "learn-workbuddy",
        "description": "从 0 搭建 WorkBuddy-style Desktop Agent Harness，覆盖 Agent Loop、工具调用、上下文工程、长期记忆、Sidecar、权限审计与真实模型评测。",
        "category": "开源项目",
        "tags": ["Agent", "项目实战", "上下文工程", "Memory", "评测"],
        "level": "进阶",
        "type": "开源项目",
        "url": "https://github.com/adongwanai/learn-workbuddy",
        "sourcePath": "",
        "date": "2026-07-27",
        "readingMinutes": 0,
        "wordCount": 0,
        "status": "已完善",
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
        "type": "专题站",
        "url": f"{SITE_ROOT}/research/",
        "sourcePath": "",
        "date": "2026-07-27",
        "readingMinutes": 0,
        "wordCount": 0,
        "status": "已完善",
        "featured": True,
        "external": False,
    },
    {
        "id": "portal-interview-guide",
        "title": "InterviewGuide AI 面经题库",
        "description": "按公司、知识点和频次筛选的 AI 算法与大模型面试题库，支持收藏、进度和高频优先。",
        "category": "面试求职",
        "tags": ["面试", "模型训练", "多模态", "Agent"],
        "level": "进阶",
        "type": "专题站",
        "url": f"{SITE_ROOT}/interview/",
        "sourcePath": "",
        "date": "2026-07-27",
        "readingMinutes": 0,
        "wordCount": 0,
        "status": "已完善",
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


def type_for_path(rel, title):
    lowered = f"{rel} {title}".lower()
    if rel.startswith("projects/"):
        return "项目"
    if rel.startswith("resources/"):
        return "论文清单" if "paper" in lowered or "论文" in title else "资源清单"
    if "/04-interview/" in f"/{rel}":
        return "题库" if re.search(r"题|question|coding|interview-bank", lowered) else "求职指南"
    if "/05-roadmaps/" in f"/{rel}":
        return "路线图"
    if "/06-research-frontiers/" in f"/{rel}":
        return "研究专题"
    if "/03-practice/" in f"/{rel}":
        return "实战指南"
    if "/00-getting-started/" in f"/{rel}":
        return "入门指南"
    if "/01-theory/" in f"/{rel}":
        return "原理教程"
    return "深度教程"


def level_for_text(rel, title, md):
    sample = f"{rel} {title} {md[:1600]}".lower()
    if re.search(r"入门|基础|从零|第一周|7天|7 天|what is|getting.started|概念", sample):
        return "入门"
    if re.search(r"高阶|高级|生产级|系统设计|源码|benchmark|评测|安全|强化学习|post.training|完整指南|deep.dive|专项", sample):
        return "高阶"
    return "进阶"


def tags_for_text(rel, title, md):
    sample = f"{rel} {title} {md[:8000]}".lower()
    tags = [name for name, pattern in TAG_RULES if re.search(pattern, sample, re.IGNORECASE)]
    return tags[:5] or [category_for_path(rel)]


def content_stats(md):
    without_code = re.sub(r"```.*?```", " ", md, flags=re.DOTALL)
    plain = clean_inline(without_code)
    units = re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+", plain)
    word_count = len(units)
    return word_count, max(1, math.ceil(word_count / 400))


def status_for_text(md, word_count):
    has_placeholder = bool(re.search(r"正在编写中|敬请期待|TODO|待补充|待完善", md, re.IGNORECASE))
    return "建设中" if has_placeholder and word_count < 800 else "已完善"


_GIT_DATE_CACHE = {}


def _git_date_uncached(path):
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%cs", "--", str(path.relative_to(ROOT))],
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
            md = read_text(path)
            title = first_heading(md, path.stem)
            word_count, reading_minutes = content_stats(md)
            item = {
                "id": "doc-" + re.sub(r"[^a-z0-9]+", "-", rel.lower()).strip("-"),
                "title": title,
                "description": first_paragraph(md),
                "category": category_for_path(rel),
                "tags": tags_for_text(rel, title, md),
                "level": level_for_text(rel, title, md),
                "type": type_for_path(rel, title),
                "url": build_url(rel),
                "sourcePath": rel,
                "date": git_date(path),
                "readingMinutes": reading_minutes,
                "wordCount": word_count,
                "status": status_for_text(md, word_count),
                "featured": rel in FEATURED_PATHS,
                "external": False,
            }
            items.append(item)

    items.extend(CURATED_EXTERNAL)
    items.sort(
        key=lambda item: (
            not item["featured"],
            item["status"] != "已完善",
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
