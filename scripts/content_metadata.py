"""Shared metadata rules for AgentGuide Markdown content.

The repository intentionally avoids a YAML dependency.  This parser accepts only
the front-matter shape used by AgentGuide: three scalar fields and one list.
"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

TYPES = (
    "入口页",
    "教程",
    "实践指南",
    "求职指南",
    "路线图",
    "题库",
    "项目蓝图",
    "资源清单",
    "论文清单",
    "研究专题",
    "迁移提示",
)
STATUSES = ("已发布", "建设中", "待补充", "已归档")
PUBLIC_STATUSES = frozenset(("已发布", "建设中"))
LEVELS = ("入门", "进阶", "高阶", "通用")
TOPICS = (
    "Agent",
    "RAG",
    "上下文工程",
    "记忆",
    "MCP",
    "多智能体",
    "评测",
    "安全",
    "模型训练",
    "推理部署",
    "多模态",
    "Coding Agent",
    "具身智能",
    "基础模型",
    "面试求职",
    "项目实战",
    "科研",
    "框架工具",
)

FIELDS = ("type", "status", "level", "topic")
EXCLUDED_FILE = "docs/PR1-CLEANUP-LOG.md"


class FrontMatterError(ValueError):
    """Raised when a leading front-matter block is malformed."""


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def parse_front_matter(text: str) -> tuple[dict[str, object], str]:
    """Parse AgentGuide's constrained front matter and return metadata/body."""
    normalized = text.replace("\r\n", "\n")
    if not normalized.startswith("---\n"):
        return {}, text

    end = normalized.find("\n---\n", 4)
    if end < 0:
        raise FrontMatterError("leading front matter has no closing ---")

    block = normalized[4:end]
    body = normalized[end + 5 :]
    if body.startswith("\n"):
        body = body[1:]
    metadata: dict[str, object] = {}
    current_list: str | None = None

    for line_number, raw_line in enumerate(block.splitlines(), start=2):
        if not raw_line.strip():
            continue
        list_item = re.fullmatch(r"\s+-\s+(.+?)\s*", raw_line)
        if list_item:
            if current_list != "topic":
                raise FrontMatterError(
                    f"line {line_number}: list item is only allowed below topic"
                )
            topics = metadata.setdefault("topic", [])
            assert isinstance(topics, list)
            topics.append(list_item.group(1))
            continue

        field = re.fullmatch(r"([a-z]+):(?:\s*(.*))?", raw_line)
        if not field:
            raise FrontMatterError(f"line {line_number}: unsupported syntax")
        key, value = field.group(1), (field.group(2) or "").strip()
        if key not in FIELDS:
            raise FrontMatterError(f"line {line_number}: unsupported field {key!r}")
        if key in metadata:
            raise FrontMatterError(f"line {line_number}: duplicate field {key!r}")
        if key == "topic":
            if value:
                raise FrontMatterError(
                    f"line {line_number}: topic must use an indented list"
                )
            metadata[key] = []
            current_list = key
        else:
            if not value:
                raise FrontMatterError(f"line {line_number}: {key} cannot be empty")
            metadata[key] = value
            current_list = None

    return metadata, body


def render_front_matter(metadata: dict[str, object]) -> str:
    """Render metadata in the repository's canonical field order."""
    lines = ["---"]
    for field in ("type", "status", "level"):
        lines.append(f"{field}: {metadata[field]}")
    lines.append("topic:")
    for topic in metadata["topic"]:
        lines.append(f"  - {topic}")
    lines.extend(("---", ""))
    return "\n".join(lines) + "\n"


def has_metadata(metadata: dict[str, object]) -> bool:
    return all(field in metadata for field in FIELDS)


def validate_metadata(metadata: dict[str, object]) -> list[str]:
    """Return validation errors for a present metadata mapping."""
    errors: list[str] = []
    missing = [field for field in FIELDS if field not in metadata]
    if missing:
        errors.append("missing fields: " + ", ".join(missing))

    if "type" in metadata and metadata["type"] not in TYPES:
        errors.append(f"invalid type: {metadata['type']}")
    if "status" in metadata and metadata["status"] not in STATUSES:
        errors.append(f"invalid status: {metadata['status']}")
    if "level" in metadata and metadata["level"] not in LEVELS:
        errors.append(f"invalid level: {metadata['level']}")

    topics = metadata.get("topic")
    if topics is not None:
        if not isinstance(topics, list):
            errors.append("topic must be a list")
        else:
            if not 1 <= len(topics) <= 3:
                errors.append("topic must contain 1-3 values")
            if len(topics) != len(set(topics)):
                errors.append("topic contains duplicate values")
            invalid = [topic for topic in topics if topic not in TOPICS]
            if invalid:
                errors.append("invalid topic: " + ", ".join(invalid))
    return errors


def is_scoped_markdown(rel: str) -> bool:
    parts = Path(rel).parts
    if not parts or parts[0] not in {"docs", "projects", "resources"}:
        return False
    if rel == EXCLUDED_FILE:
        return False
    return not (len(parts) > 1 and parts[0] == "docs" and parts[1] == "archive")


def markdown_files() -> list[Path]:
    files: list[Path] = []
    for dirname in ("docs", "projects", "resources"):
        base = ROOT / dirname
        if base.exists():
            files.extend(
                path
                for path in base.rglob("*.md")
                if is_scoped_markdown(path.relative_to(ROOT).as_posix())
            )
    return sorted(files)


def category_for_path(rel: str) -> str:
    parts = rel.split("/")
    if parts[0] == "docs" and len(parts) > 1:
        section = parts[1]
        categories = {
            "00-getting-started": "快速开始",
            "01-theory": "基础理论",
            "02-tech-stack": "技术栈",
            "03-practice": "项目实战",
            "04-interview": "面试求职",
            "05-roadmaps": "学习路线",
            "06-research-frontiers": "科研前沿",
        }
        return categories.get(section, "技术栈")
    if parts[0] == "projects":
        return "项目实战"
    if parts[0] == "resources":
        return "资源合集"
    return "其他"


def first_heading(body: str, fallback: str) -> str:
    for line in body.splitlines():
        match = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if match:
            title = re.sub(r"[`*_~]", "", match.group(1)).strip()
            return title or fallback
    return fallback


TOPIC_RULES = (
    ("Coding Agent", r"coding agent|claude code|openclaw|代码智能体"),
    ("具身智能", r"具身|\bvla\b|robot|world model|世界模型"),
    ("多模态", r"多模态|multimodal|\bvlm\b|图像生成|语音生成|视觉"),
    ("多智能体", r"multi[- ]?agent|多智能体|autogen|crewai|metagpt"),
    ("上下文工程", r"上下文工程|context engineering|context window"),
    ("记忆", r"agent memory|记忆|memory"),
    ("MCP", r"\bmcp\b|model context protocol"),
    ("RAG", r"\brag\b|检索|向量数据库|vector db|graphrag"),
    ("评测", r"评测|评估|evaluation|benchmark|eval harness"),
    ("安全", r"安全|security|safety|sandbox|沙箱|权限"),
    ("模型训练", r"训练|sft|rlhf|dpo|grpo|lora|post-training|强化学习"),
    ("推理部署", r"部署|推理优化|vllm|sglang|serving|量化|ai infra"),
    ("面试求职", r"面试|求职|简历|题库|秋招|薪资|offer|hr"),
    ("科研", r"科研|研究方向|论文|paper|rebuttal"),
    ("项目实战", r"项目|实战|production|工程|workflow|作品集"),
    ("框架工具", r"框架|工具|framework|langchain|agentscope"),
    ("基础模型", r"transformer|deepseek|llama|qwen|基础模型|llm|大模型"),
    ("Agent", r"\bagent(?:ic)?\b|智能体|react"),
)


def infer_topics(rel: str, title: str, body: str = "") -> list[str]:
    """Infer conservative topics from path/title; body is accepted for API stability."""
    del body
    sample = f"{rel}\n{title}".lower()
    seeded: list[str] = []
    if "/04-interview/" in f"/{rel}" or "/05-roadmaps/" in f"/{rel}":
        seeded.append("面试求职")
    if "/06-research-frontiers/" in f"/{rel}":
        seeded.append("科研")
    if rel.startswith("projects/") or rel.startswith("resources/project-catalogs/"):
        seeded.append("项目实战")
    if rel.startswith("resources/agent/"):
        seeded.append("Agent")
    if rel.startswith("resources/rag/"):
        seeded.append("RAG")
    if rel.startswith("resources/multimodal/"):
        seeded.extend(("多模态", "RAG"))
    if "/papers/" in f"/{rel}":
        seeded.append("科研")
    if "/03-practice/" in f"/{rel}":
        seeded.append("项目实战")
    if "/00-getting-started/" in f"/{rel}":
        seeded.append("Agent")

    matches = [name for name, pattern in TOPIC_RULES if re.search(pattern, sample, re.I)]
    topics: list[str] = []
    for topic in (*seeded, *matches):
        if topic not in topics:
            topics.append(topic)
    return topics[:3] or ["Agent"]


def infer_type(rel: str, title: str) -> str:
    lowered = f"{rel} {title}".lower()
    if Path(rel).name.lower() == "readme.md":
        return "入口页"
    if rel.startswith("projects/"):
        return "项目蓝图"
    if rel.startswith("resources/"):
        return "论文清单" if re.search(r"paper|论文", lowered) else "资源清单"
    if "/04-interview/" in f"/{rel}":
        if re.search(r"question|题|bank|coding-exercises|fundamentals", lowered):
            return "题库"
        return "求职指南"
    if "/05-roadmaps/" in f"/{rel}":
        return "路线图"
    if "/06-research-frontiers/" in f"/{rel}":
        return "研究专题"
    if "/03-practice/" in f"/{rel}":
        return "实践指南"
    return "教程"


def infer_level(rel: str, title: str, body: str) -> str:
    del body
    if Path(rel).name.lower() == "readme.md" and not re.match(
        r"projects/0[1-3]-", rel
    ):
        return "通用"
    sample = f"{rel} {title}".lower()
    if "/00-getting-started/" in f"/{rel}":
        return "入门"
    if "/05-roadmaps/" in f"/{rel}":
        return "通用"
    if "/06-research-frontiers/" in f"/{rel}":
        return "高阶"
    if rel.startswith("projects/"):
        return "进阶"
    if rel.startswith("resources/"):
        if "/papers/" in f"/{rel}" or "论文" in title:
            return "进阶"
        return "通用"
    if "/04-interview/" in f"/{rel}":
        if "/18-agent-interview-playbooks/" in f"/{rel}" or "/23-frontier-interview-guides/" in f"/{rel}":
            return "高阶"
        if re.search(r"fundamentals|career|job-hunting|salary|hr-interview|mindset|resume|storytelling", rel):
            return "入门"
        if re.search(r"specialized|company-interview|model-evaluation", rel):
            return "高阶"
        return "进阶"
    if re.search(r"what-is-agent|transformer|langchain|agentscope|vector-db-basics", rel):
        return "入门"
    if re.search(
        r"高阶|production|security|sandbox|reinforcement|post-training|"
        r"evaluation|harness|compliance|deep.dive|完整指南|专项|infra",
        sample,
    ):
        return "高阶"
    return "进阶"


def infer_status(body: str) -> str:
    if re.search(r"正在编写中|敬请期待|TODO|待补充|待完善", body, re.I):
        return "待补充"
    return "已发布"


def infer_metadata(rel: str, body: str) -> dict[str, object]:
    title = first_heading(body, Path(rel).stem)
    return {
        "type": infer_type(rel, title),
        "status": infer_status(body),
        "level": infer_level(rel, title, body),
        "topic": infer_topics(rel, title, body),
    }
