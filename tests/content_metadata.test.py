#!/usr/bin/env python3
"""Regression checks for PR 2 taxonomy, backlog, and public resources."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from content_metadata import (  # noqa: E402
    LEVELS,
    PUBLIC_STATUSES,
    TOPICS,
    TYPES,
    markdown_files,
    parse_front_matter,
    read_text,
    validate_metadata,
)
from generate_resources import collect_resources  # noqa: E402


EXPECTED_BACKLOG = {
    "docs/01-theory/02-agent-history.md",
    "docs/01-theory/03-transformer.md",
    "docs/01-theory/05-cot-and-planning.md",
    "docs/01-theory/08-agent-bench.md",
    "docs/02-tech-stack/04-langchain-guide.md",
    "docs/02-tech-stack/06-multi-agent-frameworks.md",
    "docs/02-tech-stack/07-agentscope.md",
    "docs/02-tech-stack/08-vector-db-basics.md",
    "docs/02-tech-stack/20-rag-full-pipeline.md",
    "docs/02-tech-stack/21-agent-reinforcement-learning.md",
    "docs/02-tech-stack/build-your-agent-framework.md",
    "docs/03-practice/02-high-availability-rag.md",
    "docs/03-practice/04-graduation-project.md",
    "resources/agent/papers/data_synthesis/README.md",
    "resources/rag/papers/graphrag/README.md",
    "resources/rag/papers/multimodal_rag/README.md",
}

EXPECTED_ARCHIVED = {
    "docs/00-getting-started/04-repo-gap-map.md",
}

EXPECTED_MOVED_TYPES = {
    "docs/03-practice/06-agent-production-challenges.md": "实践指南",
    "docs/03-practice/07-multimodal-rag-evaluation-checklist.md": "实践指南",
    "docs/02-tech-stack/28-multimodal-rag-pipeline.md": "教程",
    "docs/02-tech-stack/29-vector-database-selection.md": "教程",
    "resources/project-catalogs/end-to-end-agent-projects.md": "资源清单",
    "resources/project-catalogs/agent-workflows.md": "资源清单",
    "resources/project-catalogs/agent-project-collections.md": "资源清单",
}


def fail(messages: list[str]) -> int:
    print("Content taxonomy test failed:")
    for message in messages:
        print(f"- {message}")
    return 1


def main() -> int:
    errors: list[str] = []
    records: dict[str, dict[str, object]] = {}
    for path in markdown_files():
        rel = path.relative_to(ROOT).as_posix()
        metadata, _ = parse_front_matter(read_text(path))
        if not metadata:
            errors.append(f"{rel}: missing metadata")
            continue
        errors.extend(f"{rel}: {message}" for message in validate_metadata(metadata))
        records[rel] = metadata

    backlog = {rel for rel, metadata in records.items() if metadata["status"] == "待补充"}
    archived = {rel for rel, metadata in records.items() if metadata["status"] == "已归档"}
    if backlog != EXPECTED_BACKLOG:
        errors.append(f"Backlog paths differ: {sorted(backlog ^ EXPECTED_BACKLOG)}")
    if archived != EXPECTED_ARCHIVED:
        errors.append(f"Archived paths differ: {sorted(archived ^ EXPECTED_ARCHIVED)}")

    hidden_paths = EXPECTED_BACKLOG | EXPECTED_ARCHIVED
    inline_link = re.compile(r"\[[^]]*\]\(([^)]+)\)")
    public_sources = [
        rel for rel, metadata in records.items() if metadata["status"] in PUBLIC_STATUSES
    ]
    public_sources.append("README.md")
    for rel in public_sources:
        source = ROOT / rel
        for raw_target in inline_link.findall(read_text(source)):
            target = raw_target.strip()
            if target.startswith("<") and target.endswith(">"):
                target = target[1:-1]
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            resolved = (source.parent / unquote(parsed.path)).resolve()
            try:
                target_rel = resolved.relative_to(ROOT).as_posix()
            except ValueError:
                continue
            if target_rel in hidden_paths:
                errors.append(f"{rel}: public document links to hidden {target_rel}")

    backlog_text = read_text(ROOT / "BACKLOG.md")
    for rel in EXPECTED_BACKLOG:
        if rel not in backlog_text:
            errors.append(f"BACKLOG.md does not list {rel}")

    for rel, expected_type in EXPECTED_MOVED_TYPES.items():
        actual = records.get(rel, {}).get("type")
        if actual != expected_type:
            errors.append(f"{rel}: expected type {expected_type}, got {actual}")

    resources = collect_resources()
    first_party = [item for item in resources if item.get("sourcePath")]
    public_paths = {item["sourcePath"] for item in first_party}
    leaked = hidden_paths & public_paths
    if leaked:
        errors.append(f"non-public documents leaked into resources.json: {sorted(leaked)}")
    for item in resources:
        if item["status"] not in PUBLIC_STATUSES:
            errors.append(f"{item.get('sourcePath') or item['id']}: non-public status in resources.json")
        if item["type"] not in TYPES:
            errors.append(f"{item.get('sourcePath') or item['id']}: invalid type {item['type']}")
        if item["level"] not in LEVELS:
            errors.append(f"{item.get('sourcePath') or item['id']}: invalid level {item['level']}")
        tags = item.get("tags", [])
        if not 1 <= len(tags) <= 3 or any(tag not in TOPICS for tag in tags):
            errors.append(f"{item.get('sourcePath') or item['id']}: invalid generated tags {tags}")

    pdfs = sorted((ROOT / "resources").rglob("*.pdf"))
    if len(pdfs) != 58:
        errors.append(f"expected 58 PDFs, found {len(pdfs)}")
    resources_index = read_text(ROOT / "resources/README.md")
    if "共 58 份 PDF" not in resources_index:
        errors.append("resources/README.md does not contain the generated 58-PDF summary")

    if errors:
        return fail(errors)
    print(
        f"Content taxonomy test passed for {len(records)} Markdown files, "
        f"{len(EXPECTED_BACKLOG)} backlog items, and {len(first_party)} public records."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
