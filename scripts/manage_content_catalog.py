"""Validate metadata and generate AgentGuide content indexes.

Usage:
    python scripts/manage_content_catalog.py --write
    python scripts/manage_content_catalog.py --check
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

from content_metadata import (
    PUBLIC_STATUSES,
    ROOT,
    first_heading,
    has_metadata,
    infer_metadata,
    markdown_files,
    parse_front_matter,
    read_text,
    validate_metadata,
)


START = "<!-- AUTO-GENERATED-CONTENT:START -->"
END = "<!-- AUTO-GENERATED-CONTENT:END -->"
BACKLOG_HEADER = """# AgentGuide 内容 Backlog

> 本页由 `python scripts/manage_content_catalog.py --write` 自动生成。以下文档保留在原路径，但当前不进入公开导航与站点资源索引。

"""

ALLOWED_TYPES_BY_ROOT = {
    "docs": {
        "入口页",
        "教程",
        "实践指南",
        "求职指南",
        "路线图",
        "题库",
        "研究专题",
    },
    "projects": {"入口页", "项目蓝图", "迁移提示"},
    "resources": {"入口页", "资源清单", "论文清单", "迁移提示"},
}


def relative_link(source: Path, target: Path) -> str:
    rel = Path(os.path.relpath(target, source.parent)).as_posix()
    if not rel.startswith("."):
        rel = "./" + rel
    rel = rel.replace("(", "%28").replace(")", "%29")
    if any(character.isspace() or character in "[]" for character in rel):
        return f"<{rel}>"
    return rel


def escape_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def load_records() -> tuple[list[dict[str, object]], list[str], list[str]]:
    records: list[dict[str, object]] = []
    warnings: list[str] = []
    errors: list[str] = []

    for path in markdown_files():
        rel = path.relative_to(ROOT).as_posix()
        try:
            metadata, body = parse_front_matter(read_text(path))
        except ValueError as exc:
            errors.append(f"{rel}: {exc}")
            continue

        if not has_metadata(metadata):
            warnings.append(
                f"{rel}: metadata missing or incomplete; using fallback inference"
            )
        illegal = [
            message
            for message in validate_metadata(metadata)
            if not message.startswith("missing fields:")
        ]
        errors.extend(f"{rel}: {message}" for message in illegal)
        if illegal:
            continue
        resolved_metadata = {**infer_metadata(rel, body), **metadata}

        root = rel.split("/", 1)[0]
        if resolved_metadata["type"] not in ALLOWED_TYPES_BY_ROOT[root]:
            errors.append(
                f"{rel}: type {resolved_metadata['type']!r} is not valid below {root}/"
            )

        records.append(
            {
                "path": path,
                "rel": rel,
                "body": body,
                "title": first_heading(body, path.stem),
                **resolved_metadata,
            }
        )

    return records, warnings, errors


def duplicate_number_errors(records: list[dict[str, object]]) -> list[str]:
    numbered: dict[tuple[str, str], list[str]] = defaultdict(list)
    for record in records:
        path = record["path"]
        assert isinstance(path, Path)
        match = re.match(r"^(\d+)-", path.name)
        if match:
            numbered[(path.parent.as_posix(), match.group(1))].append(
                str(record["rel"])
            )

    errors: list[str] = []
    for (_, prefix), paths in sorted(numbered.items()):
        if len(paths) > 1:
            errors.append(f"duplicate sibling number {prefix}: {', '.join(paths)}")
    return errors


def public_descendants(
    index_path: Path, records: list[dict[str, object]]
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for record in records:
        path = record["path"]
        assert isinstance(path, Path)
        if path == index_path or record["status"] not in PUBLIC_STATUSES:
            continue
        try:
            path.relative_to(index_path.parent)
        except ValueError:
            continue
        results.append(record)
    return sorted(results, key=lambda item: str(item["rel"]).lower())


def render_document_table(
    index_path: Path, records: list[dict[str, object]]
) -> list[str]:
    descendants = public_descendants(index_path, records)
    lines = [
        "## 完整内容索引",
        "",
        "<!-- 下表由元数据生成，请勿手工编辑。 -->",
        "",
        "| 文档 | 类型 | 状态 | 难度 | 主题 |",
        "|:---|:---|:---|:---|:---|",
    ]
    for record in descendants:
        path = record["path"]
        assert isinstance(path, Path)
        topics = record["topic"]
        assert isinstance(topics, list)
        link = relative_link(index_path, path)
        lines.append(
            "| "
            f"[{escape_cell(record['title'])}]({link}) | "
            f"{record['type']} | {record['status']} | {record['level']} | "
            f"{escape_cell('、'.join(topics))} |"
        )
    if not descendants:
        lines.append("| 暂无已发布内容 | — | — | — | — |")
    return lines


def render_pdf_table(index_path: Path) -> list[str]:
    resources_root = ROOT / "resources"
    pdfs = sorted(
        resources_root.rglob("*.pdf"),
        key=lambda path: path.relative_to(resources_root).as_posix().casefold(),
    )
    lines = [
        "",
        "## PDF 索引",
        "",
        f"共 {len(pdfs)} 份 PDF；文件保持原路径，不参与 Markdown 元数据分类。",
        "",
        "| PDF | 所在目录 |",
        "|:---|:---|",
    ]
    for path in pdfs:
        directory = path.parent.relative_to(ROOT / "resources").as_posix()
        lines.append(
            f"| [{escape_cell(path.stem)}]({relative_link(index_path, path)}) | "
            f"{escape_cell(directory or '.')} |"
        )
    return lines


def generated_block(index_path: Path, records: list[dict[str, object]]) -> str:
    lines = render_document_table(index_path, records)
    if index_path == ROOT / "resources/README.md":
        lines.extend(render_pdf_table(index_path))
    return "\n".join((START, *lines, END))


def replace_generated_block(text: str, block: str) -> str:
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.S)
    if pattern.search(text):
        return pattern.sub(lambda _: block, text)
    return text.rstrip() + "\n\n---\n\n" + block + "\n"


def render_backlog(records: list[dict[str, object]]) -> str:
    backlog = sorted(
        (record for record in records if record["status"] == "待补充"),
        key=lambda item: str(item["rel"]).lower(),
    )
    lines = [
        BACKLOG_HEADER.rstrip(),
        "",
        f"当前共 **{len(backlog)}** 项待补内容。",
        "",
        "| 原路径 | 标题 | 类型 | 难度 | 主题 |",
        "|:---|:---|:---|:---|:---|",
    ]
    backlog_path = ROOT / "BACKLOG.md"
    for record in backlog:
        path = record["path"]
        assert isinstance(path, Path)
        topics = record["topic"]
        assert isinstance(topics, list)
        lines.append(
            "| "
            f"[`{escape_cell(record['rel'])}`]({relative_link(backlog_path, path)}) | "
            f"{escape_cell(record['title'])} | {record['type']} | {record['level']} | "
            f"{escape_cell('、'.join(topics))} |"
        )
    return "\n".join(lines) + "\n"


def expected_files(records: list[dict[str, object]]) -> dict[Path, str]:
    expected: dict[Path, str] = {}
    for record in records:
        if record["type"] != "入口页" or record["status"] not in PUBLIC_STATUSES:
            continue
        path = record["path"]
        assert isinstance(path, Path)
        text = read_text(path)
        expected[path] = replace_generated_block(text, generated_block(path, records))
    expected[ROOT / "BACKLOG.md"] = render_backlog(records)
    return expected


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()

    records, warnings, errors = load_records()
    errors.extend(duplicate_number_errors(records))
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    if errors:
        print("Content metadata validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    expected = expected_files(records)
    stale: list[str] = []
    for path, wanted in expected.items():
        current = read_text(path) if path.exists() else ""
        if current == wanted:
            continue
        if args.write:
            path.write_text(wanted, encoding="utf-8")
        else:
            stale.append(path.relative_to(ROOT).as_posix())

    if stale:
        print("Generated content is stale; run with --write:", file=sys.stderr)
        for rel in stale:
            print(f"- {rel}", file=sys.stderr)
        return 1

    action = "Updated" if args.write else "Checked"
    backlog_count = sum(record["status"] == "待补充" for record in records)
    index_count = sum(
        record["type"] == "入口页" and record["status"] in PUBLIC_STATUSES
        for record in records
    )
    print(
        f"{action} {index_count} public indexes; validated {len(records)} documents; "
        f"Backlog contains {backlog_count} items."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
