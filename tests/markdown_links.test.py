#!/usr/bin/env python3
"""Check local links in the active AgentGuide Markdown knowledge base."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit


REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = (
    REPO_ROOT,
    REPO_ROOT / "docs",
    REPO_ROOT / "projects",
    REPO_ROOT / "resources",
)
EXCLUDED_PARTS = {
    ".git",
    ".astro",
    "archive",
    "dist",
    "node_modules",
}
EXCLUDED_FILES = {
    REPO_ROOT / "PROJECT_SUMMARY.md",
}

INLINE_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
REFERENCE_LINK_RE = re.compile(r"^\s*\[[^\]]+\]:\s*(\S+)")
HTML_LINK_RE = re.compile(r"(?:href|src)\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)
TITLE_RE = re.compile(r"^(.*?)\s+[\"'][^\"']*[\"']\s*$")


def markdown_files() -> list[Path]:
    """Return tracked-style Markdown paths in the agreed active scope."""
    files = list(REPO_ROOT.glob("*.md"))
    for root in SCAN_ROOTS[1:]:
        if not root.exists():
            continue
        files.extend(
            path
            for path in root.rglob("*.md")
            if not EXCLUDED_PARTS.intersection(path.relative_to(REPO_ROOT).parts)
        )
    return sorted(set(files) - EXCLUDED_FILES)


def strip_code_and_comments(lines: list[str]) -> list[tuple[int, str]]:
    """Remove fenced code blocks and HTML comments before parsing links."""
    visible: list[tuple[int, str]] = []
    in_fence = False
    in_comment = False

    for line_number, original in enumerate(lines, start=1):
        line = original
        if line.lstrip().startswith(("```", "~~~")) and not in_comment:
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        output: list[str] = []
        cursor = 0
        while cursor < len(line):
            if in_comment:
                end = line.find("-->", cursor)
                if end == -1:
                    cursor = len(line)
                    continue
                in_comment = False
                cursor = end + 3
                continue

            start = line.find("<!--", cursor)
            if start == -1:
                output.append(line[cursor:])
                break
            output.append(line[cursor:start])
            cursor = start + 4
            in_comment = True

        cleaned = "".join(output)
        cleaned = re.sub(r"`[^`]*`", "", cleaned)
        visible.append((line_number, cleaned))

    return visible


def normalize_target(raw_target: str) -> str | None:
    """Return a local path target, or None for anchors and external links."""
    target = raw_target.strip()
    if target.startswith("<") and ">" in target:
        target = target[1 : target.index(">")]
    else:
        titled = TITLE_RE.match(target)
        if titled:
            target = titled.group(1).strip()

    if not target or target.startswith(("#", "//", "/")):
        return None

    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc:
        return None

    path_part = unquote(parsed.path).replace("\\", "/")
    return path_part or None


def targets_in(path: Path) -> list[tuple[int, str]]:
    """Extract Markdown and HTML link targets with source line numbers."""
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    targets: list[tuple[int, str]] = []
    for line_number, line in strip_code_and_comments(lines):
        raw_targets = INLINE_LINK_RE.findall(line)
        reference = REFERENCE_LINK_RE.match(line)
        if reference:
            raw_targets.append(reference.group(1))
        raw_targets.extend(HTML_LINK_RE.findall(line))
        targets.extend((line_number, target) for target in raw_targets)
    return targets


def main() -> int:
    errors: list[str] = []
    files = markdown_files()

    for source in files:
        for line_number, raw_target in targets_in(source):
            target = normalize_target(raw_target)
            if target is None:
                continue

            resolved = (source.parent / target).resolve()
            try:
                resolved.relative_to(REPO_ROOT)
            except ValueError:
                errors.append(
                    f"{source.relative_to(REPO_ROOT)}:{line_number}: "
                    f"target leaves repository: {raw_target}"
                )
                continue

            if not resolved.exists():
                errors.append(
                    f"{source.relative_to(REPO_ROOT)}:{line_number}: "
                    f"missing local target: {raw_target}"
                )

    if errors:
        print("Markdown local-link check failed:")
        for error in errors:
            print(f"- {error}")
        print(f"\n{len(errors)} broken link(s) across {len(files)} Markdown file(s).")
        return 1

    print(f"Markdown local-link check passed for {len(files)} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
