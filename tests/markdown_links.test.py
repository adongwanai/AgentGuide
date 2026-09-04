#!/usr/bin/env python3
"""Check local links in active content and both website source trees."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTENT_ROOTS = (
    REPO_ROOT / "docs",
    REPO_ROOT / "projects",
    REPO_ROOT / "resources",
    REPO_ROOT / "apps",
)
SOURCE_SUFFIXES = {".md", ".mdx", ".html", ".astro", ".css"}
EXCLUDED_PARTS = {
    ".git",
    ".astro",
    "archive",
    "dist",
    "node_modules",
    "site-dist",
}
EXCLUDED_FILES = {REPO_ROOT / "PROJECT_SUMMARY.md"}

INLINE_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
REFERENCE_LINK_RE = re.compile(r"^\s*\[[^\]]+\]:\s*(\S+)")
HTML_LINK_RE = re.compile(r"(?:href|src)\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)
CSS_URL_RE = re.compile(r"url\(\s*[\"']?([^\"')]+)[\"']?\s*\)", re.IGNORECASE)
TITLE_RE = re.compile(r"^(.*?)\s+[\"'][^\"']*[\"']\s*$")


def source_files() -> list[Path]:
    """Return active content and static website source files."""
    files = [path for path in REPO_ROOT.glob("*.md") if path not in EXCLUDED_FILES]
    for root in CONTENT_ROOTS:
        if not root.exists():
            continue
        files.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in SOURCE_SUFFIXES
            and not EXCLUDED_PARTS.intersection(path.relative_to(REPO_ROOT).parts)
        )
    return sorted(set(files))


def strip_code_and_comments(lines: list[str]) -> list[tuple[int, str]]:
    """Remove fenced code blocks and HTML comments before parsing prose links."""
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
    """Return a relative local path, or None for routes and external/dynamic links."""
    target = raw_target.strip()
    if target.startswith("<") and ">" in target:
        target = target[1 : target.index(">")]
    else:
        titled = TITLE_RE.match(target)
        if titled:
            target = titled.group(1).strip()

    if (
        not target
        or target.startswith(("#", "//", "/"))
        or "{" in target
        or "}" in target
    ):
        return None

    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc:
        return None

    path_part = unquote(parsed.path).replace("\\", "/")
    return path_part or None


def targets_in(path: Path) -> list[tuple[int, str]]:
    """Extract static link targets with source line numbers."""
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    targets: list[tuple[int, str]] = []
    for line_number, line in strip_code_and_comments(lines):
        raw_targets: list[str] = []
        if path.suffix.lower() in {".md", ".mdx"}:
            raw_targets.extend(INLINE_LINK_RE.findall(line))
            reference = REFERENCE_LINK_RE.match(line)
            if reference:
                raw_targets.append(reference.group(1))
        if path.suffix.lower() in {".md", ".mdx", ".html", ".astro"}:
            raw_targets.extend(HTML_LINK_RE.findall(line))
        if path.suffix.lower() == ".css":
            raw_targets.extend(CSS_URL_RE.findall(line))
        targets.extend((line_number, target) for target in raw_targets)
    return targets


def local_target_exists(target: Path) -> bool:
    """Accept files, directories, and extensionless Markdown/MDX references."""
    if target.exists():
        return True
    if target.suffix:
        return False
    candidates = [
        target.with_suffix(".md"),
        target.with_suffix(".mdx"),
        target.with_suffix(".html"),
        target / "README.md",
        target / "index.md",
        target / "index.mdx",
        target / "index.html",
        target / "index.astro",
    ]
    return any(candidate.exists() for candidate in candidates)


def main() -> int:
    errors: list[str] = []
    files = source_files()

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

            if not local_target_exists(resolved):
                errors.append(
                    f"{source.relative_to(REPO_ROOT)}:{line_number}: "
                    f"missing local target: {raw_target}"
                )

    if errors:
        print("Repository local-link check failed:")
        for error in errors:
            print(f"- {error}")
        print(f"\n{len(errors)} broken link(s) across {len(files)} source file(s).")
        return 1

    print(f"Repository local-link check passed for {len(files)} source file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
