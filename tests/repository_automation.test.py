#!/usr/bin/env python3
"""Regression checks for repository layout and non-mutating automation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIGRATION_STUBS = (
    "projects/04-end-to-end-projects/README.md",
    "projects/05-agent-workflows/README.md",
    "projects/06-project-collections/README.md",
    "resources/agent/ai-agent-production-challenges.md",
    "resources/multimodal/evaluation-checklist.md",
    "resources/multimodal/multimodal-rag-pipeline.md",
    "resources/rag/vector-db.md",
)


def tracked(*paths: str) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--", *paths],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def main() -> int:
    errors: list[str] = []
    workflow_dir = ROOT / ".github" / "workflows"
    ci = (workflow_dir / "ci.yml").read_text(encoding="utf-8")
    deploy = (workflow_dir / "deploy-pages.yml").read_text(encoding="utf-8")
    workflows = "\n".join(
        path.read_text(encoding="utf-8") for path in workflow_dir.glob("*.yml")
    )

    for app in ("ai-research-ebook", "InterviewGuide"):
        if not (ROOT / "apps" / app / "package.json").is_file():
            errors.append(f"apps/{app}/package.json is missing")
    if (ROOT / "external").exists():
        errors.append("legacy external/ directory still exists")
    if tracked("research"):
        errors.append("top-level research/ build snapshot is still tracked")
    if tracked("data/resources.json", "sitemap.xml"):
        errors.append("generated resources.json or sitemap.xml is still tracked")
    if (workflow_dir / "update-resources.yml").exists():
        errors.append("update-resources.yml still exists")
    if "git commit" in workflows or "git push" in workflows:
        errors.append("a workflow still writes commits back to the repository")

    for old_path in MIGRATION_STUBS:
        if (ROOT / old_path).exists():
            errors.append(f"expired migration stub still exists: {old_path}")

    required_ci_fragments = (
        "content-checks:",
        "research-build:",
        "interview-test-build:",
        "pages-assembly:",
        "npm test",
        "python tests/markdown_links.test.py",
    )
    for fragment in required_ci_fragments:
        if fragment not in ci:
            errors.append(f"ci.yml is missing: {fragment}")

    required_deploy_fragments = (
        "apps/ai-research-ebook",
        "apps/InterviewGuide",
        "python scripts/generate_resources.py",
        "python tests/pages_assembly.test.py site-dist",
    )
    for fragment in required_deploy_fragments:
        if fragment not in deploy:
            errors.append(f"deploy-pages.yml is missing: {fragment}")
    if "external/" in ci or "external/" in deploy:
        errors.append("legacy external/ path remains in CI or deployment")
    if "paths-ignore:" in ci or "paths-ignore:" in deploy:
        errors.append("CI or deployment still skips path-scoped changes")

    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in ("/data/resources.json", "/sitemap.xml", "/research/"):
        if pattern not in gitignore:
            errors.append(f".gitignore is missing: {pattern}")

    if errors:
        print("Repository automation test failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Repository layout and automation checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
