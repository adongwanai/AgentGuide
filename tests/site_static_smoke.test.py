#!/usr/bin/env python3
"""Smoke-test the root static site and freshly generated deployment data."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(text: str, fragment: str, source: str, errors: list[str]) -> None:
    if fragment not in text:
        errors.append(f"{source} is missing expected fragment: {fragment}")


def main() -> int:
    errors: list[str] = []
    index = (ROOT / "index.html").read_text(encoding="utf-8")
    script_path = ROOT / "assets" / "site.js"
    script = script_path.read_text(encoding="utf-8")

    syntax = subprocess.run(
        ["node", "--check", str(script_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if syntax.returncode:
        errors.append(f"assets/site.js failed node --check: {syntax.stderr.strip()}")

    for fragment in (
        '<link rel="canonical" href="https://adongwanai.github.io/AgentGuide/">',
        '<meta property="og:image" content="https://adongwanai.github.io/AgentGuide/assets/agentguide-social.png">',
        "application/ld+json",
        "data-github-forks",
        '<option value="通用">通用</option>',
        "/AgentGuide/interview/",
        '<li><a href="/AgentGuide/interview/">面经题库</a></li>',
    ):
        require(index, fragment, "index.html", errors)
    for fragment in ("data/resources.json", "a.status === '已发布'"):
        require(script, fragment, "assets/site.js", errors)

    resources_path = ROOT / "data" / "resources.json"
    sitemap_path = ROOT / "sitemap.xml"
    if not resources_path.is_file():
        errors.append("data/resources.json was not generated")
    else:
        resources = json.loads(resources_path.read_text(encoding="utf-8"))
        ids = [item["id"] for item in resources]
        if len(resources) < 100:
            errors.append(f"resources.json only has {len(resources)} items")
        if len(ids) != len(set(ids)):
            errors.append("resources.json contains duplicate ids")
        if not all(item["category"] and item["type"] and item["level"] for item in resources):
            errors.append("resources.json contains incomplete classification fields")
        if not all(item["status"] in {"已发布", "建设中"} for item in resources):
            errors.append("resources.json contains a non-public status")
        if not any(item["id"] == "external-learn-workbuddy" for item in resources):
            errors.append("resources.json is missing the curated learn-workbuddy entry")

    if not sitemap_path.is_file():
        errors.append("sitemap.xml was not generated")
    else:
        sitemap = sitemap_path.read_text(encoding="utf-8")
        for location in (
            "https://adongwanai.github.io/AgentGuide/",
            "https://adongwanai.github.io/AgentGuide/research/",
            "https://adongwanai.github.io/AgentGuide/interview/questions/q-1/",
        ):
            require(sitemap, f"<loc>{location}</loc>", "sitemap.xml", errors)

    if errors:
        print("Root site smoke test failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Root static site and generated deployment data passed smoke checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
