#!/usr/bin/env python3
"""Smoke-test the exact directory tree uploaded to GitHub Pages."""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def main() -> int:
    site = Path(sys.argv[1] if len(sys.argv) > 1 else "site-dist").resolve()
    required = (
        "index.html",
        "404.html",
        "robots.txt",
        "site.webmanifest",
        ".nojekyll",
        "assets/site.js",
        "data/resources.json",
        "sitemap.xml",
        "research/index.html",
        "research/docs/intro/01-overview/index.html",
        "interview/index.html",
        "interview/questions/q-1/index.html",
    )
    missing = [relative for relative in required if not (site / relative).is_file()]
    if missing:
        print("Pages assembly smoke test failed; missing files:")
        for relative in missing:
            print(f"- {relative}")
        return 1

    resources = json.loads((site / "data/resources.json").read_text(encoding="utf-8"))
    if len(resources) < 100:
        print(f"Pages assembly smoke test failed; only {len(resources)} resources")
        return 1

    sitemap = ET.parse(site / "sitemap.xml")
    namespace = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    locations = {
        node.text for node in sitemap.findall("s:url/s:loc", namespace) if node.text
    }
    expected_urls = {
        "https://adongwanai.github.io/AgentGuide/",
        "https://adongwanai.github.io/AgentGuide/research/",
        "https://adongwanai.github.io/AgentGuide/interview/questions/q-1/",
    }
    if missing_urls := expected_urls - locations:
        print("Pages assembly smoke test failed; sitemap is missing:")
        for url in sorted(missing_urls):
            print(f"- {url}")
        return 1

    print(f"Pages assembly smoke test passed for {site}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
