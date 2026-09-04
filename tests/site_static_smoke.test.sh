#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

node --check "$ROOT_DIR/assets/site.js"

# 使用 grep -qF（固定字符串）而非 rg：ripgrep 未必存在于 CI runner 镜像中，
# 而这些断言全部是字面量匹配，不需要正则。
grep -qF '<link rel="canonical" href="https://adongwanai.github.io/AgentGuide/">' "$ROOT_DIR/index.html"
grep -qF '<meta property="og:image" content="https://adongwanai.github.io/AgentGuide/assets/agentguide-social.png">' "$ROOT_DIR/index.html"
grep -qF 'application/ld+json' "$ROOT_DIR/index.html"
grep -qF 'data/resources.json' "$ROOT_DIR/assets/site.js"
grep -qF 'data-github-forks' "$ROOT_DIR/index.html"
grep -qF '<option value="通用">通用</option>' "$ROOT_DIR/index.html"
grep -qF "a.status === '已发布'" "$ROOT_DIR/assets/site.js"

python3 - "$ROOT_DIR/data/resources.json" <<'PY'
import json
import sys
from pathlib import Path

resources = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
ids = [item["id"] for item in resources]
# 下限用于捕捉「生成器产出为空或严重损坏」，不用于统计内容数量。
# 原值 120 贴着当时的实际条数（121），删除任意一篇文档都会误报。
assert len(resources) >= 100, f"resources.json only has {len(resources)} items"
assert len(ids) == len(set(ids))
assert all(item["category"] and item["type"] and item["level"] for item in resources)
assert all(item["status"] in {"已发布", "建设中"} for item in resources)
assert any(item["id"] == "external-learn-workbuddy" for item in resources)
PY

grep -qF '<loc>https://adongwanai.github.io/AgentGuide/</loc>' "$ROOT_DIR/sitemap.xml"
grep -qF '<loc>https://adongwanai.github.io/AgentGuide/research/</loc>' "$ROOT_DIR/sitemap.xml"
grep -qF '<loc>https://adongwanai.github.io/AgentGuide/interview/questions/q-1/</loc>' "$ROOT_DIR/sitemap.xml"
