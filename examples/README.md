# Examples：可复制的 Agent 最小样例

> 这里不是放大型项目，而是放你可以直接迁移到自己项目里的“小骨架”：loop、tool schema、trace、eval、README 模板。

## 文件说明

| 文件 | 用途 |
|:---|:---|
| [minimal-agent-loop.md](./minimal-agent-loop.md) | 50-150 行最小 Agent Loop 的设计模板 |
| [tool-card-template.md](./tool-card-template.md) | 工具说明卡模板，适合放进 README 或 docs |
| [trace-schema.json](./trace-schema.json) | Agent 运行轨迹 JSONL 的单条记录 schema |
| [eval-cases.jsonl](./eval-cases.jsonl) | 10 条入门 eval case，可扩展到项目中 |
| [project-readme-template.md](./project-readme-template.md) | 简历级 Agent 项目的 README 模板 |

## 推荐用法

1. 先读 `minimal-agent-loop.md`，理解 Agent 最小闭环。
2. 用 `tool-card-template.md` 给你的工具写清楚边界。
3. 每次运行 Agent 都写入 `trace-schema.json` 格式的 JSONL。
4. 用 `eval-cases.jsonl` 的格式为你的项目扩展 20-50 条评测。
5. 项目完成后套用 `project-readme-template.md`，补齐架构、运行、评测和失败分析。

最小循环的回归测试可以在仓库根目录运行：

```bash
python tests/agent_loop.test.py
```

测试直接执行模板中的 Python 代码，用脚本模型和模拟工具检查未知工具、人工拒绝、超时、工具异常、正常完成与步数上限，不需要 API Key 或第三方依赖。

## 一个简历级项目最少要交付什么

- `README.md`：别人能 clone 跑起来。
- `docs/spec.md`：目标用户、任务、约束、非目标。
- `docs/eval_report.md`：评测集、指标、失败分析。
- `traces/*.jsonl`：至少保留 3-5 条代表性运行轨迹。
- `examples/*.md`：演示脚本和输入样例。

## 相关项目蓝图

- [Paper Agent](../projects/01-paper-agent/README.md)
- [Travel Agent](../projects/02-travel-agent/README.md)
- [Web Agent](../projects/03-web-agent/README.md)
