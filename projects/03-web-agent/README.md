---
type: 项目蓝图
status: 已发布
level: 进阶
topic:
  - 项目实战
  - Coding Agent
  - 评测
---

# Web Agent：浏览器任务自动化与评测

> 项目目标：构建一个只操作公开网页的 Web Agent，完成信息查找、表单填写、页面导航等任务，并通过截图、DOM 和 trace 复盘失败。

Web Agent 是 Agent 工程里最接近真实世界的一类项目：网页会变、按钮会失效、登录和验证码会出现、DOM 和视觉信息不一致。做好这个项目，面试里很容易讲出工程深度。

## 安全边界

这个项目默认只做公开网页和测试页面，不做：

- 绕过登录、验证码、付费墙或网站访问控制。
- 自动输入真实账号、密码、支付信息、身份证件。
- 自动下单、付款、删除、发布不可撤销内容。
- 高频抓取或违反网站 robots / ToS 的行为。

## MVP 范围

| 能力 | MVP 做法 | 进阶做法 |
|:---|:---|:---|
| 页面访问 | Playwright 打开 URL | 浏览器状态持久化 |
| 观察页面 | DOM 文本 + 可访问性树 | 截图 + 多模态模型 |
| 动作执行 | click、type、select、scroll | 文件上传、下载、拖拽 |
| 任务规划 | 单 Agent loop | planner + executor + verifier |
| 失败复盘 | screenshot + trace.jsonl | DOM diff、video、span tracing |
| 评测 | 本地测试网页 + 公开文档页面 | WebArena / BrowserGym 风格任务 |

## 推荐架构

```text
Task
  -> Web Planner
  -> Browser State Observer
      - URL
      - title
      - accessibility tree
      - visible text
      - screenshot
  -> Action Selector
  -> Playwright Executor
  -> Verifier
  -> Trace / Screenshot / Report
```

## 核心工具设计

| 工具 | 输入 | 输出 | 注意点 |
|:---|:---|:---|:---|
| `browser_open` | url | title、url、状态 | 限制域名或提示风险 |
| `browser_observe` | none | 可见文本、元素摘要、截图路径 | 不把完整 DOM 全塞给模型 |
| `browser_click` | selector / text | 成功与否、页面变化 | 优先可访问性 selector |
| `browser_type` | selector、text | 成功与否 | 禁止输入敏感信息 |
| `browser_extract` | instruction | 结构化信息 | 保留来源 URL |
| `browser_screenshot` | none | 图片路径 | 失败时必留证据 |
| `ask_human` | reason、options | 用户选择 | 登录、付款等高风险动作 |

## 观察层怎么做

不要把完整 HTML 直接塞给模型。建议观察层输出：

```json
{
  "url": "https://example.com/docs",
  "title": "Docs",
  "visible_text_summary": "Install section is visible...",
  "interactive_elements": [
    {"role": "link", "name": "Quickstart", "selector": "text=Quickstart"},
    {"role": "button", "name": "Copy", "selector": "button[aria-label='Copy']"}
  ],
  "screenshot_path": "traces/run-001/step-003.png"
}
```

## Eval 设计

### 本地可控任务

先做一个本地测试站点，覆盖：

- 找到指定文本。
- 点击 tab 或 accordion。
- 填写表单但不提交敏感数据。
- 从表格里抽取信息。
- 处理按钮 disabled、网络超时、页面跳转。

### 公开网页任务

选择稳定、公开、低风险的页面：

- 官方文档中找到安装命令。
- GitHub README 中抽取运行步骤。
- arXiv 页面抽取标题、作者和 PDF 链接。

### Benchmark 延伸

- [WebArena](https://webarena.dev/)：真实网站任务评测。
- [VisualWebArena](https://github.com/web-arena-x/visualwebarena)：加入视觉理解的网页任务。
- [BrowserGym](https://github.com/ServiceNow/BrowserGym)：Web agent 环境和 benchmark 工具。
- [Mind2Web](https://github.com/OSU-NLP-Group/Mind2Web)：网页任务理解数据集。
- [OSWorld](https://github.com/xlang-ai/OSWorld)：桌面计算机使用评测。

## 指标

| 指标 | 含义 |
|:---|:---|
| `task_success_rate` | 任务最终是否完成 |
| `step_success_rate` | 单步动作是否产生预期页面变化 |
| `avg_steps` | 平均动作数，过多说明规划或观察有问题 |
| `recovery_rate` | 失败后是否能重试或改选路径 |
| `unsafe_action_block_rate` | 高风险动作是否被拦截 |
| `evidence_completeness` | 是否保存截图、URL、DOM 摘要和动作轨迹 |

## 实现里程碑

### Milestone 1：浏览器工具封装

- 用 Playwright 实现 open、observe、click、type、screenshot。
- 每步保存 trace。
- 禁止真实密码、支付和个人证件字段。

### Milestone 2：本地任务评测

- 做 10 个本地网页任务。
- 写自动 verifier。
- 统计成功率和失败原因。

### Milestone 3：公开网页任务

- 对 5 个官方文档或 GitHub 页面执行信息抽取。
- 保留 URL 和引用。
- 处理页面变化和超时。

### Milestone 4：多模态观察

- 加截图观察。
- 比较 DOM-only 和 DOM+vision 的成功率。
- 输出消融实验报告。

## 简历表达

> 构建公开网页任务 Web Agent，封装 Playwright 浏览器工具和结构化观察层，基于 URL、可访问性树、交互元素摘要和截图进行动作选择；设计本地测试站点与公开文档任务共 20 条 eval case，记录每步截图、DOM 摘要和动作 trace，统计任务成功率、平均步数、恢复率和高风险动作拦截率。

## 参考项目与资料

- [Playwright](https://playwright.dev/)
- [BrowserGym](https://github.com/ServiceNow/BrowserGym)
- [WebArena](https://webarena.dev/)
- [VisualWebArena](https://github.com/web-arena-x/visualwebarena)
- [Mind2Web](https://github.com/OSU-NLP-Group/Mind2Web)
- [OSWorld](https://github.com/xlang-ai/OSWorld)
- [OpenAI Agents SDK](https://platform.openai.com/docs/guides/agents-sdk/)
