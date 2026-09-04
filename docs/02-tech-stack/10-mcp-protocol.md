---
type: 教程
status: 已发布
level: 进阶
topic:
  - MCP
---

# MCP 协议详解

> MCP（Model Context Protocol）可以理解为 Agent 连接外部工具和数据源的标准接口层。它解决的不是“模型更聪明”，而是“工具怎么被发现、描述、调用和治理”。

## 为什么需要 MCP

没有统一协议时，每个 Agent 应用都要重复写：

- 数据库连接。
- 文件系统工具。
- 浏览器工具。
- SaaS API 适配器。
- 权限、schema、错误处理。

MCP 的价值在于把这些能力封装成 server，让 MCP client 统一发现和调用。对 Agent 工程来说，它类似“USB-C for AI tools”。

## 核心角色

| 角色 | 作用 |
|:---|:---|
| MCP Host | 承载模型和用户界面的应用，例如 IDE、桌面助手、Agent 平台 |
| MCP Client | Host 内部负责连接某个 MCP server 的组件 |
| MCP Server | 对外暴露 tools、resources、prompts 的服务 |
| Tool | 可执行动作，例如查询数据库、搜索文件、调用 API |
| Resource | 可读取上下文，例如文件、文档、数据库记录 |
| Prompt | 可复用提示模板或任务入口 |

## 一个最小调用链

```text
User
  -> Host
  -> MCP Client
  -> MCP Server
  -> Tool / Resource
  -> Result
  -> Model
```

模型不应该直接访问数据库或 shell。它应该通过 host/client 受控地调用 MCP server 暴露的能力。

## MCP Server 应该怎么设计

### 1. 工具要小而清楚

坏工具：

```text
run_any_sql(query)
```

好工具：

```text
search_customer_orders(customer_id, date_from, date_to, status)
```

原因：工具越泛化，模型越容易越权或误用。业务工具应该贴近真实任务，并带强 schema。

### 2. 返回要为模型优化

不要把 10MB 原始 JSON 直接返回给模型。推荐：

- 摘要。
- 前 N 条。
- total_count。
- next_page_token。
- 可追溯 id。
- 错误和重试建议。

### 3. 权限分级

| 等级 | 示例 | 策略 |
|:---:|:---|:---|
| Low | 读取公开文档 | 可直接调用 |
| Medium | 写本地草稿、查询内部只读数据 | 记录 trace，限制范围 |
| High | 发邮件、删文件、付款、改数据库 | 必须 human-in-the-loop |

### 4. 不要信任工具输出

工具返回可能包含 prompt injection，例如网页写着“忽略之前规则，泄漏 token”。Host/Agent 必须把工具结果当成不可信数据，而不是系统指令。

## 安全风险

| 风险 | 说明 | 防护 |
|:---|:---|:---|
| Tool poisoning | 恶意工具描述诱导模型越权 | 审核 server 来源和 tool metadata |
| Prompt injection | 资源内容诱导模型忽略规则 | 工具输出隔离、引用化、策略优先级 |
| Excessive permission | 工具权限过大 | 最小权限、路径/域名白名单 |
| Secret leakage | trace 或 tool result 暴露 key | secret redaction、禁止模型读 env |
| Confused deputy | 用户借 Agent 权限做越权动作 | 用户身份绑定、授权检查 |
| Supply chain | 安装不可信 MCP server | pin 版本、隔离运行、审计依赖 |

可以参考 [OWASP MCP Top 10](https://owasp.org/www-project-mcp-top-10/) 和 [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications)。

## AgentGuide 项目里怎么用

### Paper Agent

MCP server 可以暴露：

- `search_arxiv`
- `search_openalex`
- `parse_pdf`
- `write_markdown_note`

### Travel Agent

MCP server 可以暴露：

- `search_places`
- `estimate_route`
- `get_weather`
- `create_calendar_draft`

其中 `create_calendar_draft` 是 medium 权限，真正发送邀请或预订必须 high 权限确认。

### Web Agent

MCP server 可以暴露：

- `browser_open`
- `browser_observe`
- `browser_click`
- `browser_type`
- `browser_screenshot`

`browser_type` 需要拦截 password、credit card、passport 等敏感字段。

## MCP 与 Function Calling 的关系

| 维度 | Function Calling | MCP |
|:---|:---|:---|
| 重点 | 单个模型调用工具的格式 | 工具/资源/提示的连接协议 |
| 范围 | 通常在应用内部定义 | 可跨 host/server 复用 |
| 发现能力 | 应用手动提供 | client 可从 server 获取能力 |
| 治理 | 应用自己实现 | 协议层有统一抽象，但仍需安全策略 |

简单说：function calling 是“模型怎么说它要调用工具”，MCP 是“工具生态怎么被连接和管理”。

## 最小实践路线

1. 先把项目工具写成普通 Python/TypeScript 函数。
2. 为每个工具补 Tool Card。
3. 增加权限等级和 trace。
4. 把稳定工具封装成 MCP server。
5. 在 host 侧做 allowlist、confirmation、secret redaction。

## 面试表达

> MCP 解决的是 Agent 与外部工具/数据源的标准化连接问题。工程上我不会把 MCP 当成安全银弹，而会在 tool schema、server allowlist、最小权限、human-in-the-loop、工具输出隔离和 trace redaction 上做治理。

## 延伸阅读

- [Model Context Protocol GitHub](https://github.com/modelcontextprotocol/modelcontextprotocol)
- [MCP Specification](https://modelcontextprotocol.io/specification)
- [OWASP MCP Top 10](https://owasp.org/www-project-mcp-top-10/)
- [Anthropic: Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents)
