---
type: 教程
status: 已发布
level: 进阶
topic:
  - Coding Agent
---

# Claude Code 最佳实践指南

> 你的 Claude Code 完整操作手册 - 从入门到精通

## 📌 文档说明

**文档定位**：Claude Code 完整使用指南，涵盖从基础配置到高级工作流的全流程

**适合人群**：
- AI Agent 开发者想要提升编程效率
- 对 AI 辅助编程感兴趣的工程师
- 希望掌握 Claude Code 的团队

**核心价值**：
- ✅ 系统化的学习路径（从入门到精通）
- ✅ 实战验证的最佳实践
- ✅ 企业级的工作流模式
- ✅ 完整的命令速查表

**阅读时间**：约 60-90 分钟（建议分段学习）

---

## 📖 目录

- [快速入门](#快速入门)
- [核心配置](#核心配置)
- [基础命令](#基础命令)
- [工作流模式](#工作流模式)
- [高级功能](#高级功能)
- [性能优化](#性能优化)
- [自动化与集成](#自动化与集成)
- [最佳实践总结](#最佳实践总结)
- [命令速查表](#命令速查表)

---

## 快速入门

### 安装与启动

```bash
# 安装 Claude Code
curl -sL https://install.anthropic.com | sh

# 或使用 npm 安装
npm install -g @anthropic-ai/claude-code

# 启动交互式 REPL
claude

# 带初始提示启动
claude "总结这个项目"

# 检查版本
claude --version

# 更新到最新版本
claude update
```

### 基本操作

```bash
# 打印模式 - 执行后退出
claude -p "解释这个函数"

# 处理管道内容
cat logs.txt | claude -p "分析日志"

# 继续最近的对话
claude -c

# 通过 ID 恢复会话
claude -r "abc123" "完成这个 PR"
```

---

## 核心配置

### 1. CLAUDE.md - 最重要的文件

`CLAUDE.md` 是你代码库的"宪法"，Claude 会自动读取此文件作为上下文。

#### 创建位置

- **仓库根目录**：`CLAUDE.md`（团队共享，提交到 git）
- **本地版本**：`CLAUDE.local.md`（个人使用，加入 .gitignore）
- **home 目录**：`~/.claude/CLAUDE.md`（全局配置）
- **父目录或子目录**：支持 monorepo 场景

#### 内容建议

```markdown
# 项目名称

## Python 开发规范
- 始终使用 Python 3.11+
- 测试命令：pytest tests/
- 使用 pyenv 管理 Python 版本

## 常用 Bash 命令
- npm run build: 构建项目
- npm run typecheck: 运行类型检查

## 代码风格
- 使用 ES 模块语法（import/export），不使用 CommonJS（require）
- 尽可能解构导入（如：import { foo } from 'bar'）

## 内部工具
### <工具名称>
- 用途：xxx
- 使用示例：tool-cli --flag value
- 重要：绝不使用 --dangerous-flag，优先使用 --safe-flag

对于复杂用法或遇到 FooBarError，参见 path/to/docs.md

## 工作流
- 完成代码修改后务必进行类型检查
- 优先运行单个测试而非整个测试套件（性能考虑）
- 使用 rebase 而非 merge 来合并分支
```

#### 编写技巧和反模式

**✅ 应该做的：**

1. **从防护栏开始，不是手册**：根据 Claude 出错的地方来编写文档，而不是一开始就写全面的手册

2. **提供替代方案**：不要只说"绝不使用 X"，要说"不要使用 X，优先使用 Y"

3. **引导阅读其他文档**：不要 @-引用整个文件（会膨胀上下文），而是说明何时阅读
   ```markdown
   对于复杂的 API 使用或遇到 AuthenticationError，参见 docs/api-guide.md
   ```

4. **作为简化工具的驱动力**：如果 CLI 命令复杂冗长，不要写长篇文档解释，而是写一个简单的 bash 包装器

5. **使用 # 快捷键**：在 Claude Code 中按 `#` 键，让 Claude 自动将内容合并到相关的 CLAUDE.md 中

**❌ 不应该做的：**

- ❌ 不要在 CLAUDE.md 中 @-引用其他文档文件（会完整嵌入，浪费 token）
- ❌ 不要只写否定约束（"绝不使用..."），总是提供替代方案
- ❌ 不要让文件过于冗长，保持简洁（建议 13-25KB）

#### 维护策略

- **迭代优化**：使用 [prompt improver](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/prompt-improver) 优化指令
- **强调重点**：用 "IMPORTANT" 或 "YOU MUST" 强调关键指令
- **团队同步**：维护一份 `AGENTS.md` 与其他 AI IDE 保持兼容

### 2. settings.json 配置

位置：`.claude/settings.json` 或 `~/.claude.json`

```json
{
  "permissions": {
    "allowedTools": [
      "Edit",
      "Bash(git:*)",
      "Bash(git commit:*)"
    ],
    "allowedDomains": [
      "docs.example.com"
    ]
  },
  "ANTHROPIC_API_KEY": "your-key-here",
  "HTTPS_PROXY": "http://localhost:8080",
  "MCP_TOOL_TIMEOUT": 60000,
  "BASH_MAX_TIMEOUT_MS": 300000
}
```

**关键配置项：**

- **HTTPS_PROXY/HTTP_PROXY**：用于调试，检查原始流量和提示
- **MCP_TOOL_TIMEOUT/BASH_MAX_TIMEOUT_MS**：提高超时时间，适合运行长时间命令
- **ANTHROPIC_API_KEY**：使用企业 API key，按使用量计费而非按座位
- **permissions**：定期审计自动允许的命令列表

### 3. 工具权限管理

有四种方式管理允许的工具：

```bash
# 方式 1: 在会话中被提示时选择 "Always allow"

# 方式 2: 使用 /permissions 命令
/permissions
# 然后添加：Edit, Bash(git commit:*), mcp__puppeteer__puppeteer_navigate

# 方式 3: 手动编辑 settings.json（推荐团队共享）

# 方式 4: 使用 CLI 标志（会话级）
claude --allowedTools "Bash(git:*)" "Write" "Read"
```

**推荐白名单：**

```bash
# 安全的文件操作
--allowedTools "Edit" "Write" "Read"

# Git 操作
--allowedTools "Bash(git:*)" "Bash(git commit:*)"

# 禁止危险操作
--disallowedTools "Bash(rm:*)" "Bash(sudo:*)" "Bash(chmod:*)"
```

### 4. 安装 gh CLI（GitHub 用户）

```bash
# 安装 GitHub CLI
brew install gh  # macOS
# 或参考：https://cli.github.com

# 认证
gh auth login
```

Claude 会自动使用 `gh` CLI 与 GitHub 交互（创建 issue、开启 PR、读取评论等）。

---

## 基础命令

### 会话管理命令

```bash
/help                     # 显示帮助和可用命令
/exit                     # 退出 REPL
/clear                    # 清除对话历史（重要！频繁使用）
/config                   # 打开配置面板
/doctor                   # 检查 Claude Code 安装健康状况
/context                  # 查看当前上下文使用情况
/compact [说明]           # 压缩对话（不推荐，见性能优化）
/cos                      # 显示当前会话的成本和时长
/permissions              # 管理工具权限白名单
/ide                      # 管理 IDE 集成
/mcp                      # 访问 MCP 功能
/init                     # 自动生成 CLAUDE.md 文件
```

### 模型选择

```bash
# 切换模型
claude --model sonnet                    # 使用 Sonnet（默认）
claude --model opus                      # 使用 Opus
claude --model claude-sonnet-4-20250514  # 使用特定版本
```

### 目录管理

```bash
# 添加额外的工作目录
claude --add-dir ../apps ../lib

# 验证目录路径
claude --add-dir /path/to/project
```

### 输出格式

```bash
# 不同的输出格式
claude -p "query" --output-format json
claude -p "query" --output-format text
claude -p "query" --output-format stream-json

# 输入格式
claude -p --input-format stream-json

# 限制对话轮次
claude -p --max-turns 3 "query"

# 详细日志
claude --verbose
```

### 键盘快捷键

```
Ctrl+C          取消当前操作
Ctrl+D          退出 Claude Code
Tab             自动补全（文件名、命令等）
Up/Down         浏览命令历史
Escape          中断 Claude（保留上下文，可重定向）
Escape x2       跳回历史记录，编辑之前的提示
Shift+Tab       切换自动接受模式
```

---

## 工作流模式

### 1. 探索 → 规划 → 编码 → 提交（推荐）

这是最通用且有效的工作流：

```bash
# 第 1 步：探索代码库
> 阅读处理日志的所有文件，暂时不要编写任何代码

# （可选）使用子智能体深入调查
> 使用子智能体验证日志系统的错误处理机制

# 第 2 步：制定计划（使用扩展思考）
> think hard：如何重构日志系统以支持结构化日志？制定详细计划

# think 的层级：
# "think" < "think hard" < "think harder" < "ultrathink"

# 第 3 步：（可选）记录计划
> 将你的计划写入 PLAN.md，以便稍后可以重置到这个点

# 第 4 步：实现
> 根据计划用代码实现解决方案，在实现各部分时验证合理性

# 第 5 步：提交
> 提交结果并创建 pull request，同时更新 README 说明你的修改
```

**关键点：**
- 步骤 1-2 至关重要，防止 Claude 直接跳到编码
- 使用 "think" 关键词触发扩展思考模式
- 考虑使用子智能体来保留主上下文的可用性

### 2. 测试驱动开发（TDD）- Anthropic 内部最爱

```bash
# 第 1 步：编写测试
> 我们要进行测试驱动开发。基于预期的输入/输出编写测试，
> 不要创建模拟实现，即使功能还不存在

# 第 2 步：确认测试失败
> 运行测试并确认它们失败。不要编写任何实现代码

# 第 3 步：提交测试
> 提交这些测试

# 第 4 步：实现功能
> 编写能通过测试的代码，不要修改测试。持续迭代直到所有测试通过

# 第 5 步：验证过拟合
> 使用独立的子智能体验证实现是否对测试过度拟合

# 第 6 步：提交代码
> 提交代码并创建 PR
```

**优势：**
- Claude 有明确的迭代目标（通过测试）
- 可以持续改进直到成功
- 防止过度设计

### 3. 视觉迭代工作流（UI 开发）

```bash
# 第 1 步：设置截图能力
> 使用 Puppeteer MCP 或手动截图

# 第 2 步：提供设计稿
> 这是设计稿（拖放图片或粘贴截图）

# macOS 截图快捷键：
# cmd+ctrl+shift+4（截图到剪贴板）
# 然后 ctrl+v 粘贴（注意不是 cmd+v）

# 第 3 步：迭代实现
> 用代码实现设计，截图当前结果，比对设计稿，迭代直到匹配

# 第 4 步：提交
> 满意后提交代码
```

**技巧：**
- 经过 2-3 次迭代，结果通常会好得多
- 明确告诉 Claude 结果要"视觉上美观"

### 4. 代码库问答（入职利器）

```bash
# 直接提问，无需特殊提示
> 日志系统是如何工作的？
> 我如何创建一个新的 API 端点？
> foo.rs 文件第 134 行的 async move { ... } 是做什么的？
> CustomerOnboardingFlowImpl 处理了哪些边界情况？
> 为什么我们在第 333 行调用 foo() 而不是 bar()？
> baz.py 文件第 334 行在 Java 中对应的实现是什么？
```

**应用场景：**
- 新人入职，快速了解代码库
- 理解复杂的代码逻辑
- 查找代码的历史原因

### 5. Git 工作流（90%的 Git 交互）

```bash
# 搜索 git 历史
> v1.2.3 版本包含了哪些更改？
> 谁是这个特定功能的所有者？请查阅 git 历史
> 为什么这个 API 是这样设计的？查看 git blame 和历史

# 自动编写提交信息
> 查看我的更改和最近的历史记录，写一条提交信息

# 处理复杂操作
> 恢复 src/auth.py 的上一个版本
> 解决这个 rebase 冲突
> 比较 feature-a 和 feature-b 分支的差异
```

### 6. GitHub 工作流

```bash
# 创建 PR
> pr  # Claude 理解 "pr" 简写

# 解决代码审查评论
> 修复我 PR 上的所有评论，然后推送

# 修复失败的构建
> 查看 CI 失败日志，修复问题并推送

# 分类 issues
> 遍历所有开放的 GitHub issues，为它们添加适当的标签
```

### 7. 安全的 YOLO 模式

```bash
# 跳过所有权限检查（危险！）
claude --dangerously-skip-permissions

# 适用场景：
# - 修复 lint 错误
# - 生成样板代码
# - 快速原型开发

# ⚠️ 安全建议：
# 在没有互联网访问的 Docker 容器中使用
# 参考：https://github.com/anthropics/claude-code/.devcontainer
```

### 8. 清单和草稿板工作流（大型任务）

```bash
# 处理大量 lint 错误
> 运行 lint 命令，将所有错误（包括文件名和行号）写入 LINT_TODO.md

> 逐个解决 LINT_TODO.md 中的问题，修复并验证后勾选，然后继续下一个

# 适用场景：
# - 代码迁移
# - 批量重构
# - 复杂的构建脚本
```

### 9. Jupyter Notebook 工作流

```bash
# 数据分析
> 读取 analysis.ipynb，解释最后一个图表的输出

# 清理 notebook
> 清理这个 notebook，让它看起来美观，为同事展示做准备
> 明确：优化数据可视化的观看体验

# 推荐工作方式：
# 在 VS Code 中并排打开 Claude Code 和 .ipynb 文件
```

---

## 高级功能

### 1. 上下文管理（Compact vs Clear）

#### 查看上下文使用情况

```bash
/context  # 查看 200k token 上下文窗口的使用情况
```

在大型代码库（如 monorepo）中，新会话基线成本约 20k tokens（10%），剩余 180k 用于修改。

#### 三种上下文管理策略

**策略 1：/compact（避免使用）**

```bash
/compact "保留重要部分"
```

❌ **不推荐原因：**
- 自动压缩不透明、容易出错、优化不佳
- 可能丢失关键上下文

**策略 2：/clear + /catchup（简单重启）**

```bash
/clear  # 清除状态

# 创建自定义命令：.claude/commands/catchup.md
# 内容：Read all changed files in my git branch

/catchup  # 读取分支中所有变更的文件
```

✅ **推荐用于：** 简单任务、快速重启

**策略 3：Document & Clear（复杂重启）**

```bash
# 步骤 1：记录进度
> 将你的计划和当前进度详细记录到 PROGRESS.md

# 步骤 2：清除上下文
/clear

# 步骤 3：恢复并继续
> 阅读 PROGRESS.md 并继续工作
```

✅ **推荐用于：** 大型任务、长时间工作

### 2. 自定义斜杠命令

位置：`.claude/commands/` 或 `~/.claude/commands`

**示例 1：catchup 命令**

`.claude/commands/catchup.md`：
```markdown
Read all changed files in my current git branch and summarize the changes.
```

使用：`/catchup`

**示例 2：修复 GitHub issue**

`.claude/commands/fix-github-issue.md`：
```markdown
Please analyze and fix the GitHub issue: $ARGUMENTS.

Follow these steps:
1. Use `gh issue view` to get the issue details
2. Understand the problem described in the issue
3. Search the codebase for relevant files
4. Implement the necessary changes to fix the issue
5. Write and run tests to verify the fix
6. Ensure code passes linting and type checking
7. Create a descriptive commit message
8. Push and create a PR

Remember to use the GitHub CLI (`gh`) for all GitHub-related tasks.
```

使用：`/project:fix-github-issue 1234`

**示例 3：PR 命令**

`.claude/commands/pr.md`：
```markdown
Clean up my code, stage it, and prepare a pull request with a descriptive title and description.
```

使用：`/pr`

**⚠️ 注意：**
- 不要创建过多复杂的自定义命令（反模式）
- 斜杠命令应该是简单的个人快捷方式，不是新的"魔法命令"系统

### 3. 子智能体（Subagents）

#### 内置方式：Task(...)（推荐）

```bash
# 让主 Agent 决定何时委托
> 分析这个项目结构，必要时使用 Task(...) 委托子任务给你的克隆

# Claude 会自动使用 Task(...) 或 Explore(...) 创建克隆
```

✅ **优势：**
- 主 Agent 保留所有上下文（CLAUDE.md）
- 动态委托，由 Agent 自己决定如何分工
- "Master-Clone" 架构，灵活性高

#### 自定义子智能体（不推荐）

```bash
# 创建专门的子智能体
# 例如：.claude/subagents/python-tests.md
```

❌ **问题：**
1. **隔离上下文**：主 Agent 失去测试相关的上下文，无法整体推理
2. **强制工作流**：将人类定义的工作流强加给 Agent
3. **降低灵活性**：Agent 无法根据实际情况调整策略

**结论：** 使用 Task(...) 让主 Agent 自己管理委托，而不是预定义专门的子智能体。

### 4. MCP（Model Context Protocol）

#### MCP 的正确使用方式

**不要做：** ❌ 创建臃肿的 API 镜像
```bash
# 糟糕的 MCP 设计：
read_thing_a(), read_thing_b(), read_thing_c(),
update_thing_a(), update_thing_b(), ...
# （几十个工具，只是镜像 REST API）
```

**应该做：** ✅ 提供高级数据网关
```bash
# 好的 MCP 设计：
download_raw_data(filters...)
take_sensitive_gated_action(args...)
execute_code_in_environment_with_state(code...)
```

#### 配置 MCP

```bash
# 配置 MCP 服务器
claude --mcp

# 或通过 /mcp 命令
/mcp

# 调试 MCP
claude --mcp-debug
```

#### 三种配置方式

1. **项目配置**：在项目目录中可用
2. **全局配置**：在所有项目中可用
3. **检入的 .mcp.json**：对团队所有人可用

**示例 .mcp.json**：
```json
{
  "mcpServers": {
    "puppeteer": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-puppeteer"]
    },
    "sentry": {
      "command": "npx",
      "args": ["-y", "@your-org/sentry-mcp"]
    }
  }
}
```

#### 推荐的 MCP 使用场景

✅ **适合使用 MCP：**
- 复杂的、有状态的环境（如 Playwright、iOS Simulator）
- 需要认证和安全边界的操作
- 跨网络的数据访问

❌ **不适合使用 MCP：**
- 无状态的工具（用简单的 CLI 代替）
- 纯数据查询（提供原始数据下载 API，让 Agent 用脚本处理）

### 5. Skills（技能）- 比 MCP 更重要

#### 什么是 Skills

Skills 是 "Scripting Agent" 模型的形式化产品化：

**Agent 自主性的三个阶段：**

1. **Single Prompt**：在一个提示中给出所有上下文（不可扩展）
2. **Tool Calling**：手工制作工具，抽象现实（经典 Agent 模型，但创建新的抽象瓶颈）
3. **Scripting**：给 Agent 访问原始环境（二进制、脚本、文档），它动态编写代码交互（最灵活）

**Skills = Scripting 的形式化**

#### Skills vs MCP

- **MCP**：安全网关，管理认证、网络、安全边界
- **Skills**：可发现的、文档化的 CLI/脚本，Agent 可以动态调用

#### 如何使用 Skills

```bash
# Skills 本质上是文档化的 CLI 工具
# 在 CLAUDE.md 或专门的 SKILL.md 中记录它们

# SKILL.md 示例：
# Tool: internal-deploy-cli
# Usage: internal-deploy-cli --env prod --service api
# Description: Deploy services to production
# Examples: ...
```

**技巧：**
- 如果你已经在使用 CLI 而非 MCP，你已经在使用 Skills 的理念
- SKILL.md 只是让这些 CLI 更有组织、可共享、可发现

### 6. Hooks（钩子）- 企业级关键功能

#### 什么是 Hooks

Hooks 是确定性的"必须做"规则，补充 CLAUDE.md 中的"应该做"建议。

#### 两种 Hook 策略

**策略 1：Block-at-Submit Hooks（推荐）**

```javascript
// .claude/hooks/pre-commit.js
// PreToolUse hook 包装 Bash(git commit)

if (!fs.existsSync('/tmp/agent-pre-commit-pass')) {
  throw new Error('Tests must pass before commit. Run tests first.');
}
```

工作流：
1. Claude 编写代码
2. 尝试提交
3. Hook 检查 `/tmp/agent-pre-commit-pass` 文件
4. 如果测试未通过，阻止提交，Claude 进入"测试-修复"循环
5. 测试通过后，允许提交

✅ **优势：**
- 在最终结果处验证
- 让 Agent 完成完整的计划
- 不会中途"挫败" Agent

**策略 2：Hint Hooks（提示型）**

```javascript
// 简单的、非阻塞的提示
if (isSuboptimal(code)) {
  console.log('Hint: Consider using pattern X instead of Y');
}
```

✅ **适用于：** 提供优化建议，但不强制

**❌ 避免：Block-at-Write Hooks**

不要在 `Edit` 或 `Write` 时阻塞：
- 中途阻塞会让 Agent 困惑或"挫败"
- 破坏 Agent 的完整计划

### 7. 规划模式（Planning Mode）

```bash
# 启动规划模式
> 请制定一个详细计划来实现用户认证系统，
> 包括检查点，在每个检查点停下来展示你的工作

# 使用扩展思考
> think harder: 制定一个重构整个数据库层的计划
```

**最佳实践：**
- 对于任何"大型"功能变更都使用规划模式
- 定义"检查点"，让 Claude 在关键处停下来
- 使用规划建立对所需最小上下文的直觉

**企业级：自定义规划工具**

基于 Claude Code SDK 构建，强制执行内部最佳实践：
- 技术设计格式
- 代码结构规范
- 数据隐私和安全要求

### 8. 会话历史与恢复

```bash
# 恢复会话
claude --resume abc123 "继续之前的工作"

# 继续最近的会话
claude --continue

# 从历史中学习
> 总结你是如何解决上次的 AuthenticationError 的
```

**高级用法：挖掘历史数据**

```bash
# 会话历史位置
~/.claude/projects/

# 元分析脚本（自行编写）
# - 查找常见异常
# - 分析权限请求模式
# - 识别错误模式
# - 改进 CLAUDE.md 和工具
```

---

## 性能优化

### 1. 上下文管理技巧

```bash
# 频繁使用 /clear
/clear  # 在任务之间清除上下文

# 限制对话轮次
claude -p --max-turns 5 "focused query"

# 使用 /compact（谨慎）
/compact "保留重要部分"  # 仅在必要时使用

# 查看上下文使用
/context  # 监控 token 使用情况
```

### 2. 指令具体化

**❌ 糟糕的提示：**
```
修复这个 bug
```

**✅ 好的提示：**
```
修复 src/auth.py 中的认证 bug。问题是在用户登录后
session token 没有正确设置。请：
1. 检查 login() 函数
2. 确保 set_session_token() 被调用
3. 添加单元测试验证修复
4. 运行测试套件确保没有破坏其他功能
```

**关键原则：**
- Claude 可以推断意图，但不会读心术
- 具体性 = 更好的首次成功率 = 更少的迭代 = 更快的结果

### 3. 提供视觉和数据

#### 视觉输入

```bash
# 方式 1：粘贴截图
# macOS: cmd+ctrl+shift+4 截图到剪贴板
#        然后 ctrl+v 粘贴（不是 cmd+v！）

# 方式 2：拖放图片
# 直接拖放到提示输入框

# 方式 3：文件路径
> 查看 designs/mockup.png 并实现这个 UI
```

#### 数据输入

```bash
# 方式 1：直接粘贴（最常见）
> 分析这些数据：
> [粘贴 CSV 或 JSON]

# 方式 2：管道输入
cat logs.txt | claude -p "分析错误模式"

# 方式 3：让 Claude 拉取
> 使用 gh CLI 拉取最近 10 个 issues 并分类

# 方式 4：读取文件或 URL
> 读取 data/metrics.csv 并创建可视化
> 获取 https://docs.example.com/api 并总结
```

### 4. 提及文件和 URL

```bash
# 使用 Tab 补全快速引用文件
> 更新 src/[Tab] 中的认证逻辑
> 阅读 docs/[Tab] 了解 API 规范

# 提供 URL
> 获取并阅读 https://example.com/docs/api-guide
> 参考这个设计：https://figma.com/file/abc123

# 管理域名白名单
/permissions
# 添加 docs.example.com 到白名单
```

### 5. 路线修正技巧

#### 四种修正工具

**工具 1：提前规划**
```bash
> 制定一个详细计划来实现 X。在我确认计划可行之前，不要编写任何代码
```

**工具 2：中断（Escape）**
```
按 Escape：中断当前操作（思考、工具调用、文件编辑）
保留上下文，可以重定向或扩展指令
```

**工具 3：回退历史（Escape x2）**
```
双击 Escape：跳回历史记录
编辑之前的提示，探索不同方向
可以反复编辑和重试
```

**工具 4：撤销修改**
```bash
> 撤销你刚才对 auth.py 的修改
> 恢复到上一个版本，我们换个方法
```

**原则：**
- Claude 偶尔能一次成功，但修正通常能更快得到更好的结果
- 成为积极的合作者，而不是被动的观察者

### 6. 自动接受模式

```bash
# 切换自动接受模式
Shift+Tab

# 适用场景：
# - 简单任务
# - 高度信任的操作
# - 快速原型

# 注意：
# - 通常通过积极合作能得到更好的结果
# - 不要完全"放手"
```

---

## 自动化与集成

### 1. 无头模式（Headless Mode）

用于 CI、pre-commit hooks、构建脚本和自动化。

```bash
# 基本无头模式
claude -p "分析代码库" --output-format json > analysis.json

# 流式 JSON
claude -p "大型任务" --output-format stream-json

# 批量处理
claude -p --max-turns 1 "快速查询"
```

### 2. Issue 分类自动化

```bash
# GitHub Actions 示例
# .github/workflows/issue-triage.yml

name: Issue Triage
on:
  issues:
    types: [opened]

jobs:
  triage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - run: |
          claude -p "分析 issue #${{ github.event.issue.number }} 并添加适当的标签" \
            --output-format json \
            --allowedTools "Bash(gh:*)"
```

### 3. 主观代码审查

```bash
# 在 CI 中运行 Claude
# .github/workflows/code-review.yml

- name: Claude Code Review
  run: |
    git diff HEAD~1 | claude -p "
    审查这个 PR，检查：
    - 拼写错误
    - 过时的注释
    - 误导性的函数/变量名
    - 未处理的边界情况
    输出 JSON 格式的审查意见
    " --output-format json > review.json
```

### 4. 多 Claude 并行工作流

#### 模式 1：一个写代码，一个审查

```bash
# 终端 1
> 实现用户认证系统

# 等待完成后，终端 2
> 审查终端 1 中 Claude 的工作，检查安全性和最佳实践

# 终端 3（或终端 1 /clear）
> 阅读代码和审查反馈，根据反馈修改代码
```

#### 模式 2：多个 Git Checkouts

```bash
# 创建 3-4 个 git checkout
mkdir ../project-copy-{1,2,3,4}
git clone . ../project-copy-1
git clone . ../project-copy-2
# ...

# 在不同终端中启动
cd ../project-copy-1 && claude "实现功能 A"
cd ../project-copy-2 && claude "实现功能 B"
cd ../project-copy-3 && claude "修复 bug C"
cd ../project-copy-4 && claude "重构模块 D"

# 循环检查进度，批准权限请求
```

#### 模式 3：Git Worktrees（推荐）

```bash
# 创建 worktrees
git worktree add ../project-feature-a feature-a
git worktree add ../project-feature-b feature-b
git worktree add ../project-refactor-c refactor-c

# 在每个 worktree 中启动 Claude
cd ../project-feature-a && claude
cd ../project-feature-b && claude
cd ../project-refactor-c && claude

# 技巧：
# - 使用一致的命名约定
# - 每个 worktree 保持一个终端标签
# - Mac iTerm2 用户：设置通知提醒

# 清理
git worktree remove ../project-feature-a
```

**优势：**
- 共享 Git 历史，节省空间
- 独立的工作目录
- 适合并行的独立任务

### 5. 无头模式的两种模式

#### 模式 A：扇出（Fan-out）- 大规模处理

```bash
#!/bin/bash
# 大规模迁移或分析

# 步骤 1：生成任务列表
claude -p "生成需要从 React 迁移到 Vue 的文件列表" \
  --output-format json > tasks.json

# 步骤 2：循环处理
cat tasks.json | jq -r '.files[]' | while read file; do
  claude -p "将 $file 从 React 迁移到 Vue。\
    完成后返回 OK，失败返回 FAIL" \
    --allowedTools Edit Bash(git commit:*) \
    --max-turns 5
done
```

**适用场景：**
- 代码迁移（数千个文件）
- 日志分析（数百个日志文件）
- 批量数据处理

#### 模式 B：管道化（Pipeline）- 集成到数据流

```bash
# 集成到现有管道
cat data.csv | \
  process_step_1 | \
  claude -p "清理和标准化数据" --output-format json | \
  process_step_2 | \
  save_results

# 调试
claude -p "任务" --verbose  # 调试时使用
claude -p "任务"            # 生产环境关闭 verbose
```

### 6. Claude Code GitHub Action（GHA）

这是最强大的操作化 Claude Code 的方式。

```yaml
# .github/workflows/claude-pr.yml
name: Claude PR from Slack

on:
  repository_dispatch:
    types: [slack-request]

jobs:
  claude-fix:
    runs-on: ubuntu-latest
    container:
      image: anthropic/claude-code:latest
    
    steps:
      - uses: actions/checkout@v2
      
      - name: Run Claude
        run: |
          claude -p "${{ github.event.client_payload.request }}" \
            --allowedTools Edit Bash(git:*) \
            --dangerously-skip-permissions
      
      - name: Create PR
        run: |
          git push origin HEAD:auto-fix-${{ github.run_id }}
          gh pr create --title "Auto-fix from Claude" --body "..."
```

**应用场景：**
- 从 Slack 触发 PR
- 从 Jira 自动修复 bug
- 从 CloudWatch 告警自动修复问题

**数据驱动的改进循环：**

```bash
# 查询 GHA 日志，找到常见错误
query-claude-gha-logs --since 5d | \
  claude -p "看看其他 Claude 实例卡在什么地方，修复它，然后提交 PR"

# 改进流程：
# Bug -> 改进 CLAUDE.md / CLI -> 更好的 Agent
```

**优势：**
- 完全可定制的容器和环境
- 强大的沙箱和审计控制
- 支持所有高级功能（Hooks、MCP）
- 比 Cursor 后台 Agent 或 Codex Web UI 更可控

### 7. Claude Code SDK

Claude Code 不仅是 CLI，也是一个强大的 SDK。

#### 三种使用方式

**用途 1：大规模并行脚本**

```bash
#!/bin/bash
# 并行大规模重构

ls src/**/*.js | xargs -P 10 -I {} claude -p \
  "在 {} 中将所有 foo 引用改为 bar" \
  --max-turns 1 \
  --output-format json
```

**用途 2：内部聊天工具**

```python
# 为非技术用户构建简单的聊天界面
# 例如：安装失败时，自动调用 Claude Code SDK 修复

from claude_code import Agent

agent = Agent()
try:
    install_package()
except InstallError as e:
    # 让 Claude 修复问题
    agent.run(f"修复安装错误：{e}")
    install_package()  # 重试
```

**用途 3：快速 Agent 原型**

```python
# 快速原型新的 Agent 想法
from claude_code import Agent, Tool

# 例如：威胁调查 Agent
investigator = Agent(
    tools=[custom_cli_tool, mcp_server],
    system_prompt="你是一个安全威胁调查专家..."
)

investigator.run("调查这个可疑的登录事件")
```

**推荐：**
- 作为默认 Agent 框架（替代 LangChain/CrewAI）
- 用于编码和非编码任务
- 快速测试想法，再决定是否构建完整系统

---

## 最佳实践总结

### 核心原则

#### 1. CLAUDE.md 是关键

- ✅ 从防护栏开始，基于 Claude 的错误迭代
- ✅ 提供替代方案，不只说"不要"
- ✅ 引导阅读其他文档，不要 @-引用完整文件
- ✅ 保持简洁，作为简化工具的驱动力
- ❌ 不要试图写一本完整的手册

#### 2. 上下文管理至关重要

- ✅ 频繁使用 `/clear` 在任务之间清除上下文
- ✅ 使用 `/context` 监控 token 使用
- ✅ 对大型任务使用 "Document & Clear" 模式
- ❌ 避免依赖 `/compact`（不可靠）

#### 3. 具体化你的指令

- ✅ 提供详细的、多步骤的指令
- ✅ 使用 "think" / "think hard" / "think harder" 触发扩展思考
- ✅ 明确告诉 Claude 何时不要编码（探索阶段）
- ❌ 不要期望 Claude 读心术

#### 4. 让 Claude 看到结果

- ✅ 提供截图、设计稿、数据可视化
- ✅ 使用测试作为迭代目标
- ✅ 让 Claude 截图并比对设计
- ✅ 明确要求视觉上美观

#### 5. 积极合作，主动修正

- ✅ 在编码前要求制定计划
- ✅ 使用 Escape 中断并重定向
- ✅ 使用 Escape x2 编辑历史提示
- ✅ 让 Claude 撤销并重试
- ❌ 不要完全"放手"，保持参与

#### 6. 使用正确的架构

- ✅ 使用 Task(...) 让主 Agent 动态委托
- ✅ MCP 应该是简单的数据网关，不是 API 镜像
- ✅ 优先使用 CLI + Skills，而非复杂的 MCP
- ❌ 避免自定义子智能体（会隔离上下文）

#### 7. 测试和验证

- ✅ 使用 TDD 工作流（写测试 -> 实现）
- ✅ 使用 Hooks 在 commit 时验证（不是 write 时）
- ✅ 让独立的 Claude 审查代码
- ✅ 使用子智能体验证是否过拟合

#### 8. 扩展和自动化

- ✅ 使用 git worktrees 并行多任务
- ✅ 使用无头模式批量处理
- ✅ 使用 GHA 操作化 Claude Code
- ✅ 使用 SDK 快速原型新 Agent

### 安全建议

- ⚠️ 避免 `--dangerously-skip-permissions`（除非在容器中）
- ⚠️ 使用 `--disallowedTools` 禁止危险命令
- ⚠️ 定期审查工具权限白名单
- ⚠️ 保持 Claude Code 更新
- ⚠️ 在生产环境使用 Hooks 强制验证

### 工作流建议

- 📌 对任何"大型"功能使用规划模式
- 📌 创建 .claude/commands/ 中的简单快捷命令
- 📌 使用 JSON 输出进行自动化
- 📌 使用会话 ID 管理长时间任务
- 📌 定期 `/clear` 保持上下文专注

### 团队协作

- 👥 将 CLAUDE.md 提交到 git（团队共享）
- 👥 将 .claude/settings.json 检入源代码控制
- 👥 使用 .mcp.json 共享 MCP 配置
- 👥 记录自定义斜杠命令供团队使用
- 👥 使用 GHA 日志驱动改进循环

### 性能建议

- ⚡ 使用 `--max-turns` 限制上下文
- ⚡ 提高 MCP_TOOL_TIMEOUT 和 BASH_MAX_TIMEOUT_MS
- ⚡ 使用 Tab 补全快速引用文件
- ⚡ 管道输入大数据，不要粘贴
- ⚡ 使用流式 JSON 处理大型输出

---

## 命令速查表

### CLI 命令

| 命令 | 描述 | 示例 |
|-----|-----|-----|
| `claude` | 启动交互式 REPL | `claude` |
| `claude "query"` | 带提示启动 | `claude "解释这个项目"` |
| `claude -p "query"` | 打印模式，执行后退出 | `claude -p "解释函数"` |
| `claude -c` | 继续最近的对话 | `claude -c` |
| `claude -r "id" "query"` | 通过 ID 恢复会话 | `claude -r "abc123" "完成 PR"` |
| `claude update` | 更新到最新版本 | `claude update` |
| `claude --mcp` | 配置 MCP 服务器 | `claude --mcp` |

### CLI 标志

| 标志 | 描述 | 示例 |
|-----|-----|-----|
| `--model` | 指定模型 | `--model sonnet` |
| `--add-dir` | 添加工作目录 | `--add-dir ../apps ../lib` |
| `--allowedTools` | 允许工具无需提示 | `--allowedTools "Bash(git:*)"` |
| `--disallowedTools` | 禁止特定工具 | `--disallowedTools "Bash(rm:*)"` |
| `--output-format` | 设置输出格式 | `--output-format json` |
| `--input-format` | 设置输入格式 | `--input-format stream-json` |
| `--max-turns` | 限制对话轮次 | `--max-turns 3` |
| `--verbose` | 启用详细日志 | `--verbose` |
| `--continue` | 继续会话 | `--continue` |
| `--resume` | 恢复会话 | `--resume abc123` |
| `--dangerously-skip-permissions` | 跳过所有权限提示（危险！） | `--dangerously-skip-permissions` |
| `--mcp-debug` | 调试 MCP 配置 | `--mcp-debug` |

### 斜杠命令

| 命令 | 描述 |
|-----|-----|
| `/help` | 显示帮助和可用命令 |
| `/exit` | 退出 REPL |
| `/clear` | 清除对话历史 |
| `/config` | 打开配置面板 |
| `/doctor` | 检查安装健康状况 |
| `/cos` | 显示成本和时长 |
| `/context` | 查看上下文使用情况 |
| `/ide` | 管理 IDE 集成 |
| `/compact [说明]` | 压缩对话（谨慎使用） |
| `/mcp` | 访问 MCP 功能 |
| `/permissions` | 管理工具权限 |
| `/init` | 生成 CLAUDE.md 文件 |

### 键盘快捷键

| 快捷键 | 操作 |
|-------|-----|
| `Ctrl+C` | 取消当前操作 |
| `Ctrl+D` | 退出 Claude Code |
| `Tab` | 自动补全（文件、命令） |
| `Up/Down` | 浏览命令历史 |
| `Escape` | 中断 Claude（保留上下文） |
| `Escape x2` | 跳回历史，编辑提示 |
| `Shift+Tab` | 切换自动接受模式 |

### 常用工具白名单

```bash
# 安全文件操作
--allowedTools "Edit" "Write" "Read"

# Git 操作
--allowedTools "Bash(git:*)" "Bash(git commit:*)"

# 禁止危险操作
--disallowedTools "Bash(rm:*)" "Bash(sudo:*)" "Bash(chmod:*)"
```

### 常用工作流命令

```bash
# 探索 -> 规划 -> 编码 -> 提交
> 阅读相关文件，暂时不要编码
> think hard: 制定实现计划
> 根据计划实现代码
> 提交并创建 PR

# TDD 工作流
> 编写测试（不要创建模拟实现）
> 运行测试确认失败
> 提交测试
> 编写通过测试的代码
> 提交代码

# 视觉迭代
> 这是设计稿（粘贴截图）
> 实现设计，截图结果，迭代直到匹配
> 提交代码

# 代码库问答
> 日志系统是如何工作的？
> 查看 git 历史，解释为什么这样设计

# Git 工作流
> 写一条提交信息
> 解决这个 rebase 冲突
> v1.2.3 版本包含了哪些更改？

# GitHub 工作流
> pr
> 修复我 PR 上的评论并推送
> 查看 CI 失败日志并修复
```

### 文件和目录结构

```
project/
├── CLAUDE.md              # 项目级配置（提交到 git）
├── CLAUDE.local.md        # 个人配置（不提交）
├── .mcp.json              # MCP 配置（提交到 git）
├── .claude/
│   ├── settings.json      # 项目级设置（推荐提交）
│   ├── commands/          # 自定义斜杠命令
│   │   ├── catchup.md
│   │   ├── pr.md
│   │   └── fix-github-issue.md
│   └── hooks/             # Hooks（企业级）
│       └── pre-commit.js
├── ~/.claude/
│   ├── CLAUDE.md          # 全局配置
│   ├── settings.json      # 全局设置
│   └── commands/          # 全局命令
└── ~/.claude.json         # 全局配置（替代方案）
```

---

## 资源链接

### 官方文档

- [Claude Code 官方文档](https://docs.anthropic.com/en/docs/claude-code)
- [Claude Code GitHub 仓库](https://github.com/anthropics/claude-code)
- [Anthropic API 文档](https://docs.anthropic.com/)
- [MCP 文档](https://modelcontextprotocol.io/docs/getting-started/intro)

### 实用工具

- [Prompt Improver](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/prompt-improver) - 优化提示
- [GitHub CLI](https://cli.github.com) - gh 命令行工具
- [Puppeteer MCP Server](https://github.com/modelcontextprotocol/servers/tree/main/src/puppeteer) - 浏览器自动化
- [Docker Dev Container](https://github.com/anthropics/claude-code/tree/main/.devcontainer) - 安全的容器环境

### 进阶阅读

- [AI Can't Read Your Docs](https://blog.sshh.io/p/ai-cant-read-your-docs) - 编写 AI 友好的文档
- [Building Multi-Agent Systems](https://blog.sshh.io/p/building-multi-agent-systems-part) - Master-Clone 架构
- [Everything Wrong with MCP](https://blog.sshh.io/p/everything-wrong-with-mcp) - MCP 的正确使用方式

---

## 结语

Claude Code 代表了 AI 辅助编程向"代理式编程"（Agentic Coding）范式的重要转变。与传统的代码补全工具不同，Claude Code 通过提供接近原始模型访问能力、灵活的工具集成和多实例协作模式，为工程师提供了一个真正的 AI 编程伙伴。

**关键思维转变：**

- 从"人类编写代码、AI 辅助" → "人类定义意图、AI 实现细节"
- 从"学习固定命令" → "自然语言对话式协作"
- 从"单一工具" → "可编程的 Agent 框架"

**成功的关键：**

1. **投资 CLAUDE.md**：这是你最重要的配置文件
2. **迭代和学习**：基于实际错误优化工作流
3. **积极合作**：不要"放手"，要主动引导
4. **正确架构**：Task(...) > 自定义子智能体，CLI + Skills > 复杂 MCP

随着越来越多的工程师采用这种工作模式，我们正在见证软件开发范式的根本性转变。这份指南不仅是工具使用手册，更是这个新时代的工程方法论。

**现在就开始：**

```bash
# 安装 Claude Code
curl -sL https://install.anthropic.com | sh

# 在你的项目中初始化
cd your-project
claude /init

# 开始对话式编程
claude "帮我了解这个项目的架构"
```

祝你在 AI 辅助编程的旅程中取得成功！

---

**最后更新时间**：2025年11月

**版本**：v1.0

**维护者**：基于 Claude Code Cheat Sheet 和 Anthropic 官方最佳实践整理

---

## 附录：常见问题

### Q: CLAUDE.md 应该写多长？

**A:** 
- 个人项目：随意，通常 5-10KB
- 团队项目：建议 13-25KB
- 企业 monorepo：可达 25KB+，但要精心管理

原则：保持简洁，只记录 30%+ 工程师使用的工具和规范。

### Q: 什么时候使用 /compact vs /clear？

**A:**
- **优先使用 /clear**：简单任务、任务之间切换
- **避免 /compact**：不可靠，容易丢失上下文
- **Document & Clear**：大型任务，需要持久化进度

### Q: 自定义子智能体 vs Task(...) 哪个好？

**A:**
- **Task(...) 更好**：让主 Agent 动态决定委托，保留完整上下文
- **自定义子智能体问题**：隔离上下文，强制工作流，降低灵活性

除非有非常特定的需求，否则使用 Task(...)。

### Q: MCP vs CLI，如何选择？

**A:**
- **使用 CLI**：无状态操作、简单工具、数据查询
- **使用 MCP**：有状态环境（Playwright）、需要安全边界、跨网络访问

如果可以写成简单的 CLI，就不要用 MCP。

### Q: 如何处理大型代码库的性能问题？

**A:**
1. 频繁 `/clear` 保持上下文专注
2. 使用 `--max-turns` 限制轮次
3. 精简 CLAUDE.md，只放核心信息
4. 使用 `--add-dir` 限制工作目录
5. 定期查看 `/context` 监控使用情况

### Q: --dangerously-skip-permissions 安全吗？

**A:**
- **不安全**：可能导致数据丢失、系统损坏、数据泄露
- **安全使用方式**：仅在无互联网访问的 Docker 容器中使用
- **推荐场景**：修复 lint、生成样板代码、快速原型

生产环境避免使用。

### Q: 如何让团队采用 Claude Code？

**A:**
1. 创建团队共享的 CLAUDE.md
2. 将 .claude/settings.json 提交到 git
3. 创建常用的自定义斜杠命令
4. 分享成功案例和最佳实践
5. 使用 GHA 展示自动化能力

从入职流程开始，让新人用 Claude Code 学习代码库。

### Q: 如何调试 Claude 的错误行为？

**A:**
1. 使用 `--verbose` 查看详细日志
2. 使用 `HTTPS_PROXY` 检查原始提示
3. 检查 CLAUDE.md 是否有冲突指令
4. 查看会话历史：`~/.claude/projects/`
5. 迭代优化提示和 CLAUDE.md

### Q: 可以在 CI/CD 中使用 Claude Code 吗？

**A:**
可以！推荐方式：
1. **GitHub Actions**：使用官方 claude-code-action
2. **容器化**：在 Docker 中运行，控制环境
3. **无头模式**：`claude -p` + `--output-format json`
4. **审计**：保存日志，定期审查改进

这是操作化 Claude Code 的最强大方式。

