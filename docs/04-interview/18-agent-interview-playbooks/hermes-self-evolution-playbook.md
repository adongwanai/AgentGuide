---
type: 求职指南
status: 已发布
level: 高阶
topic:
  - 面试求职
  - Agent
---

# Hermes Agent 自进化 彻底备战手册
> **本手册定位**：面向 AI Agent 求职与面试准备的专题复习资料，用于补充 AgentGuide 面试题库与项目讲述材料。
> 内容以公开资料、项目复盘和工程实践方法论为基础做二次整理，建议结合自己的真实项目经历进行取舍与改写。

---
## 目标：把 Hermes 的自我进化机制讲得像真正读过源码、理解其设计哲学的人

> 使用建议：Hermes Agent 是 2026 年 Agent 自进化方向的标杆。面试官如果问"Agent 怎么自我进化"，你引用的不能只是 LangChain 的 Memory 或者向量数据库——那些是记忆，不是进化。Hermes 示范了真正的闭环：经验→Skill→使用→改进→再使用。这份手册覆盖了从 Skills 闭环到 KEPA 到 RL 训练到 Prompt/Context/Harness 三维分析的全部内容。
>
> 综合吸收四份材料：
> 1. `Herness.md`：仓库级极致深度解读，10 个可迁移核心模式，Nous 战略判断
> 2. `一文搞懂Hermes`：Skills 闭环 7 阶段源码级拆解（腾讯云开发者）
> 3. `拆完Hermes源码`：KEPA + 4D 记忆 + OpenClaw 对比（腾讯云开发者）
> 4. `深度解析_Hermes_Agent`：Prompt/Context/Harness + RL 训练闭环（阿里云开发者）

---

# 一、30 秒 + 2 分钟版本

## 30 秒版本
Hermes Agent 是 Nous Research 在 2026 年 2 月开源的自主进化型 Agent 框架，71.8K GitHub Stars，核心理念是"The agent that grows with you." 和 OpenClaw 的全能管家定位完全不同，Hermes 的差异化在于内置学习闭环：每次完成任务后，Agent 自动复盘、提取经验、创建或更新 Skill，下次遇到类似任务直接复用——越用越强，不需要用户手动教。

## 2 分钟版本
Hermes 的自进化由"内外"双路径驱动。外路径是**动态 Skill 生成**：复杂任务完成后，后台审查 Agent 自动判断"是否值得沉淀为 Skill"，创建结构化 skill 文件，后续通过渐进式披露按需加载。内路径是**RL 训练闭环**：批量采集 Agent 轨迹 → 压缩 → GRPO 训练 → 评估 → 部署回 Agent。两者结合：Skill 解决"快速复用"问题，RL 解决"能力内化"问题。

核心技术栈：KEPA（提示反向传播）+ 四层记忆架构（Prompt Memory / Session Search / Skills / Honcho）+ 渐进式披露 + Periodic Nudge + 14 类错误自愈 + 多平台 Gateway。和 OpenClaw 最本质的区别：OpenClaw 每次任务从零开始，Hermes 每次任务带着历史积累的经验上场。

---

# 二、核心定位：为什么 Hermes 不是"又一个 Agent 框架"

## 2.1 它要解决的根本问题

当前 Agent 生态的系统性盲区——**记忆与学习是割裂的**：

| 框架 | 致命缺陷 |
|------|---------|
| LangChain / LlamaIndex | 无状态链式调用，会话结束即遗忘 |
| AutoGen | 多 Agent 编排强但单个 agent 不会积累程序性知识 |
| CrewAI | role 是静态 prompt，无演进 |
| OpenClaw | 跨会话记忆不错，但只记录"发生过什么"，不沉淀"什么方法有效" |

一句话总结：**OpenClaw 有声明性记忆（what happened），Hermes 有程序性记忆（how to do it）。**

## 2.2 Nous Research 的真实战略（面试时最有深度的一句）

> "Hermes Agent 对 Nous 的战略价值不是'agent 产品'，而是**下一代 tool-calling 模型的训练数据生产机。**"

证据：`batch_runner.py`、`trajectory_compressor.py`、`environments/`（Atropos RL 环境）这三个在普通 agent 框架里根本不会出现的模块，在 Hermes 里是一等公民。`save_trajectories` 是 AIAgent 的构造参数。

Nous 的闭环布局：Hermes Agent（产品+数据前端）→ trajectory 数据 → Atropos RL 环境 → 训练 Hermes N+1 模型 → 部署回 Agent。**Agent 不应被当作"推理时应用"看待，而是训练-推理-数据-再训练的闭环节点。**

---

# 三、Skills 闭环：7 阶段完整拆解

这是 Hermes 最核心的机制，也是面试时最应该详细讲的。

## 3.1 全局视角

```
用户会话 → 任务执行 → 后台审查 Agent → 判断是否创建/更新 Skill
                                            ↓
下次类似任务 ← 渐进式加载 Skill ← 索引注入 System Prompt
```

## 3.2 阶段一：Skill 创建触发——谁决定"什么时候该创建"？

**Agent 自己决定。** System Prompt 中写入创建触发条件：

- **5+ tool calls**：简单任务不值得建，只有复杂流程才需要
- **fixing a tricky error**：踩过的坑是最有价值的知识
- **don't wait to be asked**：Agent 应自主判断，不等用户要求
- **Skills that aren't maintained become liabilities**：过时的 Skill 比没有更危险

四个触发器（4 选 1）：①同一任务 ≥5 次 tool call ②从一次 error 中恢复 ③用户纠正 agent ④出现 non-obvious 但有效的工作流

## 3.3 阶段二：七道安全关卡——创建流程的防护链

调用 `skill_manage(action="create", name="...", content="...")` 后经过：

**1. 原子写入**：不是 `file.write()`，而是先写临时文件 → `os.replace()` 原子替换。进程崩溃时，目标文件要么是旧内容要么是新内容，绝不会出现写了一半的损坏文件。

**2. 写入后扫描（非扫描后写入）**：避免 TOCTOU 竞态条件。先写入再扫描文件系统上的实际内容，确保扫描的是最终状态。

**3. Prompt Injection 检测**：`INJECTION_PATTERNS` 正则库检测 "ignore previous instructions" 等恶意内容。因为 Skill 内容最终注入 Agent 消息流，恶意 Skill 就是 Prompt Injection 攻击。

**4. 路径穿越防护**：`validate_within_dir()` + `has_traversal_component()` 拦截 `../../.env` 等路径逃逸。

**5. 环境变量依赖检查**：Skill 需要的 API key 未配置时，不静默失败，而是交互式提示或返回 "setup_needed": true。

**6. 安全扫描（skills_guard.py）**：90+ 种威胁正则模式，覆盖命令注入、环境变量泄露、Docker 逃逸、XSS、SQL 注入等 10 个类别。

**7. 信任分级策略**：

| 来源 | 策略 |
|------|------|
| 内置 Skill | 完全信任（经过 code review） |
| 受信任来源（OpenAI/Anthropic 官方） | 允许 caution，阻止 dangerous |
| 社区 Skill | 最严格，任何高于 safe 的发现都阻止 |
| Agent 自创建 | 允许 caution，dangerous 时询问用户 |

## 3.4 阶段三：索引构建——两层缓存的极致优化

为什么不能每次都扫描文件系统？用户可能有几十上百个 Skill，每次会话都递归扫描 + 解析 YAML frontmatter 不可接受。

**Layer 1：进程内 LRU 缓存**（`SKILLS_PROMPT_CACHE_MAX = 8`）：线程安全 OrderedDict，缓存键是五元组 `(skills_dir, available_tools, available_toolsets, platform_hint, disabled_toolsets)`。同一 Gateway 进程服务多平台，所以 platform 也是键。

**Layer 2：磁盘快照**：不对比文件内容（太慢），对比每个 SKILL.md 的 mtime 和文件大小。任何一个文件变化，manifest 不匹配，触发全量扫描。

性能对比：Layer 1 命中 ~0.001ms → Layer 2 命中 ~1ms → 全扫描 50-500ms

## 3.5 阶段四：条件激活——Skill 的智能可见性

不是所有 Skill 在所有情况下都应该出现在索引中。

- `fallback_for_toolsets: [web]`：只在 web 工具不可用时出现（有了 Firecrawl 就不需要手动 curl 搜索）
- `requires_toolsets: [terminal]`：没有 terminal 工具时自动隐藏
- `platforms: [macos]`：在 Linux 服务器上不出现

这套机制解决了**索引膨胀**问题，让 System Prompt 保持精简。

## 3.6 阶段五：渐进式披露——三级加载

这是受 Anthropic Claude Skills 启发的设计模式，核心思想：**不要一次性把所有信息倒给 Agent。**

- **Tier 1**：System Prompt 只放索引（每个 Skill 一行：名称 + 描述，约 20 tokens）
- **Tier 2**：Agent 判断需要时，调用 `skill_view(name)` 加载完整内容
- **Tier 3**：如果 Skill 有支撑文件（API 文档、模板），再按需加载

100 个 Skill 的用户，System Prompt 只增加约 2,000 tokens（100×20），而不是 500K tokens。

## 3.7 阶段六：注入策略——User Message 而非 System Prompt

**这是整个 Skills 系统中最关键的架构决策。**

Skill 内容不作为 System Prompt 追加，而是作为 **User Message** 注入到对话历史。

**为什么？四个字：Prompt Cache。**

Anthropic 的 Prompt Caching 要求 System Prompt 在整个对话中不能变化。如果每次加载 Skill 就修改 System Prompt，缓存失效，每轮对话都要重新处理整个 prompt——30 轮工具调用的复杂任务可能数十倍成本增加。

Hermes 在 AGENTS.md 中明确警告："Prompt Caching Must Not Break. Do NOT implement changes that would alter past context mid-conversation, change toolsets mid-conversation, reload memories or rebuild system prompts mid-conversation."

为弥补 User Message 指令跟随权重低于 System Prompt 的问题，Hermes 在注入消息前加 `[SYSTEM: ...]` 前缀标记，System Prompt 中 "you MUST load it" 的强制措辞也在间接提升遵循概率。

**这是一个深思熟虑的成本-效果权衡**：牺牲一点点指令跟随可靠性，换取数十倍的 API 成本节约。

## 3.8 阶段七：自改进——闭环的关键闭合点

**Patch 而非 Edit**：更新 Skill 时默认用 `skill_manage.patch(old_string, new_string)`，只改变化的部分，不重写整个文件。

**Fuzzy Match 引擎**：LLM 回忆 Skill 内容时经常有微小格式差异（多一个空格、少一个换行），严格字符串匹配会导致大量合理 patch 失败。Fuzzy Match 处理了：空白规范化、缩进差异、转义序列、块锚匹配。

**最终一致性模型**：
1. 当前对话：使用旧版 Skill，发现问题并 patch
2. Patch 成功后清除 LRU 缓存 + 磁盘快照
3. 下一个对话：索引缓存失效，重新扫描，加载更新后的 Skill

**当前对话不生效，下次会话才生效**——保护 Prompt Cache，同时保证最终一致性。

---

# 四、KEPA：对"提示"做反向传播

这是 Hermes 最精妙的设计，社区称为 **KEPA（Knowledge-Enhanced Prompt Adaptation）**。

## 4.1 核心类比

```
传统深度学习：
  前向传播：输入 → 模型权重 → 输出
  反向传播：损失 → 梯度 → 更新权重

Hermes 的做法：
  前向传播：用户意图 → Hermes(LLM + 工具 + Prompt) → 执行结果
  反向传播：执行质量 → 审查 Agent → 更新 Prompt/Skill/Memory
```

**关键差异：不是更新"模型权重"，而是更新"如何使用模型"的策略**——包括提示模板、工具调用顺序、技能定义等。

## 4.2 后台审查 Agent 的实现

整个"自我进化"的核心 = 一段 Prompt + 一个后台线程 + 文件持久化。

**没有强化学习，没有模型微调，没有向量数据库。** 纯粹的 Prompt Engineering + 文件系统。

`_spawn_background_review` 在后台异步 Fork 一个轻量级 Agent，从三个维度审查：

1. **记忆审查**（`_MEMORY_REVIEW_PROMPT`）：这段对话有什么值得记住的经验？
2. **技能审查**（`_SKILL_REVIEW_PROMPT`）：这个任务模式是否值得变成 Skill？
3. **综合审查**（`_COMBINED_REVIEW_PROMPT`）：有什么可以改进的？

## 4.3 Periodic Nudge：行为经济学在 Agent 设计中的应用

`_iters_since_skill` 计数器记录距离上次使用 skill_manage 过了多少轮。`_skill_nudge_interval = 10`——Agent 连续工作 10 轮都没有创建/修改技能时，系统注入提醒："你是不是该把刚才学到的经验整理成技能了？"

**Agent 不是无限自律的。** 周期性 nudge = 给 Agent 设置"反省时刻"，避免沉浸任务忘记 meta-level 知识抽取。

## 4.4 KEPA 的四重优势

1. **零成本进化**：审查 Agent Token 消耗极小（最多 8 次 tool call），Skill 文件就是 Markdown
2. **可解释、可编辑**：Skill 是自然语言写的，你可以随时打开看 Agent 学到了什么，不对直接改
3. **跨模型迁移**：今天用 Claude 积累的 Skill，明天换 DeepSeek 一样能用
4. **渐进式积累**：三个月后的 Hermes 和刚装好的 Hermes，体验完全不同

## 4.5 KEPA 的四重局限

1. **"自动"≠"准确"**：审查 Agent 判断力上限=底层 LLM 能力上限，可能保存噪声、遗漏重要经验、Skill 质量参差不齐
2. **只有复杂任务才触发**：简单一步到位的任务大概率 "Nothing to save"
3. **更新有延迟**：冻结快照机制下，后台更新的 Skill 到下一次会话才生效
4. **生态差距**：OpenClaw 有 ClawHub 数千插件，Hermes 第三方生态在早期

---

# 五、四层记忆架构

这是 Hermes 最精华的存储架构。**"何时加载"的划分比"存什么"更重要。**

| 层 | 存储位置 | 加载时机 | 尺寸约束 | 对应认知类型 |
|----|---------|---------|---------|------------|
| **Prompt Memory** | MEMORY.md / USER.md | 永远加载，进 system prompt | 两文件合计 ≤ 3,575 字符（硬性） | 语义/陈述性记忆 |
| **Session Search** | SQLite + FTS5 | 按需查询，agent 主动调用 session_search | 无限 | 情景记忆 |
| **Skills** | ~/.hermes/skills/*/SKILL.md | 索引常驻 + 全文按需（渐进式披露） | 每 skill 独立文件 | 程序性记忆 |
| **Honcho**（可选） | 独立服务 | 按需查询，被动建模 | 12 identity layers | 关系性记忆/自我模型 |

## 关键设计决策

**为什么是 3,575 这个奇怪的数字？** 刻意的经济约束——逼迫 agent 做 curation 而非 accumulation。给 50KB，agent 会把什么都塞进来；3,575 字符迫使只记"真正必须每次都看到的"。

**为什么 Memory 写入当前会话不生效？** 避免 agent 在一次会话中反复覆盖自己刚写下的内容导致不稳定，也让"写记忆"成为接近人类"睡眠巩固"的慢过程。

**为什么用 FTS5 而不用向量数据库？** 对结构化文本（对话、tool call），BM25/FTS5 在精确性和可解释性上优于嵌入相似度；检索后用 LLM 做摘要再注入，把"搜索质量"和"注入 token 效率"解耦。

---

# 六、RL 训练闭环：权重内化的终极进化

Skill 生成是"记笔记"，RL 训练是"练内功"——改变模型权重，实现真正的能力进化。

## 6.1 完整流程

```
任务定义 → 批量数据合成 (batch_runner.py) → 轨迹压缩 (trajectory_compressor.py)
→ GRPO 训练 (rl_cli.py) → 自动评估 → 部署回 Agent
```

## 6.2 批量数据合成（batch_runner.py）

- 用户准备 JSONL 提示词或从 Benchmark 数据集（GSM8K、HumanEval）采集
- 线程池并行处理，每条提示词创建独立 Agent 实例
- 默认用 `anthropic/claude-opus-4.6` 作为 Teacher 模型
- **工具集随机采样**：不是每次都同样工具配置，而是随机采样不同组合——训练数据覆盖各种工具搭配场景，模型学会灵活运用
- **零推理过滤**：统计 `<REASONING_SCRATCHPAD>` 和 reasoning 字段，两者都为零的样本丢弃——Agent 完全没有显式推理的样本对训练无价值
- 输出 ShareGPT 格式：生态兼容（LLaMA-Factory、FastChat 都认）

## 6.3 轨迹压缩（trajectory_compressor.py）

原始轨迹可能有几十万 Token，直接用于 RL 训练不现实。压缩策略：

**头部保护区**：第一条 system prompt + 第一条人类消息 + 第一条 GPT 回复 + 第一次工具交互——任务的"锚点"，绝不被压缩

**尾部保护区**：最后 4 轮对话——承载最终结论和验证信息，绝不被压缩

**中间压缩区**：大量中间步骤和工具调用——用轻量模型（如 Gemini Flash）生成 `[CONTEXT SUMMARY]` 摘要

目标上限：15,250 tokens（精确到 HuggingFace Tokenizer 计数）

## 6.4 GRPO 算法与奖励函数

使用 DeepSeek R1 提出的 **GRPO（Group Relative Policy Optimization）**：同一问题采样 8-16 个回答，用奖励函数打分，学习"多产出高分回答，少产出低分回答"。

**关键优势：不需要单独训练 Reward Model，直接用规则化奖励函数。**

多维度组合奖励设计：

| 维度 | 权重 | 衡量什么 |
|------|------|---------|
| 正确性 | 2.0（最高） | 最终答案是否正确 |
| 格式规范 | 0.5 | 是否遵循 `<reasoning>...<answer>` 结构 |
| 渐进格式 | 0~0.5 | 部分符合格式也给分 |

奖励函数设计黄金法则：①组合 3-5 个奖励函数，各管一个方面 ②正确性权重最高 ③给部分分 ④先单独测试再合并

通过 ToolContext 机制，奖励函数可以：执行终端命令编译验证代码、读取文件检查是否真的修改、访问网络验证搜索结果——做"真实验证"而非文本匹配。

## 6.5 为什么不直接从用户数据学习？

两个原因：
1. **隐私问题**：用户对话包含敏感信息
2. **质量问题（更关键的）**：用户对话质量参差不齐，直接拿来训练大概率把模型"训废"

RL 训练的真正目的不是"从用户那学东西"，而是做**知识蒸馏**——把 Claude Opus 的 Agent 能力压缩到 Qwen 3-4B 小模型里，实现降本、提速、数据不出机器。

---

# 七、Prompt / Context / Harness 三维分析

## 7.1 Prompt Engineering：模型异构与生态兼容

**工具使用强制指导**：不同模型对工具调用的主动性不同，Hermes 根据具体模型动态注入针对性指令补丁：
- Claude：训练时就强调工具使用，一般不需要额外提醒
- GPT/Codex：容易"只说不做"，需明确指令"你必须用工具执行"
- Gemini/Gemma：需提醒绝对路径、先读后改、并行调用工具

**生态兼容性（降低迁移成本的殺手锏）**：
- 直接读取 OpenClaw 的 AGENT.md、SOUL.md、USER.md——零成本迁移
- 支持 Claude Code 的 CLAUDE.md、Cursor 的 .cursorrules
- 适配多平台 IM 协议（WhatsApp、Slack 等各自的行为提示词）

## 7.2 Context Engineering：相对阈值 + 上下文注入

**压缩：相对阈值触发 vs 绝对阈值**

OpenClaw 用绝对阈值（固定 token 数触发压缩），Hermes 用**比例阈值**——上下文达到模型总窗口的 50% 时触发。无论底层是 200K 窗口还是 32K 窗口，都能灵活适应。

压缩策略同 OpenClaw：头部保护 + 尾部保护 + 中间摘要。

**上下文注入：从"工具调用"到"即时挂载"**

通过 `@` 符号快速资源挂载，省去 Agent 思考"是否需要调用工具"的中间环节：

| 语法 | 作用 |
|------|------|
| `@file:main.py` | 注入文件完整内容 |
| `@file:src/utils.py:10-20` | 注入指定行 |
| `@folder:src/` | 列出目录树 |
| `@diff` | 注入 git 未暂存更改 |
| `@staged` | 注入 git 已暂存更改 |
| `@git:3` | 注入最近 3 次提交（含完整补丁） |
| `@url:https://...` | 抓取网页转 Markdown |

## 7.3 Harness Engineering：约束与运行保障

**全生命周期 Hook 机制**：`on_agent_start` / `on_tool_call` / `on_tool_result` / `on_agent_end` / `on_turn_start` / `on_pre_compress` / `on_memory_write` / `on_delegation` / `on_session_end`

**14 类错误分类与自愈**：不再笼统处理"Error"，而是精细分类：

| 错误类型 | 含义 | 自动恢复策略 |
|---------|------|------------|
| auth | 认证失败 | 提示用户重新配置 |
| rate_limit | 被限流 | 指数退避重试 |
| timeout | 请求超时 | 重试 + 降级 |
| context_overflow | 上下文溢出 | 触发压缩 |
| server_error | 5xx | 切换 fallback provider |
| billing | 额度用完 | 提示用户充值 |
| ... | ... | ... |

**受控子 Agent 机制**：子 Agent 不能创建新的子 Agent（防止递归爆炸）、不能向主 Agent 反向询问（单向性）、不能访问主 Agent 完整上下文（安全隔离）。子 Agent 禁用的工具列表包括 skill_manage、memory 写入等权限操作。

**多层级安全护栏**：防 Prompt 注入 + Skill 安全扫描 + 符号链接逃逸检测 + 结构化检查（MAX_FILE_COUNT=50、MAX_FILE_SIZE=1MB、禁止 .exe 文件）

---

# 八、Hermes vs OpenClaw：面试必问的对比

| 维度 | OpenClaw | Hermes Agent |
|------|---------|-------------|
| **核心哲学** | 全能助手，插件生态 | 自我进化，越用越强 |
| **记忆能力** | 无状态（需手动配置） | 四维持久记忆（自动） |
| **技能管理** | 用户手动安装/编写 Skill | Agent 自动从经验中创建 Skill |
| **学习方式** | 不学习 | 内置闭环学习系统 |
| **插件生态** | 数千个（ClawHub） | 较少，但快速增长 |
| **部署门槛** | 中等 | 极低（$5 VPS） |
| **适合场景** | 一次性任务、多平台集成 | 长期使用、个性化需求 |

**最本质的差异**：OpenClaw 是"什么都能干但不长记性的全能管家"，Hermes 是"能力在成长的专属员工"。

**用户驱动 vs Agent 自驱动**：最理想的方案是两者结合——核心的确定性的知识（项目规范、个人偏好）用户显式定义（像 OpenClaw SOUL.md），隐性的经验性的知识（"这个 API 容易超时，最好加重试"）Agent 自动积累（像 Hermes KEPA）。Hermes 也支持用户手动编辑 Skill 和 Memory，自动进化是兜底机制。

---

# 九、10 个可迁移核心模式（面试时展示架构思维）

## 模式 1：程序性记忆 = 独立可写的 Skill 文件
把"agent 学会的工作流"物化为 Markdown 文件，与 tool（原语）和 memory（事实）分离。每次任务后 agent 自评是否值得沉淀。

## 模式 2：Periodic Nudges
在会话中按固定 tool call 间隔注入系统级消息，要求 agent 回顾最近发生的事并评估是否值得持久化。Nudge 不来自用户，不占用户 turn。

## 模式 3：Progressive Disclosure
System Prompt 只放索引（name + 一行摘要），全文加载由 agent 主动触发。Token 成本与 Skill 数量解耦。

## 模式 4：统一 Session-ID 网关架构
Session 是主键，平台是属性。一个 session_id 绑定一段 conversation state，来自/回到哪个平台由 message envelope 决定。

## 模式 5：四层记忆架构
按"何时加载 + 是否每次加载"划分至少 3 层（prompt memory / episodic search / procedural skills），用存储位置编码使用频率。

## 模式 6：带 Lineage 的上下文压缩
压缩中间 turn 时保存被压缩 turn 的 ID 作为 lineage 元数据，agent 后续需要原始内容可顺 lineage 回溯。

## 模式 7：Prompt-Cache-Aware 设计
System Prompt 前缀必须逐字节稳定。Memory/Skill 变更通过 User Message 注入，不在会话中修改 System Prompt。

## 模式 8：Subagent Delegation（RPC 式子代理）
开隔离 context 的子进程 agent，只返回最终报告给主 agent——中间 100 次 tool call 不污染主 context。

## 模式 9：开放技能标准 + 社区 Hub
Skill 格式是 agentskills.io 开放标准（YAML frontmatter + Markdown），跨框架可移植，避免 framework lock-in。

## 模式 10：Trajectory Pipeline（Agent 即训练数据源）
save_trajectories=True → 每次会话写 JSONL → 压缩 → 包装成 RL env → 训练下一代模型 → 部署回 Agent。闭环从未断开。

---

# 十、高频面试题

## Q1. Hermes Agent 的"自进化"到底是怎么实现的？

### 主答模板
两句话：**Skill 自动生成解决"快速复用"，RL 训练闭环解决"能力内化"。**

Skill 生成路径：复杂任务完成后，后台审查 Agent 自动复盘整个执行轨迹，从三个维度审查（记忆/技能/综合），判断是否值得沉淀为 Skill。如果是，调用 skill_manage.create 创建结构化的 skill 文件（YAML frontmatter + Markdown），后续通过渐进式披露按需加载。

RL 训练路径：batch_runner 批量采集 Agent 轨迹（Teacher 模型 + 工具集随机采样）→ trajectory_compressor 压缩（头尾保护+中间摘要）→ GRPO 训练（规则化奖励函数，不需要 Reward Model）→ 自动评估 → 部署。

整个机制的核心引擎叫 KEPA——不像传统深度学习更新模型权重，而是更新"如何使用模型"的策略（Prompt、Skill、Memory）。零训练成本，可解释可编辑，跨模型可迁移。

---

## Q2. 为什么 Skill 内容以 User Message 注入而不是 System Prompt？

### 主答模板
四个字：**Prompt Cache。** Anthropic 的 Prompt Caching 要求 System Prompt 在整个对话中不能发生变化——前缀必须逐字节稳定才能命中缓存。如果每次加载 Skill 就修改 System Prompt，缓存失效，每轮对话都要重新处理整个 prompt，30 轮工具调用的复杂任务可能数十倍成本增加。

以 User Message 注入的代价是指令跟随权重略低于 System Prompt。Hermes 通过两个手段弥补：注入消息前加 `[SYSTEM: ...]` 前缀标记，System Prompt 中 "you MUST load it" 的强制措辞。

这是一个深思熟虑的成本-效果权衡——牺牲一点指令跟随可靠性，换取 90%+ 的 API 成本节约。

---

## Q3. Hermes 和 OpenClaw 的本质区别是什么？

### 主答模板
OpenClaw 解决的是"怎么让 LLM 持久存在"——多平台接入、身份连续性、日程调度。Hermes 解决的是"怎么让 LLM 越用越强"——经验提取、知识沉淀、能力进化。

最本质的差异在记忆类型：OpenClaw 有声明性记忆（MEMORY.md 记录"发生过什么"），Hermes 有程序性记忆（Skills 记录"什么方法有效"）。OpenClaw 每次任务从零开始探索，Hermes 每次任务带着积累的经验上场。

另一个关键差异：学习驱动模式。OpenClaw 是用户驱动——你想让 Agent 学什么就手动写 SOUL.md/AGENTS.md。Hermes 是 Agent 自驱动——KEPA 机制让 Agent 自己决定学什么、怎么学，用户只管用。

我的判断是两者不互斥——最理想的方案是两者结合：核心的确定性知识用户显式定义，隐性的经验性知识 Agent 自动积累。

---

## Q4. Hermes 的 Periodic Nudge 是什么？为什么重要？

### 主答模板
Periodic Nudge 是 Hermes 在 Agent 连续工作 N 轮（默认 10 轮）没有创建或更新 Skill 时，系统自动注入的一条提醒消息，让 Agent 回顾最近的工作并评估是否值得沉淀。

它解决的是一类隐蔽的工程问题：**Agent 不是无限自律的。** Agent 会沉浸在任务执行中，忘记做 meta-level 的知识抽取。人类需要"定期复盘"的习惯，Agent 同样需要——但 Agent 不会自发养成这个习惯。

Nudge 不来自用户（不占用户 turn），是作为 ephemeral system message 注入。它借鉴了认知心理学的 memory consolidation 机制——定期给 Agent 设置"反省时刻"。

这个设计的精彩之处在于：它用最简单的工程手段（计数器 + 条件注入）解决了 Agent 最难的"自主学习意识"问题。

---

## Q5. Hermes 的 RL 训练闭环和普通 Fine-tuning 有什么不同？

### 主答模板
三个核心差异。

第一，**数据来源不同**。普通 Fine-tuning 用合成数据或人工标注数据。Hermes 的 RL 训练数据来自 Agent 在真实任务上的完整轨迹——包含工具调用、推理过程、纠错行为。这是真实分布数据，质量天花板完全不同。

第二，**闭环程度不同**。Hermes 的 RL 不是一次性训练，而是完整闭环：Agent 生产轨迹 → 轨迹压缩 → GRPO 训练 → 评估 → 部署回 Agent → 继续生产更好的轨迹。产品本身就是数据 pipeline 的前端。

第三，**训练目标不同**。不是微调语言能力，而是用 GRPO 强化 tool-calling 能力。奖励函数可以做真实验证——编译代码、读取文件、访问网络——而不仅仅是文本匹配。

但有一个关键认知：**RL 训练不是"从用户数据学习"，而是知识蒸馏**——把 Claude Opus 的 Agent 能力压缩到小模型里，实现降本和本地化部署。直接用用户对话训练大概率把模型训废。

---

## Q6. 你在 TinyClaw 里做的 RAG-to-Skill 和 Hermes 的 Skills 系统有什么异同？

### 主答模板
相同点：都是把经验沉淀为可复用的程序性知识，都采用渐进式加载避免 token 浪费。

核心差异：Hermes 的 Skill 创建是 Agent 自驱动的（后台审查 + KEPA），TinyClaw 的 RAG-to-Skill 是流程驱动的（IB 过滤 → Verifier-grounded SSL 编译 → 渐进式检索撤回）。Hermes 依赖 Agent 的判断力决定"什么时候该创建"，TinyClaw 依赖 Verifier 的确定性验证决定"什么知识值得编译成 Skill"。

另一个差异：Hermes 的 Skill 是纯自然语言 Markdown，TinyClaw 的 Skill 是经过结构化编译的（从 RAG 检索结果到可执行程序性知识）。这意味着 TinyClaw 的 Skill 质量更可控（verifier 验证），但灵活性不如 Hermes（不能靠 Agent 自己写自然语言解决 unforeseen 场景）。

还有一个关键差异：SkillsBench 的发现——自生成 Skill 平均 -1.3pp 负迁移。我和 Hermes 都要面对同样的核心挑战：怎么保证 Agent 自动创建的 Skill 不会反而降低能力。Hermes 靠安全扫描+信任分级，我靠 Verifier 验证+IB 过滤。

---

# 十一、面试金句

1. **"Hermes 的自进化不是 marketing wording，是 skill_manage 的 6 个 action + periodic nudge + FTS5 session search 的合力。"**

2. **"KEPA 的本质：不是更新模型权重，而是更新'如何使用模型'的策略。零训练成本、可解释可编辑、跨模型可迁移。"**

3. **"Skill 以 User Message 注入而非 System Prompt——牺牲一点指令跟随可靠性，换取 90%+ 的 API 成本节约。这是整个系统最关键的架构决策。"**

4. **"OpenClaw 有声明性记忆（what happened），Hermes 有程序性记忆（how to do it）。差别不在记忆的'量'，而在记忆的'类型'。"**

5. **"Agent 不是无限自律的。Periodic Nudge 用最简单的工程手段——计数器 + 条件注入——解决了 Agent 最难的'自主学习意识'问题。"**

6. **"Nous Research 把 Agent 产品当成训练数据 pipeline 的前端。Agent 不应被当作推理时应用，而是训练-推理-数据-再训练的闭环节点。"**

7. **"3,575 字符的硬约束不是 bug，是刻意设计——逼迫 agent 做 curation 而非 accumulation。"**

8. **"Hermes 的 10 个核心模式不是让你照搬代码，而是回答一个根本性问题：Agent 的知识，究竟应该以什么形式存在、以什么方式演化？"**

---

# 十二、和你的项目串联

面试时最强的 Hermes 串联叙事：

> "我在研究 Hermes Agent 的自进化机制时，最受启发的三个设计是：第一，KEPA——用 Prompt 更新替代权重更新实现零成本进化。这和我在 TinyClaw 里做 RAG-to-Skill 的思路是相通的——都是在不改模型的前提下让 Agent 的能力持续增长。第二，Periodic Nudge——用最简单的计数器+条件注入解决 Agent 的自主学习意识问题。这让我反思 TinyClaw 的程序性记忆编译是否也应该有一个类似的'定期复盘'触发。第三，Trajectory Pipeline——把 Agent 产品当作训练数据的前端。这和 GoAfar 里轨迹数据→GRPO 训练的思路完全一致，但我们没有做到 Hermes 那种'产品→数据→训练→部署回产品'的完整闭环——这是下一步的方向。"

---

# 十三、最后：Hermes 准备到什么程度才算真准备好了？

你至少要做到 8 件事：
1. **能讲出 Skills 闭环的 7 个阶段和每阶段的关键设计决策**
2. **能解释 KEPA 为什么叫"提示反向传播"以及和传统反向传播的本质区别**
3. **能讲出四层记忆架构和每一层的加载时机**
4. **能解释为什么 Skill 以 User Message 注入而不是 System Prompt（Prompt Cache）**
5. **能讲出 Periodic Nudge 的设计动机和行为经济学原理**
6. **能对比 Hermes 和 OpenClaw 的本质差异（声明性记忆 vs 程序性记忆）**
7. **能讲出 RL 训练闭环的完整流程和为什么不直接从用户数据学习**
8. **能把 TinyClaw/GoAfar/KTClaw 和 Hermes 的设计串联起来**
