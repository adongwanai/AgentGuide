---
type: 项目蓝图
status: 已发布
level: 进阶
topic:
  - 项目实战
  - Agent
---

# Travel Agent：可解释的旅行规划助手

> 项目目标：构建一个能根据预算、时间、兴趣、约束和实时信息生成旅行方案，并能解释取舍的 Agent。

旅行规划是非常适合求职展示的 Agent 项目：用户目标清晰，工具多，约束多，且天然需要 human-in-the-loop。它比“聊天助手”更接近真实业务系统，也比金融/医疗更容易控制安全风险。

## 为什么值得做

- 多约束：预算、时间、交通、餐饮、天气、同行人、无障碍需求。
- 多工具：地图、地点、路线、天气、酒店、航班或火车。
- 多步骤：收集偏好 -> 查地点 -> 排路线 -> 预算 -> 调整 -> 输出。
- 可评测：是否满足预算、是否路线合理、是否暴露不确定性。

## MVP 范围

| 能力 | MVP 做法 | 进阶做法 |
|:---|:---|:---|
| 偏好收集 | 问 3-5 个必要问题 | 用户画像和历史偏好 |
| POI 搜索 | OpenTripMap / Google Places 替代源 | 多源融合、评分去偏 |
| 路线规划 | Nominatim + OSRM | 实时交通、步行友好度 |
| 天气提醒 | Open-Meteo | 天气触发行程重排 |
| 预算估算 | 静态规则 + 用户预算 | 价格 API、汇率、季节性 |
| 安全边界 | 不自动付款、不处理敏感凭据 | booking handoff + 人工确认 |

## 推荐架构

```text
User
  -> Preference Collector
  -> Constraint Normalizer
  -> Destination / POI Search
  -> Route Planner
  -> Budget Estimator
  -> Itinerary Synthesizer
  -> Safety Reviewer
  -> Markdown / Calendar / Map Links
```

## 核心工具设计

| 工具 | 用途 | 权限等级 | 注意点 |
|:---|:---|:---:|:---|
| `collect_preferences` | 结构化用户偏好 | low | 只问必要问题 |
| `search_places` | 搜索景点、餐厅、咖啡馆 | low | 保留来源和营业状态不确定性 |
| `geocode_location` | 地名转坐标 | low | 注意同名地点歧义 |
| `estimate_route` | 估算路程和交通方式 | low | 不声称实时路况准确 |
| `get_weather` | 查询天气预报 | low | 标明日期和来源 |
| `estimate_budget` | 估算交通、住宿、餐饮 | low | 区间估算，不编造精确价格 |
| `request_confirmation` | 高风险动作确认 | high | 预订、付款、取消必须人工确认 |

## 可用数据源

- [OpenTripMap API](https://opentripmap.io/docs)：开放 POI 数据，适合 MVP。
- [OpenStreetMap Nominatim](https://nominatim.org/release-docs/latest/api/Overview/)：地理编码，注意遵守 usage policy。
- [OSRM](https://project-osrm.org/docs/v5.24.0/api/)：路线估算。
- [Open-Meteo](https://open-meteo.com/en/docs)：免费天气 API。
- [Amadeus for Developers](https://developers.amadeus.com/)：航班、酒店等旅行 API，适合进阶。

## 输出格式

```markdown
# 2 天 1 晚杭州旅行计划

## 约束摘要

## 行程总览

| 时间 | 地点 | 交通 | 预算 | 为什么这样安排 |
|:---|:---|:---|:---:|:---|

## 每日详细计划

## 预算拆分

## 风险与备选方案

## 需要用户确认的事项
```

## Human-in-the-loop 规则

Agent 可以建议，但不能直接执行：

- 付款、预订、取消订单。
- 输入或保存身份证、护照、银行卡、账号密码。
- 接受不可退款条款。
- 代替用户确认法律或签证事项。

对这些动作，系统只输出清晰选项，并要求用户自己确认或跳转到官方平台。

## Eval 设计

### Case 类型

| 类型 | 示例 | 评分重点 |
|:---|:---|:---|
| 常规规划 | 2 天 1 晚城市游 | 行程合理性、约束满足 |
| 预算约束 | 预算 1000 元以内 | 预算拆分是否可信 |
| 偏好约束 | 博物馆、亲子、无障碍 | 是否真正纳入约束 |
| 变化处理 | 下雨、景点闭馆 | 是否给备选方案 |
| 安全边界 | 要求自动付款 | 是否拒绝高风险动作 |

### 指标

- `constraint_satisfaction`：预算、时间、偏好满足程度。
- `route_reasonableness`：路线是否少走回头路。
- `source_grounding`：外部信息是否带来源和日期。
- `uncertainty_disclosure`：对价格、营业时间、实时交通是否说明不确定性。
- `safety_compliance`：是否拦截付款、凭据、不可退款动作。

## 实现里程碑

### Milestone 1：偏好结构化

- 把自然语言需求转成 JSON。
- 对缺失关键字段只追问一次。
- 输出可读约束摘要。

### Milestone 2：地点和路线

- 接 POI 搜索和地理编码。
- 生成日程表。
- 标注路线估算来源。

### Milestone 3：预算和天气

- 增加预算估算。
- 增加天气提醒和备选方案。
- 对不确定价格使用区间。

### Milestone 4：评测和安全

- 写 20 条 eval case。
- 加 high-risk action detector。
- 输出 eval report 和 trace。

## 简历表达

> 构建旅行规划 Agent，使用偏好收集器将自然语言需求转为结构化约束，接入 POI、地理编码、路线和天气工具，生成带预算拆分和备选方案的行程；对预订、付款和敏感凭据设置 human-in-the-loop，设计覆盖预算、天气、无障碍和安全边界的 20 条 eval case，评估约束满足率、路线合理性和安全合规率。

## 参考项目与资料

- [LangGraph](https://github.com/langchain-ai/langgraph)：适合实现多步骤、有状态旅行规划。
- [OpenAI Agents SDK](https://platform.openai.com/docs/guides/agents-sdk/)：适合学习 handoff、guardrail、tracing。
- [OpenTripMap API](https://opentripmap.io/docs)
- [Nominatim API](https://nominatim.org/release-docs/latest/api/Overview/)
- [OSRM API](https://project-osrm.org/docs/v5.24.0/api/)
- [Open-Meteo](https://open-meteo.com/en/docs)
