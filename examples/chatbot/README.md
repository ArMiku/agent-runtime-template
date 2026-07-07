# Agent 对话机器人 Demo

一个基于 `agent_runtime/` 的流式 Agent 对话机器人:能用工具做精确数值计算、联网搜索(Tavily)、连接 MCP 搜索 B 站视频,并在前端实时展示 Agent 的**思考过程**——包括每次工具调用的参数与结果。

## 能力一览

| 能力 | 说明 |
|---|---|
| 🧮 数值计算 | `calculate` 工具,基于 `ast` 白名单求值,支持 `+ - * / // % **` 和括号,拒绝任意代码执行 |
| 🔍 联网搜索 | `web_search` 工具,基于 Tavily 做实时联网搜索,返回 Top 结果(标题/链接/内容摘要)+ 生成式答案,用于新闻、最新事件、价格等时效性信息。未配置 `TAVILY_API_KEY` 时自动缺省,不影响其它能力 |
| 📺 B 站搜索 | 通过 MCP 连接 `bilibili-mcp-js`,搜索视频并返回标题 / UP主 / 播放量 / 链接 |
| 💭 思考过程可视化 | 前端实时渲染 reasoning 流、工具调用(名称+参数)、工具结果、最终回答的交错时间线 |
| 🔁 多轮对话 + 指代消解 | 按 `session_id` 保留历史,支持"它""刚才那个结果"等指代,以及跨工具的链式推理 |
| 📝 plan-and-execute | 默认开启 `planning` 扩展:面对开放式多步任务,Agent 用 `write_todos` 列 plan、逐项推进;plan 每步注入 system message,未完成时阻止过早结束。设 `CHATBOT_PLANNING=0` 可关闭,退回纯 ReAct |
| 🧭 显式编排 (`/plan`) | 输入 `/plan <任务>` 手动开启路线 1——显式 `PLAN → EXEC → REPLAN` 状态机:planner 一步产出完整 plan,executor 逐项执行,replanner 据结果修订。前端置顶渲染 plan 清单的演进 |

## 快速开始

### 1. 前置条件

- 已按 `AGENTS.md` 执行 `uv sync --extra dev`
- `node` / `npx` 可用(B 站 MCP 通过 `npx bilibili-mcp-js` 启动;首次运行会自动下载该包)
- 在项目根目录的 `.env` 中配置一个 OpenAI 兼容的 endpoint:

  ```sh
  OPENAI_API_KEY=sk-...
  OPENAI_MODEL=gpt-4o-mini            # 任意支持 function-calling 的模型
  OPENAI_API_BASE=https://api.openai.com/v1
  TAVILY_API_KEY=tvly-...             # 可选:启用 web_search 联网搜索(留空则该工具自动缺省)
  ```

  > 覆盖 OpenAI / DeepSeek / 智谱 / groq / xai / openrouter 等,改 `OPENAI_API_BASE` + `OPENAI_MODEL` 即可。

### 2. 运行

```sh
uv run --env-file .env python -m examples.chatbot
```

启动后打开 <http://127.0.0.1:8000>。

> 若 B 站 MCP 启动失败(如未装 node),服务会打印告警并继续运行——计算能力不受影响,仅视频搜索不可用。
> 同理,未安装 `tavily-python` 或未设 `TAVILY_API_KEY` 时,`web_search` 工具会自动缺省(打印告警),其余能力照常可用。

## 体验思考过程

前端底部提供两组示例:

- **单轮试试**:点一下填入输入框,自己发送。
- **多轮指代 ▶ 依次点**:三个按钮**依次点击**即可发送,演示一段同会话的多轮推理:

  1. `用工具算一下 25 乘以 4` → 调 `calculate(25*4)` = **100**
  2. `把它再加上 50` → "它"消解为 100,调 `calculate(100+50)` = **150**
  3. `刚才那个结果除以 3，用算出来的数字加上「游戏」去B站搜视频`
     → "刚才那个结果"消解为 150,先 `calculate(150/3)`=**50**,再把结果拼成 `50游戏` 调 B 站搜索

右上角 **↺ 新会话** 可清空对话并开启新 `session_id`,方便重新演示。

## 显式编排:`/plan`

在消息前加 `/plan ` 即可让**这一轮**走显式 plan-and-execute 状态机(路线 1),而非默认的 ReAct+涌现式 planning(路线 2)。例如:

```
/plan 先用 calculate 算 25 乘以 4，再搜索「AI」的B站视频
```

这一轮会:

1. **PLAN** —— planner 一步(`tool_choice="required"` + `submit_plan`)产出完整 todo 清单,前端置顶渲染为 `📋 计划` 卡片;
2. **EXEC** —— 每个 todo 委托一个**隔离的子 ReAct runner** 用 `calculate` / B 站搜索执行,工具调用与结果照常走 `tool_call` / `tool_result` 事件,只把单个 todo 的结果摘要回收进主上下文;
3. **REPLAN** —— 据摘要修订剩余 plan,卡片刷新;仍有未完成项则继续 EXEC,全完成则结束。

> 路线 1 与路线 2 正交。`/plan` 轮自包含、不复用聊天历史,且每轮重新规划(清掉上一次 `/plan` 的进度)。两条路线共享 `Todo`/`PluginStore` seam,但 `/plan` 用独立的 store,与 `write_todos` 的 plan 互不干扰。

### 接思考(reasoning)模型报 `submit_plan ... missing` 时

planner 是一次 `tool_choice="required"` 的结构化调用。思考模型可能把 output 全花在推理上、被 token 上限截断而没产出 `submit_plan` 工具调用。报错信息会明确指向下面两个旋钮之一——**本 demo 默认不设预算**（不同模型上下文窗口差几个数量级，预算是模型决策，不该写死），按你的模型按需开启：

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `CHATBOT_PLANNER_MAX_TOKENS` | _(不设)_ | planner 单次调用的输出 token 上限。报 missing 时调大(如 `16384`) |
| `CHATBOT_PLANNER_EXTRA_BODY` | _(不设)_ | JSON,作为 SDK `extra_body` 透传以关思考:`'{"thinking": false}'`(GLM/DeepSeek)或 `'{"reasoning_effort": "low"}'`(OpenAI o-series)。**首选**——planner 一步产出结构化 plan 不需要 chain-of-thought，关掉思考既防截断又省 token |

```sh
# GLM / DeepSeek 等思考模型：直接关掉 planner 的思考（推荐）
CHATBOT_PLANNER_EXTRA_BODY='{"thinking": false}' \
uv run --env-file .env python -m examples.chatbot

# 或：保留思考但给够预算
CHATBOT_PLANNER_MAX_TOKENS=16384 \
uv run --env-file .env python -m examples.chatbot
```

### `/plan` 的 liveness 兜底:撞限不再静默卡死

`/plan` 轮有三道**任务无关**的 liveness 护栏,撞任一道都转 `ERROR` 终态并把**已完成部分**作为 `done` 文本回传(不再发空 `done` 让前端静默卡住)。三者各抓一类失控:

| 旋钮 | 默认 | 抓什么 |
| --- | --- | --- |
| `CHATBOT_PLAN_MAX_STEPS` | `200` | 灾难兜底:总步数跑飞(前两道全失效时的最后防线) |
| `CHATBOT_PLAN_CALL_TIMEOUT` | `150`(秒) | 单次 LLM 调用 in-call 挂死(网络僵死,步数/总时长都看不见) |
| `CHATBOT_PLAN_TURN_DEADLINE` | `1500`(秒,25 分钟) | 整轮累积跑飞(很多次正常调用数量失控,研究类典型) |

三个默认都**故意设得很宽松**(灾难才触及)。收紧它们 = 把"这类任务不该超过 N 步/N 分钟"这种猜任务形状的误伤换个标签请回来——让任务**收敛**是 planner 的职责(它的 prompt 产出有界 todo),这些护栏只在已经失控时兜底诚实退出。所以正常的深度研究/重测试任务不会被默认值掐断;真要调,通常是放得更宽,而非收紧。

## 架构

```
examples/chatbot/
├── __main__.py     # python -m examples.chatbot 入口
├── server.py       # aiohttp 后端:装配工具 + MCP,按轮驱动 runner,SSE 推送
├── calculator.py   # calculate 工具:ast 白名单安全求值
├── web_search.py   # web_search 工具:Tavily 联网搜索(缺省式降级)
└── index.html      # 单页前端:SSE 客户端 + 思考过程时间线
```

**装配 → 执行 → 推送**的闭环:

1. **装配**(服务启动一次):`FunctionToolManager.enable_mcp_server(name, config)` 用**内联**配置直接连接 B 站 MCP(不写 `mcp_server.json`,不污染 `data/`),其工具与本地 `calculate` / `web_search` 合并进一个 `ToolSet`。开启 planning 时,`write_todos` 工具也加入该 `ToolSet`,并建一个进程级 `InMemoryPluginStore` 存 plan。
2. **执行**(每轮请求):构造 `ProviderRequest`(带上历史 `contexts`)→ `ToolLoopAgentRunner.reset(..., streaming=True)` → `step_until_done()` 逐步产出 `AgentResponse`。开启 planning 时,`agent_hooks` 传入**每轮新建**的 `PlanningHook`(指向上面的共享 store):reminder 计数每轮归零,但 plan 经 store 跨轮持久。
3. **推送**:把每个 `AgentResponse` 翻译成一条 SSE 帧发给浏览器,前端按类型渲染。`write_todos` 的调用与结果走的是既有的 `tool_call` / `tool_result` 事件,所以前端无需改动即可看到 plan 的演进。

### SSE 事件协议

`POST /api/chat`(请求体 `{"message": str, "session_id": str}`)返回 `text/event-stream`:

| event | data | 含义 |
|---|---|---|
| `delta` | `{"kind": "reasoning"\|"answer", "text": str}` | 流式增量:模型思考 / 回答正文 |
| `tool_call` | `{"id", "name", "args", "ts"}` | 一次工具调用(名称 + 参数) |
| `tool_result` | `{"id", "result", "ts"}` | 上述调用的返回结果 |
| `plan` | `{"text": str}` | `/plan` 轮的 PLAN/REPLAN 产出的 plan 清单(置顶刷新) |
| `error` | `{"message": str}` | 运行错误 |
| `done` | `{"text": str}` | 本轮结束,附最终回答全文 |

这些事件直接来自 runner 的 `step_until_done()` 产出,前端据此把"思考 → 调工具 → 看结果 → 回答"的链路逐段画出来。

## 已知限制

- 会话历史存在内存里(进程级 `dict`),重启即清空——示例不引入持久化。
- planning 的 plan 同样存在进程级 `InMemoryPluginStore`,重启即清空;要跨进程恢复,把它换成 host 注入的持久化 `PluginStore` 实现即可(plan 按 `session_id` 隔离)。
- 服务监听 `127.0.0.1:8000`,无鉴权,仅供本地体验,不要直接暴露到公网。
- 计算工具刻意只支持纯算术;不提供变量、函数调用等,以保证求值安全。

## 关于控制台的 `JSONRPCMessage` 报错

如果你直接连第三方 `bilibili-mcp-js`,可能在控制台看到一堆这样的 traceback:

```
pydantic_core._pydantic_core.ValidationError: 1 validation error for JSONRPCMessage
  Invalid JSON: ... input_value="  user_cover: '',"
Failed to parse JSONRPC message from server
```

**这不是 agent_runtime 或本 demo 的 bug,功能也完全正常。** 根因是:`bilibili-mcp-js`
在执行搜索时把原始视频对象用 `console.log` 打到了 **stdout**(应走 stderr),而 stdio MCP
协议规定 stdout 只能承载逐行 JSON-RPC 消息。MCP 客户端逐行解析时,这些 JS 对象 dump
(单引号、键名无引号)无法当 JSON 解析,于是每行都打一条 traceback——但客户端对坏行
`continue` 容错,真正的工具结果仍会正常返回。

本 demo 已在 `server.py` 里加了一个**精准过滤器**,只丢弃 "Failed to parse JSONRPC message"
这一条日志记录,保留该 logger 的其它真实错误。因此正常运行时控制台是干净的。
