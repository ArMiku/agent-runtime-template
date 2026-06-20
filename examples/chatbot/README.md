# Agent 对话机器人 Demo

一个基于 `agent_runtime/` 的流式 Agent 对话机器人:能用工具做精确数值计算、连接 MCP 搜索 B 站视频,并在前端实时展示 Agent 的**思考过程**——包括每次工具调用的参数与结果。

## 能力一览

| 能力 | 说明 |
|---|---|
| 🧮 数值计算 | `calculate` 工具,基于 `ast` 白名单求值,支持 `+ - * / // % **` 和括号,拒绝任意代码执行 |
| 📺 B 站搜索 | 通过 MCP 连接 `bilibili-mcp-js`,搜索视频并返回标题 / UP主 / 播放量 / 链接 |
| 💭 思考过程可视化 | 前端实时渲染 reasoning 流、工具调用(名称+参数)、工具结果、最终回答的交错时间线 |
| 🔁 多轮对话 + 指代消解 | 按 `session_id` 保留历史,支持"它""刚才那个结果"等指代,以及跨工具的链式推理 |

## 快速开始

### 1. 前置条件

- 已按 `AGENTS.md` 执行 `uv sync --extra dev`
- `node` / `npx` 可用(B 站 MCP 通过 `npx bilibili-mcp-js` 启动;首次运行会自动下载该包)
- 在项目根目录的 `.env` 中配置一个 OpenAI 兼容的 endpoint:

  ```sh
  OPENAI_API_KEY=sk-...
  OPENAI_MODEL=gpt-4o-mini            # 任意支持 function-calling 的模型
  OPENAI_API_BASE=https://api.openai.com/v1
  ```

  > 覆盖 OpenAI / DeepSeek / 智谱 / groq / xai / openrouter 等,改 `OPENAI_API_BASE` + `OPENAI_MODEL` 即可。

### 2. 运行

```sh
uv run --env-file .env python -m examples.chatbot
```

启动后打开 <http://127.0.0.1:8000>。

> 若 B 站 MCP 启动失败(如未装 node),服务会打印告警并继续运行——计算能力不受影响,仅视频搜索不可用。

## 体验思考过程

前端底部提供两组示例:

- **单轮试试**:点一下填入输入框,自己发送。
- **多轮指代 ▶ 依次点**:三个按钮**依次点击**即可发送,演示一段同会话的多轮推理:

  1. `用工具算一下 25 乘以 4` → 调 `calculate(25*4)` = **100**
  2. `把它再加上 50` → "它"消解为 100,调 `calculate(100+50)` = **150**
  3. `刚才那个结果除以 3，用算出来的数字加上「游戏」去B站搜视频`
     → "刚才那个结果"消解为 150,先 `calculate(150/3)`=**50**,再把结果拼成 `50游戏` 调 B 站搜索

右上角 **↺ 新会话** 可清空对话并开启新 `session_id`,方便重新演示。

## 架构

```
examples/chatbot/
├── __main__.py     # python -m examples.chatbot 入口
├── server.py       # aiohttp 后端:装配工具 + MCP,按轮驱动 runner,SSE 推送
├── calculator.py   # calculate 工具:ast 白名单安全求值
└── index.html      # 单页前端:SSE 客户端 + 思考过程时间线
```

**装配 → 执行 → 推送**的闭环:

1. **装配**(服务启动一次):`FunctionToolManager.enable_mcp_server(name, config)` 用**内联**配置直接连接 B 站 MCP(不写 `mcp_server.json`,不污染 `data/`),其工具与本地 `calculate` 合并进一个 `ToolSet`。
2. **执行**(每轮请求):构造 `ProviderRequest`(带上历史 `contexts`)→ `ToolLoopAgentRunner.reset(..., streaming=True)` → `step_until_done()` 逐步产出 `AgentResponse`。
3. **推送**:把每个 `AgentResponse` 翻译成一条 SSE 帧发给浏览器,前端按类型渲染。

### SSE 事件协议

`POST /api/chat`(请求体 `{"message": str, "session_id": str}`)返回 `text/event-stream`:

| event | data | 含义 |
|---|---|---|
| `delta` | `{"kind": "reasoning"\|"answer", "text": str}` | 流式增量:模型思考 / 回答正文 |
| `tool_call` | `{"id", "name", "args", "ts"}` | 一次工具调用(名称 + 参数) |
| `tool_result` | `{"id", "result", "ts"}` | 上述调用的返回结果 |
| `error` | `{"message": str}` | 运行错误 |
| `done` | `{"text": str}` | 本轮结束,附最终回答全文 |

这些事件直接来自 runner 的 `step_until_done()` 产出,前端据此把"思考 → 调工具 → 看结果 → 回答"的链路逐段画出来。

## 已知限制

- 会话历史存在内存里(进程级 `dict`),重启即清空——示例不引入持久化。
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
