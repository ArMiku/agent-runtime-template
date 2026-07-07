# Agent Runtime SDK 使用说明

> 一个框架无关的 Agent 运行时 + LLM Provider 模板。核心是一个 ReAct 工具循环，
> 通过四个显式接缝（Provider / ToolExecutor / Hooks / Context）对外扩展，并附带
> skills、fs、plugins、planning 等可选扩展。

- **包名**：`agent_runtime`（唯一可导入包；`examples/`、`tests/` 在包外）
- **Python**：`>=3.11`
- **License**：AGPL-3.0-or-later

---

## 目录

1. [安装](#1-安装)
2. [快速开始](#2-快速开始)
3. [核心概念](#3-核心概念)
4. [三层装配 API](#4-三层装配-api)
5. [Provider：接入 LLM](#5-provider接入-llm)
6. [Tools：定义工具](#6-tools定义工具)
7. [Hooks：生命周期钩子](#7-hooks生命周期钩子)
8. [扩展：skills / fs / plugins / planning](#8-扩展skills--fs--plugins--planning)
9. [Runner：控制流范式](#9-runner控制流范式)
10. [API 参考](#10-api-参考)
11. [常见问题](#11-常见问题)

---

## 1. 安装

```bash
# 推荐使用 uv（隔离、可复现）
uv sync --extra dev

# 或 pip
pip install -e ".[dev]"
```

环境变量放在 `.env`（已 gitignore），通过 `uv run --env-file .env ...` 加载。

---

## 2. 快速开始

### 最小示例：50 行完成一次工具调用

```python
import asyncio
import os

from agent_runtime.core import FunctionToolExecutor, SessionContext
from agent_runtime.core.hooks import BaseAgentRunHooks
from agent_runtime.core.run_context import ContextWrapper
from agent_runtime.core.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from agent_runtime.core.tool import FunctionTool, ToolSet
from agent_runtime.provider.entities import ProviderRequest
from examples.example_provider import make_openai_compat_provider


class AddTool(FunctionTool):
    """自定义工具：重写 call() 方法。"""

    def __init__(self):
        super().__init__(
            name="add",
            description="Add two integers.",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "first addend"},
                    "b": {"type": "integer", "description": "second addend"},
                },
                "required": ["a", "b"],
            },
        )

    async def call(self, context, **kwargs) -> str:
        return str(int(kwargs["a"]) + int(kwargs["b"]))


async def main():
    # 1. Provider
    provider = make_openai_compat_provider(
        api_key=os.environ["OPENAI_API_KEY"],
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        api_base=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
    )

    # 2. Tools
    tools = ToolSet()
    tools.add_tool(AddTool())

    # 3. Request
    request = ProviderRequest(
        prompt="What is 7 + 5? Use the add tool.",
        system_prompt="You are a helpful assistant.",
        func_tool=tools,
    )

    # 4. Runner
    run_context = ContextWrapper(context=SessionContext(session_id="demo"))
    runner = ToolLoopAgentRunner()
    await runner.reset(
        provider=provider,
        request=request,
        run_context=run_context,
        tool_executor=FunctionToolExecutor(provider),
        agent_hooks=BaseAgentRunHooks(),
    )

    async for _ in runner.step_until_done(max_step=10):
        pass

    final = runner.get_final_llm_resp()
    print(final.completion_text)  # "7 + 5 = 12"


if __name__ == "__main__":
    asyncio.run(main())
```

运行：

```bash
OPENAI_API_KEY=sk-... uv run --env-file .env python my_script.py
```

### 更简单：一行装配（Tier 3）

如果你需要 skills / fs / plugins，用 `build_local_agent` 免去手写 `reset` 和 step 循环：

```python
from agent_runtime.local_runtime import build_local_agent

agent = await build_local_agent(provider, prompt="greet me")
final = await agent.run(max_step=8)
print(final.completion_text)
await agent.aclose()
```

---

## 3. 核心概念

运行时围绕 **四个显式接缝**（seams）设计，彼此正交、可独立替换：

| 接缝 | 抽象基类 | 默认实现 | 职责 |
|------|----------|----------|------|
| **Provider** | `Provider` | `ProviderOpenAIOfficial` | 与 LLM 通信（text_chat / stream）|
| **ToolExecutor** | `BaseFunctionToolExecutor` | `FunctionToolExecutor` | 执行工具调用（handler/call 分派 + 子 agent handoff）|
| **Hooks** | `BaseAgentRunHooks` | `BaseAgentRunHooks`（无操作）| 生命周期钩子（注入上下文、否决完成等）|
| **Context** | `ContextWrapper[TContext]` | `SessionContext` | 承载会话状态与消息列表 |

**数据流**：

```
ProviderRequest ──> Runner.reset() ──> step_until_done()
                                          │
                                          ├─ on_llm_request  (hook)
                                          ├─ provider.text_chat()  ──> LLMResponse
                                          ├─ 有工具调用? ──> tool_executor.execute()
                                          │                     ├─ on_tool_start (hook)
                                          │                     └─ on_tool_end   (hook)
                                          ├─ 无工具调用? ──> on_before_complete (hook, 可否决)
                                          └─ done ──> on_agent_done (hook)
                                          
get_final_llm_resp() ──> LLMResponse
```

**约定**：整个契约中没有 `event` / `platform` 对象。工具的第一个参数永远是
`ContextWrapper`，返回 `str` 或 `mcp.types.CallToolResult`。

---

## 4. 三层装配 API

运行时提供三种装配粒度，从完全手工到一行调用。**它们产出的执行结果等价**（见
`examples/local_runtime_demo.py` 中的 oneshot vs manual 断言）。

### Tier 1 — 手工装配

直接导入细粒度构件，自己组装 tools/hooks 并驱动 runner。适合需要完全掌控 runner
生命周期的宿主。下面逐步拆解每个接口的构造和调用。

#### 步骤 1：构建 Provider

```python
from examples.example_provider import make_openai_compat_provider

provider = make_openai_compat_provider(
    api_key="sk-...",
    model="gpt-4o-mini",
    api_base="https://api.openai.com/v1",
)
```

或自定义实现（见[第 5 节](#5-provider接入-llm)）。

#### 步骤 2：构建 ToolSet

```python
from agent_runtime.core.tool import FunctionTool, ToolSet

# 方式 A：子类 + call()
class MyTool(FunctionTool):
    def __init__(self):
        super().__init__(
            name="my_tool",
            description="...",
            parameters={"type": "object", "properties": {...}, "required": [...]},
        )

    async def call(self, context, **kwargs) -> str:
        return "result"

# 方式 B：函数式 handler
async def _handler(run_context, **kwargs) -> str:
    return "result"

fn_tool = FunctionTool(
    name="fn_tool",
    description="...",
    parameters={"type": "object", "properties": {}},
    handler=_handler,
)

# 组装
tools = ToolSet()
tools.add_tool(MyTool())
tools.add_tool(fn_tool)
```

**ToolSet API**：

| 方法 | 签名 | 说明 |
|------|------|------|
| `add_tool` | `(tool: FunctionTool) -> None` | 添加单个工具 |
| `merge` | `(other: ToolSet) -> None` | 合并另一个 ToolSet 的全部工具 |
| `names` | `() -> list[str]` | 返回所有工具名称 |
| `get_tool` | `(name: str) -> FunctionTool \| None` | 按名称查找 |
| `empty` | `() -> bool` | 是否为空 |

#### 步骤 3：构建 ProviderRequest

```python
from agent_runtime.provider.entities import ProviderRequest

request = ProviderRequest(
    prompt="What is 7+5?",          # 用户提示词（与 contexts 二选一或叠加）
    system_prompt="You are ...",     # 系统提示词
    func_tool=tools,                 # ToolSet（runner 会自动序列化给 LLM）
    # --- 以下为可选 ---
    session_id="sess-001",           # 会话 ID
    image_urls=[],                   # 多模态：图片 URL 列表
    audio_urls=[],                   # 多模态：音频 URL/路径列表
    contexts=[],                     # OpenAI 格式历史消息（与 prompt 可叠加）
    model=None,                      # 覆盖 provider 默认模型
    extra_user_content_parts=[],     # 额外用户消息内容块
    tool_calls_result=None,          # 回传上次的工具调用结果
)
```

> `prompt` 和 `contexts` 可同时使用：`prompt` 会作为最新 user 消息追加到 `contexts` 末尾。

#### 步骤 4：构建 ContextWrapper

```python
from agent_runtime.core import SessionContext
from agent_runtime.core.run_context import ContextWrapper

run_context = ContextWrapper(
    context=SessionContext(session_id="demo", user_id=None),
    messages=[],              # 初始化为空；runner.reset 会自动填充
    tool_call_timeout=120,    # 单次工具调用超时（秒），默认 120
)
```

**SessionContext** 是默认的 `TContext` 实现（仅 `session_id` + `user_id`）。
如需自定义上下文，实现任意 dataclass 替代 `SessionContext`，泛型 `ContextWrapper[YourContext]` 即可。

**ContextWrapper 字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `context` | `TContext` | 你的自定义会话上下文 |
| `messages` | `list[Message]` | runner 维护的 LLM 消息列表（hook 可直接读写）|
| `tool_call_timeout` | `int` | 工具调用超时秒数 |

#### 步骤 5：构建 FunctionToolExecutor

```python
from agent_runtime.core import FunctionToolExecutor

executor = FunctionToolExecutor(
    provider,                       # 用于子 agent handoff
    agent_hooks=None,               # 可选：传给 HandoffTool 子 runner 的 hooks
    max_sub_agent_steps=30,         # 子 agent 最大步数（HandoffTool 场景）
)
```

**BaseFunctionToolExecutor** 抽象接口（自定义 executor 重写它）：

```python
class BaseFunctionToolExecutor(Generic[TContext]):
    def execute(
        self,
        tool: FunctionTool,
        run_context: ContextWrapper[TContext],
        **tool_args,
    ) -> AsyncGenerator[ToolExecResult, None]:
        """执行工具，yield 结果。ToolExecResult = str | mcp.types.CallToolResult"""
        ...
```

默认实现 `FunctionToolExecutor` 的分派逻辑：
1. `HandoffTool` → 启动子 ReAct runner
2. 有 `handler` → `tool.handler(run_context, **params)`
3. 否则 → `tool.call(run_context, **params)`

#### 步骤 6：构建 Hooks（可选）

```python
from agent_runtime.core.hooks import BaseAgentRunHooks

# 无操作（默认）
hooks = BaseAgentRunHooks()

# 或自定义
class MyHook(BaseAgentRunHooks):
    async def on_llm_request(self, run_context):
        # 每次 LLM 调用前触发
        ...

    async def on_before_complete(self, run_context, llm_response) -> bool:
        # 返回 False 拒绝完成
        return True
```

多个 hook 用 `ChainedAgentRunHooks` 组合：

```python
from agent_runtime.core.hooks_chain import ChainedAgentRunHooks
hooks = ChainedAgentRunHooks(MyHook(), AnotherHook())
```

#### 步骤 7：初始化 Runner（`reset`）

```python
from agent_runtime.core.runners.tool_loop_agent_runner import ToolLoopAgentRunner

runner = ToolLoopAgentRunner()
await runner.reset(
    # --- 5 个必选参数 ---
    provider=provider,
    request=request,
    run_context=run_context,
    tool_executor=executor,
    agent_hooks=hooks,
    # --- 可选参数 ---
    streaming=False,                     # True 开启流式（step 会 yield chunk）
    enforce_max_turns=-1,                # 最大轮次，-1 不限
    # 上下文压缩相关
    llm_compress_instruction=None,       # LLM 压缩时的指令
    llm_compress_keep_recent_ratio=0.15, # 压缩时保留最近消息的比例
    llm_compress_provider=None,          # 用于压缩的 provider（默认同主 provider）
    truncate_turns=1,                    # 按轮数截断时保留的轮数
    custom_token_counter=None,           # 自定义 token 计数器
    custom_compressor=None,              # 自定义上下文压缩器
    # 工具 schema 模式
    tool_schema_mode="full",             # "full" | "skills_like"（减少 token）
    # 容错
    fallback_providers=None,             # 降级 provider 列表
    # 工具结果溢出
    tool_result_overflow_dir=None,       # 过长工具结果写入此目录
    read_tool=None,                      # 配合 overflow 使用的 read 工具
)
```

**`reset` 做了什么**：
1. 把 `request.system_prompt` 作为首条 system message 写入 `run_context.messages`
2. 把 `request.prompt`（及 image/audio）作为 user message 追加到消息列表
3. 初始化上下文管理器（压缩/截断配置）
4. 准备 fallback providers 去重列表
5. 将 runner 状态置为 IDLE

#### 步骤 8：驱动执行循环

```python
# 方式 A：一次性跑完
async for response in runner.step_until_done(max_step=20):
    # response: AgentResponse
    #   .type: "llm_result" | "streaming_delta" | "err"
    #   .data.chain: MessageChain（包含文本/reasoning）
    pass

# 方式 B：手动逐步
while not runner.done():
    async for response in runner.step():
        # 处理单步结果
        ...
```

**`step_until_done(max_step)` 行为**：
- 循环调用 `step()` 直到 `done()` 返回 True 或达到 `max_step`
- 达到 `max_step` 时：拔掉所有工具 + 注入 "请总结" 提示 → 强制最后一轮 LLM（无工具可调用，必然结束）

#### 步骤 9：获取最终结果

```python
final = runner.get_final_llm_resp()  # -> LLMResponse | None

if final:
    print(final.completion_text)         # 纯文本回复
    print(final.reasoning_content)       # 推理内容（deepseek-r1/o1 等）
    print(final.usage)                   # TokenUsage(input_other, input_cached, output)
```

#### AgentResponse 结构

`step()` / `step_until_done()` yield 的中间结果：

```python
@dataclass
class AgentResponse:
    type: str                # "llm_result" | "streaming_delta" | "err"
    data: AgentResponseData  # .chain: MessageChain

class AgentResponseData(TypedDict):
    chain: MessageChain      # 消息链（含 text / reasoning 等组件）
```

| type | 含义 |
|------|------|
| `"llm_result"` | 一次完整 LLM 响应（含 completion 或 reasoning）|
| `"streaming_delta"` | 流式增量 chunk（`streaming=True` 时）|
| `"err"` | LLM 调用出错 |

#### 完整手动装配示例

```python
import asyncio
from agent_runtime.core import FunctionToolExecutor, SessionContext
from agent_runtime.core.hooks import BaseAgentRunHooks
from agent_runtime.core.run_context import ContextWrapper
from agent_runtime.core.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from agent_runtime.core.tool import FunctionTool, ToolSet
from agent_runtime.provider.entities import ProviderRequest
from examples.example_provider import make_openai_compat_provider


async def main():
    # 1. Provider
    provider = make_openai_compat_provider(api_key="sk-...", model="gpt-4o-mini")

    # 2. Tools
    async def _add(ctx, **kw) -> str:
        return str(int(kw["a"]) + int(kw["b"]))

    tools = ToolSet([FunctionTool(
        name="add", description="Add two numbers.",
        parameters={"type": "object", "properties": {
            "a": {"type": "integer"}, "b": {"type": "integer"}
        }, "required": ["a", "b"]},
        handler=_add,
    )])

    # 3. Request
    request = ProviderRequest(prompt="7+5=?", system_prompt="Use tools.", func_tool=tools)

    # 4. Context
    run_context = ContextWrapper(context=SessionContext(session_id="s1"))

    # 5. Executor
    executor = FunctionToolExecutor(provider)

    # 6. Runner
    runner = ToolLoopAgentRunner()
    await runner.reset(
        provider=provider,
        request=request,
        run_context=run_context,
        tool_executor=executor,
        agent_hooks=BaseAgentRunHooks(),
    )

    # 7. 执行
    async for resp in runner.step_until_done(max_step=10):
        if resp.type == "llm_result":
            print(f"[step] {resp.data['chain'].get_plain_text()}")

    # 8. 最终结果
    final = runner.get_final_llm_resp()
    print(f"Answer: {final.completion_text}")


asyncio.run(main())
```

### Tier 2 — `build_local_agent_basics`

把 skills（+ 可选 fs + 可选 plugin 贡献 + 可选 planning）装配成一个 bundle，
**不含** provider / runner / request，交给你自己接管 runner 生命周期。

```python
from agent_runtime.local_runtime import build_local_agent_basics

basics = build_local_agent_basics(
    include_fs=True,
    include_planning=False,
    contributions=[...],   # 插件贡献
)
# basics.skill_manager / basics.tools / basics.hooks / basics.planning_hook
```

返回 `LocalAgentBasics`：

| 字段 | 类型 | 说明 |
|------|------|------|
| `skill_manager` | `SkillManager` | skill 发现管理器 |
| `tools` | `ToolSet` | 可变工具集（`add_tool` / `merge`）|
| `hooks` | `BaseAgentRunHooks` | 已链接好的钩子（单个时退化为该钩子本身）|
| `planning_hook` | `PlanningHook \| None` | 仅 `include_planning=True` 时非空 |

### Tier 3 — `build_local_agent`

在 Tier 2 之上再构建 `ProviderRequest` / `ContextWrapper` / `FunctionToolExecutor` /
runner，并 `await runner.reset`，返回可直接 `run()` 的 `LocalAgent`。**你只需提供
`provider`**。

```python
agent = await build_local_agent(
    provider,
    prompt="...",
    system_prompt="",
    include_fs=True,
    include_planning=False,
    runner_type="react",       # 或 "plan_execute"
    max_turns=-1,              # 映射到 enforce_max_turns
    streaming=False,
    # **runner_kwargs 直通 runner.reset（fallback_providers、llm_compress_* 等）
)
final = await agent.run(max_step=20)
await agent.aclose()           # 关闭自建的 tool executor
```

`LocalAgent` 方法：`step()` / `step_until_done(max_step)` / `run(max_step=20)` /
`get_final_llm_resp()` / `done()` / `aclose()`。

> **选择建议**：只跑一次对话选 Tier 3；需要自定义 runner 循环但复用 skills/plugins 选
> Tier 2；需要完全掌控（自定义工具、无扩展）选 Tier 1。

---

## 5. Provider：接入 LLM

### 开箱即用：OpenAI 兼容系

`ProviderOpenAIOfficial` 覆盖了所有 OpenAI 兼容端点（OpenAI / DeepSeek / 智谱 /
Groq / xAI / OpenRouter / Minimax / 小米 等），只需换 `api_base` + `model`：

```python
from examples.example_provider import make_openai_compat_provider

provider = make_openai_compat_provider(
    api_key="sk-...",
    model="gpt-4o-mini",
    api_base="https://api.openai.com/v1",   # 换成你的厂商端点
    provider_id="openai",                    # 可选标识
    # **extra: timeout, custom_headers, api_version (Azure), ...
)
```

### 自定义 Provider

继承 `Provider`，实现以下抽象方法：

```python
from agent_runtime.provider.provider import Provider
from agent_runtime.provider.entities import LLMResponse

class MyProvider(Provider):
    def __init__(self):
        super().__init__(
            provider_config={"id": "my", "type": "my_type", "max_context_tokens": 128000, "modalities": []},
            provider_settings={},
        )

    def get_current_key(self) -> str:
        return "my-key"

    def set_key(self, key: str) -> None: ...

    async def get_models(self) -> list[str]:
        return ["my-model-v1"]

    async def text_chat(self, *args, **kwargs) -> LLMResponse:
        # 实现与 LLM 的通信逻辑
        ...

    async def text_chat_stream(self, *args, **kwargs):
        # 可选：流式实现
        yield await self.text_chat(*args, **kwargs)
```

`provider_config` 字典必选字段：`id`、`type`。常用可选：`key`（API key 列表）、
`api_base`、`model`、`max_context_tokens`、`modalities`。

### LLMResponse 结构

```python
@dataclass
class LLMResponse:
    role: str                              # "assistant" | "tool" | "err"
    completion_text: str                   # 纯文本完成结果
    result_chain: MessageChain | None      # 消息链（富内容）
    tools_call_name: list[str]             # 工具调用名称列表
    tools_call_args: list[dict]            # 工具调用参数列表
    tools_call_ids: list[str]              # 工具调用 ID 列表
    reasoning_content: str | None          # 推理内容（如 o1/deepseek-r1）
    raw_completion: ... | None             # 原始响应（OpenAI/Anthropic/Gemini）
    is_chunk: bool                         # 流式 chunk 标记
    usage: TokenUsage | None               # token 用量
```

---

## 6. Tools：定义工具

工具是 Agent 在运行时可调用的能力单元。支持两种定义方式：

### 方式 A：子类 `FunctionTool`（重写 `call`）

```python
from agent_runtime.core.tool import FunctionTool

class WeatherTool(FunctionTool):
    def __init__(self):
        super().__init__(
            name="get_weather",
            description="Get current weather for a city.",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        )

    async def call(self, context, **kwargs) -> str:
        city = kwargs["city"]
        # 实际调 API ...
        return f"Sunny, 25°C in {city}"
```

### 方式 B：函数式（传入 `handler`）

```python
from agent_runtime.core.tool import FunctionTool

async def _ping(run_context, **kwargs) -> str:
    return "pong"

ping_tool = FunctionTool(
    name="ping",
    description="Reply pong.",
    parameters={"type": "object", "properties": {}},
    handler=_ping,
)
```

> `handler` 与 `call` 是互斥的两条路径；有 `handler` 时 executor 走 `handler(run_context, **params)`，否则走 `tool.call(run_context, **params)`。

### ToolSet：工具集合

```python
from agent_runtime.core.tool import ToolSet

tools = ToolSet()
tools.add_tool(WeatherTool())
tools.add_tool(ping_tool)
tools.merge(another_tool_set)     # 合并另一个 ToolSet

tools.names()       # -> ["get_weather", "ping"]
tools.empty()       # -> False
tools.get_tool("ping")  # -> FunctionTool | None
```

`ToolSet` 序列化为各家 LLM 格式（`tools.get_func_desc_openai_style()` /
`tools.get_func_desc_anthropic_style()` / `tools.get_func_desc_google_genai_style()`），
由 provider 内部自动调用。

### MCP 工具

通过 `FunctionToolManager` 连接外部 MCP server，自动产出 `MCPTool`（`FunctionTool` 子类）：

```python
from agent_runtime.tools.func_tool_manager import FunctionToolManager

manager = FunctionToolManager()
summary = await manager.init_mcp_clients(raise_on_all_failed=True)
tools = manager.get_full_tool_set()  # ToolSet，含所有 MCP 暴露的工具

# 使用完毕后关闭连接
await manager.disable_mcp_server()
```

MCP 配置放在 `data/mcp_server.json`：

```json
{
  "mcpServers": {
    "my-server": {
      "command": "python",
      "args": ["/path/to/server.py"]
    }
  }
}
```

---

## 7. Hooks：生命周期钩子

Hooks 让你在 Agent 运行的关键节点注入逻辑，**不修改 runner 代码**。

### BaseAgentRunHooks

```python
from agent_runtime.core.hooks import BaseAgentRunHooks

class BaseAgentRunHooks(Generic[TContext]):
    async def on_agent_begin(self, run_context) -> None: ...
    async def on_llm_request(self, run_context) -> None: ...
    async def on_tool_start(self, run_context, tool, tool_args) -> None: ...
    async def on_tool_end(self, run_context, tool, tool_args, tool_result) -> None: ...
    async def on_before_complete(self, run_context, llm_response) -> bool: ...
    async def on_agent_done(self, run_context, llm_response) -> None: ...
```

| 事件 | 触发时机 | 常见用法 |
|------|----------|----------|
| `on_agent_begin` | runner reset 后、首次 LLM 调用前 | 初始化会话级状态 |
| `on_llm_request` | 每次 LLM 调用前 | 注入系统消息（skills 清单、plan 状态）|
| `on_tool_start` | 工具执行前 | 日志、权限拦截 |
| `on_tool_end` | 工具执行后 | 日志、结果后处理 |
| `on_before_complete` | LLM 无工具调用、即将结束时 | **返回 `False` 否决完成**（防止提前退出）|
| `on_agent_done` | 正式结束后 | 清理、持久化 |

### 自定义 Hook 示例

```python
class LoggingHook(BaseAgentRunHooks):
    async def on_llm_request(self, run_context):
        print(f"[LOG] LLM request, {len(run_context.messages)} messages in context")

    async def on_before_complete(self, run_context, llm_response) -> bool:
        # 返回 True 允许完成；返回 False 拒绝（runner 会再跑一轮）
        return True
```

### ChainedAgentRunHooks：组合多个 Hook

runner 只接受一个 `agent_hooks`，用 `ChainedAgentRunHooks` 把多个串起来：

```python
from agent_runtime.core.hooks_chain import ChainedAgentRunHooks

hooks = ChainedAgentRunHooks(
    SkillsPromptHook(skill_manager),
    LoggingHook(),
    PlanningHook(store),
)
```

- 事件按构造顺序依次分发
- 单个 hook 异常不影响后续 hook（try/except 隔离）
- `on_before_complete`：任一返回 `False` 即否决

---

## 8. 扩展：skills / fs / plugins / planning

所有扩展都是 **可选的**（懒导入）。通过 `build_local_agent_basics` 的开关控制：

| 参数 | 默认 | 效果 |
|------|------|------|
| `skills_root` | `data/skills/` | skill 发现目录 |
| `include_fs` | `True` | 加入只读 fs 工具（read_file 等）|
| `include_planning` | `False` | 加入 `write_todos` 工具 + `PlanningHook` |
| `contributions` | `[]` | 插件贡献列表 |
| `plugin_store` | `InMemoryPluginStore()` | 插件/计划状态的 KV 存储 |

### 8.1 Skills

Skills 是放在磁盘目录下的 Markdown 文件（`SKILL.md`），被 `SkillManager` 自动发现，
在 `on_llm_request` 时注入到 system prompt 中作为可用能力清单。

```
data/skills/
└── greet/
    └── SKILL.md    # ---\nname: greet\ndescription: Greet the user.\n---\n...
```

LLM 通过调用内建 `Skill` 工具加载对应 skill 的完整内容。

### 8.2 Plugins

Plugin 是一个拥有生命周期的能力单元，可以贡献 **工具** 和 **钩子**：

```python
from agent_runtime.extensions.plugins import Plugin, PluginManager, tool, on_llm_request

class MyPlugin(Plugin):
    name, author, desc, version = "my_plugin", "me", "desc", "1.0"

    async def initialize(self) -> None:
        await self.put_kv_data("key", "value")

    @tool
    async def echo(self, run_context, text: str) -> str:
        """Echo the input.

        Args:
            text(string): the text to echo
        """
        return text

    @on_llm_request
    async def inject_note(self, run_context) -> None:
        # 在每次 LLM 请求前注入自定义消息
        ...

# 注册插件
manager = PluginManager()
contribution = await manager.register(MyPlugin)

# contribution.tools → 插件贡献的工具列表
# contribution.hook_methods → {"on_llm_request": [...], ...}
```

可用的 hook 装饰器：`@on_llm_request`、`@on_agent_begin`、`@on_agent_done`、
`@on_tool_start`、`@on_tool_end`。

插件的 `@tool` 方法参数自动从 **Google-style docstring** 解析（`Args:` 块，
`name(type): description` 格式）。

### 8.3 Planning（Plan-and-Execute）

在已有的 ReAct 循环之上，通过一个工具 + 一个 hook 实现 plan-and-execute 涌现行为：

- `write_todos` 工具：LLM 调用它来创建/更新计划
- `PlanningHook`：
  - `on_llm_request`：将 plan 状态注入 system prompt
  - `on_before_complete`：当存在未完成 todo 时否决完成

```python
from agent_runtime.local_runtime import build_local_agent

agent = await build_local_agent(
    provider,
    prompt="research and summarize the topic",
    include_planning=True,          # 开启 planning 扩展
    plugin_store=InMemoryPluginStore(),  # 或注入持久化实现
)
final = await agent.run(max_step=30)
```

Plan 存储在 `PluginStore`（KV 接口），以 `session_id` 为键，独立于消息流：

```python
from agent_runtime.extensions.planning import load_plan

todos = await load_plan(store, session_id="my-session")
# -> [Todo(content="...", status=TodoStatus.COMPLETED), ...]
```

---

## 9. Runner：控制流范式

Runner 负责驱动 "LLM 调用 → 工具执行 → 再调用" 的循环。通过 `runner_type` 选择范式：

### 9.1 `"react"`（默认）

`ToolLoopAgentRunner` — 经典 ReAct 循环：

```
LLM → 有工具调用? → 执行工具 → 把结果追加到 context → 下一轮 LLM
                 → 无工具调用? → on_before_complete → 完成/被否决再循环
```

```python
from agent_runtime.core.runners.tool_loop_agent_runner import ToolLoopAgentRunner

runner = ToolLoopAgentRunner()
await runner.reset(
    provider=provider,
    request=request,
    run_context=run_context,
    tool_executor=FunctionToolExecutor(provider),
    agent_hooks=hooks,
    streaming=False,
    enforce_max_turns=-1,    # -1 表示不限制；>0 表示最大轮次
)

# 逐步执行
async for response in runner.step_until_done(max_step=20):
    # response: AgentResponse — 每步的中间结果
    pass

# 获取最终结果
final = runner.get_final_llm_resp()
```

### 9.2 `"plan_execute"`

`PlanExecuteRunner` — 显式 PLAN → EXEC → REPLAN 状态机：

- **PLAN 阶段**：LLM 生成分步计划
- **EXEC 阶段**：每一步委派给一个独立的子 ReAct runner 执行
- **REPLAN 阶段**：根据执行结果决定是否调整计划

```python
agent = await build_local_agent(
    provider,
    prompt="...",
    runner_type="plan_execute",     # 切换到 plan_execute 范式
    include_planning=True,          # 可选：子 runner 也带 planning 扩展
)
final = await agent.run(max_step=50)
```

> `runner_type` 与 `include_planning` 正交：
> - `react` + `include_planning=True`：ReAct 循环中涌现 plan-and-execute 行为
> - `plan_execute`：显式状态机，每个 EXEC step 独立跑子 ReAct
> - `plan_execute` + `include_planning=True`：子 runner 也具备 per-step 涌现式 replanning

### Runner 公共接口

所有 runner 继承 `BaseAgentRunner`，暴露统一接口：

```python
class BaseAgentRunner:
    async def reset(self, provider, request, run_context, tool_executor, agent_hooks, **kwargs) -> None
    def step(self) -> AsyncGenerator[AgentResponse, None]
    def step_until_done(self, max_step: int) -> AsyncGenerator[AgentResponse, None]
    def done(self) -> bool
    def get_final_llm_resp(self) -> LLMResponse | None
```

---

## 10. API 参考

### 核心类型

| 类 | 路径 | 说明 |
|----|------|------|
| `Provider` | `agent_runtime.provider.provider` | LLM Chat 提供者基类 |
| `ProviderOpenAIOfficial` | `agent_runtime.provider.sources.openai_source` | OpenAI 兼容实现 |
| `ProviderRequest` | `agent_runtime.provider.entities` | 请求描述（prompt, tools, contexts...）|
| `LLMResponse` | `agent_runtime.provider.entities` | LLM 响应（completion, tool_calls...）|
| `TokenUsage` | `agent_runtime.provider.entities` | Token 用量统计 |
| `FunctionTool` | `agent_runtime.core.tool` | 工具定义（name + schema + handler/call）|
| `ToolSet` | `agent_runtime.core.tool` | 工具集合容器 |
| `FunctionToolExecutor` | `agent_runtime.core.function_tool_executor` | 默认工具执行器 |
| `BaseAgentRunHooks` | `agent_runtime.core.hooks` | 生命周期钩子基类 |
| `ChainedAgentRunHooks` | `agent_runtime.core.hooks_chain` | 多 hook 链式组合 |
| `ContextWrapper` | `agent_runtime.core.run_context` | 运行上下文容器 |
| `SessionContext` | `agent_runtime.core.session_context` | 默认 TContext 实现 |
| `ToolLoopAgentRunner` | `agent_runtime.core.runners.tool_loop_agent_runner` | ReAct runner |
| `PlanExecuteRunner` | `agent_runtime.core.runners.plan_execute_runner` | Plan-Execute runner |

### 装配函数

| 函数 | 路径 | 说明 |
|------|------|------|
| `build_local_agent_basics` | `agent_runtime.local_runtime` | Tier 2：构建 tools/hooks bundle |
| `build_local_agent` | `agent_runtime.local_runtime` | Tier 3：一站式构建可运行 Agent |
| `make_openai_compat_provider` | `examples.example_provider` | 快捷构建 OpenAI 兼容 Provider |

### 扩展模块

| 模块 | 路径 | 说明 |
|------|------|------|
| Skills | `agent_runtime.extensions.skills` | `SkillManager` / `SkillsPromptHook` / `build_skill_tool` |
| FS | `agent_runtime.extensions.fs` | `build_fs_tools(allowed_roots=[...])` |
| Plugins | `agent_runtime.extensions.plugins` | `Plugin` / `PluginManager` / `@tool` / `@on_*` |
| Planning | `agent_runtime.extensions.planning` | `build_planning_extension` / `PlanningHook` / `load_plan` |
| MCP | `agent_runtime.tools.func_tool_manager` | `FunctionToolManager` — MCP 客户端管理 |

### ProviderRequest 字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `prompt` | `str \| None` | 用户提示词 |
| `system_prompt` | `str` | 系统提示词 |
| `func_tool` | `ToolSet \| None` | 可用工具集 |
| `session_id` | `str` | 会话 ID |
| `image_urls` | `list[str]` | 图片 URL 列表（多模态）|
| `audio_urls` | `list[str]` | 音频 URL/路径列表 |
| `contexts` | `list[dict]` | OpenAI 格式上下文消息列表 |
| `model` | `str \| None` | 模型覆盖（None 使用 Provider 默认）|
| `tool_calls_result` | `ToolCallsResult \| list \| None` | 上次工具调用结果回传 |
| `extra_user_content_parts` | `list[ContentPart]` | 额外用户消息内容块 |

---

## 11. 常见问题

### Q: 如何切换到 DeepSeek / 智谱 / 其他 OpenAI 兼容厂商？

只需更换 `api_base` 和 `model`：

```python
provider = make_openai_compat_provider(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    model="deepseek-chat",
    api_base="https://api.deepseek.com/v1",
    provider_id="deepseek",
)
```

### Q: 如何使用流式输出？

```python
agent = await build_local_agent(provider, prompt="...", streaming=True)

async for response in agent.step_until_done(max_step=20):
    # response 在 streaming=True 时包含 chunk 级别的中间输出
    if response.llm_response and response.llm_response.is_chunk:
        print(response.llm_response.completion_text, end="", flush=True)
```

### Q: 如何限制最大运行轮次？

```python
# Tier 3
agent = await build_local_agent(provider, prompt="...", max_turns=10)

# Tier 1
await runner.reset(..., enforce_max_turns=10)
```

`max_turns=-1` 表示不限制（默认）。

### Q: 如何添加自定义上下文压缩？

runner 的 `**runner_kwargs` 支持传入高级参数：

```python
agent = await build_local_agent(
    provider,
    prompt="...",
    truncate_turns=True,              # 开启截断
    llm_compress_threshold=80000,     # 超过多少 token 触发压缩
    # custom_compressor=MyCompressor(),  # 自定义压缩器
)
```

### Q: 如何同时使用 Plugin 和 Planning？

```python
from agent_runtime.extensions.plugins import PluginManager
from agent_runtime.extensions.plugins.store import InMemoryPluginStore
from agent_runtime.local_runtime import build_local_agent

manager = PluginManager()
contribution = await manager.register(MyPlugin)

agent = await build_local_agent(
    provider,
    prompt="...",
    contributions=[contribution],
    include_planning=True,
    plugin_store=InMemoryPluginStore(),  # 共享一个 store
)
```

### Q: `run_context.messages` 是什么？

它是 runner 实际发给 provider 的消息列表（`list[Message]`）。Hooks 通过修改它来
影响 LLM 行为：

- 在 `on_llm_request` 中 **insert** system message → 注入指令
- 在 `on_before_complete` 中 **append** user message → 强制 LLM 继续

### Q: 如何实现持久化（跨进程恢复）？

实现 `PluginStore` 协议（`get` / `put` / `delete`），替换默认的 `InMemoryPluginStore`：

```python
class RedisPluginStore:
    async def get(self, key: str) -> Any | None: ...
    async def put(self, key: str, value: Any) -> None: ...
    async def delete(self, key: str) -> None: ...

agent = await build_local_agent(
    provider,
    prompt="...",
    include_planning=True,
    plugin_store=RedisPluginStore(),
)
```

### Q: 工具返回值应该是什么类型？

`ToolExecResult = str | mcp.types.CallToolResult`

- 返回 `str`：executor 自动包装成 `CallToolResult`
- 返回 `CallToolResult`：直接使用（适合需要 `isError` 标记的场景）

---

## 附录：示例索引

| 示例 | 文件 | 演示 |
|------|------|------|
| 最小 driver | `examples/driver.py` | 四接缝 + 一个 add 工具 |
| Provider 构建 | `examples/example_provider.py` | `make_openai_compat_provider` |
| 一键装配 vs 手工 | `examples/local_runtime_demo.py` | Tier 3 vs Tier 1 等价验证 |
| MCP 工具 | `examples/mcp_demo.py` | stdio MCP server → MCPTool |
| Plugin 机制 | `examples/plugin_demo.py` | 生命周期 + KV + @tool + @on_llm_request |
| Planning 闭环 | `examples/planning_demo.py` | write_todos → injection → veto |
| Skills 发现 | `examples/skills_demo.py` | SkillManager + skill 加载 |
| FS 工具 | `examples/fs_demo.py` | build_fs_tools + 沙箱隔离 |
| Plan-Execute Runner | `examples/plan_execute_demo.py` | 显式状态机 runner |

运行示例：

```bash
uv run --env-file .env python -m examples.driver
uv run python -m examples.local_runtime_demo
uv run python -m examples.planning_demo
```

---

## 附录：依赖关系约束

代码分层由 `tests/test_audits.py` 强制保障：

```
foundation ← core ← provider/tools ← extensions
```

- `extensions/` 之间不可互相导入
- `core/` 不可导入 `extensions/`
- 任何重构都必须保持审计测试绿色
