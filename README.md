# agent_runtime

一个**框架无关的 Agent 运行时 + LLM Provider 模板**。它提供一套经过验证的"支持推理的
ReAct 工具循环"，
以及一层覆盖大半生态的 Provider 适配器——而不必拖上整个聊天机器人应用外壳（没有平台
适配器、没有插件 / Star 体系、没有消息流水线）。

约 50 行用户代码即可驱动：一个 `Provider` + 工具集 + system prompt + 一个
`ToolLoopAgentRunner` → 最终 LLM 响应。

## 目录结构

按领域分包、按依赖方向分层：每个接缝抽象与它的开箱默认实现就近同包，外层只能向内依赖，内层不知道外层的存在。

```
agent_runtime/
├── foundation/        # 最内层：宿主中立的基础设施，不含任何领域知识
│   └── log / paths / network / io_utils / string_utils / exceptions / config
├── core/              # Agent 引擎：runner 工具循环、工具、hooks、上下文管理
│   │                  #   接缝抽象 + 各自的默认实现就近同包（按领域而非角色分组）
│   ├── tool_executor.py / function_tool_executor.py   # 执行器接缝 + 默认实现（带 handoff）
│   ├── run_context.py / session_context.py            # TContext 接缝 + 默认实现
│   ├── runners/       # BaseAgentRunner 子类（plan-and-execute 规划器落点）
│   └── context/       # 上下文压缩 / 截断 / token 计数 + ContextStore 持久化接缝及默认实现
├── provider/          # LLM Provider 抽象 + 内置 sources（openai / anthropic）
├── message/           # 中性消息模型（MessageChain 等）
├── media/             # 媒体解析接缝（MediaResolver）
├── extensions/        # 最外层：可插拔扩展子系统，只向内依赖
│   ├── skills/        # skills 加载（路线图占位，见其 README）
│   └── plugins/       # 插件 / Star 体系（路线图占位，见其 README）
├── examples/          # 可运行示例（driver.py + example_provider.py 演示包装）
└── tests/             # 含 test_audits.py——校验上述 import 契约
```

`test_audits.py` 把这套依赖方向固化成可执行断言：任何反向依赖或越层 import 都会让审计测试变红。

## 安装

运行时面向 **Python 3.12+**。依赖：

```
pydantic  jsonschema  tenacity  mcp  docstring_parser
aiohttp   deprecated  typing_extensions  pillow
openai    anthropic
```

`openai` / `anthropic` 仅在使用内置 provider sources（`provider/sources/`）时需要；
`pillow` 是默认媒体解析器做图片转码用的。

## 快速上手（约 50 行的 driver）

```python
import asyncio, os

from agent_runtime.core.run_context import ContextWrapper
from agent_runtime.core.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from agent_runtime.core.tool import FunctionTool, ToolSet
from agent_runtime.core.hooks import BaseAgentRunHooks
from agent_runtime.core import FunctionToolExecutor, SessionContext
from agent_runtime.examples.example_provider import make_openai_compat_provider
from agent_runtime.provider.entities import ProviderRequest


class AddTool(FunctionTool):
    """干净的子类式工具：首参是 ContextWrapper，返回普通 str。"""

    def __init__(self) -> None:
        super().__init__(
            name="add",
            description="把两个整数相加并返回结果。",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            },
        )

    async def call(self, context, **kwargs) -> str:
        return str(int(kwargs["a"]) + int(kwargs["b"]))


async def run(prompt: str) -> str:
    provider = make_openai_compat_provider(
        api_key=os.environ["OPENAI_API_KEY"],
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        api_base=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
    )
    executor = FunctionToolExecutor(provider)

    tools = ToolSet()
    tools.add_tool(AddTool())

    request = ProviderRequest(
        prompt=prompt,
        system_prompt="你是一个乐于助人的助手，需要时使用工具。",
        func_tool=tools,
    )
    run_context = ContextWrapper(context=SessionContext(session_id="demo"))

    runner = ToolLoopAgentRunner()
    await runner.reset(
        provider=provider, request=request, run_context=run_context,
        tool_executor=executor, agent_hooks=BaseAgentRunHooks(),
    )
    async for _ in runner.step_until_done(max_step=10):
        pass

    final = runner.get_final_llm_resp()
    return (final.completion_text if final else "") or ""


if __name__ == "__main__":
    print(asyncio.run(run("7 + 5 等于几？用 add 工具算。")))
```

可直接运行的副本在 [`driver.py`](driver.py)：

```bash
OPENAI_API_KEY=sk-... python -m agent_runtime.examples.driver
```

## 四个对外契约（接缝）

你需要定制的一切都接入这四个显式接缝之一。它们都不携带 `event` / `platform` / `star`
对象——这正是本次抽离的核心目标。

| 接缝 | 位置 | 默认实现 | 职责 |
|------|------|----------|------|
| `Provider` | `provider/provider.py` | `ProviderOpenAIOfficial`（`provider/sources/`）；演示包装见 `examples/example_provider.py` | LLM 对话：流式 + 非流式，推理内容透传 |
| `BaseFunctionToolExecutor` | `core/tool_executor.py` | `FunctionToolExecutor`（`core/function_tool_executor.py`，就近同包） | 执行一次工具调用——`handler`/`call` 二分派 + 子代理委派 |
| `BaseAgentRunHooks` | `core/hooks.py` | `BaseAgentRunHooks()` 自身（空实现） | 生命周期回调；在非流式路径捕获推理 token |
| `TContext` | `core/run_context.py` | `SessionContext`（`core/session_context.py`，就近同包） | 你自己的会话 / 请求对象，经 `ContextWrapper[TContext]` 携带 |

另有两个注入点：

- **`ContextStore`**（`core/context/context_store.py`）——持久化接缝。`Protocol`，含
  `load` / `save`；`InMemoryContextStore` 是零依赖的默认实现。
- **`MediaResolver`**（`media/resolver.py`）——媒体解析接缝。含 `resolve_to_data_url` /
  `describe` / `is_file_uri`；`DefaultMediaResolver` 是一套完整的媒体处理实现
  （下载 / base64 / MIME / 图片转码），其中 QQ silk 语音被剥离为可选的音频编解码钩子。

## 定义工具——两条干净路径

两条路径首参都是 `ContextWrapper[TContext]`，返回 `ToolExecResult`。没有 `event`，
也没有 `MessageEventResult`。

**子类化 `FunctionTool` 并覆写 `call`**（见上例）——适合有状态或类形态的工具。

**传入 `handler` 函数**——适合快速的函数式工具：

```python
async def add(context, *, a: int, b: int) -> str:
    return str(a + b)

tool = FunctionTool(name="add", description="...", parameters={...}, handler=add)
```

默认的 `FunctionToolExecutor` 按工具类型分派：`HandoffTool`（子代理委派）和 `MCPTool`
走 `tool.call()`；函数式工具走 `tool.handler(run_context, **params)`；其余子类工具
回落到 `tool.call()`。

## 自定义 Provider

大多数 provider **无需新适配器**——只要用不同的 `api_base` + model 子类化 OpenAI
兼容基类即可（见覆盖表）。如需对接私有协议，子类化 `Provider`（`provider/provider.py`）
并实现对话方法。

## 自定义持久化

```python
from agent_runtime.core.context import ContextStore

class RedisContextStore:           # 结构化协议——满足 ContextStore Protocol 即可
    async def load(self, session_id: str) -> list | None: ...
    async def save(self, session_id: str, messages: list) -> None: ...
```

在运行前 load 历史、运行后持久化的位置注入你的 store。

## Provider 覆盖范围

两个基类即可覆盖大半生态（design.md §11）：

| 基类 | 文件 | 覆盖 |
|------|------|------|
| `ProviderOpenAIOfficial` | `provider/sources/openai_source.py` | OpenAI、**DeepSeek**（纯 OpenAI 兼容端点，无需适配器）、zhipu、Groq、xAI/Grok、OpenRouter、AIHubMix、LongCat、小米 |
| `ProviderAnthropic` | `provider/sources/anthropic_source.py` | Anthropic、Kimi、小米 token-plan、MiniMax token-plan |
| `ProviderGoogleGenAI`（Gemini） | — | **暂缓**——独立的单文件适配，自带 SDK + Google schema，以后加入 friction 很低 |

要用 zhipu / DeepSeek / Groq / xAI 等，只需用不同的 `api_base` + model 子类化
`ProviderOpenAIOfficial`，无需新适配器代码。

## 已知限制

- **MiniMax 协议歧义**（design.md §9）：MiniMax 同时提供 OpenAI 兼容端点和 token-plan
  端点。按端点选基类——OpenAI 端点 → `ProviderOpenAIOfficial`；token-plan →
  `ProviderAnthropic`。
- **权限三态（allow / deny / ask）不在本模板范围。** 模板不带权限管控；下游项目自行在
  `BaseFunctionToolExecutor.execute` 接缝或工具 wrapper 上实现（典型做法是一个挂起 /
  恢复的工单机制）。
- **本地 plan-and-execute 不在本模板范围。** 模板不内置规划器，无可迁移代码。请在
  `BaseAgentRunner` 接缝上从零实现 planner runner——该接缝已支持新增 runner 而不改动
  `ToolLoopAgentRunner`。
- **`MessageChain` 漂移**：`MessageChain` 是一个略偏具体的输出模型切片。当前可接受
  （中性容器）；若它长出平台耦合再重新评估。

## 测试

```bash
python -m pytest agent_runtime/tests -q
```

测试覆盖流式 / 非流式 provider 路径、工具循环、`MessageChain` 输出、`ContextStore`
注入，以及 import / 契约审计（所有 import 仅限标准库 + 声明依赖 + 本包、`handler`
类型为 `ToolExecResult`、参数过滤以 `tool.parameters` 为判别器）。
