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
agent_runtime/         # 唯一可导入的包（flat layout）
├── foundation/        # 最内层：宿主中立的基础设施，不含任何领域知识
│   └── log / paths / network / io_utils / string_utils / exceptions / config
├── core/              # Agent 引擎：runner 工具循环、工具原语、hooks、上下文管理
│   │                  #   接缝抽象 + 各自的默认实现就近同包（按领域而非角色分组）
│   ├── tool.py / mcp_client.py                         # 工具原语（FunctionTool/ToolSet/MCPTool/MCPClient）
│   ├── tool_executor.py / function_tool_executor.py   # 执行器接缝 + 默认实现（带 handoff）
│   ├── run_context.py / session_context.py            # TContext 接缝 + 默认实现
│   ├── runners/       # BaseAgentRunner 子类（plan-and-execute 规划器落点）
│   └── context/       # 上下文压缩 / 截断 / token 计数 + ContextStore 持久化接缝及默认实现
├── provider/          # LLM Provider 抽象 + 内置 sources（openai / anthropic）
├── tools/             # 工具管理层：注册表 + MCP 服务生命周期（FunctionToolManager）
│                      #   仅向内依赖 core 的工具原语（FunctionTool/ToolSet/MCPClient）
├── message/           # 中性消息模型（MessageChain 等）
├── media/             # 媒体解析接缝（MediaResolver）
├── extensions/        # 最外层：可插拔扩展子系统，只向内依赖
│   ├── skills/        # skills 渐进式加载（发现 + 按需载入 SKILL.md，见其 README）
│   ├── fs/            # 只读路径寻址文件系统工具集（fenced 到 skills/plugin 根，见其 README）
│   ├── plugins/       # 插件贡献 tools / skills / hooks（见其 README）
│   └── planning/      # plan-and-execute：write_todos 工具 + PlanningHook（注入 plan + 防早退）
└── local_runtime.py   # 组合根：把 skills + fs + plugins + planning 装配成可运行的 LocalAgent
examples/              # 包外：可运行示例（driver.py / *_demo.py + example_provider.py）
tests/                 # 包外：test_audits.py 校验 import 契约；fakes.py 共享测试替身
```

`test_audits.py` 把这套依赖方向固化成可执行断言：任何反向依赖或越层 import 都会让审计测试变红。

## 安装

本项目用 [uv](https://docs.astral.sh/uv/) 管理依赖，面向 **Python 3.12+**：

```bash
uv sync --extra dev   # 创建 .venv（自动选 Python ≥3.12）+ 装全部依赖 + dev 工具（pytest/ruff）
```

之后所有命令都走 `uv run`（不要用系统 / anaconda 的 Python，避免依赖漂移）。完整依赖见
`pyproject.toml`，其中几项说明：

- `openai` / `anthropic`：仅在使用内置 provider sources（`provider/sources/`）时需要
- `pillow`：默认媒体解析器做图片转码
- `pyyaml`：skills 子系统解析 SKILL.md frontmatter
- `socksio`：若环境设了 SOCKS 代理（`all_proxy=socks5://...`）访问 provider 时需要

## 快速上手（约 50 行的 driver）

```python
import asyncio, os

from agent_runtime.core.run_context import ContextWrapper
from agent_runtime.core.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from agent_runtime.core.tool import FunctionTool, ToolSet
from agent_runtime.core.hooks import BaseAgentRunHooks
from agent_runtime.core import FunctionToolExecutor, SessionContext
from examples.example_provider import make_openai_compat_provider
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

可直接运行的副本在 `examples/driver.py`。把 `OPENAI_API_KEY`（可选 `OPENAI_MODEL` /
`OPENAI_API_BASE`）写进项目根的 `.env`（已被 `.gitignore` 忽略），然后：

```bash
uv run --env-file .env python -m examples.driver
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

## plan-and-execute（planning 扩展）

与 `skills/`、`fs/` 平级的能力扩展，不改 ReAct 控制流内核，让现有循环"涌现"出规划行为：

```python
agent = await build_local_agent(
    provider,
    prompt="research and summarize X",
    include_planning=True,          # 装上 write_todos 工具 + PlanningHook
    plugin_store=my_persistent_store,  # 可选：注入持久化 PluginStore 以跨进程恢复
)
```

- **write_todos 工具**：LLM 写入/全量覆盖 plan 的唯一入口。每次传完整清单，新清单整体替换旧清单。
- **plan 经 PluginStore 持久化**：plan 是按 `session_id` 隔离的独立状态（非消息流），经已有的
  `PluginStore` KV seam 存取。因此上下文压缩删除历史消息时 plan 不丢；host 注入持久实现即可跨进程恢复
  （默认 `InMemoryPluginStore` 仅进程内）。
- **PlanningHook**：复用 `on_llm_request` 每步把当前 plan 注入首条 system message（独立 sentinel
  `<!-- todo-state -->`，与 skills 注入互不干扰）；并实现内核新增的 `on_before_complete` 事件做"防早退"——
  存在未完成 todo 时否决完成、追加提醒、让 `step_until_done` 续跑一轮，带 `max_reminders` 上限防死循环。
- **人工改 plan**：暂停态下 host 经 `basics.planning_hook.read_plan/write_plan` 读改 plan，与 LLM 的
  `write_todos` 落到同一份独立状态，恢复逻辑只有一套。

**两条路线正交**：planning 是"喂给 ReAct 的料"（路线 2，本扩展实现），不是新控制流。若未来要"显式编排"
（经典 planner→executor 状态机，路线 1），应作为 `core/runners/` 下的新 `BaseAgentRunner` 子类，由 `runner_type`
开关选择——planning 扩展仅依赖 `BaseAgentRunHooks` / `ToolSet` / `PluginStore` 抽象，届时可不改自身直接复用。
完整设计见 `openspec/changes/add-planning-extension/design.md`；可运行示例见 `examples/planning_demo.py`。

`on_before_complete` 是本扩展引入的唯一内核改动：`BaseAgentRunHooks` 新增一个默认放行（返回 `True`）的
"完成前否决"事件，与 `on_agent_done`（完成后通知）对称，所有现有 hook/runner 向后兼容。

## 自定义持久化

```python
from agent_runtime.core.context import ContextStore

class RedisContextStore:           # 结构化协议——满足 ContextStore Protocol 即可
    async def load(self, session_id: str) -> list | None: ...
    async def save(self, session_id: str, messages: list) -> None: ...
```

在运行前 load 历史、运行后持久化的位置注入你的 store。

## Provider 覆盖范围

两个基类即可覆盖大半生态：

| 基类 | 文件 | 覆盖 |
|------|------|------|
| `ProviderOpenAIOfficial` | `provider/sources/openai_source.py` | OpenAI、**DeepSeek**（纯 OpenAI 兼容端点，无需适配器）、zhipu、Groq、xAI/Grok、OpenRouter、AIHubMix、LongCat、小米 |
| `ProviderAnthropic` | `provider/sources/anthropic_source.py` | Anthropic、Kimi、小米 token-plan、MiniMax token-plan |
| `ProviderGoogleGenAI`（Gemini） | — | **暂缓**——独立的单文件适配，自带 SDK + Google schema，以后加入 friction 很低 |

要用 zhipu / DeepSeek / Groq / xAI 等，只需用不同的 `api_base` + model 子类化
`ProviderOpenAIOfficial`，无需新适配器代码。

## 已知限制

- **MiniMax 协议歧义**：MiniMax 同时提供 OpenAI 兼容端点和 token-plan
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
uv run pytest -q       # testpaths = ["tests"]；含 test_audits.py 契约审计
```

测试覆盖流式 / 非流式 provider 路径、工具循环、`MessageChain` 输出、`ContextStore`
注入，以及 import / 契约审计（所有 import 仅限标准库 + 声明依赖 + 本包、`handler`
类型为 `ToolExecResult`、参数过滤以 `tool.parameters` 为判别器）。
