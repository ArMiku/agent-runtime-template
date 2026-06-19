# extensions/plugins — 统一插件机制

一套**框架无关、无 event** 的插件机制：把记忆、检索等扩展能力以统一方式挂进 agent 循环。
插件通过干净的装饰器贡献两类扩展——**工具**（`FunctionTool`）与 **agent 循环钩子**——
运行时核心不认识任何具体能力种类。

## 快速上手

```python
from agent_runtime.extensions.plugins import (
    Plugin, PluginManager, CompositeAgentRunHooks, tool, on_llm_request,
)
from agent_runtime.core.message import Message


class MyPlugin(Plugin):
    name, author, desc, version = "my-plugin", "me", "示例", "0.1.0"

    async def initialize(self) -> None:
        await self.put_kv_data("count", 0)        # 私有 KV 持久化（PluginStore 接缝）

    @tool                                          # 贡献一个工具（docstring 解析参数）
    async def echo(self, run_context, text: str) -> str:
        """回显输入。

        Args:
            text(string): 要回显的文本
        """
        return text

    @on_llm_request                                # 每步 LLM 调用前触发，改 run_context.messages
    async def inject(self, run_context) -> None:
        run_context.messages.insert(0, Message(role="system", content="[my-plugin] 注入"))


manager = PluginManager()
contribution = await manager.register(MyPlugin)    # 代码注入注册 → initialize → 收集贡献
hooks = CompositeAgentRunHooks(manager.contributions)   # 聚合成单个 BaseAgentRunHooks
# contribution.tools 喂给 ProviderRequest.func_tool；hooks 传给 runner.reset(agent_hooks=...)
```

完整端到端示例见 `agent_runtime/examples/plugin_demo.py`。

## 组成

| 模块 | 作用 |
|------|------|
| `base.Plugin` | 插件基类。`__init_subclass__` 自动登记、`initialize/terminate` 生命周期、`put/get/delete_kv_data` 便捷方法 |
| `metadata.PluginMetadata` | 中性元数据（必填 `name/author/desc/version`），无 `support_platforms` 等平台字段 |
| `context.PluginContext` | 注入给插件的轻量上下文：`context_store`（跨 run 历史）+ `plugin_store`（私有 KV）。**不含** event/platform/DB 句柄 |
| `store.PluginStore` | 可注入的 KV 持久化 Protocol（按 `plugin_id` 隔离）；默认 `InMemoryPluginStore`，无 DB |
| `registry.PluginRegistry` | 进程级注册表 |
| `decorators` | `@tool` + `@on_llm_request`/`@on_agent_begin`/`@on_agent_done`/`@on_tool_start`/`@on_tool_end`（全部对 `run_context`） |
| `contributions` | `PluginContribution`（tools + hook_methods）；扫描插件实例产出 |
| `hooks.CompositeAgentRunHooks` | 按加载顺序聚合多插件钩子为单个 `BaseAgentRunHooks`，逐插件异常隔离 |
| `manager.PluginManager` | 代码注入注册、生命周期编排、元数据校验、可选目录加载 |
| `session.is_plugin_enabled_for_session` | 会话级启停（中性逻辑，启停配置经 `PluginStore`） |

## `on_llm_request` 钩子

`BaseAgentRunHooks` 新增的挂点，在 `ToolLoopAgentRunner.step()` 的 compact 之前、payload
从 `run_context.messages` 重新构建之前触发。钩子通过修改 `run_context.messages`（runner
唯一消费的真相）影响下一次 LLM 调用。**只收 `run_context`**，无 event、无 req。

## 注意事项

- **`on_llm_request` 每步触发**：ReAct 多轮工具调用时每轮都会重新触发，插件如需可自行去重/限频。
- **默认 `InMemoryPluginStore` 不持久**：进程重启即丢。需落盘的能力由使用方注入持久化 `PluginStore`。
- **注册表是进程级单例**：多运行时实例共享。

## 约束

- **依赖方向**：plugins 属于 `extensions/` 层，只能向内依赖 `core/`/`provider/`/`message/`，
  不得被核心反向依赖。
- **命名约束**：本目录与公开符号不含 `event`/`MessageEventResult` 字样；所有钩子对 `run_context`，不对 event。
- **import 契约**：新增第三方依赖须登记进 `tests/test_audits.py` 的 `ALLOWED_THIRD_PARTY`。
