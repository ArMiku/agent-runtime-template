# extensions/plugins — 路线图占位

插件 / Star 体系的预留落点。

## 现状

未实现。本目录是结构占位，确保后续移植插件体系时落点已定。

## 设计意图（接入时遵循）

- **依赖方向**：plugins 属于 `extensions/` 层，只能向内依赖 `core/` `foundation/`
  及 `extensions/skills`，不得被 `core/` 反向依赖。
- **接缝**：插件通过既有的 `BaseFunctionToolExecutor` / `ToolSet` / `BaseAgentRunHooks`
  接缝注入能力，不为插件在 `core/` 开新的反向钩子。
- **无宿主耦合**：权限三态（allow / deny / ask）按包根 README「已知限制」，由下游在 executor 接缝实现。
- **plan-and-execute**：规划器作为新的 `BaseAgentRunner` 子类落在 `core/runners/`，
  与本目录解耦；插件可贡献 planner 步骤所需的工具，但不实现 planner 本身。
- **import 契约**：新增依赖须登记进 `tests/test_audits.py` 的 `ALLOWED_THIRD_PARTY`。
