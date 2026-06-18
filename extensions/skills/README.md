# extensions/skills — 路线图占位

Skills 加载子系统的预留落点。

## 现状

未实现。本目录是结构占位，确保后续接入 skills 时落点已经定好、不再二次搬迁。

## 设计意图（接入时遵循）

- **依赖方向**：skills 属于 `extensions/` 层，只能向内依赖 `core/` `foundation/`，
  不得被 `core/` 反向依赖。
- **接缝**：skill 最终暴露为工具，挂到 `core/tool.py` 的 `ToolSet`；加载器负责发现
  + 注册，不侵入 `ToolLoopAgentRunner`。
- **import 契约**：新增依赖须登记进 `tests/test_audits.py` 的 `ALLOWED_THIRD_PARTY`，
  否则审计测试会红。
