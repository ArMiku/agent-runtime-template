# extensions/skills — 只读型 skills 子系统（本地执行）

管理 `SKILL.md` 指令包，以「清单 + 指令」两段式服务 agent 的渐进式披露。**本地执行、只读**：
运行时把 `skills_root` 与 `skills.json` 当作外部写入的输入来读取，自身**不安装、不删除、不
开关、不落盘**。写/管理路径是平台职责。

依赖方向：本层只能向内 import `core` / `foundation`；`core` 等核心包**绝不**反向 import 本层；
本层**绝不** import `extensions/plugins`（插件贡献的技能目录以纯路径列表在宿主处接线，见下文）。

## 两段式数据流

skills 对 agent 的服务分两段，分属不同寻址空间与不同角色消息：

| 段 | 寻址空间 | 谁放置 | 放哪 | 角色 |
|----|---------|--------|------|------|
| 清单（name + description + 目录） | **name**（注册表） | 运行时 push | 系统消息 `messages[0]` | `system` |
| 指令（SKILL.md） | **name**（注册表） | LLM pull `Skill(name)` | 对话尾部 | `tool` 结果 |
| 附属文件（scripts/references/assets） | **path**（文件系统） | LLM pull **通用 READ** | 对话尾部 | `tool` 结果 |

- **清单注入**：`SkillsPromptHook` 经 `on_llm_request`（每步、压缩之前触发）把活跃技能清单写入
  **前导系统消息**。系统消息会话内稳定 → 命中 provider 的自动前缀缓存；上下文压缩逐字保留前导
  系统消息，故清单不会被压缩丢弃。
- **幂等**：清单段以 `<!-- skills-inventory -->…<!-- /skills-inventory -->` 哨兵定界包裹，每步
  重算后**整段替换**（非追加），否则系统消息会随步数无界增长。无活跃技能时移除该段。

## 寻址空间边界（关键）

- `Skill(name)` 工具按 **name** 寻址（技能注册表）——是「发现按 name 披露」的对称面。它**只**接收
  `name` 一个参数，**不收** path/rel_path；技能指令的路径解析由 `SkillManager.load_skill` 内部
  完成。因此 LLM 无法借它读取 `SKILL.md` 之外的文件，且无路径穿越面。
- 通用文件读（READ）工具按 **path** 寻址（文件系统），读取 `scripts/`/`references/`/`assets/` 等
  附属文件。它是**独立关注点，由单独的 change 交付**。本子系统只声明该依赖、不实现 READ。
- 两者职责不重叠：`Skill` 不是「READ 工具穿马甲」。skills 的核心闭环（发现 + 加载指令）本就自洽，
  不依赖 READ。

## 公开 API

```python
from agent_runtime.extensions.skills import (
    SkillManager,          # 只读发现器
    SkillInfo,             # 单个技能元数据
    build_skills_prompt,   # 清单文案
    SkillsPromptHook,      # on_llm_request 清单注入
    build_skill_tool,      # Skill(name) FunctionTool
)
```

`SkillManager` 是只读发现器：

```python
class SkillManager:
    def __init__(self, skills_root=None, extra_skill_dirs=None, config_path=None): ...
    def list_skills(self, *, active_only=False) -> list[SkillInfo]: ...
    def is_plugin_skill(self, name: str) -> bool: ...
    def load_skill(self, name: str) -> str: ...   # 供 Skill 工具调用：按名加载 SKILL.md
```

- **不存在** `install_skill_from_zip` / `delete_skill` / `set_skill_active` / `_save_config` 或任何
  执行环境/远程市场同步方法——写/管理路径归平台。
- `list_skills` 每步被 `on_llm_request` 调用，故以 `skills_root` + `skills.json`（+ 注入的额外
  目录）的 mtime 为键缓存已解析的清单，任一变化才重扫。

## 运维契约（由宿主/平台负责写入）

| 运维意图 | 谁做 | 运行时反应 |
|---------|------|-----------|
| 加 skill | 把 `<name>/SKILL.md`（可选 scripts/references/assets）放进 `skills_root` | 下次 `list_skills` 扫到；`skills.json` 无该 name → 默认 active |
| 禁用某 skill | 写 `skills.json`：`{"skills":{"<name>":{"active":false}}}` | `list_skills(active_only=True)` 过滤掉（对 local 与 plugin 均生效） |
| 删除 skill | 删 `skills_root/<name>` | 下次扫描自然消失 |
| 立即生效 | 无需重启 | mtime 变化即失效缓存，下次 LLM 调用即见 |

`skills.json` 角色收敛为「对已发现 skill 的 opt-out 开关」，纯输入，运行时只读不写。**缺失即默认
启用**；宿主若要白名单语义，自行写 `skills.json` 把不要的置 false，或不放进 `skills_root`。

### 同名优先级

`list_skills` 合并 local 与 plugin 两源：**local 优先于 plugin**——`skills_root` 先占位，注入目录
里的同名条目被跳过。扩展者可用 local 覆盖插件内置技能。

legacy `skill.md`（小写）会被识别用于解析 description，但**不会被改名为 `SKILL.md`**（rename 是写
操作，归平台）。

## 插件贡献技能（声明式目录）

插件以类属性 `skills_dirs` 声明其捆绑的技能目录，与既有 `@tool`/`@on_*` 贡献同构：

```python
class MyPlugin(Plugin):
    name, author, desc, version = "my", "rt", "带技能的插件", "0.1.0"
    skills_dirs = [Path(__file__).parent / "skills"]   # 声明捆绑的 skill 目录
```

- `extensions/plugins/base.py`：`Plugin` 增可选类属性 `skills_dirs`。
- `extensions/plugins/contributions.py`：`PluginContribution.skill_dirs` 由 `collect_contribution`
  从 `type(plugin).skills_dirs` 收集。
- **接线在宿主**（skills 层不 import 插件层）：

  ```python
  extra = [d for c in plugin_manager.contributions for d in c.skill_dirs]
  skill_manager = SkillManager(extra_skill_dirs=extra)
  ```

  skills 层只认 `extra_skill_dirs: list[Path]`，不知道「插件」概念。这些技能 `readonly=True`
  （运行时不可删/改文件），但 active 读过滤**仍生效**——平台可经 `skills.json` 禁用。

## 与插件 hooks 的合成

`runner.reset(agent_hooks=...)` 只收单个 `BaseAgentRunHooks`。`SkillsPromptHook` 与插件
`CompositeAgentRunHooks` 在宿主侧链式合成（layering 禁止 skills→plugins import，故不在扩展内做）：

```python
class _ChainedHooks(BaseAgentRunHooks):
    def __init__(self, *layers):
        self._layers = layers
    async def on_llm_request(self, run_context):
        for layer in self._layers:
            try:
                await layer.on_llm_request(run_context)
            except Exception as e:
                logger.error(f"hook layer error: {e}", exc_info=True)
    # 其余事件按需转发……

runner.reset(
    agent_hooks=_ChainedHooks(SkillsPromptHook(skill_manager), CompositeAgentRunHooks(contributions)),
    ...
)
```

（也可把 `SkillsPromptHook` 纳入一个外层复合体。）钩子自身 try/except + error 日志，异常不中断
主循环，与既有钩子容错一致。

## 最小端到端示例

见 `examples/skills_demo.py`：准备 `data/skills/greet/SKILL.md` → `SkillManager()` 发现 →
`SkillsPromptHook` 把清单注入系统消息（哨兵定界仅一份）→ 模型调 `Skill(name="greet")` 经真实
执行路径拿到 SKILL.md 指令。附属文件读取部分注明待通用 READ 能力（独立 change）。

## import 契约

新增第三方依赖须登记进 `tests/test_audits.py` 的 `ALLOWED_THIRD_PARTY`（本子系统用 `yaml` 解析
frontmatter，已登记）。`tests/test_audits.py` 另审计本层依赖方向（向内、不 import 插件层、不含
宿主/执行环境/远程市场烙印字样）。
