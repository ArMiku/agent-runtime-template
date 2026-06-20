# fs — 只读型文件系统工具集（无执行隔离的安全默认）

为 agent 提供一套**按路径寻址**的只读文件访问能力：`list_dir`（发现）+ `read_file`
（读取）。它与 skills 子系统的**按名字寻址**互补——`Skill(name)` 加载 `SKILL.md` 指令，
本工具集读取指令引用的附属文件（`scripts/` / `references/` / `assets/`）及其它允许根
内的文件。

## 设计定位

- **只读**：全部工具只读，不提供写入、编辑、删除或任意命令执行。
- **窄工具、无 shell**：入参走 JSON Schema，不经任何 shell 解释器（`;` / `$()` / 反引号
  / `|` / `>` 等都是字面字符），因此不存在 shell 文法逃逸面。
- **路径围栏**：每个 `path` 先 `realpath` 归一化，再校验落在某个**允许根**之内。安全边界
  收敛为单一可测命题：`realpath(path)` 是否在某个 `allowed_roots` 之下。
- **无执行隔离依赖**：本工具集不依赖容器、landlock、seccomp 或任何隔离执行环境。它本身就是
  本地、只读场景下的**安全默认**。

## 过渡契约：可被 Bash 平替

本工具集是「**没有隔离执行环境时的安全默认**」，而非不可变设计。**当宿主提供隔离执行环境**
（如容器隔离的 shell）时，宿主**应**把整套 fs 工具从 `ToolSet` 摘除，替换为单个 Bash 工具，
以隔离兜底换取 shell 的组合力（管道、重定向、跨命令链）。落地方式：

- 宿主在 `ToolSet` 构造层整体替换，**不改 fs 扩展源码**；fs 扩展不假定自己是唯一文件访问通道。
- 每个工具的 `docstring` 开头都有一句 `safe-by-default narrow tool; superseded by an
  isolated-execution Bash tool when the host provides one`。
- 宿主替换后，agent 的工作流迁移到 Bash 惯用法（`cat -n` 带行号、`wc -l` 取行数），而非格式
  对等——Bash 输出本身不带行号 /「剩余 N 行」/ 元数据。

## 两个寻址空间

| 段 | 寻址空间 | 通道 | 范围 |
|----|---------|------|------|
| skills 清单 | **name** | 系统消息注入 | 可用技能名 + 描述 |
| skills 指令 | **name** | `Skill(name)` | 仅 `SKILL.md` 正文 |
| **附属文件** | **path** | `list_dir` / `read_file` | 允许根内的任意文件 |

`Skill(name)` 只收 name、无 path 入参、无穿越面；fs 工具按 path 读附属文件、由围栏保护。两
者**不重叠**：name 寻址与 path 寻址互补，合起来使 skills 闭环自洽。

## 工具

### `read_file(path, start_line=1, line_count=2000)`

- **强制带行号**：输出 `{行号}\t{内容}`（为后续编辑/替换提供稳定行坐标）。
- **双上限分块**：返回 `[start_line, start_line+line_count)` 区间，**并叠加每块字节上限
  `MAX_BYTES ≈ 32KB`**——累计字节先达即在当前行**中途截断**。仅按行数分块挡不住单行超长文件
  （压缩/打包），字节上限是 token 安全的兜底。
- **元数据**：尾部给出总行数、已读行数（字节数）、mtime、编码。
- **续读提示**：未读完时标注「剩余 N 行未读；续读：read_file(path, start_line=K)」；字节上限
  截断时标注截断行。
- **仅 UTF-8**：解码失败即视为不可读，提示改用专用解析工具；不做多编码探测、不静默吞码。
- **二进制拒绝**：检测 NUL 字节等二进制特征一律拒绝（含图片），不把二进制/图片字节灌入对话。
  图片（如 skills `assets/`）的读取留后续 change。
- **越界**：`start_line` 超出总行数 → 友好错误（指明合法范围 `1..N`）。

### `list_dir(path)`

- 返回目录条目：每行 `{名}\t[类型]`（`[dir]` 或 `[file, {字节数}B]`），按名排序。
- **叠加 `MAX_ENTRIES` 上限**（1000），超限标注「已截断，还有 N 条未列」。
- 解决「`read_file` 要先知道读什么」的发现问题。

### 错误规约

围栏拒绝 / 文件不存在 / 越界 `start_line` 等 handler 抛出的异常，由 runner 归一化为
`error:` 文本结果回灌 LLM（不中断主循环）：

| 情况 | 异常 | runner 归一化 |
|------|------|--------------|
| 路径越界 / 静态符号链接逃逸 | `PermissionError` | `error: ...` |
| 文件/目录不存在 | `FileNotFoundError` | `error: ...` |
| 非 UTF-8 / 二进制 / 越界 start_line / 目录路径当文件读 | `ValueError` | `error: ...` |

## 安全模型：路径围栏（containment）

```
read_file / list_dir(path)
   └─ resolve_within_roots(path, allowed_roots)
        ├─ real = os.path.realpath(path)            # 解析符号链接、..、相对段
        ├─ for root in allowed_roots:
        │      root_real = os.path.realpath(root)
        │      if real == root_real or real.startswith(root_real + os.sep):
        │           return Path(real)               # 严格前缀，防 /a 放行 /ab
        └─ raise PermissionError
```

要点：

- **`realpath` 先行**：符号链接、`..`、相对段全部解析后再判 → **静态**符号链接逃逸天然被拦。
- **严格前缀**：用 `root + os.sep` 前缀，避免 `/data/skills` 误放行 `/data/skills-secret`。
- **`allowed_roots` 宿主注入**：fs 层不硬编码根；默认 `[get_skills_dir()]`；宿主可注入仓库根
  等放宽语义。fs 层**不 import** skills/plugins，两侧只通过「路径列表」这一纯数据在宿主接线。
- **TOCTOU 取舍**：该 check-then-open 围栏不防御「校验与打开之间」的竞态。本地只读场景低危，
  作为已知取舍记录于此。

## 接线（由宿主负责）

```python
from agent_runtime.extensions.fs import build_fs_tools
from agent_runtime.foundation.paths import get_skills_dir

# 默认围栏含 skills 目录；按需追加插件提供的技能目录等。
fs_tools = build_fs_tools(allowed_roots=[get_skills_dir(), *plugin_skill_dirs])
tools = ToolSet([skill_tool, *fs_tools])
```

## 依赖方向

`extensions/fs` MAY import `core` / `foundation`；`core` 等核心包 MUST NOT import
`extensions/fs`；`extensions/fs` MUST NOT import `extensions/plugins` 或 `extensions/skills`
（`tests/test_audits.py` 强制审计）。

## 非目标

- 不实现 `glob` / `grep`（留作后续小 change）。
- 不实现 `write` / `edit` / `delete` / `shell`（需隔离执行，属另一战略）。
- 不引入容器 / 隔离执行环境。
- 不改变宿主应用自身行为。
