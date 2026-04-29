# MiniClaw 作业 2 项目分析

这是一个用 Python 实现的极简 OpenClaw 风格智能体项目。它的目标不是复刻完整的 OpenClaw 框架，而是用尽量少的代码演示智能体最核心的运行机制：接收用户自然语言输入，调用大模型进行决策，在需要时通过 Function Calling 调用本地工具，再把工具执行结果作为 Observation 交还给模型，最终生成回复。

项目对应作业文档《OpenClaw 个人 Mini 实现-作业2.docx》中“整体框架要求（必做）”部分，当前版本已经覆盖大模型接入、多轮对话上下文、工具调用、ReAct 式消息循环、本地执行安全隔离和异常提示。

## 项目结构

```text
MiniClaw/
├── main.py
├── miniclaw/
│   ├── __init__.py
│   ├── app.py
│   ├── agent.py
│   ├── cli.py
│   ├── history.py
│   ├── llm.py
│   ├── runtime.py
│   ├── subagents.py
│   ├── tools.py
│   └── web_server.py
├── tools_plugins/
│   └── example.py
├── web/
│   └── index.html
├── workspace/
│   ├── .gitkeep
│   └── hello.txt
├── OpenClaw 个人 Mini 实现-作业2.docx
├── .gitignore
└── README.md
```

各文件职责如下：

- `main.py`：CLI 启动入口，Web 不再通过它分支启动。
- `miniclaw/app.py`：FastAPI app 工厂与 Web 路由入口。
- `miniclaw/cli.py`：命令行交互层，复用共享 runtime，并处理 `/quit`、`/model`、`/clear` 指令。
- `miniclaw/agent.py`：智能体核心，维护对话历史，执行模型调用、工具调用和 Observation 回传循环。
- `miniclaw/history.py`：对话历史持久化，负责加载、保存和清空 `workspace/history.json`。
- `miniclaw/llm.py`：DeepSeek 模型配置层，基于 `pydantic-settings` 读取配置，并维护 `pydantic-ai` 的 V4 模型预设与思考模式参数。
- `miniclaw/runtime.py`：CLI 与 Web 共享的运行时，统一初始化客户端、Agent、模型切换与历史清空。
- `miniclaw/subagents.py`：分析子 Agent，用于承接主 Agent 委托的只读文件分析任务。
- `miniclaw/tools.py`：工具系统，包含工具注册表、`pydantic-ai` 工具包装、`pluggy` 插件加载、安全 shell 工具，以及默认本地工具。
- `miniclaw/web_server.py`：兼容层，保留 `run()` 启动函数并转发到新的 FastAPI app。
- `tools_plugins/`：插件目录，新增工具文件后通过 `pluggy` hook 自动注册。
- `web/`：内置静态前端页面，由 FastAPI 直接提供。
- `workspace/`：工具允许访问的安全工作区。Agent 的读写操作被限制在这个目录内。

## 核心能力

### 1. 大模型接入

项目通过 `pydantic-ai` 的原生 DeepSeek provider 接入 DeepSeek。运行时使用 `pydantic-settings` 自动读取项目根目录 `.env` 中的配置：

- `DEEPSEEK_API_KEY`：必填，DeepSeek API Key。
- `DEEPSEEK_MODEL`：可选，默认是 `deepseek-v4-flash`。
- `SERPAPI_KEY`：`search_web` 必填，SerpAPI Key。
- `SERPAPI_LOCATION`：可选，默认 `Austin, Texas, United States`。
- `SERPAPI_HL`：可选，默认 `en`。
- `SERPAPI_GL`：可选，默认 `us`。
- `SERPAPI_GOOGLE_DOMAIN`：可选，默认 `google.com`。

`llm.py` 会把 `.env` 中的配置映射到两个固定预设：

```python
MODEL_PRESETS = {
    "deepseek-v4-flash": {
        "model": "deepseek-v4-flash",
        "model_settings": {"extra_body": {"thinking": {"type": "disabled"}}},
    },
    "deepseek-v4-pro": {
        "model": "deepseek-v4-pro",
        "model_settings": {
            "extra_body": {"thinking": {"type": "enabled"}},
            "openai_reasoning_effort": "high",
        },
    },
}
```

主 Agent 和子 Agent 都通过 `Agent("deepseek:<model_id>")` 创建，并共享同一套模型预设解析逻辑。项目不再显式暴露 `OpenAIProvider`、`OpenAIChatModel` 或 `base_url=/beta` 兼容层。

CLI 支持两个模型：

- `deepseek-v4-flash`：快速预设，默认关闭 thinking。
- `deepseek-v4-pro`：高质量预设，默认开启 thinking，并附带 `reasoning_effort=high`。

### 2. 多轮对话上下文

`MiniClawAgent` 内部维护 `messages` 列表，初始化时放入 system prompt，之后每轮用户输入、模型回复和工具 Observation 都会追加进去。

对话历史会自动保存到 `workspace/history.json`。下次启动时，如果该文件存在，CLI 会自动加载历史消息继续对话。

使用 `/clear` 可以清空当前内存中的对话历史，并删除本地历史文件。

### 3. Function Calling 工具调用

工具系统由 `Tool` 和 `ToolRegistry` 两个核心抽象组成：

- `Tool` 保存工具名称、描述、JSON Schema 参数定义和实际 Python 函数。
- `ToolRegistry` 负责注册工具、导出模型可见的工具 schema，并根据模型返回的工具名和参数执行对应函数。

当前内置工具包括：

- `list_files(relative_dir)`：列出 `workspace/` 下某个相对目录的直接子文件和子目录；列根目录时传 `"."`。
- `read_text_file(relative_path)`：读取 `workspace/` 下的 UTF-8 文本文件。
- `list_directory_tree(relative_dir, max_depth)`：递归列出 `workspace/` 下的目录树，适合快速了解结构。
- `search_files(pattern, relative_dir, case_sensitive, max_matches)`：在 `workspace/` 下搜索 UTF-8 文件内容，返回匹配行。
- `write_text_file(relative_path, content)`：向 `workspace/` 下写入 UTF-8 文本文件，并自动创建父目录。
- `append_text_file(relative_path, content)`：向 `workspace/` 下的文本文件追加内容，不存在则创建。
- `replace_text_in_file(relative_path, old_text, new_text, count)`：在 `workspace/` 下的文本文件中替换指定文本。
- `search_web(query, max_results, site)`：使用 SerpAPI 的 Google 搜索联网检索公开网页信息，支持返回数量和站点限定。
- `fetch_web_page(url, max_chars)`：打开公开网页 URL，提取标题、正文文本和链接摘要。
- `get_current_time(timezone)`：获取指定 IANA 时区的当前日期和时间。
- `calculate_expression(expression)`：安全计算数学表达式，支持常见 `math` 函数。
- `run_shell_command(command, args, working_dir)`：在 `workspace/` 内执行安全的只读 shell 命令。
- `delegate_file_analysis(relative_path, task)`：把文件分析任务委托给只读分析子 Agent。

工具执行结果统一序列化为 JSON 字符串，包含 `ok`、`tool` 和具体返回内容或错误信息，便于模型继续理解执行结果。

项目支持基于 `pluggy` 的工具插件。只需要在 `tools_plugins/*.py` 中声明 `register_tools()` hook，并返回工具列表：

```python
from miniclaw.tools import hookimpl, tool


@hookimpl
def register_tools():
    @tool(
        name="echo_text",
        description="返回用户传入的一段文本。",
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "要原样返回的文本。",
                }
            },
        },
    )
    def echo_text(text: str) -> dict[str, str]:
        return {"text": text}

    return [echo_text]
```

插件工具会通过 `pydantic-ai` 自动生成工具 schema，并保持现有工具名与调用语义不变。

`run_shell_command` 只允许 `pwd`、`ls`、`cat`、`head`、`tail`、`wc`、`rg`，并拒绝管道、重定向、命令连接符、越界路径和非白名单参数。命令通过 `subprocess.run(..., shell=False)` 执行，工作目录和文件参数都必须位于 `workspace/` 内。

`delegate_file_analysis` 会创建一个短期分析子 Agent。子 Agent 与主 Agent 共享同一个模型预设选择逻辑，但只拥有 `list_files` 和 `read_text_file` 两个只读工具，也不会写入主对话历史。

### 4. ReAct 式消息循环

`agent.py` 中的 `run_turn()` 是整个项目的核心流程：

```text
用户输入
  -> 追加到 messages
  -> 调用大模型
  -> 如果模型直接回复，则返回最终答案
  -> 如果模型请求工具调用，则执行工具
  -> 将工具结果作为 tool message 写回 messages
  -> 继续调用大模型
  -> 直到模型给出最终答案或达到最大轮数
```

这个流程对应作业要求中的 ReAct 思路：

- Thought：由大模型内部完成推理和决策。
- Action：通过 Function Calling 选择工具和参数。
- Observation：本地工具执行结果回传给模型。
- Final Answer：模型基于 Observation 给出最终回复。

控制台会按轮次展示 `[Step N Thought]`、`[Step N Action]`、`[Step N Observation]`。当当前预设开启 thinking（如 `deepseek-v4-pro`）时，CLI 会把 `reasoning_content` 展示在 `Reasoning>` 分区；Observation 支持工具执行过程中的增量输出，最终仍会展示完整工具 JSON。

项目设置了 `max_steps=6`，避免模型持续调用工具导致无限循环。

### 5. 本地执行安全隔离

安全边界主要在 `tools.py` 的 `_safe_workspace_path()` 中实现。它会拒绝：

- 绝对路径。
- 包含 `..` 的路径。
- 解析后不位于 `workspace/` 目录内的路径。

因此，即使用户要求读取项目根目录下的作业文档或其他文件，工具也会拒绝越界访问。这一点符合“本地执行与安全隔离”的作业要求。

shell 工具额外使用命令白名单和参数校验，只支持只读命令，不允许写文件、环境变量展开、管道或重定向。需要联网时应使用 `search_web`（底层通过 SerpAPI 调 Google 搜索），而不是通过 shell 绕过限制。

## 运行方式

推荐在项目根目录创建 `.env`（已被 `.gitignore` 忽略）：

```dotenv
DEEPSEEK_API_KEY=你的 DeepSeek API Key
DEEPSEEK_MODEL=deepseek-v4-flash
SERPAPI_KEY=你的 SerpAPI Key
SERPAPI_LOCATION=Austin, Texas, United States
SERPAPI_HL=en
SERPAPI_GL=us
SERPAPI_GOOGLE_DOMAIN=google.com
```

也可以继续使用临时环境变量（仅当前终端会话有效）：

```powershell
$env:DEEPSEEK_API_KEY="你的 DeepSeek API Key"
```

可选配置：

```powershell
$env:DEEPSEEK_MODEL="deepseek-v4-flash"
```

说明：

- 旧的 `DEEPSEEK_BASE_URL` 现在会被静默忽略，不再用于启用 strict tool calling。
- DeepSeek 官方已说明 `deepseek-chat` 与 `deepseek-reasoner` 将于 `2026-07-24` 停用，因此本项目默认直接切到 V4 预设。

推荐使用 `MiniClaw` conda 环境，不要混用 `base`、仓库里的 `.conda` 和 `MiniClaw` 三套解释器：

```bash
conda activate MiniClaw
pip install -r requirements.txt
```

启动 Web：

```bash
uvicorn miniclaw.app:app --host 127.0.0.1 --port 8000 --reload
```

打开：

```text
http://127.0.0.1:8000
```

启动 CLI：

```bash
python main.py
```

输入 `/quit` 退出。普通 `quit` 会作为用户消息发送给模型。

历史指令：

```text
/clear
```

模型指令：

```text
/model
/model deepseek-v4-flash
/model deepseek-v4-pro
```

## 示例指令

```text
你是谁？
/model deepseek-v4-pro
列出 workspace 里的文件
创建 hello.txt，写入“你好，MiniClaw”，然后读取确认内容
在 workspace 中搜索 main 函数
把 hello.txt 末尾追加一行“继续扩展工具”
调用 echo_text 工具返回 hello plugin
搜索 DeepSeek API 最新文档，并打开一条结果核验正文
计算 sin(pi / 2) + 2^8
告诉我 Asia/Shanghai 当前时间
用 shell 列出 workspace 根目录
分析 workspace/hello.txt 的主要内容
尝试读取 ../OpenClaw 个人 Mini 实现-作业2.docx
/clear
/quit
```

最后一条会被安全检查拒绝，因为工具只能访问 `workspace/`。

## 运行检查

可以使用下面的命令检查 Python 文件是否能正常编译：

```powershell
python -m compileall main.py miniclaw tools_plugins
```

当前检查结果：`main.py`、`miniclaw/` 和 `tools_plugins/` 均可通过编译。

## 项目优点

- 结构清晰：入口、CLI、Agent、大模型客户端、工具系统拆分明确，适合学习智能体框架的基本组成。
- 依赖仍较轻量：核心只依赖 `pydantic-ai`、FastAPI 和少量工具库，环境搭建成本较低。
- 核心链路完整：已经具备模型决策、工具调用、工具结果回传和最终回答的闭环。
- 安全意识明确：所有工具访问都限制在 `workspace/` 下，并提供友好的错误返回。
- 可扩展性有雏形：`ToolRegistry` 已经把工具注册和工具执行统一起来，后续增加工具比较自然。

## 当前限制

- 这是一个教学版 demo，没有额外保留自动化测试文件。
- shell 工具目前只支持只读白名单命令，不能执行写入、联网或复杂管道任务。
- 多 Agent 协作仍是雏形，目前只有主 Agent 委托文件分析子 Agent。

## Web 前端模式

项目新增了基于 FastAPI + SSE 的前端模式，提供：

- Markdown 渲染：支持 GFM（标题、列表、表格、代码块等）。
- 代码高亮：前端使用 highlight.js 渲染代码块。
- 数学公式：支持 $inline$ 与 $$block$$ 公式渲染（KaTeX）。
- 单框时间线展示：在同一个 assistant 气泡内按时间顺序展示 Thought / Action / Observation / Final Answer。

说明：

- Web 模式与 CLI 模式并存，但现在以 FastAPI Web 入口为主。
- `python main.py` 仍然保留为 CLI 交互模式。
- 标准 Web 启动方式是 `uvicorn miniclaw.app:app`。
- TAO 事件由后端按步骤推送；Action 和 Observation 都会在同一个 assistant 消息时间线中增量更新，工具完成后再显示完整 Observation JSON。

## 子 Agent 手工测试

当前只有一种子 Agent：`delegate_file_analysis` 触发的只读文件分析子 Agent。它不是通用任务路由器，也不会并行调度多个子 Agent。

建议先准备一个测试文件，例如在 `workspace/hello.txt` 中写入 2 到 5 行文本，然后分别在 Web 或 CLI 中测试以下场景：

1. 基础委托：

```text
请使用 delegate_file_analysis 分析 workspace/hello.txt 的主要内容，并给出一句总结。
```

2. 自然语言委托：

```text
请分析 workspace/hello.txt，但不要直接抄全文，先委托子 Agent 再总结。
```

3. 越界失败：

```text
请分析 ../README.md
```

4. 普通工具对照：

```text
列出 workspace 文件
```

判断标准如下：

- Web 前端：同一个 assistant 消息时间线里应出现 `delegate_file_analysis` 这个工具名；对应 Observation JSON 里应包含 `analysis` 字段。
- CLI：终端输出里应出现 `[Step N Action]` 下的 `delegate_file_analysis(...)`，以及 `[Step N Observation]` 中包含 `analysis`。
- 最终回答：应基于 `analysis` 字段做总结，而不是只原样返回工具 JSON。
- 越界测试：应体现 `workspace/` 路径限制，不能成功读取 `../README.md`。
- 对照测试：`列出 workspace 文件` 更可能调用 `list_files`，用于区分普通工具调用和子 Agent 委托。

## 总结

MiniClaw 是一个面向教学和作业场景的迷你智能体框架。它没有追求复杂功能，而是抓住了智能体系统最重要的骨架：大模型负责理解和决策，工具负责真实执行，Agent 循环负责在两者之间传递上下文和 Observation。

从完成度看，它已经满足作业必做要求；从工程角度看，它的模块边界清楚、安全边界明确，后续可以沿着持久化、插件化、测试、流式输出和更多工具能力继续演进。
