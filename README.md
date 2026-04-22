# MiniClaw 作业 2 项目分析

这是一个用 Python 实现的极简 OpenClaw 风格智能体项目。它的目标不是复刻完整的 OpenClaw 框架，而是用尽量少的代码演示智能体最核心的运行机制：接收用户自然语言输入，调用大模型进行决策，在需要时通过 Function Calling 调用本地工具，再把工具执行结果作为 Observation 交还给模型，最终生成回复。

项目对应作业文档《OpenClaw 个人 Mini 实现-作业2.docx》中“整体框架要求（必做）”部分，当前版本已经覆盖大模型接入、多轮对话上下文、工具调用、ReAct 式消息循环、本地执行安全隔离和异常提示。

## 项目结构

```text
MiniClaw/
├── main.py
├── miniclaw/
│   ├── __init__.py
│   ├── agent.py
│   ├── cli.py
│   ├── history.py
│   ├── llm.py
│   └── tools.py
├── tools_plugins/
│   └── example.py
├── workspace/
│   ├── .gitkeep
│   └── hello.txt
├── OpenClaw 个人 Mini 实现-作业2.docx
├── .gitignore
└── README.md
```

各文件职责如下：

- `main.py`：程序入口，只负责调用 `miniclaw.cli.main()`。
- `miniclaw/cli.py`：命令行交互层，读取用户输入，初始化大模型客户端和 Agent，并处理 `/quit`、`/model`、`/clear` 指令。
- `miniclaw/agent.py`：智能体核心，维护对话历史，执行模型调用、工具调用和 Observation 回传循环。
- `miniclaw/history.py`：对话历史持久化，负责加载、保存和清空 `workspace/history.json`。
- `miniclaw/llm.py`：DeepSeek API 客户端，负责配置读取、OpenAI SDK 调用、响应解析和错误封装。
- `miniclaw/tools.py`：工具系统，包含工具注册表、工具 schema、工具执行入口、装饰器插件加载，以及三个默认本地文件工具。
- `tools_plugins/`：装饰器工具插件目录，新增工具文件后会在启动时自动注册。
- `workspace/`：工具允许访问的安全工作区。Agent 的读写操作被限制在这个目录内。

## 核心能力

### 1. 大模型接入

项目通过 `DeepSeekClient` 接入 DeepSeek Chat Completions API。运行时会先尝试加载项目根目录的 `.env`，再读取配置：

- `DEEPSEEK_API_KEY`：必填，DeepSeek API Key。
- `DEEPSEEK_BASE_URL`：可选，默认是 `https://api.deepseek.com/beta`，用于启用 strict tool calling。
- `DEEPSEEK_MODEL`：可选，默认是 `deepseek-chat`。

`llm.py` 使用 OpenAI SDK 发起兼容 DeepSeek Chat Completions API 的请求，客户端初始化方式如下：

```python
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/beta",
)
```

请求体中包含：

- `model`
- `messages`
- `tools`
- `tool_choice="auto"`
- `stream=True`

这意味着是否调用工具由模型自行判断，符合 Function Calling 的基本设计。工具 schema 使用 DeepSeek strict mode：每个 function 都带 `strict=true`，参数 object 禁止额外字段。

CLI 支持两个模型：

- `deepseek-chat`：普通对话模型。
- `deepseek-reasoner`：会在最终回答前输出 `reasoning_content`，CLI 中显示为 `Reasoning>` 分区。

### 2. 多轮对话上下文

`MiniClawAgent` 内部维护 `messages` 列表，初始化时放入 system prompt，之后每轮用户输入、模型回复和工具 Observation 都会追加进去。

对话历史会自动保存到 `workspace/history.json`。下次启动时，如果该文件存在，CLI 会自动加载历史消息继续对话。

使用 `/clear` 可以清空当前内存中的对话历史，并删除本地历史文件。

### 3. Function Calling 工具调用

工具系统由 `Tool` 和 `ToolRegistry` 两个核心抽象组成：

- `Tool` 保存工具名称、描述、JSON Schema 参数定义和实际 Python 函数。
- `ToolRegistry` 负责注册工具、导出模型可见的工具 schema，并根据模型返回的工具名和参数执行对应函数。

当前内置三个工具：

- `list_files(relative_dir)`：列出 `workspace/` 下某个相对目录的直接子文件和子目录；列根目录时传 `"."`。
- `read_text_file(relative_path)`：读取 `workspace/` 下的 UTF-8 文本文件。
- `write_text_file(relative_path, content)`：向 `workspace/` 下写入 UTF-8 文本文件，并自动创建父目录。

工具执行结果统一序列化为 JSON 字符串，包含 `ok`、`tool` 和具体返回内容或错误信息，便于模型继续理解执行结果。

项目支持装饰器式工具插件。只需要在 `tools_plugins/*.py` 中新增函数并使用 `@tool(...)` 标注，启动时会自动注册：

```python
from miniclaw.tools import tool


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
```

插件工具同样使用 DeepSeek strict mode。工具 schema 会自动补充 `strict=true`、`additionalProperties=false`，并把所有 properties 写入 required。

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

项目设置了 `max_steps=6`，避免模型持续调用工具导致无限循环。

### 5. 本地执行安全隔离

安全边界主要在 `tools.py` 的 `_safe_workspace_path()` 中实现。它会拒绝：

- 绝对路径。
- 包含 `..` 的路径。
- 解析后不位于 `workspace/` 目录内的路径。

因此，即使用户要求读取项目根目录下的作业文档或其他文件，工具也会拒绝越界访问。这一点符合“本地执行与安全隔离”的作业要求。

## 运行方式

推荐在项目根目录创建 `.env`（已被 `.gitignore` 忽略）：

```dotenv
DEEPSEEK_API_KEY=你的 DeepSeek API Key
DEEPSEEK_BASE_URL=https://api.deepseek.com/beta
DEEPSEEK_MODEL=deepseek-chat
```

也可以继续使用临时环境变量（仅当前终端会话有效）：

```powershell
$env:DEEPSEEK_API_KEY="你的 DeepSeek API Key"
```

可选配置：

```powershell
$env:DEEPSEEK_BASE_URL="https://api.deepseek.com/beta"
$env:DEEPSEEK_MODEL="deepseek-chat"
```

启动程序：

```powershell
pip install -r requirements.txt
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
/model deepseek-chat
/model deepseek-reasoner
```

## 示例指令

```text
你是谁？
/model deepseek-reasoner
列出 workspace 里的文件
创建 hello.txt，写入“你好，MiniClaw”，然后读取确认内容
调用 echo_text 工具返回 hello plugin
尝试读取 ../OpenClaw 个人 Mini 实现-作业2.docx
/clear
/quit
```

最后一条会被安全检查拒绝，因为工具只能访问 `workspace/`。

## 静态检查

可以使用下面的命令检查 Python 文件是否能正常编译：

```powershell
python -m compileall main.py miniclaw
```

当前检查结果：`main.py` 和 `miniclaw/` 均可通过编译。

## 项目优点

- 结构清晰：入口、CLI、Agent、大模型客户端、工具系统拆分明确，适合学习智能体框架的基本组成。
- 依赖轻量：除 OpenAI SDK 外只使用 Python 标准库，环境搭建成本低。
- 核心链路完整：已经具备模型决策、工具调用、工具结果回传和最终回答的闭环。
- 安全意识明确：所有工具访问都限制在 `workspace/` 下，并提供友好的错误返回。
- 可扩展性有雏形：`ToolRegistry` 已经把工具注册和工具执行统一起来，后续增加工具比较自然。

## 当前限制

- 工具数量较少，目前只覆盖基础文件列表、读取和写入。
- ReAct 过程只打印 Observation，没有更细粒度地展示模型思考、行动和最终结论的结构化过程。
- 缺少自动化单元测试，目前主要依赖 `compileall` 做静态编译检查。
- 未实现 shell 执行工具，因此无法完成更复杂的本地自动化任务。

## 后续扩展建议

1. 增加单元测试，重点覆盖路径安全检查、工具参数错误、文件读写、历史持久化和插件加载。
2. 增加更清晰的 ReAct 日志展示，将每轮工具调用、Observation 和最终回答分层输出。
3. 在安全白名单基础上实现有限的 shell 工具，支持更多真实系统操作。
4. 把 LLM 客户端抽象成通用接口，未来可以切换 DeepSeek、OpenAI、Claude 或本地模型。

## 总结

MiniClaw 是一个面向教学和作业场景的迷你智能体框架。它没有追求复杂功能，而是抓住了智能体系统最重要的骨架：大模型负责理解和决策，工具负责真实执行，Agent 循环负责在两者之间传递上下文和 Observation。

从完成度看，它已经满足作业必做要求；从工程角度看，它的模块边界清楚、安全边界明确，后续可以沿着持久化、插件化、测试、流式输出和更多工具能力继续演进。
