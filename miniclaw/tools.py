from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_ROOT / "workspace"
PLUGINS_DIR = PROJECT_ROOT / "tools_plugins"

ToolFunction = Callable[..., dict[str, Any]]
_PLUGIN_TOOLS: list["Tool"] = []
_LOADED_PLUGIN_FILES: set[Path] = set()
_SHELL_COMMANDS = {"pwd", "ls", "cat", "head", "tail", "wc", "rg"}
_SHELL_META_CHARS = {"|", "&", ";", "<", ">", "`", "$", "(", ")", "{", "}"}
_MAX_SHELL_OUTPUT = 4000


class ToolError(Exception):
    """Raised for safe, user-facing tool errors."""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    function: ToolFunction

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": _strict_parameters(self.parameters),
                "strict": True,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def run(self, name: str, arguments: dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return _json_result(ok=False, tool=name, error=f"未知工具：{name}")

        try:
            result = tool.function(**arguments)
            return _json_result(ok=True, tool=name, **result)
        except TypeError as exc:
            return _json_result(ok=False, tool=name, error=f"工具参数错误：{exc}")
        except ToolError as exc:
            return _json_result(ok=False, tool=name, error=str(exc))
        except Exception as exc:
            return _json_result(ok=False, tool=name, error=f"工具执行异常：{exc}")


def create_default_registry(client: Any | None = None) -> ToolRegistry:
    WORKSPACE_DIR.mkdir(exist_ok=True)

    registry = ToolRegistry()
    register_file_tools(registry, writable=True)
    register_shell_tool(registry)
    if client is not None:
        register_delegate_tool(registry, client)
    load_plugins(registry)
    return registry


def create_read_only_registry() -> ToolRegistry:
    registry = ToolRegistry()
    register_file_tools(registry, writable=False)
    return registry


def register_file_tools(registry: ToolRegistry, writable: bool = True) -> None:
    registry.register(
        Tool(
            name="list_files",
            description="列出 workspace 安全目录内某个相对目录的直接子文件和子目录。",
            parameters={
                "type": "object",
                "properties": {
                    "relative_dir": {
                        "type": "string",
                        "description": "workspace 内的相对目录；列根目录时使用 .",
                    }
                },
                "required": ["relative_dir"],
                "additionalProperties": False,
            },
            function=list_files,
        )
    )
    registry.register(
        Tool(
            name="read_text_file",
            description="读取 workspace 安全目录内的 UTF-8 文本文件。",
            parameters={
                "type": "object",
                "properties": {
                    "relative_path": {
                        "type": "string",
                        "description": "workspace 内要读取的相对文件路径。",
                    }
                },
                "required": ["relative_path"],
                "additionalProperties": False,
            },
            function=read_text_file,
        )
    )
    if writable:
        registry.register(
            Tool(
                name="write_text_file",
                description="向 workspace 安全目录内写入 UTF-8 文本文件，会自动创建父目录。",
                parameters={
                    "type": "object",
                    "properties": {
                        "relative_path": {
                            "type": "string",
                            "description": "workspace 内要写入的相对文件路径。",
                        },
                        "content": {
                            "type": "string",
                            "description": "要写入文件的文本内容。",
                        },
                    },
                    "required": ["relative_path", "content"],
                    "additionalProperties": False,
                },
                function=write_text_file,
            )
        )


def register_shell_tool(registry: ToolRegistry) -> None:
    registry.register(
        Tool(
            name="run_shell_command",
            description="在 workspace 内安全执行只读 shell 命令。只允许 pwd、ls、cat、head、tail、wc、rg。",
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "允许的命令：pwd、ls、cat、head、tail、wc、rg。",
                    },
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "命令参数。无参数时使用空数组。",
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "workspace 内的工作目录；默认意图请传 .。",
                    },
                },
            },
            function=run_shell_command,
        )
    )


def register_delegate_tool(registry: ToolRegistry, client: Any) -> None:
    registry.register(
        Tool(
            name="delegate_file_analysis",
            description="将 workspace 文件分析任务委托给只读分析子 Agent。",
            parameters={
                "type": "object",
                "properties": {
                    "relative_path": {
                        "type": "string",
                        "description": "workspace 内要分析的文件路径。",
                    },
                    "task": {
                        "type": "string",
                        "description": "希望分析子 Agent 完成的具体分析任务。",
                    },
                },
            },
            function=_make_delegate_file_analysis(client),
        )
    )


def tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
) -> Callable[[ToolFunction], ToolFunction]:
    def decorator(function: ToolFunction) -> ToolFunction:
        _PLUGIN_TOOLS.append(
            Tool(
                name=name,
                description=description,
                parameters=parameters,
                function=function,
            )
        )
        return function

    return decorator


def load_plugins(registry: ToolRegistry, plugins_dir: Path = PLUGINS_DIR) -> None:
    if not plugins_dir.is_dir():
        return

    for plugin_path in sorted(plugins_dir.glob("*.py")):
        if plugin_path.name == "__init__.py" or plugin_path in _LOADED_PLUGIN_FILES:
            continue

        module_name = f"miniclaw_plugin_{plugin_path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, plugin_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        _LOADED_PLUGIN_FILES.add(plugin_path)

    for plugin_tool in _PLUGIN_TOOLS:
        registry.register(plugin_tool)


def list_files(relative_dir: str = ".") -> dict[str, Any]:
    directory = _safe_workspace_path(relative_dir)
    if not directory.exists():
        raise ToolError(f"目录不存在：{relative_dir}")
    if not directory.is_dir():
        raise ToolError(f"不是目录：{relative_dir}")

    entries = []
    for child in sorted(directory.iterdir(), key=lambda item: item.name.lower()):
        entries.append(
            {
                "name": child.name,
                "type": "directory" if child.is_dir() else "file",
                "size": None if child.is_dir() else child.stat().st_size,
            }
        )

    return {"relative_dir": relative_dir, "entries": entries}


def read_text_file(relative_path: str) -> dict[str, Any]:
    path = _safe_workspace_path(relative_path)
    if not path.exists():
        raise ToolError(f"文件不存在：{relative_path}")
    if not path.is_file():
        raise ToolError(f"不是文件：{relative_path}")

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ToolError(f"文件不是有效 UTF-8 文本：{relative_path}") from exc

    return {"relative_path": relative_path, "content": content}


def write_text_file(relative_path: str, content: str) -> dict[str, Any]:
    path = _safe_workspace_path(relative_path)
    if path.exists() and path.is_dir():
        raise ToolError(f"目标路径是目录，无法写入文件：{relative_path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

    return {
        "relative_path": relative_path,
        "bytes": len(content.encode("utf-8")),
        "message": "文件写入成功。",
    }


def run_shell_command(
    command: str,
    args: list[str],
    working_dir: str,
) -> dict[str, Any]:
    command_args = _validate_shell_command(command, args)
    cwd = _safe_workspace_path(working_dir)
    if not cwd.is_dir():
        raise ToolError(f"工作目录不存在或不是目录：{working_dir}")

    completed = subprocess.run(
        [command, *command_args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=5,
        shell=False,
        check=False,
    )

    stdout = _truncate(completed.stdout)
    stderr = _truncate(completed.stderr)
    return {
        "command": command,
        "args": command_args,
        "working_dir": working_dir,
        "returncode": completed.returncode,
        "stdout": stdout,
        "stderr": stderr,
    }


def _safe_workspace_path(relative_path: str) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise ToolError("安全限制：不允许使用绝对路径。")
    if any(part == ".." for part in candidate.parts):
        raise ToolError("安全限制：不允许使用包含 .. 的路径。")

    base = WORKSPACE_DIR.resolve()
    resolved = (base / candidate).resolve()
    if resolved != base and not resolved.is_relative_to(base):
        raise ToolError("安全限制：路径必须位于 workspace 目录内。")

    return resolved


def _make_delegate_file_analysis(client: Any) -> ToolFunction:
    def delegate_file_analysis(relative_path: str, task: str) -> dict[str, Any]:
        from .subagents import AnalysisSubAgent

        analysis = AnalysisSubAgent(client).analyze(relative_path=relative_path, task=task)
        return {
            "relative_path": relative_path,
            "task": task,
            "analysis": analysis,
        }

    return delegate_file_analysis


def _validate_shell_command(command: str, args: list[str]) -> list[str]:
    _reject_shell_syntax(command)
    if command not in _SHELL_COMMANDS:
        raise ToolError(f"不允许执行命令：{command}")
    if not isinstance(args, list):
        raise ToolError("args 必须是字符串数组。")

    for arg in args:
        _reject_shell_syntax(arg)

    validators = {
        "pwd": _validate_pwd_args,
        "ls": _validate_ls_args,
        "cat": _validate_cat_args,
        "head": _validate_head_tail_args,
        "tail": _validate_head_tail_args,
        "wc": _validate_wc_args,
        "rg": _validate_rg_args,
    }
    return validators[command](args)


def _reject_shell_syntax(value: str) -> None:
    if "\n" in value or "\x00" in value:
        raise ToolError("shell 参数不允许包含换行或 NUL 字符。")
    if any(char in value for char in _SHELL_META_CHARS):
        raise ToolError(f"shell 参数包含不允许的语法字符：{value}")


def _validate_pwd_args(args: list[str]) -> list[str]:
    if args:
        raise ToolError("pwd 不允许携带参数。")
    return []


def _validate_ls_args(args: list[str]) -> list[str]:
    allowed_flags = {"-l", "-a", "-la", "-al"}
    return _validate_flags_and_paths(args, allowed_flags, allow_directory=True)


def _validate_cat_args(args: list[str]) -> list[str]:
    if not args:
        raise ToolError("cat 需要至少一个 workspace 内文件路径。")
    return [_validate_file_arg(arg) for arg in args]


def _validate_head_tail_args(args: list[str]) -> list[str]:
    result: list[str] = []
    index = 0
    if len(args) >= 2 and args[0] == "-n":
        if not args[1].isdigit():
            raise ToolError("-n 后必须是数字。")
        result.extend(args[:2])
        index = 2

    paths = args[index:]
    if not paths:
        raise ToolError("head/tail 需要至少一个 workspace 内文件路径。")
    result.extend(_validate_file_arg(path) for path in paths)
    return result


def _validate_wc_args(args: list[str]) -> list[str]:
    allowed_flags = {"-l", "-w", "-c"}
    return _validate_flags_and_paths(args, allowed_flags, allow_directory=False)


def _validate_rg_args(args: list[str]) -> list[str]:
    flags: list[str] = []
    rest = list(args)
    while rest and rest[0] in {"-n", "-i"}:
        flags.append(rest.pop(0))

    if not rest:
        raise ToolError("rg 需要搜索 pattern。")

    pattern = rest.pop(0)
    if rest:
        if len(rest) > 1:
            raise ToolError("rg 只允许一个可选搜索目录。")
        return [*flags, pattern, _validate_dir_arg(rest[0])]

    return [*flags, pattern]


def _validate_flags_and_paths(
    args: list[str],
    allowed_flags: set[str],
    allow_directory: bool,
) -> list[str]:
    result: list[str] = []
    for arg in args:
        if arg.startswith("-"):
            if arg not in allowed_flags:
                raise ToolError(f"不允许的参数：{arg}")
            result.append(arg)
            continue

        result.append(_validate_dir_arg(arg) if allow_directory else _validate_file_arg(arg))
    return result


def _validate_file_arg(arg: str) -> str:
    path = _safe_workspace_path(arg)
    if not path.is_file():
        raise ToolError(f"不是 workspace 内文件：{arg}")
    return arg


def _validate_dir_arg(arg: str) -> str:
    path = _safe_workspace_path(arg)
    if not path.is_dir():
        raise ToolError(f"不是 workspace 内目录：{arg}")
    return arg


def _truncate(text: str) -> str:
    if len(text) <= _MAX_SHELL_OUTPUT:
        return text
    return text[:_MAX_SHELL_OUTPUT] + "\n...[truncated]"


def _json_result(ok: bool, tool: str, **payload: Any) -> str:
    return json.dumps({"ok": ok, "tool": tool, **payload}, ensure_ascii=False)


def _strict_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    strict = dict(parameters)
    properties = strict.get("properties", {})
    strict["additionalProperties"] = False
    strict["required"] = list(properties)
    return strict
