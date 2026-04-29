from __future__ import annotations

import importlib.util
import ast
import contextvars
import datetime as dt
import html
import inspect
import ipaddress
import json
import math
import re
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urljoin, urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
import pluggy
import serpapi
import trafilatura
from pydantic import ValidationError
from trafilatura.metadata import extract_metadata
from pydantic_ai.tools import RunContext
from pydantic_ai.tools import Tool as PydanticTool

from .settings import SerpApiSettings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_ROOT / "workspace"
PLUGINS_DIR = PROJECT_ROOT / "tools_plugins"

ToolFunction = Callable[..., dict[str, Any]]
ToolEventCallback = Callable[[dict[str, Any]], None]
_LOADED_PLUGIN_FILES: set[Path] = set()
_LOADED_PLUGIN_MODULES: dict[Path, Any] = {}
_TOOL_EVENT_CALLBACK: contextvars.ContextVar[ToolEventCallback | None] = (
    contextvars.ContextVar("miniclaw_tool_event_callback", default=None)
)
hookspec = pluggy.HookspecMarker("miniclaw")
hookimpl = pluggy.HookimplMarker("miniclaw")
_SHELL_COMMANDS = {"pwd", "ls", "cat", "head", "tail", "wc", "rg"}
_SHELL_META_CHARS = {"|", "&", ";", "<", ">", "`", "$", "(", ")", "{", "}"}
_MAX_SHELL_ARGS = 8
_MAX_SHELL_ARG_LENGTH = 200
_MAX_SHELL_FILES = 3
_MAX_RG_PATTERN_LENGTH = 120
_MAX_HEAD_TAIL_LINES = 200
_SHELL_POLICY_VERSION = "read-only-sandbox-v2"
_MAX_SHELL_OUTPUT = 4000
_MAX_FILE_READ_BYTES = 1_000_000
_MAX_WEB_BYTES = 1_500_000
_MAX_WEB_TEXT = 12000
_MAX_FILE_MATCHES = 80
_DEFAULT_SEARCH_RESULTS = 5
_USER_AGENT = "MiniClaw/0.2 (+https://example.local)"


class ToolError(Exception):
    """Raised for safe, user-facing tool errors."""


class PluginSpec:
    @hookspec
    def register_tools(self) -> list["Tool"]:
        """Return plugin tool definitions."""


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

    def as_pydantic_tool(
        self,
        on_trace: Callable[[RunContext[Any], str, dict[str, Any]], None] | None = None,
    ) -> PydanticTool[Any]:
        function = self.function
        original_signature = inspect.signature(function)

        def wrapped(ctx: RunContext[Any], **kwargs: Any) -> dict[str, Any]:
            callback = None
            if on_trace:
                callback = lambda event: on_trace(ctx, self.name, event)
            token = _TOOL_EVENT_CALLBACK.set(callback)
            try:
                return function(**kwargs)
            except TypeError as exc:
                return {"ok": False, "tool": self.name, "error": f"工具参数错误：{exc}"}
            except ToolError as exc:
                return {"ok": False, "tool": self.name, "error": str(exc)}
            except Exception as exc:
                return {"ok": False, "tool": self.name, "error": f"工具执行异常：{exc}"}
            finally:
                _TOOL_EVENT_CALLBACK.reset(token)

        wrapped.__name__ = function.__name__
        wrapped.__doc__ = self.description
        wrapped.__annotations__ = {"ctx": RunContext[Any], **getattr(function, "__annotations__", {})}
        wrapped.__signature__ = original_signature.replace(
            parameters=[
                inspect.Parameter(
                    "ctx",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=RunContext[Any],
                ),
                *original_signature.parameters.values(),
            ]
        )

        return PydanticTool(
            wrapped,
            takes_ctx=True,
            name=self.name,
            description=self.description,
            strict=True,
        )


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def as_pydantic_tools(
        self,
        on_trace: Callable[[RunContext[Any], str, dict[str, Any]], None] | None = None,
    ) -> list[PydanticTool[Any]]:
        return [tool.as_pydantic_tool(on_trace=on_trace) for tool in self._tools.values()]

    def run(
        self,
        name: str,
        arguments: dict[str, Any],
        on_event: ToolEventCallback | None = None,
    ) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return _json_result(ok=False, tool=name, error=f"未知工具：{name}")

        token = _TOOL_EVENT_CALLBACK.set(on_event)
        try:
            result = tool.function(**arguments)
            return _json_result(ok=True, tool=name, **result)
        except TypeError as exc:
            return _json_result(ok=False, tool=name, error=f"工具参数错误：{exc}")
        except ToolError as exc:
            return _json_result(ok=False, tool=name, error=str(exc))
        except Exception as exc:
            return _json_result(ok=False, tool=name, error=f"工具执行异常：{exc}")
        finally:
            _TOOL_EVENT_CALLBACK.reset(token)


def create_default_registry(client: Any | None = None) -> ToolRegistry:
    WORKSPACE_DIR.mkdir(exist_ok=True)

    registry = ToolRegistry()
    register_file_tools(registry, writable=True)
    register_shell_tool(registry)
    register_web_tools(registry)
    register_utility_tools(registry)
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
    registry.register(
        Tool(
            name="list_directory_tree",
            description="递归列出 workspace 安全目录内的目录树，适合快速了解项目结构。",
            parameters={
                "type": "object",
                "properties": {
                    "relative_dir": {
                        "type": "string",
                        "description": "workspace 内的相对目录；列根目录时使用 .",
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "递归深度，建议 1-4；超过限制会自动截断。",
                    },
                },
                "required": ["relative_dir", "max_depth"],
                "additionalProperties": False,
            },
            function=list_directory_tree,
        )
    )
    registry.register(
        Tool(
            name="search_files",
            description="在 workspace 安全目录内用正则或普通文本搜索 UTF-8 文件内容，返回匹配行。",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "要搜索的文本或正则表达式。",
                    },
                    "relative_dir": {
                        "type": "string",
                        "description": "workspace 内搜索目录；搜索根目录时使用 .。",
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "是否区分大小写。",
                    },
                    "max_matches": {
                        "type": "integer",
                        "description": "最多返回多少条匹配，建议 20-80。",
                    },
                },
                "required": ["pattern", "relative_dir", "case_sensitive", "max_matches"],
                "additionalProperties": False,
            },
            function=search_files,
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
        registry.register(
            Tool(
                name="append_text_file",
                description="向 workspace 安全目录内的 UTF-8 文本文件末尾追加内容，不存在则创建。",
                parameters={
                    "type": "object",
                    "properties": {
                        "relative_path": {
                            "type": "string",
                            "description": "workspace 内要追加写入的相对文件路径。",
                        },
                        "content": {
                            "type": "string",
                            "description": "要追加的文本内容。",
                        },
                    },
                    "required": ["relative_path", "content"],
                    "additionalProperties": False,
                },
                function=append_text_file,
            )
        )
        registry.register(
            Tool(
                name="replace_text_in_file",
                description="在 workspace 安全目录内的 UTF-8 文本文件中替换指定文本。",
                parameters={
                    "type": "object",
                    "properties": {
                        "relative_path": {
                            "type": "string",
                            "description": "workspace 内要修改的相对文件路径。",
                        },
                        "old_text": {
                            "type": "string",
                            "description": "要查找并替换的原文本。",
                        },
                        "new_text": {
                            "type": "string",
                            "description": "替换后的新文本。",
                        },
                        "count": {
                            "type": "integer",
                            "description": "最多替换次数；0 表示替换全部。",
                        },
                    },
                    "required": ["relative_path", "old_text", "new_text", "count"],
                    "additionalProperties": False,
                },
                function=replace_text_in_file,
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
            description="将 workspace 文件分析任务委托给只读分析子 Agent。凡是分析、总结、解释、审查文件内容或结构的任务，都应优先使用这个工具。",
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


def register_web_tools(registry: ToolRegistry) -> None:
    registry.register(
        Tool(
            name="search_web",
            description="使用 SerpAPI 的 Google 搜索联网检索公开网页信息，支持限定站点和指定返回数量。",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要搜索的关键词或问题。",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "最多返回多少条结果，建议 3-10。",
                    },
                    "site": {
                        "type": "string",
                        "description": "可选站点限定，例如 python.org；不限定时传空字符串。",
                    },
                },
                "required": ["query", "max_results", "site"],
                "additionalProperties": False,
            },
            function=search_web,
        )
    )
    registry.register(
        Tool(
            name="fetch_web_page",
            description="抓取公开网页 URL，提取标题、正文文本和链接摘要，适合打开搜索结果进一步核验。",
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "要抓取的 http 或 https 公开网页 URL。",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "最多返回多少正文字符，建议 3000-12000。",
                    },
                },
                "required": ["url", "max_chars"],
                "additionalProperties": False,
            },
            function=fetch_web_page,
        )
    )


def register_web_search_tool(registry: ToolRegistry) -> None:
    register_web_tools(registry)


def register_utility_tools(registry: ToolRegistry) -> None:
    registry.register(
        Tool(
            name="get_current_time",
            description="获取指定 IANA 时区的当前日期和时间。",
            parameters={
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "IANA 时区名，例如 Asia/Shanghai、UTC、America/New_York。",
                    }
                },
                "required": ["timezone"],
                "additionalProperties": False,
            },
            function=get_current_time,
        )
    )
    registry.register(
        Tool(
            name="calculate_expression",
            description="安全计算数学表达式，支持四则运算、幂、括号和常见 math 函数。",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "要计算的数学表达式，例如 (2 + 3) * sin(pi / 2)。",
                    }
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
            function=calculate_expression,
        )
    )


def tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
) -> Callable[[ToolFunction], ToolFunction]:
    def decorator(function: ToolFunction) -> ToolFunction:
        return Tool(
            name=name,
            description=description,
            parameters=parameters,
            function=function,
        )

    return decorator


def emit_tool_event(
    kind: str,
    content: str,
    data: dict[str, Any] | None = None,
) -> None:
    callback = _TOOL_EVENT_CALLBACK.get()
    if callback is None:
        return

    callback(
        {
            "type": kind,
            "content": content,
            "data": data or {},
        }
    )


def load_plugins(registry: ToolRegistry, plugins_dir: Path = PLUGINS_DIR) -> None:
    if not plugins_dir.is_dir():
        return

    manager = pluggy.PluginManager("miniclaw")
    manager.add_hookspecs(PluginSpec)

    for plugin_path in sorted(plugins_dir.glob("*.py")):
        if plugin_path.name == "__init__.py":
            continue

        module_name = f"miniclaw_plugin_{plugin_path.stem}"
        if plugin_path in _LOADED_PLUGIN_MODULES:
            module = _LOADED_PLUGIN_MODULES[plugin_path]
        else:
            spec = importlib.util.spec_from_file_location(module_name, plugin_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            _LOADED_PLUGIN_FILES.add(plugin_path)
            _LOADED_PLUGIN_MODULES[plugin_path] = module

        manager.register(module, name=module_name)

    for plugin_tools in manager.hook.register_tools():
        for plugin_tool in plugin_tools or []:
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


def list_directory_tree(relative_dir: str = ".", max_depth: int = 3) -> dict[str, Any]:
    directory = _safe_workspace_path(relative_dir)
    if not directory.exists():
        raise ToolError(f"目录不存在：{relative_dir}")
    if not directory.is_dir():
        raise ToolError(f"不是目录：{relative_dir}")

    depth = _clamp_int(max_depth, minimum=0, maximum=6, default=3)
    entries: list[dict[str, Any]] = []
    base = directory.resolve()

    def walk(current: Path, current_depth: int) -> None:
        if current_depth > depth:
            return

        for child in sorted(current.iterdir(), key=lambda item: item.name.lower()):
            relative_child = child.resolve().relative_to(base).as_posix()
            entries.append(
                {
                    "path": "." if not relative_child else relative_child,
                    "type": "directory" if child.is_dir() else "file",
                    "size": None if child.is_dir() else child.stat().st_size,
                    "depth": current_depth,
                }
            )
            if child.is_dir() and current_depth < depth:
                walk(child, current_depth + 1)

    walk(directory, 0)
    return {"relative_dir": relative_dir, "max_depth": depth, "entries": entries}


def search_files(
    pattern: str,
    relative_dir: str = ".",
    case_sensitive: bool = False,
    max_matches: int = 40,
) -> dict[str, Any]:
    normalized_pattern = pattern.strip()
    if not normalized_pattern:
        raise ToolError("搜索 pattern 不能为空。")

    directory = _safe_workspace_path(relative_dir)
    if not directory.exists():
        raise ToolError(f"目录不存在：{relative_dir}")
    if not directory.is_dir():
        raise ToolError(f"不是目录：{relative_dir}")

    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        compiled = re.compile(normalized_pattern, flags)
    except re.error:
        compiled = re.compile(re.escape(normalized_pattern), flags)

    limit = _clamp_int(max_matches, minimum=1, maximum=_MAX_FILE_MATCHES, default=40)
    matches: list[dict[str, Any]] = []
    skipped_binary_or_large = 0

    for path in sorted(directory.rglob("*"), key=lambda item: item.as_posix().lower()):
        if not path.is_file():
            continue
        if path.stat().st_size > _MAX_FILE_READ_BYTES:
            skipped_binary_or_large += 1
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            skipped_binary_or_large += 1
            continue

        for line_number, line in enumerate(text.splitlines(), start=1):
            if not compiled.search(line):
                continue
            matches.append(
                {
                    "relative_path": path.relative_to(WORKSPACE_DIR).as_posix(),
                    "line": line_number,
                    "text": line[:300],
                }
            )
            if len(matches) >= limit:
                return {
                    "pattern": normalized_pattern,
                    "relative_dir": relative_dir,
                    "matches": matches,
                    "truncated": True,
                    "skipped_binary_or_large": skipped_binary_or_large,
                }

    return {
        "pattern": normalized_pattern,
        "relative_dir": relative_dir,
        "matches": matches,
        "truncated": False,
        "skipped_binary_or_large": skipped_binary_or_large,
    }


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


def append_text_file(relative_path: str, content: str) -> dict[str, Any]:
    path = _safe_workspace_path(relative_path)
    if path.exists() and path.is_dir():
        raise ToolError(f"目标路径是目录，无法追加写入：{relative_path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(content)

    return {
        "relative_path": relative_path,
        "bytes_appended": len(content.encode("utf-8")),
        "message": "文件追加成功。",
    }


def replace_text_in_file(
    relative_path: str,
    old_text: str,
    new_text: str,
    count: int = 0,
) -> dict[str, Any]:
    if not old_text:
        raise ToolError("old_text 不能为空。")

    path = _safe_workspace_path(relative_path)
    if not path.exists():
        raise ToolError(f"文件不存在：{relative_path}")
    if not path.is_file():
        raise ToolError(f"不是文件：{relative_path}")
    if path.stat().st_size > _MAX_FILE_READ_BYTES:
        raise ToolError(f"文件过大，拒绝替换：{relative_path}")

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ToolError(f"文件不是有效 UTF-8 文本：{relative_path}") from exc

    replace_count = 0 if count < 0 else count
    occurrences = content.count(old_text)
    if occurrences == 0:
        return {
            "relative_path": relative_path,
            "replacements": 0,
            "message": "没有找到需要替换的文本。",
        }

    updated = content.replace(old_text, new_text, replace_count)
    path.write_text(updated, encoding="utf-8")
    actual_replacements = occurrences if replace_count == 0 else min(occurrences, replace_count)
    return {
        "relative_path": relative_path,
        "replacements": actual_replacements,
        "message": "文本替换成功。",
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
        "validated": True,
        "policy": _SHELL_POLICY_VERSION,
        "returncode": completed.returncode,
        "stdout": stdout,
        "stderr": stderr,
    }


def search_web(query: str, max_results: int = _DEFAULT_SEARCH_RESULTS, site: str = "") -> dict[str, Any]:
    normalized_query = query.strip()
    if not normalized_query:
        raise ToolError("搜索词不能为空。")

    limit = _clamp_int(
        max_results,
        minimum=1,
        maximum=10,
        default=_DEFAULT_SEARCH_RESULTS,
    )
    site = site.strip()
    search_query = f"site:{site} {normalized_query}" if site else normalized_query
    settings = _load_serpapi_settings()
    params: dict[str, Any] = {
        "engine": "google",
        "q": search_query,
        "num": limit,
        "google_domain": settings.google_domain,
        "hl": settings.hl,
        "gl": settings.gl,
    }
    if settings.location:
        params["location"] = settings.location

    emit_tool_event(
        "observation_delta",
        "正在通过 SerpAPI 搜索 Google...",
        {"provider": "serpapi", "engine": "google", "status": "started"},
    )
    client = serpapi.Client(api_key=settings.api_key, timeout=settings.timeout)
    try:
        response = client.search(params)
    except serpapi.HTTPError as exc:
        raise ToolError(f"SerpAPI 搜索失败：HTTP {exc.status_code} {exc.error}") from exc
    except serpapi.TimeoutError as exc:
        raise ToolError(f"SerpAPI 搜索超时：{exc}") from exc
    except Exception as exc:
        raise ToolError(f"SerpAPI 搜索失败：{exc}") from exc

    response_error = _clean_search_text(str(response.get("error") or ""))
    if response_error:
        raise ToolError(f"SerpAPI 搜索失败：{response_error}")

    organic_results = response.get("organic_results") or []
    results: list[dict[str, str]] = []
    for item in organic_results:
        title = _clean_search_text(item.get("title"))
        result_url = str(item.get("link") or "").strip()
        snippet = _clean_search_text(item.get("snippet"))
        displayed_link = _clean_search_text(item.get("displayed_link"))
        if not title or not result_url:
            continue
        results.append(
            {
                "title": title,
                "url": result_url,
                "snippet": snippet,
                "source": displayed_link or "google",
            }
        )

    results = _dedupe_search_results(results)
    if site:
        results = [result for result in results if _url_matches_site(result.get("url", ""), site)]
    results = results[:limit]

    emit_tool_event(
        "observation_delta",
        f"SerpAPI 返回 {len(results)} 条结果。",
        {"provider": "serpapi", "engine": "google", "status": "ok", "count": len(results)},
    )

    if not results:
        raise ToolError("没有找到搜索结果。")

    return {
        "query": normalized_query,
        "site": site,
        "provider": "serpapi",
        "engines": ["google"],
        "search_metadata": {
            "id": str((response.get("search_metadata") or {}).get("id") or ""),
        },
        "results": results,
    }


def fetch_web_page(url: str, max_chars: int = _MAX_WEB_TEXT) -> dict[str, Any]:
    safe_url = _validate_public_url(url)
    limit = _clamp_int(max_chars, minimum=500, maximum=_MAX_WEB_TEXT, default=6000)
    headers = {
        "User-Agent": _USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,text/plain;q=0.8,*/*;q=0.5",
    }

    try:
        final_url, content_type, payload = _fetch_public_payload(
            safe_url,
            headers=headers,
            timeout=12,
        )
    except httpx.HTTPStatusError as exc:
        raise ToolError(f"网页抓取失败：HTTP {exc.response.status_code}") from exc
    except httpx.HTTPError as exc:
        raise ToolError(f"网页抓取失败：{exc}") from exc

    encoding = _guess_encoding(content_type)
    text = payload.decode(encoding, errors="replace")
    metadata = extract_metadata(text, default_url=final_url)
    title = _clean_search_text(getattr(metadata, "title", None))
    body_text = (
        trafilatura.extract(
            text,
            url=final_url,
            include_comments=False,
            include_links=False,
            include_images=False,
            include_formatting=False,
            output_format="txt",
        )
        or ""
    )
    links = _extract_links(text, final_url)
    cleaned_text = _clean_search_text(body_text)
    truncated = len(cleaned_text) > limit

    return {
        "url": safe_url,
        "final_url": final_url,
        "content_type": content_type,
        "title": title,
        "text": cleaned_text[:limit],
        "links": links[:20],
        "truncated": truncated,
    }


def get_current_time(timezone: str = "Asia/Shanghai") -> dict[str, Any]:
    timezone_name = timezone.strip() or "Asia/Shanghai"
    try:
        tz = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise ToolError(f"未知时区：{timezone_name}") from exc

    now = dt.datetime.now(tz)
    return {
        "timezone": timezone_name,
        "iso": now.isoformat(timespec="seconds"),
        "date": now.date().isoformat(),
        "time": now.strftime("%H:%M:%S"),
        "weekday": now.strftime("%A"),
        "utc_offset": now.strftime("%z"),
    }


def calculate_expression(expression: str) -> dict[str, Any]:
    normalized_expression = expression.strip()
    if not normalized_expression:
        raise ToolError("表达式不能为空。")
    if len(normalized_expression) > 500:
        raise ToolError("表达式过长。")

    try:
        tree = ast.parse(normalized_expression, mode="eval")
        value = _eval_math_node(tree.body)
    except (SyntaxError, ValueError, TypeError, OverflowError) as exc:
        raise ToolError(f"表达式无法计算：{exc}") from exc

    if isinstance(value, float) and not math.isfinite(value):
        raise ToolError("计算结果不是有限数。")

    return {"expression": normalized_expression, "result": value}


def _load_serpapi_settings() -> SerpApiSettings:
    try:
        return SerpApiSettings()
    except ValidationError as exc:
        raise ToolError(
            "请先设置 SERPAPI_KEY。你可以在项目根目录 .env 中写入："
            "SERPAPI_KEY=你的 key"
        ) from exc


def _dedupe_search_results(results: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen: set[str] = set()

    for result in results:
        url = result.get("url", "").strip()
        key = url.split("#", 1)[0].rstrip("/")
        if not key or key in seen:
            continue

        seen.add(key)
        deduped.append(result)

    return deduped


def _url_matches_site(url: str, site: str) -> bool:
    hostname = urlparse(url).hostname
    if not hostname:
        return False

    normalized_hostname = hostname.lower()
    normalized_site = site.strip().lower().lstrip(".")
    return normalized_hostname == normalized_site or normalized_hostname.endswith(
        f".{normalized_site}"
    )


def _validate_public_url(url: str) -> str:
    normalized = url.strip()
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"}:
        raise ToolError("只允许抓取 http 或 https URL。")
    if not parsed.hostname:
        raise ToolError("URL 缺少主机名。")

    hostname = parsed.hostname.lower()
    if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(".local"):
        raise ToolError("安全限制：不允许抓取本机或本地域名。")

    try:
        addresses = socket.getaddrinfo(hostname, parsed.port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ToolError(f"无法解析 URL 主机：{hostname}") from exc

    for address in addresses:
        ip_address = ipaddress.ip_address(address[4][0])
        if (
            ip_address.is_private
            or ip_address.is_loopback
            or ip_address.is_link_local
            or ip_address.is_multicast
            or ip_address.is_reserved
            or ip_address.is_unspecified
        ):
            raise ToolError("安全限制：不允许抓取本地、内网或保留地址。")

    return normalized


def _guess_encoding(content_type: str) -> str:
    match = re.search(r"charset=([\w.-]+)", content_type, re.IGNORECASE)
    if match:
        return match.group(1)
    return "utf-8"


def _extract_links(page: str, base_url: str) -> list[dict[str, str]]:
    links: list[dict[str, str]] = []
    for match in re.finditer(
        r'<a\s+[^>]*href=["\'](?P<href>[^"\']+)["\'][^>]*>(?P<label>.*?)</a>',
        page,
        re.DOTALL | re.IGNORECASE,
    ):
        href = html.unescape(match.group("href")).strip()
        label = _clean_search_text(match.group("label"))
        if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue
        links.append({"text": label[:120], "url": urljoin(base_url, href)})
        if len(links) >= 20:
            break

    return links


def _clamp_int(value: int, minimum: int, maximum: int, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


_MATH_FUNCTIONS: dict[str, Callable[..., Any]] = {
    name: value
    for name, value in vars(math).items()
    if callable(value) and not name.startswith("_")
}
_MATH_FUNCTIONS.update({"abs": abs, "round": round})
_MATH_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}


def _eval_math_node(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("只允许数字常量。")

    if isinstance(node, ast.Name):
        if node.id in _MATH_CONSTANTS:
            return _MATH_CONSTANTS[node.id]
        raise ValueError(f"未知名称：{node.id}")

    if isinstance(node, ast.UnaryOp):
        value = _eval_math_node(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +value
        if isinstance(node.op, ast.USub):
            return -value
        raise ValueError("不支持的单目运算。")

    if isinstance(node, ast.BinOp):
        left = _eval_math_node(node.left)
        right = _eval_math_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.Pow):
            if abs(right) > 1000:
                raise ValueError("幂指数过大。")
            return left**right
        raise ValueError("不支持的二元运算。")

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("只允许调用白名单数学函数。")
        function = _MATH_FUNCTIONS.get(node.func.id)
        if function is None:
            raise ValueError(f"不允许的函数：{node.func.id}")
        if node.keywords:
            raise ValueError("不支持关键字参数。")
        args = [_eval_math_node(arg) for arg in node.args]
        return function(*args)

    raise ValueError("表达式包含不允许的语法。")


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

        emit_tool_event(
            "delegation_start",
            f"主 Agent 准备把文件分析任务委托给子 Agent：{relative_path}",
            {
                "relative_path": relative_path,
                "task": task,
                "status": "delegation_started",
                "agent_role": "main",
                "agent_name": "MiniClaw",
            },
        )
        subagent = AnalysisSubAgent(client)
        analysis = subagent.analyze(
            relative_path=relative_path,
            task=task,
            on_event=lambda event: emit_tool_event(
                str(event.get("type") or "delegation_progress"),
                str(event.get("content") or ""),
                event.get("data") or {},
            ),
        )
        emit_tool_event(
            "delegation_result",
            "子 Agent 已返回分析结果，主 Agent 恢复处理。",
            {
                "relative_path": relative_path,
                "task": task,
                "status": "delegation_completed",
                "agent_role": "main",
                "agent_name": "MiniClaw",
                "subagent_result": analysis,
            },
        )
        return {
            "relative_path": relative_path,
            "task": task,
            "analysis": analysis,
            "agent_role": "sub",
            "agent_name": "AnalysisSubAgent",
            "phase": "subagent_result",
        }

    return delegate_file_analysis


def _validate_shell_command(command: str, args: list[str]) -> list[str]:
    if not isinstance(command, str):
        raise ToolError("command 必须是字符串。")
    _reject_shell_syntax(command, field_name="command")
    if command not in _SHELL_COMMANDS:
        raise ToolError(f"不允许执行命令：{command}")
    if not isinstance(args, list):
        raise ToolError("args 必须是字符串数组。")
    if len(args) > _MAX_SHELL_ARGS:
        raise ToolError(f"参数数量超限：最多允许 {_MAX_SHELL_ARGS} 个参数。")

    for arg in args:
        if not isinstance(arg, str):
            raise ToolError("args 必须是字符串数组。")
        _reject_shell_syntax(arg, field_name="参数")

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


def _reject_shell_syntax(value: str, field_name: str = "shell 参数") -> None:
    if value != value.strip():
        raise ToolError(f"{field_name} 不允许包含前后空白。")
    if not value:
        raise ToolError(f"{field_name} 不能为空。")
    if len(value) > _MAX_SHELL_ARG_LENGTH:
        raise ToolError(f"{field_name} 过长，最多允许 {_MAX_SHELL_ARG_LENGTH} 个字符。")
    if "\n" in value or "\x00" in value:
        raise ToolError(f"{field_name} 不允许包含换行或 NUL 字符。")
    if any(char in value for char in _SHELL_META_CHARS):
        raise ToolError(f"{field_name} 包含不允许的语法字符：{value}")


def _validate_pwd_args(args: list[str]) -> list[str]:
    if args:
        raise ToolError("pwd 不允许携带参数。")
    return []


def _validate_ls_args(args: list[str]) -> list[str]:
    allowed_flags = {"-l", "-a", "-la", "-al"}
    result: list[str] = []
    path_count = 0
    for arg in args:
        if arg.startswith("-"):
            if arg not in allowed_flags:
                raise ToolError(f"ls 不允许的参数：{arg}")
            result.append(arg)
            continue
        path_count += 1
        if path_count > 1:
            raise ToolError("ls 最多只允许一个目录路径。")
        result.append(_validate_dir_arg(arg))
    return result


def _validate_cat_args(args: list[str]) -> list[str]:
    if not args:
        raise ToolError("cat 需要至少一个 workspace 内文件路径。")
    if len(args) > _MAX_SHELL_FILES:
        raise ToolError(f"cat 最多只允许 {_MAX_SHELL_FILES} 个文件。")
    return [_validate_file_arg(arg, command_name="cat") for arg in args]


def _validate_head_tail_args(args: list[str]) -> list[str]:
    result: list[str] = []
    index = 0
    if len(args) >= 2 and args[0] == "-n":
        if not args[1].isdigit():
            raise ToolError("-n 后必须是正整数。")
        line_count = int(args[1])
        if line_count < 1 or line_count > _MAX_HEAD_TAIL_LINES:
            raise ToolError(f"-n 取值超限：只允许 1 到 {_MAX_HEAD_TAIL_LINES}。")
        result.extend(args[:2])
        index = 2
    elif args and args[0].startswith("-"):
        raise ToolError("head/tail 仅允许可选参数 -n <数字>。")

    paths = args[index:]
    if not paths:
        raise ToolError("head/tail 需要至少一个 workspace 内文件路径。")
    if len(paths) > _MAX_SHELL_FILES:
        raise ToolError(f"head/tail 最多只允许 {_MAX_SHELL_FILES} 个文件。")
    result.extend(_validate_file_arg(path, command_name="head/tail") for path in paths)
    return result


def _validate_wc_args(args: list[str]) -> list[str]:
    allowed_flags = {"-l", "-w", "-c"}
    files = _validate_flags_and_paths(args, allowed_flags, allow_directory=False, command_name="wc")
    path_count = len([arg for arg in files if not arg.startswith("-")])
    if path_count == 0:
        raise ToolError("wc 需要至少一个 workspace 内文件路径。")
    if path_count > _MAX_SHELL_FILES:
        raise ToolError(f"wc 最多只允许 {_MAX_SHELL_FILES} 个文件。")
    return files


def _validate_rg_args(args: list[str]) -> list[str]:
    flags: list[str] = []
    rest = list(args)
    while rest and rest[0] in {"-n", "-i"}:
        flags.append(rest.pop(0))

    if not rest:
        raise ToolError("rg 需要搜索 pattern。")

    pattern = rest.pop(0)
    if not pattern.strip():
        raise ToolError("rg 的 pattern 不能为空。")
    if len(pattern) > _MAX_RG_PATTERN_LENGTH:
        raise ToolError(f"rg 的 pattern 过长，最多允许 {_MAX_RG_PATTERN_LENGTH} 个字符。")
    if rest:
        if len(rest) > 1:
            raise ToolError("rg 只允许一个可选搜索目录。")
        return [*flags, pattern, _validate_dir_arg(rest[0])]

    return [*flags, pattern]


def _validate_flags_and_paths(
    args: list[str],
    allowed_flags: set[str],
    allow_directory: bool,
    command_name: str,
) -> list[str]:
    result: list[str] = []
    for arg in args:
        if arg.startswith("-"):
            if arg not in allowed_flags:
                raise ToolError(f"{command_name} 不允许的参数：{arg}")
            result.append(arg)
            continue

        result.append(
            _validate_dir_arg(arg)
            if allow_directory
            else _validate_file_arg(arg, command_name=command_name)
        )
    return result


def _validate_file_arg(arg: str, command_name: str = "命令") -> str:
    if arg.startswith("-"):
        raise ToolError(f"{command_name} 不允许使用以 - 开头的伪路径参数：{arg}")
    path = _safe_workspace_path(arg)
    if not path.is_file():
        raise ToolError(f"不是 workspace 内文件：{arg}")
    if path.stat().st_size > _MAX_FILE_READ_BYTES:
        raise ToolError(f"文件过大，拒绝通过 {command_name} 读取：{arg}")
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


def _clean_search_text(text: str | None) -> str:
    if not text:
        return ""

    normalized = html.unescape(text)
    normalized = re.sub(r"<[^>]+>", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _strict_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    strict = dict(parameters)
    properties = strict.get("properties", {})
    strict["additionalProperties"] = False
    strict["required"] = list(properties)
    return strict


def _fetch_public_payload(
    url: str,
    headers: dict[str, str],
    timeout: int,
    max_bytes: int | None = None,
) -> tuple[str, str, bytes]:
    current_url = url
    redirect_count = 0

    with httpx.Client(timeout=timeout, follow_redirects=False) as client:
        while True:
            _validate_public_url(current_url)
            with client.stream("GET", current_url, headers=headers) as response:
                response.raise_for_status()

                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise ToolError("重定向响应缺少 location。")
                    current_url = urljoin(str(response.url), location)
                    redirect_count += 1
                    if redirect_count > 5:
                        raise ToolError("重定向次数过多。")
                    continue

                content_type = response.headers.get("Content-Type", "")
                limit = _MAX_WEB_BYTES if max_bytes is None else max_bytes
                payload = bytearray()
                for chunk in response.iter_bytes():
                    payload.extend(chunk)
                    if len(payload) > limit:
                        raise ToolError("响应内容过大。")
                final_url = str(response.url)
                _validate_public_url(final_url)
                return final_url, content_type, bytes(payload)
