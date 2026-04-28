from __future__ import annotations

import importlib.util
import ast
import contextvars
import datetime as dt
import html
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
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote_plus, unquote, urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from xml.etree import ElementTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_ROOT / "workspace"
PLUGINS_DIR = PROJECT_ROOT / "tools_plugins"

ToolFunction = Callable[..., dict[str, Any]]
ToolEventCallback = Callable[[dict[str, Any]], None]
_PLUGIN_TOOLS: list["Tool"] = []
_LOADED_PLUGIN_FILES: set[Path] = set()
_TOOL_EVENT_CALLBACK: contextvars.ContextVar[ToolEventCallback | None] = (
    contextvars.ContextVar("miniclaw_tool_event_callback", default=None)
)
_SHELL_COMMANDS = {"pwd", "ls", "cat", "head", "tail", "wc", "rg"}
_SHELL_META_CHARS = {"|", "&", ";", "<", ">", "`", "$", "(", ")", "{", "}"}
_MAX_SHELL_OUTPUT = 4000
_MAX_FILE_READ_BYTES = 1_000_000
_MAX_WEB_BYTES = 1_500_000
_MAX_WEB_TEXT = 12000
_MAX_FILE_MATCHES = 80
_DEFAULT_SEARCH_RESULTS = 5
_USER_AGENT = "MiniClaw/0.2 (+https://example.local)"


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


def register_web_tools(registry: ToolRegistry) -> None:
    registry.register(
        Tool(
            name="search_web",
            description="使用 Bing、Baidu、Google 联网搜索公开网页信息，支持限定站点和指定返回数量。",
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

    engines: list[tuple[str, Callable[[str, int], list[dict[str, str]]]]] = [
        ("bing", _search_bing),
        ("baidu", _search_baidu),
        ("google", _search_google),
    ]
    all_results: list[dict[str, str]] = []
    errors: dict[str, str] = {}

    for engine_name, search_function in engines:
        emit_tool_event(
            "observation_delta",
            f"正在搜索 {engine_name}...",
            {"engine": engine_name, "status": "started"},
        )
        try:
            engine_results = search_function(search_query, limit)
        except Exception as exc:
            errors[engine_name] = str(exc)
            emit_tool_event(
                "observation_delta",
                f"{engine_name} 搜索失败：{exc}",
                {"engine": engine_name, "status": "error", "error": str(exc)},
            )
            continue

        all_results.extend(engine_results)
        emit_tool_event(
            "observation_delta",
            f"{engine_name} 返回 {len(engine_results)} 条候选结果。",
            {"engine": engine_name, "status": "ok", "count": len(engine_results)},
        )

    results = _dedupe_search_results(
        all_results
    )
    if site:
        results = [result for result in results if _url_matches_site(result.get("url", ""), site)]
    results = results[:limit]

    if not results:
        if errors:
            raise ToolError(f"没有找到搜索结果。搜索错误：{errors}")
        raise ToolError("没有找到搜索结果。")

    return {
        "query": normalized_query,
        "site": site,
        "engines": [engine_name for engine_name, _ in engines],
        "errors": errors,
        "results": results,
    }


def fetch_web_page(url: str, max_chars: int = _MAX_WEB_TEXT) -> dict[str, Any]:
    safe_url = _validate_public_url(url)
    limit = _clamp_int(max_chars, minimum=500, maximum=_MAX_WEB_TEXT, default=6000)
    request = Request(
        safe_url,
        headers={
            "User-Agent": _USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.8,*/*;q=0.5",
        },
    )

    try:
        opener = build_opener(_PublicRedirectHandler)
        with opener.open(request, timeout=12) as response:
            content_type = response.headers.get("Content-Type", "")
            payload = response.read(_MAX_WEB_BYTES + 1)
            final_url = response.geturl()
    except HTTPError as exc:
        raise ToolError(f"网页抓取失败：HTTP {exc.code}") from exc
    except URLError as exc:
        raise ToolError(f"网页抓取失败：{exc.reason}") from exc
    except TimeoutError as exc:
        raise ToolError("网页抓取超时，请稍后重试。") from exc

    if len(payload) > _MAX_WEB_BYTES:
        raise ToolError("网页过大，已拒绝抓取。")

    encoding = _guess_encoding(content_type)
    text = payload.decode(encoding, errors="replace")
    title, body_text, links = _extract_web_text(text, final_url)
    truncated = len(body_text) > limit

    return {
        "url": safe_url,
        "final_url": final_url,
        "content_type": content_type,
        "title": title,
        "text": body_text[:limit],
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


def _read_search_response(
    request: Request,
    timeout: int = 10,
    max_bytes: int | None = None,
    attempts: int = 3,
) -> bytes:
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            with urlopen(request, timeout=timeout) as response:
                if max_bytes is None:
                    return response.read()
                return response.read(max_bytes)
        except HTTPError:
            raise
        except (URLError, TimeoutError, OSError) as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise ToolError("搜索请求失败。")


def _search_bing(query: str, limit: int) -> list[dict[str, str]]:
    url = f"https://www.bing.com/search?cc=us&q={quote_plus(query)}&format=rss"
    request = Request(
        url,
        headers={
            "User-Agent": _USER_AGENT,
            "Accept": "application/rss+xml, application/xml;q=0.9, text/xml;q=0.8",
        },
    )

    try:
        payload = _read_search_response(request)
    except HTTPError as exc:
        raise ToolError(f"Bing 搜索失败：HTTP {exc.code}") from exc
    except URLError as exc:
        raise ToolError(f"Bing 搜索失败：{exc.reason}") from exc
    except OSError as exc:
        raise ToolError(f"Bing 搜索失败：{exc}") from exc
    except TimeoutError as exc:
        raise ToolError("Bing 搜索超时。") from exc

    try:
        root = ElementTree.fromstring(payload)
    except ElementTree.ParseError as exc:
        raise ToolError("Bing 返回了无法解析的结果。") from exc

    results = []
    for item in root.findall("./channel/item")[:limit]:
        title = _clean_search_text(item.findtext("title"))
        link = (item.findtext("link") or "").strip()
        snippet = _clean_search_text(item.findtext("description"))

        if not title and not link:
            continue

        results.append(
            {
                "title": title,
                "url": link,
                "snippet": snippet,
                "source": "bing",
            }
        )

    return results


def _search_baidu(query: str, limit: int) -> list[dict[str, str]]:
    url = f"https://www.baidu.com/s?wd={quote_plus(query)}&rn={limit}"
    request = Request(
        url,
        headers={
            "User-Agent": _USER_AGENT,
            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.5",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.6",
        },
    )

    try:
        html_payload = _read_search_response(request, max_bytes=_MAX_WEB_BYTES)
    except HTTPError as exc:
        raise ToolError(f"Baidu 搜索失败：HTTP {exc.code}") from exc
    except URLError as exc:
        raise ToolError(f"Baidu 搜索失败：{exc.reason}") from exc
    except OSError as exc:
        raise ToolError(f"Baidu 搜索失败：{exc}") from exc
    except TimeoutError as exc:
        raise ToolError("Baidu 搜索超时。") from exc

    page = html_payload.decode("utf-8", errors="replace")
    if "百度安全验证" in page or "网络不给力" in page:
        raise ToolError("Baidu 返回了安全验证或临时不可用页面。")

    results: list[dict[str, str]] = []
    blocks = re.split(r'<div[^>]+class="[^"]*\bresult\b[^"]*"', page, flags=re.IGNORECASE)

    for block in blocks[1:]:
        link_match = re.search(
            r'<h3[^>]*>.*?<a[^>]+href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>.*?</h3>',
            block,
            re.DOTALL | re.IGNORECASE,
        )
        if not link_match:
            continue

        snippet_match = re.search(
            r'<div[^>]+class="[^"]*(?:c-abstract|content-right|result-op)[^"]*"[^>]*>'
            r"(?P<snippet>.*?)</div>",
            block,
            re.DOTALL | re.IGNORECASE,
        )
        result_url = html.unescape(link_match.group("url"))
        if result_url.startswith("/"):
            result_url = urljoin("https://www.baidu.com", result_url)
        if "baidu.com/link" in result_url:
            result_url = _resolve_search_redirect(result_url)

        title = _clean_search_text(link_match.group("title"))
        snippet = _clean_search_text(snippet_match.group("snippet")) if snippet_match else ""
        if title and result_url:
            results.append(
                {
                    "title": title,
                    "url": result_url,
                    "snippet": snippet,
                    "source": "baidu",
                }
            )
        if len(results) >= limit:
            break

    return results


def _search_google(query: str, limit: int) -> list[dict[str, str]]:
    url = f"https://www.google.com/search?q={quote_plus(query)}&num={limit}&hl=zh-CN"
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "Chrome/121.0 Safari/537.36 MiniClaw/0.2"
            ),
            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.5",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.6",
        },
    )

    try:
        html_payload = _read_search_response(request, max_bytes=_MAX_WEB_BYTES)
    except HTTPError as exc:
        raise ToolError(f"Google 搜索失败：HTTP {exc.code}") from exc
    except URLError as exc:
        raise ToolError(f"Google 搜索失败：{exc.reason}") from exc
    except OSError as exc:
        raise ToolError(f"Google 搜索失败：{exc}") from exc
    except TimeoutError as exc:
        raise ToolError("Google 搜索超时。") from exc

    page = html_payload.decode("utf-8", errors="replace")
    results: list[dict[str, str]] = []

    for match in re.finditer(
        r'<a[^>]+href="(?P<href>/url\?q=[^"]+|https?://[^"]+)"[^>]*>.*?'
        r"<h3[^>]*>(?P<title>.*?)</h3>",
        page,
        re.DOTALL | re.IGNORECASE,
    ):
        result_url = _decode_google_url(html.unescape(match.group("href")))
        if not result_url or "google.com" in (urlparse(result_url).hostname or ""):
            continue
        title = _clean_search_text(match.group("title"))
        if not title:
            continue

        snippet = _google_snippet_after(page, match.end())
        results.append(
            {
                "title": title,
                "url": result_url,
                "snippet": snippet,
                "source": "google",
            }
        )
        if len(results) >= limit:
            break

    return results


def _decode_google_url(href: str) -> str:
    if href.startswith("/url?"):
        params = parse_qs(urlparse(href).query)
        return unquote(params.get("q", [""])[0])
    return href


def _google_snippet_after(page: str, start_index: int) -> str:
    window = page[start_index : start_index + 1800]
    candidates = re.findall(
        r'<div[^>]+(?:data-sncf|class="[^"]*(?:VwiC3b|IsZvec|GI74Re)[^"]*")[^>]*>'
        r"(?P<snippet>.*?)</div>",
        window,
        re.DOTALL | re.IGNORECASE,
    )
    for candidate in candidates:
        snippet = _clean_search_text(candidate)
        if snippet:
            return snippet[:500]
    return ""


def _resolve_search_redirect(url: str) -> str:
    try:
        _validate_public_url(url)
        request = Request(url, headers={"User-Agent": _USER_AGENT, "Accept": "*/*"})
        opener = build_opener(_PublicRedirectHandler)
        with opener.open(request, timeout=5) as response:
            final_url = response.geturl()
    except (HTTPError, URLError, TimeoutError, ToolError):
        return url

    return final_url or url


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


def _extract_web_text(page: str, base_url: str) -> tuple[str, str, list[dict[str, str]]]:
    title_match = re.search(r"<title[^>]*>(.*?)</title>", page, re.DOTALL | re.IGNORECASE)
    title = _clean_search_text(title_match.group(1)) if title_match else ""

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

    text = re.sub(r"<(script|style|noscript|svg|canvas)[^>]*>.*?</\1>", " ", page, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = _clean_search_text(text)
    return title, text, links


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


class _PublicRedirectHandler(HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Any,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Request | None:
        _validate_public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


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
