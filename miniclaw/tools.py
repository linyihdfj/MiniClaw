from __future__ import annotations

import importlib.util
import json
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


def create_default_registry() -> ToolRegistry:
    WORKSPACE_DIR.mkdir(exist_ok=True)

    registry = ToolRegistry()
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
    load_plugins(registry)
    return registry


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


def _json_result(ok: bool, tool: str, **payload: Any) -> str:
    return json.dumps({"ok": ok, "tool": tool, **payload}, ensure_ascii=False)


def _strict_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    strict = dict(parameters)
    properties = strict.get("properties", {})
    strict["additionalProperties"] = False
    strict["required"] = list(properties)
    return strict
