from __future__ import annotations

from miniclaw.tools import hookimpl, tool


@hookimpl
def register_tools() -> list[object]:
    # 最小插件示例：声明一个工具并通过 hook 返回给主注册表加载。
    @tool(
        name="echo_text",
        description="返回用户传入的一段文本，用于演示插件工具。",
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
