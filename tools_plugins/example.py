from __future__ import annotations

from miniclaw.tools import tool


@tool(
    name="echo_text",
    description="返回用户传入的一段文本，用于演示装饰器插件工具。",
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
