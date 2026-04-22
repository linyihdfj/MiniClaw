from __future__ import annotations

from .agent import MiniClawAgent
from .llm import ConfigError, DeepSeekClient
from .tools import create_default_registry


MODELS = ("deepseek-chat", "deepseek-reasoner")


def main() -> None:
    print("MiniClaw - Python 智能体作业 2")
    print("输入 /quit 退出，输入 /model 查看或切换模型。\n")

    try:
        client = DeepSeekClient.from_env()
    except ConfigError as exc:
        print(f"配置错误：{exc}")
        return

    agent = MiniClawAgent(client=client, tools=create_default_registry())

    while True:
        try:
            user_input = input("User> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            return

        if not user_input:
            continue
        if user_input == "/quit":
            print("再见。")
            return
        if user_input == "/model" or user_input.startswith("/model "):
            _handle_model_command(client, user_input)
            continue

        printed_reasoning = False
        printed_content = False

        def print_reasoning(content: str) -> None:
            nonlocal printed_reasoning
            if not printed_reasoning:
                print("Reasoning> ", end="", flush=True)
                printed_reasoning = True
            print(content, end="", flush=True)

        def print_content(content: str) -> None:
            nonlocal printed_content
            if not printed_content:
                if printed_reasoning:
                    print("\nMiniClaw> ", end="", flush=True)
                else:
                    print("MiniClaw> ", end="", flush=True)
                printed_content = True
            print(content, end="", flush=True)

        answer = agent.run_turn(
            user_input,
            on_content_delta=print_content,
            on_reasoning_delta=print_reasoning,
        )
        if answer:
            if not printed_content:
                if printed_reasoning:
                    print("\nMiniClaw> ", end="")
                else:
                    print("MiniClaw> ", end="")
            print(answer, end="")
        print("\n")


def _handle_model_command(client: DeepSeekClient, command: str) -> None:
    parts = command.split()
    if len(parts) == 1:
        print(f"当前模型：{client.model}")
        print(f"可选模型：{', '.join(MODELS)}\n")
        return

    model = parts[1]
    if model not in MODELS:
        print(f"未知模型：{model}")
        print(f"可选模型：{', '.join(MODELS)}\n")
        return

    client.model = model
    print(f"已切换模型：{client.model}\n")
