from __future__ import annotations

import json

from .agent import MiniClawAgent
from .history import clear_history, load_history
from .llm import ConfigError, DeepSeekClient
from .tools import create_default_registry


MODELS = ("deepseek-chat", "deepseek-reasoner")


def main() -> None:
    print("MiniClaw - Python 智能体作业 2")
    print("输入 /quit 退出，输入 /model 查看或切换模型，输入 /clear 清空历史。\n")

    try:
        client = DeepSeekClient.from_env()
    except ConfigError as exc:
        print(f"配置错误：{exc}")
        return

    history = load_history()
    if history:
        agent = MiniClawAgent(
            client=client,
            tools=create_default_registry(client=client),
            messages=history,
        )
        print(f"已加载历史对话：{len(history)} 条消息。\n")
    else:
        agent = MiniClawAgent(client=client, tools=create_default_registry(client=client))

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
        if user_input == "/clear":
            agent.reset_messages()
            clear_history()
            print("已清空对话历史。\n")
            continue
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
            on_trace=_print_trace,
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


def _print_trace(event: dict) -> None:
    step = event["step"]
    event_type = event["type"]
    if event_type == "thought":
        print(f"\n[Step {step} Thought]")
        print(event["content"])
    elif event_type == "action":
        print(f"\n[Step {step} Action]")
        arguments = json.dumps(event["arguments"], ensure_ascii=False)
        print(f"{event['tool']}({arguments})")
    elif event_type == "observation_start":
        print(f"\n[Step {step} Observation]")
        print(event.get("content") or "开始执行工具。")
    elif event_type == "observation_delta":
        print(event.get("content") or "")
    elif event_type == "observation":
        print(event["content"])
