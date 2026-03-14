"""Agentic loop: send messages, handle tool calls, manage conversation."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from tax_advisor.config import Settings
from tax_advisor.models import chat_completion
from tax_advisor.tools import TOOL_DEFINITIONS, execute_tool


MAX_TOOL_ROUNDS = 10


class Agent:
    """Manages conversation state and the agentic tool-call loop."""

    def __init__(self, settings: Settings, console: Console) -> None:
        self.settings = settings
        self.console = console
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": settings.system_prompt},
        ]

    def clear_history(self) -> None:
        """Reset conversation history, keeping the system prompt."""
        self.messages = [
            {"role": "system", "content": self.settings.system_prompt},
        ]

    def run(self, user_input: str) -> None:
        """Process a user message through the agentic loop.

        Sends the message to the model, handles any tool calls, and streams
        the final response to the console.
        """
        self.messages.append({"role": "user", "content": user_input})

        for _ in range(MAX_TOOL_ROUNDS):
            response = self._get_response(stream=True)
            assistant_message = self._handle_stream(response)
            self.messages.append(assistant_message)

            # If no tool calls, we're done
            tool_calls = assistant_message.get("tool_calls")
            if not tool_calls:
                break

            # Execute each tool call and add results
            for tool_call in tool_calls:
                fn = tool_call["function"]
                self.console.print(
                    f"  [dim]calling tool:[/dim] [cyan]{fn['name']}[/cyan]"
                )
                result = execute_tool(fn["name"], fn["arguments"])
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": result,
                    }
                )
        else:
            self.console.print(
                "[yellow]Warning: reached maximum tool call rounds.[/yellow]"
            )

    def _get_response(self, stream: bool = True) -> Any:
        """Send the current conversation to the model."""
        return chat_completion(
            messages=self.messages,
            model=self.settings.model,
            temperature=self.settings.temperature,
            tools=TOOL_DEFINITIONS,
            stream=stream,
            bedrock_profile=self.settings.bedrock_profile,
            api_key=self.settings.openai_api_key or None,
        )

    def _handle_stream(self, response: Any) -> dict[str, Any]:
        """Consume a streaming response, printing text and collecting tool calls.

        Returns the assembled assistant message dict.
        """
        content_parts: list[str] = []
        tool_calls_by_index: dict[int, dict[str, Any]] = {}

        for chunk in response:
            delta = chunk.choices[0].delta

            # Accumulate text content
            if delta.content:
                # Print tokens as they arrive
                self.console.print(delta.content, end="")
                content_parts.append(delta.content)

            # Accumulate tool calls
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_by_index:
                        tool_calls_by_index[idx] = {
                            "id": tc.id or "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    entry = tool_calls_by_index[idx]
                    if tc.id:
                        entry["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            entry["function"]["name"] += tc.function.name
                        if tc.function.arguments:
                            entry["function"]["arguments"] += tc.function.arguments

        # Newline after streamed content
        if content_parts:
            self.console.print()

        # Build the assistant message
        message: dict[str, Any] = {"role": "assistant"}
        content = "".join(content_parts)
        if content:
            message["content"] = content
        if tool_calls_by_index:
            message["tool_calls"] = [
                tool_calls_by_index[i] for i in sorted(tool_calls_by_index)
            ]
        return message

    def run_non_streaming(self, user_input: str) -> None:
        """Process a user message without streaming (for simpler use cases)."""
        self.messages.append({"role": "user", "content": user_input})

        for _ in range(MAX_TOOL_ROUNDS):
            response = chat_completion(
                messages=self.messages,
                model=self.settings.model,
                temperature=self.settings.temperature,
                tools=TOOL_DEFINITIONS,
                stream=False,
                bedrock_profile=self.settings.bedrock_profile,
                api_key=self.settings.openai_api_key or None,
            )
            choice = response.choices[0]
            assistant_msg = choice.message.model_dump()
            self.messages.append(assistant_msg)

            if not choice.message.tool_calls:
                self.console.print(
                    Panel(Markdown(choice.message.content or ""), title="Assistant")
                )
                break

            for tool_call in choice.message.tool_calls:
                fn = tool_call.function
                self.console.print(
                    f"  [dim]calling tool:[/dim] [cyan]{fn.name}[/cyan]"
                )
                result = execute_tool(fn.name, fn.arguments)
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )
        else:
            self.console.print(
                "[yellow]Warning: reached maximum tool call rounds.[/yellow]"
            )
