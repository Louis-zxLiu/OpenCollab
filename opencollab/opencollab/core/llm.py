"""Minimal LLM interface — thin wrapper over OpenAI-compatible API.

Supports any OpenAI-compatible provider (OpenAI, DeepSeek, local models)
and Anthropic natively. No custom message format — uses standard dicts.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import openai

# ---------------------------------------------------------------------------
# Response containers
# ---------------------------------------------------------------------------


@dataclass
class LLMResponse:
    """Single LLM completion result."""

    content: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: Usage = field(default_factory=lambda: Usage())
    finish_reason: str | None = None
    raw: Any = None


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class StreamDelta:
    """A single chunk from streaming response."""

    content: str | None = None
    tool_call_index: int | None = None
    tool_call_id: str | None = None
    tool_call_name: str | None = None
    tool_call_args_delta: str | None = None
    finish_reason: str | None = None


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------


class LLMClient:
    """Provider-agnostic LLM client. Uses OpenAI SDK which works with any
    compatible endpoint (OpenAI, DeepSeek, Together, Ollama, vLLM, etc.).

    For Anthropic: set base_url="https://api.anthropic.com/v1" and use
    anthropic-compatible proxy, or use the dedicated Anthropic SDK path.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        provider: str = "openai",
    ):
        self.model = model
        self.provider = provider

        if provider == "anthropic":
            import anthropic

            self._anthropic = anthropic.AsyncAnthropic(
                api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            )
            self._openai = None
        else:
            self._openai = openai.AsyncOpenAI(
                api_key=api_key or os.environ.get("OPENAI_API_KEY"),
                base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
            )
            self._anthropic = None

    # ---- Non-streaming completion ----

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Single-shot completion. Returns full response."""
        start = time.monotonic()
        if self._anthropic:
            return await self._complete_anthropic(messages, tools, temperature)
        return await self._complete_openai(messages, tools, temperature)

    async def _complete_openai(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        temperature: float,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        resp = await self._openai.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message

        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                })

        return LLMResponse(
            content=msg.content,
            tool_calls=tool_calls,
            usage=Usage(
                input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
                output_tokens=resp.usage.completion_tokens if resp.usage else 0,
            ),
            finish_reason=choice.finish_reason,
            raw=resp,
        )

    async def _complete_anthropic(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        temperature: float,
    ) -> LLMResponse:
        # Extract system from messages
        system_parts = []
        conv_messages = []
        for m in messages:
            if m["role"] == "system":
                system_parts.append(m["content"])
            else:
                conv_messages.append(m)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": conv_messages,
            "max_tokens": 8192,
            "temperature": temperature,
        }
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)

        if tools:
            # Convert OpenAI tool format to Anthropic format
            anthropic_tools = []
            for t in tools:
                func = t["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                })
            kwargs["tools"] = anthropic_tools

        resp = await self._anthropic.messages.create(**kwargs)

        content = ""
        tool_calls = []
        for block in resp.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                import json

                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {"name": block.name, "arguments": json.dumps(block.input)},
                })

        return LLMResponse(
            content=content or None,
            tool_calls=tool_calls,
            usage=Usage(
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
            ),
            finish_reason=resp.stop_reason,
            raw=resp,
        )

    # ---- Streaming completion (OpenAI path only for now) ----

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
    ) -> AsyncIterator[StreamDelta]:
        """Streaming completion. Yields deltas."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        if self._anthropic:
            async for delta in self._stream_anthropic(messages, tools, temperature):
                yield delta
            return

        stream = await self._openai.chat.completions.create(**kwargs)
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            finish = chunk.choices[0].finish_reason

            # Text content
            if delta.content:
                yield StreamDelta(content=delta.content)

            # Tool calls
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    yield StreamDelta(
                        tool_call_index=tc.index,
                        tool_call_id=tc.id,
                        tool_call_name=tc.function.name if tc.function and tc.function.name else None,
                        tool_call_args_delta=tc.function.arguments if tc.function else None,
                    )

            if finish:
                yield StreamDelta(finish_reason=finish)

    async def _stream_anthropic(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        temperature: float,
    ) -> AsyncIterator[StreamDelta]:
        system_parts = []
        conv_messages = []
        for m in messages:
            if m["role"] == "system":
                system_parts.append(m["content"])
            else:
                conv_messages.append(m)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": conv_messages,
            "max_tokens": 8192,
            "temperature": temperature,
        }
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)
        if tools:
            anthropic_tools = []
            for t in tools:
                func = t["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                })
            kwargs["tools"] = anthropic_tools

        async with self._anthropic.messages.stream(**kwargs) as stream:
            async for event in stream:
                if event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        yield StreamDelta(content=event.delta.text)
                    elif event.delta.type == "input_json_delta":
                        yield StreamDelta(tool_call_args_delta=event.delta.partial_json)
                elif event.type == "content_block_start":
                    if event.content_block.type == "tool_use":
                        yield StreamDelta(
                            tool_call_index=event.index,
                            tool_call_id=event.content_block.id,
                            tool_call_name=event.content_block.name,
                        )
                elif event.type == "message_delta":
                    if hasattr(event.delta, "stop_reason") and event.delta.stop_reason:
                        yield StreamDelta(finish_reason=event.delta.stop_reason)


def estimate_tokens(text: str) -> int:
    """Rough token estimation: ~4 chars per token for English, ~2 for CJK."""
    return max(1, len(text) // 3)


def estimate_messages_tokens(messages: list[dict]) -> int:
    """Estimate total tokens across a message list."""
    total = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    total += estimate_tokens(part["text"])
    return total
