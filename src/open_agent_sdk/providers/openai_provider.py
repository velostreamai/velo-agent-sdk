"""OpenAI Chat Completions API Provider.

Converts between the SDK's internal Anthropic-like message format
and OpenAI's Chat Completions API format.

Uses urllib.request with asyncio.get_event_loop().run_in_executor()
to avoid adding new dependencies (same approach as WebFetchTool).
"""

from __future__ import annotations

import asyncio
import json
import urllib.request
import urllib.error
from typing import Any

from open_agent_sdk.providers.types import (
    ApiType,
    CreateMessageParams,
    CreateMessageResponse,
    NormalizedTool,
)



_MAX_TOOL_INPUT_DECODE_DEPTH = 4


def _normalise_tool_input(raw: Any) -> dict[str, Any]:
    """Coerce a provider's tool-call `arguments` into the dict a tool expects.

    A tool-use block's `input` is typed `dict[str, Any]` and was never enforced.
    Two paths handed a `str` downstream instead, and the first tool method to
    call `.get()` on it died with

        AttributeError: 'str' object has no attribute 'get'

    killing the run mid-flight, after arbitrary spend, with nothing committed.
    One dispatch burned 9.9M input tokens that way.

    The two paths differ, and only one is obvious:

    1. The old `except` branch assigned the raw string verbatim when
       `json.loads` raised — violating the declared type by design.
    2. **Double-encoded arguments.** The provider JSON-encodes the argument
       string a second time, so `json.loads` SUCCEEDS and returns a `str`. No
       exception, and nothing downstream checked the type. This is why "the SDK
       does not parse arguments" was never an accurate description: it parses,
       then hands back a string.

    So decode repeatedly while the result is still a string, and fail CLOSED —
    anything that will not resolve to a dict becomes `{}` rather than reaching a
    tool. A tool given `{}` does nothing; a tool given a half-decoded string is
    how a shell command gets misread.
    """
    value = raw
    for _ in range(_MAX_TOOL_INPUT_DECODE_DEPTH):
        if isinstance(value, dict):
            return value
        if not isinstance(value, str):
            break
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError, ValueError):
            break
    return value if isinstance(value, dict) else {}


class OpenAIProvider:
    """LLM provider for OpenAI Chat Completions API."""

    def __init__(
        self,
        *,
        api_key: str = "",
        base_url: str = "",
        provider_routing: dict[str, Any] | None = None,
        include_usage: bool = False,
    ):
        """
        provider_routing:
            Gateway routing preferences, sent as the request body's `provider`
            field. OpenRouter's shape is `{"order": [...], "allow_fallbacks": bool}`.

            This matters more than it looks. A gateway load-balances each request
            across upstream endpoints, and a prefix cache is PER ENDPOINT. An agent
            loop re-sends its whole transcript every turn, so without pinning it
            lands somewhere new each time and misses the cache on EVERY turn.
            Measured on identical prompts: unpinned, three requests hit three
            endpoints with 0 cached tokens; pinned, all three hit one endpoint with
            1280 cached and cost 3.7x less.

            `allow_fallbacks` should normally stay True — a pin that cannot fail
            over turns a slow or down endpoint into a failed run, which costs far
            more than a cache miss.

        include_usage:
            Ask the gateway for real usage accounting (`usage.include`). Without
            it the response carries no cost and no `prompt_tokens_details`, so
            `cached_tokens` cannot be read at all — which is why the cache problem
            above stayed invisible.

        Both default OFF and are omitted from the body entirely when unset: a
        plain OpenAI endpoint rejects unknown fields, so they must never be sent
        speculatively.
        """
        self._api_key = api_key
        self._base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self._provider_routing = provider_routing
        self._include_usage = include_usage

    @property
    def api_type(self) -> ApiType:
        return "openai-completions"

    async def create_message(self, params: CreateMessageParams) -> CreateMessageResponse:
        messages = self._convert_messages(params.system, params.messages)
        tools = self._convert_tools(params.tools) if params.tools else None

        body: dict[str, Any] = {
            "model": params.model,
            "max_tokens": params.max_tokens,
            "messages": messages,
        }

        if tools:
            body["tools"] = tools

        data = await self._post_chat_completions(body)
        return self._convert_response(data)

    # --------------------------------------------------------------------------
    # HTTP
    # --------------------------------------------------------------------------

    async def _post_chat_completions(self, body: dict[str, Any]) -> dict[str, Any]:
        # Applied HERE rather than in create_message: this is the one place every
        # request passes through with a finished body, including any future call
        # site that builds a body of its own.
        #
        # `setdefault`, not assignment — an explicit value from the caller wins.
        if self._provider_routing is not None:
            body.setdefault("provider", self._provider_routing)
        if self._include_usage:
            body.setdefault("usage", {"include": True})

        url = f"{self._base_url}/chat/completions"
        payload = json.dumps(body).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
        )

        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: urllib.request.urlopen(req, timeout=300),
            )
            response_data = response.read().decode("utf-8")
            return json.loads(response_data)
        except urllib.error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode("utf-8")
            except Exception:
                pass
            raise RuntimeError(
                f"OpenAI API error: {e.code} {e.reason}: {err_body}"
            ) from e

    # --------------------------------------------------------------------------
    # Message Conversion: Internal -> OpenAI
    # --------------------------------------------------------------------------

    def _convert_messages(
        self,
        system: str,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []

        if system:
            result.append({"role": "system", "content": system})

        for msg in messages:
            role = msg.get("role", "user")
            if role == "user":
                self._convert_user_message(msg, result)
            elif role == "assistant":
                self._convert_assistant_message(msg, result)

        return result

    def _convert_user_message(
        self,
        msg: dict[str, Any],
        result: list[dict[str, Any]],
    ) -> None:
        content = msg.get("content", "")

        if isinstance(content, str):
            result.append({"role": "user", "content": content})
            return

        # Content blocks may contain text and/or tool_result blocks
        text_parts: list[str] = []
        tool_results: list[dict[str, Any]] = []

        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type", "")
            if block_type == "text":
                text_parts.append(block.get("text", ""))
            elif block_type == "tool_result":
                tool_results.append({
                    "tool_use_id": block.get("tool_use_id", ""),
                    "content": block.get("content", ""),
                })

        # Tool results become separate tool messages
        for tr in tool_results:
            content_val = tr["content"]
            if not isinstance(content_val, str):
                content_val = json.dumps(content_val)
            result.append({
                "role": "tool",
                "tool_call_id": tr["tool_use_id"],
                "content": content_val,
            })

        # Text parts become a user message
        if text_parts:
            result.append({"role": "user", "content": "\n".join(text_parts)})

    def _convert_assistant_message(
        self,
        msg: dict[str, Any],
        result: list[dict[str, Any]],
    ) -> None:
        content = msg.get("content", "")

        if isinstance(content, str):
            result.append({"role": "assistant", "content": content})
            return

        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type", "")
            if block_type == "text":
                text_parts.append(block.get("text", ""))
            elif block_type == "tool_use":
                input_val = block.get("input", {})
                tool_calls.append({
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": input_val if isinstance(input_val, str) else json.dumps(input_val),
                    },
                })

        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": "\n".join(text_parts) if text_parts else None,
        }

        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls

        result.append(assistant_msg)

    # --------------------------------------------------------------------------
    # Tool Conversion: Internal -> OpenAI
    # --------------------------------------------------------------------------

    def _convert_tools(self, tools: list[NormalizedTool]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in tools
        ]

    # --------------------------------------------------------------------------
    # Response Conversion: OpenAI -> Internal
    # --------------------------------------------------------------------------

    def _convert_response(self, data: dict[str, Any]) -> CreateMessageResponse:
        choices = data.get("choices", [])
        if not choices:
            return CreateMessageResponse(
                content=[{"type": "text", "text": ""}],
                stop_reason="end_turn",
                usage={"input_tokens": 0, "output_tokens": 0,
                       "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
            )

        choice = choices[0]
        message = choice.get("message", {})
        content: list[dict[str, Any]] = []

        # Text content
        if message.get("content"):
            content.append({"type": "text", "text": message["content"]})

        # Tool calls
        for tc in (message.get("tool_calls") or []):
            func = tc.get("function", {})
            input_val = _normalise_tool_input(func.get("arguments"))

            content.append({
                "type": "tool_use",
                "id": tc.get("id", ""),
                "name": func.get("name", ""),
                "input": input_val,
            })

        if not content:
            content.append({"type": "text", "text": ""})

        # Map finish_reason
        finish_reason = choice.get("finish_reason", "stop")
        stop_reason = self._map_finish_reason(finish_reason)

        usage_data = data.get("usage", {})
        return CreateMessageResponse(
            content=content,
            stop_reason=stop_reason,
            usage={
                "input_tokens": usage_data.get("prompt_tokens", 0),
                "output_tokens": usage_data.get("completion_tokens", 0),
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        )

    @staticmethod
    def _map_finish_reason(reason: str) -> str:
        mapping = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
        }
        return mapping.get(reason, reason)
