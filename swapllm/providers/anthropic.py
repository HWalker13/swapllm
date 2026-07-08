from __future__ import annotations

import anthropic
import httpx

from .._normalize import VendorExceptionMap, normalize_exception
from ..exceptions import ProviderResponseValidationError
from .base import Message

# The one place anthropic's own exception classes are named. Everything else
# in swapllm, including the Router, only ever sees swapllm.exceptions types.
_VMAP = VendorExceptionMap(
    rate_limit=anthropic.RateLimitError,
    timeout=anthropic.APITimeoutError,
    connection=anthropic.APIConnectionError,
    server=anthropic.InternalServerError,
)

# Anthropic's Messages API requires max_tokens and has no server-side default
# (unlike Groq/OpenAI's OpenAI-compatible endpoints) - callers of the shared
# Provider.complete() interface don't supply one, so this adapter needs its
# own floor.
_DEFAULT_MAX_TOKENS = 4096


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=api_key, http_client=http_client)

    def complete(self, messages: list[Message]) -> str:
        # Unlike Groq/OpenAI's OpenAI-compatible shape, Anthropic's Messages
        # API takes `system` as a separate top-level request field, not a
        # "system"-role entry inside `messages` - a system-role message left
        # in `messages` is rejected by the API. Split it out here so callers
        # can still pass one Message list shape to every provider.
        system, conversation = _split_system(messages)
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system if system is not None else anthropic.NOT_GIVEN,
                messages=conversation,
            )
            content = "".join(block.text for block in response.content if block.type == "text")
            if not content:
                raise ProviderResponseValidationError(self.name, "provider returned no text content")
            return content
        except ProviderResponseValidationError:
            raise
        except Exception as exc:
            raise normalize_exception(self.name, exc, _VMAP) from exc


def _split_system(messages: list[Message]) -> tuple[str | None, list[Message]]:
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    conversation = [m for m in messages if m["role"] != "system"]
    system = "\n\n".join(system_parts) if system_parts else None
    return system, conversation
