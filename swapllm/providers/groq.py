from __future__ import annotations

import groq
import httpx

from .._normalize import VendorExceptionMap, normalize_exception
from ..exceptions import ProviderResponseValidationError
from .base import Message, validate_messages

# The one place groq's own exception classes are named. Everything else in
# swapllm, including the Router, only ever sees swapllm.exceptions types.
_VMAP = VendorExceptionMap(
    rate_limit=groq.RateLimitError,
    timeout=groq.APITimeoutError,
    connection=groq.APIConnectionError,
    server=groq.InternalServerError,
)


class GroqProvider:
    name = "groq"

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        # max_retries=0: the SDK would otherwise retry 429s, 5xx and timeouts itself (twice, with
        # backoff) before swapLLM ever sees the failure, which delays fallback by seconds and
        # contradicts the README's "no same-provider retry". Fallback is the retry strategy.
        self._client = groq.Groq(api_key=api_key, http_client=http_client, max_retries=0)

    def complete(self, messages: list[Message]) -> str:
        validate_messages(messages)
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            content = response.choices[0].message.content
            if content is None:
                raise ProviderResponseValidationError(self.name, "provider returned no text content")
            return content
        except ProviderResponseValidationError:
            raise
        except Exception as exc:
            raise normalize_exception(self.name, exc, _VMAP) from exc
