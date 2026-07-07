from __future__ import annotations

import groq
import httpx

from .._normalize import VendorExceptionMap, normalize_exception
from ..exceptions import ProviderRequestError
from .base import Message

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
        self._client = groq.Groq(api_key=api_key, http_client=http_client)

    def complete(self, messages: list[Message]) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            content = response.choices[0].message.content
            if content is None:
                raise ProviderRequestError(self.name, "provider returned no text content")
            return content
        except ProviderRequestError:
            raise
        except Exception as exc:
            raise normalize_exception(self.name, exc, _VMAP) from exc
