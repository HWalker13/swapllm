from __future__ import annotations

import httpx
import openai

from .._normalize import VendorExceptionMap, normalize_exception
from ..exceptions import ProviderResponseValidationError
from .base import Message

# The one place openai's own exception classes are named. Everything else in
# swapllm, including the Router, only ever sees swapllm.exceptions types.
_VMAP = VendorExceptionMap(
    rate_limit=openai.RateLimitError,
    timeout=openai.APITimeoutError,
    connection=openai.APIConnectionError,
    server=openai.InternalServerError,
)

# openai raises these from its structured-output parsing helper when a
# response is truncated (finish_reason="length") or declined by the content
# filter (finish_reason="content_filter") - the request succeeded, but there's
# no usable text, same category as the null-content case in GroqProvider. No
# Anthropic SDK exception corresponds to either one. Handled ahead of the
# generic Exception catch-all so a content-filter false-positive triggers
# Router failover instead of surfacing as an unrecognized ProviderRequestError.
_UNUSABLE_RESPONSE: tuple[type[Exception], ...] = (
    openai.LengthFinishReasonError,
    openai.ContentFilterFinishReasonError,
)


class OpenAIProvider:
    name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self._client = openai.OpenAI(api_key=api_key, http_client=http_client)

    def complete(self, messages: list[Message]) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            content = response.choices[0].message.content
            if content is None:
                raise ProviderResponseValidationError(self.name, "provider returned no text content")
            return content
        except _UNUSABLE_RESPONSE as exc:
            raise ProviderResponseValidationError(self.name, str(exc)) from exc
        except ProviderResponseValidationError:
            raise
        except Exception as exc:
            raise normalize_exception(self.name, exc, _VMAP) from exc
