"""Contract tests for OpenAIProvider (SPEC.md S5): mocked HTTP only, no real
API calls. Uses httpx.MockTransport injected via OpenAIProvider's http_client
param, so tests never touch the network and never need an API key.
"""

from __future__ import annotations

import httpx
import openai
import pytest

from swapllm import (
    OpenAIProvider,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderResponseValidationError,
    ProviderServerError,
    ProviderTimeoutError,
)


def _chat_completion_body(content: str | None, *, finish_reason: str = "stop") -> dict:
    """Shape of a real openai ChatCompletion response (verified against
    openai.types.chat.ChatCompletion / Choice / ChatCompletionMessage)."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        },
    }


def _error_body(message: str) -> dict:
    return {"error": {"message": message, "type": "invalid_request_error"}}


def _provider(handler) -> OpenAIProvider:
    transport = httpx.MockTransport(handler)
    return OpenAIProvider(
        api_key="test-key",
        model="gpt-4o-mini",
        http_client=httpx.Client(transport=transport),
    )


def test_complete_returns_normalized_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_chat_completion_body("hello from openai"))

    provider = _provider(handler)
    result = provider.complete([{"role": "user", "content": "hi"}])

    assert result == "hello from openai"


def test_rate_limit_normalizes_to_provider_rate_limit_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json=_error_body("rate limit exceeded"))

    provider = _provider(handler)
    with pytest.raises(ProviderRateLimitError) as exc_info:
        provider.complete([{"role": "user", "content": "hi"}])

    assert exc_info.value.provider == "openai"
    assert exc_info.value.original is not None


@pytest.mark.parametrize("status", [500, 502, 503])
def test_5xx_normalizes_to_provider_server_error(status: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=_error_body("upstream error"))

    provider = _provider(handler)
    with pytest.raises(ProviderServerError) as exc_info:
        provider.complete([{"role": "user", "content": "hi"}])

    assert exc_info.value.provider == "openai"


def test_timeout_normalizes_to_provider_timeout_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    provider = _provider(handler)
    with pytest.raises(ProviderTimeoutError) as exc_info:
        provider.complete([{"role": "user", "content": "hi"}])

    assert exc_info.value.provider == "openai"


def test_connection_error_normalizes_to_provider_timeout_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    provider = _provider(handler)
    with pytest.raises(ProviderTimeoutError) as exc_info:
        provider.complete([{"role": "user", "content": "hi"}])

    assert exc_info.value.provider == "openai"


def test_bad_request_normalizes_to_provider_request_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json=_error_body("bad request"))

    provider = _provider(handler)
    with pytest.raises(ProviderRequestError) as exc_info:
        provider.complete([{"role": "user", "content": "hi"}])

    assert exc_info.value.provider == "openai"


def test_missing_content_normalizes_to_provider_response_validation_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_chat_completion_body(None, finish_reason="content_filter"))

    provider = _provider(handler)
    with pytest.raises(ProviderResponseValidationError):
        provider.complete([{"role": "user", "content": "hi"}])


def test_length_finish_reason_normalizes_to_provider_response_validation_error() -> None:
    """LengthFinishReasonError has no Anthropic equivalent - it means the
    provider truncated the response before it could be used. This must
    trigger Router failover (ProviderResponseValidationError), not surface
    as an unrecognized ProviderRequestError via the generic catch-all."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_chat_completion_body("truncated...", finish_reason="length"))

    provider = _provider(handler)
    completion = openai.types.chat.ChatCompletion.model_validate(
        _chat_completion_body("truncated...", finish_reason="length")
    )

    def raise_length(*args, **kwargs):
        raise openai.LengthFinishReasonError(completion=completion)

    provider._client.chat.completions.create = raise_length

    with pytest.raises(ProviderResponseValidationError) as exc_info:
        provider.complete([{"role": "user", "content": "hi"}])

    assert exc_info.value.provider == "openai"


def test_content_filter_finish_reason_normalizes_to_provider_response_validation_error() -> None:
    """ContentFilterFinishReasonError has no Anthropic equivalent - the
    request succeeded but the content filter declined the content. A
    content-filter false-positive from one vendor should fall back to the
    next provider rather than surfacing as an unrecognized hard failure."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_chat_completion_body(None, finish_reason="content_filter"))

    provider = _provider(handler)

    def raise_content_filter(*args, **kwargs):
        raise openai.ContentFilterFinishReasonError()

    provider._client.chat.completions.create = raise_content_filter

    with pytest.raises(ProviderResponseValidationError) as exc_info:
        provider.complete([{"role": "user", "content": "hi"}])

    assert exc_info.value.provider == "openai"
