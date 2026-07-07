"""Contract tests for GroqProvider (SPEC.md S5): mocked HTTP only, no real
API calls. Uses httpx.MockTransport injected via GroqProvider's http_client
param, so tests never touch the network and never need an API key.
"""

from __future__ import annotations

import httpx
import pytest

from swapllm import (
    GroqProvider,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderResponseValidationError,
    ProviderServerError,
    ProviderTimeoutError,
)


def _chat_completion_body(content: str) -> dict:
    """Shape of a real groq ChatCompletion response (verified against
    groq.types.chat.ChatCompletion / Choice / ChatCompletionMessage)."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "llama-3.1-70b",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
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


def _provider(handler) -> GroqProvider:
    transport = httpx.MockTransport(handler)
    return GroqProvider(
        api_key="test-key",
        model="llama-3.1-70b",
        http_client=httpx.Client(transport=transport),
    )


def test_complete_returns_normalized_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_chat_completion_body("hello from groq"))

    provider = _provider(handler)
    result = provider.complete([{"role": "user", "content": "hi"}])

    assert result == "hello from groq"


def test_rate_limit_normalizes_to_provider_rate_limit_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json=_error_body("rate limit exceeded"))

    provider = _provider(handler)
    with pytest.raises(ProviderRateLimitError) as exc_info:
        provider.complete([{"role": "user", "content": "hi"}])

    assert exc_info.value.provider == "groq"
    assert exc_info.value.original is not None


@pytest.mark.parametrize("status", [500, 502, 503])
def test_5xx_normalizes_to_provider_server_error(status: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=_error_body("upstream error"))

    provider = _provider(handler)
    with pytest.raises(ProviderServerError) as exc_info:
        provider.complete([{"role": "user", "content": "hi"}])

    assert exc_info.value.provider == "groq"


def test_timeout_normalizes_to_provider_timeout_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    provider = _provider(handler)
    with pytest.raises(ProviderTimeoutError) as exc_info:
        provider.complete([{"role": "user", "content": "hi"}])

    assert exc_info.value.provider == "groq"


def test_connection_error_normalizes_to_provider_timeout_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    provider = _provider(handler)
    with pytest.raises(ProviderTimeoutError) as exc_info:
        provider.complete([{"role": "user", "content": "hi"}])

    assert exc_info.value.provider == "groq"


def test_bad_request_normalizes_to_provider_request_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json=_error_body("bad request"))

    provider = _provider(handler)
    with pytest.raises(ProviderRequestError) as exc_info:
        provider.complete([{"role": "user", "content": "hi"}])

    assert exc_info.value.provider == "groq"


def test_missing_content_normalizes_to_provider_response_validation_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = _chat_completion_body("placeholder")
        body["choices"][0]["message"]["content"] = None
        return httpx.Response(200, json=body)

    provider = _provider(handler)
    with pytest.raises(ProviderResponseValidationError):
        provider.complete([{"role": "user", "content": "hi"}])
