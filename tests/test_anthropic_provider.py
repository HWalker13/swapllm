"""Contract tests for AnthropicProvider (SPEC.md S5): mocked HTTP only, no
real API calls. Uses httpx.MockTransport injected via AnthropicProvider's
http_client param, so tests never touch the network and never need an API
key.
"""

from __future__ import annotations

import httpx
import pytest

from swapllm import (
    AnthropicProvider,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderResponseValidationError,
    ProviderServerError,
    ProviderTimeoutError,
)


def _message_body(text: str | None) -> dict:
    """Shape of a real anthropic Message response (verified against
    anthropic.types.Message / TextBlock)."""
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-4-5",
        "content": [] if text is None else [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def _error_body(message: str) -> dict:
    return {"type": "error", "error": {"type": "invalid_request_error", "message": message}}


def _provider(handler) -> AnthropicProvider:
    transport = httpx.MockTransport(handler)
    return AnthropicProvider(
        api_key="test-key",
        model="claude-haiku-4-5",
        http_client=httpx.Client(transport=transport),
    )


def test_complete_returns_normalized_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_message_body("hello from anthropic"))

    provider = _provider(handler)
    result = provider.complete([{"role": "user", "content": "hi"}])

    assert result == "hello from anthropic"


def test_system_message_is_split_out_of_conversation() -> None:
    """Anthropic's API takes `system` as a top-level field, not a
    "system"-role entry in `messages` - a system-role message left in
    `messages` would be rejected. Assert the adapter strips it out and the
    call still succeeds, proving callers can use the same Message shape
    across every provider."""

    seen_bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen_bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_message_body("hello from anthropic"))

    provider = _provider(handler)
    result = provider.complete(
        [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "hi"},
        ]
    )

    assert result == "hello from anthropic"
    assert seen_bodies[0]["system"] == "be nice"
    assert seen_bodies[0]["messages"] == [{"role": "user", "content": "hi"}]


def test_rate_limit_normalizes_to_provider_rate_limit_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json=_error_body("rate limit exceeded"))

    provider = _provider(handler)
    with pytest.raises(ProviderRateLimitError) as exc_info:
        provider.complete([{"role": "user", "content": "hi"}])

    assert exc_info.value.provider == "anthropic"
    assert exc_info.value.original is not None


@pytest.mark.parametrize("status", [500, 502, 503])
def test_5xx_normalizes_to_provider_server_error(status: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=_error_body("upstream error"))

    provider = _provider(handler)
    with pytest.raises(ProviderServerError) as exc_info:
        provider.complete([{"role": "user", "content": "hi"}])

    assert exc_info.value.provider == "anthropic"


def test_timeout_normalizes_to_provider_timeout_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    provider = _provider(handler)
    with pytest.raises(ProviderTimeoutError) as exc_info:
        provider.complete([{"role": "user", "content": "hi"}])

    assert exc_info.value.provider == "anthropic"


def test_connection_error_normalizes_to_provider_timeout_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    provider = _provider(handler)
    with pytest.raises(ProviderTimeoutError) as exc_info:
        provider.complete([{"role": "user", "content": "hi"}])

    assert exc_info.value.provider == "anthropic"


def test_bad_request_normalizes_to_provider_request_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json=_error_body("bad request"))

    provider = _provider(handler)
    with pytest.raises(ProviderRequestError) as exc_info:
        provider.complete([{"role": "user", "content": "hi"}])

    assert exc_info.value.provider == "anthropic"


def test_missing_content_normalizes_to_provider_response_validation_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_message_body(None))

    provider = _provider(handler)
    with pytest.raises(ProviderResponseValidationError):
        provider.complete([{"role": "user", "content": "hi"}])
