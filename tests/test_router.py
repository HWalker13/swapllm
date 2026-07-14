"""Router tests (SPEC.md S3-S4): fallback across providers on retryable
failures, immediate propagation on non-retryable ones, and
AllProvidersFailedError population when every provider fails.

Mirrors test_provider_parity.py's structure: mocks all three vendors via
httpx.MockTransport so the Router is exercised against the real adapters,
not a hand-rolled test double, proving the fallback loop and each adapter's
own exception normalization work together end-to-end.
"""

from __future__ import annotations

from typing import Callable

import httpx
import pytest

from swapllm import (
    AllProvidersFailedError,
    AnthropicProvider,
    GroqProvider,
    OpenAIProvider,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderServerError,
    Router,
)

Handler = Callable[[httpx.Request], httpx.Response]


# --- body shapes, one set per vendor (see test_provider_parity.py - each
# vendor gets its own builders on purpose, duplicated here rather than
# imported, matching this codebase's existing per-test-file convention).


def _openai_shaped_success_body(content: str | None) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _openai_shaped_error_body(message: str) -> dict:
    return {"error": {"message": message, "type": "invalid_request_error"}}


def _anthropic_success_body(content: str) -> dict:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "test-model",
        "content": [{"type": "text", "text": content}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def _anthropic_error_body(message: str) -> dict:
    return {"type": "error", "error": {"type": "invalid_request_error", "message": message}}


def _build_groq(handler: Handler) -> GroqProvider:
    return GroqProvider(
        api_key="test-key", model="test-model", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )


def _build_openai(handler: Handler) -> OpenAIProvider:
    return OpenAIProvider(
        api_key="test-key", model="test-model", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )


def _build_anthropic(handler: Handler) -> AnthropicProvider:
    return AnthropicProvider(
        api_key="test-key", model="test-model", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )


def _always(status: int, body: dict) -> Handler:
    return lambda request: httpx.Response(status, json=body)


def _always_ok(body: dict) -> Handler:
    return lambda request: httpx.Response(200, json=body)


def _unreachable_handler(request: httpx.Request) -> httpx.Response:
    raise AssertionError(
        "this provider should not have been called - non-retryable errors and "
        "validate_messages() ValueErrors must not trigger fallback"
    )


# --- the four retryable failure types, each shaped as a first-provider
# (groq) response/exception that must cause the Router to move on.


def _rate_limit_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(429, json=_openai_shaped_error_body("rate limited"))


def _server_error_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(503, json=_openai_shaped_error_body("upstream error"))


def _timeout_handler(request: httpx.Request) -> httpx.Response:
    raise httpx.TimeoutException("timed out", request=request)


def _missing_content_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=_openai_shaped_success_body(None))


@pytest.mark.parametrize(
    "failing_handler",
    [_rate_limit_handler, _server_error_handler, _timeout_handler, _missing_content_handler],
    ids=["rate_limit", "server_error", "timeout", "missing_content"],
)
def test_complete_advances_to_next_provider_on_retryable_failure(failing_handler: Handler) -> None:
    groq = _build_groq(failing_handler)
    openai_provider = _build_openai(_always_ok(_openai_shaped_success_body("answer from openai")))
    anthropic = _build_anthropic(_unreachable_handler)

    router = Router(providers=[groq, openai_provider, anthropic], fallback_order=["groq", "openai", "anthropic"])

    result = router.complete([{"role": "user", "content": "hi"}])

    assert result == "answer from openai"


def test_complete_advances_through_multiple_failures_to_last_provider() -> None:
    groq = _build_groq(_rate_limit_handler)
    openai_provider = _build_openai(_server_error_handler)
    anthropic = _build_anthropic(_always_ok(_anthropic_success_body("answer from anthropic")))

    router = Router(providers=[groq, openai_provider, anthropic], fallback_order=["groq", "openai", "anthropic"])

    result = router.complete([{"role": "user", "content": "hi"}])

    assert result == "answer from anthropic"


def test_complete_propagates_provider_request_error_without_fallback() -> None:
    groq = _build_groq(_always(400, _openai_shaped_error_body("bad request")))
    openai_provider = _build_openai(_unreachable_handler)
    anthropic = _build_anthropic(_unreachable_handler)

    router = Router(providers=[groq, openai_provider, anthropic], fallback_order=["groq", "openai", "anthropic"])

    with pytest.raises(ProviderRequestError):
        router.complete([{"role": "user", "content": "hi"}])


def test_complete_propagates_validate_messages_value_error_without_fallback() -> None:
    groq = _build_groq(_unreachable_handler)
    openai_provider = _build_openai(_unreachable_handler)
    anthropic = _build_anthropic(_unreachable_handler)

    router = Router(providers=[groq, openai_provider, anthropic], fallback_order=["groq", "openai", "anthropic"])
    messages = [
        {"role": "system", "content": "be nice"},
        {"role": "system", "content": "be extra nice"},
        {"role": "user", "content": "hi"},
    ]

    with pytest.raises(ValueError):
        router.complete(messages)


def test_all_providers_failed_error_populates_failures_with_provider_and_original() -> None:
    groq = _build_groq(_always(429, _openai_shaped_error_body("groq rate limited")))
    openai_provider = _build_openai(_always(503, _openai_shaped_error_body("openai upstream error")))
    anthropic = _build_anthropic(_always(500, _anthropic_error_body("anthropic upstream error")))

    router = Router(providers=[groq, openai_provider, anthropic], fallback_order=["groq", "openai", "anthropic"])

    with pytest.raises(AllProvidersFailedError) as exc_info:
        router.complete([{"role": "user", "content": "hi"}])

    failures = exc_info.value.failures
    assert [f.provider for f in failures] == ["groq", "openai", "anthropic"]
    assert isinstance(failures[0], ProviderRateLimitError)
    assert isinstance(failures[1], ProviderServerError)
    assert isinstance(failures[2], ProviderServerError)
    for failure in failures:
        assert failure.original is not None


def test_init_rejects_fallback_order_referencing_unknown_provider() -> None:
    groq = _build_groq(_unreachable_handler)

    with pytest.raises(ValueError):
        Router(providers=[groq], fallback_order=["groq", "openai"])


def test_init_rejects_empty_fallback_order() -> None:
    groq = _build_groq(_unreachable_handler)

    with pytest.raises(ValueError):
        Router(providers=[groq], fallback_order=[])


def test_init_rejects_duplicate_fallback_order_names() -> None:
    groq = _build_groq(_unreachable_handler)
    openai_provider = _build_openai(_unreachable_handler)

    with pytest.raises(ValueError):
        Router(providers=[groq, openai_provider], fallback_order=["groq", "groq", "openai"])
