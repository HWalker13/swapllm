"""Cross-provider parity contract test (SPEC.md S5): "mock each provider's
HTTP response and assert the *normalized* output shape matches across all
three, regardless of how differently each vendor structures its raw
response... proving the abstraction genuinely abstracts, not just that each
adapter can parse its own vendor's JSON."

The per-provider test files (test_groq_provider.py, test_openai_provider.py,
test_anthropic_provider.py) each prove one adapter parses its own vendor's
response shape correctly. None of them prove the three adapters agree with
each other - a bug that swapped GroqProvider's return value shape, or that
made AnthropicProvider raise a different exception type than the other two
for the same failure class, would pass every one of those files individually.
This file is the one that would catch it: every test below drives all three
GroqProvider / OpenAIProvider / AnthropicProvider instances through the same
scenario and asserts they produce the identical normalized result, even
though each vendor's mocked HTTP body is shaped completely differently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import httpx
import pytest

from swapllm import (
    AnthropicProvider,
    GroqProvider,
    OpenAIProvider,
    Provider,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderResponseValidationError,
    ProviderServerError,
    ProviderTimeoutError,
)

Handler = Callable[[httpx.Request], httpx.Response]

# Used as the mock transport handler for tests that assert validate_messages()
# rejects bad input before any adapter builds a vendor request - if rejection
# ever regresses and an adapter proceeds to make the HTTP call, this fails the
# test with a clear assertion instead of silently passing or hitting the
# network.
_UNREACHABLE_HANDLER: Handler = lambda request: (_ for _ in ()).throw(  # noqa: E731
    AssertionError("validate_messages() should reject the input before any HTTP call is made")
)


# --- Groq/OpenAI share the OpenAI-compatible chat.completion response shape;
# Anthropic's Message shape is structurally different (content is a list of
# typed blocks, not a single string field; errors are `{"type": "error", ...}`
# rather than `{"error": {...}}`). Each vendor gets its own body builders on
# purpose - the point of this file is that despite that difference, the
# *normalized* result is identical.


def _openai_shaped_success_body(content: str) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _openai_shaped_missing_content_body() -> dict:
    body = _openai_shaped_success_body("placeholder")
    body["choices"][0]["message"]["content"] = None
    return body


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


def _anthropic_missing_content_body() -> dict:
    body = _anthropic_success_body("placeholder")
    body["content"] = []
    return body


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


@dataclass(frozen=True)
class _ProviderSpec:
    name: str
    build: Callable[[Handler], Provider]
    success_body: Callable[[str], dict]
    error_body: Callable[[str], dict]
    missing_content_body: Callable[[], dict]


_PROVIDER_SPECS = [
    _ProviderSpec(
        "groq", _build_groq, _openai_shaped_success_body, _openai_shaped_error_body, _openai_shaped_missing_content_body
    ),
    _ProviderSpec(
        "openai",
        _build_openai,
        _openai_shaped_success_body,
        _openai_shaped_error_body,
        _openai_shaped_missing_content_body,
    ),
    _ProviderSpec(
        "anthropic", _build_anthropic, _anthropic_success_body, _anthropic_error_body, _anthropic_missing_content_body
    ),
]
_SPEC_IDS = [spec.name for spec in _PROVIDER_SPECS]


@pytest.mark.parametrize("spec", _PROVIDER_SPECS, ids=_SPEC_IDS)
def test_complete_returns_identical_text_regardless_of_vendor(spec: _ProviderSpec) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=spec.success_body("same answer, any vendor"))

    provider = spec.build(handler)
    result = provider.complete([{"role": "user", "content": "hi"}])

    assert result == "same answer, any vendor"


@pytest.mark.parametrize("spec", _PROVIDER_SPECS, ids=_SPEC_IDS)
def test_rate_limit_normalizes_identically_regardless_of_vendor(spec: _ProviderSpec) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json=spec.error_body("rate limit exceeded"))

    provider = spec.build(handler)
    with pytest.raises(ProviderRateLimitError) as exc_info:
        provider.complete([{"role": "user", "content": "hi"}])

    assert exc_info.value.provider == spec.name


@pytest.mark.parametrize("status", [500, 502, 503])
@pytest.mark.parametrize("spec", _PROVIDER_SPECS, ids=_SPEC_IDS)
def test_5xx_normalizes_identically_regardless_of_vendor(spec: _ProviderSpec, status: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=spec.error_body("upstream error"))

    provider = spec.build(handler)
    with pytest.raises(ProviderServerError) as exc_info:
        provider.complete([{"role": "user", "content": "hi"}])

    assert exc_info.value.provider == spec.name


@pytest.mark.parametrize("spec", _PROVIDER_SPECS, ids=_SPEC_IDS)
def test_timeout_normalizes_identically_regardless_of_vendor(spec: _ProviderSpec) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    provider = spec.build(handler)
    with pytest.raises(ProviderTimeoutError) as exc_info:
        provider.complete([{"role": "user", "content": "hi"}])

    assert exc_info.value.provider == spec.name


@pytest.mark.parametrize("spec", _PROVIDER_SPECS, ids=_SPEC_IDS)
def test_connection_error_normalizes_identically_regardless_of_vendor(spec: _ProviderSpec) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    provider = spec.build(handler)
    with pytest.raises(ProviderTimeoutError) as exc_info:
        provider.complete([{"role": "user", "content": "hi"}])

    assert exc_info.value.provider == spec.name


@pytest.mark.parametrize("spec", _PROVIDER_SPECS, ids=_SPEC_IDS)
def test_bad_request_normalizes_identically_regardless_of_vendor(spec: _ProviderSpec) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json=spec.error_body("bad request"))

    provider = spec.build(handler)
    with pytest.raises(ProviderRequestError) as exc_info:
        provider.complete([{"role": "user", "content": "hi"}])

    assert exc_info.value.provider == spec.name


@pytest.mark.parametrize("spec", _PROVIDER_SPECS, ids=_SPEC_IDS)
def test_missing_content_normalizes_identically_regardless_of_vendor(spec: _ProviderSpec) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=spec.missing_content_body())

    provider = spec.build(handler)
    with pytest.raises(ProviderResponseValidationError) as exc_info:
        provider.complete([{"role": "user", "content": "hi"}])

    assert exc_info.value.provider == spec.name


@pytest.mark.parametrize("spec", _PROVIDER_SPECS, ids=_SPEC_IDS)
def test_multiple_system_messages_rejected_regardless_of_vendor(spec: _ProviderSpec) -> None:
    provider = spec.build(_UNREACHABLE_HANDLER)
    messages = [
        {"role": "system", "content": "be nice"},
        {"role": "system", "content": "be extra nice"},
        {"role": "user", "content": "hi"},
    ]

    with pytest.raises(ValueError):
        provider.complete(messages)


@pytest.mark.parametrize("spec", _PROVIDER_SPECS, ids=_SPEC_IDS)
def test_misplaced_system_message_rejected_regardless_of_vendor(spec: _ProviderSpec) -> None:
    provider = spec.build(_UNREACHABLE_HANDLER)
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "be nice"},
    ]

    with pytest.raises(ValueError):
        provider.complete(messages)
