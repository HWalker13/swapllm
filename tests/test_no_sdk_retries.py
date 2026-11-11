"""No hidden retries (README "Fallback behavior": no same-provider retry).

The vendor SDKs retry 429s and 5xx responses on their own by default (max_retries=2, with
backoff). swapLLM turns that off, so a failure reaches the Router immediately and fallback
moves on to the next provider. These tests count the HTTP requests each provider actually
sends: one failure must cost exactly one request. Mocked HTTP only, no API keys.
"""

from __future__ import annotations

import collections

import httpx
import pytest

from swapllm import (
    AnthropicProvider,
    GroqProvider,
    OpenAIProvider,
    ProviderRateLimitError,
    ProviderServerError,
    Router,
)

OPENAI_STYLE_ERROR = {"error": {"message": "mocked failure", "type": "server_error"}}
ANTHROPIC_ERROR = {"type": "error", "error": {"type": "api_error", "message": "mocked failure"}}
ANTHROPIC_OK = {
    "id": "msg_test",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-4-5",
    "content": [{"type": "text", "text": "hello from anthropic"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 1, "output_tokens": 1},
}
MESSAGES = [{"role": "user", "content": "hi"}]


def counting_client(counter: collections.Counter, name: str, status: int, body: dict) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        counter[name] += 1
        return httpx.Response(status, json=body)

    return httpx.Client(transport=httpx.MockTransport(handler))


def make(provider: str, counter: collections.Counter, status: int):
    if provider == "groq":
        return GroqProvider(api_key="k", model="llama-3.1-8b-instant", http_client=counting_client(counter, provider, status, OPENAI_STYLE_ERROR))
    if provider == "openai":
        return OpenAIProvider(api_key="k", model="gpt-4o-mini", http_client=counting_client(counter, provider, status, OPENAI_STYLE_ERROR))
    return AnthropicProvider(api_key="k", model="claude-haiku-4-5", http_client=counting_client(counter, provider, status, ANTHROPIC_ERROR))


@pytest.mark.parametrize("provider", ["groq", "openai", "anthropic"])
@pytest.mark.parametrize(("status", "error"), [(429, ProviderRateLimitError), (503, ProviderServerError)])
def test_one_failure_sends_exactly_one_request(provider, status, error) -> None:
    counter: collections.Counter = collections.Counter()
    with pytest.raises(error):
        make(provider, counter, status).complete(MESSAGES)
    assert counter[provider] == 1


def test_router_cascade_sends_one_request_per_provider() -> None:
    counter: collections.Counter = collections.Counter()
    router = Router(
        providers=[
            make("groq", counter, 429),
            make("openai", counter, 503),
            AnthropicProvider(api_key="k", model="claude-haiku-4-5", http_client=counting_client(counter, "anthropic", 200, ANTHROPIC_OK)),
        ],
        fallback_order=["groq", "openai", "anthropic"],
    )
    assert router.complete(MESSAGES) == "hello from anthropic"
    assert counter == {"groq": 1, "openai": 1, "anthropic": 1}
