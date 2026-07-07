"""Normalized exception hierarchy shared across all provider adapters.

Groq, OpenAI, and Anthropic each ship their own SDK exception hierarchy
(``groq.RateLimitError``, ``openai.RateLimitError``, ``anthropic.RateLimitError``,
etc.) rooted in their own ``GroqError`` / ``OpenAIError`` / ``AnthropicError``
base - there is no shared base class across vendors to catch centrally. Every
adapter is required to catch its vendor's raw exceptions at the SDK call site
and re-raise one of the types below via ``swapllm._normalize.normalize_exception``,
so the Router (SPEC.md S4) can make fallback decisions against one closed set
of types instead of vendor-specific ones.
"""

from __future__ import annotations


class SwapLLMError(Exception):
    """Base class for every error swapllm itself raises."""


class ProviderError(SwapLLMError):
    """A single provider's request failed.

    Carries the provider's short id (matching ``fallback_order`` entries, e.g.
    ``"groq"``) and the original vendor exception, so ``AllProvidersFailedError``
    can report every provider's failure reason without callers needing to know
    vendor-specific exception types.
    """

    def __init__(self, provider: str, message: str, *, original: Exception | None = None) -> None:
        super().__init__(f"[{provider}] {message}")
        self.provider = provider
        self.original = original


class ProviderRateLimitError(ProviderError):
    """The provider responded 429.

    Per SPEC.md S4, the Router fails over to the next provider immediately -
    a rate-limited provider will not un-limit itself in the next few seconds,
    so there is no same-provider retry.
    """


class ProviderTimeoutError(ProviderError):
    """The provider timed out, or the request never reached it at all
    (DNS failure, connection refused, connection dropped).

    These are grouped under one type on purpose: both mean "this provider is
    not answering right now," and the Router's response is identical either
    way - fail over to the next provider. Distinguishing "timed out" from
    "never connected" would add a category with no behavioral difference.
    """


class ProviderServerError(ProviderError):
    """The provider responded with a 5xx status.

    Per SPEC.md S4 (confirmed as a deliberate v1 decision, not a default):
    5xx is treated identically to rate-limit and timeout - immediate failover,
    no same-provider retry. A more elaborate v2 could retry a single 5xx once
    before failing over, on the theory that a 5xx is more often a transient
    blip than a rate-limit is; v1 stays simple and uniform instead.
    """


class ProviderResponseValidationError(ProviderError):
    """The provider returned a successful HTTP response, but the content
    failed schema validation when a ``schema=`` was requested.

    Raised by the Router (SPEC.md S4/S4-schema layer), not by adapters -
    adapters only return raw text and know nothing about the caller's
    Pydantic schema. Included here because it is still a per-provider failure
    that belongs in ``AllProvidersFailedError.failures`` alongside the
    SDK-level errors above.
    """


class ProviderRequestError(ProviderError):
    """Catch-all for vendor errors that are not rate-limit, timeout, or 5xx
    (e.g. 400 bad request, 401/403 auth failures, or genuinely unexpected
    vendor SDK exceptions).

    Not a Router failover trigger in v1: these almost always mean
    misconfiguration (bad API key, malformed request) rather than a
    provider-side outage, and switching to the next provider would silently
    mask that misconfiguration rather than surface it. Callers should expect
    this to propagate rather than trigger fallback.
    """


class AllProvidersFailedError(SwapLLMError):
    """Raised by the Router when every provider in ``fallback_order`` failed.

    Carries every individual :class:`ProviderError` so the caller can inspect
    why each provider failed, rather than swapllm silently returning ``None``
    or an empty string on total failure.
    """

    def __init__(self, failures: list[ProviderError]) -> None:
        self.failures = failures
        detail = "; ".join(f"{f.provider}: {type(f).__name__}: {f.args[0]}" for f in failures)
        super().__init__(f"All providers failed: {detail}")
