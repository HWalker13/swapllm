"""Shared vendor-exception -> swapllm-exception mapping.

Groq, OpenAI, and Anthropic's Python SDKs are all generated from the same
Stainless OpenAPI tooling and are structurally identical one level below their
vendor-specific base class: each exposes ``RateLimitError`` (429) and
``InternalServerError`` (any 5xx) as siblings under ``APIStatusError``, and
``APITimeoutError`` as a subclass of the broader ``APIConnectionError``. None
of these classes are shared *across* vendors, so each adapter builds one
``VendorExceptionMap`` from its own SDK's classes at import time - that map is
the only place a provider-specific exception class name appears. Everything
else, including this function, works only in terms of the normalized types in
``swapllm.exceptions``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .exceptions import (
    ProviderError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderServerError,
    ProviderTimeoutError,
)


@dataclass(frozen=True)
class VendorExceptionMap:
    """The small set of a vendor SDK's own exception classes this library
    cares about, gathered in one place per adapter."""

    rate_limit: type[Exception]
    timeout: type[Exception]
    connection: type[Exception]
    server: type[Exception]


def normalize_exception(provider: str, exc: Exception, vmap: VendorExceptionMap) -> ProviderError:
    """Translate one vendor SDK exception into its normalized swapllm equivalent.

    Check order matters: ``timeout`` must be checked before ``connection``
    because every vendor's ``APITimeoutError`` subclasses its own
    ``APIConnectionError`` - checking the broader class first would shadow
    the more specific one.
    """
    if isinstance(exc, vmap.rate_limit):
        return ProviderRateLimitError(provider, str(exc), original=exc)
    if isinstance(exc, vmap.timeout):
        return ProviderTimeoutError(provider, str(exc), original=exc)
    if isinstance(exc, vmap.connection):
        return ProviderTimeoutError(provider, str(exc), original=exc)
    if isinstance(exc, vmap.server):
        return ProviderServerError(provider, str(exc), original=exc)
    return ProviderRequestError(provider, str(exc), original=exc)
