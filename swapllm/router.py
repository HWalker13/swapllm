"""Router: SPEC.md S3-S4 fallback orchestration across provider adapters.

Tries each provider in ``fallback_order``, in order, stopping at the first
one that succeeds. A provider is only skipped in favor of the next one for
failure classes that mean "this specific provider cannot answer right now"
- rate limit, timeout, 5xx, or an unusable/invalid response
(``ProviderRateLimitError`` / ``ProviderTimeoutError`` / ``ProviderServerError``
/ ``ProviderResponseValidationError``). Every other exception -
``ProviderRequestError``, or the plain ``ValueError`` that
``validate_messages()`` raises for a caller-input problem - propagates
immediately without trying any other provider, since switching providers
would not fix a caller's own misconfiguration or malformed request (SPEC.md
S4; see also ``swapllm.providers.base.validate_messages``).

Schema validation (SPEC.md S3-S4's optional ``schema=`` param) is Day 4
scope and is deliberately not implemented here yet.
"""

from __future__ import annotations

from .exceptions import (
    AllProvidersFailedError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseValidationError,
    ProviderServerError,
    ProviderTimeoutError,
)
from .providers import Message, Provider

# The failure classes that mean "this provider, not the request itself, is
# the problem" - the only ones that trigger fallback to the next provider.
# ProviderRequestError and the plain ValueError from validate_messages() are
# deliberately absent: both mean the caller's input or config is the issue,
# and failing over would just retry the same bad input against the next
# provider instead of surfacing it.
_RETRYABLE: tuple[type[ProviderError], ...] = (
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderServerError,
    ProviderResponseValidationError,
)


class Router:
    """Tries providers in ``fallback_order`` until one succeeds, falling
    over to the next provider only on a retryable failure and giving up with
    ``AllProvidersFailedError`` only once every provider in the order has
    been tried and failed.

    What advances to the next provider: ``ProviderRateLimitError``,
    ``ProviderTimeoutError``, ``ProviderServerError``, or
    ``ProviderResponseValidationError`` - failure classes that mean "this
    specific provider cannot answer right now," not "this request is
    broken."

    What does NOT advance, and instead propagates immediately with no
    further providers tried: ``ProviderRequestError`` (near-always a caller
    misconfiguration - bad API key, malformed request - that a different
    provider would not fix), and the plain ``ValueError`` raised by
    ``validate_messages()`` for a caller-input problem (see
    ``swapllm.providers.base.validate_messages``). Both are deliberately
    absent from the Router's retry set for the same reason: failing over
    would just retry the same bad input or config against the next
    provider instead of surfacing it.

    What happens if every provider in ``fallback_order`` fails a retryable
    way: ``Router.complete`` raises ``AllProvidersFailedError``, carrying
    one ``ProviderError`` per attempt, in the order each provider was tried
    (SPEC.md S4) - never a silent ``None`` or empty string.

    ``providers`` and ``fallback_order`` are kept as two separate params
    (matching SPEC.md S3) rather than inferring order from list position:
    ``providers`` is the set of already-instantiated adapters, keyed
    internally by their ``.name``; ``fallback_order`` is what actually
    controls attempt order. Keeping them separate means the attempt order
    can be reconfigured (or reused across multiple routers with different
    orders) without re-instantiating the underlying provider objects, which
    each hold a live vendor SDK client.
    """

    def __init__(self, providers: list[Provider], fallback_order: list[str]) -> None:
        if not fallback_order:
            raise ValueError("fallback_order must contain at least one provider name")

        by_name = {p.name: p for p in providers}
        unknown = [name for name in fallback_order if name not in by_name]
        if unknown:
            raise ValueError(f"fallback_order references provider(s) not in providers: {unknown}")

        self.providers = list(providers)
        self.fallback_order = list(fallback_order)
        self._by_name = by_name

    def complete(self, messages: list[Message]) -> str:
        """Return the first successful completion, trying providers in
        ``fallback_order``.

        What advances to the next provider, what doesn't, and what happens
        if every provider fails: see the ``Router`` class docstring.
        """
        failures: list[ProviderError] = []
        for name in self.fallback_order:
            provider = self._by_name[name]
            try:
                return provider.complete(messages)
            except _RETRYABLE as exc:
                failures.append(exc)
        raise AllProvidersFailedError(failures)
