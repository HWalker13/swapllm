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

Schema validation (SPEC.md S3-S4's optional ``schema=`` param, Day 4 scope):
when ``complete()`` is called with ``schema=``, a provider's raw text is
parsed as JSON and validated against it before being returned. A provider
whose text fails that parse/validate step reuses
``ProviderResponseValidationError`` - per SPEC.md S4, "a provider
technically 'responding' with garbage isn't success" applies identically
whether the garbage is null content (the adapter's own check) or text that
isn't valid JSON matching ``schema`` (the Router's check) - so it triggers
the same fallback via the existing ``_RETRYABLE`` set, with no separate
exception type and no Router control-flow change.

Before validating, a single leading/trailing markdown code fence (` ```json
`...`\n``` ` or a bare ` ```...\n``` `) is stripped if the provider's *entire*
text is that one fenced block - see ``_strip_markdown_json_fence``. This is
narrow on purpose: chat-completion APIs without strict JSON mode routinely
wrap structured output in a fence, and which vendor does this is exactly the
kind of per-vendor quirk this package exists to hide (SPEC.md S4's "callers
can swap providers without the behavior changing for the same input") - but
text with anything before or after the fence is left untouched and still
fails validation, since guessing at arbitrary text-around-JSON is a
different, unbounded problem this does not attempt to solve.
"""

from __future__ import annotations

import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .exceptions import (
    AllProvidersFailedError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseValidationError,
    ProviderServerError,
    ProviderTimeoutError,
)
from .providers import Message, Provider

_SchemaT = TypeVar("_SchemaT", bound=BaseModel)

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

# Matches only when the *whole* (stripped) text is one fenced block - an
# optional "json" language tag on the opening fence, then the body, then a
# closing fence with nothing else around either one. Deliberately not
# matching a fence anywhere inside a larger string: text before/after the
# fence means this isn't the narrow "vendor wrapped its JSON in markdown"
# case this exists for, and should still surface as a validation failure.
_MARKDOWN_JSON_FENCE_RE = re.compile(r"```(?:json)?[ \t]*\n(.*?)\n```[ \t]*", re.DOTALL | re.IGNORECASE)


def _strip_markdown_json_fence(text: str) -> str:
    match = _MARKDOWN_JSON_FENCE_RE.fullmatch(text.strip())
    return match.group(1) if match else text


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
    (SPEC.md S4) - never a silent ``None`` or empty string. This applies
    identically whether the failures are provider-outage failures or, with
    ``schema=`` given, every provider's text failing schema validation - no
    separate "everyone failed schema validation" exception exists.

    ``complete()``'s return type depends on whether ``schema=`` is given
    (SPEC.md S3: "optional - validates before returning"): plain ``str``
    text when omitted, or the validated ``schema`` instance when provided -
    never raw unvalidated text alongside a schema request.

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

        # Duplicates are rejected here, not just left to "work" by retrying
        # the same provider instance twice. This isn't a style preference -
        # it enforces the Day 1 decision, documented on ProviderServerError,
        # that v1 does no same-provider retry (a rate-limited or 5xx-ing
        # provider won't fix itself in the next few seconds; SPEC.md S4).
        # Allowing a duplicate name in fallback_order would let that decision
        # be silently bypassed through a different mechanism than the one it
        # was originally about. If same-provider retry is wanted later, it
        # should be a deliberate v2 feature (e.g. a retry-count param), not
        # an emergent side effect of unvalidated list contents.
        duplicates = {name for name in fallback_order if fallback_order.count(name) > 1}
        if duplicates:
            raise ValueError(f"fallback_order contains duplicate provider name(s): {sorted(duplicates)}")

        by_name = {p.name: p for p in providers}
        unknown = [name for name in fallback_order if name not in by_name]
        if unknown:
            raise ValueError(f"fallback_order references provider(s) not in providers: {unknown}")

        self.providers = list(providers)
        self.fallback_order = list(fallback_order)
        self._by_name = by_name

    def complete(self, messages: list[Message], schema: type[_SchemaT] | None = None) -> str | _SchemaT:
        """Return the first successful completion, trying providers in
        ``fallback_order``.

        With ``schema=`` given, each provider's raw text must parse as JSON
        and validate against ``schema`` before it counts as success; a
        provider whose text fails that step is treated the same as any
        other retryable provider failure (see class docstring) and the
        Router moves on to the next provider.

        What advances to the next provider, what doesn't, what's returned,
        and what happens if every provider fails: see the ``Router`` class
        docstring.
        """
        failures: list[ProviderError] = []
        for name in self.fallback_order:
            provider = self._by_name[name]
            try:
                text = provider.complete(messages)
            except _RETRYABLE as exc:
                failures.append(exc)
                continue

            if schema is None:
                return text

            try:
                return schema.model_validate_json(_strip_markdown_json_fence(text))
            except ValidationError as exc:
                failures.append(ProviderResponseValidationError(name, str(exc), original=exc))

        raise AllProvidersFailedError(failures)
