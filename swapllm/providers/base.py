"""The provider interface every adapter (Groq, OpenAI, Anthropic) implements.

Design choice: ``Protocol``, not ``ABC``.

Adapters hold no shared state or shared behavior worth inheriting - each one
wraps a different vendor SDK client (``groq.Groq``, ``openai.OpenAI``,
``anthropic.Anthropic``), constructed differently, with its own
``VendorExceptionMap`` (see ``swapllm._normalize``). There is no template
method or common ``__init__`` logic to factor upward into a base class; an
ABC here would exist only to declare a shape, which is exactly what
``Protocol`` is for. ``Protocol`` also lets a caller pass any object satisfying
this shape (e.g. a hand-rolled test double) without it needing to inherit from
a swapllm base class - structural typing, not nominal typing. If real shared
behavior emerges later (e.g. a common retry wrapper all adapters call into),
that's a reason to introduce a small ABC or mixin at that point, not before.
"""

from __future__ import annotations

from typing import Protocol, TypedDict, runtime_checkable


class Message(TypedDict):
    role: str
    content: str


def validate_messages(messages: list[Message]) -> None:
    """Reject message shapes a vendor SDK would otherwise handle inconsistently.

    Anthropic's Messages API takes `system` as a separate top-level field, so
    an adapter has to pull system-role entries out of `messages` itself
    (see AnthropicProvider._split_system). Groq/OpenAI's OpenAI-compatible
    endpoints have no such extraction step and would just pass multiple or
    misplaced system-role entries straight through to the vendor - meaning
    the same input (two system messages, or one not at index 0) would
    silently behave differently per provider. That breaks the one guarantee
    this package exists to provide: callers can swap providers without the
    behavior changing for the same input (SPEC.md S4).

    It lives here in the shared base module, and every adapter calls it
    identically as the first line of ``complete()``, precisely so callers get
    the same rejection behavior regardless of provider - duplicating this
    check into each adapter instead would let the three copies drift apart
    over time, which is the exact failure mode this function exists to rule
    out.

    A caller-input error, not a vendor failure - deliberately a plain
    ValueError rather than a ProviderError subclass, since it must NOT
    trigger Router fallback the way a vendor failure does (misplaced system
    messages are not a provider outage, and failing over would just retry
    the same bad input against the next provider).
    """
    system_count = sum(1 for m in messages if m["role"] == "system")
    if system_count > 1:
        raise ValueError("multiple system messages are not supported")
    if system_count == 1 and messages[0]["role"] != "system":
        raise ValueError("a system message must be the first message")


@runtime_checkable
class Provider(Protocol):
    """A single LLM provider adapter.

    name: short id matching entries in ``Router(fallback_order=[...])``,
    e.g. ``"groq"``.
    """

    name: str

    def complete(self, messages: list[Message]) -> str:
        """Send ``messages`` to the vendor and return the raw text completion.

        Must raise only ``swapllm.exceptions.ProviderError`` subclasses on
        failure - never a raw vendor SDK exception. See
        ``swapllm._normalize.normalize_exception`` for how adapters convert
        their vendor's exceptions before they reach this boundary.

        Schema validation is explicitly not this method's job: it always
        returns unvalidated text, and the Router (SPEC.md S3-S4) applies the
        optional ``schema=`` on top of whatever provider actually answered.
        """
        ...
