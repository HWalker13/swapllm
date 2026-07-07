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
