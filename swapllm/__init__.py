"""swapllm - swap between LLM providers through one interface.

Day 3 scope (SPEC.md S8): Router adds fallback orchestration on top of Day
1/2's three normalized adapters - rate-limit/timeout/5xx/invalid-response
failures advance to the next provider in ``fallback_order``, everything else
propagates immediately. Schema validation (SPEC.md S3-S4) is Day 4 scope and
not implemented yet.
"""

from .exceptions import (
    AllProvidersFailedError,
    ProviderError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderResponseValidationError,
    ProviderServerError,
    ProviderTimeoutError,
    SwapLLMError,
)
from .providers import AnthropicProvider, GroqProvider, Message, OpenAIProvider, Provider
from .router import Router

__all__ = [
    "GroqProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "Message",
    "Provider",
    "Router",
    "SwapLLMError",
    "ProviderError",
    "ProviderRateLimitError",
    "ProviderTimeoutError",
    "ProviderServerError",
    "ProviderResponseValidationError",
    "ProviderRequestError",
    "AllProvidersFailedError",
]
