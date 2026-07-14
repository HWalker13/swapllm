"""swapllm - swap between LLM providers through one interface.

Day 4 scope (SPEC.md S8): Router.complete()'s optional ``schema=`` param
validates a provider's text as JSON against a Pydantic model before
returning it, reusing Day 1/2's ProviderResponseValidationError so a
schema-validation failure triggers the same fallback (SPEC.md S4) as
Day 3's rate-limit/timeout/5xx/invalid-response failures - no separate
exception type or Router control-flow change needed.
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
