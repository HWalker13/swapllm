"""swapllm - swap between LLM providers through one interface.

Day 2 scope (SPEC.md S8): OpenAI + Anthropic adapters added alongside Day 1's
Groq adapter, all three normalized through the same exception taxonomy.
Router lands on Day 3 and is deliberately not exported yet.
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

__all__ = [
    "GroqProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "Message",
    "Provider",
    "SwapLLMError",
    "ProviderError",
    "ProviderRateLimitError",
    "ProviderTimeoutError",
    "ProviderServerError",
    "ProviderResponseValidationError",
    "ProviderRequestError",
    "AllProvidersFailedError",
]
