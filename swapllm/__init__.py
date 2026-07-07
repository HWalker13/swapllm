"""swapllm - swap between LLM providers through one interface.

Day 1 scope (SPEC.md S8): provider interface + Groq adapter + exception
normalization only. Router, OpenAIProvider, and AnthropicProvider land on
Day 2-3 and are deliberately not exported yet.
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
from .providers import GroqProvider, Message, Provider

__all__ = [
    "GroqProvider",
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
