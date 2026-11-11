# Changelog

## 0.1.1

- Disable SDK default retries so fallback happens immediately, matching the README.
  The Groq, OpenAI and Anthropic SDK clients are now created with `max_retries=0`. Before,
  each SDK retried a 429, 5xx or timeout twice on its own (with backoff) before swapLLM saw
  the failure, so one failing provider cost 3 HTTP requests and over a second before fallback.
  New tests (`tests/test_no_sdk_retries.py`) assert one failure sends exactly one request.

## 0.1.0

- First release: `Router.complete()` across Groq, OpenAI and Anthropic with automatic fallback,
  a typed exception hierarchy, and optional Pydantic schema validation.
