# swapllm

Swap between LLM providers — Groq, OpenAI, Anthropic — through one interface, with automatic fallback on rate-limit/timeout/outage and optional schema-validated responses. Write your call site once against `Router.complete()`, and changing providers (or falling back to a second one when the first is down) is a config change, not a code change.

## Install

```bash
pip install swapllm
```

## Quickstart

```python
from swapllm import Router, GroqProvider, OpenAIProvider, AnthropicProvider

router = Router(
    providers=[
        GroqProvider(api_key=..., model="llama-3.1-70b"),
        OpenAIProvider(api_key=..., model="gpt-4o-mini"),
        AnthropicProvider(api_key=..., model="claude-haiku-4-5"),
    ],
    fallback_order=["groq", "openai", "anthropic"],
)

response = router.complete(
    messages=[{"role": "user", "content": "..."}],
    schema=MyPydanticSchema,   # optional — validates before returning
)
```

## Fallback behavior

`Router.complete()` tries each provider in `fallback_order`, in order, and returns the first successful completion.

**Triggers fallback to the next provider:**
- Rate limit (`429`)
- Timeout, or the request never reaching the provider at all
- A `5xx` from the provider
- A response that fails schema validation (when `schema=` is given) — a provider "responding" with unusable content isn't success

**Does not trigger fallback — propagates immediately instead:**
- `ProviderRequestError` (e.g. `400`/`401`/`403`, a malformed request, a bad API key) — this is almost always a caller misconfiguration, and switching providers would silently mask it rather than surface it
- A plain `ValueError` from malformed input (e.g. multiple or misplaced `system` messages) — same reasoning: a different provider won't fix bad input

There is also no same-provider retry: a rate-limited or 5xx-ing provider won't un-limit itself in the next few seconds, so a failure always advances to the *next* provider, never the same one again. If every provider in `fallback_order` fails, `Router.complete()` raises `AllProvidersFailedError` with every provider's individual failure reason attached — it never silently returns `None` or an empty string.

## Schema validation

`schema=` is optional. Pass a Pydantic model and `Router.complete()` parses the winning provider's text as JSON, validates it against the schema, and returns the validated model instance instead of a raw string. A leading/trailing markdown code fence (```` ```json ... ``` ````) is stripped first, since providers without strict JSON mode routinely wrap structured output that way.

Validation failure is treated as a provider failure, not a caller error — it triggers fallback to the next provider like any other retryable failure, since a provider returning malformed JSON isn't meaningfully different from a provider returning nothing at all.

Skip `schema=` entirely for the common case of just wanting a chat reply back as a string.

## Testing / contributing

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Tests are contract tests against mocked provider HTTP responses — no real API calls are made, and none are required to run the suite.

## License

MIT — see [LICENSE](LICENSE).
