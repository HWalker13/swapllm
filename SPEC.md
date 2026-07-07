# swapllm — Full Project Rundown
*(working name locked — verified available on PyPI as of today; `llm-router`, `anyllm`, `llmswap`, `model-router`, and `routellm` are all already taken)*

This is the spec — read this before writing any code, and treat it as the single source of truth if a session gets picked back up cold.

---

## 1. What This Is

A standalone, published Python package that lets any application swap between LLM providers (Groq, OpenAI, Anthropic) through one stable interface, with automatic fallback on rate-limit/timeout/outage and schema-validated responses. Not a new idea — it's the AI-provider abstraction layer that already existed inside Pantry Pal, extracted, generalized, and shipped as something anyone can `pip install`, not just a file buried in one app's `services/` folder.

**Why this exists (the resume logic, stated plainly):** Pantry Pal itself is retiring — "full-stack CRUD + AI wrapper" is redundant signal next to Patch and Lovely. But the abstraction layer inside it was the one genuinely interesting piece of engineering, and it's *already proven* across three separate projects (Pantry Pal, Patch's BaaS routing, Lovely's dual-provider config) — see Provider Abstraction Pattern. Publishing it standalone turns "I built this pattern three times" into "I built this pattern three times, then packaged it so I only had to build it once."

---

## 2. Scope for v1 (ship in 3–5 days — do not let this creep)

**In scope:**
- Sync-only API (async is an explicit stretch goal, not v1 — see §7)
- Three provider adapters: Groq, OpenAI, Anthropic
- Plain text/chat completion only — no function-calling, no streaming, no vision
- Config-based fallback order with automatic retry on rate-limit (429), timeout, and 5xx
- Optional Pydantic schema validation on the response before it's returned to the caller
- Pytest suite with mocked provider responses (no real API calls in CI)
- README quickstart, MIT license, published to PyPI as `swapllm`

**Explicitly out of scope for v1** (real providers differ in more than response shape — function-calling formats, streaming semantics, and token counting are all genuinely different problems, and trying to abstract all of it at once is how a 3-day project becomes a 3-week one):
- Streaming responses
- Function/tool calling
- Async support
- Usage/cost tracking (Lovely already solved this in its own domain — worth revisiting as a v2 feature, not now)
- A CLI

---

## 3. The Interface (the actual design)

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

**Why a single `.complete()` method, not provider-specific methods:** the entire point is that calling code never needs to know which provider actually answered. If the interface leaked provider-specific method names or kwargs, you'd be back to vendor lock-in with extra steps.

**Why the schema param is optional, not required:** forcing every caller to define a Pydantic model is friction that would kill adoption for the simple "just get me a string back" case. Optional validation is the same design decision Pantry Pal made — validate when it matters (structured recipe data), skip it when it doesn't (a plain chat reply).

---

## 4. Fallback Logic (the part that actually needs to be correct)

- On `429` (rate limit) or timeout from the current provider, retry against the next provider in `fallback_order` — not the same one again, since a rate-limited provider won't magically un-limit itself in the next few seconds
- A provider that fails validation (returns malformed JSON when a schema was requested) is treated the same as a hard failure and triggers fallback — a provider technically "responding" with garbage isn't success
- If every configured provider fails, raise one typed exception (`AllProvidersFailedError`) carrying the individual failure reasons — never silently return `None` or an empty string, since that's exactly the "fails closed but invisibly" bug pattern that caused a real conversation-history bug in Lovely

---

## 5. Testing Strategy

Contract tests, not live API tests, for CI: mock each provider's HTTP response and assert the *normalized* output shape matches across all three, regardless of how differently each vendor structures its raw response. This is the test suite's actual job — proving the abstraction genuinely abstracts, not just that each adapter can parse its own vendor's JSON.

A separate, manually-triggered "live smoke test" (behind an env flag requiring real API keys) hits the actual APIs once before a release, to catch the case where a vendor changed their response shape since the mocks were written.

---

## 6. Packaging & Publishing

- `pyproject.toml` (not legacy `setup.py`)
- MIT license — matters for a package other people might actually depend on
- Semantic versioning starting at `0.1.0`
- Publish via `python -m build` + `twine upload`
- README needs: install command, the four-line quickstart above, and one line on *why* this exists (provider swap without touching call sites) — recruiters and other devs alike should be able to understand the point in under 30 seconds

---

## 7. Stretch Goals (post-v1 — do not start these until v1 ships)

- Async support (`httpx`-based, mirroring the sync interface)
- Streaming responses
- A fourth provider (likely Gemini, since it rounds out "the big four")
- Basic usage/cost logging, reusing the *idea* (not the code) from Lovely's per-user/global spend caps

---

## 8. Timeline

| Day | Milestone |
|---|---|
| 1 | Provider interface (`Protocol`/ABC) + Groq adapter + first passing tests |
| 2 | OpenAI + Anthropic adapters, normalized response shape confirmed across all three |
| 3 | Fallback logic — rate-limit/timeout detection, retry-to-next-provider, `AllProvidersFailedError` |
| 4 | Schema validation layer + full contract test suite |
| 5 | README, `pyproject.toml`, PyPI publish |

---

## 9. What Goes on the Resume When This Ships

- Extracted Pantry Pal's AI-provider abstraction into a standalone open-source Python package (PyPI), letting any app swap between Groq/OpenAI/Anthropic with a single config change and automatic fallback on rate-limit or outage
- Wrote a pytest suite validating response-contract parity across all three providers, plus a README quickstart, turning an internal implementation detail into a reusable public tool

## Links
- Origin: Pantry Pal (retired, pattern extracted)
- Same pattern, different domain: Patch's BaaS abstraction, Lovely's dual-AI-provider config
- Concept: Provider Abstraction Pattern
