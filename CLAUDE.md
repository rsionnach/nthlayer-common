# nthlayer-common

Shared utilities package for the NthLayer ecosystem. Provides the unified LLM interface used by all agentic components (measure, correlate, respond, learn).

<!-- AUTO-MANAGED: module-description -->
## Purpose

- Single place for cross-cutting utilities shared across NthLayer runtime components
- No third-party LLM libraries — direct `httpx` calls to provider APIs only
- Public API: `llm_call`, `LLMResponse`, `LLMError` (from `nthlayer_common`)
<!-- END AUTO-MANAGED -->

<!-- AUTO-MANAGED: architecture -->
## Structure

```
src/nthlayer_common/
    __init__.py     # Re-exports: llm_call, LLMResponse, LLMError
    llm.py          # Unified LLM wrapper
tests/
    test_llm.py     # Unit tests (mocked httpx.post)
```
<!-- END AUTO-MANAGED -->

<!-- AUTO-MANAGED: build-commands -->
## Commands

```bash
# Run tests
uv run pytest

# Lint
uv run ruff check src/ tests/
```
<!-- END AUTO-MANAGED -->

<!-- AUTO-MANAGED: conventions -->
## LLM Interface Conventions

**Model format:** `"provider/model-name"` — e.g. `anthropic/claude-sonnet-4-20250514`, `openai/gpt-4o`, `ollama/llama3.1`

**Provider routing:**
- `anthropic/*` → Anthropic Messages API (`api.anthropic.com/v1/messages`)
- Everything else → OpenAI-compatible Chat Completions API

**Environment variables:**
- `NTHLAYER_MODEL` — override default model (default: `anthropic/claude-sonnet-4-20250514`)
- `NTHLAYER_LLM_TIMEOUT` — request timeout in seconds (default: 60)
- `ANTHROPIC_API_KEY` — required for `anthropic/*` models
- `OPENAI_API_KEY` — for OpenAI-compatible providers (optional for Ollama/vLLM)
- `OPENAI_API_BASE` — override endpoint URL for any provider
- `AZURE_OPENAI_ENDPOINT` — Azure OpenAI resource URL

**Default base URLs by provider:**
- `openai` → `https://api.openai.com/v1`
- `ollama` → `http://localhost:11434/v1`
- `vllm` → `http://localhost:8000/v1`
- `lmstudio` → `http://localhost:1234/v1`
- `together` → `https://api.together.xyz/v1`
- `groq` → `https://api.groq.com/openai/v1`
- `mistral` → `https://api.mistral.ai/v1`

**Bare model name guessing** (`_guess_provider`): `claude*` → anthropic, `gpt*/o1*/o3*` → openai, `llama*/mistral*/gemma*` → ollama, else → openai.

**LLMResponse fields:** `text`, `model`, `provider` — all strings.

**LLMError:** carries `provider`, `model`, `cause` attributes alongside message.
<!-- END AUTO-MANAGED -->

<!-- AUTO-MANAGED: dependencies -->
## Dependencies

- `httpx>=0.27` — HTTP client for all provider API calls (no `requests`, no LiteLLM)
- `pytest>=8.0` (dev) — test framework
- `ruff>=0.8` (dev) — linter
<!-- END AUTO-MANAGED -->
