# nthlayer-common

Shared utilities package for the NthLayer ecosystem. Provides the unified LLM interface, provider infrastructure, identity resolution, error hierarchy, tier definitions, and data models used by all ecosystem components.

<!-- AUTO-MANAGED: module-description -->
## Purpose

- Single place for cross-cutting utilities shared across all NthLayer components
- No third-party LLM libraries — direct `httpx` calls to provider APIs only
- Public API: `llm_call`, `LLMResponse`, `LLMError` (from `nthlayer_common`)
- Ships `py.typed` marker (PEP 561) — mypy/pyright use inline annotations directly
- License: Apache 2.0
<!-- END AUTO-MANAGED -->

<!-- AUTO-MANAGED: architecture -->
## Structure

```
src/nthlayer_common/
    __init__.py          # Re-exports: llm_call, LLMResponse, LLMError
    llm.py               # Unified LLM wrapper (model-agnostic, httpx-based)
    errors.py            # Error hierarchy: NthLayerError, ExitCode, @main_with_error_handling
    tiers.py             # Tier definitions: Tier, TIER_CONFIGS, normalize_tier, get_slo_targets
    slo_models.py        # SLO, ErrorBudget, SLOStatus, TimeWindow
    dependency_models.py # DependencyGraph, DependencyType, BlastRadiusResult
    domain_models.py     # Run, Finding, Team, Service
    gate_models.py       # GateResult, GatePolicy, DeploymentGateCheck
    slack.py             # SlackNotifier (webhook, fail-open)
    slack_web.py         # SlackWebClient (Web API: post_message, update_message, verify_signature)
    prompts.py           # load_prompt, render_user_prompt, validate_response
    parsing.py           # Shared parsing utilities
    explanation.py       # BudgetExplanation dataclass + format_explanation() (table/json/markdown); shared across observe/respond
    py.typed             # PEP 561 marker
    clients/
        base.py          # BaseHTTPClient (httpx, retry, circuit breaker)
        cortex.py        # CortexClient
        mimir.py         # MimirRulerProvider (canonical source); MimirRulerError, RulerPushResult — BaseHTTPClient subclass
        pagerduty.py     # PagerDutyClient
        slack.py         # SlackAPIClient
    providers/
        prometheus.py    # PrometheusProvider
        grafana.py       # GrafanaProvider
        pagerduty.py     # PagerDutyProvider
        mimir.py         # Re-export shim → nthlayer_common.clients.mimir (MimirRulerProvider, MimirRulerError, RulerPushResult, DEFAULT_USER_AGENT)
        registry.py      # ProviderRegistry
        base.py          # Provider base classes
        lock.py          # Provider locking utilities
    identity/
        models.py        # ServiceIdentity, IdentityMatch
        normalizer.py    # normalize_service_name, DEFAULT_RULES
        resolver.py      # IdentityResolver (7-strategy resolution)
        ownership.py     # OwnershipResolver, OwnershipSignal, OwnershipAttribution
        ownership_providers/  # Backstage, Kubernetes, PagerDuty providers
tests/
    test_llm.py
    test_errors.py
    test_tiers.py
    test_models.py
    test_providers.py
    test_identity.py
    test_slack.py
    test_prompts.py
    test_explanation.py
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

**LLMResponse fields:** `text` (str), `model` (str), `provider` (str), `input_tokens` (int|None), `output_tokens` (int|None) — `@dataclass`.

**LLMError:** carries `provider`, `model`, `cause`, `status_code` attributes alongside message.

**Retry behavior (`llm_call` default `retry=3`):**
- Transient errors retried with exponential backoff + full jitter: 429, 408, 502, 503, connection errors, timeouts
- Permanent errors fail immediately: 400, 401, 403, 404, 422
- `Retry-After` header parsed and respected; timeout budget checked before each sleep
- Pass `retry=0` to disable retries
- API key guard: raises `LLMError` if model string starts with `sk-ant-`, `sk-`, `key-`, or `Bearer `
<!-- END AUTO-MANAGED -->

<!-- AUTO-MANAGED: dependencies -->
## Dependencies

- `httpx>=0.27` — HTTP client for all provider API calls (no `requests`, no LiteLLM)
- `pyyaml>=6.0` — YAML parsing
- `structlog>=24.1.0` — structured logging throughout the package
- `pydantic>=2.7.0,<3.0.0` — data validation and serialization
- `pagerduty>=6.0.0,<7.0.0` — PagerDuty API client
- `cachetools>=5.3.0,<7.0.0` — in-memory caching utilities
- `tenacity>=8.2.3,<10.0.0` — retry with exponential backoff + jitter for LLM and HTTP client calls
- `circuitbreaker>=2.0.0,<3.0.0` — circuit breaker pattern for provider calls
- `pytest>=8.0` (dev) — test framework
- `pytest-asyncio>=0.23` (dev) — async test support
- `ruff>=0.8` (dev) — linter

## Public API Summary

**LLM:** `llm_call(system, user, model?, max_tokens=2000, timeout?, retry=3)` → `LLMResponse(text, model, provider, input_tokens, output_tokens)`

**Providers:** `PrometheusProvider`, `GrafanaProvider`, `PagerDutyProvider`, `MimirRulerProvider`, `ProviderRegistry`

**Identity:** `IdentityResolver` (7-strategy), `normalize_service_name()`, `OwnershipResolver`

**HTTP Clients:** `BaseHTTPClient`, `CortexClient`, `MimirRulerProvider` (canonical: `clients.mimir`), `PagerDutyClient`, `SlackAPIClient`

**Slack:** `SlackNotifier` (`slack.py` — Block Kit webhook, fail-open); `SlackWebClient` (`slack_web.py` — Web API: `post_message`, `update_message`, `verify_signature`; lazy httpx client, fail-open)

**Errors:** `NthLayerError` → `ConfigurationError`, `ProviderError`, `ValidationError`, `BlockedError`; `ExitCode` (SUCCESS=0, WARNING=1, BLOCKED=2, CONFIG_ERROR=10, PROVIDER_ERROR=11, VALIDATION_ERROR=12); `@main_with_error_handling()` decorator

**Tiers:** `Tier` (CRITICAL/STANDARD/LOW + tier-1/2/3 aliases), `TIER_CONFIGS`, `normalize_tier()`, `get_tier_config()`, `get_slo_targets()`

**Data Models:** SLO (`SLO`, `ErrorBudget`, `SLOStatus`, `TimeWindow`), Dependency (`DependencyGraph`, `DependencyType`, `BlastRadiusResult`), Domain (`Run`, `Finding`, `Team`, `Service`), Gate (`GateResult`, `GatePolicy`, `DeploymentGateCheck`)

**Prompts:** `load_prompt(path)`, `render_user_prompt(template, **kwargs)`, `validate_response(data, schema)`

**Explanation:** `BudgetExplanation(service, slo_name, headline, body, causes, recommended_actions, severity)` dataclass; `format_explanation(explanation, fmt)` → str (fmt: "table"|"json"|"markdown"); produced by nthlayer-observe, consumable by nthlayer-respond
<!-- END AUTO-MANAGED -->
