# nthlayer-common

Shared utilities for the [NthLayer](https://nthlayer.io) ecosystem. Provides the unified LLM interface, provider infrastructure, identity resolution, and data models used by all ecosystem components.

## Install

```bash
pip install nthlayer-common
```

## LLM Interface

Model-agnostic — one function, any provider. No LiteLLM, no SDKs. Direct HTTP calls via `httpx`.

```python
from nthlayer_common import llm_call

result = llm_call(
    system="You are a triage agent...",
    user="Evaluate this incident...",
)
print(result.text)
```

### Provider support

Two API formats cover the entire market:

| Provider | Model format | API |
|----------|-------------|-----|
| Anthropic | `anthropic/claude-sonnet-4-20250514` | Messages API |
| OpenAI | `openai/gpt-4o` | Chat Completions |
| Ollama | `ollama/llama3.1` | Chat Completions |
| Azure | `azure/my-deployment` | Chat Completions |
| Together | `together/meta-llama/Llama-3-70b` | Chat Completions |
| Groq | `groq/llama-3.1-70b-versatile` | Chat Completions |
| Mistral | `mistral/mistral-large-latest` | Chat Completions |
| vLLM | `vllm/my-model` | Chat Completions |
| LM Studio | `lmstudio/my-model` | Chat Completions |

### Configuration

```bash
NTHLAYER_MODEL="anthropic/claude-sonnet-4-20250514"   # default
NTHLAYER_LLM_TIMEOUT="60"                              # seconds
ANTHROPIC_API_KEY="sk-ant-..."                         # for anthropic/* models
OPENAI_API_KEY="sk-..."                                # for openai/*, together/*, groq/*, etc.
OPENAI_API_BASE="http://localhost:11434/v1"            # override endpoint URL
```

### Retry & resilience

Built-in retry with exponential backoff and jitter for transient errors (429, 502, 503, timeouts). Respects `Retry-After` headers. Permanent errors (400, 401, 403) fail immediately. Default: 3 retries.

## Provider Infrastructure

Shared async providers for infrastructure services, migrated from nthlayer-generate so all ecosystem components use the same clients:

- **`PrometheusProvider`** — query, query_range, get_sli_value, health_check
- **`GrafanaProvider`** — dashboard CRUD, datasource management, folder operations
- **`PagerDutyProvider`** — service, team, and escalation policy management
- **`MimirRulerProvider`** — Prometheus rule push to Grafana Mimir
- **`ProviderRegistry`** — register/create/list providers by name

## Identity Resolution

Cross-provider service name normalization and ownership attribution:

- **`IdentityResolver`** — 7-strategy resolution (explicit mapping → external ID → exact → alias → normalized → fuzzy → attribute correlation)
- **`normalize_service_name()`** — strips env suffixes, version tags, Java package prefixes, type suffixes
- **`OwnershipResolver`** — queries multiple ownership sources concurrently, selects highest-confidence signal
- **Ownership providers** — Backstage, Kubernetes, PagerDuty (live); CODEOWNERS, Declared (static, in nthlayer)

## HTTP Clients

Shared HTTP client infrastructure with per-instance retry and circuit breaker:

- **`BaseHTTPClient`** — httpx-based with configurable retry (tenacity) + circuit breaker
- **`CortexClient`** — Cortex API client
- **`PagerDutyClient`** — PagerDuty REST API client
- **`SlackAPIClient`** — Slack Web API (token-based, `post_message`)

## Slack Notifications

- **`SlackNotifier`** — Block Kit messages via incoming webhook (fail-open, never blocks pipelines)
- **`SlackWebClient`** — Web API for interactive messages (buttons, message updates, signature verification)

## Error Handling

Unified error hierarchy for the entire ecosystem:

- **`ExitCode`** — SUCCESS=0, WARNING=1, BLOCKED=2, CONFIG_ERROR=10, PROVIDER_ERROR=11, VALIDATION_ERROR=12
- **`NthLayerError`** → `ConfigurationError`, `ProviderError`, `ValidationError`, `BlockedError`
- **`@main_with_error_handling()`** — decorator for CLI main functions with automatic exit code conversion

## Tier Definitions

Single source of truth for service tier configuration:

- **`Tier`** — CRITICAL, STANDARD, LOW (with legacy aliases tier-1/2/3)
- **`TIER_CONFIGS`** — availability targets, latency thresholds, error budget warning/blocking percentages, PagerDuty urgency
- **`normalize_tier()`**, **`get_tier_config()`**, **`get_slo_targets()`**

## Data Models

Shared Pydantic/dataclass models used across the ecosystem:

- **SLO models** — `SLO`, `ErrorBudget`, `SLOStatus`, `TimeWindow`
- **Dependency models** — `DependencyGraph`, `DependencyType`, `BlastRadiusResult`
- **Domain models** — `Run`, `Finding`, `Team`, `Service`
- **Gate models** — `GateResult`, `GatePolicy`, `DeploymentGateCheck`

## Prompt Loader

Shared YAML prompt loader for agentic components:

- **`load_prompt(path)`** — reads YAML, renders schema block into system prompt
- **`render_user_prompt(template, **kwargs)`** — simple `{{ variable }}` interpolation
- **`validate_response(data, schema)`** — validates model output against expected schema

## Typed package

Ships with a `py.typed` marker (PEP 561) — mypy and pyright use inline type annotations directly without requiring a separate stub package.

## License

Apache 2.0
