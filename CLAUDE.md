# nthlayer-common

Shared utilities package for the NthLayer ecosystem. Provides the unified LLM interface, provider infrastructure, identity resolution, error hierarchy, tier definitions, and data models used by all ecosystem components.

<!-- AUTO-MANAGED: module-description -->
## Purpose

- Single place for cross-cutting utilities shared across all NthLayer components
- Two LLM interfaces: `llm_call` (raw text, httpx-based) and `structured_call` (validated Pydantic models, Instructor-based)
- Public API: `llm_call`, `LLMResponse`, `LLMError`, `structured_call` (from `nthlayer_common.llm` / `nthlayer_common.llm_structured`)
- Ships `py.typed` marker (PEP 561) — mypy/pyright use inline annotations directly
- License: Apache 2.0
<!-- END AUTO-MANAGED -->

<!-- AUTO-MANAGED: architecture -->
## Structure

```
src/nthlayer_common/
    __init__.py          # Re-exports: errors (NthLayerError, ConfigurationError, ProviderError, ValidationError, BlockedError, WarningResult, DegradedError, TransientError, PermanentError, ExitCode, classify_http_error, main_with_error_handling, retry), llm (llm_call, LLMResponse, LLMError), parsing (clamp, strip_markdown_fences), prompts (PromptSpec, extract_confidence, load_prompt, render_user_prompt, validate_response), SlackNotifier, SlackWebClient
    config.py            # Unified config loading — Config dataclass (deployment/store/llm/prometheus/core/workers/bench); Config.load(), Config.from_dict(), Config.get(dotpath); bare/None-valued YAML sections coerced to {}; canonical import: from nthlayer_common.config import Config
    llm.py               # Unified LLM wrapper (model-agnostic, httpx-based) — returns raw text; lazy-imports llm_stub for CI fast-path
    llm_structured.py    # Structured LLM outputs via Instructor — structured_call() returns validated Pydantic models; lazy-imports llm_stub for CI fast-path
    llm_stub.py          # CI integration test stub — NTHLAYER_LLM_STUB=canned short-circuits llm_call/structured_call before any HTTP; role detection via system prompt markers; _STRUCTURED_FACTORIES registry keyed by response_model.__name__; DO NOT enable in production
    errors.py            # Error hierarchy: NthLayerError, ExitCode, @main_with_error_handling; tier-boundary errors: TransientError, PermanentError, DegradedError; classify_http_error(); @retry() decorator
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
    api_client.py        # CoreAPIClient — async httpx client for nthlayer-core API; APIResult(ok, status_code, data, error, detail); never raises on API errors; endpoints: health, verdicts (submit/get/get_verdicts/get_ancestors/get_descendants/resolve_outcome), assessments (submit/get_assessments), cases (create/get/get_cases/acquire_lease/release_lease/resolve_case), change-freezes, heartbeats, manifests (get_manifests/get_manifest), component-state, monitoring
    cloudevents.py       # CloudEvents v1.0 envelope helpers — wrap_verdict, wrap_assessment, parse_cloudevent, validate_cloudevent; type taxonomy frozen from v1.5 onwards (spec: NTHLAYER-TELEMETRY-ENVELOPE-v1 §3); ASSESSMENT_KINDS (public): {slo_status, judgment_slo_evaluation, burn_rate, drift_signal, portfolio_status, deploy_gate, dependency_graph, correlation_snapshot, topology_drift, contract_divergence, retrospective, calibration_signal}
    metrics.py           # Self-observability Prometheus metrics — cycle_duration_seconds, verdicts_written_total, assessments_written_total, heartbeats_emitted_total, llm_calls_total, errors_total, api_requests_total (labels: method/route/status — route=URL template, never raw path), store_size_bytes, wal_size_bytes, stuck_action_requests; render_metrics(), metrics_content_type(); canonical import: from nthlayer_common.metrics import ...; spec: NTHLAYER-COMMON-v1 §7.3
    telemetry.py         # OTel telemetry emission — emit_llm_event(*, model, provider, caller, ...) emits "nthlayer.llm.call" span event with gen_ai.* attributes; graceful no-op when OTel SDK not configured or no active recording span; is_otel_available() → bool; canonical import: from nthlayer_common.telemetry import emit_llm_event; spec: NTHLAYER-COMMON-v1 §3.4, §7
    py.typed             # PEP 561 marker
    clients/
        base.py          # BaseHTTPClient (httpx, retry, circuit breaker)
        cortex.py        # CortexClient
        mimir.py         # MimirRulerProvider (canonical source); MimirRulerError, RulerPushResult — BaseHTTPClient subclass
        pagerduty.py     # PagerDutyClient
        slack.py         # SlackAPIClient
    providers/
        prometheus.py    # PrometheusProvider — get_sli_value() returns None (empty result/malformed/non-numeric) vs 0.0 (total outage, opensrm-e1gk); get_sli_time_series() → list[{timestamp, sli_value, duration_seconds}]; _parse_step_to_seconds() suffix lookup dict (s/m/h/d) + numeric string fallback, default 300.0; health_check() uses /api/v1/query?query=up
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
    records/
        __init__.py      # Re-exports all public record types, hashing, store, verification, verdict_bridge, errors
        errors.py        # RecordStoreError, RecordStoreCorrupt, RecordStoreLocked, ChainForkError, InvalidTransitionError
        models.py        # Assessment, Verdict, Evaluation, Incident; Summaries; all enums; ZERO_HASH
        hashing.py       # canonical_json, compute_hash, verify_hash (SHA-256 content addressing)
        verdict_bridge.py  # build_decision_verdict(), write_decision_verdict(), hash_content() — shared factory and atomic writer for content-addressed Verdict records
        store.py         # DecisionRecordStore Protocol (structural typing interface; includes get_chain_tail)
        sqlite_store.py  # SQLiteDecisionRecordStore — WAL mode, thread-local conns, append-only with chain fork detection
        verification.py  # verify_chain, verify_incident; ChainVerificationResult, IncidentVerificationResult
    verdicts/
        __init__.py      # Re-exports all public verdict types, operations, store, serialisation (moved from nthlayer-learn)
        models.py        # Verdict, Producer, Subject, Judgment, Outcome, Lineage, Metadata, Override, GroundTruth, AccuracyReport; VALID_* frozensets including VALID_VERDICT_TYPES; TTL_DEFAULT=90d; v1.5 fields on Verdict: verdict_type, pipeline_latency_ms, chain_depth, parent_ids, service; cost_currency on Metadata
        core.py          # create(), link(), resolve(), supersede() — thread-safe ID generation "vrd-{date}-{uuid8}-{seq:05d}"
        store.py         # VerdictStore(ABC), MemoryStore, VerdictFilter, AccuracyFilter
        sqlite_store.py  # SQLiteVerdictStore — WAL mode, thread-local conns, atomic conditional UPDATE for resolve()
        serialise.py     # to_dict/to_json/from_dict/from_json — datetime↔ISO strings via dataclasses.asdict; to_dict renames timestamp→created_at and verdict_type→type (HTTP-canonical wire names for POST /verdicts); from_dict accepts both wire-canonical (type, created_at) and legacy internal names (verdict_type, timestamp) for compat with data written before opensrm-saun.1.2; field precedence: created_at canonical, explicit None falls back to legacy timestamp, empty string is malformed; round-trips v1.5 fields (verdict_type, pipeline_latency_ms, chain_depth, parent_ids, service, cost_currency)
    manifest/
        __init__.py      # Public API: load_manifest, is_manifest_file, ManifestLoadError, LegacyFormatWarning, OpenSRMParseError, OpenSRMV2ParseError; re-exports all model types and constants
        models.py        # Unified internal model (all dataclasses): ReliabilityManifest, SLODefinition, Dependency, Ownership, DeploymentConfig, ReliabilityContract, Instrumentation, and all sub-models; VALID_TIERS, VALID_SERVICE_TYPES, SERVICE_TYPE_ALIASES, JUDGMENT_SLO_TYPES, STANDARD_SLO_TYPES, VALID_EXHAUSTION_BEHAVIORS; SourceFormat, DependencyCriticality enums; SLODefinition.target canonical convention: 0-100 percentage for all SLO types (opensrm-5fff)
        target_validation.py  # Load-time SLO target convention validator (opensrm-5fff.1); TargetConventionWarning(UserWarning) — filterable via warnings.filterwarnings(action, category=TargetConventionWarning); warn_target_convention_mismatches(manifest) → None — called by load_manifest after every successful parse; heuristic: any target in (0, 1) range warns as likely ratio author error for both classical and judgment SLOs; target==1.0 ambiguous (silent), target<=0 or >100 out-of-range (different concern, skipped); never rejects
        v1_compat.py     # v1→v2 compat defaults: default_statistical_requirements(judgment_type), default_measurement(judgment_type, window), convert_v1_contract(service_name, availability, latency, judgment)
        openslo/
            __init__.py  # OpenSLO v1 subpackage
            parser.py    # parse_openslo_slos(slo_list, base_dir?) — inline + $ref resolution; ratioMetric/thresholdMetric indicator support; OpenSLOParseError
        parser/
            __init__.py  # parser subpackage
            _shared.py   # parse_observability(obs_data) -> Observability | None — shared observability parsing helper used by v1 and v2 parsers
            loader.py    # load_manifest(path, environment?, format="auto", suppress_deprecation_warning?) — auto-detects srm/v1/opensrm_v2/legacy; _find_template_dir walks up 10 dirs for .nthlayer/templates/ or templates/; legacy parser inline; calls warn_target_convention_mismatches(manifest) after every successful parse (all formats) — cross-subsystem drift check, never rejects; ManifestLoadError, LegacyFormatWarning
            v1.py        # parse_srm_v1(data, source_file?), parse_srm_v1_file(path), resolve_template(manifest_data, template_dir) → (data, warnings); SLO type inference dict + keyword fallback; _extract_indicator_query reads canonical indicator.query nested form only (top-level query not accepted); OpenSRMParseError
            v2.py        # parse_opensrm_v2(data, source_file?, base_dir?), parse_opensrm_v2_file(path); Backstage entity ref resolution; judgment SLO parsing (8 types); contract/dependency/instrumentation parsing; resolve_v2_template with append/replace override directives; OpenSRMV2ParseError
tests/
    test_llm.py
    test_errors.py
    test_tiers.py
    test_models.py
    test_providers.py          # TestGetSliValueNoneVsZero (5 async tests, opensrm-e1gk): empty→None, zero→0.0, short-tuple→None, non-numeric→None, real→float; TestPrometheusProvider: test_parse_step_to_seconds (5m/1h/30s/1d/unknown), test_parse_step_numeric_string (bare integers)
    test_identity.py
    test_slack.py
    test_prompts.py
    test_explanation.py
    test_verdicts_core.py
    test_verdicts_models.py
    test_verdicts_serialise.py
    test_verdicts_sqlite_concurrency.py
    test_verdicts_store.py
    test_llm_structured.py
    test_api_client.py
    test_cloudevents.py          # envelope helpers (wrap_verdict, wrap_assessment, parse_cloudevent, validate_cloudevent); TestWrapRealVerdict (opensrm-saun.1.2) locks end-to-end contract: verdict_create→to_dict→wrap_verdict must produce correct CE type never .unknown.v1; TestEnvelopeRoundTrip: wrap→unwrap→unmarshal preserves inner record; ASSESSMENT_KINDS public constant coverage
    test_config.py
    test_metrics.py
    test_telemetry.py
    test_tier_errors.py
    test_verdicts_v15_fields.py
    test_manifest_models.py
    test_manifest_parser.py
    test_manifest_real_specs.py  # regression: loads real demo/specs from disk (not synthetic fixtures); skipped if demo/specs/ absent; guards against parser silently dropping indicator.query; spec path resolution tries nthlayer/demo/specs (post-consolidation) then demo/specs fallback; includes test_fraud_detect_judgment_slo_round_trips asserting reversal_rate SLO has judgment_type=="reversal_rate" and indicator_query contains gen_ai_overrides_total
    test_llm_stub.py             # 26 tests for NTHLAYER_LLM_STUB=canned: role detection, all 4 agent shapes, structured callers, env-var variants, name-collision guard
    test_target_validation.py    # 17 tests (5 classes) pinning SLO target convention (opensrm-5fff.1): TestPercentageRange (99.9/99.99/50%/98.5/99% clean — canonical percentage passes silently), TestRatioWarns (0.999/0.5/0.985/0.015 → warn with "ratio" + suggested *100 correction, both classical and judgment), TestEdgeCases (target==1.0 silent; 0/negative/above-100 skipped), TestMixedManifests (one clean + one dirty warns once; warning names service), TestRealDemoSpecs (fraud-detect.yaml loads with 0 TargetConventionWarnings post-migration — regression guard)
    records/
        conftest.py      # shared builders: build_test_assessment, build_test_verdict, build_test_evaluation (correct content-addressed hashes)
        test_hashing.py
        test_models.py
        test_store.py
        test_verification.py
        test_verdict_bridge.py
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

## CI integration test stub (`NTHLAYER_LLM_STUB=canned`)

Module: `nthlayer_common/llm_stub.py`. Added for opensrm-saun.1 (three-tier integration test). Setting `NTHLAYER_LLM_STUB=canned` in the environment short-circuits both `llm_call()` and `structured_call()` / `structured_call_with_usage()` *before any HTTP request* and returns a deterministic canned response. **Not a behavioural fake** — every call of a given role returns the same data regardless of input. Purpose is to exercise wiring (verdict shape, lineage propagation, store writes) without a real LLM API key. **Do not enable in production.**

**Activation point:** lazy `from nthlayer_common.llm_stub import …` inside the `llm_call` / `structured_call` / `structured_call_with_usage` body — the stub module is not imported during normal use; this keeps the import graph cycle-safe with `nthlayer_common.llm`.

**`llm_call()` raw-text dispatch** — role detected via case-insensitive substring match in the system prompt (markers in `_ROLE_MARKERS`):

| System prompt contains | Canned JSON shape |
|---|---|
| `"you are a triage agent"` | `{severity, blast_radius, affected_slos, assigned_team, reasoning, confidence}` |
| `"you are a communication agent"` | `{updates: [{channel, update_type, content}], reasoning, confidence}` |
| `"you are an investigation agent"` | `{hypotheses: [{description, confidence, evidence, change_candidate}], root_cause, root_cause_confidence, reasoning, confidence}` |
| `"you are a remediation agent"` | `{proposed_action: "rollback", target: "fraud-detect", risk_assessment, requires_human_approval: true, reasoning, confidence}` |
| (no marker) | `{reasoning, confidence}` |

All canned text is JSON shaped to match the schema each respond agent's `_parse_json` expects (so e.g. `RemediationAgent`'s safe-action registry check accepts `"rollback"` because that action exists in the registry with `requires_approval=true`).

**`structured_call()` / `structured_call_with_usage()` dispatch** — registry keyed by `response_model.__name__`:

| `response_model` | Returns |
|---|---|
| `EvaluationResult` (measure evaluator) | one passing `DimensionScore` (score=0.85), `confidence=0.8` |
| `SnapshotSummary` (correlate snapshot summary) | stub summary string, empty `notable_omissions` |
| `TriageResponse` (respond triage agent, P3-E.2) | severity=2, blast_radius=["fraud-detect"], assigned_team="payments", confidence=0.7 |
| `InvestigationResponse` (respond investigation agent, P3-E.2) | single deploy-regression hypothesis, root_cause set, confidence=0.8 |
| `CommunicationResponse` (respond communication agent, P3-E.2) | single status_page update, update_type="initial", confidence=0.7 |
| `RemediationResponse` (respond remediation agent, P3-E.2) | proposed_action="rollback", target="fraud-detect", requires_human_approval=True, confidence=0.7 |
| (anything else) | `NotImplementedError` — prevents new structured-call sites silently producing garbage |

`structured_call_with_usage()` wraps the canned model in `StructuredCallResult(data=…, usage=StructuredCallUsage(0, 0))`.

**Public helpers:** `is_stub_enabled() -> bool`, `stub_text_response(system, model) -> LLMResponse`, `stub_structured_response(response_model) -> T`. Currently only the `llm_call`/`structured_call` wrappers consume them.

**Adding coverage:**
- New agent role → extend `_ROLE_MARKERS` (ordered tuple) and `_TEXT_BY_ROLE`.
- New structured-call site → register a factory in `_STRUCTURED_FACTORIES`.

**Tests:** `tests/test_llm_stub.py` — 26 tests covering role detection (incl. `None` system, case-insensitive marker match), all four respond agent shapes, both structured callers, env-var case/whitespace variants (`CANNED`, ` canned `, `canned\n`), env-var-unset preserves HTTP path, unknown structured model raises clearly, name-collision with incompatible shape raises with qualified path.

## SLO target convention

**0-100 percentage canonical** (opensrm-5fff, decision recorded at `nthlayer/docs/superpowers/specs/2026-05-06-slo-target-convention-decision.md`).

`manifest.SLODefinition.target` uses 0-100 percentage convention across all NthLayer-internal consumers. Examples:

- Classical SLO: `availability target=99.9` for 99.9% availability
- Judgment SLO: `reversal_rate target=98.5` (SLI is `1 - reversal_rate * 100`; breach when SLI drops below 98.5)

`observe.collector` and `measure.worker` both compute against percentage targets directly. The `measure.worker._evaluate_slo` path scales the Prometheus SLI (always 0.0-1.0 ratio from PromQL) to percentage with `current_pct = current_value * 100` before comparing to the target.

The OpenSLO surface (`nthlayer_common.slo_models.SLO`) uses 0.0-1.0 ratio. Conversion happens at the boundary in `nthlayer-generate.slos.pipeline._build_slo_from_manifest` which divides by 100.0 unconditionally.

A load-time validator in `nthlayer_common.manifest.target_validation` flags targets in the (0, 1) range as likely ratio author errors via `TargetConventionWarning` (subclass of `UserWarning`). Filterable via `warnings.filterwarnings(action, category=TargetConventionWarning)`. Tests in `tests/test_target_validation.py` pin behaviour at the boundaries.

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
- `instructor>=1.5` — JSON schema enforcement + validation retry for structured LLM outputs
- `anthropic>=0.40` — Anthropic SDK (used by `structured_call` via Instructor)
- `openai>=1.50` — OpenAI SDK (used by `structured_call` via Instructor)
- `opentelemetry-api>=1.28` — OTel API for cost-accounting telemetry (`telemetry.py`); graceful no-op if SDK not configured
- `prometheus-client>=0.21` — Prometheus metrics exposition for self-observability (`metrics.py`)
- `pygments>=2.20.0` — **transitive security pin** (CVE-2026-4539, bead opensrm-9uow.3); `rich ← instructor/typer` pulls pygments; pin explicit because `rich`'s own lower bound doesn't yet carry the fix; remove once it does
- `pytest>=8.0` (dev) — test framework
- `pytest-asyncio>=0.23` (dev) — async test support
- `ruff>=0.8` (dev) — linter

## Public API Summary

**Config (`nthlayer_common.config`):** `Config.load(path?) -> Config` — resolution order: explicit path → `NTHLAYER_CONFIG` env var → `./nthlayer.yaml` → empty defaults; applies env overrides after loading. `Config.from_dict(data)` for tests (`source="dict"`). `Config.get(dotpath, default=None)` for nested access (e.g. `config.get("workers.observe.cycle_interval_seconds")`). Convenience properties: `store_path` (default `"nthlayer.db"`), `deployment_id` (default `"default"`), `default_model` (default `"anthropic/claude-sonnet-4-20250514"`), `prometheus_url` (default `None`). Env overrides: `NTHLAYER_CONFIG` (file path), `NTHLAYER_STORE_PATH` (store.path), `NTHLAYER_MODEL` (llm.default_model), `NTHLAYER_DEPLOYMENT_ID` (deployment.id), `NTHLAYER_PROMETHEUS_URL` (prometheus.url). Spec: NTHLAYER-COMMON-v1 §10.

**LLM (raw text):** `llm_call(system, user, model?, max_tokens=2000, timeout?, retry=3)` → `LLMResponse(text, model, provider, input_tokens, output_tokens)`

**LLM (structured):** `structured_call(system, user, response_model, model?, max_tokens=2000, timeout?, max_retries=3)` → validated `BaseModel` instance; uses Instructor for JSON schema enforcement + retry on malformed responses; raises `LLMError` on provider errors or exhausted retries

**Providers:** `PrometheusProvider`, `GrafanaProvider`, `PagerDutyProvider`, `MimirRulerProvider`, `ProviderRegistry`

**Identity:** `IdentityResolver` (7-strategy), `normalize_service_name()`, `OwnershipResolver`

**HTTP Clients:** `BaseHTTPClient(base_url, *, timeout=30.0, max_retries=3, backoff_factor=2.0, circuit_failure_threshold=5, circuit_recovery_timeout=60)` — async httpx base with tenacity retry + circuit breaker; retryable statuses: 408, 429, 500, 502, 503, 504; override `_headers()` and `_auth_tuple()` to customise; `is_retryable_status(code)` helper exported. `MimirRulerProvider(ruler_url, *, tenant_id, api_key, username, password, timeout=30.0, user_agent, max_retries=3, backoff_factor=2.0)` — canonical in `clients.mimir`; `push_rules(namespace, rules_yaml)` → `RulerPushResult(success, namespace, status_code, message, groups_pushed)`; also `delete_rules()`, `list_rules()`, `health_check()`. Other clients: `CortexClient`, `PagerDutyClient`, `SlackAPIClient`. `CoreAPIClient(base_url="http://localhost:8000", max_retries=3, initial_backoff=1.0, max_backoff=30.0, timeout=30.0)` (`api_client.py`) — async httpx client for nthlayer-core API; returns `APIResult(ok, status_code, data, error, detail)`, never raises; transient codes (502, 503, 504, 429) retried with exponential backoff; connection errors (ConnectError, ConnectTimeout, OSError) reset httpx client before retry; timeout errors retried without client reset; 4xx returned immediately; all retries exhausted → `status_code=0, error="connection_failed"`; canonical import: `from nthlayer_common.api_client import CoreAPIClient`. Full endpoint coverage: `health()`; verdicts: `submit_verdict`, `get_verdict`, `get_verdicts(*, service, verdict_type, created_after, created_before, limit=100)`, `get_ancestors(id, max_hops?)`, `get_descendants(id)`, `resolve_outcome(id, outcome)`; assessments: `submit_assessment`, `get_assessments(*, service, kind, limit=100)`; cases: `create_case`, `get_case`, `get_cases(*, state, priority, service, limit=100)`, `acquire_lease(case_id, holder, expires_at)`, `release_lease(case_id)`, `resolve_case(case_id, resolution_id)`; change-freezes: `create_change_freeze`, `get_active_freezes`, `lift_change_freeze(name, lifted_by)`; heartbeats: `heartbeat(component, instance_id, state?)`, `get_heartbeats(threshold=30)`; manifests: `get_manifests()`, `get_manifest(service)`; component-state: `put_component_state(component, state)`, `get_component_state(component)`; monitoring: `get_stuck_action_requests(threshold=60)`.

**Slack:** `SlackNotifier` (`slack.py` — Block Kit webhook, fail-open); `SlackWebClient` (`slack_web.py` — Web API: `post_message`, `update_message`, `verify_signature`; lazy httpx client, fail-open)

**Errors:** `NthLayerError` → `ConfigurationError`, `ProviderError`, `ValidationError`, `BlockedError`, `PolicyAuditError`, `WarningResult`; **tier-boundary errors** (classify by retryability, not CLI exit code): `TransientError` (retryable: HTTP 502/503/504/429, connection errors, timeouts → PROVIDER_ERROR exit), `PermanentError` (non-retryable: HTTP 400/401/403/404/409/422 → VALIDATION_ERROR exit), `DegradedError` (continue with degraded output → WARNING exit); `classify_http_error(status_code, message?, detail?) -> TransientError | PermanentError` — maps HTTP status to TransientError/PermanentError (unknown 5xx→transient, unknown 4xx→permanent); `@retry(*, on=TransientError, max_attempts=3, initial_backoff=1.0, max_backoff=30.0, backoff_factor=2.0)` — async-only decorator with exponential backoff; raises `TypeError` at decoration time if applied to a sync function; `ExitCode` (SUCCESS=0, WARNING=1, BLOCKED=2, CONFIG_ERROR=10, PROVIDER_ERROR=11, VALIDATION_ERROR=12, UNKNOWN_ERROR=127); `@main_with_error_handling(*, show_traceback=False, log_errors=True)` — CLI decorator, `KeyboardInterrupt` → exit 130; `format_error_message(error) -> str` — formats message + details for display

**Tiers:** `Tier` (CRITICAL/STANDARD/LOW + tier-1/2/3 aliases), `TIER_CONFIGS`, `normalize_tier()`, `get_tier_config()`, `get_slo_targets()`

**Data Models:** SLO (`SLO`, `ErrorBudget`, `SLOStatus`, `TimeWindow`), Dependency (`DependencyGraph`, `DependencyType`, `BlastRadiusResult`), Domain (`Run`, `Finding`, `Team`, `Service`), Gate (`GateResult`, `GatePolicy`, `DeploymentGateCheck`)

**Prompts:** `load_prompt(path)`, `render_user_prompt(template, **kwargs)`, `validate_response(data, schema)`

**Explanation:** `BudgetExplanation(service, slo_name, headline, body, causes, recommended_actions, severity)` dataclass; `format_explanation(explanation, fmt)` → str (fmt: "table"|"json"|"markdown"); produced by nthlayer-observe, consumable by nthlayer-respond

**Manifest (`nthlayer_common.manifest`):** Unified OpenSRM manifest parsing — v1 (srm/v1), v2 (opensrm.nthlayer.io/v2), and legacy NthLayer formats. Canonical import: `from nthlayer_common.manifest import load_manifest, ReliabilityManifest`. Implements C-X.2 (manifest format migration).
- `load_manifest(file_path, environment?, format="auto", suppress_deprecation_warning?) -> ReliabilityManifest` — auto-detects format from `apiVersion`; v1 resolves templates (missing template = warning); v2 resolves templates (missing = parse error); legacy emits `LegacyFormatWarning`; `ManifestLoadError` wraps all parse failures
- `is_manifest_file(path) -> bool` — detects manifests by content (apiVersion, kind, or service+name+team+tier+type keys)
- **Model (`ReliabilityManifest`):** dataclass; required: `name`, `team`, `tier` (critical/high/standard/low), `type` (api/worker/stream/ai-gate/batch/database/web); optional: `slos`, `dependencies`, `ownership`, `observability`, `deployment`, `contracts`, `instrumentation`, `alerting` (raw dict for nthlayer-generate); `source_format` (SourceFormat enum), `source_file`; `is_ai_gate()`, `get_judgment_slos()`, `get_standard_slos()`, `validate_contracts() -> list[str]`
- **SLO (`SLODefinition`):** `name`, `target`, `slo_type` (required: availability/latency/error_rate/throughput), `window="30d"`, `indicator_query`, `total_query`/`good_query` (OpenSLO ratio); judgment SLOs: `judgment_type` (one of 8 JUDGMENT_SLO_TYPES), `measurement` (JudgmentMeasurement), `breach_actions` (list[BreachAction]), `statistical_requirements`; `source_ref` for OpenSLO $ref provenance; `is_judgment_slo() -> bool`
- **Parsers:** `parse_srm_v1` / `parse_opensrm_v2` / `parse_openslo_slos`; format detection via `is_srm_v1_format` / `is_opensrm_v2_format`; v1 `resolve_template` (no-chaining, warn on missing); v2 `resolve_v2_template` (kind=ServiceManifestTemplate, one level, error on missing; append/replace override directives). **v1 SLO indicator parsing (opensrm-saun.1 fix):** `_extract_indicator_query` reads `indicator.query` from the nested `indicator:` object only — top-level `query` was previously also accepted but is not in any real spec and is no longer read; SLOs without an `indicator` block correctly produce `indicator_query=None` (NO_DATA at collection time)
- **v2 specifics:** Backstage entity refs (`kind:namespace/name`) resolved at parse time — failure is a parse error; service type inferred from labels.type > judgment_slo presence > decision events; OpenSLO $refs resolved relative to manifest's directory
- **v1_compat:** `default_statistical_requirements(judgment_type)` (95% CI, method from type); `default_measurement(judgment_type, window)` (source/method/bins per type); `convert_v1_contract(service_name, ...) -> ReliabilityContract` (name="{service_name}-api", judgment direction="below")
- **Errors:** `ManifestLoadError`, `LegacyFormatWarning`, `OpenSRMParseError`, `OpenSRMV2ParseError`, `OpenSLOParseError`

**Verdicts (`nthlayer_common.verdicts`):** Atomic AI judgment model — migrated from nthlayer-learn to break circular dependencies. Distinct from `records`: verdicts are "what did the AI decide"; records are the immutable content-addressed audit trail.
- **Models (dataclasses):** `Verdict(id, version, timestamp, producer, subject, judgment, outcome, lineage, metadata, verdict_type, pipeline_latency_ms, chain_depth, parent_ids, service)` — v1.5 transitional fields: `verdict_type` (str|None, from VALID_VERDICT_TYPES, None for backward compat), `pipeline_latency_ms` (int|None, cumulative chain latency), `chain_depth` (int=0), `parent_ids` (list[str]=[]), `service` (str|None, denormalized); `Producer(system, instance, model, prompt_version)`; `Subject(type, ref, summary, agent, service, environment, content_hash)` — type validated against `VALID_SUBJECT_TYPES`; `Judgment(action, confidence 0.0–1.0, score, dimensions, reasoning, tags)` — action validated against `VALID_ACTIONS`; `Outcome(status="pending", ...)`, `Lineage(parent, children, context)`, `Metadata(ttl=TTL_DEFAULT=90d, cost_tokens, cost_currency, latency_ms, custom)`; `Override`, `GroundTruth`, `AccuracyReport`
- **Constants:** `VALID_SUBJECT_TYPES` (agent_output, correlation, triage, evaluation, retrospective, …); `VALID_ACTIONS` (approve, reject, flag, escalate, defer, custom); `VALID_OUTCOME_STATUSES` (confirmed, overridden, partial, superseded, expired); `VALID_VERDICT_TYPES` (action_request/approval/capability/denial/execution/operator_note from RBAC §10; autonomy_change/quality_breach from measure; triage/investigation/communication/remediation from respond agents; outcome_resolution — topology_drift/contract_divergence/correlation_snapshot are observations → ASSESSMENT_KINDS, not verdicts; "assessment" removed in opensrm-saun.1.2); `TTL_DEFAULT`
- **Operations:** `create(subject, judgment, producer, metadata=None) -> Verdict` (thread-safe ID "vrd-{date}-{uuid8}-{seq:05d}", pending outcome); `link(verdict, parent, context)` (mutates lineage); `resolve(verdict, status, ...)` (validates pending→resolved); `supersede(old, new) -> tuple` (bidirectional lineage)
- **Store:** `VerdictStore(ABC)` — `put`, `get`, `query(VerdictFilter)`, `update_outcome`, `accuracy(AccuracyFilter)`, `by_lineage(verdict_id, direction="both")`, `expire() -> int`, `resolve()` (convenience); `MemoryStore` (thread-safe, BFS, expire by TTL); `SQLiteVerdictStore` (WAL, thread-local conns, atomic conditional UPDATE for resolve, `close()`/context manager); `VerdictFilter` (from_time/to_time must be timezone-aware); `AccuracyFilter`
- **Serialisation:** `to_dict/to_json/from_dict/from_json` — dataclasses.asdict + datetime↔ISO strings; `to_dict` renames `timestamp`→`created_at` and `verdict_type`→`type` at the wire boundary (HTTP-canonical names for nthlayer-core POST /verdicts); `from_dict` accepts both wire-canonical (`type`, `created_at`) and legacy internal names (`verdict_type`, `timestamp`) for round-trip compat with data written before opensrm-saun.1.2; round-trips all v1.5 fields (verdict_type, pipeline_latency_ms, chain_depth, parent_ids, service, cost_currency)

**Metrics (`nthlayer_common.metrics`):** Self-observability Prometheus metrics. Spec: NTHLAYER-COMMON-v1 §7.3. Worker metrics: `cycle_duration_seconds` (Histogram, [component], buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)), `verdicts_written_total` (Counter, [component, type]), `assessments_written_total` (Counter, [component, kind]), `heartbeats_emitted_total` (Counter, [component]), `llm_calls_total` (Counter, [component, model, outcome]), `errors_total` (Counter, [component, error_type]). Core metrics: `api_requests_total` (Counter, [method, route, status] — route=URL template e.g. `/verdicts/{id}`, never raw path), `store_size_bytes` (Gauge), `wal_size_bytes` (Gauge), `stuck_action_requests` (Gauge, [service]). Helpers: `render_metrics() -> bytes`, `metrics_content_type() -> str`.

**Telemetry (`nthlayer_common.telemetry`):** OTel cost-accounting for LLM calls. Spec: NTHLAYER-COMMON-v1 §3.4, §7. `emit_llm_event(*, model, provider, caller, input_tokens?, output_tokens?, cached_tokens?, reasoning_tokens?, verdict_id?, duration_ms?, success=True, error?) -> None` — adds `"nthlayer.llm.call"` event to current span with `gen_ai.*` attributes; no-op if no active recording span or OTel not available. `is_otel_available() -> bool`.

**CloudEvents (`nthlayer_common.cloudevents`):** CloudEvents v1.0 envelope helpers for NthLayer events. Canonical import: `from nthlayer_common.cloudevents import wrap_verdict, wrap_assessment`. Envelope format frozen from v1.5 onwards (spec: NTHLAYER-TELEMETRY-ENVELOPE-v1 §3). `NTHLAYER_DEPLOYMENT_ID` env var sets default deployment ID (default: "default").
- `wrap_verdict(verdict, *, component="unknown", deployment_id=None) -> dict` — requires `id` field; sets `subject` from `service` as `"component:default/{service}"`; type from `_VERDICT_TYPES` (delegates to `VALID_VERDICT_TYPES`: action_request, approval, capability, denial, execution, operator_note, autonomy_change, quality_breach, triage, investigation, communication, remediation, outcome_resolution), unknown → `io.nthlayer.verdict.unknown.v1`
- `wrap_assessment(assessment, *, component="unknown", deployment_id=None) -> dict` — requires `id` field; kind from `ASSESSMENT_KINDS` (slo_status, judgment_slo_evaluation, burn_rate, drift_signal, portfolio_status, deploy_gate, dependency_graph, correlation_snapshot, topology_drift, contract_divergence, retrospective, calibration_signal), unknown → `io.nthlayer.assessment.unknown.v1`
- `parse_cloudevent(envelope) -> dict` — extracts `data` payload; validates specversion/type/source/id present and specversion=="1.0"; raises `ValueError` on missing attrs or wrong specversion; returns `{}` if no `data` key
- `validate_cloudevent(envelope) -> list[str]` — non-raising batch validator; returns list of issues (empty = valid); checks required attrs, specversion, type matches `io.nthlayer.*`, source matches `urn:nthlayer:*`, datacontenttype is `application/json`

**Decision Records (`nthlayer_common.records`):** Content-addressed append-only audit trail for the full NthLayer decision chain.
- **Models:** `Assessment` (observe→), `Verdict` (agentic components→), `Evaluation` (learn→), `Incident` (mutable index); `Summaries(technical, plain, executive)` with `truncated()`; enums: `AssessmentType`, `Severity`, `VerdictOutcome`, `EvaluationMethod`, `EvaluationOutcome`, `IncidentStatus`; `ZERO_HASH` genesis sentinel
- **Hashing:** `canonical_json(record)` → sorted-keys UTF-8 JSON bytes (excludes `hash`/`previous_hash`); `compute_hash(canonical)` → SHA-256 hex; `verify_hash(record)` → bool
- **Verdict Bridge:** Two public functions — `build_decision_verdict(*, agent, incident_id, timestamp, model, reasoning, action, outcome, prompt_text, response_text, input_hashes, previous_hash, summaries_technical, summaries_plain, summaries_executive=None) -> Verdict` (pure factory: hashes prompt/response via `hash_content(text) -> str` SHA-256 hex, truncates summaries technical/plain→280 chars/executive→140 chars, builds placeholder then returns fully content-addressed Verdict); `write_decision_verdict(store, *, agent, incident_id, timestamp, model, reasoning, action, outcome, prompt_text, response_text, input_hashes=None, summaries_technical, summaries_plain, summaries_executive=None, max_retries=1) -> None` (atomic writer: reads chain tail via `store.get_chain_tail("verdict", agent)`, builds record, writes verdict + prompt + response, retries once on `ChainForkError`, fail-open — all errors logged, never raised). Used by nthlayer-measure, nthlayer-correlate, nthlayer-respond. Use `write_decision_verdict` in all callers; `build_decision_verdict` for cases where you need the record object before writing.
- **Store:** `DecisionRecordStore` Protocol — `put_assessment/verdict/evaluation`, `create_incident`, `update_incident_status`, `get_by_hash`, `get_chain(record_type, stream_or_agent, *, limit=0)`, `get_chain_tail(record_type, chain_key) -> record | None`, `get_incident`, `get_incident_records`, `put/get_prompt`, `put/get_response`
- **SQLite:** `SQLiteDecisionRecordStore(db_path)` — WAL mode, thread-local connections, 5s busy timeout; 6 tables: assessments, verdicts, evaluations, incidents, prompts, responses; stores canonical JSON alongside each record. Record tables (assessments/verdicts/evaluations) use INSERT + IntegrityError catch: same hash → idempotent return; different hash claiming same chain position → raises `ChainForkError("Chain fork detected")`; UNIQUE constraints: `(stream, previous_hash)` assessments, `(agent, previous_hash)` verdicts, `(incident_id, previous_hash)` evaluations. Incidents and prompts/responses use INSERT OR IGNORE (idempotent). Context manager supported (`with SQLiteDecisionRecordStore(path) as store`); call `store.close()` to release thread-local connections.
- **Verification:** `verify_chain(store, record_type, stream_or_agent)` → `ChainVerificationResult` (checks hash integrity, chain linkage, genesis zero-hash, timestamp contiguity); `verify_incident(store, incident_id)` → `IncidentVerificationResult` (checks all record hashes valid; verdict `input_hashes` resolve via `get_by_hash`; verdict `prompt_hash`/`response_hash` resolve via `get_prompt`/`get_response`; evaluation `verdict_hash` and `evidence_hashes` resolve)
<!-- END AUTO-MANAGED -->
