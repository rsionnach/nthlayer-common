# nthlayer-common architecture

Internal reference for the package's module layout and public API
surface. Reflects the post-consolidation state (opensrm-hty.7,
2026-04-26) where observe/measure/correlate/respond/learn were merged
into nthlayer-workers. The verdicts subpackage moved here from
nthlayer-learn to break a circular dependency.

This file is the canonical home for the "what lives where" reference
that previously lived in CLAUDE.md AUTO-MANAGED blocks. Source code
(particularly each `__init__.py`) remains authoritative for the public
surface — this file is a fast cross-reference, not a contract.

## Top-level layout

```
src/nthlayer_common/
    __init__.py          # Re-exports across the public surface
    config.py            # Unified Config dataclass + Config.load()
    llm.py               # Raw-text LLM wrapper (httpx-based)
    llm_structured.py    # Structured LLM outputs via Instructor
    llm_stub.py          # CI stub (NTHLAYER_LLM_STUB=canned) — see llm-interface.md
    errors.py            # Error hierarchy + @main_with_error_handling + @retry
    tiers.py             # Tier, TIER_CONFIGS, normalize_tier, get_slo_targets
    slo_models.py        # SLO, ErrorBudget, SLOStatus, TimeWindow
    dependency_models.py # DependencyGraph, DependencyType, BlastRadiusResult
    domain_models.py     # Run, Finding, Team, Service
    gate_models.py       # GateResult, GatePolicy, DeploymentGateCheck
    slack.py             # SlackNotifier (webhook, fail-open)
    slack_web.py         # SlackWebClient (Web API, fail-open)
    prompts.py           # load_prompt, render_user_prompt, validate_response
    parsing.py           # clamp, strip_markdown_fences
    explanation.py       # BudgetExplanation + format_explanation()
    api_client.py        # CoreAPIClient — async client for nthlayer-core
    cloudevents.py       # CloudEvents v1.0 envelope helpers (frozen v1.5+)
    metrics.py           # Self-observability Prometheus metrics
    telemetry.py         # OTel cost-accounting for LLM calls
    outcomes.py          # FinancialImpact + compute/estimate helpers (opensrm-jmy.1)
    py.typed             # PEP 561 marker
    clients/             # HTTP clients (BaseHTTPClient + service-specific)
    providers/           # PrometheusProvider, GrafanaProvider, etc.
    identity/            # ServiceIdentity, IdentityResolver, OwnershipResolver
    records/             # Decision Records — content-addressed audit trail
    verdicts/            # Atomic AI judgment model (moved from nthlayer-learn)
    overrides/           # Human override ingestion (gen_ai.override, opensrm-jmy.4)
    governance_bridge/   # External governance webhook protocol (opensrm-jmy.5)
    manifest/            # Unified OpenSRM manifest parsing (v1, v2, OpenSLO, legacy)
                         #   + scan.py: walking a directory that holds
                         #     manifests alongside other YAML
```

## Subsystem references

The detailed cross-reference for each subsystem (records, verdicts,
overrides, governance_bridge, manifest, LLM stub, cloudevents) is
captured here so an agent does not need to read source first to know
what exists. **Source remains authoritative** — these notes can drift.

### Config (`nthlayer_common.config`)

- `Config.load(path?) -> Config` — resolution order: explicit path →
  `NTHLAYER_CONFIG` env → `./nthlayer.yaml` → empty defaults.
- `Config.from_dict(data)` (tests, `source="dict"`).
- `Config.get(dotpath, default=None)` — nested access.
- Convenience properties: `store_path`, `deployment_id`, `default_model`,
  `prometheus_url`.
- Env overrides: `NTHLAYER_CONFIG`, `NTHLAYER_STORE_PATH`,
  `NTHLAYER_MODEL`, `NTHLAYER_DEPLOYMENT_ID`, `NTHLAYER_PROMETHEUS_URL`.
- Spec: NTHLAYER-COMMON-v1 §10.

### Errors (`nthlayer_common.errors`)

- Hierarchy: `NthLayerError` → `ConfigurationError`, `ProviderError`,
  `ValidationError`, `BlockedError`, `PolicyAuditError`, `WarningResult`.
- **Tier-boundary errors** classify by retryability (not CLI exit code):
  `TransientError` (502/503/504/429, connection errors, timeouts →
  PROVIDER_ERROR exit), `PermanentError` (400/401/403/404/409/422 →
  VALIDATION_ERROR exit), `DegradedError` (continue with degraded output
  → WARNING exit).
- `classify_http_error(status, message?, detail?) ->
  TransientError | PermanentError` — unknown 5xx → transient,
  unknown 4xx → permanent.
- `@retry(*, on=TransientError, max_attempts=3, initial_backoff=1.0,
  max_backoff=30.0, backoff_factor=2.0)` — async-only; raises TypeError
  at decoration if applied to a sync function.
- `ExitCode`: SUCCESS=0, WARNING=1, BLOCKED=2, CONFIG_ERROR=10,
  PROVIDER_ERROR=11, VALIDATION_ERROR=12, UNKNOWN_ERROR=127.
- `@main_with_error_handling(*, show_traceback=False, log_errors=True)`:
  CLI decorator; KeyboardInterrupt → exit 130.
- `format_error_message(error) -> str`.

### Tiers (`nthlayer_common.tiers`)

- `Tier`: CRITICAL / STANDARD / LOW (+ tier-1/2/3 aliases).
- `TIER_CONFIGS`, `normalize_tier()`, `get_tier_config()`,
  `get_slo_targets()`.

### Verdicts (`nthlayer_common.verdicts`)

Atomic AI judgment model — migrated from nthlayer-learn to break a
circular dependency. **Distinct from `records`**: verdicts are *what
did the AI decide*; records are the immutable content-addressed audit
trail.

- Models (dataclasses): `Verdict`, `Producer`, `Subject`, `Judgment`,
  `Outcome`, `Lineage`, `Metadata`, `Override`, `GroundTruth`,
  `AccuracyReport`. v1.5 transitional fields on Verdict:
  `verdict_type`, `pipeline_latency_ms`, `chain_depth`, `parent_ids`,
  `service`.
- Constants: `VALID_SUBJECT_TYPES`, `VALID_ACTIONS`,
  `VALID_OUTCOME_STATUSES`, `VALID_VERDICT_TYPES`, `TTL_DEFAULT=90d`.
  Note: `assessment` is **not** a verdict_type (removed in
  opensrm-saun.1.2) — topology_drift/contract_divergence/
  correlation_snapshot are observations → ASSESSMENT_KINDS.
- Operations: `create`, `link`, `resolve`, `supersede`. Thread-safe ID
  format: `vrd-{date}-{uuid8}-{seq:05d}`.
- Store: `VerdictStore(ABC)` with `MemoryStore` and
  `SQLiteVerdictStore` (WAL mode, thread-local connections).
  `update_outcome(verdict_id, outcome, *, expected_status=None)` —
  unconditional last-writer-wins by default; supplying
  `expected_status` turns it into compare-and-swap and raises
  `OutcomeStatusMismatch(ValueError)` on miss (opensrm-jmy.11).
- Serialisation: `to_dict` / `to_json` / `from_dict` / `from_json` —
  wire boundary renames `timestamp`→`created_at`,
  `verdict_type`→`type`; `from_dict` accepts both wire-canonical and
  legacy internal names.

### Records (`nthlayer_common.records`)

Content-addressed append-only audit trail. Six tables: assessments,
verdicts, evaluations, incidents, prompts, responses.

- Models: `Assessment`, `Verdict`, `Evaluation`, `Incident`;
  `Summaries(technical, plain, executive)` with `truncated()`;
  `ZERO_HASH` genesis sentinel.
- Hashing: `canonical_json(record)` (sorted keys UTF-8, excludes
  `hash`/`previous_hash`) → `compute_hash(canonical)` → SHA-256 hex.
  `verify_hash(record) -> bool`.
- Verdict bridge:
  `build_decision_verdict(...) -> Verdict` (pure factory; truncates
  technical/plain summaries to 280 chars, executive to 140 chars);
  `write_decision_verdict(store, ..., max_retries=1) -> None` (atomic
  writer; retries once on `ChainForkError`; fail-open).
- Store: `DecisionRecordStore` Protocol; `SQLiteDecisionRecordStore`
  WAL mode + thread-local conns + 5s busy timeout.
  - Record tables use INSERT + IntegrityError: same hash → idempotent
    return; different hash claiming same chain position →
    `ChainForkError("Chain fork detected")`.
  - Incidents and prompts/responses use INSERT OR IGNORE.
- Verification: `verify_chain(...)` and `verify_incident(...)` return
  structured results checking hash integrity, chain linkage, genesis
  zero-hash, timestamp contiguity, and (for incidents) that all
  referenced hashes resolve.

### Overrides (`nthlayer_common.overrides`)

Human override ingestion — canonical OTel `gen_ai.override` event
schema, privacy controls, and verdict binding (opensrm-jmy.4, spec §4).

- `OverrideEvent` — required: `decision_id`, `service`,
  `corrected_action`, `reviewer`. Optional: `original_action`,
  `reason`, `confidence_at_decision` ([0.0, 1.0]), `source_system`,
  `timestamp` (tz-aware, default utcnow). `is_high_confidence_failure`
  property (confidence > 0.85). `to_otel_attributes()` and `to_dict()`
  are deliberately distinct.
- `OverridePrivacyConfig(pre_redacted=False, plaintext_reviewer=False,
  exclude_reason=False)`. `plaintext_reviewer` is a DEPRECATED alias
  for `pre_redacted` (both trigger identical `trust_wire = ...`
  predicate).
- `hash_reviewer(reviewer) -> str` (stable SHA-256 hex; GDPR
  pseudonymisation baseline, not anonymisation).
- `map_webhook_to_override(payload, mapping, *, defaults=None)` —
  dotted-path adapter for arbitrary webhook shapes. Required-field
  guard uses `is None` (not falsy) so JSON null surfaces distinctly
  from `""`.
- `apply_override_to_verdict(store, event, *, privacy=None) ->
  Verdict | None` — first-writer-wins via store-level CAS; returns
  None on unmatched ID, terminal status, audit-trail conflict, or CAS
  miss; structured warnings on every None path.

### Governance bridge (`nthlayer_common.governance_bridge`)

Webhook protocol for external governance platforms (opensrm-jmy.5,
spec §5). Foundation only: signal schemas + outbound emitter. Worker
integration and the inbound adapter sidecar are tracked in follow-ups
(jmy.13/14/15/16).

- Outbound: `AutonomyChangeSignal`, `PolicyRecommendationSignal`,
  `IncidentOpenedSignal`.
- Inbound (delivered via the bridge adapter sidecar, not yet
  implemented): `PolicyUpdatedSignal`, `AutonomyRestoreRequest`.
- All signals carry `version="v1"` and an immutable `signal_type`
  (init=False — prevents one signal kind being forged as another on
  the wire). `to_payload()` returns the wire dict per spec §5
  (`signal_type` → `"type"` on the wire; None optionals dropped).
  Tz-aware timestamps enforced.
- `GovernanceBridgeEmitter(webhook_url, *, auth_token=None,
  max_attempts=3, initial_backoff=1.0, max_backoff=30.0,
  backoff_factor=2.0, timeout=10.0, client=None)` — async, async
  context manager. `emit(signal) -> EmitResult` never raises
  (fail-open per spec §5 principle 6). Retries 408/429/5xx + connection
  errors with exponential backoff; non-retryable 4xx → permanent
  failure (still fail-open). Bearer auth attached when `auth_token`
  supplied; injectable client for tests.
- Idempotency: receivers dedup on `(type, service, timestamp)` per
  spec principle 3; `signal.to_payload()` is pure and stable across
  retries.

### Manifest (`nthlayer_common.manifest`)

Unified OpenSRM manifest parsing — v1 (srm/v1), v2
(opensrm.nthlayer.io/v2), OpenSLO subpackage, legacy NthLayer formats.
Implements C-X.2 (manifest format migration).

**Scanning a mixed directory** (`manifest/scan.py`). `load_manifest`
raises for any YAML it cannot parse as a manifest, including files that
never claimed to be one. Callers walking a specs directory need to
separate "a manifest that was aiming to load and failed" from "foreign
YAML sharing the directory" — count the first, and the operator learns
their view is partial; count the second, and a coverage caveat fires on
every mixed-directory run until nobody reads it.

- `iter_manifest_files(dir) -> list[Path]` — every `.yaml`/`.yml` FILE
  directly under `dir`, sorted. Both suffixes always: `.yml` invisibility
  is a silent subset reached by file extension.
- `foreign_yaml_reason(path) -> str | None` — `None` when the file was
  aiming to be a manifest (so the caller counts it), a short reason when
  it plainly was not (so the caller can log rather than drop it silently).
  Recovers intent only while evidence of intent survives; see its
  docstring for the stated limit.
- `MANIFEST_SUFFIXES` — the two suffixes, for callers doing their own walk.

Used by `nthlayer_workers` observe and learn. **Look here before writing
a fourth directory walk** — three existed before this was shared
(opensrm-oh27, opensrm-3470).

- `load_manifest(path, environment?, format="auto",
  suppress_deprecation_warning?) -> ReliabilityManifest` —
  auto-detects format from `apiVersion`. v1 resolves templates
  (missing → warning). v2 resolves templates (missing → parse error).
  Legacy emits `LegacyFormatWarning`. All parse failures wrapped in
  `ManifestLoadError`.
- `is_manifest_file(path) -> bool` — content-based (apiVersion, kind,
  or service+name+team+tier+type).
- `ReliabilityManifest` dataclass; required: `name`, `team`, `tier`,
  `type`. Optional: `slos`, `dependencies`, `ownership`,
  `observability`, `deployment`, `contracts`, `instrumentation`,
  `alerting` (raw dict for nthlayer-generate), `outcomes` (opensrm-jmy.1).
- SLO target convention: **0-100 percentage canonical** across all
  NthLayer-internal consumers. The OpenSLO surface
  (`slo_models.SLO`) uses 0.0-1.0 ratio; conversion happens at the
  boundary in nthlayer-generate. Load-time validator
  (`target_validation.py`) flags targets in (0, 1) as likely ratio
  author errors via `TargetConventionWarning(UserWarning)` — filterable
  via `warnings.filterwarnings(action,
  category=TargetConventionWarning)`. See
  `nthlayer/docs/superpowers/specs/2026-05-06-slo-target-convention-decision.md`
  (opensrm-5fff).
- v2 specifics: Backstage entity refs (`kind:namespace/name`)
  resolved at parse time (failure → parse error). Service type is
  READ from the required `spec.service.type` field, never inferred;
  `metadata.labels.type` is not consulted (opensrm-ih0v). Valid values
  are the spec's six plus an `^x-[a-z][a-z0-9-]*$` extension branch.
  Consumers should call `resolve_service_type` (resolves aliases, then
  validates — that order matters, since an alias is not itself a valid
  type); `is_valid_service_type` and `valid_service_types_phrase` are the
  underlying predicate and the shared error wording. Only an `ai-gate` may declare
  `judgment_slo`, matching schema.json's `ServiceManifest.allOf`.
  OpenSLO `$refs` resolved relative to manifest's directory.
- v1→v2 compat helpers in `v1_compat.py`:
  `default_statistical_requirements`, `default_measurement`,
  `convert_v1_contract`, `convert_v1_to_v2(v1_data: dict) -> dict`.

### CloudEvents (`nthlayer_common.cloudevents`)

CloudEvents v1.0 envelope helpers. Envelope format frozen from v1.5
onwards (spec NTHLAYER-TELEMETRY-ENVELOPE-v1 §3). `NTHLAYER_DEPLOYMENT_ID`
env sets default deployment ID (default `"default"`).

- `wrap_verdict(verdict, *, component="unknown", deployment_id=None)
  -> dict` — requires `id`; subject from `service` as
  `"component:default/{service}"`. Type from `_VERDICT_TYPES`
  (delegates to `VALID_VERDICT_TYPES`); unknown →
  `io.nthlayer.verdict.unknown.v1`.
- `wrap_assessment(...)` — same shape; kind from `ASSESSMENT_KINDS`
  (12 values: slo_status, judgment_slo_evaluation, burn_rate,
  drift_signal, portfolio_status, deploy_gate, dependency_graph,
  correlation_snapshot, topology_drift, contract_divergence,
  retrospective, calibration_signal). Unknown →
  `io.nthlayer.assessment.unknown.v1`.
- `parse_cloudevent(envelope)` — extracts `data`; validates
  specversion/type/source/id and `specversion=="1.0"`; raises on
  missing attrs or wrong specversion. Empty dict if no `data` key.
- `validate_cloudevent(envelope) -> list[str]` — non-raising batch
  validator; returns issue list (empty = valid).

### Metrics + Telemetry

- `metrics.py` — Prometheus metrics (spec NTHLAYER-COMMON-v1 §7.3).
  Worker: `cycle_duration_seconds` (Histogram, [component],
  buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)),
  `verdicts_written_total` ([component, type]),
  `assessments_written_total` ([component, kind]),
  `heartbeats_emitted_total` ([component]),
  `llm_calls_total` ([component, model, outcome]),
  `errors_total` ([component, error_type]).
  Core: `api_requests_total` ([method, route, status] — route is the
  URL template e.g. `/verdicts/{id}`, never raw path),
  `store_size_bytes`, `wal_size_bytes`, `stuck_action_requests`
  ([service]). Helpers: `render_metrics() -> bytes`,
  `metrics_content_type() -> str`.
- `telemetry.py` — OTel cost-accounting for LLM calls (spec §3.4, §7).
  `emit_llm_event(*, model, provider, caller, input_tokens?,
  output_tokens?, cached_tokens?, reasoning_tokens?, verdict_id?,
  duration_ms?, success=True, error?) -> None` — adds
  `"nthlayer.llm.call"` event to current span with `gen_ai.*`
  attributes; no-op if no recording span or OTel unavailable.
  `is_otel_available() -> bool`.

### Outcomes (`nthlayer_common.outcomes`)

Financial-impact primitives for retrospective analysis (opensrm-jmy.1
spec §1).

- `FinancialImpact(estimated, currency, decisions_affected,
  failure_mode, volume_source)` dataclass.
- `compute_financial_impact(outcomes, *, decisions_affected,
  failure_mode, volume_source) -> FinancialImpact | None` — multiplies
  per-failure cost by impacted decision count. Returns None when
  outcomes lacks cost for the requested failure_mode.
- `estimate_decisions_in_window(outcomes, *, window) -> int | None` —
  spec-fallback decision count prorated from
  `estimated_daily_decisions` (floor to 1 when 0 < raw < 1 and daily
  > 0; None when no volume estimate or non-positive window).
- `VolumeSource = Literal["metric", "spec_estimate"]`.
- `FailureMode = Literal["false_positive", "false_negative"]`.
- Depends on `nthlayer_common.manifest.models.Outcomes`.

### Identity (`nthlayer_common.identity`)

- `ServiceIdentity`, `IdentityMatch`.
- `normalize_service_name`, `DEFAULT_RULES`.
- `IdentityResolver` (7-strategy resolution).
- `OwnershipResolver`, `OwnershipSignal`, `OwnershipAttribution`.
- Providers: Backstage, Kubernetes, PagerDuty.

### HTTP clients (`nthlayer_common.clients`)

- `BaseHTTPClient(base_url, *, timeout=30.0, max_retries=3,
  backoff_factor=2.0, circuit_failure_threshold=5,
  circuit_recovery_timeout=60)` — async httpx base with tenacity retry
  + circuit breaker. Retryable: 408, 429, 500, 502, 503, 504. Override
  `_headers()` and `_auth_tuple()` to customise.
  `is_retryable_status(code)` exported.
- `CortexClient`, `PagerDutyClient`, `SlackAPIClient`.
- `MimirRulerProvider(ruler_url, *, tenant_id, api_key, username,
  password, timeout=30.0, user_agent, max_retries=3,
  backoff_factor=2.0)` — canonical in `clients.mimir`.
  `push_rules(namespace, rules_yaml)` →
  `RulerPushResult(success, namespace, status_code, message,
  groups_pushed)`. Also `delete_rules()`, `list_rules()`,
  `health_check()`.

### CoreAPIClient (`nthlayer_common.api_client`)

Async httpx client for nthlayer-core API. Returns
`APIResult(ok, status_code, data, error, detail)`; never raises.

- `CoreAPIClient(base_url="http://localhost:8000", max_retries=3,
  initial_backoff=1.0, max_backoff=30.0, timeout=30.0)`.
- Transient codes (502, 503, 504, 429) retried with exponential
  backoff. Connection errors (ConnectError, ConnectTimeout, OSError)
  reset the httpx client before retry. Timeout errors retried without
  client reset. 4xx returned immediately. Retries exhausted →
  `status_code=0, error="connection_failed"`.
- Endpoints: health; verdicts (submit, get, get_verdicts(*, service,
  verdict_type, created_after, created_before, limit=100),
  get_ancestors(id, max_hops?), get_descendants(id),
  resolve_outcome(id, outcome), `apply_override(verdict_id, payload)`
  — POST /verdicts/{id}/override (opensrm-jmy.18): 200=applied/idempotent,
  404=verdict_not_found, 409=conflict, 422=validation_error,
  0=connection_failed); assessments (submit, get,
  get_assessments(*, service, kind, limit=100)); cases (create, get,
  get_cases(*, state, priority, service, limit=100),
  acquire_lease(case_id, holder, expires_at), release_lease(case_id),
  resolve_case(case_id, resolution_id)); change-freezes
  (create_change_freeze, get_active_freezes,
  lift_change_freeze(name, lifted_by)); heartbeats
  (heartbeat(component, instance_id, state?),
  get_heartbeats(threshold=30)); manifests (get_manifests,
  get_manifest(service)); component-state
  (put_component_state, get_component_state); monitoring
  (get_stuck_action_requests(threshold=60)).

### Data models

- SLO: `SLO`, `ErrorBudget`, `SLOStatus`, `TimeWindow`
  (`slo_models.py`).
- Dependency: `DependencyGraph`, `DependencyType`, `BlastRadiusResult`
  (`dependency_models.py`).
- Domain: `Run`, `Finding`, `Team`, `Service` (`domain_models.py`).
- Gate: `GateResult`, `GatePolicy`, `DeploymentGateCheck`
  (`gate_models.py`).

### Slack

- `SlackNotifier` (`slack.py`) — Block Kit webhook, fail-open.
- `SlackWebClient` (`slack_web.py`) — Web API:
  `post_message`, `update_message`, `verify_signature`. Lazy httpx
  client, fail-open.

### Prompts / explanation

- `prompts.py`: `load_prompt(path)`, `render_user_prompt(template,
  **kwargs)`, `validate_response(data, schema)`.
- `explanation.py`:
  `BudgetExplanation(service, slo_name, headline, body, causes,
  recommended_actions, severity)` dataclass;
  `format_explanation(explanation, fmt)` → str (fmt:
  "table" / "json" / "markdown"). Produced by nthlayer observe,
  consumed by nthlayer respond.

## Test suite (758 tests at last full count)

See `tests/` — one file per source module. Notable:

- `tests/test_providers.py` — `TestGetSliValueNoneVsZero` (5 async
  tests, opensrm-e1gk): empty → None, zero → 0.0, short-tuple → None,
  non-numeric → None, real → float. `TestPrometheusProvider` covers
  `_parse_step_to_seconds` (5m / 1h / 30s / 1d / unknown) and numeric
  string fallback.
- `tests/test_cloudevents.py` — `TestWrapRealVerdict` (opensrm-saun.1.2)
  locks the end-to-end contract: `verdict_create` → `to_dict` →
  `wrap_verdict` must produce the correct CE type, never `.unknown.v1`.
- `tests/test_v1_to_v2_migration.py` — 14 tests across 3 classes;
  `TestDemoSpecsRoundTrip` (4) covers every demo spec under
  `nthlayer/demo/specs/`.
- `tests/test_llm_stub.py` — 26 tests for the canned-LLM stub. See
  `llm-interface.md` for the stub's role detection and structured
  factory registry.
- `tests/smoke/test_imports.py` — walks every module via `pkgutil`
  and asserts every `__all__` symbol resolves via `getattr`. Runs in
  the release-please publish gate.

## See also

- `llm-interface.md` — provider routing, env vars, retry behaviour,
  LLM stub canned shapes.
- `nthlayer/docs/superpowers/specs/2026-05-06-slo-target-convention-decision.md`
  — SLO target convention decision record (opensrm-5fff).
- Ecosystem-wide spec docs: `nthlayer/docs/specs/`.
