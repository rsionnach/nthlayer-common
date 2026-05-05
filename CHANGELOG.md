# Changelog — nthlayer-common

`nthlayer-common` is the shared utility library for the NthLayer ecosystem.
Imported by `nthlayer-core`, `nthlayer-workers`, and `nthlayer-bench`. License:
Apache-2.0.

## v1.5.0 — 2026-05-03

First lockstep release with the rest of the v1.5 ecosystem. Major changes:

**LLM stub for CI integration tests** (opensrm-saun.1.1).
`nthlayer_common.llm_stub` short-circuits `llm_call()`, `structured_call()`,
and `structured_call_with_usage()` when `NTHLAYER_LLM_STUB=canned` is set.
Returns deterministic canned responses indexed by detected agent role
(triage / investigation / communication / remediation) for raw text and by
`response_model` class name (`EvaluationResult`, `SnapshotSummary`) for
structured calls. Activation predicate `is_stub_enabled()` is
case-insensitive and trim-tolerant. Three call sites in `llm.py` and
`llm_structured.py` import the stub lazily for cycle safety. Documented in
the package's CLAUDE.md; not advertised on PyPI as it is intended for CI
integration tests only.

**Wire-canonical field names in `to_dict(Verdict)`** (opensrm-saun.1.2).
`to_dict` now emits HTTP-canonical `type` and `created_at` (renamed from
internal `verdict_type` and `timestamp`) at the serialisation boundary.
The `Verdict` dataclass keeps its internal field names. `from_dict` accepts
both name pairs for round-trip compatibility with stored data; precedence
rule is "explicit None on canonical name falls back to legacy". This
aligns the wire format with what `nthlayer-core`'s `POST /verdicts` API
contract expects.

**CloudEvents type taxonomy consolidated** (opensrm-saun.1.2).
`nthlayer_common.cloudevents._VERDICT_TYPES` now imports
`VALID_VERDICT_TYPES` from `nthlayer_common.verdicts.models` rather than
maintaining a parallel set — one canonical source. The verdict-type set
itself was reshaped: dropped `"assessment"` (a category error — verdicts
are decisions, assessments are continuous observations) and added the four
respond agent role names (`triage`, `investigation`, `communication`,
`remediation`).

**v1 SRM manifest parser reads canonical `indicator.query`** (opensrm-saun.1).
`parser.v1.py` previously read top-level `config["query"]` (a non-canonical
shape that no published spec or example produces). The fix routes through
a new `_extract_indicator_query()` helper that reads the canonical
`indicator.query` nested form per `nthlayer-generate/documentation/SCHEMA.md`.
Discovered when the saun.1 three-tier integration test exercised the
worker → core `/manifests` path for the first time and found every SLO
was being collected as `NO_DATA` because the indicator PromQL was being
silently dropped at parse time. Regression test
(`tests/test_manifest_real_specs.py`) loads each `demo/specs/*.yaml`
from disk and asserts `indicator_query` populates.

**Provenance.** This package was created during the six-repo
consolidation decided 2026-04-21 (see
`docs/superpowers/specs/2026-04-21-repo-consolidation-recommendation.md`
in the `opensrm` repo). It hosts the unified LLM wrapper, provider
infrastructure (Prometheus, Grafana, PagerDuty, Mimir), identity
resolution, HTTP clients, Slack notifiers, error hierarchy, tier
definitions, shared data models, prompt loader, content-addressed
decision records, verdict model, manifest parsing, CloudEvents envelope
helpers, and self-observability metrics. Detailed surface documented in
this package's `CLAUDE.md`.
