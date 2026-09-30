# nthlayer-common

Shared utilities for the NthLayer ecosystem: unified LLM interface,
provider infrastructure, identity resolution, error hierarchy, tier
definitions, decision records, verdicts, manifests, governance bridge.
Pure library; no console scripts.

## Stack

Python ≥3.11, `uv`-managed. PEP 561 — ships `py.typed`.

## Build / test / lint commands

→ See `AGENTS.md`. (Canonical home for build/test/lint/typecheck;
`pyproject.toml` is the lock for dependencies and tool config.)

## Hard rules

These are load-bearing — wrong-side mistakes cause silent breakage.

1. **SLO target convention is 0-100 percentage.**
   `SLODefinition.target` uses 0-100 percentage across every
   NthLayer-internal consumer (classical *and* judgment SLOs).
   - Examples: `availability target=99.9` for 99.9% availability;
     `reversal_rate target=98.5` (SLI is `1 - reversal_rate * 100`).
   - The OpenSLO surface (`slo_models.SLO`) uses 0.0-1.0 ratio.
     Conversion happens **at both boundaries**:
     - OUTBOUND — `nthlayer-generate.slos.pipeline._build_slo_from_manifest`
       divides by 100.0; `v1_compat._v1_slo_to_openslo` likewise for
       classical SLOs, and `_v1_slo_to_judgment` via
       `judgment_target_ratio()` for judgment ones.
     - INBOUND — `manifest/target_validation.py` owns the judgment
       polarity convention and both parsers use it (opensrm-ocvu):
       `JUDGMENT_TARGET_FIELDS` (type → target field),
       `TARGET_FIELD_IS_CEILING` (complement / scale / leave alone),
       `judgment_target_percent()`, `judgment_target_ratio()`,
       `judgment_promise_direction()` and `judgment_promise()`.
       It lives there rather than in either parser so v1 and v2 cannot
       hold different conventions for one shared model. Package-internal
       but cross-module, so deliberately not re-exported (hard rule 3).
     - Only the three MAXIMA complement (`reversal_rate`,
       `high_confidence_failure`, `escalation`). `outcomes` and
       `audit_sampling` are already floors — scale only. Error
       magnitudes (`segments`, `stability`, `calibration`) are left
       unconverted pending decision 3c.
   - Load-time validator in `manifest/target_validation.py` flags
     targets in `(0, 1)` as likely ratio author errors via
     `TargetConventionWarning(UserWarning)`. Tests in
     `tests/test_target_validation.py` pin the boundaries.
   - Decision record (opensrm-5fff):
     `nthlayer/docs/superpowers/specs/2026-05-06-slo-target-convention-decision.md`.

2. **Lint floor is frozen.** Ruff
   `select=["E4","E7","E9","F","I","UP","SIM","B"]`.
   `E501` and the full `W` family are separate hygiene calls, not
   part of the floor. No `per-file-ignores` — keep tests' imports
   above `pytestmark` / `pytest.importorskip` blocks. See `AGENTS.md`
   for full lint discipline.

3. **Public API is the top of each `__init__.py`.** Changing or
   removing a re-export breaks downstream consumers (observe, measure,
   correlate, respond, learn workers, plus core and bench). Add to
   the re-export list when a new symbol is meant to be public; never
   silently rename.

4. **`verdicts` and `records` are distinct subsystems.** Verdicts =
   *what did the AI decide* (mutable, queryable). Records =
   content-addressed append-only audit trail. They share concepts but
   not types — do not collapse them.

5. **`assessment` is not a `verdict_type`.** Removed in
   opensrm-saun.1.2. Topology drift, contract divergence, and
   correlation snapshots are observations → use `ASSESSMENT_KINDS`
   (`cloudevents.py`), not `VALID_VERDICT_TYPES`.

6. **Tests must use the structured-data primitives the spec
   prescribes.** Don't assert on raw stdout/stderr strings; assert
   on exit codes, enum values, dataclass fields, store-returned
   records. Captured-text assertions break under any formatting
   change and miss real regressions.

## Where to find detail

- Module layout, public API, subsystem cross-reference: `docs/architecture.md`.
- LLM provider routing, retry behaviour, env vars, CI canned-LLM stub:
  `docs/llm-interface.md`.
- Ecosystem-wide specs (envelope, telemetry, decision records,
  manifest formats): `nthlayer/docs/specs/`.
- Project memory / Rob's preferences across sessions:
  `~/.claude/projects/-Users-robfox-Documents-GitHub-nthlayer-ecosystem/memory/MEMORY.md`.
- Beads (issue tracking): `cd opensrm && bd ready --json`.

## Where this fits in the ecosystem

- Consumed by every other ecosystem member except `opensrm` (spec
  repo, no Python).
- Distributed as `nthlayer-common` on PyPI under Apache 2.0.
- Released via `release-please-action@v4` + trusted-publishing; a
  Docker-based smoke gate (`tests/smoke/test_imports.py`) runs against
  the freshly-built wheel before publish blocks the release.
