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
     Conversion happens at the boundary in
     `nthlayer-generate.slos.pipeline._build_slo_from_manifest` which
     divides by 100.0.
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
