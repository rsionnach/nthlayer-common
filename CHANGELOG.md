# Changelog — nthlayer-common

`nthlayer-common` is the shared utility library for the NthLayer ecosystem.
Imported by `nthlayer-core`, `nthlayer-workers`, and `nthlayer-bench`. License:
Apache-2.0.

## [1.6.0](https://github.com/rsionnach/nthlayer-common/compare/v1.5.0...v1.6.0) (2026-05-10)


### Features

* add BudgetExplanation data model and formatter (nthlayer-hmj) ([e0a8266](https://github.com/rsionnach/nthlayer-common/commit/e0a82666461b1a950ac1293fda9c0081604bdc3e))
* add retry with backoff for transient LLM failures ([adefc7c](https://github.com/rsionnach/nthlayer-common/commit/adefc7cd13121fa5f7e7a7700b57c2cafc1c8f1c))
* add SlackWebClient for interactive messages (buttons, updates) ([a4a037b](https://github.com/rsionnach/nthlayer-common/commit/a4a037bc1e910237f25b92f6faf77f7183d644c0))
* add status code classification and Retry-After parsing for llm_call ([e50c2ef](https://github.com/rsionnach/nthlayer-common/commit/e50c2ef3fc681f50ca25c30cc4c82b092ae28f62))
* add token counts to LLMResponse, add CLAUDE.md ([3b66abf](https://github.com/rsionnach/nthlayer-common/commit/3b66abfaddd21594b3a888c4af802e294ef8407c))
* **cloudevents:** CloudEvents v1.0 envelope helpers ([0086d07](https://github.com/rsionnach/nthlayer-common/commit/0086d0721c2b9775a0c7ab34dae9d8ab6b59c1bb))
* **config:** unified Config dataclass with env override resolution ([9b16795](https://github.com/rsionnach/nthlayer-common/commit/9b16795c122419ff858593c8573dc31ba0c26bf1))
* content-addressed decision records package ([5d2b605](https://github.com/rsionnach/nthlayer-common/commit/5d2b605f451bec69d0a96b60ed5786f1798d201e))
* content-addressed decision records package (records/) ([e98cb9b](https://github.com/rsionnach/nthlayer-common/commit/e98cb9b4ac465c4c4d34067819e629cd1ff10b2f))
* **errors:** add tier-boundary error hierarchy ([ab6f2d1](https://github.com/rsionnach/nthlayer-common/commit/ab6f2d18fcdac01b1ab5f2e6a38df18f07c358a2))
* expose Phase 1 subsystem APIs from package root ([c274cf1](https://github.com/rsionnach/nthlayer-common/commit/c274cf1eea3f84cae181873c45febe69e454e732))
* **llm_stub:** NTHLAYER_LLM_STUB=canned mode for CI integration tests ([9018f6f](https://github.com/rsionnach/nthlayer-common/commit/9018f6fcd463c33a8ea06a7831b3e4e8f8bc1090))
* **llm-stub:** canned factories for respond agent response models ([ee91822](https://github.com/rsionnach/nthlayer-common/commit/ee91822827db6990ac633b0be93a5ae382b57b99))
* **manifest:** convert_v1_to_v2 library function (P2-A.2) ([a65b519](https://github.com/rsionnach/nthlayer-common/commit/a65b519bae66def9b116bf6a5e87d901c10d24b5))
* **manifest:** SLO target convention validator + honest documentation (opensrm-pa2w) ([57b9de1](https://github.com/rsionnach/nthlayer-common/commit/57b9de123cb1b84dc3bd0e9c655895a30cb5efb8))
* **manifest:** unified OpenSRM manifest parsing — v1, v2, legacy ([d340f28](https://github.com/rsionnach/nthlayer-common/commit/d340f28c504aa42193c3a967e2da2b7ebf2a775b))
* **metrics:** self-observability Prometheus metrics ([3b9e666](https://github.com/rsionnach/nthlayer-common/commit/3b9e666875bc428c51e36254646230d5deb002af))
* model-agnostic LLM wrapper for NthLayer ecosystem ([a8ad6b0](https://github.com/rsionnach/nthlayer-common/commit/a8ad6b07eb59bf767fdd7d026e2aa73c0577a30f))
* **outcomes:** business outcome binding foundation (opensrm-jmy.1) ([437eb33](https://github.com/rsionnach/nthlayer-common/commit/437eb337a018d8953b3e707d4ff1f59fd1c6fcc4))
* **overrides:** human override ingestion foundation (opensrm-jmy.4) ([f03c3cf](https://github.com/rsionnach/nthlayer-common/commit/f03c3cf22fce52d185c19de25507dedb8106601d))
* shared infrastructure migration + py.typed + README ([f36b49b](https://github.com/rsionnach/nthlayer-common/commit/f36b49bd9a202f67508161edbb0c380900c3382b))
* shared prompt loader with YAML schemas and response validation ([8bc5555](https://github.com/rsionnach/nthlayer-common/commit/8bc555529e35f9848d352926a4e315f7d9bea262))
* SlackNotifier transport — Block Kit messages via webhook, fail-open ([5b1f328](https://github.com/rsionnach/nthlayer-common/commit/5b1f328b0c67bb504ec773ca00e685bd7397a89a))
* SlackWebClient + LLM retry with backoff (v0.1.4) ([3f0801e](https://github.com/rsionnach/nthlayer-common/commit/3f0801e988a6c628af1e82920189e633af48d270))
* **verdicts:** atomic AI judgment model migrated from nthlayer-learn ([21d5f79](https://github.com/rsionnach/nthlayer-common/commit/21d5f79d06c802765af1d0b4ca96daed328edbbe))
* **verdicts:** wire-canonical names + 4 respond-agent verdict types ([dccb520](https://github.com/rsionnach/nthlayer-common/commit/dccb520878e737f5bd4f5d61a567963f2fe6c0e7))


### Bug Fixes

* add missing runtime dependencies to pyproject.toml ([11c36c2](https://github.com/rsionnach/nthlayer-common/commit/11c36c2d0d234022a0f2d282d73da49e1e48aa82))
* add shared parsing utilities, fix Azure URL, add timeout docs ([15fcded](https://github.com/rsionnach/nthlayer-common/commit/15fcdedf05f5a8007d75d1325602ec70e39bee2f))
* catch BaseHTTPClient error types in MimirRulerProvider (R5 finding) ([b7977ea](https://github.com/rsionnach/nthlayer-common/commit/b7977ea28af4d0dfe888d33880ac86a9c88679cb))
* **deps:** pin pygments&gt;=2.20.0 to clear CVE-2026-4539 ([65880e5](https://github.com/rsionnach/nthlayer-common/commit/65880e5cf3ae3dd683a966b47cd0cf0595b2c1ab))
* detect API keys accidentally used as model names ([3a44f5b](https://github.com/rsionnach/nthlayer-common/commit/3a44f5b09b62a392beeb60650f1cf5e738c41f95))
* export error hierarchy from __init__.py + bump to 0.1.7 ([b77cc51](https://github.com/rsionnach/nthlayer-common/commit/b77cc5122ab269f1507ba983e7aeeb412ce714c2))
* handle non-UTF-8 body in verify_signature ([d73e924](https://github.com/rsionnach/nthlayer-common/commit/d73e9246af3221da984ab334db03d927b16b9dac))
* harden llm.py from R5 review findings ([b1a103a](https://github.com/rsionnach/nthlayer-common/commit/b1a103a7115a75a7a6afb0a338e9b594c1d6a77e))
* **llm-stub:** include confidence in SnapshotSummary canned response ([992c0c3](https://github.com/rsionnach/nthlayer-common/commit/992c0c378f03cac02289fe8c67f5e9992b7bec75))
* **manifest/v1:** read indicator.query from nested form (opensrm-0i5h) ([0fb5096](https://github.com/rsionnach/nthlayer-common/commit/0fb5096d10a1d477e67f72763f54477ad965477b))
* **prometheus:** distinguish empty-result from 0.0 in get_sli_value ([1874e2b](https://github.com/rsionnach/nthlayer-common/commit/1874e2b8d9092da4c1d65a382557e70362908c9d))
* R5 edge case findings in BaseHTTPClient + MimirRulerProvider ([e702447](https://github.com/rsionnach/nthlayer-common/commit/e702447b694f1a4543b521c30cbbebcea35e9e37))


### Code Refactoring

* move MimirRulerProvider to clients/, extend BaseHTTPClient (nthlayer-2xe) ([2403945](https://github.com/rsionnach/nthlayer-common/commit/2403945ce70092951601c1fd737f16c1d200e9f7))
* **overrides:** clarity nits — comment + docstring trim (opensrm-jmy.12) ([11de001](https://github.com/rsionnach/nthlayer-common/commit/11de001a847abce77284b7d8a0b682635b9600a5))
* **overrides:** R5 review followups (opensrm-jmy.4) ([8139492](https://github.com/rsionnach/nthlayer-common/commit/8139492897c8d5e4b24ca0637ae99aa944a86308))
* share httpx client in SlackWebClient for connection pooling ([0a9ecf3](https://github.com/rsionnach/nthlayer-common/commit/0a9ecf35eb9e0720748ca4699dbe104571cc9eb1))
* **slo:** adopt 0-100 percentage canonical for SLO target convention ([0ca1296](https://github.com/rsionnach/nthlayer-common/commit/0ca129688c72b8e8c6745db85c2345fdf74c3b9f))


### Documentation

* add README.md for PyPI description, bump to 0.1.1 ([f642ff2](https://github.com/rsionnach/nthlayer-common/commit/f642ff2efda3c14a545a5c4a0c05b4209b803f2a))
* **CLAUDE.md:** catalogue Outcomes module + financial impact tests (opensrm-jmy.1) ([4bbca0d](https://github.com/rsionnach/nthlayer-common/commit/4bbca0d371bbd881a4955ea180a609341c7bf9ad))
* **CLAUDE.md:** document 8-judgment-type parametrised parser test ([5ad876a](https://github.com/rsionnach/nthlayer-common/commit/5ad876a8d80c8ea09960f380acd058a0a44a29f0))
* **CLAUDE.md:** document convert_v1_to_v2 and migration tests ([0da4c54](https://github.com/rsionnach/nthlayer-common/commit/0da4c5480f451cc2dc0326c2dc89bfb1e383f210))
* **CLAUDE.md:** document get_sli_value None-vs-zero semantics + tests ([674e1fe](https://github.com/rsionnach/nthlayer-common/commit/674e1fe1a01d8a490e605d2fb0b61f0fb7de3086))
* **CLAUDE.md:** document pygments transitive pin for CVE-2026-4539 ([6f672fe](https://github.com/rsionnach/nthlayer-common/commit/6f672fe6aeac9d0e39912a9c9764f5d775fe189f))
* **CLAUDE.md:** document release-please + smoke gate + Dependabot ([392ba4d](https://github.com/rsionnach/nthlayer-common/commit/392ba4df6bcc467aeaddcdfb680ef8fb1c0c0d35))
* **CLAUDE.md:** document respond-agent stub factories ([6ff38f5](https://github.com/rsionnach/nthlayer-common/commit/6ff38f5308bb7cff8e096ade3176a8994af4fdf2))
* **CLAUDE.md:** document target_validation module + SLO convention divergence ([60bf7da](https://github.com/rsionnach/nthlayer-common/commit/60bf7da7546b6684a261e81a47b3e540627dae1f))
* **CLAUDE.md:** expand terminal-status enumeration (opensrm-jmy.12) ([588f786](https://github.com/rsionnach/nthlayer-common/commit/588f786f2a3caaca86f176a8cc00d15b0a23a486))
* **CLAUDE.md:** note confidence=0.5 in SnapshotSummary stub factory ([0df2256](https://github.com/rsionnach/nthlayer-common/commit/0df2256b37e61c47d4e297fd976c0b66faf09f11))
* **CLAUDE.md:** tighten override test catalogue post-R5 ([726384e](https://github.com/rsionnach/nthlayer-common/commit/726384ea36eae8698e3097138ed3186233b48d50))
* **CLAUDE.md:** update SLO target convention + target_validation entries ([0a5a227](https://github.com/rsionnach/nthlayer-common/commit/0a5a2272592b6234dfe2e7c295874769611e9734))
* **comments:** inline pointers to verdict-vs-assessment decision ([008941e](https://github.com/rsionnach/nthlayer-common/commit/008941eabcfae5cd30bd3989d1cfa57703e8fb27))
* fix self-referential _headers() docstring (R5 clarity) ([384eab8](https://github.com/rsionnach/nthlayer-common/commit/384eab8ccc539062b058fbd09b98c4bc73d24d91))
* update README to reflect full shared infrastructure scope ([7e74923](https://github.com/rsionnach/nthlayer-common/commit/7e74923e300f04074658938034ee6aa4bfcd8922))

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
