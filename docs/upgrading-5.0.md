# Upgrading to nthlayer-common 5.0

5.0 tracks two OpenSRM changes to judgment thresholds:

- **opensrm-7zpz** — a `segments` threshold names the rate it compares
  (`metric`, OPENSRM-CORE-v2 §5.5.1), and every threshold entry is closed: a key
  its type does not define is an error, never a silently ignored field
  (OPENSRM-CORE-v2 §5.5).
- **opensrm-1rnr** — the srm/v1 threshold lift stops dropping keys it cannot
  carry, and honours `unit` on `feedback_latency`.

The release notes link here for detail. 4.0's changes are in
[upgrading-4.0.md](upgrading-4.0.md).

## What changed

| | 4.x | 5.0 |
|---|---|---|
| `segments` without `metric` | accepted | `OpenSRMV2ParseError`: `metric` is required |
| `segments` `metric` | not read (any value ignored) | one of `reversal_rate`, `high_confidence_failure`, `outcomes`, `escalation`; `audit_sampling` is rejected with its own reason (it measures sampling, not performance) |
| a key a threshold's type does not define (entry level, or inside `measurement`, `probe` or `outcome_signal`) | ignored | `OpenSRMV2ParseError` naming the key and CORE §5.5 |
| `segments` `measurement` | ignored | `window` (Duration, > 0, default `30d`), `source` (required iff `reversal_rate`; `lineage` / `calibration_sample` / `downstream_signal`), `confidence_threshold` (Ratio, required iff `high_confidence_failure`); each rejected with any other metric |
| `segments` `outcome_signal` | ignored | `{source, window}`, required iff `metric: outcomes`, rejected otherwise; `window` is a Duration and may be zero |
| pre-3c `segments` under `judgment_slo` | fix-it block loaded unchanged | fix-it block carries a `metric` placeholder and says how to choose; it loads once a metric is filled in |
| srm/v1 `calibration` / `feedback_latency` keys other than `target`, `window`, `unit` (`feedback_latency` only), `description`, `labels` | ignored (dropped) | `ValueError` (`OpenSRMParseError` from the parser) naming the key |
| srm/v1 `feedback_latency` `unit` | ignored: the target was always read as seconds | honoured: absent or `s` → `"<n>s"`; `ms`, `m`, `h`, `d`, `w` → `"<n><unit>"`; anything else rejected |
| srm/v1 `feedback_latency` `window` | ignored | absent or `30d` accepted (v2 fixes the window at 30d, CORE §5.5.5); anything else rejected |
| srm/v1 `calibration` `unit` | ignored | rejected (the target is an ECE ratio) |

`description` and `labels` on an srm/v1 threshold entry are still accepted and
dropped. The existing srm/v1 rejections (`p50`, `p90`, `bins`, `method`,
`maximum_*`) are unchanged.

## Threshold entries are closed

Each entry may hold `type`, `bounds` and the keys its type defines, nothing else:

| type | entry keys besides `type` / `bounds` | nested, closed |
|---|---|---|
| `segments` | `metric`, `segment_dimension`, `segment_values`, `measurement`, `outcome_signal` | `measurement`: `window`, `source`, `confidence_threshold`; `outcome_signal`: `source`, `window` |
| `stability` | `probe` | `probe`: `frozen_input_set`, `frequency` |
| `calibration` | `measurement` | `measurement`: `bins`, `window` (`method` keeps its own error) |
| `audit_backlog` | none | |
| `feedback_latency` | none | |

This applies to every source of an entry: a template's, the manifest's, and one
appended through `overrides.judgment_threshold` (the schema does not validate
`template.overrides`, so the parser is the only check there; opensrm-0myj).

## segments: choosing the metric

```yaml
judgment_threshold:
  - type: segments
    metric: reversal_rate          # the rate compared across segments
    measurement:
      source: lineage              # reversal_rate's parameter
      window: 7d                   # optional population window; default 30d
    segment_dimension: customer_tier
    segment_values: [enterprise, smb, consumer]
    bounds:
      maximum_variance_from_overall: 0.05
```

| `metric` | its parameter |
|---|---|
| `reversal_rate` | `measurement.source` |
| `high_confidence_failure` | `measurement.confidence_threshold` |
| `outcomes` | `outcome_signal: {source, window}` (`window` = how long to wait for each outcome, not the population window) |
| `escalation` | none (the escalation rate, not the human agreement rate) |

One entry compares one rate; declare an entry per rate that must hold evenly.

## API

- New: `OutcomeSignal(source: str, window: str)`, a frozen dataclass,
  re-exported from `nthlayer_common.manifest`.
- `ThresholdMeasurement` gains four fields, all defaulting to `None` and set only
  on `segments`:

| Field | Holds |
|---|---|
| `metric` | The compared rate, one of the four above. |
| `source` | `reversal_rate` only. |
| `confidence_threshold` | `high_confidence_failure` only; a float in [0, 1]. |
| `outcome_signal` | `outcomes` only; an `OutcomeSignal`. |

- On `segments`, `ThresholdMeasurement.window` is the population window and is
  always set: the declared `measurement.window`, or `"30d"` when absent.
- `threshold_to_v2_entry` emits `metric`, `measurement` (always with `window`)
  and `outcome_signal` for `segments`.

Worked example: the outcomes entry from the spec's
`examples/judgment-thresholds/segments-two-dimensions.yaml`. The real repr from
`parse_opensrm_v2`:

```
JudgmentThreshold(type='segments', bounds={'maximum_variance_from_overall': 0.08}, measurement=ThresholdMeasurement(bins=None, window='7d', probe=None, segment_dimension='region', segment_values=['eu', 'us', 'apac'], metric='outcomes', source=None, confidence_threshold=None, outcome_signal=OutcomeSignal(source='retro-consumer', window='1h')))
```

`ThresholdMeasurement` now has `source` and `confidence_threshold`; it still has
no `method`, `frequency` or `sampling`.

## What consumers must change

- Raise the nthlayer-common requirement to `>=5.0.0,<6.0.0`.
- Add `metric` (and its parameter) to every `segments` threshold in manifests,
  templates and test fixtures, and remove any key a threshold's type does not
  define.
- Serialise a threshold with `nthlayer_common.manifest.threshold_to_v2_entry`
  (public from 5.0.0) rather than copying `ThresholdMeasurement` fields: it
  emits the exact `spec.judgment_threshold` entry shape, validates what it
  emits (raising `ValueError` for a threshold the parser could never produce), and picks up future fields without a consumer change. nthlayer-core's
  `/manifests` serialiser must move to it (and its inline `segments` fixtures
  gain `metric`).
- **New in 5.1.0:** read an entry back with
  `nthlayer_common.manifest.threshold_from_v2_entry` (opensrm-908x) rather
  than parsing the dict by hand. It turns a served entry (e.g. an item of
  core 3.0's `/manifests` `judgment_thresholds`) into a `JudgmentThreshold`
  with exactly the manifest parser's per-entry rules. Pass `where` as a path
  to the entry in your own document, e.g.
  `threshold_from_v2_entry(item, where=f"/manifests[{i}].judgment_thresholds[{j}]")`;
  errors then read `<where>: <reason>`. It raises `OpenSRMV2ParseError`, which
  is not a `ValueError` (what `threshold_to_v2_entry` raises): catch both.
  Entries are closed, so a key a newer nthlayer-common adds is an error here:
  upgrade nthlayer-common no later than the service serving the entries.
- nthlayer-generate and nthlayer-workers read no `ThresholdMeasurement` field
  today, so they need only the version bump and any `segments` fixtures; an
  evaluator in workers should get its thresholds through
  `threshold_from_v2_entry` and evaluate them as in the 5.2.0 bullet below.
- **New in 5.2.0:** evaluate thresholds with
  `evaluate_threshold(threshold, Samples(...))` (pure; consumers supply
  measurements per `THRESHOLD_SERIES`) and emit `ThresholdOutcome.to_dict()` as
  a `judgment_threshold_evaluation` assessment. A sampler that can only bound a
  value from below (a histogram quantile in the `+Inf` bucket) supplies
  `AtLeast(v)`, which breaches iff `v >= limit` (reason `lower_bound`) and is
  otherwise not evaluable with reason `beyond_top_bucket`. Result reasons are
  `no_data_source`, `no_data`, `insufficient_window`, `lower_bound` and
  `beyond_top_bucket`; `ThresholdOutcome.from_dict` preserves reason strings it
  does not know, so a reader survives a newer writer. `Status`, `MissingReason`
  and `ResultReason` are exported for typing.
- **New in 5.3.0:** `load_manifest` rejects `apiVersion: opensrm/v1` (the
  published v1 schema) with an error naming `nthlayer-generate
  migrate-manifest`, instead of "service.name is required". (From
  nthlayer-core 4.0.0 the bare `nthlayer` script belongs to nthlayer-generate as
  well, but `nthlayer-generate` works with every version; opensrm-dww5.)
  `convert_opensrm_v1_to_v2(data, *, service_type=None) ->
  MigrationResult(manifest, unmapped)` converts it to v2 following
  AUTHORING §7. `unmapped` lists every field not carried over exactly,
  each with a `kind`: `"dropped"` (no v2 home) or `"assumed"` (converted
  with a value v1 does not declare). `MigrationResult.lossy` is True only
  when something was dropped; assumed entries are informational. It raises
  `OpenSRMV1MigrationError`, a `ValueError`, for input it cannot convert. v2
  `spec.dependencies[].service` now accepts `resource:` refs.
  `is_manifest_file` now recognises `opensrm/v1` documents by content, so
  directory scans surface them instead of skipping them.
- Evaluators compare the rate `metric` names, per CORE §5.5.1. For `outcomes`,
  count only decisions in `measurement.window` whose `outcome_signal.window` has
  elapsed; a listed segment with none is not evaluable (CORE §5.5.6 E3).
- srm/v1 manifests: move any `feedback_latency` with a non-30d `window`, and
  any threshold entry using keys outside `target`/`window`/`unit`/`description`/
  `labels`, to a v2 `judgment_threshold`. Check `unit` on `feedback_latency`:
  a `unit: ms` that 4.x ignored now changes the bound.
