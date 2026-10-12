# Upgrading to nthlayer-common 4.0

4.0 implements OpenSRM decision 3c (opensrm-l33e): judgment error magnitudes
are no longer SLOs. The release notes link here for detail.

## What changed

| | 3.x | 4.0 |
|---|---|---|
| segments / stability / calibration | `SLODefinition` in `manifest.slos`, `judgment_type` set | `JudgmentThreshold` in `manifest.judgment_thresholds` |
| audit backlog bound | `audit_sampling.target.audit_backlog_maximum_age` (never read) | `JudgmentThreshold(type="audit_backlog", bounds={"maximum_age": ...})`; the old key is a parse error that prints the replacement block |
| feedback latency | srm/v1 only, parsed as a latency `SLODefinition` with `judgment_type=None` | `JudgmentThreshold(type="feedback_latency", bounds={"maximum_p50": "<n>s"})` |
| calibration bounds | only `maximum_brier_score` read; `maximum_expected_calibration_error` was dropped | each declared bound kept |
| `JUDGMENT_SLO_TYPES` | 8 names | **removed, no alias**; use `V2_JUDGMENT_SLO_TYPES` (5) and `V2_JUDGMENT_THRESHOLD_TYPES` (5) |
| contract promise on a threshold or unknown type | silently matched nothing | parse error. The published opensrm/v1 schema's `ContractJudgment` keys (`max_reversal_rate`, `max_hcf_rate`, `max_escalation_rate`, ...) are not supported by the srm/v1 parser and now raise (tracked in opensrm-pnmv) |
| srm/v1 threshold config with bound-like keys | `p50`, `p90`, `bins`, `method`, `maximum_*` silently ignored | `ValueError` (`OpenSRMParseError` from the parser) naming the key: srm/v1 cannot express them; declare the threshold in an `opensrm.nthlayer.io/v2` manifest under `spec.judgment_threshold` (OPENSRM-CORE-v2 §5.5) |
| `TargetConventionWarning` on magnitudes | fired (e.g. "write '5.0'" for `maximum_drift: 0.05`) | never |

The five judgment SLO types are `reversal_rate`, `high_confidence_failure`,
`escalation`, `outcomes` and `audit_sampling`. The five threshold types are
`segments`, `stability`, `calibration`, `audit_backlog` and `feedback_latency`.

Code that imported `JUDGMENT_SLO_TYPES` now fails at import time. Code that read
`segments`, `stability` or `calibration` out of `manifest.slos` finds nothing
there: read `ReliabilityManifest.judgment_thresholds` instead.

## Evaluating a threshold

A breach is the measured value exceeding ANY declared bound. Bounds are stored as
declared: Ratio bounds are floats in [0, 1], Duration bounds are strings such as
"24h" (`<integer><ms|s|m|h|d|w>`); `duration_seconds("24h")` returns `86400.0`.
There is no complement and no 0-100 conversion.

Evaluate every entry, including several of the same type (CORE §5.5.6 E1): each
entry is evaluated independently.

This parser does not evaluate anything. The evaluation rules (strictly-exceeds,
units, not-evaluable, precedence) are OPENSRM-CORE-v2.md section 5.5.6 E1-E4, for
evaluators such as nthlayer-workers.

## Manifests

v2 manifests must move these entries from `judgment_slo` to `judgment_threshold`.
The parser's error shows the replacement block. srm/v1 manifests need no change
for `calibration` or `feedback_latency` (an srm/v1 `calibration` target reads as
`maximum_expected_calibration_error`); srm/v1 `segments`, `stability` and `audit_backlog`
are now rejected and must be declared in a v2 manifest.

## Threshold entries

- Entries are flat (`type`, `bounds`, ...), not wrapped in `metadata`/`spec`.
- Other keys that do not belong to a threshold's type are ignored. (5.0 closes
  entries: such keys are errors. See [upgrading-5.0.md](upgrading-5.0.md).)
- `breach_actions`, `target`, `judgment_type` and `service` on a threshold raise
  `OpenSRMV2ParseError`, as does `measurement.method` on a `calibration` threshold.
- `segment_dimension` and every `segment_values` entry must be non-empty strings.
- An integral float `bins` (for example `10.0`) is accepted and stored as an int.

## Templates

A template's `judgment_threshold` list CONCATENATES with the manifest's: template
entries first, then manifest entries. Unlike every other list, the manifest's does
not replace the template's, so a manifest cannot remove a template's threshold.
`overrides.judgment_threshold` may only `append` (mapping form, with a list value).
`replace`, the list form and any other shape raise `OpenSRMV2ParseError`
(OPENSRM-CORE-v2.md section 5.5; the section 9 list form is tracked in
opensrm-r78t). The ai-gate requirement applies to the resolved manifest.

## API

- `JudgmentThreshold`
- `ThresholdMeasurement`
- `duration_seconds(value: str) -> float` (Duration string to seconds; raises
  `ValueError` for anything that is not `<integer><ms|s|m|h|d|w>`)
- `ReliabilityManifest.judgment_thresholds`
- `get_judgment_thresholds()`
- `V2_JUDGMENT_SLO_TYPES`
- `V2_JUDGMENT_THRESHOLD_TYPES`
- `SRM_V1_JUDGMENT_THRESHOLD_TYPES`

### `JudgmentThreshold` fields

| Field | Holds |
|---|---|
| `type` | One of `V2_JUDGMENT_THRESHOLD_TYPES`. |
| `bounds` | Dict of bound name to value, stored as declared: a float for Ratio bounds, a duration string such as `"24h"` for Duration bounds. |
| `measurement` | A `ThresholdMeasurement`. calibration: `bins` (int) and `window`. stability: `probe` (`frozen_input_set`, `frequency`). segments: `segment_dimension` and `segment_values`. Other types: empty (all fields at their defaults). It is not a `JudgmentMeasurement`: it has no `source`, `confidence_threshold`, `method`, `frequency` or `sampling` (those are judgment-SLO fields), so a probe's frequency is `measurement.probe.frequency`. |

Worked example. An srm/v1 `calibration: {target: 0.05, window: 30d}` becomes
`JudgmentThreshold(type="calibration", bounds={"maximum_expected_calibration_error": 0.05}, measurement=ThresholdMeasurement(bins=10, window="30d"))`.
The real repr from `parse_srm_v1` (unset fields shown as the parser leaves them):

```
JudgmentThreshold(type='calibration', bounds={'maximum_expected_calibration_error': 0.05}, measurement=ThresholdMeasurement(bins=10, window='30d', probe=None, segment_dimension=None, segment_values=[]))
```

## What consumers must change

- Replace `JUDGMENT_SLO_TYPES` imports with `V2_JUDGMENT_SLO_TYPES` (SLOs) or
  `V2_JUDGMENT_THRESHOLD_TYPES` (thresholds).
- Read thresholds from `judgment_thresholds`, not `slos`.
- `JudgmentThreshold.measurement` is a `ThresholdMeasurement`, not a
  `JudgmentMeasurement`: read a stability probe's frequency from
  `measurement.probe.frequency` (there is no `measurement.frequency`), and there is no
  `measurement.method`.
- Widen the nthlayer-common requirement to `>=4.0.0,<5.0.0`.
- Evaluators implement OPENSRM-CORE-v2 §5.5.6 E1-E4.
- Reconcile local copies of the old vocabulary. For example nthlayer-generate
  `src/nthlayer_generate/specs/manifest.py` defines its own `JUDGMENT_SLO_TYPES`
  (including calibration and feedback_latency), used at about line 267.
- Grep consumers for `JUDGMENT_SLO_TYPES`, "eight judgment", and the departed type
  names (`segments`, `stability`, `calibration`). For example nthlayer-workers
  `measure/adapters/prometheus.py` carries comments about "the 8 JUDGMENT_SLO_TYPES".
