# Upgrading to nthlayer-common 4.0

4.0 implements OpenSRM decision 3c (opensrm-l33e): judgment error magnitudes
are no longer SLOs. This file exists because the generated `CHANGELOG.md` keeps
only the first paragraph of a `BREAKING CHANGE:` footer; the footer links here
for the rest.

## What changed

| | 3.x | 4.0 |
|---|---|---|
| segments / stability / calibration | `SLODefinition` in `manifest.slos`, `judgment_type` set | `JudgmentThreshold` in `manifest.judgment_thresholds` |
| audit backlog bound | `audit_sampling.target.audit_backlog_maximum_age` (never read) | `JudgmentThreshold(type="audit_backlog", bounds={"maximum_age": ...})`; the old key is a parse error that prints the replacement block |
| feedback latency | srm/v1 only, parsed as a latency `SLODefinition` with `judgment_type=None` | `JudgmentThreshold(type="feedback_latency", bounds={"maximum_p50": "<n>s"})` |
| calibration bounds | only `maximum_brier_score` read; `maximum_expected_calibration_error` was dropped | each declared bound kept |
| `JUDGMENT_SLO_TYPES` | 8 names | **removed, no alias**; use `V2_JUDGMENT_SLO_TYPES` (5) and `V2_JUDGMENT_THRESHOLD_TYPES` (5) |
| contract promise on a threshold or unknown type | silently matched nothing | parse error |
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
"24h" (`<integer><ms|s|m|h|d|w>`). There is no complement and no 0-100 conversion.
This parser does not evaluate anything. The evaluation rules (strictly-exceeds,
units, not-evaluable, precedence) are OPENSRM-CORE-v2.md section 5.5.6 E1-E4, for
evaluators such as nthlayer-workers.

## Manifests

v2 manifests must move these entries from `judgment_slo` to `judgment_threshold`.
The parser's error shows the replacement block. srm/v1 manifests need no change
for `calibration` or `feedback_latency` (an srm/v1 `calibration` target reads as
`maximum_expected_calibration_error`); srm/v1 `segments` and `stability` are
now rejected and must be declared in a v2 manifest.

## Threshold entries

- Entries are flat (`type`, `bounds`, ...), not wrapped in `metadata`/`spec`.
- Keys that do not belong to a threshold's type are ignored.
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
