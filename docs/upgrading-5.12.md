# Upgrading to nthlayer-common 5.12

5.12 is a minor release, but the OpenSLO parser is now stricter: it enforces
OpenSLO v1's `ratioMetric` rules where 5.11 accepted malformed shapes without
complaint (opensrm-zo0o). No manifest in the NthLayer ecosystem is affected.
The rules come from the OpenSLO specification, vendored at
`tests/fixtures/openslo/ratio-metric-v1.md`: a `ratioMetric` is
"{good, total}, {bad, total} or raw".

## What is new

- `SLODefinition.bad_query`: the `bad` query of an OpenSLO `ratioMetric` with
  `bad` and `total`. 5.11 dropped it, so such an SLO had no indicator at all.
  `indicator_query` stays unset for it; the good count is `total - bad`.
- The legacy format's plural `indicators[].success_ratio` now fills
  `good_query` and `total_query`. It comes from the first entry that has both
  as non-blank strings. 5.11 read only each entry's `type`.
- A `bad`/`total` ratio defaults to `availability` when the SLO's name gives no
  type hint, as a `good`/`total` ratio already did.

## What now raises `OpenSLOParseError`

These used to load. A v2 manifest reports them as an `OpenSRMV2ParseError`.

| `ratioMetric` | 5.11 | 5.12 |
|---|---|---|
| `good` and `bad` together | `bad` ignored | refused (OpenSLO: one of them) |
| `total` alone, or `good` / `bad` without `total` | loaded with no usable ratio | refused |
| `raw` together with `good`, `bad` or `total` | `raw` ignored | refused |
| `raw` without a `rawType` of `success` or `failure` | loaded | refused |
| `rawType` without `raw` | ignored | refused (OpenSLO: "Not to be used with good and bad") |
| not a mapping, or a query that is not a string | `AttributeError`, or passed through | refused (NthLayer's rule) |

A blank query is treated as missing (NthLayer's rule). A `raw` ratio with a
valid `rawType` loads as before, with no good, bad or total query.
