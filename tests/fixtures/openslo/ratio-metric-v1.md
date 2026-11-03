# OpenSLO v1 ratioMetric — vendored excerpt

Source: https://github.com/OpenSLO/OpenSLO/blob/e0259752f252276485cfa2708dac2db00c2f3464/website/docs/specification.md
(lines 348-382 and 394-417, verbatim). Vendored for opensrm-zo0o so the
parser's ratioMetric fixtures in tests/test_slo_good_total.py trace to
the spec, not to the implementation. Apache-2.0, (c) the OpenSLO authors.

```yaml
  ratioMetric: # either thresholdMetric or ratioMetric must be provided
    counter:
      true | false # true if the metric is a monotonically increasing counter,
      # or false, if it is a single number that can arbitrarily go up or down
      # ignored when using "raw"
    good: # the numerator, either "good" or "bad" must be provided if "total" is used
      metricSource:
        metricSourceRef: string # optional
        type: string # optional
        spec:
          # arbitrary chosen fields for every data source type to make it comfortable to use.
    bad: # the numerator, either "good" or "bad" must be provided if "total" is used
      metricSource:
        metricSourceRef: string # optional
        type: string # optional
        spec:
          # arbitrary chosen fields for every data source type to make it comfortable to use.
    total: # the denominator used with either "good" or "bad", either this or "raw" must be used
      metricSource:
        metricSourceRef: string # optional
        type: string # optional
        spec:
          # arbitrary chosen fields for every data source type to make it comfortable to use.

    rawType:
      success | failure # required with "raw", indicates how the stored ratio was calculated:
      #  success – good/total
      #  failure – bad/total
    raw: # the precomputed ratio stored as a metric, can't be used together with good/bad/total
      metricSource:
        metricSourceRef: string # optional
        type: string # optional
        spec:
          # arbitrary chosen fields for every data source type to make it comfortable to use.
```

- **ratioMetric** _Metric {good, total}, {bad, total} or raw_.

  - **counter** _enum(true \| false)_, specifies whether the metric is a monotonically
    increasing counter. Has no effect when using a `raw` query.

  - **good** represents the query used for gathering data from metric sources used
    as the numerator. Received data is used to compare objectives (threshold)
    values to find good values. If `bad` is defined then `good` must not be set.

  - **bad** represents the query used for gathering data from metric sources used
    as the numerator. Received data is used to compare objectives (threshold)
    values to find bad values. If `good` is defined then `bad` must not be set.

  - **total** represents the query used for gathering data from metric sources
    that is used as the denominator. Received data is used to compare objectives
    (threshold) values to find total number of metrics.

  - **rawType** _enum(success \| failure)_, required when using `raw`, specifies
    whether the ratios represented by the "raw" ratio metric are of successes or failures.
    Not to be used with `good` and `bad` as picking one of those determines the type of
    ratio.

  - **raw** represents the query used for gathering already precomputed ratios.
    The type of ratio (_success_ or _failure_) is specified using `rawType`.
