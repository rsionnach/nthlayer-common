# OpenSLO v1 SLO fields, objectives and thresholdMetric — vendored excerpt

Source: https://github.com/OpenSLO/OpenSLO/blob/e0259752f252276485cfa2708dac2db00c2f3464/website/docs/specification.md
(lines 186-204, 245-252, 341-347, 388-392 and 513-524, verbatim; the same
commit as ratio-metric-v1.md). Vendored for opensrm-8his so the OpenSLO
documents in tests/test_load_manifest_shape_contract.py (V2_FULL: the
thresholdMetric indicator, `timeWindow`, and an objective's `op`, `value`
and `targetPercent`) trace to the spec, not to the implementation.
Apache-2.0, (c) the OpenSLO authors.

## SLO `spec` fields (lines 186-204)

```yaml
  description: string # optional up to 1050 characters
  service: string # name of the service to associate this SLO with, may refer (depends on implementation) to existing object Kind: Service
  indicator: # see SLI below for details
  indicatorRef: string # name of the SLI. Required if indicator is not given.
  timeWindow:
    # exactly one item; one of possible: rolling or calendar–aligned time window
    ## rolling time window
    - duration: duration-shorthand # duration of the window eg 1d, 4w
      isRolling: true
    # or
    ## calendar–aligned time window
    - duration: duration-shorthand # duration of the window eg 1M, 1Q, 1Y
      calendar:
        startTime: 2020-01-21 12:30:00 # date with time in 24h format, format without time zone
        timeZone: America/New_York # name as in IANA Time Zone Database
      isRolling: false # if omitted assumed `false` if `calendar:` is present
  budgetingMethod: Occurrences | Timeslices | RatioTimeslices
  objectives: # see objectives below for details
  alertPolicies: # see alert policies below for details
```

## Objectives (lines 245-252)

```yaml
objectives:
  - displayName: string # optional
    op: lte | gte | lt | gt # conditional operator used to compare the SLI against the value. Only needed when using a thresholdMetric
    value: numeric # required, value used to compare threshold metrics. Only needed when using a thresholdMetric
    target: numeric [0.0, 1.0) # budget target for given objective of the SLO, can't be used with targetPercent
    targetPercent: numeric [0.0, 100) # budget target for given objective of the SLO, can't be used with target
    timeSliceTarget: numeric (0.0, 1.0] # required only when budgetingMethod is set to Timeslices
    timeSliceWindow: numeric | duration-shorthand # required only when budgetingMethod is set to Timeslices or RatioTimeslices
```

## SLI `thresholdMetric` (lines 341-347)

```yaml
  thresholdMetric: # either thresholdMetric or ratioMetric must be provided
    metricSource:
      metricSourceRef: string # optional, this field can be used to refer to DataSource object
      type: string # optional, this field describes predefined metric source type e.g. Prometheus, Datadog, etc.
      spec:
        # arbitrary chosen fields for every data source type to make it comfortable to use
        # anything that is valid YAML can be put here.
```

## Notes (SLI) (lines 388-392)

Either `ratioMetric` or `thresholdMetric` must be used.

- **thresholdMetric** _Metric_, represents the query used for
  gathering data from metric sources. Raw data is used to compare objectives
  (threshold) values.

## Example (lines 513-524)

An example of **thresholdMetric**:

```yaml
thresholdMetric:
  metricSource:
    metricSourceRef: redshift-datasource
    spec:
      region: eu-central-1
      clusterId: metrics-cluster
      databaseName: metrics-db
      query: SELECT value, timestamp FROM metrics WHERE timestamp BETWEEN :date_from AND :date_to
```
