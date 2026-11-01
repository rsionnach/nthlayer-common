"""The classical-SLO recording-rule contract (opensrm-h0at, opensrm-wo62.3).

Series that nthlayer-generate writes as recording rules and nthlayer-workers
reads back. Both sides take the name from here so the writer and the reader
cannot drift: measure and correlate once queried ``slo:error_budget:ratio``, a
series nothing wrote, so measure read no data for every availability SLO
(opensrm-h0at). The front door's docs/metrics-contract.md documents these
series; until opensrm-h0at's front-door change lands it still shows the old
name.

Only the error budget lives here so far. wo62.3 moves across the other
classical-SLO recording rules generate writes (availability, latency,
error rate, throughput).
"""

from __future__ import annotations

from dataclasses import dataclass

# PromQL double-quoted strings take Go escapes. Backslash first, so the
# backslashes added for the other two are not themselves doubled.
_PROMQL_ESCAPES = (("\\", "\\\\"), ('"', '\\"'), ("\n", "\\n"))


def _promql_string(value: object) -> str:
    # str() first, to render a value as generate's rule file carries it.
    # Exact for objective (generate stores str(target)) and for an int name
    # (srm/v1 loads `name: 404` as an int; YAML dumps it as 404). A bool or
    # float-like name is not: YAML dumps True as `true` (opensrm-kr94).
    value = str(value)
    for raw, escaped in _PROMQL_ESCAPES:
        value = value.replace(raw, escaped)
    return f'"{value}"'


@dataclass(frozen=True)
class SloSeries:
    """One recording-rule output. ``labels`` are the labels generate attaches
    to every sample; a reader may filter on any subset of them."""

    metric: str
    labels: tuple[str, ...]

    def selector(self, **matchers: object) -> str:
        """An instant-vector selector, ``metric{label="value",...}``.

        Refuses a label the series does not carry: a matcher on one would
        select nothing, and an empty result reads downstream as no data.
        Values render with ``str()``, matching how generate writes them.
        """
        unknown = sorted(set(matchers) - set(self.labels))
        if unknown:
            raise ValueError(
                f"{self.metric} has no label(s) {unknown}; it carries {list(self.labels)}"
            )
        if not matchers:
            return self.metric
        body = ",".join(f"{k}={_promql_string(v)}" for k, v in matchers.items())
        return f"{self.metric}{{{body}}}"


# Written once per availability-typed classical SLO. Signed: 1.0 is a full
# budget, 0.0 is spent, negative is overspent. ``slo`` is the SLO's name and
# ``objective`` is ``str(SLODefinition.target)``, its 0-100 target as a float
# (99 is written "99.0"), so pass the target itself, not a formatted string.
ERROR_BUDGET_REMAINING = SloSeries(
    "slo:error_budget_remaining:ratio", ("service", "slo", "objective")
)
