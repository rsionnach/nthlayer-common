"""The parsers keep an SLO's own good/total (and bad) queries (opensrm-zo0o).

Ruling Z1 (Rob, 2026-11-03): every classical SLO is good/total against its
objective, as OpenSLO defines it, so nthlayer-generate can write an error
budget for each from the SLO's own queries. That needs the parsers to keep
them. Two shapes were dropped:

- OpenSLO v1's ``ratioMetric`` takes ``total`` and exactly one of ``good``
  or ``bad`` (OpenSLO v1 spec, "Ratio Metric": "good" and "bad" are
  mutually exclusive). The ``bad`` form was read as no indicator at all.
- The legacy format's plural ``indicators[]`` carries a ``success_ratio``
  with ``total_query`` and ``good_query`` (the shape of
  nthlayer-generate/examples/services/analytics-stream.yaml); only the
  entry's ``type`` was read.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nthlayer_common.manifest import load_manifest
from nthlayer_common.manifest.openslo.parser import OpenSLOParseError, parse_openslo_slos

TOTAL = 'sum(rate(http_requests_total{service="payment-api"}[5m]))'
GOOD = 'sum(rate(http_requests_total{service="payment-api",code!~"5.."}[5m]))'
BAD = 'sum(rate(http_requests_total{service="payment-api",code=~"5.."}[5m]))'


def _source(query: str) -> dict:
    return {"metricSource": {"type": "Prometheus", "spec": {"query": query}}}


def _openslo(ratio: dict) -> dict:
    """An OpenSLO v1 SLO document, per the spec's SLO schema."""
    return {
        "apiVersion": "openslo/v1",
        "kind": "SLO",
        "metadata": {"name": "payment-availability"},
        "spec": {
            "service": "payment-api",
            "indicator": {
                "metadata": {"name": "availability"},
                "spec": {"ratioMetric": {"counter": True, **ratio}},
            },
            "timeWindow": [{"duration": "30d", "isRolling": True}],
            "objectives": [{"displayName": "99.9% of requests succeed", "target": 0.999}],
        },
    }


class TestOpenSLORatioMetric:
    def test_bad_over_total_keeps_both_queries(self) -> None:
        (slo,) = parse_openslo_slos([_openslo({"bad": _source(BAD), "total": _source(TOTAL)})])

        assert slo.bad_query == BAD
        assert slo.total_query == TOTAL
        assert slo.good_query is None
        # indicator_query is the composed good ratio; consumers that read it
        # (generate's Sloth path) are unchanged by this bead.
        assert slo.indicator_query is None

    def test_good_over_total_is_unchanged(self) -> None:
        (slo,) = parse_openslo_slos([_openslo({"good": _source(GOOD), "total": _source(TOTAL)})])

        assert slo.good_query == GOOD
        assert slo.total_query == TOTAL
        assert slo.bad_query is None
        assert slo.indicator_query == f"({GOOD}) / ({TOTAL})"

    def test_good_and_bad_together_is_an_error(self) -> None:
        """OpenSLO allows one of them; with both, which is the numerator?"""
        doc = _openslo({"good": _source(GOOD), "bad": _source(BAD), "total": _source(TOTAL)})
        with pytest.raises(OpenSLOParseError, match="good.*bad"):
            parse_openslo_slos([doc])


LEGACY = """\
apiVersion: nthlayer.dev/v1
kind: Service
service:
  name: analytics-stream
  team: data
  tier: high
  type: stream-processor
resources:
  - kind: SLO
    name: processing-availability
    spec:
      objective: 99.5
      window: 30d
      indicators:
        - type: availability
          success_ratio:
            total_query: sum(rate(events_processed_total{service="analytics-stream"}[5m]))
            good_query: sum(rate(events_processed_total{service="analytics-stream",status="success"}[5m]))
  - kind: SLO
    name: stream-lag
    spec:
      objective: 99.0
      window: 30d
      indicators:
        - type: custom
          threshold: 60
          metric: max(kafka_consumer_lag_seconds{service="analytics-stream",topic="events"})
"""


class TestLegacySuccessRatio:
    def _slos(self, tmp_path: Path) -> dict:
        path = tmp_path / "analytics-stream.yaml"
        path.write_text(LEGACY)
        return {s.name: s for s in load_manifest(path).slos}

    def test_success_ratio_queries_are_kept(self, tmp_path: Path) -> None:
        slo = self._slos(tmp_path)["processing-availability"]

        assert slo.total_query == 'sum(rate(events_processed_total{service="analytics-stream"}[5m]))'
        assert slo.good_query == (
            'sum(rate(events_processed_total{service="analytics-stream",status="success"}[5m]))'
        )
        assert slo.bad_query is None
        # The singular indicator.query is what fed indicator_query; still absent.
        assert slo.indicator_query is None

    def test_an_indicator_without_a_success_ratio_has_none(self, tmp_path: Path) -> None:
        slo = self._slos(tmp_path)["stream-lag"]

        assert slo.good_query is None
        assert slo.total_query is None
