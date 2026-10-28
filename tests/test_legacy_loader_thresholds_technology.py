"""Legacy-format load: latency thresholds and cache/queue technology
(opensrm-wo62.4.7, rulings R1-R2).

Every shape below is taken from what nthlayer-generate writes or reads for
the legacy format, not from nthlayer-common's output:

- ``spec.indicator.threshold_ms``: generate's built-in critical-api
  template and ``nthlayer init`` scaffold (cli/init.py), and the key
  generate's legacy sloth builder read (generators/sloth.py);
- top-level ``spec.threshold_ms``: ``nthlayer setup`` (cli/setup.py), the
  override hint ``nthlayer init`` writes, and examples/services/auth-api.yaml;
- ``spec.latency_threshold``: the key generate's legacy recording-rule
  builder read (recording_rules/builder.py);
- ``queues``/``caches`` entries with ``type:``: examples/services/email-worker.yaml
  and notification-service.yaml, read by generate's legacy dashboard builder
  (dashboards/builder_sdk.py).

R1 (Rob, 2026-10-28): the three threshold keys are read in that order.
R2: a cache's or queue's ``type:`` is carried in ``Dependency.database_type``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import yaml

from nthlayer_common.manifest import (
    Dependency,
    ManifestLoadError,
    ReliabilityManifest,
    SLODefinition,
    load_manifest,
)


def _load(path: Path) -> ReliabilityManifest:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_manifest(path)


def _write(tmp_path: Path, resources: list[dict]) -> Path:
    doc = {
        "service": {"name": "svc", "team": "platform", "tier": "critical", "type": "api"},
        "resources": resources,
    }
    path = tmp_path / "svc.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return path


def _latency(spec_extra: dict, indicator_extra: dict | None = None) -> dict:
    indicator = {"type": "latency", "percentile": 95, **(indicator_extra or {})}
    return {
        "kind": "SLO",
        "name": "latency-p95",
        "spec": {"objective": 99.0, "window": "30d", "indicator": indicator, **spec_extra},
    }


def _slo(path: Path, name: str = "latency-p95") -> SLODefinition:
    return {s.name: s for s in _load(path).slos}[name]


# =============================================================================
# R1: latency threshold
# =============================================================================


@pytest.mark.parametrize(
    ("spec_extra", "indicator_extra", "expected"),
    [
        pytest.param({}, {"threshold_ms": 500}, 500.0, id="indicator.threshold_ms"),
        pytest.param({"threshold_ms": 300}, {}, 300.0, id="spec.threshold_ms"),
        pytest.param({"latency_threshold": 250}, {}, 250.0, id="spec.latency_threshold"),
        pytest.param({}, {}, None, id="none-declared"),
    ],
)
def test_each_threshold_key_is_read(tmp_path, spec_extra, indicator_extra, expected):
    slo = _slo(_write(tmp_path, [_latency(spec_extra, indicator_extra)]))

    assert slo.threshold_ms == expected


@pytest.mark.parametrize(
    ("spec_extra", "indicator_extra", "expected"),
    [
        pytest.param(
            {"threshold_ms": 300, "latency_threshold": 250},
            {"threshold_ms": 500},
            500.0,
            id="indicator-beats-both",
        ),
        pytest.param(
            {"threshold_ms": 300, "latency_threshold": 250}, {}, 300.0, id="spec-beats-latency"
        ),
    ],
)
def test_threshold_precedence(tmp_path, spec_extra, indicator_extra, expected):
    slo = _slo(_write(tmp_path, [_latency(spec_extra, indicator_extra)]))

    assert slo.threshold_ms == expected


def test_objective_is_not_mistaken_for_the_threshold(tmp_path):
    """The wo62.4.3 regression: a 99% objective became a 99 ms threshold."""
    slo = _slo(_write(tmp_path, [_latency({}, {"threshold_ms": 500})]))

    assert (slo.target, slo.threshold_ms) == (99.0, 500.0)


def test_setup_scaffold_shape(tmp_path):
    """The literal shape ``nthlayer setup`` writes (generate cli/setup.py): a
    latency-p99 SLO with the threshold beside an indicator that has none."""
    resource = {
        "kind": "SLO",
        "name": "latency-p99",
        "spec": {
            "objective": 99.0,
            "window": "30d",
            "threshold_ms": 200,
            "indicator": {"type": "latency", "percentile": 99},
        },
    }

    assert _slo(_write(tmp_path, [resource]), "latency-p99").threshold_ms == 200.0


def test_numeric_string_threshold_is_accepted(tmp_path):
    """Same rule as targets (target_validation.require_number): a string
    spelling a number has always been read as one."""
    slo = _slo(_write(tmp_path, [_latency({}, {"threshold_ms": "500"})]))

    assert slo.threshold_ms == 500.0


@pytest.mark.parametrize("bad", ["500ms", True, [500], float("nan"), 0, -500], ids=repr)
def test_unusable_threshold_is_rejected(tmp_path, bad):
    """Legacy sloth divided the value by 1000.0: a non-number crashed it and
    zero or a negative wrote a query that compares against nothing real.

    Rejected at load, naming the SLO, rather than carried to a consumer.
    ``True`` is an int to Python but never a threshold.
    """
    path = _write(tmp_path, [_latency({}, {"threshold_ms": bad})])

    with pytest.raises(ManifestLoadError, match="latency-p95"):
        _load(path)


def test_template_default_reaches_the_slo(tmp_path):
    """The built-in critical-api template declares indicator.threshold_ms: 500."""
    path = tmp_path / "api-minimal.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "service": {
                    "name": "api-minimal",
                    "team": "platform",
                    "tier": "critical",
                    "type": "api",
                    "template": "critical-api",
                }
            }
        )
    )

    assert _slo(path).threshold_ms == 500.0


def test_override_replacing_a_template_slo_keeps_its_own_threshold(tmp_path):
    """examples/services/auth-api.yaml: the override replaces the template's
    latency-p95 whole (wo62.4.1's replace-not-merge rule), so the template's
    indicator.threshold_ms: 500 is gone and the top-level 200 is the only one."""
    path = tmp_path / "auth-api.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "service": {
                    "name": "auth-api",
                    "team": "platform",
                    "tier": "critical",
                    "type": "api",
                    "template": "critical-api",
                },
                "resources": [
                    {
                        "kind": "SLO",
                        "name": "latency-p95",
                        "spec": {"threshold_ms": 200, "objective": 99.5},
                    }
                ],
            },
            sort_keys=False,
        )
    )

    assert _slo(path).threshold_ms == 200.0


# =============================================================================
# R2: cache and queue technology
# =============================================================================


def _deps(tmp_path: Path, spec: dict) -> dict[str, Dependency]:
    path = _write(tmp_path, [{"kind": "Dependencies", "name": "upstream", "spec": spec}])
    return {d.name: d for d in _load(path).dependencies}


def test_queue_and_cache_type_is_carried(tmp_path):
    deps = _deps(
        tmp_path,
        {
            "queues": [{"name": "email-queue", "type": "rabbitmq", "criticality": "high"}],
            "caches": [{"name": "session-cache", "type": "memcached"}],
        },
    )

    assert (deps["email-queue"].type, deps["email-queue"].database_type) == ("queue", "rabbitmq")
    assert (deps["session-cache"].type, deps["session-cache"].database_type) == (
        "cache",
        "memcached",
    )


def test_undeclared_queue_and_cache_type_stays_none(tmp_path):
    """No type means no technology: the consumer's default applies, as before."""
    deps = _deps(tmp_path, {"queues": [{"name": "jobs"}], "caches": [{"name": "c"}]})

    assert deps["jobs"].database_type is None
    assert deps["c"].database_type is None


def test_database_type_unchanged(tmp_path):
    deps = _deps(tmp_path, {"databases": [{"name": "orders-db", "type": "postgresql"}]})

    assert deps["orders-db"].database_type == "postgresql"


# =============================================================================
# R4: an environment overlay's threshold replaces the base's
# =============================================================================
#
# Rob, 2026-10-28, after the edge-cases review: the overlay is deep-merged into
# the base SLO spec, so without this the base's indicator.threshold_ms outranked
# the overlay's top-level threshold_ms under R1's order. Shapes from generate's
# examples/services/payment-api.yaml and examples/environments/payment-api-prod.yaml
# ("threshold_ms: 300  # Stricter than default 500ms").


def _with_overlay(tmp_path: Path, base_spec: dict, overlay_spec: dict) -> Path:
    path = _write(tmp_path, [_latency(base_spec["extra"], base_spec["indicator"])])
    (tmp_path / "environments").mkdir()
    (tmp_path / "environments" / "svc-prod.yaml").write_text(
        yaml.safe_dump(
            {
                "environment": "prod",
                "resources": [{"kind": "SLO", "name": "latency-p95", "spec": overlay_spec}],
            }
        )
    )
    return path


def _slo_in(path: Path, environment: str | None) -> SLODefinition:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        manifest = load_manifest(path, environment=environment)
    return {s.name: s for s in manifest.slos}["latency-p95"]


@pytest.mark.parametrize(
    ("overlay_spec", "expected"),
    [
        pytest.param({"objective": 99.5, "threshold_ms": 300}, 300.0, id="spec.threshold_ms"),
        pytest.param({"latency_threshold": 200}, 200.0, id="spec.latency_threshold"),
        pytest.param({"indicator": {"threshold_ms": 250}}, 250.0, id="indicator.threshold_ms"),
    ],
)
def test_overlay_threshold_replaces_the_base(tmp_path, overlay_spec, expected):
    path = _with_overlay(tmp_path, {"extra": {}, "indicator": {"threshold_ms": 500}}, overlay_spec)

    assert _slo_in(path, None).threshold_ms == 500.0
    assert _slo_in(path, "prod").threshold_ms == expected


def test_overlay_threshold_replaces_a_lower_priority_base_key(tmp_path):
    """Every base key goes, not just indicator.threshold_ms: a surviving base
    threshold_ms would outrank the overlay's latency_threshold under R1."""
    path = _with_overlay(
        tmp_path, {"extra": {"threshold_ms": 500}, "indicator": {}}, {"latency_threshold": 200}
    )

    assert _slo_in(path, "prod").threshold_ms == 200.0


def test_overlay_without_threshold_keeps_the_base(tmp_path):
    path = _with_overlay(
        tmp_path, {"extra": {}, "indicator": {"threshold_ms": 500}}, {"objective": 99.5}
    )

    slo = _slo_in(path, "prod")
    assert (slo.target, slo.threshold_ms) == (99.5, 500.0)


def test_overlay_threshold_keeps_the_rest_of_the_indicator(tmp_path):
    """Only threshold keys are removed from the base; the query survives."""
    query = "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))"
    path = _with_overlay(
        tmp_path,
        {"extra": {}, "indicator": {"threshold_ms": 500, "query": query}},
        {"threshold_ms": 300},
    )

    slo = _slo_in(path, "prod")
    assert (slo.threshold_ms, slo.indicator_query) == (300.0, query)
