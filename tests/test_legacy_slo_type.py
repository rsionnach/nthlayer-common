"""Legacy-format load: inferring an SLO's slo_type (opensrm-e9ts).

The loader read only a singular ``spec.indicator.type`` and otherwise fell
back to name keywords, then to "availability". Real legacy manifests declare
a plural ``indicators:`` list, so lag and freshness SLOs loaded as
availability, nthlayer-generate wrote them an error budget from request
success, and nthlayer-workers' measure read it as theirs.

Each positive shape below is a resource copied verbatim from nthlayer-generate's
legacy corpus (cited per test), not from this loader's output.

R1 (Rob, 2026-11-02): the types in an ``indicators:`` list must agree, or the
load fails naming the SLO; untyped entries are ignored.
R2: a typeless SLO that declares a latency threshold
(legacy.LATENCY_THRESHOLD_KEYS) is latency.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import yaml

from nthlayer_common.manifest import ManifestLoadError, load_manifest


def _slo_type(tmp_path: Path, resource: dict) -> str:
    doc = {
        "service": {"name": "svc", "team": "platform", "tier": "critical", "type": "api"},
        "resources": [resource],
    }
    path = tmp_path / "svc.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (slo,) = load_manifest(path).slos
    return slo.slo_type


# nthlayer-generate examples/services/analytics-stream.yaml
STREAM_LAG = {
    "kind": "SLO",
    "name": "stream-lag",
    "spec": {
        "objective": 99.0,
        "window": "30d",
        "indicators": [
            {
                "type": "custom",
                "threshold": 60,
                "metric": 'max(kafka_consumer_lag_seconds{service="analytics-stream",topic="events"})',
            }
        ],
    },
}

# nthlayer-generate examples/services/notification-worker.yaml
QUEUE_LAG = {
    "kind": "SLO",
    "name": "queue-lag",
    "spec": {
        "objective": 99.0,
        "window": "30d",
        "indicators": [
            {
                "type": "custom",
                "threshold": 100,
                "metric": 'max(kafka_consumer_lag_seconds{service="notification-worker",topic="notifications"})',
            }
        ],
    },
}

# nthlayer-generate examples/services/notification-worker.yaml
DELIVERY_SUCCESS = {
    "kind": "SLO",
    "name": "delivery-success",
    "spec": {
        "objective": 99.5,
        "window": "30d",
        "indicators": [
            {
                "type": "availability",
                "success_ratio": {
                    "total_query": 'sum(rate(notifications_sent_total{service="notification-worker"}[5m]))',
                    "good_query": 'sum(rate(notifications_sent_total{service="notification-worker",status="delivered"}[5m]))',
                },
            }
        ],
    },
}

# nthlayer-generate examples/services/analytics-pipeline.yaml
FRESHNESS_P95 = {
    "kind": "SLO",
    "name": "freshness-p95",
    "spec": {"threshold_ms": 43200000, "objective": 90.0},
}

# nthlayer-common tests/fixtures/legacy_loader/services/substitution-defaults.yaml: declares
# nothing a type could come from.
SUCCESS_RATE = {
    "kind": "SLO",
    "name": "success-rate",
    "spec": {"objective": 99.0, "window": "30d", "description": "d"},
}


@pytest.mark.parametrize(
    ("resource", "expected"),
    [
        (STREAM_LAG, "custom"),
        (QUEUE_LAG, "custom"),
        (DELIVERY_SUCCESS, "availability"),
    ],
    ids=["stream-lag", "queue-lag", "delivery-success"],
)
def test_plural_indicators_type_is_read(tmp_path, resource, expected):
    assert _slo_type(tmp_path, resource) == expected


def test_a_typeless_slo_with_a_latency_threshold_is_latency(tmp_path):
    assert _slo_type(tmp_path, FRESHNESS_P95) == "latency"


def test_an_slo_declaring_nothing_still_defaults_to_availability(tmp_path):
    assert _slo_type(tmp_path, SUCCESS_RATE) == "availability"


def test_untyped_entries_do_not_count(tmp_path):
    resource = {
        **STREAM_LAG,
        "spec": {
            **STREAM_LAG["spec"],
            "indicators": [{"threshold": 5}, *STREAM_LAG["spec"]["indicators"]],
        },
    }
    assert _slo_type(tmp_path, resource) == "custom"


def test_agreeing_entries_are_one_type(tmp_path):
    entry = STREAM_LAG["spec"]["indicators"][0]
    resource = {**STREAM_LAG, "spec": {**STREAM_LAG["spec"], "indicators": [entry, entry]}}
    assert _slo_type(tmp_path, resource) == "custom"


def test_disagreeing_entries_fail_the_load_naming_the_slo(tmp_path):
    # R1: a list whose entries declare different types has no single
    # slo_type, and guessing one is how lag SLOs became availability.
    resource = {
        **STREAM_LAG,
        "spec": {
            **STREAM_LAG["spec"],
            "indicators": [
                *STREAM_LAG["spec"]["indicators"],
                DELIVERY_SUCCESS["spec"]["indicators"][0],
            ],
        },
    }
    with pytest.raises(ManifestLoadError, match="stream-lag"):
        _slo_type(tmp_path, resource)


def test_a_singular_indicator_type_still_wins(tmp_path):
    # nthlayer-generate examples/services/search-api.yaml's search-quality declares `indicator: {type: custom}`;
    # the singular key keeps its precedence over a plural list.
    resource = {
        **DELIVERY_SUCCESS,
        "spec": {**DELIVERY_SUCCESS["spec"], "indicator": {"type": "custom"}},
    }
    assert _slo_type(tmp_path, resource) == "custom"


@pytest.mark.parametrize("bad", [["latency"], {"kind": "latency"}, 3, True], ids=repr)
def test_a_non_string_indicator_type_fails_the_load_naming_the_slo(tmp_path, bad):
    # Without the check, a list or mapping raised a raw TypeError out of
    # load_manifest (it wraps only ValueError), and a scalar became slo_type.
    resource = {**STREAM_LAG, "spec": {**STREAM_LAG["spec"], "indicators": [{"type": bad}]}}
    with pytest.raises(ManifestLoadError, match="stream-lag"):
        _slo_type(tmp_path, resource)


@pytest.mark.parametrize(
    "bad",
    [{"type": "custom"}, ["custom"], "custom"],
    ids=["mapping", "list-of-str", "str"],
)
def test_a_malformed_indicators_list_fails_the_load_naming_the_slo(tmp_path, bad):
    # Ignoring it would fall back to availability: the failure this bead fixes.
    resource = {**STREAM_LAG, "spec": {**STREAM_LAG["spec"], "indicators": bad}}
    with pytest.raises(ManifestLoadError, match="stream-lag"):
        _slo_type(tmp_path, resource)


def test_a_blank_type_counts_as_untyped(tmp_path):
    resource = {
        **STREAM_LAG,
        "spec": {
            **STREAM_LAG["spec"],
            "indicators": [{"type": "  "}, *STREAM_LAG["spec"]["indicators"]],
        },
    }
    assert _slo_type(tmp_path, resource) == "custom"


def test_types_compare_as_written(tmp_path):
    # Pinned, not normalised: the loader passes a type through as declared
    # (opensrm-hiw2 tracks normalising), so two spellings are two types.
    entry = STREAM_LAG["spec"]["indicators"][0]
    resource = {
        **STREAM_LAG,
        "spec": {**STREAM_LAG["spec"], "indicators": [entry, {**entry, "type": "Custom"}]},
    }
    with pytest.raises(ManifestLoadError, match="stream-lag"):
        _slo_type(tmp_path, resource)


def test_a_name_keyword_outranks_a_latency_threshold(tmp_path):
    resource = {"kind": "SLO", "name": "errors-p99", "spec": {"threshold_ms": 500, "objective": 99.0}}
    assert _slo_type(tmp_path, resource) == "error_rate"
