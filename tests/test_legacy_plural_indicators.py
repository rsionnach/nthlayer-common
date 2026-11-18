"""Legacy-format load: the plural ``indicators:`` list's latency threshold
(opensrm-wo62.4.24).

Real legacy manifests declare a latency SLO's bound only inside the plural
list (``indicators: [{type: latency, threshold_ms: 800, ...}]``). The loader
read ``indicator.threshold_ms``, ``threshold_ms`` and ``latency_threshold``
(opensrm-wo62.4.7) and never the list, so those SLOs loaded with no
threshold and nthlayer-generate gave them no error budget.

The shapes are taken from four of generate's own example files, snapshotted
in ``tests/fixtures/legacy_plural_indicators/`` (see the README there), and
the expected numbers are read off those files, not off the loader.

The list's rule is the one opensrm-e9ts set for its ``type``: entries must
agree and an entry that declares nothing is ignored.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import yaml

from nthlayer_common.manifest import ManifestLoadError, SLODefinition, load_manifest
from nthlayer_common.manifest.parser import legacy

CORPUS = Path(__file__).parent / "fixtures" / "legacy_plural_indicators"

# (file, SLO name, the threshold_ms its indicators[] entry declares), as
# written in generate's examples/services/ at 8230fb6.
CORPUS_THRESHOLDS = [
    ("analytics-stream.yaml", "event-latency-p99", 10000),
    ("checkout-service.yaml", "latency-p95", 800),
    ("checkout-service.yaml", "latency-p99", 1500),
    ("identity-service.yaml", "latency-p95", 200),
    ("identity-service.yaml", "latency-p99", 500),
    ("notification-worker.yaml", "processing-latency-p95", 5000),
]


def _load(path: Path, environment: str | None = None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_manifest(path, environment=environment)


def _slos(path: Path, environment: str | None = None) -> dict[str, SLODefinition]:
    return {s.name: s for s in _load(path, environment).slos}


def _write(tmp_path: Path, spec: dict, name: str = "checkout-latency") -> Path:
    doc = {
        "service": {"name": "svc", "team": "platform", "tier": "critical", "type": "api"},
        "resources": [{"kind": "SLO", "name": name, "spec": {"objective": 99.0, **spec}}],
    }
    path = tmp_path / "svc.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return path


def _threshold(tmp_path: Path, spec: dict) -> float | None:
    return _slos(_write(tmp_path, spec))["checkout-latency"].threshold_ms


def _entry(**extra) -> dict:
    return {"type": "latency", "percentile": 95, **extra}


# =============================================================================
# The corpus
# =============================================================================


def test_the_table_is_what_the_corpus_files_declare():
    """The numbers above against the files themselves, read as plain YAML:
    a wrong row here would otherwise pass against a matching loader bug."""
    declared = []
    for path in sorted(CORPUS.glob("*.yaml")):
        for resource in yaml.safe_load(path.read_text())["resources"]:
            for entry in resource.get("spec", {}).get("indicators", []):
                if "threshold_ms" in entry:
                    declared.append((path.name, resource["name"], entry["threshold_ms"]))

    assert declared == CORPUS_THRESHOLDS


@pytest.mark.parametrize(("file", "name", "declared"), CORPUS_THRESHOLDS)
def test_a_corpus_latency_slo_carries_its_declared_threshold(file, name, declared):
    slo = _slos(CORPUS / file)[name]

    assert (slo.slo_type, slo.threshold_ms) == ("latency", float(declared))


@pytest.mark.parametrize(
    ("file", "name"),
    [("analytics-stream.yaml", "stream-lag"), ("notification-worker.yaml", "queue-lag")],
)
def test_a_custom_indicators_threshold_is_not_a_latency_threshold(file, name):
    """``threshold: 60`` on a ``type: custom`` entry is a bound on the SLO's
    own metric (seconds of lag), not milliseconds of latency."""
    slo = _slos(CORPUS / file)[name]

    assert (slo.slo_type, slo.threshold_ms) == ("custom", None)


def test_a_corpus_availability_slo_has_no_threshold():
    assert _slos(CORPUS / "checkout-service.yaml")["availability"].threshold_ms is None


# =============================================================================
# Order: singular indicator, plural list, then the SLO's own keys
# =============================================================================


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        pytest.param({"indicators": [_entry(threshold_ms=800)]}, 800.0, id="list-alone"),
        pytest.param(
            {"indicator": {"threshold_ms": 500}, "indicators": [_entry(threshold_ms=800)]},
            500.0,
            id="singular-beats-list",
        ),
        pytest.param(
            {"threshold_ms": 300, "indicators": [_entry(threshold_ms=800)]},
            800.0,
            id="list-beats-spec.threshold_ms",
        ),
        pytest.param(
            {"latency_threshold": 250, "indicators": [_entry(threshold_ms=800)]},
            800.0,
            id="list-beats-spec.latency_threshold",
        ),
        pytest.param({"threshold_ms": 300, "indicators": [_entry()]}, 300.0, id="list-silent"),
    ],
)
def test_threshold_order(tmp_path, spec, expected):
    assert _threshold(tmp_path, spec) == expected


def test_a_singular_threshold_means_the_list_is_not_read(tmp_path):
    """Only the first declared threshold is validated (opensrm-wo62.4.7)."""
    spec = {"indicator": {"threshold_ms": 500}, "indicators": [_entry(threshold_ms="soon")]}

    assert _threshold(tmp_path, spec) == 500.0


# =============================================================================
# The list's rule
# =============================================================================


@pytest.mark.parametrize(
    ("entries", "expected"),
    [
        pytest.param([_entry(threshold_ms=800), _entry(threshold_ms=800)], 800.0, id="same"),
        pytest.param([_entry(threshold_ms=800), _entry(threshold_ms="800.0")], 800.0, id="same-n"),
        pytest.param([_entry(), _entry(threshold_ms=800)], 800.0, id="undeclared-first"),
        pytest.param([_entry(threshold_ms=None), _entry(threshold_ms=800)], 800.0, id="null"),
        pytest.param([_entry(), _entry()], None, id="none-declared"),
        pytest.param([], None, id="empty-list"),
    ],
)
def test_entries_that_agree_give_one_threshold(tmp_path, entries, expected):
    assert _threshold(tmp_path, {"indicators": entries}) == expected


def test_entries_that_disagree_fail_the_load_naming_the_slo(tmp_path):
    path = _write(tmp_path, {"indicators": [_entry(threshold_ms=800), _entry(threshold_ms=1500)]})

    with pytest.raises(ManifestLoadError) as raised:
        _load(path)

    assert "checkout-latency" in str(raised.value)


@pytest.mark.parametrize("bad", ["500ms", True, [500], float("nan"), 0, -500], ids=repr)
@pytest.mark.parametrize("position", [0, 1])
def test_an_unusable_threshold_in_any_entry_is_rejected(tmp_path, bad, position):
    """The same values the other threshold keys refuse (opensrm-wo62.4.7),
    in whichever entry: every declared one is validated."""
    entries = [_entry(threshold_ms=800), _entry(threshold_ms=800)]
    entries[position]["threshold_ms"] = bad
    path = _write(tmp_path, {"indicators": entries})

    with pytest.raises(ManifestLoadError) as raised:
        _load(path)

    assert "checkout-latency" in str(raised.value)


@pytest.mark.parametrize(
    ("indicators", "expected"),
    [
        pytest.param("latency", None, id="string"),
        pytest.param({"threshold_ms": 800}, None, id="mapping"),
        pytest.param([_entry(threshold_ms=800), "latency"], 800.0, id="non-mapping-entry"),
    ],
)
def test_a_malformed_list_beside_a_singular_type_still_loads(tmp_path, indicators, expected):
    """Such a file loaded before the list's threshold was read, so it still
    does: what is not a mapping entry of a list declares no threshold."""
    spec = {"indicator": {"type": "latency"}, "indicators": indicators}

    assert _threshold(tmp_path, spec) == expected


@pytest.mark.parametrize("indicators", ["latency", [_entry(threshold_ms=800), "latency"]])
def test_a_malformed_list_that_types_the_slo_still_fails_the_load(tmp_path, indicators):
    """opensrm-e9ts: with no singular indicator.type the list is read for
    the type, and must be a list of mappings."""
    path = _write(tmp_path, {"indicators": indicators})

    with pytest.raises(ManifestLoadError) as raised:
        _load(path)

    assert "checkout-latency" in str(raised.value)


def test_an_untyped_entry_with_a_threshold_makes_the_slo_latency(tmp_path):
    """A declared latency threshold types an SLO that nothing else types
    (opensrm-e9ts, R2); the list's threshold counts as one."""
    path = _write(tmp_path, {"indicators": [{"threshold_ms": 800}]}, name="checkout-speed")

    slo = _slos(path)["checkout-speed"]
    assert (slo.slo_type, slo.threshold_ms) == ("latency", 800.0)


# =============================================================================
# Environment overlay: an overlay's threshold replaces the base's (R4,
# opensrm-wo62.4.7), whichever of the four places either uses.
# =============================================================================


def _with_overlay(tmp_path: Path, base_spec: dict, overlay_spec: dict) -> Path:
    path = _write(tmp_path, base_spec)
    (tmp_path / "environments").mkdir()
    (tmp_path / "environments" / "svc-prod.yaml").write_text(
        yaml.safe_dump(
            {
                "environment": "prod",
                "resources": [{"kind": "SLO", "name": "checkout-latency", "spec": overlay_spec}],
            }
        )
    )
    return path


def _prod(path: Path) -> SLODefinition:
    return _slos(path, "prod")["checkout-latency"]


@pytest.mark.parametrize(
    "base_spec",
    [
        pytest.param({"indicator": {"type": "latency", "threshold_ms": 500}}, id="singular"),
        pytest.param({"threshold_ms": 500}, id="spec.threshold_ms"),
        pytest.param({"indicators": [_entry(threshold_ms=500)]}, id="list"),
    ],
)
def test_an_overlay_list_threshold_replaces_the_base(tmp_path, base_spec):
    path = _with_overlay(tmp_path, base_spec, {"indicators": [_entry(threshold_ms=300)]})

    assert _slos(path)["checkout-latency"].threshold_ms == 500.0
    assert _prod(path).threshold_ms == 300.0


@pytest.mark.parametrize(
    "overlay_spec",
    [
        pytest.param({"threshold_ms": 300}, id="spec.threshold_ms"),
        pytest.param({"latency_threshold": 300}, id="spec.latency_threshold"),
        pytest.param({"indicator": {"threshold_ms": 300}}, id="singular"),
    ],
)
def test_an_overlay_threshold_replaces_a_base_list_threshold(tmp_path, overlay_spec):
    """Without removing the base list's threshold, it would outrank the
    overlay's top-level key under the order above."""
    path = _with_overlay(tmp_path, {"indicators": [_entry(threshold_ms=500)]}, overlay_spec)

    slo = _prod(path)
    # The rest of the base entry survives: its type still types the SLO.
    assert (slo.threshold_ms, slo.slo_type) == (300.0, "latency")


def test_an_overlay_without_a_threshold_keeps_the_base_list_threshold(tmp_path):
    path = _with_overlay(tmp_path, {"indicators": [_entry(threshold_ms=500)]}, {"objective": 99.5})

    slo = _prod(path)
    assert (slo.target, slo.threshold_ms) == (99.5, 500.0)


# =============================================================================
# legacy.indicators_thresholds: shared by the loader and the overlay merge
# =============================================================================


@pytest.mark.parametrize(
    ("indicators", "expected"),
    [
        pytest.param([{"threshold_ms": 800}, {"type": "latency"}, {"threshold_ms": 5}], [800, 5]),
        pytest.param([{"threshold_ms": None}], []),
        pytest.param([{"threshold_ms": 0}], [0]),
        pytest.param(["latency", {"threshold_ms": 800}], [800]),
        pytest.param(None, []),
        pytest.param("latency", []),
        pytest.param({"threshold_ms": 800}, []),
    ],
)
def test_indicators_thresholds(indicators, expected):
    assert legacy.indicators_thresholds(indicators) == expected
