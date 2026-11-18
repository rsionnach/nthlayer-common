"""Legacy-format load: the SLO target is a finite number (opensrm-1lhl).

The legacy loader was the last of the five writers of
``SLODefinition.target`` left unguarded by opensrm-ocvu. Measured on it
then, and written here as the YAML an author would type:

- ``objective: .nan`` loaded as a NaN target, and a NaN target makes every
  comparison in ``validate_contracts()`` false, so a breach reported clean;
- ``objective: {}`` (any falsy value) made the SLO vanish from the manifest
  through ``objective or target``, with nothing reported.

Rob's rulings (2026-11-18): a non-finite objective is refused; a present,
non-numeric falsy objective is refused; a numeric 0 is kept as a 0% target;
a missing or null objective still drops the SLO (opensrm-wo62.4.17). Range
is out of scope, as it was in opensrm-ocvu: ``objective: -5`` loads.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import pytest

from nthlayer_common.manifest import ManifestLoadError, SLODefinition, load_manifest

MANIFEST = """\
service: {{name: svc, team: platform, tier: critical, type: api}}
resources:
  - kind: SLO
    name: checkout-availability
    spec:
{spec}
  - kind: SLO
    name: other
    spec: {{objective: 99.0}}
"""


def _slos(tmp_path: Path, *spec_lines: str) -> dict[str, SLODefinition]:
    path = tmp_path / "svc.yaml"
    path.write_text(MANIFEST.format(spec="\n".join(f"      {line}" for line in spec_lines)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {s.name: s for s in load_manifest(path).slos}


def _refused(tmp_path: Path, *spec_lines: str) -> ManifestLoadError:
    with pytest.raises(ManifestLoadError) as raised:
        _slos(tmp_path, *spec_lines)
    return raised.value


@pytest.mark.parametrize("key", ["objective", "target"])
@pytest.mark.parametrize("value", [".nan", ".inf", "-.inf", "'nan'", "'inf'", "1e400"])
def test_a_non_finite_target_is_refused_naming_the_slo(tmp_path, key, value):
    error = _refused(tmp_path, f"{key}: {value}")

    assert "checkout-availability" in str(error)
    assert error.backstop is False


@pytest.mark.parametrize("value", ["{}", "[]", "''", "false"])
def test_a_falsy_objective_that_is_no_number_is_refused_not_dropped(tmp_path, value):
    error = _refused(tmp_path, f"objective: {value}")

    assert "checkout-availability" in str(error)


@pytest.mark.parametrize("key", ["objective", "target"])
@pytest.mark.parametrize("value", ["true", "false", "soon", "[99]", "{value: 99}", "2024-01-01"])
def test_a_target_that_is_no_number_is_refused(tmp_path, key, value):
    """``true`` used to load as 1.0: a boolean is an int to Python, never a
    target. ``soon`` used to leave as a bare ValueError."""
    error = _refused(tmp_path, f"{key}: {value}")

    assert "checkout-availability" in str(error)


@pytest.mark.parametrize("key", ["objective", "target"])
@pytest.mark.parametrize(
    ("value", "expected"),
    [("99.9", 99.9), ("'99.9'", 99.9), ("0", 0.0), ("0.0", 0.0), ("-5", -5.0), ("250", 250.0)],
)
def test_a_number_is_the_target_as_declared(tmp_path, key, value, expected):
    """0 included: it used to drop the SLO. A string spelling a number has
    always loaded, and range is not this guard's."""
    slo = _slos(tmp_path, f"{key}: {value}")["checkout-availability"]

    assert slo.target == expected and math.isfinite(slo.target)


def test_a_zero_objective_is_not_passed_over_for_the_target_key(tmp_path):
    """``objective or target`` read past a 0 objective to ``target``."""
    slo = _slos(tmp_path, "objective: 0", "target: 99.0")["checkout-availability"]

    assert slo.target == 0.0


@pytest.mark.parametrize("objective", ["objective:", "objective: null", "window: 30d"])
def test_a_null_or_missing_objective_falls_back_to_target(tmp_path, objective):
    slo = _slos(tmp_path, objective, "target: 99.5")["checkout-availability"]

    assert slo.target == 99.5


@pytest.mark.parametrize("spec", ["window: 30d", "objective: null", "target:"])
def test_an_slo_that_declares_no_target_is_still_dropped(tmp_path, spec):
    """Unchanged here; whether it should be is opensrm-wo62.4.17's."""
    assert list(_slos(tmp_path, spec)) == ["other"]
