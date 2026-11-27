"""PagerDuty integration keys stay out of reprs, and two legacy warnings
name their manifest (opensrm-wo62.4.14).

An integration key is a PagerDuty routing secret: whoever holds one can
open incidents on the service. ``PagerDutyConfig`` is a dataclass, so its
default repr printed both keys, and ``ReliabilityManifest``'s repr printed
them twice more: through ``ownership.pagerduty`` and through ``raw_data``,
the document as loaded. A manifest that is logged, or that appears in a
traceback, then put the keys in the log.

The keys below are made up for this test.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from structlog.testing import capture_logs

from nthlayer_common.manifest import load_manifest
from nthlayer_common.manifest.models import PagerDutyConfig

INTEGRATION_KEY = "R0UTINGKEYa1b2c3d4e5f6a7b8c9d0e1f2"
SRE_INTEGRATION_KEY = "SREKEYf6e5d4c3b2a1f6e5d4c3b2a1f6e5"

LEGACY = f"""\
service:
  name: checkout
  team: payments
  tier: critical
  type: api
resources:
  - kind: PagerDuty
    name: pagerduty
    spec:
      service_id: PSVC1
      escalation_policy: PESC1
      integration_key: {INTEGRATION_KEY}
      sre_integration_key: {SRE_INTEGRATION_KEY}
"""


def _load(tmp_path: Path, text: str):
    path = tmp_path / "checkout.yaml"
    path.write_text(text)
    return path, load_manifest(path, suppress_deprecation_warning=True)


def test_the_keys_load_and_are_read_as_before(tmp_path):
    # Hiding them from the repr must not lose them: generate reads both.
    _, manifest = _load(tmp_path, LEGACY)

    pagerduty = manifest.ownership.pagerduty
    assert pagerduty.integration_key == INTEGRATION_KEY
    assert pagerduty.sre_integration_key == SRE_INTEGRATION_KEY
    assert pagerduty.service_id == "PSVC1"


def test_the_config_repr_shows_neither_key():
    config = PagerDutyConfig(
        service_id="PSVC1",
        integration_key=INTEGRATION_KEY,
        sre_integration_key=SRE_INTEGRATION_KEY,
    )

    shown = repr(config)

    assert INTEGRATION_KEY not in shown
    assert SRE_INTEGRATION_KEY not in shown
    # The rest of the config is still shown: the repr is not blanked.
    assert "PSVC1" in shown


@pytest.mark.parametrize("show", [repr, str], ids=["repr", "str"])
def test_a_loaded_manifest_shows_neither_key(show, tmp_path):
    _, manifest = _load(tmp_path, LEGACY)

    shown = show(manifest)

    assert INTEGRATION_KEY not in shown
    assert SRE_INTEGRATION_KEY not in shown
    assert "checkout" in shown


def test_the_raw_document_is_kept_but_not_shown(tmp_path):
    _, manifest = _load(tmp_path, LEGACY)

    # Consumers still get the document as loaded...
    assert manifest.raw_data["resources"][0]["spec"]["integration_key"] == INTEGRATION_KEY
    # ...and it is what would have shown the keys a second time.
    assert "raw_data" not in repr(manifest)


def test_configs_with_different_keys_are_still_different():
    # repr=False must not have become compare=False.
    one = PagerDutyConfig(integration_key=INTEGRATION_KEY)
    other = PagerDutyConfig(integration_key=SRE_INTEGRATION_KEY)

    assert one != other
    assert one == PagerDutyConfig(integration_key=INTEGRATION_KEY)


def test_the_pagerduty_warning_names_the_service_and_the_file(tmp_path):
    text = LEGACY.replace("  type: api\n", "  type: api\n  pagerduty: not-a-mapping\n")
    with capture_logs() as logs:
        path, _ = _load(tmp_path, text)

    (event,) = [e for e in logs if e["event"] == "legacy_service_pagerduty_not_mapping"]
    assert event["service"] == "checkout"
    assert event["source_file"] == str(path)
    assert event["value_type"] == "str"


def test_the_observability_warning_names_the_service_and_the_file(tmp_path):
    text = LEGACY + "  - kind: Observability\n    name: observability\n    spec: not-a-mapping\n"
    with capture_logs() as logs:
        path, _ = _load(tmp_path, text)

    (event,) = [e for e in logs if e["event"] == "legacy_observability_spec_not_mapping"]
    assert event["service"] == "checkout"
    assert event["source_file"] == str(path)
    assert event["spec_type"] == "str"
