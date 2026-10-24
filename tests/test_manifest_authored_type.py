"""ReliabilityManifest.authored_type keeps the spelling the author wrote.

The invariant is opensrm-z3ab's (see the `type`/`authored_type` handling in nthlayer-generate's specs/models.py):
`type` is resolved for internal comparisons; `authored_type` is what `${type}`
substitutes into generated PromQL, where rewriting `web` to `x-web` selects
zero series in the operator's Prometheus. Alias spellings are common's
SERVICE_TYPE_ALIASES in nthlayer_common.manifest.models, the documented input conveniences; the
extension spelling is opensrm/spec/v2/examples/x-service-type.yaml's.
"""
import dataclasses
from pathlib import Path

import pytest
import yaml

from nthlayer_common.manifest import ReliabilityManifest, load_manifest
from nthlayer_common.manifest.parser.v1 import parse_srm_v1
from nthlayer_common.manifest.parser.v2 import parse_opensrm_v2

OPENSRM = Path(__file__).resolve().parents[2] / "opensrm"
X_TYPE_EXAMPLE = OPENSRM / "spec" / "v2" / "examples" / "x-service-type.yaml"
ALIASES = [("web", "x-web"), ("background-job", "worker"), ("pipeline", "batch")]


def _v1(service_type: str) -> dict:
    return {"apiVersion": "srm/v1", "kind": "ServiceReliabilityManifest",
            "metadata": {"name": "shop", "team": "t", "tier": "critical"},
            "spec": {"type": service_type}}


def _v2(service_type: str) -> dict:
    return {"apiVersion": "opensrm.nthlayer.io/v2", "kind": "ServiceManifest",
            "metadata": {"name": "shop", "labels": {"tier": "critical"}},
            "spec": {"owner": {"group": "group:default/t"},
                     "service": {"name": "shop", "type": service_type}}}


@pytest.mark.parametrize("authored, resolved", ALIASES)
def test_srm_v1_keeps_the_authored_alias(authored, resolved):
    m = parse_srm_v1(_v1(authored))
    assert (m.type, m.authored_type) == (resolved, authored)


@pytest.mark.parametrize("authored, resolved", ALIASES)
def test_v2_keeps_the_authored_alias(authored, resolved):
    m = parse_opensrm_v2(_v2(authored))
    assert (m.type, m.authored_type) == (resolved, authored)


@pytest.mark.parametrize("authored, resolved", ALIASES)
def test_legacy_keeps_the_authored_alias(tmp_path, authored, resolved):
    p = tmp_path / "svc.yaml"
    p.write_text(yaml.safe_dump({"service": {"name": "shop", "team": "t",
                                             "tier": "critical", "type": authored}}))
    m = load_manifest(p, suppress_deprecation_warning=True)
    assert (m.type, m.authored_type) == (resolved, authored)


def test_the_shipped_extension_example_is_still_missing_tier():
    """Tripwire: fails when opensrm-ta0m fixes the example, so the injection below can go."""
    assert X_TYPE_EXAMPLE.is_file(), "the opensrm sibling is missing"
    doc = yaml.safe_load(X_TYPE_EXAMPLE.read_text())
    assert doc["spec"]["service"]["type"] == "x-web"
    assert "tier" not in (doc["metadata"].get("labels") or {})


def test_the_spec_extension_example_keeps_its_type(tmp_path):
    # Provenance: copy of opensrm/spec/v2/examples/x-service-type.yaml at opensrm
    # 859424a. The ONLY change is injecting metadata.labels.tier: critical, because
    # the shipped example omits it and common's v2 parser requires it (a spec/parser
    # divergence tracked as opensrm-ta0m). x-web is a canonical value (the alias
    # target of `web`), so this exercises the no-alias path. spec.service.type is read from
    # the file, not restated here.
    doc = yaml.safe_load(X_TYPE_EXAMPLE.read_text())
    declared = doc["spec"]["service"]["type"]
    doc["metadata"].setdefault("labels", {})["tier"] = "critical"
    p = tmp_path / "x.yaml"
    p.write_text(yaml.safe_dump(doc))
    m = load_manifest(p)
    assert m.authored_type == m.type == declared


def test_authored_type_survives_dataclasses_replace():
    m = ReliabilityManifest(name="s", team="t", tier="critical", type="web")
    assert dataclasses.replace(m, tier="high").authored_type == "web"


def test_explicit_authored_type_is_not_overwritten():
    m = ReliabilityManifest(name="s", team="t", tier="critical", type="x-web", authored_type="web")
    assert (m.type, m.authored_type) == ("x-web", "web")
