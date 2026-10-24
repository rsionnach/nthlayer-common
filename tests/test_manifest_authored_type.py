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
from nthlayer_common.manifest.models import SERVICE_TYPE_ALIASES
from nthlayer_common.manifest.parser.loader import ManifestLoadError
from nthlayer_common.manifest.parser.v1 import OpenSRMParseError, parse_srm_v1
from nthlayer_common.manifest.parser.v2 import OpenSRMV2ParseError, parse_opensrm_v2

OPENSRM = Path(__file__).resolve().parents[2] / "opensrm"
X_TYPE_EXAMPLE = OPENSRM / "spec" / "v2" / "examples" / "x-service-type.yaml"
# Literal on purpose (spec-facing fixture); test_aliases_fixture_covers_every_alias
# fails when a new alias is added to SERVICE_TYPE_ALIASES without coverage here.
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


# --- r5 edge-cases (opensrm-wo62.1) -----------------------------------------
REJECTED = (ValueError, ManifestLoadError, OpenSRMParseError, OpenSRMV2ParseError)


def _m(**kw):
    return ReliabilityManifest(name="s", team="t", tier="critical", **kw)


@pytest.mark.parametrize("bad", ["", "nonsense", "api"])
def test_explicit_authored_type_must_be_valid_and_resolve_to_type(bad):
    # "api" is valid but resolves to api, not x-web: a mismatch, not an alias.
    with pytest.raises(ValueError, match="authored_type") as e:
        _m(type="x-web", authored_type=bad)
    assert "x-web" in str(e.value)


def test_non_string_authored_type_is_rejected():
    with pytest.raises(ValueError, match="authored_type"):
        _m(type="web", authored_type=["web"])


def test_replace_type_without_authored_type_raises():
    m = _m(type="web")
    with pytest.raises(ValueError, match="authored_type"):
        dataclasses.replace(m, type="api")
    assert dataclasses.replace(m, type="api", authored_type="api").authored_type == "api"


@pytest.mark.parametrize("variant", [" Web ", "WEB", "Web", "web "])
def test_case_and_whitespace_variants_are_rejected_not_normalised(tmp_path, variant):
    with pytest.raises(REJECTED):
        parse_srm_v1(_v1(variant))
    with pytest.raises(REJECTED):
        parse_opensrm_v2(_v2(variant))
    p = tmp_path / "svc.yaml"
    p.write_text(yaml.safe_dump({"service": {"name": "shop", "team": "t",
                                             "tier": "critical", "type": variant}}))
    with pytest.raises(REJECTED):
        load_manifest(p, suppress_deprecation_warning=True)


def test_deepcopy_and_pickle_keep_authored_type():
    import copy
    import pickle
    m = _m(type="web")
    for clone in (copy.deepcopy(m), pickle.loads(pickle.dumps(m))):
        assert (clone.type, clone.authored_type) == ("x-web", "web")


def test_v2_judgment_guard_names_the_resolved_type():
    data = _v2("web")
    data["spec"]["judgment_slo"] = {}
    with pytest.raises(OpenSRMV2ParseError, match="'x-web'"):
        parse_opensrm_v2(data)


def test_equality_ignores_authored_type_but_repr_shows_it():
    a, b = _m(type="web"), _m(type="x-web")
    assert a.authored_type != b.authored_type
    assert a == b
    assert "authored_type='web'" in repr(a)


def test_authored_type_none_means_derive_from_type():
    assert _m(type="web", authored_type=None).authored_type == "web"


def test_aliases_fixture_covers_every_alias():
    assert dict(ALIASES) == SERVICE_TYPE_ALIASES
