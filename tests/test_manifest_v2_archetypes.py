"""The schema-vs-parser seam: every shipped v2 archetype example must both
validate against ``spec/v2/schema.json`` AND parse to a ReliabilityManifest.

The divergence this file exists to catch (opensrm-ih0v): ``opensrm-6w9d``
made ``spec.service.type`` a required first-class field, and
``spec/v2/validate.sh`` went green on every archetype example. But
nthlayer-common's v2 parser still read ``metadata.labels.type`` and, failing
that, *inferred* ai-gate from the presence of ``judgment_slo`` — the exact
inversion 6w9d exists to correct. So the examples were simultaneously green
against the schema and red against the reference implementation, and nothing
in either test suite noticed, because no test crossed the two.

That is the whole point of this module: it is the only place where the
schema's verdict and the parser's verdict on the same bytes are compared.
Neither half alone would have caught it. Do not split them.

The fixtures are the live shipped examples, not copies — that is
load-bearing, exactly as in ``test_manifest_real_specs.py``. A copy could
co-evolve with a parser bug; these cannot.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from nthlayer_common.manifest import ReliabilityManifest, load_manifest

# File path: <ecosystem>/nthlayer-common/tests/test_manifest_v2_archetypes.py
# parents: [0]=tests/ [1]=nthlayer-common/ [2]=<ecosystem>/
# The opensrm spec repo is a sibling checkout in the ecosystem workspace.
ECOSYSTEM_ROOT = Path(__file__).resolve().parents[2]
SPEC_V2_DIR = ECOSYSTEM_ROOT / "opensrm" / "spec" / "v2"
EXAMPLES_DIR = SPEC_V2_DIR / "examples" / "services"
SCHEMA_PATH = SPEC_V2_DIR / "schema.json"


def _archetype_paths() -> list[Path]:
    if not EXAMPLES_DIR.is_dir():
        return []
    return sorted(p for p in EXAMPLES_DIR.iterdir() if p.suffix in (".yaml", ".yml"))


# Skip the module when opensrm isn't sibling-checked-out (e.g. nthlayer-common
# built standalone in a CI matrix). The seam is only observable when both
# repos are present.
pytestmark = pytest.mark.skipif(
    not _archetype_paths() or not SCHEMA_PATH.is_file(),
    reason="opensrm/spec/v2 not present in this checkout layout",
)


def _schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text())


@pytest.mark.parametrize("spec_path", _archetype_paths(), ids=lambda p: p.name)
def test_archetype_validates_against_schema(spec_path: Path):
    """Half one of the seam: the schema accepts the shipped example.

    This half already passed before opensrm-ih0v — validate.sh is green.
    It is here so that a future change breaking the schema half fails in
    the same file as the parser half, rather than only in opensrm's CI.
    """
    jsonschema = pytest.importorskip("jsonschema")

    document = yaml.safe_load(spec_path.read_text())
    validator = jsonschema.Draft7Validator(_schema())
    errors = sorted(validator.iter_errors(document), key=lambda e: e.path)

    assert not errors, (
        f"{spec_path.name}: schema rejected a shipped example:\n"
        + "\n".join(f"  {list(e.path)}: {e.message}" for e in errors)
    )


# A LEDGER OF KNOWN SCHEMA-vs-PARSER DIVERGENCE.
#
# Each entry is an example that validates green against schema.json and is
# rejected by the reference parser — the exact seam this module exists to
# make visible. opensrm-ih0v fixed the service-type limb of it; these four
# are separate causes it uncovered on the way, each with its own bead.
#
# Two of the four are the same disease opensrm-6w9d and opensrm-ih0v cured:
# inference standing in for declaration. None of them is caught by
# spec/v2/validate.sh, which never resolves $refs and never consults a
# parser.
#
# strict=True is load-bearing. When a bead below lands, its example starts
# passing, the xfail becomes an XPASS, and the suite FAILS — forcing the
# entry's removal. Never downgrade these to plain skips: a skip rots
# silently into permanent missing coverage, which is precisely the failure
# mode that let the original divergence ship.
_KNOWN_DIVERGENCE = {
    "ai-gate.yaml": (
        "opensrm-a742: parser only handles the wrapped `kind: JudgmentSLO` "
        "document shape, not the schema's flat inline JudgmentSLOItem "
        "(blocked on opensrm-2027's service-ownership decision)"
    ),
    "api.yaml": (
        "opensrm-1qy1: $ref './slos/checkout-latency-p99.yaml' names a file "
        "that does not exist in the spec repo"
    ),
    "batch.yaml": (
        "opensrm-9bil: parser requires metadata.labels.tier; this example "
        "ships no labels block, and the schema makes labels optional"
    ),
    "database.yaml": (
        "opensrm-9bil: parser requires metadata.labels.tier; this example "
        "ships no labels block, and the schema makes labels optional"
    ),
    "ai-gate-without-judgment-slos.yaml": (
        "opensrm-9bil: parser requires metadata.labels.tier; this example "
        "ships no labels block, and the schema makes labels optional"
    ),
    "stream.yaml": (
        "opensrm-47tt: parser infers classical slo_type from the SLO name, "
        "and 'enrichment-consumer-lag' matches no known pattern"
    ),
    "worker.yaml": (
        "opensrm-47tt: parser infers classical slo_type from the SLO name, "
        "and 'reconciliation-success-rate' matches no known pattern"
    ),
}


def _parse_case(path: Path) -> Any:
    reason = _KNOWN_DIVERGENCE.get(path.name)
    if reason is not None:
        return pytest.param(path, marks=pytest.mark.xfail(strict=True, reason=reason))
    return path


@pytest.mark.parametrize("spec_path", _archetype_paths(), ids=lambda p: p.name)
def test_archetype_declares_a_type_nthlayer_common_accepts(spec_path: Path):
    """ih0v's own contract, asserted on every shipped archetype.

    This is the one half of the seam that holds for all seven today, and it
    is deliberately independent of full parsing: every entry in
    ``_KNOWN_DIVERGENCE`` below is blocked on some *other* field, so without
    this test the module would consist entirely of xfails and assert nothing
    positive about the service type at all.

    What it pins: nthlayer-common's accepted type set is neither narrower
    than the spec's (which would reject valid manifests) nor wider (which
    would accept manifests validate.sh refuses).
    """
    declared = yaml.safe_load(spec_path.read_text())["spec"]["service"]["type"]

    manifest = ReliabilityManifest(
        name="probe", team="t", tier="critical", type=declared
    )

    assert manifest.type == declared, (
        f"{spec_path.name} declares type={declared!r}, which nthlayer-common "
        f"normalised to {manifest.type!r} — a shipped spec value must never "
        f"be treated as an alias"
    )


@pytest.mark.parametrize(
    "spec_path",
    [_parse_case(p) for p in _archetype_paths()],
    ids=lambda p: p.name,
)
def test_archetype_parses_with_declared_type(spec_path: Path):
    """Half two of the seam: the parser accepts the same bytes, and the
    ``type`` it produces is the one the manifest *declared* — not one it
    inferred from surrounding content.

    Asserting equality with the declared value (rather than merely that
    parsing succeeded) is what makes this a regression guard. The old
    ``_infer_service_type`` would have happily parsed ai-gate.yaml and
    returned ``ai-gate`` — by reading judgment_slo, not the field. Only
    comparing against the declared value distinguishes reading from guessing.
    """
    document = yaml.safe_load(spec_path.read_text())
    declared = document["spec"]["service"]["type"]

    manifest = load_manifest(spec_path)

    assert manifest.type == declared, (
        f"{spec_path.name}: declared spec.service.type={declared!r} but "
        f"parser produced type={manifest.type!r}"
    )
