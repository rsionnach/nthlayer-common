"""``spec.service.type`` — the v2 service type discriminator (opensrm-ih0v).

opensrm-6w9d made the field required and first-class, with the six-value
enum plus an ``^x-[a-z][a-z0-9-]*$`` extension branch. This module covers
nthlayer-common's side: the parser READS the declared field (it no longer
infers one), the model accepts the extension branch, and v1 upconversion
writes the field rather than a label.
"""
from __future__ import annotations

import pytest

from nthlayer_common.manifest import ReliabilityManifest
from nthlayer_common.manifest.parser.v2 import OpenSRMV2ParseError, parse_opensrm_v2
from nthlayer_common.manifest.v1_compat import convert_v1_to_v2


def _v2_doc(service: dict[str, object], **spec_extra: object) -> dict[str, object]:
    """A minimal valid v2 ServiceManifest, with spec.service swapped in."""
    return {
        "apiVersion": "opensrm.nthlayer.io/v2",
        "kind": "ServiceManifest",
        "metadata": {"name": "svc", "labels": {"tier": "critical"}},
        "spec": {
            "owner": {"group": "group:default/team"},
            "service": service,
            **spec_extra,
        },
    }


# =============================================================================
# Parser reads the declared field
# =============================================================================


def test_parser_reads_declared_type():
    manifest = parse_opensrm_v2(_v2_doc({"name": "svc", "type": "batch"}))
    assert manifest.type == "batch"


def test_parser_rejects_missing_type_naming_the_field_path():
    """The error must name ``spec.service.type``.

    The pre-ih0v message told authors to set ``metadata.labels.type``, which
    is now neither read nor written — following it would not have helped.
    """
    with pytest.raises(OpenSRMV2ParseError) as excinfo:
        parse_opensrm_v2(_v2_doc({"name": "svc"}))

    message = str(excinfo.value)
    assert "spec.service.type" in message
    assert "metadata.labels.type" not in message


def test_parser_ignores_labels_type():
    """``metadata.labels.type`` has no authority post-ih0v.

    Pre-ih0v it was the primary source. A manifest carrying a stale label
    that contradicts the declared field must resolve to the FIELD, not the
    label — otherwise the old inference path survives by the back door.
    """
    document = _v2_doc({"name": "svc", "type": "worker"})
    document["metadata"]["labels"]["type"] = "ai-gate"  # type: ignore[index]

    manifest = parse_opensrm_v2(document)

    assert manifest.type == "worker"


def test_worker_declaring_judgment_slo_is_rejected():
    """The inversion opensrm-6w9d exists to correct, closed from both ends.

    ``_infer_service_type`` returned ai-gate for anything carrying a
    judgment_slo, silently promoting a misconfigured worker into ai-gate-only
    codepaths. Deleting it stops the reclassification — but on its own that
    only converts the manifest from wrongly-accepted-as-ai-gate to
    wrongly-accepted-as-worker: ``get_judgment_slos()`` would still return
    judgment SLOs for a service the spec says cannot have them, which is the
    same harm arriving by a different road.

    schema.json's ServiceManifest.allOf forbids ``judgment_slo`` outright
    whenever ``spec.service.type`` is present and not ``ai-gate``, so the
    parser must reject it too. Anything else is a parser-wider-than-schema
    divergence — precisely what this bead exists to eliminate.
    """
    # Deliberately the wrapped `kind: JudgmentSLO` document shape, not the
    # schema's flat inline one: the flat shape is blocked on opensrm-a742,
    # and this test is about reclassification, not item shape. Using the
    # shape the parser already handles keeps the two seams independent.
    document = _v2_doc(
        {"name": "svc", "type": "worker"},
        judgment_slo=[
            {
                "metadata": {"name": "svc-reversal-rate"},
                "spec": {
                    "service": "svc",
                    "judgment_type": "reversal_rate",
                    "target": {"maximum_reversal_rate": 0.05},
                },
            }
        ],
    )

    with pytest.raises(OpenSRMV2ParseError, match="judgment_slo"):
        parse_opensrm_v2(document)


def test_ai_gate_declaring_judgment_slo_is_accepted():
    """The permitted direction — guards against over-correcting into a rule
    that forbids judgment SLOs everywhere."""
    document = _v2_doc(
        {"name": "svc", "type": "ai-gate"},
        judgment_slo=[
            {
                "metadata": {"name": "svc-reversal-rate"},
                "spec": {
                    "service": "svc",
                    "judgment_type": "reversal_rate",
                    "target": {"maximum_reversal_rate": 0.05},
                },
            }
        ],
    )

    manifest = parse_opensrm_v2(document)

    assert manifest.is_ai_gate()
    assert len(manifest.get_judgment_slos()) == 1


def test_empty_judgment_slo_list_still_rejected_on_worker():
    """``"judgment_slo": false`` in the schema rejects ANY value, so an empty
    list is invalid too — the property must be absent, not merely empty.

    Keying the parser check on truthiness rather than presence would accept
    ``judgment_slo: []`` on a worker and reopen the divergence for the one
    input most likely to be produced by a template or codegen path.
    """
    document = _v2_doc({"name": "svc", "type": "worker"}, judgment_slo=[])

    with pytest.raises(OpenSRMV2ParseError, match="judgment_slo"):
        parse_opensrm_v2(document)


def test_parser_does_not_infer_ai_gate_from_decision_events():
    """The other inference limb, removed with the same cut."""
    document = _v2_doc(
        {"name": "svc", "type": "stream"},
        instrumentation={"required_events": [{"type": "decision.made"}]},
    )

    assert parse_opensrm_v2(document).type == "stream"


# =============================================================================
# The x- extension branch
# =============================================================================


@pytest.mark.parametrize("service_type", ["api", "worker", "stream", "ai-gate", "batch", "database"])
def test_all_six_spec_types_accepted(service_type: str):
    assert ReliabilityManifest(name="s", team="t", tier="critical", type=service_type).type == service_type


@pytest.mark.parametrize(
    "service_type",
    [
        "x-web",
        "x-ml",
        "x-a",
        "x-edge-cache",
        # The spec's pattern is `^x-[a-z][a-z0-9-]*$` — `-` is inside the
        # character class, so a trailing hyphen is permitted. It is ugly and
        # it is legal; nthlayer-common matches the spec rather than its own
        # taste, because a stricter parser would reject manifests that
        # validate.sh accepts, recreating the very seam ih0v closes.
        "x-web-",
    ],
)
def test_extension_types_accepted(service_type: str):
    assert ReliabilityManifest(name="s", team="t", tier="critical", type=service_type).type == service_type


@pytest.mark.parametrize(
    "service_type",
    [
        "x-",  # nothing after the prefix
        "x-1web",  # must start with a letter
        "x-Web",  # uppercase not permitted
        "X-web",  # uppercase prefix
        "xweb",  # missing the hyphen
        "nonsense",
    ],
)
def test_invalid_types_rejected(service_type: str):
    with pytest.raises(ValueError, match="Invalid type"):
        ReliabilityManifest(name="s", team="t", tier="critical", type=service_type)


def test_trailing_newline_rejected_in_extension_type():
    """The two-validator trap opensrm-6w9d pinned, one repo over.

    YAML ``type: |`` followed by ``x-web`` yields ``'x-web\\n'``.
    ``check-jsonschema`` (ECMA-262, strict ``$``) rejects it; Python
    ``jsonschema`` accepts it, because ``re.search`` lets ``$`` match before
    a final newline. nthlayer-common must agree with the STRICT engine —
    validate.sh runs it, so anything laxer means the parser accepts bytes
    the spec's own gate rejects. This is why the check uses ``fullmatch``.
    """
    with pytest.raises(ValueError, match="Invalid type"):
        ReliabilityManifest(name="s", team="t", tier="critical", type="x-web\n")


# =============================================================================
# Aliases: accepted on input, never stored
# =============================================================================


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [("background-job", "worker"), ("pipeline", "batch"), ("web", "x-web")],
)
def test_aliases_normalise_on_construction(alias: str, canonical: str):
    """Aliases are an nthlayer-common input convenience with no spec standing.

    They resolve in ``__post_init__`` before validation, so a manifest never
    STORES an alias — which is what keeps them from leaking into anything
    that round-trips back out to a spec document.
    """
    assert ReliabilityManifest(name="s", team="t", tier="critical", type=alias).type == canonical


# =============================================================================
# v1 upconversion writes the field
# =============================================================================


def _v1_doc(service_type: str | None) -> dict[str, object]:
    spec: dict[str, object] = {"slos": {}}
    if service_type is not None:
        spec["type"] = service_type
    return {
        "apiVersion": "srm/v1",
        "kind": "ServiceReliabilityManifest",
        "metadata": {"name": "svc", "team": "payments", "tier": "critical"},
        "spec": spec,
    }


def test_v1_upconversion_writes_spec_service_type():
    v2 = convert_v1_to_v2(_v1_doc("api"))

    assert v2["spec"]["service"]["type"] == "api"


def test_v1_upconversion_no_longer_writes_labels_type():
    """``metadata.labels.type`` is dead post-ih0v — its only reader was
    ``_infer_service_type``. Continuing to write it would leave a field that
    looks authoritative and is not."""
    v2 = convert_v1_to_v2(_v1_doc("api"))

    assert "type" not in v2["metadata"].get("labels", {})


@pytest.mark.parametrize(
    ("v1_type", "expected"),
    [("background-job", "worker"), ("pipeline", "batch"), ("web", "x-web")],
)
def test_v1_upconversion_normalises_aliases(v1_type: str, expected: str):
    """Upconversion must emit a SCHEMA-valid value.

    SERVICE_TYPE_ALIASES resolves at the dataclass layer, but convert_v1_to_v2() output
    is checked by schema.json — which knows no aliases. Emitting a raw
    ``background-job`` would produce a v2 document that the spec rejects,
    so the alias map has to be applied here too.
    """
    v2 = convert_v1_to_v2(_v1_doc(v1_type))

    assert v2["spec"]["service"]["type"] == expected


@pytest.mark.parametrize(
    "v1_type",
    [
        "",  # falsy but not None — the `is None` guard let this through
        "Frontend",  # uppercase, matches neither the enum nor the x- branch
        "ml",  # a plausible-looking value that is simply not in the enum
        "x-Web",  # right prefix, wrong case for the extension pattern
    ],
)
def test_v1_upconversion_rejects_values_the_schema_would_reject(v1_type: str):
    """Upconversion must not emit a document schema.json refuses.

    Guarding only ``is None`` was not enough: any other invalid value was
    written verbatim. The failure then surfaced far from its cause — for a
    non-empty value it escaped the ValueError that
    nthlayer-generate's migrate_manifest catches around this call, and
    resurfaced later as an uncaught error from ReliabilityManifest's own
    validation.

    Note ``""``: v1_compat used ``is None`` while parser/v2 used a falsy
    check, so the empty string was rejected on one path and accepted on the
    other. Both now use the same predicate.
    """
    with pytest.raises(ValueError, match=r"spec\.type|Invalid type"):
        convert_v1_to_v2(_v1_doc(v1_type))


def test_v1_upconversion_fails_loudly_on_missing_type():
    """v1's ``spec.type`` was optional; ``spec.service.type`` is required.

    A v1 manifest omitting it cannot produce a valid v2 document, so
    upconversion raises rather than inventing a default — a guessed ``api``
    would be indistinguishable from a declared one downstream.
    """
    with pytest.raises(ValueError, match="spec.type"):
        convert_v1_to_v2(_v1_doc(None))
