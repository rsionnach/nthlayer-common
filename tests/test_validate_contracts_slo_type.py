"""validate_contracts() finds a service's availability SLOs by type
(opensrm-kgec).

It used to look for an SLO named exactly ``availability``. srm/v1 keys its
SLOs by kind, so that name exists there; a v2 SLO is an OpenSLO document
with a DNS-style name such as ``payment-api-availability``, which never
matched, so a v2 manifest's availability SLO was never compared with the
contract it backs.

The v2 manifest is the spec's own migration example (the snapshot in
fixtures/load_shape_matrix): one contract promising 0.9995, and four SLOs
named ``payment-api-<kind>`` with ratio targets.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from nthlayer_common.manifest import load_manifest
from nthlayer_common.manifest.models import (
    ContractPromise,
    ReliabilityContract,
    ReliabilityManifest,
    SLODefinition,
)

EXAMPLE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "load_shape_matrix"
    / "v2-spec-examples"
    / "migration"
    / "api-full-v2.yaml"
)
EXAMPLE = yaml.safe_load(EXAMPLE_PATH.read_text())


def _load(tmp_path: Path, document: dict[str, Any]) -> ReliabilityManifest:
    path = tmp_path / "service.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    return load_manifest(path)


def _slo(document: dict[str, Any], name: str) -> dict[str, Any]:
    (found,) = [s for s in document["spec"]["slo"] if s["metadata"]["name"] == name]
    return found


def test_the_example_is_what_the_cases_assume(tmp_path):
    manifest = _load(tmp_path, EXAMPLE)

    assert manifest.contracts[0].promise.availability == 0.9995
    assert {slo.name: (slo.slo_type, slo.target) for slo in manifest.slos} == {
        "payment-api-availability": ("availability", 99.95),
        "payment-api-latency": ("latency", 99.0),
        "payment-api-error-rate": ("error_rate", 99.9),
        "payment-api-throughput": ("throughput", 99.0),
    }


def test_the_example_as_shipped_keeps_its_promise(tmp_path):
    # Its availability SLO equals the promise. Its latency, error-rate and
    # throughput targets are all below 99.95 and are not availability SLOs:
    # none of them is a breach of an availability promise.
    manifest = _load(tmp_path, EXAMPLE)

    assert manifest.validate_contracts() == []


def test_a_v2_availability_slo_looser_than_the_contract_is_reported(tmp_path):
    document = copy.deepcopy(EXAMPLE)
    _slo(document, "payment-api-availability")["spec"]["objectives"][0]["target"] = 0.999

    manifest = _load(tmp_path, document)
    errors = manifest.validate_contracts()

    # One finding, for the availability SLO, naming it and the contract.
    assert len(errors) == 1
    assert "payment-api-availability" in errors[0]
    assert "'payment-api'" in errors[0]


def test_a_v2_availability_slo_tighter_than_the_contract_is_not(tmp_path):
    document = copy.deepcopy(EXAMPLE)
    _slo(document, "payment-api-availability")["spec"]["objectives"][0]["target"] = 0.9999

    assert _load(tmp_path, document).validate_contracts() == []


def _manifest(slos: list[SLODefinition], promise: ContractPromise) -> ReliabilityManifest:
    return ReliabilityManifest(
        name="svc",
        team="t",
        tier="critical",
        type="ai-gate",
        slos=slos,
        contracts=[ReliabilityContract(name="svc-api", promise=promise)],
    )


def test_the_type_decides_not_the_name():
    # An SLO that happens to be called "availability" but is a latency
    # SLO is not an availability SLO; one called anything else that is
    # one, is.
    manifest = _manifest(
        [
            SLODefinition(name="availability", target=90.0, slo_type="latency"),
            SLODefinition(name="uptime", target=99.0, slo_type="availability"),
        ],
        ContractPromise(availability=0.999),
    )

    errors = manifest.validate_contracts()

    assert len(errors) == 1
    assert "'uptime'" in errors[0]


def test_every_looser_availability_slo_is_reported():
    manifest = _manifest(
        [
            SLODefinition(name="read-availability", target=99.0, slo_type="availability"),
            SLODefinition(name="write-availability", target=99.5, slo_type="availability"),
            SLODefinition(name="admin-availability", target=99.99, slo_type="availability"),
        ],
        ContractPromise(availability=0.999),
    )

    errors = manifest.validate_contracts()

    assert len(errors) == 2
    assert "'read-availability'" in errors[0]
    assert "'write-availability'" in errors[1]


def test_a_judgment_slo_is_not_an_availability_slo():
    # A judgment SLO carries slo_type "availability" because it tracks a
    # rate. Its target answers a judgment promise, not an availability one:
    # 95.0 here would read as far looser than 99.9% availability.
    manifest = _manifest(
        [
            SLODefinition(
                name="reversal_rate",
                target=95.0,
                slo_type="availability",
                judgment_type="reversal_rate",
            )
        ],
        ContractPromise(availability=0.999),
    )

    assert manifest.validate_contracts() == []


def test_an_srm_v1_availability_slo_is_still_found():
    # srm/v1's SLO is named for its kind, and is of that type: found
    # before by name, now by type.
    manifest = _manifest(
        [SLODefinition(name="availability", target=99.5, slo_type="availability")],
        ContractPromise(availability=0.999),
    )

    errors = manifest.validate_contracts()

    assert len(errors) == 1
    assert "'availability'" in errors[0]
