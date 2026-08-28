"""Scanning a directory that holds manifests *and other things*.

``load_manifest`` raises ``ManifestLoadError`` for any YAML it cannot parse
as a manifest — including files that never claimed to be one. A caller
walking a specs directory therefore has to answer a question the parser
does not: **was this file aiming to be a manifest and failing, or is it
simply something else sharing the directory?**

Getting that wrong in either direction is a real bug, and both directions
have now happened in nthlayer-workers:

- Count everything, and the "computed over a subset" caveat fires on every
  mixed-directory run. A caveat that always fires stops being read.
- Count nothing, and a broken manifest drops out of a financial figure or a
  set of measured SLOs in silence (opensrm-oh27, opensrm-3470).

``foreign_yaml_reason`` is that question, promoted here from
``nthlayer_workers.learn.retrospective`` so the three call sites that need
it stop reaching into each other (opensrm-3470). ``iter_manifest_files`` comes with it — see MANIFEST_SUFFIXES for why both
suffixes, always.

These tests are ported from the workers suite deliberately: the promotion
must be behaviour-preserving, and porting the contract is how that is
demonstrated rather than asserted.
"""
from __future__ import annotations

import pytest

from nthlayer_common.manifest import foreign_yaml_reason, iter_manifest_files


def _write(tmp_path, name: str, body: str):
    p = tmp_path / name
    p.write_text(body)
    return p


# =============================================================================
# foreign_yaml_reason — "was this aiming to be a manifest?"
# =============================================================================

_AIMING = {
    "v2 manifest": (
        "apiVersion: opensrm.nthlayer.io/v2\nkind: ServiceManifest\n"
        "metadata: {name: s}\nspec:\n  service: {name: s, type: api}\n"
    ),
    "v1 manifest": (
        "apiVersion: srm/v1\nkind: ServiceReliabilityManifest\n"
        "metadata: {name: s, team: t, tier: critical}\nspec: {type: api, slos: {}}\n"
    ),
    "legacy service shape": "service:\n  name: s\n  team: t\n",
    # Header stripped by a bad merge or a mis-indent, body intact. This is the
    # common corruption, and the case the gate's last extension exists for.
    "headerless, spec.service mapping": "spec:\n  service:\n    name: s\n    type: api\n",
    # v1 writes spec.slos as a MAPPING of name -> config. Encoding it as a
    # list here is what let the inert-guard bug hide: the fixture agreed
    # with the broken predicate instead of with the parser.
    "headerless, spec.slos mapping": "spec:\n  slos:\n    availability:\n      target: 99.9\n",
    "headerless, spec.outcomes": "spec:\n  outcomes:\n    decision_value: {}\n",
    # A typo'd kind or a drifted API group is an ordinary way a real manifest
    # breaks: aiming at us and missing still counts.
    "near-miss apiVersion": "apiVersion: opensrm.nthlayer.io/v3\nkind: ServiceManifest\n",
    "near-miss kind": "apiVersion: srm/v9\nkind: ServiceManifestX\n",
    # Unreadable or empty: a syntax error or a truncated write inside a specs
    # directory is a deployment error either way.
    "yaml syntax error": "a: [1,\n",
    "zero bytes": "",
    "comments only": "# nothing here\n",
}

_FOREIGN = {
    "kustomization": "apiVersion: kustomize.config.k8s.io/v1beta1\nkind: Kustomization\nresources: []\n",
    "k8s deployment": "apiVersion: apps/v1\nkind: Deployment\nmetadata: {name: d}\n",
    "prometheus rules": "groups:\n  - name: g\n    rules: []\n",
    # OpenSLO writes spec.service as a STRING; a ServiceManifest writes it as a
    # mapping. That polysemy is the discriminator.
    "openslo": "apiVersion: openslo/v1\nkind: SLO\nspec:\n  service: my-svc\n  objectives: []\n",
    "headerless openslo": "spec:\n  service: my-svc\n  objectives: []\n",
    "top-level list": "- a\n- b\n",
    "top-level scalar": "just-a-string\n",
}


@pytest.mark.parametrize("label", sorted(_AIMING))
def test_files_aiming_to_be_manifests_are_not_foreign(tmp_path, label):
    """Returns None — the caller should COUNT these as broken manifests."""
    path = _write(tmp_path, "x.yaml", _AIMING[label])

    assert foreign_yaml_reason(path) is None, (
        f"{label!r} was treated as foreign YAML; it should count as a "
        f"manifest that failed to load"
    )


@pytest.mark.parametrize("label", sorted(_FOREIGN))
def test_files_that_are_plainly_something_else_are_foreign(tmp_path, label):
    """Returns a reason — the caller should DROP these without counting."""
    path = _write(tmp_path, "x.yaml", _FOREIGN[label])

    reason = foreign_yaml_reason(path)

    assert reason is not None, f"{label!r} should have been recognised as foreign"
    assert reason.strip(), "the reason must be non-empty — it is what gets logged"


def test_a_file_that_vanished_is_not_foreign(tmp_path):
    """A second read failing where load_manifest's first read succeeded means
    the file moved underneath us. That is not evidence it was foreign, so it
    counts."""
    assert foreign_yaml_reason(tmp_path / "never-existed.yaml") is None


def test_non_utf8_bytes_are_not_foreign(tmp_path):
    """Undecodable bytes are too malformed to inspect. A specs directory
    holding one is a deployment error either way, so it counts rather than
    being dismissed."""
    path = tmp_path / "x.yaml"
    path.write_bytes(b"\xff\xfe\x00binary\n")

    assert foreign_yaml_reason(path) is None


# =============================================================================
# iter_manifest_files — both suffixes, deterministically
# =============================================================================


def test_iter_manifest_files_sees_both_yaml_and_yml(tmp_path):
    """``.yml`` invisibility is the same silent-subset failure as an
    uncounted parse error, reached by file extension: the manifest is
    dropped and nothing anywhere says so."""
    for name in ("a.yaml", "b.yml", "c.yaml"):
        _write(tmp_path, name, "spec: {}\n")

    found = {p.name for p in iter_manifest_files(tmp_path)}

    assert found == {"a.yaml", "b.yml", "c.yaml"}


def test_iter_manifest_files_ignores_other_suffixes(tmp_path):
    for name in ("a.yaml", "notes.md", "script.py", "data.json", "noext"):
        _write(tmp_path, name, "x\n")

    assert [p.name for p in iter_manifest_files(tmp_path)] == ["a.yaml"]


def test_iter_manifest_files_is_sorted(tmp_path):
    """Callers dedupe on a first-wins basis, so filesystem ordering would
    make which manifest wins non-deterministic across machines."""
    for name in ("z.yaml", "a.yml", "m.yaml"):
        _write(tmp_path, name, "spec: {}\n")

    names = [p.name for p in iter_manifest_files(tmp_path)]

    assert names == sorted(names)


def test_iter_manifest_files_on_a_missing_directory_is_empty(tmp_path):
    """Callers check is_dir() for their own error messages; this returning
    empty rather than raising keeps that their decision."""
    assert iter_manifest_files(tmp_path / "nope") == []


def test_iter_manifest_files_skips_directories(tmp_path):
    """A DIRECTORY named ``foo.yaml`` is not a manifest file — every caller
    would otherwise count it as a manifest that failed to load."""
    (tmp_path / "subdir.yaml").mkdir()
    (tmp_path / "real.yaml").write_text("spec: {}\n")

    assert [p.name for p in iter_manifest_files(tmp_path)] == ["real.yaml"]


# =============================================================================
# The body-recovery step must fire for REAL manifest shapes
# =============================================================================
#
# opensrm-oh27's fifth and final gate iteration added body recovery: a bad
# merge or a mis-indent can strip the header while leaving the body intact,
# and that is the common corruption. It was landed as "the last extension"
# on the grounds that the remaining leak was unclosable in principle.
#
# It was inert. `spec.slos` was tested with isinstance(..., list), but v1
# writes it as a MAPPING (parser/v1.py does slos_data.items()), and v2 does
# not use `spec.slos` at all — its keys are `spec.slo` and
# `spec.judgment_slo`. So the branch never fired for any real manifest, in
# either format, and a header-stripped manifest was dropped as foreign with
# parse_failures == 0: the exact silent subset this bead family exists to
# remove, inside the guard meant to prevent it.
#
# Found by opensrm-3470's edge-cases pass. Not caught earlier because the
# promotion was verified against the ORIGINAL, which had the same defect —
# faithful copying proves faithfulness, not correctness.


@pytest.mark.parametrize(
    ("label", "body"),
    [
        # v1: spec.slos is a mapping of name -> config
        ("v1 slos mapping", "spec:\n  type: api\n  slos:\n    availability:\n      target: 99.9\n"),
        # v2: a list of OpenSLO documents
        ("v2 slo list", "spec:\n  slo:\n    - apiVersion: openslo/v1\n      kind: SLO\n"),
        # v2: judgment SLOs
        ("v2 judgment_slo", "spec:\n  judgment_slo:\n    - metadata: {name: j}\n"),
        # already covered, kept so a regression in either direction shows here
        ("v2 service mapping", "spec:\n  service:\n    name: s\n    type: api\n"),
        ("v2 outcomes", "spec:\n  outcomes:\n    decision_value: {}\n"),
    ],
)
def test_headerless_manifest_bodies_are_recovered(tmp_path, label, body):
    """A header-stripped manifest must COUNT, not be dropped as foreign."""
    path = tmp_path / "x.yaml"
    path.write_text(body)

    assert foreign_yaml_reason(path) is None, (
        f"{label}: a headerless manifest body was classified as foreign YAML, "
        f"so it would be dropped with parse_failures == 0"
    )


def test_openslo_still_excluded_after_widening_the_body_check(tmp_path):
    """Widening body recovery must not start swallowing OpenSLO documents.

    OpenSLO writes spec.service as a STRING and has spec.objectives; it has
    no spec.slos/slo/judgment_slo. If widening broke this, every OpenSLO file
    in a specs directory would start counting as a broken manifest.
    """
    path = tmp_path / "x.yaml"
    path.write_text("spec:\n  service: my-svc\n  objectives:\n    - target: 0.99\n")

    assert foreign_yaml_reason(path) is not None


@pytest.mark.parametrize("name", ["a.YAML", "b.Yml", "c.YmL"])
def test_iter_manifest_files_is_case_insensitive_on_suffix(tmp_path, name):
    """`.YAML` was excluded — never loaded, never counted, never logged.

    The same silent-subset-by-extension failure MANIFEST_SUFFIXES exists to
    prevent, one case-fold away."""
    (tmp_path / name).write_text("spec: {}\n")

    assert [p.name for p in iter_manifest_files(tmp_path)] == [name]


def test_a_dangling_symlink_is_still_listed(tmp_path):
    """is_file() follows symlinks, so a broken one vanished from the listing.

    Before the is_file() guard it reached load_manifest and raised, so it was
    counted. Silently dropping a stale overlay symlink is a regression the
    guard introduced; excluding directories was the point, not excluding
    everything that is not a readable file.
    """
    (tmp_path / "broken.yaml").symlink_to(tmp_path / "does-not-exist.yaml")

    assert [p.name for p in iter_manifest_files(tmp_path)] == ["broken.yaml"]


@pytest.mark.parametrize(
    ("label", "body"),
    [
        # parser/v1.py:91 — "spec.type is required"
        ("v1 required type", "spec:\n  type: api\n  dependencies:\n    - name: x\n"),
        # parser/v2.py:104 — "spec.owner is required in v2 manifests"
        ("v2 required owner", "spec:\n  owner:\n    group: group:default/t\n  contracts: []\n"),
    ],
)
def test_body_recovery_covers_the_keys_the_parsers_require(tmp_path, label, body):
    """Body recovery checked OPTIONAL keys and omitted the required ones.

    A file carrying the one key its format cannot be parsed without is about
    as strong a claim to being a manifest as a body can make — stronger than
    the optional keys already accepted. Same class as the inert-list bug: a
    predicate reasoning about shapes without checking what the parsers
    actually demand.
    """
    path = tmp_path / "x.yaml"
    path.write_text(body)

    assert foreign_yaml_reason(path) is None, f"{label} was dropped as foreign"


def test_backstage_component_is_not_a_manifest(tmp_path):
    """A Backstage catalog-info.yaml carries `spec.type` and `spec.owner` too.

    Accepting any string `spec.type` as a manifest marker counts every
    Backstage Component in a specs directory as a broken manifest — the
    noisy direction, which is how a coverage caveat stops being read.

    `spec.type` only counts when it is a value OpenSRM would accept:
    `service` is not one, `api` is. Backstage's `spec.owner` is a string
    where v2's is a mapping, so that marker already discriminates.
    """
    path = tmp_path / "catalog-info.yaml"
    path.write_text(
        "apiVersion: backstage.io/v1alpha1\n"
        "kind: Component\n"
        "metadata: {name: svc}\n"
        "spec: {type: service, lifecycle: production, owner: team-a}\n"
    )

    assert foreign_yaml_reason(path) is not None
