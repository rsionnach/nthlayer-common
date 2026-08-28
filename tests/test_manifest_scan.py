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
    "headerless, spec.slos": "spec:\n  slos:\n    - name: availability\n",
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
