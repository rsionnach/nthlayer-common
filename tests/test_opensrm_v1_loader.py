"""load_manifest rejects opensrm/v1 clearly (opensrm-pnmv)."""
import re
import shlex
import warnings

import pytest
import yaml

from nthlayer_common import manifest
from nthlayer_common.manifest import LegacyFormatWarning, ManifestLoadError, load_manifest

_DOC = {"apiVersion": "opensrm/v1", "kind": "ServiceReliabilityManifest",
        "metadata": {"name": "my-service", "team": "t", "tier": "standard"},
        "spec": {"type": "api"}}


@pytest.mark.parametrize("fmt", ["auto", "srm_v1", "opensrm_v2", "legacy", None])
def test_opensrm_v1_is_rejected_with_the_migrate_command(tmp_path, fmt):
    path = tmp_path / "svc.yaml"
    path.write_text(yaml.safe_dump(_DOC))
    with warnings.catch_warnings():
        warnings.simplefilter("error", LegacyFormatWarning)   # never reaches the legacy branch
        with pytest.raises(ManifestLoadError) as exc:
            load_manifest(path, format=fmt)
    msg = str(exc.value)
    assert "opensrm/v1" in msg and "nthlayer-generate migrate-manifest" in msg
    # Bare script is ambiguous (opensrm-dww5). The lookbehind keeps this from
    # matching inside "nthlayer-generate migrate-manifest".
    assert re.search(r"(?<![\w-])nthlayer migrate-manifest", msg) is None
    assert "service.name is required" not in msg


def test_migration_api_is_public():
    for name in ("convert_opensrm_v1_to_v2", "is_opensrm_v1_format", "MigrationResult",
                 "UnmappedField", "UnmappedKind", "OPENSRM_V1_API_VERSION",
                 "OPENSRM_V1_REJECTION", "OpenSRMV1MigrationError"):
        assert name in manifest.__all__ and hasattr(manifest, name), name


def test_migration_error_is_a_value_error():
    # Existing `except ValueError` callers keep catching it.
    assert issubclass(manifest.OpenSRMV1MigrationError, ValueError)


def test_exported_rejection_is_the_one_the_loader_raises(tmp_path):
    path = tmp_path / "svc.yaml"
    path.write_text(yaml.safe_dump(_DOC))
    with pytest.raises(ManifestLoadError) as exc:
        load_manifest(path)
    assert str(exc.value) == manifest.OPENSRM_V1_REJECTION.format(path=shlex.quote(str(path)))


def test_is_manifest_file_counts_opensrm_v1_so_scans_reach_the_rejection(tmp_path):
    path = tmp_path / "svc.yaml"          # not *.reliability.yaml, not service.yaml
    path.write_text(yaml.safe_dump(_DOC))
    assert manifest.is_manifest_file(path)
    with pytest.raises(ManifestLoadError, match="nthlayer-generate migrate-manifest"):
        load_manifest(path)


def test_foreign_yaml_reason_does_not_treat_opensrm_v1_as_foreign(tmp_path):
    path = tmp_path / "svc.yaml"
    path.write_text(yaml.safe_dump(_DOC))
    assert manifest.foreign_yaml_reason(path) is None


def test_rejection_command_is_shell_safe_and_ends_the_message(tmp_path):
    d = tmp_path / "my dir"
    d.mkdir()
    path = d / "svc.yaml"
    path.write_text(yaml.safe_dump(_DOC))
    with pytest.raises(ManifestLoadError) as exc:
        load_manifest(path)
    msg = str(exc.value)
    marker = "nthlayer-generate migrate-manifest "
    assert msg.count(marker) == 1
    tail = msg.split(marker)[1]          # nothing may follow the quoted path
    assert shlex.split(tail) == [str(path)]
    assert msg.endswith(tail)
