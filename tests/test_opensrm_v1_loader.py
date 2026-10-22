"""load_manifest rejects opensrm/v1 clearly (opensrm-pnmv)."""
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
    assert "opensrm/v1" in msg and "nthlayer migrate-manifest" in msg
    assert "service.name is required" not in msg


def test_migration_api_is_public():
    for name in ("convert_opensrm_v1_to_v2", "is_opensrm_v1_format", "MigrationResult",
                 "UnmappedField", "OPENSRM_V1_API_VERSION"):
        assert name in manifest.__all__ and hasattr(manifest, name), name
