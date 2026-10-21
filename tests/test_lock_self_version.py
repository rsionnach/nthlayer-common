"""uv.lock must record this project's own version [opensrm-g35i]."""
import tomllib
from pathlib import Path


def test_uv_lock_records_this_projects_version():
    """uv.lock's own [[package]] entry must carry pyproject's version.

    release-please bumps pyproject.toml; a toml extra-files updater in
    release-please-config.json bumps this entry in the same release PR. That
    updater only WARNS when its jsonpath matches nothing (a renamed package, a
    lock-format change), so without this test the lock would silently go stale
    again and every `uv sync` after a release would dirty it [opensrm-g35i].
    Release PRs run CI, so a skipped bump fails here, on the release PR.
    """
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    lock = tomllib.loads((root / "uv.lock").read_text())
    entries = [p for p in lock.get("package", []) if p.get("name") == project["name"]]
    assert len(entries) == 1, (
        f"uv.lock has {len(entries)} [[package]] entries named {project['name']!r}"
    )
    assert entries[0].get("version") == project["version"], (
        f"uv.lock records {project['name']} {entries[0].get('version')}, pyproject says "
        f"{project['version']}: run `uv lock` (and check release-please-config.json extra-files)"
    )
