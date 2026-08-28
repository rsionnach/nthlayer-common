"""Scanning a directory that holds manifests alongside other YAML.

``load_manifest`` raises ``ManifestLoadError`` for any YAML it cannot parse
as a manifest, including files that never claimed to be one. A caller
walking a specs directory therefore has to answer a question the parser
does not: was this file *aiming* to be a manifest and failing, or is it
something else sharing the directory?

Both wrong answers are real bugs, and both have happened:

- Treat everything as a broken manifest, and a "computed over a subset"
  caveat fires on every mixed-directory run. A caveat that always fires
  stops being read.
- Treat nothing as one, and a broken manifest drops out of a financial
  figure or a set of measured SLOs without a word (opensrm-oh27,
  opensrm-3470).

Promoted here from ``nthlayer_workers.learn.retrospective`` once a third
consumer appeared, so the call sites stop reaching into one another
(opensrm-3470).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from nthlayer_common.manifest.parser.v1 import is_srm_v1_format
from nthlayer_common.manifest.parser.v2 import is_opensrm_v2_format

# Suffixes a manifest may carry. Both, always: `.yml` invisibility is the
# same silent-subset failure as an uncounted parse error, reached by file
# extension instead — the manifest is dropped and nothing says so.
MANIFEST_SUFFIXES = (".yaml", ".yml")


def iter_manifest_files(specs_dir: str | Path) -> list[Path]:
    """Every ``.yaml``/``.yml`` file directly under ``specs_dir``, sorted.

    Sorted because callers dedupe on a first-wins basis: unsorted, which
    manifest wins for a service with two files would vary by filesystem.

    Returns empty for a path that is not a directory rather than raising,
    so a caller that wants its own error message for that case keeps the
    decision.
    """
    path = Path(specs_dir)
    if not path.is_dir():
        return []
    # is_file() as well as the suffix: a DIRECTORY named `foo.yaml` would
    # otherwise be yielded, and every caller then treats it as a manifest
    # that failed to load — load_manifest raises IsADirectoryError (an
    # OSError they catch) and foreign_yaml_reason returns None on the same
    # error, meaning 'this was aiming to be a manifest'. Wrong in the noisy
    # direction rather than the silent one, but wrong.
    return sorted(
        p for p in path.iterdir() if p.is_file() and p.suffix in MANIFEST_SUFFIXES
    )


def foreign_yaml_reason(spec_file: str | Path) -> str | None:
    """Why a file that failed to load should NOT count as a broken manifest.

    Returns ``None`` when the file was aiming at being a manifest — so the
    caller counts it — or a short reason when it plainly was not, so the
    caller can drop it and log the reason rather than dropping it silently.

    Evidence of intent is checked in three widening steps: the canonical
    v1/v2 format predicates, so this cannot drift from what the parser
    accepts; then a near miss on those headers, since a typo'd kind or a
    drifted API group is an ordinary way a real manifest breaks; then the
    body, since a bad merge or a mis-indent can strip the header while
    leaving ``spec.service`` (as a *mapping*), ``spec.slos`` or
    ``spec.outcomes`` intact.

    ``spec.service`` as a mapping is what separates a headerless manifest
    from a headerless OpenSLO document, which writes the same key as a
    plain string. Anything unreadable or empty counts too — a syntax error
    or a truncated write inside a specs directory is a deployment error
    either way.

    LIMIT, stated rather than deferred: this recovers intent only while
    evidence of intent survives. A file that has lost both its header and
    its body is indistinguishable in principle from any other headerless
    YAML mapping — both are dicts with no apiVersion, no kind, no
    ``spec.service``. No content heuristic separates those, and adding more
    markers would not change it. Such a file is dropped; the caller should
    record it at debug so an investigator chasing a number that looks wrong
    can pick the trail back up.
    """
    path = Path(spec_file)
    try:
        data = yaml.safe_load(path.read_text())
    except (OSError, ValueError, yaml.YAMLError):
        # ValueError covers UnicodeDecodeError on non-UTF-8 bytes. Too
        # malformed to inspect, so it counts. A read failing here where
        # load_manifest's read succeeded also means the file moved
        # underneath us, which is not evidence it was foreign.
        return None

    if not data:
        # None (zero bytes, whitespace, comments) or an empty container. A
        # truncated or interrupted write is precisely what the count exists
        # for.
        return None

    if not isinstance(data, dict):
        return f"top-level YAML is {type(data).__name__}, not a mapping"

    # "service" at the top level is the legacy pre-apiVersion shape that
    # load_manifest still accepts.
    if is_opensrm_v2_format(data) or is_srm_v1_format(data) or "service" in data:
        return None

    api_version = data.get("apiVersion")
    kind = data.get("kind")
    near_miss = (
        isinstance(api_version, str)
        and (api_version.startswith("srm/") or "opensrm" in api_version)
    ) or (
        isinstance(kind, str)
        and kind.startswith(("ServiceManifest", "ServiceReliabilityManifest"))
    )
    if near_miss:
        return None

    spec = data.get("spec")
    if isinstance(spec, dict) and (
        isinstance(spec.get("service"), dict)
        or isinstance(spec.get("slos"), list)
        or isinstance(spec.get("outcomes"), dict)
    ):
        return None

    return f"no manifest markers (apiVersion={api_version!r}, kind={kind!r})"
