"""Unified configuration loading for the NthLayer ecosystem.

Loads from a single ``nthlayer.yaml`` with sections for each tier:
``deployment:``, ``store:``, ``llm:``, ``prometheus:``, ``core:``,
``workers:``, ``bench:``. Falls back to per-component YAML files
if ``nthlayer.yaml`` is not found.

Environment variable overrides:
    NTHLAYER_CONFIG       — path to nthlayer.yaml (default: ./nthlayer.yaml)
    NTHLAYER_STORE_PATH   — override store.path
    NTHLAYER_MODEL        — override llm.default_model
    NTHLAYER_DEPLOYMENT_ID — override deployment.id

Canonical import: ``from nthlayer_common.config import Config``

Spec: NTHLAYER-COMMON-v1 §10
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    """Unified NthLayer configuration.

    Access tier-specific config via attributes: config.core, config.workers, config.bench.
    Access shared config via: config.deployment, config.store, config.llm, config.prometheus.
    """

    deployment: dict[str, Any] = field(default_factory=dict)
    store: dict[str, Any] = field(default_factory=dict)
    llm: dict[str, Any] = field(default_factory=dict)
    prometheus: dict[str, Any] = field(default_factory=dict)
    core: dict[str, Any] = field(default_factory=dict)
    workers: dict[str, Any] = field(default_factory=dict)
    bench: dict[str, Any] = field(default_factory=dict)
    _raw: dict[str, Any] = field(default_factory=dict, repr=False)
    _source: str = field(default="defaults", repr=False)

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        """Load configuration with auto-detection.

        Resolution order:
        1. Explicit path argument
        2. NTHLAYER_CONFIG env var
        3. ./nthlayer.yaml (current directory)
        4. Empty defaults (all sections empty dicts)

        After loading from YAML, environment variable overrides are applied.
        """
        config_path = _resolve_config_path(path)

        if config_path and config_path.exists():
            raw = _load_yaml(config_path)
            source = str(config_path)
        else:
            raw = {}
            source = "defaults"

        # Use `or {}` to coalesce None-valued YAML sections (bare keys like `store:`)
        deployment = raw.get("deployment") or {}
        store = raw.get("store") or {}
        llm = raw.get("llm") or {}
        prometheus = raw.get("prometheus") or {}
        core = raw.get("core") or {}
        workers = raw.get("workers") or {}
        bench = raw.get("bench") or {}

        # Ensure _raw references the same dicts as the section attributes
        # so that env overrides via _apply_env_overrides are visible to get()
        raw.setdefault("deployment", deployment)
        raw.setdefault("store", store)
        raw.setdefault("llm", llm)
        raw.setdefault("prometheus", prometheus)
        raw.setdefault("core", core)
        raw.setdefault("workers", workers)
        raw.setdefault("bench", bench)
        # Also fix None-valued keys already present in raw
        for key in ("deployment", "store", "llm", "prometheus", "core", "workers", "bench"):
            if raw[key] is None:
                raw[key] = locals()[key]

        config = cls(
            deployment=raw["deployment"],
            store=raw["store"],
            llm=raw["llm"],
            prometheus=raw["prometheus"],
            core=raw["core"],
            workers=raw["workers"],
            bench=raw["bench"],
            _raw=raw,
            _source=source,
        )

        _apply_env_overrides(config)
        return config

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        """Create config from a dict (for testing)."""
        for key in ("deployment", "store", "llm", "prometheus", "core", "workers", "bench"):
            data.setdefault(key, {})
            if data[key] is None:
                data[key] = {}
        return cls(
            deployment=data["deployment"],
            store=data["store"],
            llm=data["llm"],
            prometheus=data["prometheus"],
            core=data["core"],
            workers=data["workers"],
            bench=data["bench"],
            _raw=data,
            _source="dict",
        )

    def get(self, dotpath: str, default: Any = None) -> Any:
        """Get a value by dot-separated path. E.g., 'store.path' or 'workers.observe.cycle_interval'."""
        keys = dotpath.split(".")
        current: Any = self._raw
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current

    @property
    def source(self) -> str:
        """Where the config was loaded from."""
        return self._source

    @property
    def store_path(self) -> str:
        """Convenience: the SQLite store path."""
        return self.store.get("path", "nthlayer.db")

    @property
    def deployment_id(self) -> str:
        """Convenience: the deployment identifier."""
        return self.deployment.get("id", "default")

    @property
    def default_model(self) -> str:
        """Convenience: the default LLM model."""
        return self.llm.get("default_model", "anthropic/claude-sonnet-4-20250514")

    @property
    def prometheus_url(self) -> str | None:
        """Convenience: the Prometheus URL."""
        return self.prometheus.get("url")


def _resolve_config_path(explicit: str | Path | None) -> Path | None:
    """Resolve the config file path."""
    if explicit is not None:
        return Path(explicit)

    env_path = os.environ.get("NTHLAYER_CONFIG")
    if env_path:
        return Path(env_path)

    default = Path("nthlayer.yaml")
    if default.exists():
        return default

    return None


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load and parse a YAML config file. Empty files return empty dict."""
    with open(path) as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping, got {type(data).__name__}")
    return data


def _apply_env_overrides(config: Config) -> None:
    """Apply environment variable overrides to config."""
    store_path = os.environ.get("NTHLAYER_STORE_PATH")
    if store_path:
        config.store["path"] = store_path

    model = os.environ.get("NTHLAYER_MODEL")
    if model:
        config.llm["default_model"] = model

    deployment_id = os.environ.get("NTHLAYER_DEPLOYMENT_ID")
    if deployment_id:
        config.deployment["id"] = deployment_id

    prometheus_url = os.environ.get("NTHLAYER_PROMETHEUS_URL")
    if prometheus_url:
        config.prometheus["url"] = prometheus_url
