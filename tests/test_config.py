"""Tests for unified configuration loading."""


import pytest

from nthlayer_common.config import Config


@pytest.fixture
def config_yaml(tmp_path):
    """Write a sample nthlayer.yaml and return its path."""
    content = """\
deployment:
  id: production-eu
  signing_key_id: ed25519-2026q2

store:
  path: /var/lib/nthlayer/store.db
  connection_pool_size: 10

llm:
  default_model: anthropic/claude-opus-4-7
  summary_model: anthropic/claude-haiku-4-5

prometheus:
  url: http://prometheus:9090

core:
  host: 0.0.0.0
  port: 8000

workers:
  observe:
    cycle_interval_seconds: 60
  measure:
    cycle_interval_seconds: 60
    self_calibration_interval: 24h

bench:
  team: sre
  polling_interval: 5
"""
    p = tmp_path / "nthlayer.yaml"
    p.write_text(content)
    return p


class TestLoadFromFile:
    def test_load_explicit_path(self, config_yaml):
        config = Config.load(config_yaml)
        assert config.deployment_id == "production-eu"
        assert config.store_path == "/var/lib/nthlayer/store.db"
        assert config.default_model == "anthropic/claude-opus-4-7"
        assert config.prometheus_url == "http://prometheus:9090"

    def test_load_core_section(self, config_yaml):
        config = Config.load(config_yaml)
        assert config.core["host"] == "0.0.0.0"
        assert config.core["port"] == 8000

    def test_load_workers_section(self, config_yaml):
        config = Config.load(config_yaml)
        assert config.workers["observe"]["cycle_interval_seconds"] == 60
        assert config.workers["measure"]["self_calibration_interval"] == "24h"

    def test_load_bench_section(self, config_yaml):
        config = Config.load(config_yaml)
        assert config.bench["team"] == "sre"

    def test_source_is_file_path(self, config_yaml):
        config = Config.load(config_yaml)
        assert str(config_yaml) in config.source


class TestLoadDefaults:
    def test_no_file_returns_defaults(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("NTHLAYER_CONFIG", raising=False)
        config = Config.load()
        assert config.source == "defaults"
        assert config.deployment_id == "default"
        assert config.store_path == "nthlayer.db"
        assert config.default_model == "anthropic/claude-sonnet-4-20250514"
        assert config.prometheus_url is None

    def test_missing_sections_default_to_empty(self, tmp_path):
        p = tmp_path / "minimal.yaml"
        p.write_text("deployment:\n  id: minimal\n")
        config = Config.load(p)
        assert config.deployment_id == "minimal"
        assert config.core == {}
        assert config.workers == {}
        assert config.bench == {}


class TestEnvVarOverrides:
    def test_store_path_override(self, config_yaml, monkeypatch):
        monkeypatch.setenv("NTHLAYER_STORE_PATH", "/tmp/override.db")
        config = Config.load(config_yaml)
        assert config.store_path == "/tmp/override.db"

    def test_model_override(self, config_yaml, monkeypatch):
        monkeypatch.setenv("NTHLAYER_MODEL", "openai/gpt-4o")
        config = Config.load(config_yaml)
        assert config.default_model == "openai/gpt-4o"

    def test_deployment_id_override(self, config_yaml, monkeypatch):
        monkeypatch.setenv("NTHLAYER_DEPLOYMENT_ID", "us-east-staging")
        config = Config.load(config_yaml)
        assert config.deployment_id == "us-east-staging"

    def test_prometheus_url_override(self, config_yaml, monkeypatch):
        monkeypatch.setenv("NTHLAYER_PROMETHEUS_URL", "http://override:9090")
        config = Config.load(config_yaml)
        assert config.prometheus_url == "http://override:9090"

    def test_env_overrides_apply_to_defaults(self, tmp_path, monkeypatch):
        """Env vars work even without a config file."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("NTHLAYER_CONFIG", raising=False)
        monkeypatch.setenv("NTHLAYER_STORE_PATH", "/env/store.db")
        config = Config.load()
        assert config.store_path == "/env/store.db"


class TestEnvVarConfigPath:
    def test_nthlayer_config_env(self, config_yaml, monkeypatch):
        monkeypatch.setenv("NTHLAYER_CONFIG", str(config_yaml))
        config = Config.load()
        assert config.deployment_id == "production-eu"


class TestDotPathAccess:
    def test_top_level(self, config_yaml):
        config = Config.load(config_yaml)
        assert config.get("deployment.id") == "production-eu"

    def test_nested(self, config_yaml):
        config = Config.load(config_yaml)
        assert config.get("workers.observe.cycle_interval_seconds") == 60

    def test_missing_returns_default(self, config_yaml):
        config = Config.load(config_yaml)
        assert config.get("nonexistent.key", "fallback") == "fallback"

    def test_deeply_missing_returns_default(self, config_yaml):
        config = Config.load(config_yaml)
        assert config.get("workers.observe.nonexistent", 42) == 42


class TestFromDict:
    def test_creates_config(self):
        config = Config.from_dict({
            "deployment": {"id": "test"},
            "store": {"path": "/test.db"},
        })
        assert config.deployment_id == "test"
        assert config.store_path == "/test.db"
        assert config.source == "dict"


class TestGetDotpathWithEnvOverrides:
    def test_get_sees_env_override_when_section_missing(self, tmp_path, monkeypatch):
        """get() must return env-overridden values even if section absent from YAML."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("NTHLAYER_CONFIG", raising=False)
        monkeypatch.setenv("NTHLAYER_STORE_PATH", "/env/store.db")
        config = Config.load()
        assert config.get("store.path") == "/env/store.db"

    def test_get_sees_env_override_when_section_present(self, config_yaml, monkeypatch):
        monkeypatch.setenv("NTHLAYER_STORE_PATH", "/override.db")
        config = Config.load(config_yaml)
        assert config.get("store.path") == "/override.db"


class TestNoneValuedSections:
    def test_bare_section_key_treated_as_empty(self, tmp_path):
        """YAML with bare section key (store: with no value) → empty dict."""
        p = tmp_path / "bare.yaml"
        p.write_text("deployment:\n  id: prod\nstore:\n")
        config = Config.load(p)
        assert config.store == {}
        assert config.store_path == "nthlayer.db"

    def test_empty_yaml_loads_as_defaults(self, tmp_path):
        """Empty YAML file (just ---) loads with empty defaults."""
        p = tmp_path / "empty.yaml"
        p.write_text("---\n")
        config = Config.load(p)
        assert config.deployment_id == "default"
        assert config.store_path == "nthlayer.db"


class TestInvalidConfig:
    def test_non_mapping_raises(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("- just\n- a\n- list\n")
        with pytest.raises(ValueError, match="must contain a YAML mapping"):
            Config.load(p)
