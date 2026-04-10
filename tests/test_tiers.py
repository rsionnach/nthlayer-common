"""Tests for nthlayer_common.tiers module."""

from nthlayer_common.tiers import (
    TIER_CONFIGS,
    TIER_NAMES,
    VALID_TIERS,
    Tier,
    TierConfig,
    get_slo_targets,
    get_tier_config,
    get_tier_thresholds,
    is_valid_tier,
    normalize_tier,
)


class TestTier:
    def test_enum_values(self):
        assert Tier.CRITICAL == "critical"
        assert Tier.STANDARD == "standard"
        assert Tier.LOW == "low"


class TestTierConfigs:
    def test_three_tiers(self):
        assert len(TIER_CONFIGS) == 3
        assert set(TIER_CONFIGS.keys()) == {"critical", "standard", "low"}

    def test_critical_config(self):
        c = TIER_CONFIGS["critical"]
        assert c.availability_target == 99.95
        assert c.latency_p99_ms == 200
        assert c.error_budget_blocking_pct == 10.0
        assert c.pagerduty_urgency == "high"

    def test_standard_blocking_is_advisory(self):
        assert TIER_CONFIGS["standard"].error_budget_blocking_pct is None

    def test_low_blocking_is_advisory(self):
        assert TIER_CONFIGS["low"].error_budget_blocking_pct is None


class TestNormalizeTier:
    def test_canonical_names(self):
        assert normalize_tier("critical") == "critical"
        assert normalize_tier("standard") == "standard"
        assert normalize_tier("low") == "low"

    def test_aliases(self):
        assert normalize_tier("tier-1") == "critical"
        assert normalize_tier("tier-2") == "standard"
        assert normalize_tier("tier-3") == "low"

    def test_case_insensitive(self):
        assert normalize_tier("CRITICAL") == "critical"
        assert normalize_tier("Tier-1") == "critical"

    def test_invalid_raises(self):
        import pytest
        with pytest.raises(ValueError, match="Invalid tier"):
            normalize_tier("nonexistent")


class TestGetTierConfig:
    def test_returns_config(self):
        config = get_tier_config("critical")
        assert isinstance(config, TierConfig)
        assert config.name == "critical"

    def test_alias_resolves(self):
        config = get_tier_config("tier-1")
        assert config.name == "critical"


class TestIsValidTier:
    def test_valid_canonical(self):
        assert is_valid_tier("critical")
        assert is_valid_tier("standard")
        assert is_valid_tier("low")

    def test_valid_aliases(self):
        assert is_valid_tier("tier-1")
        assert is_valid_tier("tier-2")
        assert is_valid_tier("tier-3")

    def test_invalid(self):
        assert not is_valid_tier("nonexistent")
        assert not is_valid_tier("")


class TestGetTierThresholds:
    def test_critical_has_blocking(self):
        t = get_tier_thresholds("critical")
        assert t["warning"] == 20.0
        assert t["blocking"] == 10.0

    def test_standard_no_blocking(self):
        t = get_tier_thresholds("standard")
        assert t["blocking"] is None


class TestGetSloTargets:
    def test_critical_targets(self):
        t = get_slo_targets("critical")
        assert t["availability"] == 99.95
        assert t["latency_ms"] == 200

    def test_low_targets(self):
        t = get_slo_targets("low")
        assert t["availability"] == 99.5
        assert t["latency_ms"] == 1000


class TestConstants:
    def test_tier_names(self):
        assert TIER_NAMES == ("critical", "standard", "low")

    def test_valid_tiers_includes_aliases(self):
        assert "tier-1" in VALID_TIERS
        assert "critical" in VALID_TIERS
