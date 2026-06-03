"""Tests for nthlayer_common.identity module."""

from nthlayer_common.identity import (
    IdentityResolver,
    NormalizationRule,
    ServiceIdentity,
    extract_from_pattern,
    extract_service_name,
    normalize_service_name,
)


class TestNormalizeServiceName:
    """Tests for normalize_service_name."""

    def test_removes_environment_suffix(self):
        assert normalize_service_name("payment-api-prod") == "payment"

    def test_removes_version_suffix(self):
        assert normalize_service_name("payment-api-v2") == "payment"

    def test_removes_java_package_prefix(self):
        assert normalize_service_name("com.acme.payment-service") == "payment"

    def test_removes_type_suffix(self):
        assert normalize_service_name("payment-service") == "payment"

    def test_removes_type_prefix(self):
        assert normalize_service_name("svc-payment") == "payment"

    def test_lowercases(self):
        assert normalize_service_name("PAYMENT-API") == "payment"

    def test_normalizes_separators(self):
        assert normalize_service_name("payment.api") == "payment"
        assert normalize_service_name("payment_api") == "payment"

    def test_empty_string(self):
        assert normalize_service_name("") == ""

    def test_custom_rules(self):
        rules = [NormalizationRule(pattern=r"-custom$", replacement="", description="test")]
        assert normalize_service_name("payment-custom", rules=rules) == "payment"

    def test_combined_stripping(self):
        # Rules apply single-pass in order: env→version→java→type-suffix→type-prefix
        # -v2 stripped first, then com.acme. stripped, leaving payment-api-prod
        # -prod not re-matched because env rule already ran
        assert normalize_service_name("com.acme.payment-api-prod-v2") == "payment-api-prod"
        # Simpler case where rules don't conflict
        assert normalize_service_name("com.acme.payment-service") == "payment"


class TestExtractFromPattern:
    """Tests for extract_from_pattern."""

    def test_extracts_named_group(self):
        result = extract_from_pattern(
            "component:default/payment-api",
            r"^component:(?P<namespace>[^/]+)/(?P<name>.+)$",
            "name",
        )
        assert result == "payment-api"

    def test_returns_none_on_no_match(self):
        result = extract_from_pattern("no-match", r"^component:(.+)$", "name")
        assert result is None

    def test_returns_none_on_missing_named_group(self):
        result = extract_from_pattern(
            "test",
            r"^(?P<name>.+)$",
            "nonexistent",
        )
        assert result is None

    def test_returns_none_on_empty_string(self):
        result = extract_from_pattern("", r"^(?P<name>.+)$", "name")
        assert result is None


class TestExtractServiceName:
    """Tests for extract_service_name with provider patterns."""

    def test_backstage_pattern(self):
        assert extract_service_name("component:default/payment-api", "backstage") == "payment"

    def test_kubernetes_pattern(self):
        assert extract_service_name("production/payment-api", "kubernetes") == "payment"

    def test_consul_pattern(self):
        assert extract_service_name("dc1.payment-api-prod", "consul") == "payment"

    def test_unknown_provider_normalizes(self):
        assert extract_service_name("payment-api-prod", "unknown") == "payment"

    def test_provider_case_insensitive(self):
        assert extract_service_name("component:default/payment-api", "Backstage") == "payment"


class TestServiceIdentity:
    """Tests for ServiceIdentity."""

    def test_matches_canonical_name(self):
        si = ServiceIdentity(canonical_name="payment")
        assert si.matches("payment")

    def test_matches_alias(self):
        si = ServiceIdentity(canonical_name="payment", aliases={"pay-api"})
        assert si.matches("pay-api")

    def test_matches_external_id(self):
        si = ServiceIdentity(
            canonical_name="payment",
            external_ids={"consul": "pay-api-prod"},
        )
        assert si.matches("pay-api-prod", provider="consul")

    def test_matches_normalized_form(self):
        si = ServiceIdentity(canonical_name="payment")
        assert si.matches("payment-api-prod")

    def test_no_match(self):
        si = ServiceIdentity(canonical_name="payment")
        assert not si.matches("totally-different")

    def test_merge_from(self):
        si1 = ServiceIdentity(canonical_name="payment", confidence=0.8)
        si2 = ServiceIdentity(
            canonical_name="pay-api",
            aliases={"payment-api"},
            external_ids={"consul": "pay"},
            confidence=0.9,
        )
        si1.merge_from(si2)
        assert "payment-api" in si1.aliases
        assert "pay-api" in si1.aliases
        assert si1.external_ids["consul"] == "pay"
        assert si1.confidence == 0.9

    def test_to_dict(self):
        si = ServiceIdentity(canonical_name="payment")
        d = si.to_dict()
        assert d["canonical_name"] == "payment"
        assert "aliases" in d
        assert "created_at" in d


class TestIdentityResolver:
    """Tests for IdentityResolver."""

    def test_exact_match(self):
        resolver = IdentityResolver()
        si = ServiceIdentity(canonical_name="payment")
        resolver.register(si)
        match = resolver.resolve("payment")
        assert match.found
        assert match.match_type == "exact"
        assert match.confidence == 1.0

    def test_alias_match(self):
        resolver = IdentityResolver()
        si = ServiceIdentity(canonical_name="payment", aliases={"pay-api"})
        resolver.register(si)
        match = resolver.resolve("pay-api")
        assert match.found
        assert match.match_type == "alias"

    def test_normalized_match(self):
        resolver = IdentityResolver()
        si = ServiceIdentity(canonical_name="payment")
        resolver.register(si)
        match = resolver.resolve("payment-api-prod")
        assert match.found
        assert match.match_type == "normalized"

    def test_no_match(self):
        resolver = IdentityResolver()
        match = resolver.resolve("unknown-service")
        assert not match.found
        assert match.match_type == "none"
        assert match.confidence == 0.0

    def test_external_id_match(self):
        resolver = IdentityResolver()
        si = ServiceIdentity(
            canonical_name="payment",
            external_ids={"consul": "pay-api-prod"},
        )
        resolver.register(si)
        match = resolver.resolve("pay-api-prod", provider="consul")
        assert match.found
        assert match.match_type == "external_id"

    def test_explicit_mapping(self):
        resolver = IdentityResolver()
        si = ServiceIdentity(canonical_name="payment")
        resolver.register(si)
        resolver.add_mapping("legacy-pay", "payment")
        match = resolver.resolve("legacy-pay")
        assert match.found
        assert match.match_type == "explicit_mapping"

    def test_register_from_discovery(self):
        resolver = IdentityResolver()
        identity = resolver.register_from_discovery("payment-api-prod", "consul")
        assert identity.canonical_name == "payment"
        assert "payment-api-prod" in identity.aliases
        assert identity.external_ids["consul"] == "payment-api-prod"

    def test_register_from_discovery_rejects_empty_name(self):
        resolver = IdentityResolver()
        import pytest
        with pytest.raises(ValueError, match="normalizes to empty string"):
            resolver.register_from_discovery("api", "consul")

    def test_register_from_discovery_clears_cache(self):
        resolver = IdentityResolver()
        # First resolve — miss
        match1 = resolver.resolve("payment-api-prod")
        assert not match1.found
        # Register via discovery
        resolver.register_from_discovery("payment-api-prod", "consul")
        # Second resolve — should hit now (cache cleared)
        match2 = resolver.resolve("payment-api-prod")
        assert match2.found

    def test_merge_existing(self):
        resolver = IdentityResolver()
        si1 = ServiceIdentity(canonical_name="payment")
        resolver.register(si1)
        si2 = ServiceIdentity(canonical_name="payment", aliases={"pay"})
        resolver.register(si2, merge_existing=True)
        assert "pay" in resolver.identities["payment"].aliases

    def test_cache_works(self):
        resolver = IdentityResolver()
        si = ServiceIdentity(canonical_name="payment")
        resolver.register(si)
        match1 = resolver.resolve("payment")
        match2 = resolver.resolve("payment")
        assert match1 is match2  # same cached object

    def test_clear_cache(self):
        resolver = IdentityResolver()
        si = ServiceIdentity(canonical_name="payment")
        resolver.register(si)
        resolver.resolve("payment")
        resolver.clear_cache()
        # After clear, cache is empty — next resolve creates new match
        match = resolver.resolve("payment")
        assert match.found
