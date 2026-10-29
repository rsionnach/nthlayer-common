"""Legacy-format load: environment overlays, template expansion, ${var}
substitution and PagerDuty keys (opensrm-wo62.4.1, rulings D1-D4), plus the
PagerDuty SRE/integration keys and the Observability resource
(opensrm-wo62.4.13, rulings P1-P5).

Every expected value below is derived from nthlayer-generate's legacy
semantics, not from nthlayer-common's implementation:

- overlay lookup order and env-name rules: generate specs/environments.py
  (EnvironmentLoader.find_environment_file, validate_environment_name);
- overlay merge: generate specs/environment_merger.py (merge_service_config:
  the service block is deep-merged; resources are keyed by (kind, name) and
  only their spec is deep-merged; new resources are appended);
- template expansion: generate specs/parser.py (parse_service_file applies
  the overlay FIRST, then _merge_resources: template resources first, a user
  resource with the same (kind, name) REPLACES the template one whole, new
  user resources follow) and custom_templates.py / template_loader.py
  (custom templates are keyed by their `name` field and override built-ins;
  an invalid template file is skipped);
- substitution: generate specs/template.py over ServiceContext.to_dict
  (type as authored, support_model default "self", language/framework ""
  when unset, env only when an environment is given; unknown left as-is);
- PagerDuty SRE/integration keys: generate orchestration/handlers.py
  (PagerDutyHandler.generate reads service.pagerduty.timezone and
  .sre_escalation_policy_id, and the first PagerDuty resource's
  spec.integration_key and spec.sre_integration_key);
- Observability: generate orchestrator.py (ResourceDetector.detect plans a
  dashboard when any `kind: Observability` resource exists, whatever its
  spec).

The fixture tree (tests/fixtures/legacy_loader) is copied into tmp_path
for each test so that the .nthlayer/templates walk-up cannot reach a
directory outside the test.
"""

from __future__ import annotations

import shutil
import warnings
from importlib.resources import files
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from nthlayer_common.manifest import (
    ManifestLoadError,
    Observability,
    PagerDutyConfig,
    load_manifest,
)
from nthlayer_common.manifest.models import SourceFormat
from nthlayer_common.manifest.parser.loader import _find_template_dir

FIXTURES = Path(__file__).parent / "fixtures" / "legacy_loader"

BUILTIN_TEMPLATE_NAMES = {
    "background-job",
    "critical-api",
    "low-api",
    "pipeline",
    "standard-api",
}

# Template query from builtin critical-api.yaml with ${service} = api-minimal.
CRITICAL_API_AVAILABILITY_QUERY = (
    'sum(rate(http_requests_total{service="api-minimal",code!~"5.."}[5m]))\n'
    "/\n"
    'sum(rate(http_requests_total{service="api-minimal"}[5m]))\n'
)

# payment-api.yaml's availability query with ${service} = payment-api.
PAYMENT_API_AVAILABILITY_QUERY = (
    'sum(rate(http_requests{service="payment-api",code!~"5.."}[5m]))\n'
    "/\n"
    'sum(rate(http_requests{service="payment-api"}[5m]))\n'
)


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A private copy of the fixture tree; returns its services/ dir."""
    # A tripwire, not an inventory: the files the lookup-order and template
    # tests rely on. Others (substitution-*, *-pagerduty*) load directly.
    expected = [
        "services/payment-api.yaml",
        "services/payment-api-with-envs.yaml",
        "services/api-minimal.yaml",
        "services/auth-api.yaml",
        "services/email-worker.yaml",
        "services/environments/payment-api-prod.yaml",
        "services/environments/prod.yaml",
        "services/environments/staging.yaml",
        "services/environments/dev.yaml",
        "services/.nthlayer/environments/payment-api-staging.yaml",
        "services/.nthlayer/environments/payment-api-canary.yaml",
        "custom_templates/critical-api.yaml",
    ]
    for rel in expected:
        assert (FIXTURES / rel).is_file(), f"fixture missing: {rel}"
    root = tmp_path / "legacy_loader"
    shutil.copytree(FIXTURES, root)
    services = root / "services"
    # The copies must not pick up a template dir from outside the test.
    assert _find_template_dir(services / "api-minimal.yaml") is None
    return services


def _load(path: Path, environment: str | None = None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_manifest(path, environment=environment)


def _slos(manifest) -> dict:
    return {s.name: s for s in manifest.slos}


def _resource_keys(manifest) -> list[tuple[str, str]]:
    return [(r["kind"], r["name"]) for r in manifest.raw_data["resources"]]


def _resource(manifest, kind: str, name: str) -> dict:
    matches = [r for r in manifest.raw_data["resources"] if (r["kind"], r["name"]) == (kind, name)]
    assert len(matches) == 1
    return matches[0]


# =============================================================================
# D1 — environment overlay: lookup order
# =============================================================================


class TestOverlayLookupOrder:
    def test_step1_service_specific_file_beats_shared(self, tree):
        # environments/payment-api-prod.yaml and environments/prod.yaml both exist.
        m = _load(tree / "payment-api.yaml", "prod")
        metadata = m.raw_data["service"]["metadata"]
        assert metadata == {"criticality": "mission-critical", "compliance": "pci-dss"}
        assert "monitoring_level" not in metadata  # prod.yaml was not read
        assert _slos(m)["availability"].target == 99.99

    def test_step2_shared_file_when_no_service_specific(self, tree):
        # Stem payment-api-with-envs has no {stem}-prod.yaml -> environments/prod.yaml.
        m = _load(tree / "payment-api-with-envs.yaml", "prod")
        assert m.raw_data["service"]["metadata"] == {
            "environment": "prod",
            "monitoring_level": "comprehensive",
        }

    def test_step2_beats_step3(self, tree):
        # environments/staging.yaml (tier standard) and
        # .nthlayer/environments/payment-api-staging.yaml (tier low) both exist.
        m = _load(tree / "payment-api.yaml", "staging")
        assert m.tier == "standard"
        assert _slos(m)["availability"].target == 99.5

    def test_step3_nthlayer_dir_when_nothing_else(self, tree):
        m = _load(tree / "payment-api.yaml", "canary")
        assert m.tier == "standard"
        assert m.raw_data["service"]["metadata"]["testing_mode"] is True

    def test_step3_is_only_service_specific(self, tree):
        # .nthlayer/environments has payment-api-staging.yaml only; a stem with
        # no matching file there and no environments/ match gets no overlay.
        (tree / "environments" / "staging.yaml").unlink()
        m = _load(tree / "payment-api-with-envs.yaml", "staging")
        assert m.tier == "critical"

    def test_lookup_uses_lowercased_name(self, tree):
        m = _load(tree / "payment-api.yaml", "PROD")
        assert _slos(m)["availability"].target == 99.99
        # The manifest records the environment as given (legacy stored the raw arg).
        assert m.environment == "PROD"

    def test_missing_overlay_is_a_no_op(self, tree):
        base = _load(tree / "payment-api.yaml")
        m = _load(tree / "payment-api.yaml", "qa")
        assert m.environment == "qa"
        assert m.tier == base.tier == "critical"
        assert m.raw_data == base.raw_data

    def test_no_environment_reads_no_overlay(self, tree):
        m = _load(tree / "payment-api.yaml")
        assert m.environment is None
        assert _slos(m)["availability"].target == 99.95

    def test_empty_environment_is_treated_as_none(self, tree):
        # Legacy guarded with `if environment:`, so "" neither validates nor looks up.
        m = _load(tree / "payment-api.yaml", "")
        assert _slos(m)["availability"].target == 99.95


# =============================================================================
# D1 — environment overlay: merge
# =============================================================================


class TestOverlayMerge:
    def test_service_block_is_deep_merged(self, tree):
        m = _load(tree / "payment-api.yaml", "canary")
        service = m.raw_data["service"]
        assert service["tier"] == "standard"  # overridden
        assert service["language"] == "java"  # kept from base
        assert service["framework"] == "spring-boot"
        assert m.language == "java"

    def test_resource_spec_is_deep_merged(self, tree):
        m = _load(tree / "payment-api.yaml", "prod")
        spec = _resource(m, "SLO", "availability")["spec"]
        assert spec["objective"] == 99.99
        assert spec["window"] == "30d"
        # Base indicator keys survive; the overlay's success_rate is added.
        assert spec["indicator"]["type"] == "availability"
        assert set(spec["indicator"]["success_rate"]) == {"total_query", "error_query"}
        assert _slos(m)["availability"].indicator_query == PAYMENT_API_AVAILABILITY_QUERY

    def test_nested_spec_keys_survive_a_partial_override(self, tree):
        m = _load(tree / "payment-api.yaml", "prod")
        spec = _resource(m, "SLO", "latency-p95")["spec"]
        assert spec["objective"] == 99.5
        assert spec["threshold_ms"] == 300
        # The overlay declares a threshold, so the base's indicator.threshold_ms
        # is removed before the merge (R4, opensrm-wo62.4.7); its other nested
        # keys survive.
        assert spec["indicator"] == {"type": "latency", "percentile": 95}
        assert _slos(m)["latency-p95"].threshold_ms == 300.0

    def test_overlay_pagerduty_spec_is_merged(self, tree):
        m = _load(tree / "payment-api.yaml", "prod")
        spec = _resource(m, "PagerDuty", "primary")["spec"]
        assert spec == {
            "escalation_policy": "payment-critical",
            "urgency": "high",
            "auto_create": True,
            "auto_resolve": False,
            "acknowledgement_timeout": "5m",
        }
        assert m.ownership.pagerduty.escalation_policy_id == "payment-critical"
        assert m.ownership.escalation == "payment-critical"

    def test_new_overlay_resources_are_appended_in_order(self, tree):
        m = _load(tree / "payment-api.yaml", "prod")
        assert _resource_keys(m) == [
            ("SLO", "availability"),
            ("SLO", "latency-p95"),
            ("PagerDuty", "primary"),
            ("Dependencies", "upstream"),
            ("SLO", "latency-p99"),
            ("Dependencies", "databases"),
        ]
        assert [s.name for s in m.slos] == ["availability", "latency-p95", "latency-p99"]
        assert _slos(m)["latency-p99"].target == 99.0

    def test_overlay_does_not_mutate_the_fixture(self, tree):
        before = (tree / "payment-api.yaml").read_text()
        _load(tree / "payment-api.yaml", "prod")
        assert (tree / "payment-api.yaml").read_text() == before


# =============================================================================
# D1 — environment overlay: errors
# =============================================================================


class TestOverlayErrors:
    @pytest.mark.parametrize(
        "content",
        [
            "service: [unclosed\n",  # invalid YAML
            "- environment: broken\n",  # top level is a list
            "",  # empty document
            "environment: broken\nservice: standard\n",  # service not a mapping
            "environment: broken\nservice:\n",  # service null
            "environment: broken\nresources:\n  kind: SLO\n",  # resources not a list
            "environment: broken\nresources:\n  - SLO\n",  # resource not a mapping
            # R5 correctness: these raised raw AttributeError/TypeError.
            "resources:\n  - {kind: SLO, name: availability, spec: null}\n",
            "resources:\n  - {kind: SLO, name: availability, spec: [1, 2]}\n",
            "resources:\n  - {kind: [SLO], name: availability, spec: {}}\n",
            "resources:\n  - {kind: SLO, name: {a: 1}, spec: {}}\n",
        ],
        ids=[
            "invalid-yaml",
            "list-top-level",
            "empty",
            "service-scalar",
            "service-null",
            "resources-mapping",
            "resource-scalar",
            "spec-null",
            "spec-list",
            "kind-list",
            "name-mapping",
        ],
    )
    def test_malformed_overlay_names_the_overlay_file(self, tree, content):
        overlay = tree / "environments" / "broken.yaml"
        overlay.write_text(content)
        with pytest.raises(ManifestLoadError) as exc:
            _load(tree / "payment-api.yaml", "broken")
        assert str(overlay) in str(exc.value)

    def test_unhashable_resource_name_in_the_manifest_names_the_manifest(self, tree):
        """R5 correctness: a template merge keyed on a mapping name raised a raw TypeError."""
        manifest = tree / "bad-name.yaml"
        manifest.write_text(
            "service:\n  name: bad-name\n  team: t\n  tier: critical\n  type: api\n"
            "  template: critical-api\n"
            "resources:\n  - {kind: SLO, name: {a: 1}, spec: {}}\n"
        )
        with pytest.raises(ManifestLoadError) as exc:
            _load(manifest)
        assert str(manifest) in str(exc.value)

    @pytest.mark.parametrize(
        "name",
        ["../prod", "prod/eu", "prod\\eu", "-prod", "prod-", "prod_eu", "pr od", "prod.eu"],
    )
    def test_invalid_environment_name_is_rejected(self, tree, name):
        with pytest.raises(ManifestLoadError):
            _load(tree / "payment-api.yaml", name)

    @pytest.mark.parametrize("name", ["p", "prod", "prod-us", "eu-west-1", "  Prod  "])
    def test_valid_environment_names_load(self, tree, name):
        m = _load(tree / "payment-api.yaml", name)
        assert m.name == "payment-api"


# =============================================================================
# D2 — template expansion
# =============================================================================


class TestTemplateExpansion:
    def test_builtin_template_supplies_resources(self, tree):
        m = _load(tree / "api-minimal.yaml")
        slos = _slos(m)
        assert [s.name for s in m.slos] == ["availability", "latency-p95"]
        assert slos["availability"].target == 99.9
        assert slos["availability"].window == "30d"
        assert slos["availability"].indicator_query == CRITICAL_API_AVAILABILITY_QUERY
        assert slos["latency-p95"].target == 99.0
        assert m.ownership.pagerduty.urgency == "high"
        assert m.template == "critical-api"

    def test_user_resource_replaces_template_resource_whole(self, tree):
        # auth-api's latency-p95 has objective + threshold_ms only. Legacy
        # replaces the template resource by (kind, name) without merging,
        # so the template's window and indicator are gone.
        m = _load(tree / "auth-api.yaml")
        assert _resource(m, "SLO", "latency-p95")["spec"] == {
            "threshold_ms": 200,
            "objective": 99.5,
        }
        p95 = _slos(m)["latency-p95"]
        assert p95.target == 99.5
        assert p95.indicator_query is None
        assert _resource_keys(m) == [
            ("SLO", "availability"),
            ("SLO", "latency-p95"),
            ("PagerDuty", "primary"),
            ("SLO", "token-generation-latency"),
            ("Dependencies", "upstream"),
        ]
        assert _slos(m)["availability"].target == 99.9
        assert 'service="auth-api"' in _slos(m)["token-generation-latency"].indicator_query

    def test_user_override_without_objective_drops_the_template_slo(self, tree):
        # email-worker overrides processing-latency-p95 with threshold_ms only.
        m = _load(tree / "email-worker.yaml")
        assert _resource(m, "SLO", "processing-latency-p95")["spec"] == {"threshold_ms": 120000}
        assert [s.name for s in m.slos] == ["success-rate"]
        assert m.ownership.pagerduty.urgency == "low"

    def test_overlay_applies_before_template_expansion(self, tree):
        # payment-api-with-envs has no resources; environments/dev.yaml's
        # resources therefore become user resources and replace the
        # critical-api template resources whole (generate parser.py order).
        m = _load(tree / "payment-api-with-envs.yaml", "dev")
        assert m.tier == "low"
        assert _resource_keys(m) == [
            ("SLO", "availability"),
            ("SLO", "latency-p95"),
            ("PagerDuty", "primary"),
        ]
        slos = _slos(m)
        assert slos["availability"].target == 99.0
        assert slos["availability"].indicator_query is None
        assert slos["latency-p95"].target == 90.0
        assert _resource(m, "PagerDuty", "primary")["spec"] == {"urgency": "low"}

    def test_template_without_overlay(self, tree):
        m = _load(tree / "payment-api-with-envs.yaml")
        assert _slos(m)["availability"].target == 99.9
        assert 'service="payment-api"' in _slos(m)["availability"].indicator_query

    def test_custom_template_overrides_builtin_by_name_field(self, tree):
        custom_dir = tree / ".nthlayer" / "templates"
        custom_dir.mkdir()
        # Installed under a different file name: registration is by `name`.
        shutil.copy(tree.parent / "custom_templates" / "critical-api.yaml", custom_dir / "org.yaml")
        m = _load(tree / "api-minimal.yaml")
        assert [s.name for s in m.slos] == ["availability"]
        assert _slos(m)["availability"].target == 99.95
        assert m.ownership.pagerduty is None

    def test_custom_template_found_by_walking_up(self, tree):
        custom_dir = tree.parent / ".nthlayer" / "templates"
        custom_dir.mkdir(parents=True)
        shutil.copy(tree.parent / "custom_templates" / "critical-api.yaml", custom_dir / "a.yaml")
        m = _load(tree / "api-minimal.yaml")
        assert _slos(m)["availability"].target == 99.95

    def test_custom_template_with_unhashable_kind_is_skipped(self, tree):
        """R5 edge-cases: it raised a raw TypeError; generate skipped bad templates."""
        custom_dir = tree / ".nthlayer" / "templates"
        custom_dir.mkdir()
        (custom_dir / "bad.yaml").write_text(
            "name: bad-kind\ndescription: d\ntier: critical\ntype: api\n"
            "resources:\n  - {kind: [SLO], name: a, spec: {}}\n"
        )
        manifest = tree / "uses-bad.yaml"
        manifest.write_text(
            "service:\n  name: x\n  team: t\n  tier: critical\n  type: api\n  template: bad-kind\n"
        )
        with pytest.raises(ManifestLoadError, match="Unknown template"):
            _load(manifest)

    def test_invalid_custom_template_is_skipped(self, tree):
        custom_dir = tree / ".nthlayer" / "templates"
        custom_dir.mkdir()
        # Missing the required `description` (template_loader.py) -> skipped,
        # so the built-in critical-api stays in force.
        (custom_dir / "bad.yaml").write_text("name: critical-api\ntier: critical\ntype: api\n")
        m = _load(tree / "api-minimal.yaml")
        assert _slos(m)["availability"].target == 99.9

    def test_unknown_template_lists_available_templates(self, tree):
        manifest = tree / "unknown.yaml"
        manifest.write_text(
            (tree / "api-minimal.yaml").read_text().replace("critical-api", "no-such-template")
        )
        with pytest.raises(ManifestLoadError) as exc:
            _load(manifest)
        message = str(exc.value)
        assert "no-such-template" in message
        for name in BUILTIN_TEMPLATE_NAMES:
            assert name in message

    def test_unknown_template_lists_custom_templates_too(self, tree):
        custom_dir = tree / ".nthlayer" / "templates"
        custom_dir.mkdir()
        (custom_dir / "team.yaml").write_text(
            "name: team-api\ndescription: d\ntier: low\ntype: api\nresources: []\n"
        )
        manifest = tree / "unknown.yaml"
        manifest.write_text(
            (tree / "api-minimal.yaml").read_text().replace("critical-api", "no-such-template")
        )
        with pytest.raises(ManifestLoadError) as exc:
            _load(manifest)
        assert "team-api" in str(exc.value)


class TestBuiltinTemplatesShipAsPackageData:
    # Checks the source tree; tests/smoke/test_package_data.py checks the
    # installed wheel. Both, because package-data config can differ.
    def test_all_five_builtin_templates_are_package_resources(self):
        package = files("nthlayer_common.manifest.legacy_templates")
        names = {p.name for p in package.iterdir() if p.name.endswith(".yaml")}
        assert names == {f"{n}.yaml" for n in BUILTIN_TEMPLATE_NAMES}

    @pytest.mark.parametrize("name", sorted(BUILTIN_TEMPLATE_NAMES))
    def test_each_builtin_template_expands(self, tree, name):
        manifest = tree / f"uses-{name}.yaml"
        manifest.write_text(
            f"service:\n  name: svc\n  team: t\n  tier: standard\n  type: api\n  template: {name}\n"
        )
        m = _load(manifest)
        assert m.slos, name
        assert all("${service}" not in (s.indicator_query or "") for s in m.slos)


# =============================================================================
# D3 — ${var} substitution
# =============================================================================


class TestSubstitution:
    def test_every_variable_with_environment(self, tree):
        m = _load(tree / "substitution-demo.yaml", "qa")
        assert _slos(m)["availability"].description == (
            "payment-api|payments|critical|api|shared|java|spring-boot|qa|${unknown}"
        )

    def test_env_left_literal_without_environment(self, tree):
        m = _load(tree / "substitution-demo.yaml")
        assert _slos(m)["availability"].description == (
            "payment-api|payments|critical|api|shared|java|spring-boot|${env}|${unknown}"
        )

    def test_env_is_the_environment_as_given(self, tree):
        m = _load(tree / "substitution-demo.yaml", "QA")
        assert "|QA|" in _slos(m)["availability"].description

    def test_service_block_is_substituted_only_with_an_environment(self, tree):
        """generate parser.py:100-108: only the environment pass reaches the
        service block, and only for ${env}/${service}/${team}. Without an
        environment, generate substituted resource specs alone
        (render_resource_spec), so the service block stays literal."""
        assert _load(tree / "substitution-demo.yaml").description == "${team} own ${service}"
        m = _load(tree / "substitution-demo.yaml", "qa")
        assert m.description == "payments own payment-api"

    def test_env_in_service_name_reaches_every_query(self, tree):
        """R5 edge-cases. generate parser.py:100-108 substitutes ${env} in
        service.name before ServiceContext is built, so render_resource_spec's
        ${service} is the substituted name. A one-pass substitution leaked
        a literal ${env} into every PromQL selector."""
        manifest = tree / "env-named.yaml"
        manifest.write_text(
            "service:\n  name: api-${env}\n  team: t\n  tier: standard\n  type: api\n"
            "  template: standard-api\n"
        )
        m = _load(manifest, "prod")
        assert m.name == "api-prod"
        queries = [s.indicator_query for s in m.slos if s.indicator_query]
        assert queries
        for query in queries:
            assert "${" not in query
            assert 'service="api-prod"' in query

    def test_env_in_template_name_selects_the_template(self, tree):
        """generate's environment pass runs before template lookup."""
        manifest = tree / "env-template.yaml"
        manifest.write_text(
            "service:\n  name: x\n  team: t\n  tier: critical\n  type: api\n"
            "  template: ${env}-api\n"
        )
        m = _load(manifest, "critical")
        assert [s.name for s in m.slos] == ["availability", "latency-p95"]

    def test_type_as_authored_and_defaults(self, tree):
        m = _load(tree / "substitution-defaults.yaml")
        assert m.type == "worker"  # resolved by the model
        assert m.authored_type == "background-job"
        # ${type} is the authored spelling; support_model defaults to self;
        # unset language/framework substitute as "" (ServiceContext.to_dict).
        assert _slos(m)["success-rate"].description == "background-job|self||"

    def test_overlay_content_is_substituted(self, tree):
        m = _load(tree / "payment-api.yaml", "canary")
        assert m.raw_data["service"]["metadata"]["deployment"] == "payment-api-canary"

    def test_tier_after_overlay_is_substituted_in_resource_specs(self, tree):
        # The spec pass runs after the overlay, so ${tier} is the merged
        # tier. In the service block ${tier} stays literal: generate's
        # environment pass knows only env, service and team.
        overlay = tree / "environments" / "qa.yaml"
        overlay.write_text(
            "environment: qa\nservice:\n  tier: low\n  metadata:\n    t: '${tier}'\n"
            "resources:\n  - kind: SLO\n    name: availability\n"
            "    spec:\n      description: '${tier}'\n"
        )
        m = _load(tree / "payment-api.yaml", "qa")
        assert _resource(m, "SLO", "availability")["spec"]["description"] == "low"
        assert m.raw_data["service"]["metadata"]["t"] == "${tier}"

    def test_non_string_values_are_untouched(self, tree):
        m = _load(tree / "substitution-demo.yaml", "qa")
        assert _resource(m, "SLO", "availability")["spec"]["objective"] == 99.95


# =============================================================================
# D4 — PagerDuty keys
# =============================================================================


class TestPagerDutyKeys:
    def test_new_fields_default_to_none(self):
        pd = PagerDutyConfig()
        assert pd.urgency is None
        assert pd.auto_resolve_timeout is None
        assert pd.create_escalation_policy is None

    # These fixtures set no template:, so loading in place cannot walk up
    # into a .nthlayer/templates outside the test (unlike `tree` users).
    def test_urgency_and_auto_resolve_timeout(self):
        m = _load(FIXTURES / "services" / "payment-api-pagerduty-demo.yaml")
        assert m.ownership.pagerduty == PagerDutyConfig(
            service_id=None,
            escalation_policy_id=None,
            urgency="high",
            auto_resolve_timeout=14400,
            create_escalation_policy=None,
        )

    def test_create_escalation_policy(self):
        m = _load(FIXTURES / "services" / "full-pagerduty.yaml")
        pd = m.ownership.pagerduty
        assert pd.escalation_policy_id == "platform-oncall"
        assert pd.urgency == "high"
        assert pd.auto_resolve_timeout == 14400
        assert pd.create_escalation_policy == {
            "name": "auto-created-policy",
            "users": ["user@example.com"],
        }


# =============================================================================
# P1, P2, P4, P5 — PagerDuty SRE/integration keys (opensrm-wo62.4.13)
# =============================================================================


class TestPagerDutySreAndIntegrationKeys:
    """The keys PagerDutyHandler.generate reads."""

    def test_sre_and_integration_fields_default_to_none(self):
        pd = PagerDutyConfig()
        assert pd.timezone is None
        assert pd.sre_escalation_policy_id is None
        assert pd.integration_key is None
        assert pd.sre_integration_key is None

    def test_service_and_resource_keys_are_read(self):
        m = _load(FIXTURES / "services" / "pagerduty-sre-keys.yaml")
        assert m.ownership.pagerduty == PagerDutyConfig(
            service_id=None,
            escalation_policy_id=None,
            urgency="high",
            auto_resolve_timeout=14400,
            create_escalation_policy=None,
            timezone="Europe/Dublin",
            sre_escalation_policy_id="PSRE001",
            integration_key="${PAGERDUTY_INTEGRATION_KEY}",
            sre_integration_key="${PAGERDUTY_SRE_INTEGRATION_KEY}",
        )

    def test_unset_timezone_is_not_defaulted(self):
        # P4: generate applies America/New_York itself; common carries what
        # was authored.
        m = _load(FIXTURES / "services" / "payment-api-pagerduty-demo.yaml")
        assert m.ownership.pagerduty.timezone is None
        assert m.ownership.pagerduty.sre_escalation_policy_id is None

    def test_service_pagerduty_without_resource_builds_nothing(self, tree):
        # P5: generate's detector plans pagerduty only for a PagerDuty
        # resource, so service.pagerduty alone must not create a config.
        path = tree / "pd-service-only.yaml"
        path.write_text(
            "service:\n  name: svc\n  team: t\n  tier: critical\n  type: api\n"
            "  pagerduty:\n    timezone: Europe/Dublin\n"
            "    sre_escalation_policy_id: PSRE001\n"
        )
        m = _load(path)
        assert m.ownership.pagerduty is None

    def test_null_service_pagerduty_reads_as_unset(self, tree):
        path = tree / "pd-null.yaml"
        path.write_text(
            "service:\n  name: svc\n  team: t\n  tier: critical\n  type: api\n"
            "  pagerduty:\n"
            "resources:\n  - kind: PagerDuty\n    name: primary\n"
            "    spec:\n      urgency: high\n"
        )
        pd = _load(path).ownership.pagerduty
        assert pd.urgency == "high"
        assert pd.timezone is None
        assert pd.sre_escalation_policy_id is None

    @pytest.mark.parametrize("value", ["foo", "[a, b]"], ids=["scalar", "list"])
    def test_non_mapping_service_pagerduty_is_ignored_with_warning(self, tree, value):
        # service.pagerduty was never read before wo62.4.13; a malformed one
        # must not turn a load that succeeded into a crash.
        path = tree / "pd-malformed.yaml"
        path.write_text(
            "service:\n  name: svc\n  team: t\n  tier: critical\n  type: api\n"
            f"  pagerduty: {value}\n"
            "resources:\n  - kind: PagerDuty\n    name: primary\n"
            "    spec:\n      urgency: high\n"
        )
        with capture_logs() as logs:
            pd = _load(path).ownership.pagerduty
        assert pd.urgency == "high"
        assert pd.timezone is None
        assert pd.sre_escalation_policy_id is None
        assert [e["event"] for e in logs] == ["legacy_service_pagerduty_not_mapping"]

    def test_template_provided_resource_takes_service_keys(self, tree):
        # The builtin critical-api template supplies the PagerDuty resource;
        # the service-block keys still apply (handler reads them from
        # service_def regardless of where the resource came from).
        path = tree / "pd-templated.yaml"
        path.write_text(
            "service:\n  name: api-minimal\n  team: platform\n  tier: critical\n"
            "  type: api\n  template: critical-api\n"
            "  pagerduty:\n    timezone: Europe/Dublin\n"
        )
        pd = _load(path).ownership.pagerduty
        assert pd.urgency == "high"  # from the template's resource
        assert pd.timezone == "Europe/Dublin"

    def test_overlay_service_pagerduty_is_merged(self, tree):
        # The overlay deep-merges the service block (generate
        # merge_service_config), so an env can change the timezone and keep
        # the base's SRE policy.
        overlay = tree / "environments" / "pagerduty-sre-keys-qa.yaml"
        overlay.write_text(
            "environment: qa\nservice:\n  pagerduty:\n    timezone: UTC\n"
            "resources:\n  - kind: PagerDuty\n    name: primary\n"
            "    spec:\n      integration_key: qa-key\n"
        )
        pd = _load(tree / "pagerduty-sre-keys.yaml", "qa").ownership.pagerduty
        assert pd.timezone == "UTC"
        assert pd.sre_escalation_policy_id == "PSRE001"
        assert pd.integration_key == "qa-key"
        assert pd.sre_integration_key == "${PAGERDUTY_SRE_INTEGRATION_KEY}"


# =============================================================================
# P3 — Observability resource (opensrm-wo62.4.13)
# =============================================================================


def _with_observability(tree: Path, resource_yaml: str) -> Path:
    path = tree / "obs.yaml"
    path.write_text(
        "service:\n  name: obs\n  team: t\n  tier: high\n  type: worker\n"
        "resources:\n" + resource_yaml
    )
    return path


class TestObservabilityResource:
    def test_observability_only_fixture_carries_presence(self):
        # The legacy spec keys (metrics, logs, health) have no Observability
        # field, so presence is all that carries; that is what the legacy
        # dashboard detection keys on.
        m = _load(FIXTURES / "services" / "notification-observability-only.yaml")
        assert m.observability == Observability()

    def test_no_observability_resource_is_none(self):
        m = _load(FIXTURES / "services" / "payment-api-pagerduty-demo.yaml")
        assert m.observability is None

    def test_shared_keys_are_mapped(self, tree):
        path = _with_observability(
            tree,
            "  - kind: Observability\n    name: observability\n    spec:\n"
            "      metrics_prefix: obs\n      prometheus_job: obs-job\n"
            "      labels:\n        env: prod\n",
        )
        assert _load(path).observability == Observability(
            metrics_prefix="obs", prometheus_job="obs-job", labels={"env": "prod"}
        )

    @pytest.mark.parametrize(
        "spec_yaml",
        ["    spec: {}\n", "    spec:\n", ""],
        ids=["empty", "null", "missing"],
    )
    def test_empty_or_missing_spec_still_carries_presence(self, tree, spec_yaml):
        # P3: the legacy detector plans a dashboard for any Observability
        # resource, so an empty spec must not read as "no observability".
        path = _with_observability(
            tree, "  - kind: Observability\n    name: observability\n" + spec_yaml
        )
        assert _load(path).observability == Observability()

    @pytest.mark.parametrize(
        "spec_yaml", ["    spec: [a, b]\n", "    spec: hello\n"], ids=["list", "scalar"]
    )
    def test_non_mapping_spec_is_presence_with_warning(self, tree, spec_yaml):
        # The legacy detector never reads the spec, so a malformed one still
        # plans a dashboard; common must not crash where it used to load.
        path = _with_observability(
            tree, "  - kind: Observability\n    name: observability\n" + spec_yaml
        )
        with capture_logs() as logs:
            obs = _load(path).observability
        assert obs == Observability()
        assert [e["event"] for e in logs] == ["legacy_observability_spec_not_mapping"]

    def test_first_observability_resource_wins(self, tree):
        path = _with_observability(
            tree,
            "  - kind: Observability\n    name: a\n    spec:\n      metrics_prefix: first\n"
            "  - kind: Observability\n    name: b\n    spec:\n      metrics_prefix: second\n",
        )
        assert _load(path).observability.metrics_prefix == "first"


# =============================================================================
# srm/v1 and v2 are unaffected
# =============================================================================


class TestOtherFormatsUnaffected:
    @pytest.fixture
    def v1(self, tree) -> Path:
        # Same stem as the legacy fixture, so environments/staging.yaml and the
        # .nthlayer overlays would apply if the v1 branch read them.
        path = tree / "payment-api.yaml"
        shutil.copy(FIXTURES / "srm-v1-payment-api.yaml", path)
        return path

    @pytest.fixture
    def v2(self, tree) -> Path:
        path = tree / "payment-api.yaml"
        shutil.copy(FIXTURES / "opensrm-v2-payment-api.yaml", path)
        return path

    def test_v1_ignores_overlay_and_substitution(self, v1):
        m = _load(v1, "staging")
        assert m.source_format == SourceFormat.SRM_V1
        assert m.tier == "critical"
        slo = _slos(m)["availability"]
        assert slo.target == 99.99
        assert '{service="${service}"' in slo.indicator_query

    def test_v1_pagerduty_has_no_legacy_keys(self, v1):
        m = _load(v1)
        assert m.ownership.pagerduty == PagerDutyConfig(
            service_id="PABC123", escalation_policy_id="PEP456"
        )

    def test_v1_pagerduty_ignores_legacy_service_keys(self, tree):
        # P1: the four wo62.4.13 keys are legacy-only; v1 does not define them.
        path = tree / "v1-pd.yaml"
        text = (FIXTURES / "srm-v1-payment-api.yaml").read_text()
        anchor = "      escalation_policy_id: PEP456\n"
        assert text.count(anchor) == 1
        path.write_text(
            text.replace(
                anchor,
                anchor + "      timezone: Europe/Dublin\n"
                "      sre_escalation_policy_id: PSRE001\n"
                "      integration_key: k\n      sre_integration_key: s\n",
            )
        )
        m = _load(path)
        assert m.source_format == SourceFormat.SRM_V1
        assert m.ownership.pagerduty == PagerDutyConfig(
            service_id="PABC123", escalation_policy_id="PEP456"
        )

    @pytest.mark.parametrize("name", ["../prod", "prod_eu"])
    def test_v1_does_not_validate_environment_name(self, v1, name):
        assert _load(v1, name).source_format == SourceFormat.SRM_V1

    def test_v2_ignores_overlay_and_substitution(self, v2):
        m = _load(v2, "staging")
        assert m.source_format == SourceFormat.OPENSRM_V2
        assert m.tier == "critical"
        assert m.description == "Payments for ${team}"

    @pytest.mark.parametrize("name", ["../prod", "prod_eu"])
    def test_v2_does_not_validate_environment_name(self, v2, name):
        assert _load(v2, name).source_format == SourceFormat.OPENSRM_V2


# =============================================================================
# R5 provenance: ported generate behaviour that had no test
# =============================================================================


def _custom_template(tree: Path, body: str) -> Path:
    """Install one custom template named `custom` and a manifest that uses it."""
    custom_dir = tree / ".nthlayer" / "templates"
    custom_dir.mkdir(exist_ok=True)
    (custom_dir / "custom.yaml").write_text(body)
    manifest = tree / "uses-custom.yaml"
    manifest.write_text(
        "service:\n  name: svc\n  team: t\n  tier: critical\n  type: api\n  template: custom\n"
    )
    return manifest


_VALID_RESOURCE = "resources:\n  - {kind: SLO, name: availability, spec: {objective: 99.9}}\n"


class TestCustomTemplateValidation:
    """generate specs/templates.py ServiceTemplate.__post_init__ and Resource
    reject these; generate's loader then skipped the file with a warning, so
    the template is unknown when a manifest asks for it."""

    @pytest.mark.parametrize(
        "header",
        [
            # templates.py:35: tier must be one of TIER_NAMES.
            "name: custom\ndescription: d\ntier: ultra\ntype: api\n",
            # templates.py: type must resolve to a manifest service type.
            "name: custom\ndescription: d\ntier: critical\ntype: spaceship\n",
        ],
        ids=["invalid-tier", "invalid-type"],
    )
    def test_invalid_template_header_is_skipped(self, tree, header):
        manifest = _custom_template(tree, header + _VALID_RESOURCE)
        with pytest.raises(ManifestLoadError, match="Unknown template"):
            _load(manifest)

    def test_template_resource_without_kind_is_skipped(self, tree):
        manifest = _custom_template(
            tree,
            "name: custom\ndescription: d\ntier: critical\ntype: api\n"
            "resources:\n  - {name: availability, spec: {}}\n",
        )
        with pytest.raises(ManifestLoadError, match="Unknown template"):
            _load(manifest)

    def test_template_resources_keep_only_kind_name_and_spec(self, tree):
        """generate's Resource had only kind, name and spec (+ context)."""
        manifest = _custom_template(
            tree,
            "name: custom\ndescription: d\ntier: critical\ntype: api\n"
            "resources:\n  - {kind: SLO, name: availability, spec: {objective: 99.9},"
            " labels: {x: y}}\n",
        )
        (resource,) = _load(manifest).raw_data["resources"]
        assert set(resource) == {"kind", "name", "spec"}

    def test_env_from_a_custom_template_is_the_environment_as_given(self, tree):
        """The environment pass runs before templates expand, so ${env} in a
        template reaches only the per-spec pass (render_resource_spec), whose
        env is the environment exactly as passed."""
        manifest = _custom_template(
            tree,
            "name: custom\ndescription: d\ntier: critical\ntype: api\n"
            "resources:\n  - {kind: SLO, name: availability,"
            " spec: {objective: 99.9, description: 'in ${env}'}}\n",
        )
        assert (
            _resource(_load(manifest, "  Qa "), "SLO", "availability")["spec"]["description"]
            == "in   Qa "
        )


def test_malformed_base_with_an_overlay_names_the_manifest(tree):
    """generate EnvironmentMerger could not merge into these; parse_service_file
    wrapped the failure in a ServiceParseError naming the file."""
    manifest = tree / "bad-base.yaml"
    manifest.write_text(
        "service:\n  name: bad-base\n  team: t\n  tier: critical\n  type: api\n"
        "resources:\n  kind: SLO\n"
    )
    assert (tree / "environments" / "prod.yaml").is_file()  # the shared overlay applies
    with pytest.raises(ManifestLoadError) as exc:
        _load(manifest, "prod")
    assert str(manifest) in str(exc.value)


def test_scalar_base_service_with_an_overlay_names_the_manifest(tree):
    """generate EnvironmentMerger cannot deep-merge into a scalar service block;
    parse_service_file wrapped the failure in a ServiceParseError naming the file."""
    manifest = tree / "scalar-service.yaml"
    manifest.write_text("service: x\nresources: []\n")
    assert (tree / "environments" / "prod.yaml").is_file()
    with pytest.raises(ManifestLoadError) as exc:
        _load(manifest, "prod")
    assert str(manifest) in str(exc.value)


def test_overlay_deep_merges_nested_service_mappings(tree):
    """generate environment_merger.py _merge_dict recurses into nested
    mappings, so an overlay's service.metadata keeps the base's other keys."""
    manifest = tree / "nested.yaml"
    manifest.write_text(
        "service:\n  name: nested\n  team: t\n  tier: critical\n  type: api\n"
        "  metadata:\n    owner: base-owner\n    region: eu\n"
    )
    (tree / "environments" / "nested-qa.yaml").write_text(
        "environment: qa\nservice:\n  metadata:\n    owner: qa-owner\n"
    )
    metadata = _load(manifest, "qa").raw_data["service"]["metadata"]
    assert metadata == {"owner": "qa-owner", "region": "eu"}


def test_spec_substitution_reaches_list_items_but_not_keys(tree):
    """generate specs/template.py:42-44 substitutes values, recursing into
    dict values and list items; dict keys are never substituted."""
    manifest = tree / "lists.yaml"
    manifest.write_text(
        "service:\n  name: lists\n  team: t\n  tier: critical\n  type: api\n"
        "resources:\n  - kind: SLO\n    name: availability\n    spec:\n"
        "      objective: 99.9\n      labels: ['${service}', '${tier}']\n"
        "      '${tier}-note': kept\n"
    )
    spec = _resource(_load(manifest), "SLO", "availability")["spec"]
    assert spec["labels"] == ["lists", "critical"]
    assert spec["${tier}-note"] == "kept"


_HEADER = "name: custom\ndescription: d\ntier: critical\ntype: api\n"


class TestCustomTemplateEdges:
    """More of generate's template loading (custom_templates.py, templates.py,
    parser.py) that the iteration-3 kill check showed untested."""

    def test_only_yaml_files_are_read_and_subdirectories_are_ignored(self, tree):
        # custom_templates.py:50 globs *.yaml, so a .yml file is not a template
        # and a subdirectory is never opened.
        custom_dir = tree / ".nthlayer" / "templates"
        (custom_dir / "nested").mkdir(parents=True)
        (custom_dir / "custom.yml").write_text(_HEADER + _VALID_RESOURCE)
        manifest = tree / "uses-yml.yaml"
        manifest.write_text(
            "service:\n  name: s\n  team: t\n  tier: critical\n  type: api\n  template: custom\n"
        )
        with pytest.raises(ManifestLoadError, match="Unknown template"):
            _load(manifest)

    @pytest.mark.parametrize(
        "resources",
        [
            "resources:\n  - {kind: SLO, spec: {}}\n",  # no name (Resource needs one)
            "resources:\n  - {kind: SLO, name: a, spec: null}\n",  # null spec
            "resources: null\n",  # resources not a list
        ],
        ids=["no-name", "null-spec", "null-resources"],
    )
    def test_malformed_template_resources_are_skipped(self, tree, resources):
        manifest = _custom_template(tree, _HEADER + resources)
        with pytest.raises(ManifestLoadError, match="Unknown template"):
            _load(manifest)

    def test_non_string_template_name_is_skipped(self, tree):
        manifest = _custom_template(
            tree, "name: [custom]\ndescription: d\ntier: critical\ntype: api\n" + _VALID_RESOURCE
        )
        with pytest.raises(ManifestLoadError, match="Unknown template"):
            _load(manifest)

    def test_empty_template_name_is_a_no_op(self, tree):
        # parser.py:141 expands only when service_context.template is truthy.
        manifest = tree / "empty-template.yaml"
        manifest.write_text(
            "service:\n  name: s\n  team: t\n  tier: critical\n  type: api\n  template: ''\n"
            + _VALID_RESOURCE
        )
        assert [s.name for s in _load(manifest).slos] == ["availability"]

    def test_non_string_template_in_the_manifest_is_a_load_error(self, tree):
        manifest = tree / "list-template.yaml"
        manifest.write_text(
            "service:\n  name: s\n  team: t\n  tier: critical\n  type: api\n  template: [a]\n"
        )
        with pytest.raises(ManifestLoadError):
            _load(manifest)


def test_env_pass_substitutes_a_missing_service_name_and_team_as_empty():
    """variable_substitution.py VariableSubstitutor defaults service and team
    to "". Tested on the pass itself: the loader rejects a manifest that still
    lacks service.name, so this is reachable only before the parse."""
    from nthlayer_common.manifest.parser import legacy

    data = {"service": {"tier": "critical"}, "x": "${service}|${team}|${env}"}
    assert legacy.substitute_environment(data, "prod")["x"] == "||prod"


def test_environment_is_normalised_before_the_overlay_lookup(tree):
    """environments.py: the lookup uses the lowercased, stripped name, so the
    result does not depend on a case-insensitive filesystem."""
    lower = _load(tree / "payment-api.yaml", "prod")
    mixed = _load(tree / "payment-api.yaml", "  Prod  ")
    assert _slos(mixed)["availability"].target == _slos(lower)["availability"].target
    assert (
        _slos(lower)["availability"].target
        != _slos(_load(tree / "payment-api.yaml"))["availability"].target
    )
