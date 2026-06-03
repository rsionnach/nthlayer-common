"""Tests for nthlayer_common model modules: slo_models, dependency_models, domain_models, gate_models."""

from datetime import timedelta

import pytest

from nthlayer_common.dependency_models import (
    DependencyDirection,
    DependencyGraph,
    DependencyType,
    DiscoveredDependency,
)
from nthlayer_common.domain_models import (
    Finding,
    Run,
    RunStatus,
    Team,
    TeamSource,
)
from nthlayer_common.gate_models import (
    DeploymentGateCheck,
    GatePolicy,
    GateResult,
)
from nthlayer_common.slo_models import (
    ErrorBudget,
    SLOStatus,
    TimeWindow,
)

# --- SLO Models ---

class TestSLOStatus:
    def test_values(self):
        assert SLOStatus.HEALTHY.value == "healthy"
        assert SLOStatus.WARNING.value == "warning"
        assert SLOStatus.CRITICAL.value == "critical"
        assert SLOStatus.EXHAUSTED.value == "exhausted"


class TestTimeWindow:
    def test_to_timedelta_days(self):
        assert TimeWindow(duration="30d").to_timedelta() == timedelta(days=30)

    def test_to_timedelta_hours(self):
        assert TimeWindow(duration="7h").to_timedelta() == timedelta(hours=7)

    def test_to_timedelta_minutes(self):
        assert TimeWindow(duration="15m").to_timedelta() == timedelta(minutes=15)

    def test_to_timedelta_weeks(self):
        assert TimeWindow(duration="2w").to_timedelta() == timedelta(weeks=2)

    def test_invalid_duration(self):
        with pytest.raises(ValueError, match="Invalid duration"):
            TimeWindow(duration="30s").to_timedelta()

    def test_empty_duration(self):
        with pytest.raises(ValueError, match="Invalid duration"):
            TimeWindow(duration="").to_timedelta()

    def test_malformed_duration(self):
        with pytest.raises(ValueError, match="Invalid duration"):
            TimeWindow(duration="30days").to_timedelta()


class TestErrorBudget:
    def test_percent_consumed(self):
        eb = ErrorBudget(
            slo_id="test", service="svc",
            period_start=None, period_end=None,
            total_budget_minutes=100, burned_minutes=25, remaining_minutes=75,
        )
        assert eb.percent_consumed == 25.0
        assert eb.percent_remaining == 75.0

    def test_calculate_status(self):
        def make_eb(burned):
            return ErrorBudget(
                slo_id="t", service="s",
                period_start=None, period_end=None,
                total_budget_minutes=100, burned_minutes=burned, remaining_minutes=100-burned,
            )
        assert make_eb(10).calculate_status() == SLOStatus.HEALTHY
        assert make_eb(55).calculate_status() == SLOStatus.WARNING
        assert make_eb(85).calculate_status() == SLOStatus.CRITICAL
        assert make_eb(96).calculate_status() == SLOStatus.EXHAUSTED

    def test_zero_total_budget(self):
        eb = ErrorBudget(
            slo_id="t", service="s",
            period_start=None, period_end=None,
            total_budget_minutes=0, burned_minutes=0, remaining_minutes=0,
        )
        assert eb.percent_consumed == 0.0


# --- Dependency Models ---

class TestDependencyType:
    def test_values(self):
        assert DependencyType.SERVICE.value == "service"
        assert DependencyType.DATASTORE.value == "datastore"
        assert DependencyType.INFRASTRUCTURE.value == "infra"

    def test_direction_values(self):
        assert DependencyDirection.UPSTREAM.value == "upstream"
        assert DependencyDirection.DOWNSTREAM.value == "downstream"


class TestDiscoveredDependency:
    def test_to_dict(self):
        d = DiscoveredDependency(
            source_service="a", target_service="b", provider="prometheus",
        )
        result = d.to_dict()
        assert result["source_service"] == "a"
        assert result["provider"] == "prometheus"
        assert result["dep_type"] == "service"


class TestDependencyGraph:
    def test_empty_graph(self):
        g = DependencyGraph()
        assert g.get_service_count() == 0
        assert g.get_edge_count() == 0


# --- Domain Models ---

class TestRunStatus:
    def test_values(self):
        assert RunStatus.queued == "queued"
        assert RunStatus.running == "running"
        assert RunStatus.succeeded == "succeeded"
        assert RunStatus.failed == "failed"


class TestTeam:
    def test_create(self):
        t = Team(id="t1", name="Payments")
        assert t.name == "Payments"
        assert t.managers == []

    def test_with_sources(self):
        t = Team(id="t1", name="Payments", sources=TeamSource(pagerduty_id="PD123"))
        assert t.sources.pagerduty_id == "PD123"


class TestRun:
    def test_defaults(self):
        r = Run(job_id="j1", type="validate")
        assert r.status == RunStatus.queued
        assert r.started_at is None


class TestFinding:
    def test_create(self):
        f = Finding(run_id="r1", entity_ref="svc:payment", action="update")
        assert f.action == "update"
        assert f.outcome is None


# --- Gate Models ---

class TestGateResult:
    def test_values(self):
        assert GateResult.APPROVED == 0
        assert GateResult.WARNING == 1
        assert GateResult.BLOCKED == 2

    def test_is_intenum(self):
        assert int(GateResult.BLOCKED) == 2


class TestGatePolicy:
    def test_defaults(self):
        p = GatePolicy()
        assert p.warning is None
        assert p.blocking is None
        assert p.on_exhausted == []

    def test_from_spec(self):
        spec = {
            "thresholds": {"warning": 20.0, "blocking": 10.0},
            "on_exhausted": ["freeze_deploys"],
        }
        p = GatePolicy.from_spec(spec)
        assert p.warning == 20.0
        assert p.blocking == 10.0
        assert p.on_exhausted == ["freeze_deploys"]


class TestDeploymentGateCheck:
    def test_properties(self):
        check = DeploymentGateCheck(
            service="payment", tier="critical", result=GateResult.APPROVED,
            budget_total_minutes=100, budget_consumed_minutes=10,
            budget_remaining_minutes=90, budget_remaining_percentage=90.0,
            warning_threshold=20.0, blocking_threshold=10.0,
            downstream_services=[], high_criticality_downstream=[],
            message="ok", recommendations=[],
        )
        assert check.is_approved
        assert not check.is_blocked
        assert not check.is_warning
