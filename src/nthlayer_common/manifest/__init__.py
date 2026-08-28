"""
OpenSRM manifest parsing and data models.

Public API:
    from nthlayer_common.manifest import load_manifest, ReliabilityManifest
    from nthlayer_common.manifest.models import SLODefinition, Dependency, ...
"""

from nthlayer_common.manifest.models import (
    JUDGMENT_SLO_TYPES,
    SERVICE_TYPE_ALIASES,
    STANDARD_SLO_TYPES,
    VALID_EXHAUSTION_BEHAVIORS,
    VALID_SERVICE_TYPES,
    VALID_TIERS,
    APIRef,
    AuditConfig,
    BreachAction,
    BreachSemantics,
    BudgetPolicy,
    BudgetThresholds,
    ContractPromise,
    DecisionValue,
    Dependency,
    DependencyCriticality,
    DependencySLO,
    DeploymentConfig,
    DeploymentGates,
    ErrorBudgetGate,
    FailureCost,
    FallbackDeclaration,
    Instrumentation,
    JudgmentMeasurement,
    JudgmentPromise,
    ManifestEscalationStep,
    Observability,
    OnCallConfig,
    Outcomes,
    Override,
    Ownership,
    PagerDutyConfig,
    ProbeConfig,
    RecentIncidentsGate,
    ReliabilityContract,
    ReliabilityManifest,
    RequiredEvent,
    RequiredLog,
    RequiredMetric,
    RequiredTrace,
    RevenueAttribution,
    RollbackConfig,
    RosterMember,
    RotationConfig,
    SamplingConfig,
    SLOComplianceGate,
    SLODefinition,
    SourceFormat,
    StatisticalRequirements,
    StratifiedSample,
    TelemetryEvent,
    VolumeEstimate,
    is_valid_service_type,
    resolve_service_type,
    valid_service_types_phrase,
)
from nthlayer_common.manifest.openslo.parser import OpenSLOParseError
from nthlayer_common.manifest.parser._shared import extract_declared_dependencies
from nthlayer_common.manifest.parser.loader import (
    LegacyFormatWarning,
    ManifestLoadError,
    is_manifest_file,
    load_manifest,
)
from nthlayer_common.manifest.parser.v1 import OpenSRMParseError
from nthlayer_common.manifest.parser.v2 import OpenSRMV2ParseError
from nthlayer_common.manifest.scan import (
    MANIFEST_SUFFIXES,
    foreign_yaml_reason,
    iter_manifest_files,
)

__all__ = [
    # Loader
    "load_manifest",
    "is_manifest_file",
    "extract_declared_dependencies",
    "ManifestLoadError",
    "LegacyFormatWarning",
    "OpenSLOParseError",
    "OpenSRMParseError",
    "OpenSRMV2ParseError",
    # Constants
    "JUDGMENT_SLO_TYPES",
    "STANDARD_SLO_TYPES",
    "SERVICE_TYPE_ALIASES",
    "VALID_EXHAUSTION_BEHAVIORS",
    "VALID_SERVICE_TYPES",
    "VALID_TIERS",
    # Service-type helpers
    "is_valid_service_type",
    # Directory scanning (opensrm-3470)
    "MANIFEST_SUFFIXES",
    "foreign_yaml_reason",
    "iter_manifest_files",
    "resolve_service_type",
    "valid_service_types_phrase",
    # Enums
    "DependencyCriticality",
    "SourceFormat",
    # Manifest
    "ReliabilityManifest",
    # SLO
    "SLODefinition",
    "JudgmentMeasurement",
    "BreachAction",
    "StatisticalRequirements",
    "ProbeConfig",
    "SamplingConfig",
    "StratifiedSample",
    # Contract
    "ReliabilityContract",
    "ContractPromise",
    "JudgmentPromise",
    "APIRef",
    "BreachSemantics",
    # Dependency
    "Dependency",
    "DependencySLO",
    "FallbackDeclaration",
    # Ownership
    "Ownership",
    "PagerDutyConfig",
    "RosterMember",
    "Override",
    "ManifestEscalationStep",
    "RotationConfig",
    "OnCallConfig",
    # Observability
    "Observability",
    # Deployment
    "DeploymentConfig",
    "DeploymentGates",
    "ErrorBudgetGate",
    "SLOComplianceGate",
    "RecentIncidentsGate",
    "BudgetPolicy",
    "BudgetThresholds",
    "RollbackConfig",
    "AuditConfig",
    # Instrumentation
    "Instrumentation",
    "TelemetryEvent",
    "RequiredMetric",
    "RequiredTrace",
    "RequiredLog",
    "RequiredEvent",
    # Outcomes (Missing Capabilities § 1)
    "Outcomes",
    "DecisionValue",
    "FailureCost",
    "RevenueAttribution",
    "VolumeEstimate",
]
