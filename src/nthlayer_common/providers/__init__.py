"""
Provider infrastructure for the NthLayer ecosystem.

Canonical source for Provider protocol, registry, and all provider implementations.
"""

from nthlayer_common.providers.base import (
    PlanChange,
    PlanResult,
    Provider,
    ProviderHealth,
    ProviderResource,
    ProviderResourceSchema,
)
from nthlayer_common.providers.grafana import (
    GrafanaProvider,
    GrafanaProviderError,
)
from nthlayer_common.providers.lock import (
    DEFAULT_LOCK_PATH,
    ProviderLock,
    load_lock,
    save_lock,
)

# MimirRulerProvider is NOT a Provider protocol implementer — it's a standalone
# HTTP client for the Mimir Ruler API. Import directly from
# nthlayer_common.providers.mimir, not from this package namespace.
from nthlayer_common.providers.pagerduty import (
    PagerDutyProvider,
    PagerDutyProviderError,
)
from nthlayer_common.providers.prometheus import (
    PrometheusProvider,
    PrometheusProviderError,
)
from nthlayer_common.providers.registry import (
    ProviderFactory,
    ProviderRegistry,
    ProviderSpec,
    create_provider,
    list_providers,
    provider_registry,
    register_provider,
)

__all__ = [
    # Base protocols
    "Provider",
    "ProviderResource",
    "ProviderResourceSchema",
    "ProviderHealth",
    "PlanChange",
    "PlanResult",
    # Registry
    "ProviderFactory",
    "ProviderSpec",
    "ProviderRegistry",
    "provider_registry",
    "register_provider",
    "create_provider",
    "list_providers",
    # Lock
    "ProviderLock",
    "DEFAULT_LOCK_PATH",
    "load_lock",
    "save_lock",
    # Prometheus
    "PrometheusProvider",
    "PrometheusProviderError",
    # Grafana
    "GrafanaProvider",
    "GrafanaProviderError",
    # PagerDuty
    "PagerDutyProvider",
    "PagerDutyProviderError",
]
