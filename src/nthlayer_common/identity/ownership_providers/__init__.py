"""
Ownership providers for the NthLayer ecosystem.

Live ownership providers that query external systems (Kubernetes, Backstage, PagerDuty).
Static providers (declared.py, codeowners.py) remain in nthlayer generate.
"""

from nthlayer_common.identity.ownership_providers.backstage import BackstageOwnershipProvider
from nthlayer_common.identity.ownership_providers.base import (
    BaseOwnershipProvider,
    OwnershipProviderHealth,
)
from nthlayer_common.identity.ownership_providers.kubernetes import KubernetesOwnershipProvider
from nthlayer_common.identity.ownership_providers.pagerduty import PagerDutyOwnershipProvider

__all__ = [
    "BaseOwnershipProvider",
    "OwnershipProviderHealth",
    "BackstageOwnershipProvider",
    "KubernetesOwnershipProvider",
    "PagerDutyOwnershipProvider",
]
