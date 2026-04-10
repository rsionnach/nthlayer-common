"""
Identity resolution for service name normalization.

Canonical source for ServiceIdentity, IdentityResolver, and normalization utilities.
"""

from nthlayer_common.identity.models import IdentityMatch, ServiceIdentity
from nthlayer_common.identity.normalizer import (
    DEFAULT_RULES,
    PROVIDER_PATTERNS,
    NormalizationRule,
    extract_from_pattern,
    extract_service_name,
    normalize_service_name,
)
from nthlayer_common.identity.ownership import (
    DEFAULT_CONFIDENCE,
    OwnershipAttribution,
    OwnershipResolver,
    OwnershipSignal,
    OwnershipSource,
    create_demo_attribution,
)
from nthlayer_common.identity.resolver import IdentityResolver

__all__ = [
    # Models
    "ServiceIdentity",
    "IdentityMatch",
    # Normalizer
    "normalize_service_name",
    "extract_from_pattern",
    "extract_service_name",
    "NormalizationRule",
    "DEFAULT_RULES",
    "PROVIDER_PATTERNS",
    # Resolver
    "IdentityResolver",
    # Ownership
    "OwnershipSource",
    "OwnershipSignal",
    "OwnershipAttribution",
    "OwnershipResolver",
    "DEFAULT_CONFIDENCE",
    "create_demo_attribution",
]
