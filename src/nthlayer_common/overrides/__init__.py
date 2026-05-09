"""Human override ingestion — canonical event schema, privacy, and verdict binding.

Models the OTel ``gen_ai.override`` event and binds incoming overrides
to the verdict store. nthlayer-measure consumes overrides via OTel;
this module is what the consumer calls when an event arrives.

Canonical import: ``from nthlayer_common.overrides import OverrideEvent``.
"""

from nthlayer_common.overrides.models import (
    OverrideEvent,
    OverridePrivacyConfig,
    hash_reviewer,
    map_webhook_to_override,
)
from nthlayer_common.overrides.ingestion import apply_override_to_verdict

__all__ = [
    "OverrideEvent",
    "OverridePrivacyConfig",
    "apply_override_to_verdict",
    "hash_reviewer",
    "map_webhook_to_override",
]
