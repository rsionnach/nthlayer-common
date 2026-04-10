"""Re-export shim — canonical source is nthlayer_common.clients.mimir.

This shim maintains backward compatibility for existing imports.
MimirRulerProvider is a BaseHTTPClient subclass (standalone HTTP client
with retry + circuit breaker), NOT a Provider protocol implementer.
"""
from nthlayer_common.clients.mimir import (  # noqa: F401
    DEFAULT_USER_AGENT,
    MimirRulerError,
    MimirRulerProvider,
    RulerPushResult,
)

__all__ = [
    "DEFAULT_USER_AGENT",
    "MimirRulerError",
    "MimirRulerProvider",
    "RulerPushResult",
]
