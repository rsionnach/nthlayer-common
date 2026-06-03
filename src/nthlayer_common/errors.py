"""
Unified error handling for the NthLayer ecosystem.

This module provides standardized error handling, exit codes, and
error reporting for all ecosystem components.

Exit Codes:
- 0: Success
- 1: Warning (advisory, operation succeeded with warnings)
- 2: Blocked (operation blocked, e.g., by deployment gate)
- 10: Configuration error
- 11: Provider error (external service failure)
- 12: Validation error
- 127: Unknown/internal error
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import sys
import traceback
from enum import IntEnum
from typing import Any, TypeVar
from collections.abc import Callable

import structlog


class ExitCode(IntEnum):
    """Standardized exit codes for CLI commands."""

    SUCCESS = 0
    WARNING = 1
    BLOCKED = 2
    CONFIG_ERROR = 10
    PROVIDER_ERROR = 11
    VALIDATION_ERROR = 12
    UNKNOWN_ERROR = 127


class NthLayerError(Exception):
    """Base exception for NthLayer errors with exit code support."""

    exit_code: ExitCode = ExitCode.UNKNOWN_ERROR
    show_traceback: bool = False

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(NthLayerError):
    """Raised for configuration-related errors."""

    exit_code = ExitCode.CONFIG_ERROR


class ProviderError(NthLayerError):
    """Raised when an external provider/service fails."""

    exit_code = ExitCode.PROVIDER_ERROR


class ValidationError(NthLayerError):
    """Raised for validation failures."""

    exit_code = ExitCode.VALIDATION_ERROR


class BlockedError(NthLayerError):
    """Raised when an operation is blocked (e.g., by deployment gate)."""

    exit_code = ExitCode.BLOCKED


class PolicyAuditError(NthLayerError):
    """Raised for policy audit failures."""

    exit_code = ExitCode.VALIDATION_ERROR


class WarningResult(NthLayerError):
    """Raised to indicate success with warnings."""

    exit_code = ExitCode.WARNING


# -- Tier-boundary errors (for inter-tier communication) --
# These classify errors by retryability, not by CLI exit code.
# Workers use these to decide: retry, skip, or mark output degraded.


class TransientError(NthLayerError):
    """Retryable error. Core temporarily unavailable, rate limited, or timed out.

    Workers should retry with backoff. Maps from HTTP 502, 503, 504, 429,
    connection refused, and timeout errors.
    """

    exit_code = ExitCode.PROVIDER_ERROR


class PermanentError(NthLayerError):
    """Non-retryable error. Bad input, not found, or authorization failure.

    Workers should log and skip — retrying will not help.
    Maps from HTTP 400, 401, 403, 404, 409, 422.
    """

    exit_code = ExitCode.VALIDATION_ERROR


class DegradedError(NthLayerError):
    """Degraded output. Worker can continue but should mark results as degraded.

    Used when an LLM call fails, calibration data is unavailable, or a
    non-critical dependency is down. The worker produces output but flags
    it as lower confidence.
    """

    exit_code = ExitCode.WARNING


# HTTP status code → error type mapping for CoreAPIClient consumers
_TRANSIENT_HTTP_CODES = frozenset({502, 503, 504, 429})
_PERMANENT_HTTP_CODES = frozenset({400, 401, 403, 404, 409, 422})


def classify_http_error(status_code: int, message: str = "", detail: dict | None = None) -> TransientError | PermanentError:
    """Classify an HTTP error response into TransientError or PermanentError.

    Used by worker modules to convert CoreAPIClient error results into
    typed exceptions for their error handling logic.
    """
    if status_code in _TRANSIENT_HTTP_CODES:
        return TransientError(message, detail)
    if status_code in _PERMANENT_HTTP_CODES:
        return PermanentError(message, detail)
    if status_code >= 500:
        return TransientError(message, detail)  # Unknown 5xx assumed transient
    return PermanentError(message, detail)  # Unknown 4xx assumed permanent


def retry(
    *,
    on: type[Exception] | tuple[type[Exception], ...] = TransientError,
    max_attempts: int = 3,
    initial_backoff: float = 1.0,
    max_backoff: float = 30.0,
    backoff_factor: float = 2.0,
) -> Callable:
    """Decorator for retrying async functions on transient errors.

    Usage::

        @retry(on=TransientError, max_attempts=3)
        async def call_core():
            ...
    """

    def decorator(func: Callable) -> Callable:
        if not inspect.iscoroutinefunction(func):
            raise TypeError(
                f"@retry can only decorate async functions, got {func.__qualname__}"
            )

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            backoff = initial_backoff
            last_error: Exception | None = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except on as e:
                    last_error = e
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(min(backoff, max_backoff))
                        backoff *= backoff_factor
            raise last_error  # type: ignore[misc]

        return wrapper

    return decorator


# Type variable for decorated functions
F = TypeVar("F", bound=Callable[..., int])


def main_with_error_handling(
    *,
    show_traceback: bool = False,
    log_errors: bool = True,
) -> Callable[[F], F]:
    """
    Decorator for CLI main functions that provides unified error handling.

    Catches exceptions and converts them to appropriate exit codes with
    consistent error reporting.

    Args:
        show_traceback: If True, show full traceback for unexpected errors
        log_errors: If True, log errors to structlog

    Usage:
        @main_with_error_handling()
        def my_command() -> int:
            # command implementation
            return 0

    Exit codes:
        - NthLayerError subclasses: Uses the error's exit_code
        - KeyboardInterrupt: Returns 130 (standard for SIGINT)
        - Other exceptions: Returns 127 (unknown error)
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> int:
            try:
                return func(*args, **kwargs)
            except NthLayerError as e:
                if log_errors:
                    structlog.get_logger().error(
                        "command_error",
                        error_type=type(e).__name__,
                        message=e.message,
                        exit_code=e.exit_code,
                        **e.details,
                    )
                if e.show_traceback or show_traceback:
                    traceback.print_exc(file=sys.stderr)
                return e.exit_code
            except KeyboardInterrupt:
                if log_errors:
                    structlog.get_logger().info("command_interrupted")
                return 130  # Standard exit code for SIGINT
            except Exception as e:
                if log_errors:
                    structlog.get_logger().error(
                        "unexpected_error",
                        error_type=type(e).__name__,
                        message=str(e),
                        exit_code=ExitCode.UNKNOWN_ERROR,
                    )
                if show_traceback:
                    traceback.print_exc(file=sys.stderr)
                return ExitCode.UNKNOWN_ERROR

        return wrapper  # type: ignore[return-value]

    return decorator


def format_error_message(error: NthLayerError) -> str:
    """Format an error message for display to users."""
    msg = error.message
    if error.details:
        detail_str = ", ".join(f"{k}={v}" for k, v in error.details.items())
        msg = f"{msg} ({detail_str})"
    return msg
