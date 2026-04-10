"""Tests for nthlayer_common.errors module."""


from nthlayer_common.errors import (
    BlockedError,
    ConfigurationError,
    ExitCode,
    NthLayerError,
    PolicyAuditError,
    ProviderError,
    ValidationError,
    WarningResult,
    format_error_message,
    main_with_error_handling,
)


class TestExitCode:
    def test_values(self):
        assert ExitCode.SUCCESS == 0
        assert ExitCode.WARNING == 1
        assert ExitCode.BLOCKED == 2
        assert ExitCode.CONFIG_ERROR == 10
        assert ExitCode.PROVIDER_ERROR == 11
        assert ExitCode.VALIDATION_ERROR == 12
        assert ExitCode.UNKNOWN_ERROR == 127


class TestErrorHierarchy:
    def test_all_inherit_from_nthlayer_error(self):
        assert issubclass(ConfigurationError, NthLayerError)
        assert issubclass(ProviderError, NthLayerError)
        assert issubclass(ValidationError, NthLayerError)
        assert issubclass(BlockedError, NthLayerError)
        assert issubclass(PolicyAuditError, NthLayerError)
        assert issubclass(WarningResult, NthLayerError)

    def test_exit_codes(self):
        assert ConfigurationError.exit_code == ExitCode.CONFIG_ERROR
        assert ProviderError.exit_code == ExitCode.PROVIDER_ERROR
        assert ValidationError.exit_code == ExitCode.VALIDATION_ERROR
        assert BlockedError.exit_code == ExitCode.BLOCKED
        assert PolicyAuditError.exit_code == ExitCode.VALIDATION_ERROR
        assert WarningResult.exit_code == ExitCode.WARNING

    def test_error_message_and_details(self):
        err = ProviderError("grafana down", {"url": "http://grafana:3000"})
        assert err.message == "grafana down"
        assert err.details == {"url": "http://grafana:3000"}

    def test_error_default_details(self):
        err = NthLayerError("bare error")
        assert err.details == {}


class TestFormatErrorMessage:
    def test_without_details(self):
        err = ProviderError("connection failed")
        assert format_error_message(err) == "connection failed"

    def test_with_details(self):
        err = ProviderError("timeout", {"host": "grafana", "timeout_ms": 5000})
        msg = format_error_message(err)
        assert "timeout" in msg
        assert "host=grafana" in msg
        assert "timeout_ms=5000" in msg


class TestMainWithErrorHandling:
    def test_success(self):
        @main_with_error_handling()
        def cmd():
            return 0

        assert cmd() == 0

    def test_nthlayer_error(self):
        @main_with_error_handling()
        def cmd():
            raise ProviderError("fail")

        assert cmd() == ExitCode.PROVIDER_ERROR

    def test_blocked_error(self):
        @main_with_error_handling()
        def cmd():
            raise BlockedError("gate blocked")

        assert cmd() == ExitCode.BLOCKED

    def test_keyboard_interrupt(self):
        @main_with_error_handling()
        def cmd():
            raise KeyboardInterrupt()

        assert cmd() == 130

    def test_unexpected_exception(self):
        @main_with_error_handling()
        def cmd():
            raise RuntimeError("unexpected")

        assert cmd() == ExitCode.UNKNOWN_ERROR
