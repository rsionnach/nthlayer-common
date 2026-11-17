"""An API key mis-set as the model name must not reach errors or telemetry
(opensrm-7um7).

The shapes are the ones the model guard already refuses (llm.py and
llm_structured.py: a model starting ``sk-ant-``, ``sk-``, ``key-`` or
``Bearer ``): Anthropic keys are ``sk-ant-...``, OpenAI's ``sk-...`` and
``sk-proj-...``, ``Bearer `` is an Authorization header value pasted whole.
Every key below is made up.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from nthlayer_common._api_keys import redact_model, scrub_model
from nthlayer_common.llm import LLMError, llm_call
from nthlayer_common.llm_structured import structured_call, structured_call_with_usage
from nthlayer_common.telemetry import capture_events, emit_llm_event

# The part of each made-up key that is secret: it must appear nowhere.
SECRET = "Zq7Xw2Lm9Tn4Vb6Kp1Rs8Yd3"

# (model as mis-set, what may be shown instead)
KEY_SHAPED = [
    (f"sk-ant-api03-{SECRET}", "sk-ant-[REDACTED]"),
    (f"sk-{SECRET}", "sk-[REDACTED]"),
    (f"sk-proj-{SECRET}", "sk-[REDACTED]"),
    (f"key-{SECRET}", "key-[REDACTED]"),
    (f"Bearer {SECRET}", "Bearer [REDACTED]"),
    (f"Bearer sk-ant-api03-{SECRET}", "Bearer [REDACTED]"),
    # Pasted with stray whitespace, as env files and YAML leave it.
    (f"  sk-ant-api03-{SECRET}\n", "sk-ant-[REDACTED]"),
    # A key where the model name goes, after a provider.
    (f"anthropic/sk-ant-api03-{SECRET}", "anthropic/sk-ant-[REDACTED]"),
    (f"openai/sk-{SECRET}", "openai/sk-[REDACTED]"),
    # ...and after a nested provider path, as openai-compatible routers take.
    (f"openrouter/anthropic/sk-ant-api03-{SECRET}", "openrouter/anthropic/sk-ant-[REDACTED]"),
    (f"openai/ sk-{SECRET}", "openai/sk-[REDACTED]"),
]

# What the guard itself refuses: the key is the whole model string.
GUARDED = [
    (f"sk-ant-api03-{SECRET}", "sk-ant-[REDACTED]"),
    (f"sk-proj-{SECRET}", "sk-[REDACTED]"),
    (f"key-{SECRET}", "key-[REDACTED]"),
    (f"Bearer {SECRET}", "Bearer [REDACTED]"),
]

NOT_KEYS = [
    "anthropic/claude-sonnet-4-20250514",
    "openai/gpt-4o",
    "gpt-4o",
    "skylark-2",  # starts "sk", not "sk-"
    "keystone-1",  # starts "key", not "key-"
    "together/meta-llama/Llama-3.3-70B",
    "bearer-of-news",  # "Bearer " needs its space and its capital
    "",
]


class EvalResult(BaseModel):
    score: float


@pytest.fixture
def no_stub(monkeypatch):
    """The canned-LLM stub returns before the guard; switch it off."""
    monkeypatch.delenv("NTHLAYER_LLM_STUB", raising=False)


class TestRedactModel:
    @pytest.mark.parametrize(("model", "shown"), KEY_SHAPED)
    def test_a_key_shaped_model_is_redacted(self, model, shown):
        assert redact_model(model) == shown

    @pytest.mark.parametrize("model", NOT_KEYS)
    def test_a_model_name_is_left_alone(self, model):
        assert redact_model(model) == model

    @pytest.mark.parametrize(("model", "shown"), KEY_SHAPED)
    def test_scrub_replaces_the_key_inside_other_text(self, model, shown):
        text = f"POST /v1/deployments/{model.strip()}/chat failed"
        scrubbed = scrub_model(text, model)
        assert SECRET not in scrubbed
        assert scrubbed.startswith("POST /v1/deployments/")
        assert scrubbed.endswith("/chat failed")

    def test_scrub_passes_a_non_string_through(self):
        # emit_llm_event never raises, whatever a caller hands it as error.
        problem = TimeoutError("timed out")
        assert scrub_model(problem, f"sk-{SECRET}") is problem

    @pytest.mark.parametrize("model", NOT_KEYS)
    def test_scrub_leaves_text_about_a_model_name_alone(self, model):
        text = f"model {model} not found"
        assert scrub_model(text, model) == text


class TestLLMError:
    @pytest.mark.parametrize(("model", "shown"), KEY_SHAPED)
    def test_the_key_is_in_no_part_of_the_error(self, model, shown):
        err = LLMError(f"model {model.strip()} not found", "unknown", model)

        assert err.model == shown
        assert SECRET not in str(err)
        assert SECRET not in repr(err)
        assert not [a for a in err.args if SECRET in str(a)]

    def test_a_model_name_is_kept_as_given(self):
        err = LLMError("timed out", "anthropic", "claude-sonnet-4-20250514")

        assert err.model == "claude-sonnet-4-20250514"
        assert err.provider == "anthropic"
        assert err.args == ("[anthropic/claude-sonnet-4-20250514] timed out",)


class TestGuards:
    """The three guards raise an LLMError that shows no part of the key:
    they used to quote its first 20 characters and carry all of it."""

    @pytest.mark.parametrize(("model", "shown"), GUARDED)
    def test_llm_call(self, no_stub, model, shown):
        with pytest.raises(LLMError) as raised:
            llm_call("sys", "user", model=model)

        assert raised.value.model == shown
        assert SECRET[:6] not in str(raised.value)

    @pytest.mark.parametrize(("model", "shown"), GUARDED)
    def test_structured_call(self, no_stub, model, shown):
        with pytest.raises(LLMError) as raised:
            structured_call("sys", "user", EvalResult, model=model)

        assert raised.value.model == shown
        assert SECRET[:6] not in str(raised.value)

    @pytest.mark.parametrize(("model", "shown"), GUARDED)
    def test_structured_call_with_usage(self, no_stub, model, shown):
        with pytest.raises(LLMError) as raised:
            structured_call_with_usage("sys", "user", EvalResult, model=model)

        assert raised.value.model == shown
        assert SECRET[:6] not in str(raised.value)


class TestEmitLLMEvent:
    @pytest.mark.parametrize(("model", "shown"), KEY_SHAPED)
    def test_the_model_attribute_is_redacted(self, model, shown):
        with capture_events() as cap:
            emit_llm_event(model=model, provider="unknown", caller="correlate.summary")

        (event,) = cap.named("nthlayer.llm.call")
        assert event.attributes["gen_ai.request.model"] == shown

    @pytest.mark.parametrize(("model", "shown"), KEY_SHAPED)
    def test_the_error_attribute_is_scrubbed(self, model, shown):
        # What a caller holding an error from before this fix, or one not
        # raised by this package, would pass.
        error = f"[unknown/{model.strip()}] not a model"
        with capture_events() as cap:
            emit_llm_event(
                model=model, provider="unknown", caller="respond.agent",
                success=False, error=error,
            )

        (event,) = cap.named("nthlayer.llm.call")
        assert SECRET not in event.attributes["nthlayer.llm.error"]
        assert not [v for v in event.attributes.values() if SECRET in str(v)]

    def test_a_model_name_and_its_error_are_emitted_as_given(self):
        with capture_events() as cap:
            emit_llm_event(
                model="claude-sonnet-4-20250514", provider="anthropic",
                caller="measure.evaluator", success=False, error="timed out",
            )

        (event,) = cap.named("nthlayer.llm.call")
        assert event.attributes["gen_ai.request.model"] == "claude-sonnet-4-20250514"
        assert event.attributes["nthlayer.llm.error"] == "timed out"
