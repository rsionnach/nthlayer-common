"""An API key mis-set as the model name must not reach a provider, an error
or telemetry (opensrm-7um7, opensrm-ua2m).

The shapes start from the prefixes the model guard has always refused
(llm.py and llm_structured.py: ``sk-ant-``, ``sk-``, ``key-`` or
``Bearer ``): Anthropic keys are ``sk-ant-...``, OpenAI's ``sk-...`` and
``sk-proj-...``, ``Bearer `` is an Authorization header value pasted whole.
Every key below is made up.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from nthlayer_common._api_keys import (
    key_refusal,
    looks_like_api_key,
    redact_model,
    scrub_call,
    scrub_model,
)
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


def _must_not_be_called(*args, **kwargs):
    raise AssertionError("a key-shaped model reached a provider call")


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

    def test_scrub_finds_the_secret_percent_encoded_in_a_url(self):
        # Azure puts the model in the URL path, where the space after
        # "Bearer" becomes %20 and the key as written no longer matches.
        text = f"404 for url 'https://x/openai/deployments/Bearer%20{SECRET}/chat'"
        scrubbed = scrub_model(text, f"azure/Bearer {SECRET}")
        assert SECRET not in scrubbed
        assert scrubbed.endswith("/chat'")

    def test_scrub_finds_a_secret_whose_characters_are_percent_encoded(self):
        secret = "Zq7+Xw2/Lm9=Tn4"
        text = "GET https://x/deployments/Zq7%2BXw2%2FLm9%3DTn4 refused"
        assert scrub_model(text, f"key-{secret}") == "GET https://x/deployments/[REDACTED] refused"

    def test_scrub_does_not_rewrite_words_that_share_a_short_tail(self):
        # "sk-ab" has no secret worth the name; "ab" must stay in "about".
        assert scrub_model("task about sk-ab", "sk-ab") == "task about sk-[REDACTED]"

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

    def test_a_key_in_the_provider_slot_is_redacted(self):
        # " <key>/claude" passes the guard (leading space) and is split at
        # the slash, so the key arrives as the provider.
        provider = f" sk-ant-api03-{SECRET}"
        err = LLMError(f"no route to {provider.strip()}", provider, "claude")

        assert err.provider == "sk-ant-[REDACTED]"
        assert err.model == "claude"
        assert err.args == ("[sk-ant-[REDACTED]/claude] no route to sk-ant-[REDACTED]",)

    def test_a_model_name_is_kept_as_given(self):
        err = LLMError("timed out", "anthropic", "claude-sonnet-4-20250514")

        assert err.model == "claude-sonnet-4-20250514"
        assert err.provider == "anthropic"
        assert err.args == ("[anthropic/claude-sonnet-4-20250514] timed out",)


class TestGuards:
    """The three guards refuse exactly what redaction recognises
    (opensrm-ua2m), and the error shows no part of the key: they used to
    quote its first 20 characters and carry all of it."""

    @pytest.mark.parametrize(("model", "shown"), KEY_SHAPED)
    def test_the_guards_test_is_the_redactions(self, model, shown):
        assert looks_like_api_key(model)

    @pytest.mark.parametrize("model", NOT_KEYS)
    def test_a_model_name_passes_the_guards_test(self, model):
        assert not looks_like_api_key(model)

    def test_the_refusal_names_the_shapes_and_quotes_the_redacted_form(self):
        assert key_refusal(f"openai/sk-{SECRET}") == (
            "'openai/sk-[REDACTED]' looks like an API key, not a model name: "
            "a model, or a path segment of one, starting sk-, key- or 'Bearer ' "
            "is refused."
        )

    @pytest.mark.parametrize(("model", "shown"), KEY_SHAPED)
    def test_llm_call(self, no_stub, monkeypatch, model, shown):
        monkeypatch.setattr("nthlayer_common.llm._call_anthropic", _must_not_be_called)
        monkeypatch.setattr("nthlayer_common.llm._call_openai_compat", _must_not_be_called)

        with pytest.raises(LLMError) as raised:
            llm_call("sys", "user", model=model, retry=0)

        assert raised.value.model == shown
        assert raised.value.provider == "unknown"
        assert SECRET[:6] not in str(raised.value)

    @pytest.mark.parametrize(("model", "shown"), KEY_SHAPED)
    def test_structured_call(self, no_stub, monkeypatch, model, shown):
        monkeypatch.setattr(
            "nthlayer_common.llm_structured._get_anthropic_client", _must_not_be_called
        )
        monkeypatch.setattr(
            "nthlayer_common.llm_structured._get_openai_client", _must_not_be_called
        )

        with pytest.raises(LLMError) as raised:
            structured_call("sys", "user", EvalResult, model=model)

        assert raised.value.model == shown
        assert raised.value.provider == "unknown"
        assert SECRET[:6] not in str(raised.value)

    @pytest.mark.parametrize(("model", "shown"), KEY_SHAPED)
    def test_structured_call_with_usage(self, no_stub, monkeypatch, model, shown):
        monkeypatch.setattr(
            "nthlayer_common.llm_structured._get_anthropic_client", _must_not_be_called
        )
        monkeypatch.setattr(
            "nthlayer_common.llm_structured._get_openai_client", _must_not_be_called
        )

        with pytest.raises(LLMError) as raised:
            structured_call_with_usage("sys", "user", EvalResult, model=model)

        assert raised.value.model == shown
        assert raised.value.provider == "unknown"
        assert SECRET[:6] not in str(raised.value)


class TestScrubCall:
    def test_a_key_in_either_slot_is_scrubbed(self):
        key = f"sk-ant-api03-{SECRET}"
        assert scrub_call(f"no route to {key}", f" {key}", "claude") == (
            "no route to sk-ant-[REDACTED]"
        )
        assert scrub_call(f"model {key} not found", "openai", key) == (
            "model sk-ant-[REDACTED] not found"
        )

    def test_text_about_a_model_name_is_left_alone(self):
        text = "anthropic/claude-sonnet-4-20250514 timed out"
        assert scrub_call(text, "anthropic", "claude-sonnet-4-20250514") == text


class TestEmitLLMEvent:
    def test_a_key_in_the_provider_slot_is_redacted(self):
        provider = f" sk-ant-api03-{SECRET}"
        with capture_events() as cap:
            emit_llm_event(
                model="claude", provider=provider, caller="respond.agent",
                success=False, error=f"[{provider}/claude] refused",
            )

        (event,) = cap.named("nthlayer.llm.call")
        assert event.attributes["gen_ai.provider.name"] == "sk-ant-[REDACTED]"
        assert not [v for v in event.attributes.values() if SECRET in str(v)]

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
