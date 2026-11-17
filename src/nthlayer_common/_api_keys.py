"""Keeping an API key that was mis-set as the model name out of errors and
telemetry (opensrm-7um7).

``NTHLAYER_MODEL`` set to a key instead of a model is a known slip: the model
guards in ``llm.py`` and ``llm_structured.py`` refuse it. The model string
also travels into ``LLMError`` and ``emit_llm_event``, and from there into
logs and OTel, so those show the redacted form built here. The guards
refuse exactly what the redaction recognises (opensrm-ua2m).

Package-internal: not re-exported, and not a general secret scanner. It
knows only the shapes below.
"""

from __future__ import annotations

from urllib.parse import quote

# What a key starts with: Anthropic (sk-ant-), OpenAI and its look-alikes
# (sk-), key- style keys, and an Authorization header value pasted whole.
# Longest first where one prefix starts another, so the redacted form names
# the most specific one.
API_KEY_PREFIXES = ("sk-ant-", "sk-", "key-", "Bearer ")

REDACTED = "[REDACTED]"

# scrub_model also replaces what follows the prefix wherever it stands
# alone (a URL writes "Bearer%20<token>"), but only when it is at least
# this long: replacing a two-letter tail would rewrite unrelated words.
_MIN_BARE_SECRET = 8


def looks_like_api_key(model: str) -> bool:
    """The model guards' test: ``model``, or what follows one of its
    slashes, starts like a key. The same test redaction uses, so nothing
    that would be shown redacted is ever sent to a provider."""
    return _key_in(model) is not None


def key_refusal(model: str) -> str:
    """What a guard says when it refuses ``model``, quoting only the
    redacted form."""
    return (
        f"'{redact_model(model)}' looks like an API key, not a model name: "
        "a model, or a path segment of one, starting sk-, key- or 'Bearer ' "
        "is refused."
    )


def scrub_call(text: str, provider: str, model: str) -> str:
    """``text`` scrubbed of a key-shaped ``model`` and ``provider``. Both,
    because splitting ``<key>/name`` at its slash puts the key in the
    provider slot."""
    return scrub_model(scrub_model(text, model), provider)


def _key_in(model: str) -> tuple[str, str, str] | None:
    """``(lead, key, prefix)`` when ``model`` holds a key-shaped string, else
    None. ``key`` is the key to the end of the string, without surrounding
    whitespace; ``lead`` is what stands before it, ``"provider/"``, a longer
    path, or nothing.

    A key counts wherever a pasted one ends up: with surrounding whitespace,
    and where a model name goes, after any slash (``provider/<key>``,
    ``openrouter/anthropic/<key>``).
    """
    if not isinstance(model, str):
        return None
    # The whole string first, then what follows each slash, left to right:
    # the earliest start gives the longest key, so nothing after it shows.
    starts = [0] + [i + 1 for i, char in enumerate(model) if char == "/"]
    for start in starts:
        rest = model[start:].strip()
        for prefix in API_KEY_PREFIXES:
            if rest.startswith(prefix):
                return model[:start].strip(), rest, prefix
    return None


def redact_model(model: str) -> str:
    """``model`` as it may be shown: unchanged unless it is key-shaped, then
    the prefix that gave it away and ``[REDACTED]`` (``sk-ant-[REDACTED]``,
    ``openai/sk-[REDACTED]``)."""
    found = _key_in(model)
    if found is None:
        return model
    lead, _key, prefix = found
    return f"{lead}{prefix}{REDACTED}"


def scrub_model(text: str, model: str) -> str:
    """``text`` with every occurrence of a key-shaped ``model`` redacted.
    For messages that quote the model: a provider's error body, a URL.

    The key as written becomes its prefix and ``[REDACTED]``. What follows
    the prefix is also replaced where it stands without it, plain or
    percent-encoded, since a URL separates ``Bearer`` from its token with
    ``%20``.
    """
    found = _key_in(model)
    if found is None or not isinstance(text, str):
        # No key in the model: nothing to look for. Not a string: nothing
        # to scrub, and emit_llm_event must not raise.
        return text
    _lead, key, prefix = found
    if key == prefix:
        # Nothing follows the prefix: no secret, and replacing a bare
        # "sk-" would rewrite unrelated words.
        return text
    text = text.replace(key, f"{prefix}{REDACTED}")
    secret = key[len(prefix):].strip()
    if len(secret) >= _MIN_BARE_SECRET:
        text = text.replace(secret, REDACTED).replace(quote(secret, safe=""), REDACTED)
    return text
