# nthlayer-common LLM interface

Two public surfaces in `nthlayer_common.llm` and
`nthlayer_common.llm_structured`, plus a CI stub
(`nthlayer_common.llm_stub`).

## Model format

`"provider/model-name"` — e.g. `anthropic/claude-sonnet-4-20250514`,
`openai/gpt-4o`, `ollama/llama3.1`.

## Provider routing

- `anthropic/*` → Anthropic Messages API
  (`api.anthropic.com/v1/messages`).
- Everything else → OpenAI-compatible Chat Completions API.

## Environment variables

- `NTHLAYER_MODEL` — override default model (default
  `anthropic/claude-sonnet-4-20250514`).
- `NTHLAYER_LLM_TIMEOUT` — request timeout in seconds (default 60).
- `ANTHROPIC_API_KEY` — required for `anthropic/*` models.
- `OPENAI_API_KEY` — for OpenAI-compatible providers (optional for
  Ollama / vLLM).
- `OPENAI_API_BASE` — override endpoint URL for any provider.
- `AZURE_OPENAI_ENDPOINT` — Azure OpenAI resource URL.

## Default base URLs by provider

| Provider | URL |
|---|---|
| `openai` | `https://api.openai.com/v1` |
| `ollama` | `http://localhost:11434/v1` |
| `vllm` | `http://localhost:8000/v1` |
| `lmstudio` | `http://localhost:1234/v1` |
| `together` | `https://api.together.xyz/v1` |
| `groq` | `https://api.groq.com/openai/v1` |
| `mistral` | `https://api.mistral.ai/v1` |

## Bare model name guessing (`_guess_provider`)

- `claude*` → anthropic
- `gpt*` / `o1*` / `o3*` → openai
- `llama*` / `mistral*` / `gemma*` → ollama
- else → openai

## LLMResponse / LLMError

- `LLMResponse` (`@dataclass`): `text` (str), `model` (str), `provider`
  (str), `input_tokens` (int | None), `output_tokens` (int | None).
- `LLMError` carries `provider`, `model`, `cause`, `status_code`
  alongside the message.

## Retry behaviour (`llm_call` default `retry=3`)

- Transient errors retried with exponential backoff + full jitter:
  429, 408, 502, 503, connection errors, timeouts.
- Permanent errors fail immediately: 400, 401, 403, 404, 422.
- `Retry-After` header parsed and respected; the timeout budget is
  checked before each sleep.
- Pass `retry=0` to disable retries.
- **API-key guard**: raises `LLMError` if the `model` string starts
  with `sk-ant-`, `sk-`, `key-`, or `Bearer ` — caught common foot-gun
  where the env var content gets passed as the model name.

## Structured calls (`llm_structured.py`)

`structured_call(system, user, response_model, model?,
max_tokens=2000, timeout?, max_retries=3)` returns a validated
`pydantic.BaseModel` instance. Uses Instructor for JSON-schema
enforcement and retry on malformed responses. Raises `LLMError` on
provider errors or exhausted retries.

`structured_call_with_usage(...)` returns
`StructuredCallResult(data=..., usage=StructuredCallUsage(input,
output))`.

## CI stub (`NTHLAYER_LLM_STUB=canned`)

Module: `nthlayer_common.llm_stub`. Added for opensrm-saun.1
(three-tier integration test).

Setting `NTHLAYER_LLM_STUB=canned` in the environment short-circuits
`llm_call()`, `structured_call()`, and `structured_call_with_usage()`
**before any HTTP request** and returns a deterministic canned
response. Not a behavioural fake — every call of a given role returns
the same data regardless of input. Purpose: exercise wiring (verdict
shape, lineage propagation, store writes) without a real LLM API key.

**Do not enable in production.**

Activation point: lazy `from nthlayer_common.llm_stub import …` inside
the `llm_call` / `structured_call` / `structured_call_with_usage`
body. The stub module is not imported during normal use; this keeps
the import graph cycle-safe with `nthlayer_common.llm`.

### `llm_call()` raw-text dispatch

Role detected via case-insensitive substring match in the system
prompt (markers in `_ROLE_MARKERS`):

| System prompt contains | Canned JSON shape |
|---|---|
| `"you are a triage agent"` | `{severity, blast_radius, affected_slos, assigned_team, reasoning, confidence}` |
| `"you are a communication agent"` | `{updates: [{channel, update_type, content}], reasoning, confidence}` |
| `"you are an investigation agent"` | `{hypotheses: [{description, confidence, evidence, change_candidate}], root_cause, root_cause_confidence, reasoning, confidence}` |
| `"you are a remediation agent"` | `{proposed_action: "rollback", target: "fraud-detect", risk_assessment, requires_human_approval: true, reasoning, confidence}` |
| (no marker) | `{reasoning, confidence}` |

All canned text is JSON-shaped to match the schema each respond
agent's `_parse_json` expects (so e.g. `RemediationAgent`'s
safe-action registry check accepts `"rollback"` because that action
exists in the registry with `requires_approval=true`).

### `structured_call()` / `structured_call_with_usage()` dispatch

Registry keyed by `response_model.__name__`:

| `response_model` | Returns |
|---|---|
| `EvaluationResult` (measure evaluator) | one passing `DimensionScore` (score=0.85), `confidence=0.8` |
| `SnapshotSummary` (correlate snapshot summary) | stub summary string, empty `notable_omissions`, `confidence=0.5` (ungrounded but non-zero — passes confidence>0 filters) |
| `TriageResponse` (respond triage agent, P3-E.2) | severity=2, blast_radius=["fraud-detect"], assigned_team="payments", confidence=0.7 |
| `InvestigationResponse` (respond investigation agent, P3-E.2) | single deploy-regression hypothesis, root_cause set, confidence=0.8 |
| `CommunicationResponse` (respond communication agent, P3-E.2) | single status_page update, update_type="initial", confidence=0.7 |
| `RemediationResponse` (respond remediation agent, P3-E.2) | proposed_action="rollback", target="fraud-detect", requires_human_approval=True, confidence=0.7 |
| (anything else) | `NotImplementedError` — prevents new structured-call sites silently producing garbage |

`structured_call_with_usage()` wraps the canned model in
`StructuredCallResult(data=…, usage=StructuredCallUsage(0, 0))`.

### Public helpers

- `is_stub_enabled() -> bool`
- `stub_text_response(system, model) -> LLMResponse`
- `stub_structured_response(response_model) -> T`

Currently only the `llm_call`/`structured_call` wrappers consume them.

### Adding coverage

- New agent role → extend `_ROLE_MARKERS` (ordered tuple) and
  `_TEXT_BY_ROLE`.
- New structured-call site → register a factory in
  `_STRUCTURED_FACTORIES`.

### Tests

`tests/test_llm_stub.py` — 26 tests covering role detection (incl.
`None` system, case-insensitive marker match), all four respond agent
shapes, both structured callers, env-var case/whitespace variants
(`CANNED`, ` canned `, `canned\n`), env-var-unset preserves HTTP path,
unknown structured model raises clearly, name-collision with
incompatible shape raises with the qualified path.
