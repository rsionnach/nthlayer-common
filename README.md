# nthlayer-common

Shared utilities for the [NthLayer](https://nthlayer.io) ecosystem. Model-agnostic LLM interface — one function, any provider.

## Install

```bash
pip install nthlayer-common
```

## Usage

```python
from nthlayer_common import llm_call

result = llm_call(
    system="You are a triage agent...",
    user="Evaluate this incident...",
)
print(result.text)
```

## Provider support

Two API formats cover the entire market:

| Provider | Model format | API |
|----------|-------------|-----|
| Anthropic | `anthropic/claude-sonnet-4-20250514` | Messages API |
| OpenAI | `openai/gpt-4o` | Chat Completions |
| Ollama | `ollama/llama3.1` | Chat Completions |
| Azure | `azure/my-deployment` | Chat Completions |
| Together | `together/meta-llama/Llama-3-70b` | Chat Completions |
| Groq | `groq/llama-3.1-70b-versatile` | Chat Completions |
| Mistral | `mistral/mistral-large-latest` | Chat Completions |
| vLLM | `vllm/my-model` | Chat Completions |
| LM Studio | `lmstudio/my-model` | Chat Completions |

## Configuration

```bash
NTHLAYER_MODEL="anthropic/claude-sonnet-4-20250514"   # default
NTHLAYER_LLM_TIMEOUT="60"                              # seconds
ANTHROPIC_API_KEY="sk-ant-..."                         # for anthropic/* models
OPENAI_API_KEY="sk-..."                                # for openai/*, together/*, groq/*, etc.
OPENAI_API_BASE="http://localhost:11434/v1"            # override endpoint URL
```

## Also includes

- `strip_markdown_fences(text)` — clean LLM response parsing
- `clamp(value, low=0.0, high=1.0)` — value clamping utility
- `LLMResponse` — dataclass with `text`, `model`, `provider`, `input_tokens`, `output_tokens`
- `LLMError` — structured error with `provider`, `model`, `cause` context

## No third-party LLM libraries

One dependency: `httpx`. No LiteLLM, no SDKs. Raw HTTP calls to provider APIs.

## License

Apache 2.0
