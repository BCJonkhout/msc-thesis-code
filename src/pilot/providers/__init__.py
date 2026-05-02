"""Provider abstraction layer.

Five concrete adapters implement a uniform `AnswererProvider` interface
so the cost ledger sees the same fields regardless of provider:

  - anthropic     Claude Sonnet 4.6 / Opus 4.7
  - openai        GPT-5.4 (and any future OpenAI model)
  - gemini        Gemini 3.1 Pro Preview (via google-genai SDK)
  - dashscope     Alibaba first-party Qwen
  - openrouter    OpenRouter aggregator (via OpenAI-compatible adapter)

The OpenAI-compatible adapter (openai_compatible_provider.py) is
parameterised by base_url + api_key_env_var; OpenRouter is the first
concrete subclass. Adding Groq / DeepInfra / Together / Fireworks
later only requires another small subclass that sets two strings.

Ollama adapter is deferred until the deployment decision in Phase G
picks path B (local serving).

See pilot plan § 4.3 (KV-cache verification per provider) and § 4.1
(uniform cost-ledger fields).
"""
from pilot.providers.base import (
    AnswererProvider,
    CacheControl,
    ProviderResult,
    UnknownProviderError,
)
from pilot.providers.factory import get_provider

__all__ = [
    "AnswererProvider",
    "CacheControl",
    "ProviderResult",
    "UnknownProviderError",
    "get_provider",
]
