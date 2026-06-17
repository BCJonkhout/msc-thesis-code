"""Provider abstraction layer.

Several concrete adapters implement a uniform `AnswererProvider`
interface so the cost ledger sees the same fields regardless of
provider:

  - anthropic     Claude Sonnet 4.6 / Opus 4.7
  - openai        GPT-5.4 (and any future OpenAI model)
  - gemini        Gemini 3.1 family (via google-genai SDK)
  - dashscope     Alibaba first-party Qwen
  - openrouter    OpenRouter aggregator (via OpenAI-compatible adapter)

The completed 4-architecture main study runs a single answerer on the
``gemini`` adapter (gemini-3.1-flash-lite-preview, T=0), with a single
Grok robustness slice via the OpenAI-compatible adapter. The remaining
adapters are retained because their model families were evaluated and
recorded as rejected candidates (configs/models.yaml#rejected_candidates),
so the slate stays reinstateable for a future revision.

The OpenAI-compatible adapter (openai_compatible_provider.py) is
parameterised by base_url + api_key_env_var; OpenRouter is the first
concrete subclass. Adding Groq / DeepInfra / Together / Fireworks
later only requires another small subclass that sets two strings.

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
