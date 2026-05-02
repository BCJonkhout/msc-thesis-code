"""Provider abstraction layer.

Four concrete adapters (anthropic, openai, gemini, dashscope) implement
a uniform `AnswererProvider` interface so the cost ledger sees the
same fields regardless of provider. Ollama adapter is deferred until
the deployment decision in Phase G picks path B.

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
