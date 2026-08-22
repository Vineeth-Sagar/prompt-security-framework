"""Picks the active target-LLM adapter based on settings.target_llm_provider.

The one place that knows about every concrete adapter — everything else
(pipeline.py, routes) depends only on BaseLLMAdapter and calls
get_llm_adapter(), never importing a concrete adapter class directly.
Adding a new provider means adding a branch here, not touching callers.
"""

from app.config import get_settings
from app.llm_gateway.base import BaseLLMAdapter

_adapter: BaseLLMAdapter | None = None


def get_llm_adapter() -> BaseLLMAdapter:
    """Process-wide default adapter (lazy singleton) — a plain function
    (not @lru_cache), same pattern as get_context_buffer()/
    get_policy_engine(), so it stays overridable (e.g. pipeline.py's
    injectable `llm_adapter` param, or a future FastAPI dependency).

    Raises:
        ValueError: if settings.target_llm_provider isn't a known provider.
    """
    global _adapter
    if _adapter is None:
        provider = get_settings().target_llm_provider.lower()

        if provider == "anthropic":
            from app.llm_gateway.anthropic_adapter import AnthropicAdapter

            _adapter = AnthropicAdapter()
        elif provider == "gemini":
            from app.llm_gateway.gemini_adapter import GeminiAdapter

            _adapter = GeminiAdapter()
        else:
            raise ValueError(
                f"Unknown TARGET_LLM_PROVIDER {provider!r} — expected 'anthropic' or 'gemini'."
            )

    return _adapter
