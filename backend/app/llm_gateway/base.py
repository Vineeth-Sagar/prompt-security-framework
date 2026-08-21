"""Common interface every target-LLM adapter implements.

Only sanitized prompts (policy.action != BLOCK, using policy.final_text
— see pipeline.py) are ever supposed to reach `generate()`. The adapter
itself doesn't enforce that; the pipeline orchestrator does, by only
calling it on that path. This interface exists so the target LLM is
swappable (Anthropic today, OpenAI/local later) without touching
anything upstream of it.
"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class LLMResponse(BaseModel):
    """Normalized response from any target LLM.

    Attributes:
        text: The model's completion text.
        model: Which model actually served the request (adapters may
            fall back/retry onto a different model string than what was
            requested; this reports what really ran).
        usage: Token usage, adapter-specific keys (e.g.
            {"input_tokens": .., "output_tokens": ..}) — kept as a
            free-form dict rather than a fixed schema since different
            providers report usage differently.
        latency_ms: Wall-clock time the call took, for the Phase 12
            latency metric.
    """

    text: str
    model: str
    usage: dict[str, int]
    latency_ms: float


class BaseLLMAdapter(ABC):
    """Base class for a target-LLM adapter."""

    @abstractmethod
    async def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Generate a completion for `prompt`.

        Implementations should raise a clean, typed exception (not let
        a raw SDK exception propagate) for the failure modes callers
        need to handle explicitly — see anthropic_adapter.py's
        LLMTimeoutError/LLMRateLimitExceededError.
        """
        raise NotImplementedError
