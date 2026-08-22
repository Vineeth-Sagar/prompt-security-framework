"""End-to-end orchestrator: input -> preprocessing -> context buffer ->
SWCSA -> IFS-R -> policy -> target LLM, in one call, with per-stage timing.

This is the thing Phase 12's latency metric actually measures, and the
thing a future API route (replacing/extending api/routes/input.py) will
eventually call instead of duplicating pipeline logic inline. For now
it's a standalone, directly-testable function — no route wires it up
yet.

Design decision worth being explicit about: the turn persisted into the
session's context buffer is the *normalized* input text, not
`policy.final_text` (which may be redacted/rewritten, or None for
BLOCK). If a redacted version were stored instead, an attacker's actual
escalation pattern across turns would be erased from the conversation
history IFS-R/SWCSA read on the *next* turn — exactly the signal
`window_role_escalation`/`drift_trend` exist to catch. So the buffer
always sees what was really said, regardless of what got forwarded.

The target LLM is only ever called when `policy.action != "BLOCK"`,
with `policy.final_text` (the sanitized/rewritten/unmodified prompt,
never the raw input) — a BLOCKed prompt short-circuits with no LLM call
at all and a static rejection message instead.
"""

import time

from pydantic import BaseModel

from app.context_buffer.redis_buffer import ContextBuffer, TurnRecord, get_context_buffer
from app.ifsr.fragmenter import fragment
from app.ifsr.reconstructor import ReconstructionResult, reconstruct
from app.ifsr.subintent_classifier import classify
from app.input_layer.base import InputResult
from app.input_layer.router import get_handler, resolve_modality
from app.llm_gateway.base import BaseLLMAdapter, LLMResponse
from app.llm_gateway.factory import get_llm_adapter
from app.policy.engine import PolicyDecision, get_policy_engine
from app.preprocessing.normalizer import normalize
from app.swcsa.drift_score import DriftBreakdown, compute_drift

BLOCK_REJECTION_MESSAGE = (
    "This request was blocked by the policy engine and was not sent to the target LLM."
)


class StageTiming(BaseModel):
    stage: str
    duration_ms: float


class PipelineResult(BaseModel):
    """Everything one call to `run_pipeline` produced, for both the
    caller (final_text/llm_response) and the explainability dashboard
    (everything else).

    Attributes:
        llm_response: The target LLM's response, or None when
            `policy.action == "BLOCK"` (the LLM was never called).
        rejection_message: A static explanation, populated only when
            blocked — `llm_response` staying None isn't itself
            distinguishable from "not populated yet" to a caller that
            doesn't also check `policy.action`.
    """

    session_id: str
    input_result: InputResult
    drift: DriftBreakdown
    ifsr: ReconstructionResult
    policy: PolicyDecision
    llm_response: LLMResponse | None
    rejection_message: str | None
    stage_timings: list[StageTiming]
    total_duration_ms: float


async def run_pipeline(
    session_id: str,
    raw_input: str | bytes,
    modality: str | None = "text",
    content_type: str | None = None,
    filename: str | None = None,
    buffer: ContextBuffer | None = None,
    llm_adapter: BaseLLMAdapter | None = None,
) -> PipelineResult:
    """Run `raw_input` through the full pipeline for `session_id`.

    Args:
        session_id: Groups turns into a conversation for the context
            buffer / SWCSA's drift window.
        raw_input: Text (str) for the text modality (the default), or
            bytes for audio/image — pass modality="audio"/"image"
            explicitly, or rely on content_type/filename, the same
            fallback order api/routes/input.py uses.
        buffer: Injected ContextBuffer, defaulting to the process-wide
            singleton — overridable so tests/eval scripts don't need a
            real Redis server.
        llm_adapter: Injected BaseLLMAdapter, defaulting to the
            process-wide Anthropic singleton — overridable so tests can
            assert the adapter is never called for a BLOCKed prompt
            without making a real API call.
    """
    stage_timings: list[StageTiming] = []
    pipeline_start = time.perf_counter()
    buffer = buffer if buffer is not None else get_context_buffer()
    llm_adapter = llm_adapter if llm_adapter is not None else get_llm_adapter()

    # --- 1. input layer ---
    stage_start = time.perf_counter()
    resolved_modality = resolve_modality(modality, content_type, filename)
    handler = get_handler(resolved_modality)
    input_result = await handler.process(raw_input)
    stage_timings.append(_elapsed("input_layer", stage_start))

    # --- 2. preprocessing ---
    stage_start = time.perf_counter()
    normalized = normalize(input_result.text)
    input_result.text = normalized.text
    input_result.metadata["normalized"] = {
        "text_cased": normalized.text_cased,
        "tokens": normalized.tokens,
    }
    stage_timings.append(_elapsed("preprocessing", stage_start))

    # --- 3. context buffer: read the existing window BEFORE this turn ---
    stage_start = time.perf_counter()
    window = await buffer.get_window(session_id)
    stage_timings.append(_elapsed("context_buffer_read", stage_start))

    # --- 4. SWCSA ---
    stage_start = time.perf_counter()
    drift = compute_drift(window, normalized.text, normalized.text_cased)
    stage_timings.append(_elapsed("swcsa", stage_start))

    # --- 5. IFS-R ---
    stage_start = time.perf_counter()
    fragments = fragment(normalized.text_cased)
    verdicts = [classify(f) for f in fragments]
    ifsr_result = reconstruct(fragments, verdicts)
    stage_timings.append(_elapsed("ifsr", stage_start))

    # --- 6. policy ---
    stage_start = time.perf_counter()
    decision = get_policy_engine().decide(drift, ifsr_result)
    stage_timings.append(_elapsed("policy", stage_start))

    # --- 7. target LLM: only called when policy didn't BLOCK ---
    stage_start = time.perf_counter()
    llm_response: LLMResponse | None = None
    rejection_message: str | None = None
    if decision.action == "BLOCK":
        rejection_message = BLOCK_REJECTION_MESSAGE
    else:
        llm_response = await llm_adapter.generate(decision.final_text)
    stage_timings.append(_elapsed("llm_gateway", stage_start))

    # --- persist this turn for future calls' drift window ---
    stage_start = time.perf_counter()
    await buffer.add_turn(session_id, TurnRecord(text=normalized.text, role="user"))
    stage_timings.append(_elapsed("context_buffer_write", stage_start))

    total_duration_ms = (time.perf_counter() - pipeline_start) * 1000

    return PipelineResult(
        session_id=session_id,
        input_result=input_result,
        drift=drift,
        ifsr=ifsr_result,
        policy=decision,
        llm_response=llm_response,
        rejection_message=rejection_message,
        stage_timings=stage_timings,
        total_duration_ms=total_duration_ms,
    )


def _elapsed(stage: str, start: float) -> StageTiming:
    return StageTiming(stage=stage, duration_ms=(time.perf_counter() - start) * 1000)
