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

Output governance runs on whatever the LLM actually returned, before
any of it is treated as safe to show the user. PII scanning always
runs; the sandbox only runs when the response contains a fenced code
block. The two operate on different inputs deliberately: the sandbox
executes the LLM's *raw* text (so it observes the code's real behavior,
not a version mangled by `[REDACTED:...]` substitutions mid-syntax),
while `final_response_text` — the thing actually meant to reach the
user — is built from the *redacted* text, so a leaked secret sitting in
a code comment or string still gets caught in what's delivered even
though the sandbox ran the original.

The very last thing every call does — regardless of outcome, including
BLOCK — is persist a DecisionLog and publish a live-feed event
(decision_logger.log_decision). This happens after `PipelineResult` is
fully built, as a side effect rather than a timed pipeline stage: timing
"how long logging took" would have to either exclude itself from its
own snapshot or log a second time after appending its own duration,
neither of which is worth the complexity for what's fundamentally
fire-and-forget bookkeeping, not a decision-making stage.
"""

import time

from pydantic import BaseModel
from sqlmodel.ext.asyncio.session import AsyncSession

from app.context_buffer.redis_buffer import ContextBuffer, TurnRecord, get_context_buffer
from app.db import get_engine
from app.ifsr.fragmenter import fragment
from app.ifsr.reconstructor import ReconstructionResult, reconstruct
from app.ifsr.subintent_classifier import classify
from app.input_layer.base import InputResult
from app.input_layer.router import get_handler, resolve_modality
from app.llm_gateway.base import BaseLLMAdapter, LLMResponse
from app.llm_gateway.factory import get_llm_adapter
from app.logging.decision_logger import log_decision
from app.output_governance.pii_scanner import PIIScanResult, scan
from app.output_governance.sandbox_runner import SandboxResult, sandbox_llm_output
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
        llm_response: The target LLM's raw response, or None when
            `policy.action == "BLOCK"` (the LLM was never called). Kept
            unredacted for audit/explainability — `final_response_text`
            is what should actually reach the user.
        pii_scan: PII scan of `llm_response.text`, or None when the LLM
            was never called.
        sandbox_result: Result of executing the first fenced code block
            found in `llm_response.text`, or None when the LLM was
            never called *or* the response contained no code block
            (the common case — this is not an error signal).
        final_response_text: The text that should actually be shown to
            the user — `pii_scan.redacted_text` when the LLM was
            called, None for BLOCK (use `rejection_message` instead).
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
    pii_scan: PIIScanResult | None
    sandbox_result: SandboxResult | None
    final_response_text: str | None
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
    db_session: AsyncSession | None = None,
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
        db_session: Injected AsyncSession for decision logging,
            defaulting to a fresh session from the process-wide engine
            — overridable so tests can point logging at an in-memory
            SQLite database instead of real Postgres.
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

    # --- 8. output governance: only runs on an actual LLM response ---
    stage_start = time.perf_counter()
    pii_scan: PIIScanResult | None = None
    sandbox_result: SandboxResult | None = None
    final_response_text: str | None = None
    if llm_response is not None:
        pii_scan = scan(llm_response.text)
        # Sandbox the *raw* response — a redacted code block is mangled
        # syntax, not the code the LLM actually produced (see module
        # docstring). sandbox_llm_output returns None by itself when
        # there's no fenced code block, so no extra branch is needed here.
        sandbox_result = sandbox_llm_output(llm_response.text)
        final_response_text = pii_scan.redacted_text
    stage_timings.append(_elapsed("output_governance", stage_start))

    # --- persist this turn for future calls' drift window ---
    stage_start = time.perf_counter()
    await buffer.add_turn(session_id, TurnRecord(text=normalized.text, role="user"))
    stage_timings.append(_elapsed("context_buffer_write", stage_start))

    total_duration_ms = (time.perf_counter() - pipeline_start) * 1000

    pipeline_result = PipelineResult(
        session_id=session_id,
        input_result=input_result,
        drift=drift,
        ifsr=ifsr_result,
        policy=decision,
        llm_response=llm_response,
        pii_scan=pii_scan,
        sandbox_result=sandbox_result,
        final_response_text=final_response_text,
        rejection_message=rejection_message,
        stage_timings=stage_timings,
        total_duration_ms=total_duration_ms,
    )

    # --- explainable logging: persist + broadcast, regardless of outcome ---
    # Reuses `buffer`'s own Redis client for the publish step (rather
    # than the separate process-wide singleton) so injecting a fake
    # buffer for tests also redirects decision-log broadcasts — one
    # Redis connection to override, not two.
    if db_session is not None:
        await log_decision(pipeline_result, db_session, redis_client=buffer.client)
    else:
        async with AsyncSession(get_engine()) as session:
            await log_decision(pipeline_result, session, redis_client=buffer.client)

    return pipeline_result


def _elapsed(stage: str, start: float) -> StageTiming:
    return StageTiming(stage=stage, duration_ms=(time.perf_counter() - start) * 1000)
