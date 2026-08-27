/**
 * TypeScript mirrors of the backend's Pydantic response models.
 *
 * Kept as one hand-maintained file rather than generated from an
 * OpenAPI schema — the backend's response shapes are documented in
 * HANDOFF.md and change rarely enough that codegen isn't worth the
 * build-step complexity for this project. Field names/types below were
 * checked directly against the backend source, not just the docstrings:
 * app/pipeline.py, app/policy/engine.py, app/swcsa/drift_score.py,
 * app/ifsr/reconstructor.py, app/input_layer/base.py,
 * app/llm_gateway/base.py, app/output_governance/{pii_scanner,sandbox_runner}.py,
 * app/auth/models.py, app/api/routes/{auth,logs,ws}.py,
 * app/logging/decision_logger.py.
 */

export type Modality = "text" | "image" | "audio";

export type PolicyAction = "BLOCK" | "SAFE_REWRITE" | "PASS";

export type UserRole = "admin" | "analyst" | "viewer";

// --- auth ---

export interface UserPublic {
  id: number;
  email: string;
  role: UserRole;
  is_active: boolean;
  created_at: string;
}

export interface TokenPair {
  access_token: string;
  refresh_token: string;
  token_type: string;
}

export interface AccessTokenResponse {
  access_token: string;
  token_type: string;
}

// --- pipeline (POST /api/v1/pipeline/run) ---

export interface InputResult {
  text: string;
  modality: Modality;
  confidence: number | null;
  metadata: Record<string, unknown>;
}

export interface DriftBreakdown {
  semantic: number;
  role_escalation: number;
  topic_entropy: number;
  window_role_escalation: number;
  drift_trend: number;
  matched_patterns: string[];
  aggregate: number;
}

export interface ReconstructionResult {
  safe_text: string;
  removed: string[];
  suspicious: string[];
  modified: boolean;
  blocked: boolean;
}

export interface PolicyDecision {
  action: PolicyAction;
  matched_rule: string;
  final_text: string | null;
}

export interface LLMResponse {
  text: string;
  model: string;
  usage: Record<string, number>;
  latency_ms: number;
}

export interface PIIMatch {
  type: string;
  // Character offsets [start, end) into the pre-redaction text.
  span: [number, number];
}

export interface PIIScanResult {
  found: PIIMatch[];
  redacted_text: string;
}

export interface SandboxResult {
  stdout: string;
  stderr: string;
  exit_code: number | null;
  timed_out: boolean;
  code: string;
}

export interface StageTiming {
  stage: string;
  duration_ms: number;
}

export interface PipelineResult {
  session_id: string;
  input_result: InputResult;
  drift: DriftBreakdown;
  ifsr: ReconstructionResult;
  policy: PolicyDecision;
  llm_response: LLMResponse | null;
  pii_scan: PIIScanResult | null;
  sandbox_result: SandboxResult | null;
  final_response_text: string | null;
  rejection_message: string | null;
  stage_timings: StageTiming[];
  total_duration_ms: number;
  /** The persisted DecisionLog row's id — always populated by the time
   * the response reaches the client (logging happens unconditionally,
   * see app/pipeline.py), so treat this as non-null in practice despite
   * the wire type; a full trace lives at GET /api/v1/logs/{log_id}. */
  log_id: number | null;
}

// --- logs (GET /api/v1/logs, GET /api/v1/logs/{id}) ---

export interface DecisionLogPublic {
  id: number;
  session_id: string;
  user_id: number | null;
  user_email: string | null;
  input_modality: string;
  drift_breakdown: DriftBreakdown;
  ifsr_result: ReconstructionResult;
  policy_action: PolicyAction;
  matched_rule: string;
  pii_found: PIIMatch[];
  latency_ms_per_stage: Record<string, number>;
  created_at: string;
}

export interface PaginatedLogs {
  items: DecisionLogPublic[];
  total: number;
  limit: number;
  offset: number;
}

export interface LogFilters {
  session_id?: string;
  /** Admin-only: filter to one user's decisions (ignored server-side for non-admins). */
  user_id?: number;
  action?: PolicyAction;
  start_date?: string;
  end_date?: string;
  limit?: number;
  offset?: number;
}

// --- WS /ws/live-decisions ---

/** The compact event ws.py forwards — not the full DecisionLogPublic row. */
export interface LiveDecisionEvent {
  id: number;
  session_id: string;
  user_id: number | null;
  user_email: string | null;
  input_modality: string;
  policy_action: PolicyAction;
  matched_rule: string;
  created_at: string;
}
