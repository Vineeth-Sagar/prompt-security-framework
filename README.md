# Adaptive Context-Aware Multi-Layer Prompt Security & Output Governance Framework

**Status: Phase 0 — Foundations**

A layered security gateway that sits in front of a target LLM. It inspects
multi-modal input (text, audio, image), tracks conversational context across
turns to detect slow-burn / multi-step prompt injection, fragments and
safely reconstructs risky prompts instead of hard-blocking them, enforces
policy before forwarding to the target LLM, and governs LLM output for
leaks/unsafe content — with full decision logging for an explainability
dashboard.

## Architecture

```
Text/Audio/Image → Multi-Modal Input Layer → Text Normalization (FastAPI gateway)
→ Redis Sliding Window Buffer (3–5 msgs) → SWCSA (context drift/role escalation/topic hopping)
→ Drift Score → IFS-R (fragment → detect → sanitize → rebuild) → Policy Engine (OPA-lite)
→ [BLOCK | SAFE-REWRITE | PASS] → Target LLM → Output Governance (PII/leak/sandbox)
→ Explainable AI logs → Admin Dashboard (RBAC)
```

| Component | Role |
|---|---|
| **Input Layer** | Normalizes text, transcribes audio (faster-whisper), OCRs images (pytesseract) into a common text representation |
| **Preprocessing** | spaCy-based tokenization/cleaning/unicode normalization |
| **Context Buffer** | Redis-backed sliding window (3–5 turns) per session, with TTL |
| **SWCSA** | Sliding Window Context Shift Analyzer — semantic drift (embeddings), role-escalation heuristics, topic-hopping entropy, aggregated into a drift score |
| **IFS-R** | Intent Fragmentation & Safe Reconstruction — splits a prompt into micro-intents, classifies risk per fragment, sanitizes, rebuilds a safe prompt |
| **Policy Engine** | Data-driven rules (YAML) consuming drift score + IFS-R verdict → BLOCK / SAFE_REWRITE / PASS |
| **LLM Gateway** | Pluggable `BaseLLMAdapter` (Anthropic, Gemini — `TARGET_LLM_PROVIDER` selects which) — only sanitized prompts reach here |
| **Output Governance** | PII/sensitive-data scanning on LLM responses, sandboxed execution for generated code |
| **Explainable Logging** | Full per-request decision trace persisted to Postgres, streamed live via WebSocket |
| **Admin Dashboard** | Next.js frontend — playground, live monitoring, explainability drill-down, audit logs, RBAC admin |

## Repo layout

```
backend/
  app/
    input_layer/        # text/audio/image handlers
    preprocessing/       # normalization
    context_buffer/      # Redis sliding window
    swcsa/                # drift analyzer + score
    ifsr/                 # intent fragmentation & reconstruction
    policy/               # rule engine
    llm_gateway/          # target LLM adapter
    output_governance/    # PII scan, sandbox
    auth/                 # JWT/RBAC
    logging/              # decision logger
    api/routes/           # REST + WS endpoints
  tests/                  # pytest, one file per module
  datasets/               # synthetic attack/benign corpora for eval
  eval/                   # accuracy/FP/latency benchmark scripts
frontend/
  app/{dashboard,logs,playground,admin}
```

## Local development

Requirements: Docker Desktop, Python 3.11, Node 18+.

```bash
docker compose up --build
```

This brings up: `backend` (FastAPI, http://localhost:8000, docs at `/docs`),
`redis`, `postgres`, and `frontend` (Next.js, http://localhost:3000).

Backend only, without Docker:

```bash
cd backend
python -m venv .venv && .venv/Scripts/activate  # or source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Run tests:

```bash
cd backend
pytest
```

## Status

Build is in progress, following a phased plan (Foundations → Multi-Modal
Input → Preprocessing → Context Buffer → SWCSA → IFS-R → Policy → LLM
Gateway → Output Governance → Auth/RBAC → Explainable Logging → Frontend →
Evaluation → Dockerize). See commit history for progress; each phase lands
as a series of small, reviewable commits.
