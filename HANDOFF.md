# Handoff: Backend Complete, Frontend Next

Context document for continuing this project in a new session. Backend
Phases 0–10 are done, tested, and green on CI. Frontend (Phase 11) has
not been started — `frontend/` exists but is empty.

## What this project is

**Adaptive Context-Aware Multi-Layer Prompt Security & Output Governance
Framework** — a B.E. major project. A layered security gateway sitting
in front of a target LLM:

```
Text/Audio/Image → Multi-Modal Input Layer → Text Normalization
→ Redis Sliding Window Buffer → SWCSA (drift/role-escalation/topic-hopping)
→ Drift Score → IFS-R (fragment → classify → sanitize → rebuild)
→ Policy Engine → [BLOCK | SAFE_REWRITE | PASS] → Target LLM
→ Output Governance (PII redaction / sandboxed code execution)
→ Explainable Logging (Postgres + WebSocket live feed)
```

Repo: **https://github.com/Vineeth-Sagar/prompt-security-framework**
(public), branch `main`. Local path:
`D:\E drive\Major Project\prompt-security-framework`.

The original phased build plan (all 14 phases) is at
`D:\E drive\Major Project\Prepare\build-plan.md` — worth reading for
the full intended scope (Phase 12 eval, Phase 13 dockerize come after
the frontend).

## Build status: Phases 0–10 done, CI green throughout

| Phase | What | Status |
|---|---|---|
| 0 | Repo, Docker Compose, CI, FastAPI skeleton | ✅ |
| 1 | Multi-modal input (text/audio/image handlers) | ✅ |
| 2 | Preprocessing (spaCy normalization, unicode-smuggling defense) | ✅ |
| 3 | Redis sliding-window context buffer | ✅ |
| 4 | SWCSA (semantic drift, role-escalation, topic-hopping, aggregator) | ✅ (tuned) |
| 5 | IFS-R (fragmenter, sub-intent classifier, MinHash rewrite detector, reconstructor) | ✅ |
| 6 | Policy engine (YAML rules → BLOCK/SAFE_REWRITE/PASS) + pipeline orchestrator | ✅ |
| 7 | LLM Gateway (Anthropic + Gemini adapters, pluggable via factory) | ✅ |
| 8 | Output Governance (PII scanner, sandboxed code execution) | ✅ |
| 9 | Auth/RBAC (JWT access+refresh, admin/analyst/viewer roles) | ✅ |
| 10 | Explainable Logging (Postgres decision log, WebSocket live feed) | ✅ |
| 11 | **Frontend Dashboard — NOT STARTED** | ⬜ |
| 12 | Evaluation (accuracy/FP/latency benchmark) | ⬜ |
| 13 | Dockerize & final docs | ⬜ |

63 commits, all authored as `Vineeth-Sagar <vineethsagarhl0@gmail.com>`
(repo-level git config, not global). 271 backend tests passing, 5 skip
locally (platform-specific — see "Known gotchas" below) but run for
real on Linux CI.

## ⚠️ Critical gap the frontend will hit immediately

**No HTTP route runs the full pipeline yet.** `backend/app/pipeline.py`'s
`run_pipeline()` — the function that actually does SWCSA → IFS-R →
policy → LLM → output governance → logging — is only ever called
directly in Python (tests, manual smoke tests). `POST /api/v1/input`
(Phase 1/2) only does input normalization; it stops before SWCSA.

**Before the Playground page can work, something needs to add a route**
(e.g. `POST /api/v1/pipeline` or extend `/api/v1/input`) that calls
`run_pipeline(session_id, raw_input, ...)` and returns a `PipelineResult`
as JSON. This is the first backend gap to close in Phase 11, not
something already done.

## Current API surface (all mounted in `backend/app/main.py`)

| Method | Path | Auth | Notes |
|---|---|---|---|
| GET | `/health` | none | liveness probe |
| POST | `/api/v1/input` | none | input_layer + preprocessing only (see gap above) |
| POST | `/api/v1/auth/register` | none (bootstrap) / admin | first user ever = admin, no auth; after that, admin-only |
| POST | `/api/v1/auth/login` | none | OAuth2 password flow (`username` = email) → `{access_token, refresh_token, token_type}` |
| POST | `/api/v1/auth/refresh` | none | body `{refresh_token}` → new access token |
| GET | `/api/v1/auth/me` | any valid access token | current user's public profile |
| GET | `/api/v1/logs` | admin/analyst | paginated (`limit`/`offset`), filterable by `session_id`/`action`/`start_date`/`end_date` |
| GET | `/api/v1/logs/{id}` | admin/analyst | full DecisionLog detail |
| WS | `/ws/live-decisions?token=...` | admin/analyst | live feed, one JSON event per DecisionLog write |

CORS is already configured (`FRONTEND_ORIGIN` setting, default
`http://localhost:3000` — matches Next.js's default dev port).

### Auth model specifics

- JWT, `HS256`, secret from `JWT_SECRET` setting.
- Access tokens: 30 min. Refresh tokens: 7 days. Distinguished by a
  `type` claim — a refresh token cannot authenticate a request, only
  mint a new access token via `/refresh`.
- Roles: `admin`, `analyst`, `viewer` (`app/auth/models.py`'s
  `UserRole` enum). `require_role(*roles)` is the FastAPI dependency
  factory used to gate routes.
- **Bootstrap**: the very first `/auth/register` call (empty `users`
  table) needs no auth and always creates an `admin`. Every
  registration after that requires an authenticated admin caller.
- WebSocket auth is via `?token=` query param (browsers can't set
  custom headers on a WS handshake) — same access token as HTTP routes.

## Data shapes the frontend will consume

`PipelineResult` (from `app/pipeline.py`, once a route exposes it) —
the thing a Playground page needs:
```
{
  session_id, input_result, drift, ifsr, policy,
  llm_response, pii_scan, sandbox_result,
  final_response_text,        # <- what to actually show the user
  rejection_message,          # <- populated only when policy.action == "BLOCK"
  stage_timings, total_duration_ms
}
```
`policy.action` is one of `"BLOCK" | "SAFE_REWRITE" | "PASS"`.
`policy.matched_rule` names which rule fired (for an explainability
view). `drift.aggregate` is 0–1. `ifsr.removed`/`ifsr.suspicious` list
the fragments that got dropped/flagged.

`DecisionLogPublic` (from `GET /api/v1/logs`) — for the Logs/Audit page:
```
{ id, session_id, input_modality, drift_breakdown, ifsr_result,
  policy_action, matched_rule, pii_found, latency_ms_per_stage, created_at }
```

WS live-feed event (compact, not the full row):
```
{ id, session_id, input_modality, policy_action, matched_rule, created_at }
```

## Running the backend locally

Docker Desktop's daemon is **not running** on this dev machine, and
never has been all session — every backend feature so far was built and
tested **without** a live Postgres/Redis, using `fakeredis` and
in-memory SQLite in tests. If you want to actually run the FastAPI
server for frontend integration testing, you have two options:

1. **Start Docker Desktop**, then `docker compose up --build` from the
   repo root (brings up backend + redis + postgres). Compose file and
   Dockerfile are already written and validated (`docker compose config`
   passes), just never actually run end-to-end.
2. **Run without Docker**: needs a real Redis and Postgres reachable at
   the URLs in `backend/.env` (`REDIS_URL`, `DATABASE_URL`), then:
   ```bash
   cd backend
   source .venv/Scripts/activate   # venv already exists and is populated
   alembic upgrade head             # applies both migrations
   uvicorn app.main:app --reload
   ```

`backend/.venv` already exists with everything installed matching
`requirements.txt`. `backend/.env` already exists (gitignored) with a
working `GEMINI_API_KEY` and an `ANTHROPIC_API_KEY` (valid but **no
credit balance** — Anthropic calls will 400 until credits are added at
console.anthropic.com). `TARGET_LLM_PROVIDER=gemini` is active.

Run tests: `cd backend && pytest tests -q` (no external services
needed — everything's mocked/faked).

## Known gotchas already found and fixed (don't re-discover these)

All were caught by testing against a **genuinely fresh venv install**,
not just the incrementally-patched dev venv — that habit is why these
were caught before CI, mostly:

- `numpy` must stay `<2.0` (pinned `1.26.4`) — spaCy's `thinc` backend
  ABI-crashes on numpy 2.x, not an install-time conflict, a runtime one.
- `bcrypt` must stay `<4.1` (pinned `4.0.1`) — `passlib` (unmaintained)
  probes a `bcrypt` attribute removed in 4.1+.
- `pydantic>=2.12.5` / `httpx>=0.28.1` required once `google-genai` was
  added (pinned `2.13.4` / `0.28.1`; `pydantic-settings` bumped to
  match).
- Alembic's `--autogenerate` output against a SQLModel model always
  omits `import sqlmodel` even when it references
  `sqlmodel.sql.sqltypes.AutoString()` — add the import by hand every
  time before applying a generated migration.
- Any code path that calls `get_redis_client()` (the real
  singleton, unreachable without a running Redis) must be made
  injectable for tests — this bit both `decision_logger.py` and would
  bite any new pub/sub code the same way.
- 5 tests skip on this Windows dev machine specifically (no `resource`
  module for POSIX rlimits, no local `tesseract` binary) — they run for
  real on CI's Linux runners. Don't be alarmed by local skips; check CI.

## Git/GitHub workflow notes

- Commit convention: Conventional Commits (`feat(scope): ...`,
  `test(scope): ...`, `fix(deps): ...`), one logical unit per commit,
  fairly detailed bodies explaining *why*, not just *what* — that
  history is meant to be citable evidence of iterative development for
  the project report.
- Every phase's commits get pushed and CI-verified green before moving
  to the next phase.
- `git push` in this environment sometimes triggers a Windows
  Credential Manager "Select an account" GUI dialog that only a human
  can click through — if a push seems to hang, that's usually why.
- History was rewritten once (all 63 commits' author identity fixed
  from a stray `claude` GitHub account to `Vineeth-Sagar`) via
  `git filter-branch` + force-push. Not expected to recur.

## Phase 11 spec (frontend — the actual next task)

Reproducing the original build-plan wording for continuity:

> **Frontend Dashboard** — Next.js 14 (App Router), TypeScript, Tailwind
> + shadcn/ui, WebSockets for live decision feed.
>
> - Scaffold: Next.js app, Tailwind + shadcn/ui, API client, auth pages.
> - **Playground** — submit a prompt (text/audio/image), see the
>   pipeline's live decision.
> - **Live monitoring** — WS-fed feed of recent decisions across all
>   sessions.
> - **Explainability view** — per-request drill-down (drift score
>   breakdown, fragments, which rule fired).
> - **Logs/audit page** — filter/search historical decisions.
> - **Admin/RBAC page** — manage users/roles.
> - Build and commit one page/feature at a time; wire to the real
>   backend API, no mocked data once the backend endpoint exists.

Repo layout planned in the original build plan:
```
frontend/
  app/{dashboard,logs,playground,admin}
  components/
  lib/api-client.ts
  package.json
```

**Suggested order, given the gap above:**
1. Close the pipeline-endpoint gap (small backend addition) before or
   alongside starting the Playground page — nothing else in the
   frontend has real data to show without it.
2. Scaffold Next.js + auth pages (login against `/api/v1/auth/login`,
   store the token, attach it to subsequent requests).
3. Playground (calls the new pipeline endpoint).
4. Live monitoring (WS client against `/ws/live-decisions`).
5. Explainability drill-down (likely reuses `GET /api/v1/logs/{id}`).
6. Logs/audit page (`GET /api/v1/logs` with filters).
7. Admin/RBAC page (would need new user-management routes — `/auth`
   currently only has register/login/refresh/me, no list/update/delete
   user endpoints yet; that's another small backend gap to close).

Same daily-commit, phase-by-phase, test-as-you-go discipline as the
backend phases — small reviewable commits, push and confirm CI green
after each.
