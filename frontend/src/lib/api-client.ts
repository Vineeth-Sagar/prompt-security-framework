/**
 * Typed fetch wrapper for the backend API (see repo-root HANDOFF.md for
 * the full route table this is built against).
 *
 * ## Auth storage strategy (decided here, per Phase 11.1's brief)
 *
 * Tokens are kept in `localStorage` (see auth-storage.ts), attached as
 * a `Bearer` header on every request by this module — not httpOnly
 * cookies. That's a deliberate call, not the default Next.js would
 * nudge you toward:
 *
 * - The backend is a separate FastAPI service on its own origin
 *   (`NEXT_PUBLIC_API_URL`), issuing its own JWTs via OAuth2-password
 *   login. It has no notion of a Next.js-managed session cookie, and
 *   making it accept one would mean either putting Next in front of it
 *   as a proxy for every request, or configuring cross-origin
 *   SameSite=None cookies — real infrastructure for a local prototype.
 * - The WebSocket live feed (`/ws/live-decisions?token=...`) connects
 *   directly from the browser to the backend and authenticates via a
 *   query param, because browsers can't set custom headers on a WS
 *   handshake. That *requires* the access token to be readable by
 *   client-side JS regardless of how HTTP auth is done — so an
 *   httpOnly cookie wouldn't avoid client-side token exposure here, it
 *   would just add a second, inconsistent auth path for the one
 *   endpoint that can't use it.
 *
 * Trade-off, stated plainly: `localStorage` is readable by any script
 * on the page, so an XSS bug becomes a token-theft bug. For a local B.E.
 * major-project prototype talking to a decoupled JWT API, that's an
 * accepted trade for a single consistent auth mechanism across HTTP and
 * WS. A production deployment of this architecture would want a
 * same-origin BFF (Next Route Handlers proxying to the backend, cookie
 * to the browser) — out of scope here.
 *
 * On a 401 from any authenticated call, this module makes one silent
 * attempt to refresh via `/auth/refresh` and retries the original
 * request once; if that also fails, it clears stored tokens and
 * re-throws so the caller (ultimately AuthProvider) can redirect to
 * /login.
 */

import { clearTokens, getAccessToken, getRefreshToken, setTokens } from "@/lib/auth-storage";
import type {
  AccessTokenResponse,
  LogFilters,
  PaginatedLogs,
  DecisionLogPublic,
  Modality,
  PipelineResult,
  TokenPair,
  UserPublic,
  UserRole,
} from "@/lib/types";

export class ApiError extends Error {
  status: number;
  detail: unknown;

  constructor(status: number, detail: unknown) {
    super(typeof detail === "string" ? detail : `Request failed with status ${status}`);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

function getApiBaseUrl(): string {
  const url = process.env.NEXT_PUBLIC_API_URL;
  if (!url) {
    throw new Error(
      "NEXT_PUBLIC_API_URL is not set — copy frontend/.env.local.example to .env.local."
    );
  }
  return url.replace(/\/$/, "");
}

/** Endpoints that must never trigger the 401-refresh-retry dance —
 * doing so on /auth/login or /auth/refresh itself would infinite-loop. */
const NO_REFRESH_PATHS = ["/api/v1/auth/login", "/api/v1/auth/refresh", "/api/v1/auth/register"];

interface RequestOptions {
  method?: string;
  body?: unknown;
  form?: FormData;
  auth?: boolean; // default true
  searchParams?: Record<string, string | number | undefined>;
}

async function rawRequest<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { method = "GET", body, form, auth = true, searchParams } = options;

  const url = new URL(getApiBaseUrl() + path);
  if (searchParams) {
    for (const [key, value] of Object.entries(searchParams)) {
      if (value !== undefined) url.searchParams.set(key, String(value));
    }
  }

  const headers: Record<string, string> = {};
  if (body !== undefined) headers["Content-Type"] = "application/json";
  if (auth) {
    const token = getAccessToken();
    if (token) headers["Authorization"] = `Bearer ${token}`;
  }

  const response = await fetch(url.toString(), {
    method,
    headers,
    body: form ?? (body !== undefined ? JSON.stringify(body) : undefined),
  });

  if (response.ok) {
    if (response.status === 204) return undefined as T;
    return (await response.json()) as T;
  }

  if (response.status === 401 && auth && !NO_REFRESH_PATHS.includes(path)) {
    const refreshed = await tryRefresh();
    if (refreshed) {
      return rawRequest<T>(path, options);
    }
    clearTokens();
  }

  let detail: unknown;
  try {
    detail = (await response.json()).detail;
  } catch {
    detail = response.statusText;
  }
  throw new ApiError(response.status, detail);
}

let refreshInFlight: Promise<boolean> | null = null;

/** De-duplicated so concurrent 401s (e.g. several widgets fetching at
 * once) share one refresh call instead of racing separate ones. */
async function tryRefresh(): Promise<boolean> {
  if (!refreshInFlight) {
    refreshInFlight = (async () => {
      const refreshToken = getRefreshToken();
      if (!refreshToken) return false;
      try {
        const response = await rawRequest<AccessTokenResponse>("/api/v1/auth/refresh", {
          method: "POST",
          body: { refresh_token: refreshToken },
          auth: false,
        });
        setTokens(response.access_token);
        return true;
      } catch {
        return false;
      }
    })();
  }

  try {
    return await refreshInFlight;
  } finally {
    refreshInFlight = null;
  }
}

// --- auth ---

export async function register(
  email: string,
  password: string,
  role?: UserRole
): Promise<UserPublic> {
  return rawRequest<UserPublic>("/api/v1/auth/register", {
    method: "POST",
    body: { email, password, ...(role ? { role } : {}) },
    // Bootstrap registration (first-ever user) needs no auth; later
    // registrations need an admin's token, which — if present — is
    // still attached since `auth` defaults to true here.
  });
}

export async function login(email: string, password: string): Promise<TokenPair> {
  const url = new URL(getApiBaseUrl() + "/api/v1/auth/login");
  // OAuth2 password flow is form-encoded, not JSON — the one route
  // that doesn't go through rawRequest's JSON body handling.
  const body = new URLSearchParams({ username: email, password });

  const response = await fetch(url.toString(), {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: body.toString(),
  });

  if (!response.ok) {
    let detail: unknown;
    try {
      detail = (await response.json()).detail;
    } catch {
      detail = response.statusText;
    }
    throw new ApiError(response.status, detail);
  }

  const tokens = (await response.json()) as TokenPair;
  setTokens(tokens.access_token, tokens.refresh_token);
  return tokens;
}

export function logout(): void {
  clearTokens();
}

export async function me(): Promise<UserPublic> {
  return rawRequest<UserPublic>("/api/v1/auth/me");
}

// --- pipeline ---

export async function runPipeline(params: {
  file: File | Blob;
  filename: string;
  modality: Modality;
  sessionId?: string;
}): Promise<PipelineResult> {
  const form = new FormData();
  form.append("file", params.file, params.filename);
  form.append("modality", params.modality);
  if (params.sessionId) form.append("session_id", params.sessionId);

  return rawRequest<PipelineResult>("/api/v1/pipeline/run", {
    method: "POST",
    form,
  });
}

/** Convenience wrapper for the common case: plain text submitted as a
 * file, matching what the backend's multipart route expects. */
export async function runPipelineText(text: string, sessionId?: string): Promise<PipelineResult> {
  const blob = new Blob([text], { type: "text/plain" });
  return runPipeline({ file: blob, filename: "prompt.txt", modality: "text", sessionId });
}

// --- user management (admin only) ---

export async function listUsers(): Promise<UserPublic[]> {
  return rawRequest<UserPublic[]>("/api/v1/users");
}

export async function updateUser(
  id: number,
  patch: { role?: UserRole; is_active?: boolean }
): Promise<UserPublic> {
  return rawRequest<UserPublic>(`/api/v1/users/${id}`, {
    method: "PATCH",
    body: patch,
  });
}

// --- logs ---

export async function listLogs(filters: LogFilters = {}): Promise<PaginatedLogs> {
  return rawRequest<PaginatedLogs>("/api/v1/logs", {
    searchParams: {
      session_id: filters.session_id,
      action: filters.action,
      // Only meaningful for admins; the backend ignores it for other
      // roles (they're hard-scoped to their own decisions server-side).
      user_id: filters.user_id,
      start_date: filters.start_date,
      end_date: filters.end_date,
      limit: filters.limit,
      offset: filters.offset,
    },
  });
}

export async function getLog(id: number): Promise<DecisionLogPublic> {
  return rawRequest<DecisionLogPublic>(`/api/v1/logs/${id}`);
}

// --- websocket ---

/** Builds the live-decisions WS URL with the current access token as
 * `?token=` — the only auth mechanism the browser's WebSocket API
 * supports without a custom-header workaround (see module docstring). */
export function buildLiveDecisionsWsUrl(): string | null {
  const token = getAccessToken();
  if (!token) return null;

  const httpUrl = new URL(getApiBaseUrl());
  const wsProtocol = httpUrl.protocol === "https:" ? "wss:" : "ws:";
  return `${wsProtocol}//${httpUrl.host}/ws/live-decisions?token=${encodeURIComponent(token)}`;
}
