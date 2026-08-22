/**
 * Token persistence for the auth strategy documented in api-client.ts.
 *
 * Deliberately just `localStorage`, guarded for SSR — no cookies, no
 * server-side session. See api-client.ts's module docstring for why.
 * Kept in its own file (rather than inline in api-client.ts) so
 * anything that only needs "am I logged in" / "log out" doesn't have to
 * pull in the whole fetch-wrapper module.
 */

const ACCESS_TOKEN_KEY = "psf.access_token";
const REFRESH_TOKEN_KEY = "psf.refresh_token";

function isBrowser(): boolean {
  return typeof window !== "undefined";
}

export function getAccessToken(): string | null {
  if (!isBrowser()) return null;
  return window.localStorage.getItem(ACCESS_TOKEN_KEY);
}

export function getRefreshToken(): string | null {
  if (!isBrowser()) return null;
  return window.localStorage.getItem(REFRESH_TOKEN_KEY);
}

export function setTokens(accessToken: string, refreshToken?: string): void {
  if (!isBrowser()) return;
  window.localStorage.setItem(ACCESS_TOKEN_KEY, accessToken);
  if (refreshToken !== undefined) {
    window.localStorage.setItem(REFRESH_TOKEN_KEY, refreshToken);
  }
}

export function clearTokens(): void {
  if (!isBrowser()) return;
  window.localStorage.removeItem(ACCESS_TOKEN_KEY);
  window.localStorage.removeItem(REFRESH_TOKEN_KEY);
}

export function isAuthenticated(): boolean {
  return getAccessToken() !== null;
}

/**
 * Best-effort claim extraction from the access token's JWT payload —
 * `sub` (email), `type`, `iat`/`exp`, per app/auth/security.py's
 * `_create_token`. Notably no `role` claim; the backend doesn't embed
 * one, so the user's role has to come from GET /api/v1/auth/me instead
 * (see AuthProvider in components/auth-provider.tsx). Only useful for
 * optimistic checks (e.g. "is there a token at all, is it expired") —
 * never an authorization decision, the backend re-checks on every
 * request via require_role().
 */
export function decodeAccessToken(): { sub: string; type: string; exp: number } | null {
  const token = getAccessToken();
  if (!token) return null;

  try {
    const payload = token.split(".")[1];
    // atob expects base64, JWTs use base64url — normalize before decoding.
    const base64 = payload.replace(/-/g, "+").replace(/_/g, "/");
    return JSON.parse(atob(base64));
  } catch {
    return null;
  }
}
