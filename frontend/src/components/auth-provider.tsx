"use client";

import { createContext, useCallback, useContext, useEffect, useState } from "react";
import type { ReactNode } from "react";

import * as api from "@/lib/api-client";
import { isAuthenticated as hasStoredToken } from "@/lib/auth-storage";
import type { UserPublic } from "@/lib/types";

interface AuthContextValue {
  /** null while `loading`, or once resolved: the signed-in user, or
   * null if there isn't one. */
  user: UserPublic | null;
  /** True until the initial GET /auth/me check (using any token already
   * in localStorage) has resolved — lets consumers avoid a flash of
   * "logged out" UI before that check completes. */
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
  /** Re-runs the GET /auth/me check — e.g. after a role change. */
  refresh: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

/**
 * Resolves the current user once on mount from whatever token is
 * already in localStorage (api-client.ts's rawRequest handles the
 * refresh-on-401 dance transparently), and exposes login/logout that
 * keep this context in sync with lib/api-client.ts's token storage.
 *
 * Wraps the whole app in RootLayout so any page/component can call
 * useAuth() without prop-drilling the current user down through
 * layouts.
 */
export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<UserPublic | null>(null);
  // Always starts true, on both the server render and the client's
  // first (hydration) render — an earlier version lazy-initialized
  // this from hasStoredToken() to skip straight to loading=false when
  // there's no token, which seemed like a harmless optimization but
  // isn't: localStorage is client-only, so the server always computes
  // false while the client can compute true, and React requires a
  // component's first client render to match the server-rendered HTML
  // exactly. That mismatch corrupted hydration for every consumer of
  // this value (e.g. nav-bar.tsx renders a completely different DOM
  // shape for loading/signed-out/signed-in), not just this component.
  // Resolved to the real value in the effect below instead, which only
  // ever runs client-side, after hydration has already succeeded.
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const current = await api.me();
      setUser(current);
    } catch {
      // Token missing/expired and refresh (handled inside rawRequest)
      // also failed — treat as signed out rather than surfacing an
      // error on a page the user never actively acted on.
      api.logout();
      setUser(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    // Defined and invoked inline (rather than just calling `refresh`
    // directly) so the setState calls are lexically inside the effect,
    // after an `await` — the pattern React's docs recommend for
    // fetch-on-mount, and what react-hooks/set-state-in-effect expects
    // to see rather than a same-render setState it can't prove is async.
    let ignore = false;

    async function verifyStoredToken() {
      if (!hasStoredToken()) {
        // Still await something before setState, even though there's
        // no real async work to do — keeps this branch's setState
        // provably post-microtask too, not just the token-present one.
        await Promise.resolve();
        if (!ignore) setLoading(false);
        return;
      }
      await refresh();
    }

    void verifyStoredToken().catch(() => {
      if (!ignore) setLoading(false);
    });

    return () => {
      ignore = true;
    };
  }, [refresh]);

  const login = useCallback(async (email: string, password: string) => {
    await api.login(email, password);
    const current = await api.me();
    setUser(current);
  }, []);

  const logout = useCallback(() => {
    api.logout();
    setUser(null);
  }, []);

  return (
    <AuthContext.Provider value={{ user, loading, login, logout, refresh }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error("useAuth() must be called within an <AuthProvider>.");
  }
  return ctx;
}
