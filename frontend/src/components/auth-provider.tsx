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
  // Lazy-initialized from localStorage so the no-token case starts
  // already resolved (loading=false, user=null) without needing a
  // setState call inside the mount effect below — only the "there's a
  // token, go verify it" path actually needs to run asynchronously.
  const [loading, setLoading] = useState<boolean>(() => hasStoredToken());

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
      if (!hasStoredToken()) return;
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
