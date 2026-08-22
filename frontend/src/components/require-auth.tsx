"use client";

import { useEffect } from "react";
import type { ReactNode } from "react";
import { useRouter } from "next/navigation";

import { useAuth } from "@/components/auth-provider";

/**
 * Client-side route guard for pages that need a signed-in user
 * (Playground, Logs, Admin, ...). Renders nothing and redirects to
 * /login once the initial auth check (AuthProvider's GET /auth/me)
 * resolves without a user.
 *
 * This is an optimistic UX nicety, not the security boundary — every
 * protected backend route re-checks auth/role itself
 * (get_current_user / require_role in app/auth/security.py) regardless
 * of what the frontend does. Its only job is avoiding a flash of a page
 * that would immediately 401 against the real API.
 */
export function RequireAuth({ children }: { children: ReactNode }) {
  const { user, loading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!loading && !user) {
      router.replace("/login");
    }
  }, [loading, user, router]);

  if (loading || !user) {
    return null;
  }

  return <>{children}</>;
}
