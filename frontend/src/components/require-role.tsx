"use client";

import type { ReactNode } from "react";

import { useAuth } from "@/components/auth-provider";
import { RequireAuth } from "@/components/require-auth";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import type { UserRole } from "@/lib/types";

/**
 * Combines RequireAuth (redirect to /login if not signed in) with a
 * role check for pages gated the same way the backend gates them —
 * /dashboard and /logs require admin or analyst (matching
 * app/api/routes/{logs,ws}.py's _ALLOWED_ROLES /
 * _LOG_READER_ROLES), /admin will require admin only. Renders an
 * explanatory message rather than redirecting when the role doesn't
 * match: the user *is* signed in, just not authorized for this one
 * page, which is a different situation from not being signed in at
 * all and deserves a different response than silently bouncing them
 * to the login screen.
 *
 * As with RequireAuth, this is optimistic UI only — the backend
 * re-checks the role on every request regardless (require_role() for
 * HTTP, ws.py's _authenticate() for the WebSocket).
 */
export function RequireRole({ roles, children }: { roles: UserRole[]; children: ReactNode }) {
  return (
    <RequireAuth>
      <RoleCheck roles={roles}>{children}</RoleCheck>
    </RequireAuth>
  );
}

function RoleCheck({ roles, children }: { roles: UserRole[]; children: ReactNode }) {
  const { user } = useAuth();

  // RequireAuth only renders `children` once `user` is resolved and
  // non-null, so this is defensive, not a real null case in practice.
  if (!user || !roles.includes(user.role)) {
    return (
      <div className="mx-auto w-full max-w-md p-6">
        <Alert variant="destructive">
          <AlertTitle>Insufficient permissions</AlertTitle>
          <AlertDescription>
            This page requires the {roles.join(" or ")} role. Your account is a{" "}
            {user?.role ?? "unknown"} role.
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  return <>{children}</>;
}
