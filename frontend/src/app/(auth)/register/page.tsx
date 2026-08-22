"use client";

import { useState } from "react";
import type { FormEvent } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";

import { register as apiRegister, ApiError } from "@/lib/api-client";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

/**
 * Registration model (decided here, per Phase 11.2's brief): admin-invited
 * only, with the one exception the backend itself carves out — the very
 * first user ever registered (empty `users` table) bootstraps as admin
 * with no auth required (app/api/routes/auth.py's `register()`). Every
 * registration after that requires an authenticated admin caller, or a
 * 403.
 *
 * So this page is only ever useful in two situations: bootstrapping a
 * fresh deployment's first admin account, or an admin creating a user
 * (Phase 11.7's Admin page will call the same `register()` client
 * function with a role, from an authenticated session). An anonymous
 * visitor hitting this page on a deployment that already has an admin
 * gets the backend's 403 surfaced as-is, with a pointer to ask an
 * admin — there's no self-serve signup flow beyond the bootstrap case.
 */
export default function RegisterPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [createdRole, setCreatedRole] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      const user = await apiRegister(email, password);
      setCreatedRole(user.role);
      if (user.role === "admin") {
        // Bootstrap case — no one was signed in, so send them to log in.
        setTimeout(() => router.push("/login"), 1500);
      }
    } catch (err) {
      if (err instanceof ApiError && err.status === 403) {
        setError(
          "Self-registration is invite-only. Ask an administrator to create your account."
        );
      } else if (err instanceof ApiError) {
        setError(String(err.detail ?? err.message));
      } else {
        setError("Could not reach the server.");
      }
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="flex flex-1 items-center justify-center p-8">
      <Card className="w-full max-w-sm">
        <CardHeader>
          <CardTitle>Create account</CardTitle>
          <CardDescription>
            Only succeeds to bootstrap the first admin on a fresh deployment, or when an
            admin is already signed in.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {createdRole ? (
            <Alert>
              <AlertTitle>Account created</AlertTitle>
              <AlertDescription>
                {createdRole === "admin"
                  ? "Redirecting you to sign in…"
                  : `Account created for ${email}.`}
              </AlertDescription>
            </Alert>
          ) : (
            <form onSubmit={handleSubmit} className="flex flex-col gap-4">
              <div className="flex flex-col gap-2">
                <Label htmlFor="email">Email</Label>
                <Input
                  id="email"
                  type="email"
                  autoComplete="email"
                  required
                  value={email}
                  onChange={(event) => setEmail(event.target.value)}
                />
              </div>
              <div className="flex flex-col gap-2">
                <Label htmlFor="password">Password</Label>
                <Input
                  id="password"
                  type="password"
                  autoComplete="new-password"
                  required
                  minLength={8}
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                />
              </div>
              {error && (
                <Alert variant="destructive">
                  <AlertTitle>Couldn&apos;t register</AlertTitle>
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}
              <Button type="submit" disabled={submitting}>
                {submitting ? "Creating…" : "Create account"}
              </Button>
            </form>
          )}
          <p className="text-muted-foreground mt-4 text-sm">
            Already have an account?{" "}
            <Link href="/login" className="text-foreground underline underline-offset-4">
              Sign in
            </Link>
            .
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
