"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";

import { useAuth } from "@/components/auth-provider";
import { Button, buttonVariants } from "@/components/ui/button";

export function NavBar() {
  const { user, loading, logout } = useAuth();
  const router = useRouter();

  function handleSignOut() {
    logout();
    router.push("/login");
  }

  return (
    <header className="border-border border-b">
      <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-3">
        <Link href="/" className="font-semibold tracking-tight">
          Prompt Security Framework
        </Link>
        <div className="flex items-center gap-3 text-sm">
          {loading ? null : user ? (
            <>
              <span className="text-muted-foreground hidden sm:inline">
                {user.email} · {user.role}
              </span>
              <Button variant="outline" size="sm" onClick={handleSignOut}>
                Sign out
              </Button>
            </>
          ) : (
            <Link href="/login" className={buttonVariants({ size: "sm" })}>
              Sign in
            </Link>
          )}
        </div>
      </div>
    </header>
  );
}
