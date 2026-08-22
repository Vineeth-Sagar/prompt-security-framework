"use client";

import { useEffect } from "react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

/**
 * Route-level error boundary (Next's `error.tsx` file convention) —
 * catches unexpected render-time exceptions anywhere under this route
 * tree, not API-call failures. Those are already handled per-page with
 * inline Alerts (or a toast, for transient actions — see app/admin) so
 * a failed fetch shows a specific, in-context message instead of
 * replacing the whole page with this generic fallback. This exists for
 * the case those pages don't otherwise handle: an actual bug throwing
 * during render.
 */
export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error(error);
  }, [error]);

  return (
    <div className="flex flex-1 items-center justify-center p-8">
      <Card className="w-full max-w-md">
        <CardHeader>
          <CardTitle>Something went wrong</CardTitle>
          <CardDescription>
            An unexpected error occurred while rendering this page. Retrying might work if it
            was transient.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Button onClick={reset}>Try again</Button>
        </CardContent>
      </Card>
    </div>
  );
}
