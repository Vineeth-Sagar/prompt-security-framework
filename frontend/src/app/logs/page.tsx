"use client";

import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import Link from "next/link";

import { RequireAuth } from "@/components/require-auth";
import { useAuth } from "@/components/auth-provider";
import { ApiError, listLogs, listUsers } from "@/lib/api-client";
import type { DecisionLogPublic, PolicyAction, UserPublic } from "@/lib/types";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

const ACTION_BADGE_VARIANT: Record<PolicyAction, "destructive" | "secondary" | "default"> = {
  BLOCK: "destructive",
  SAFE_REWRITE: "secondary",
  PASS: "default",
};

const PAGE_SIZE = 25;

interface Filters {
  sessionId: string;
  action: PolicyAction | "any";
  userId: string; // user id as string, or "all" (admin-only filter)
  startDate: string; // yyyy-mm-dd, or ""
  endDate: string; // yyyy-mm-dd, or ""
}

const EMPTY_FILTERS: Filters = {
  sessionId: "",
  action: "any",
  userId: "all",
  startDate: "",
  endDate: "",
};

export default function LogsPage() {
  // Every authenticated user can reach this page now; the backend scopes
  // what they actually see (a viewer/analyst gets only their own
  // decisions, an admin gets everyone's). So the gate is just "logged
  // in", not a role check — the old admin/analyst RequireRole would have
  // wrongly kept viewers out of their own audit trail.
  return (
    <RequireAuth>
      <LogsContent />
    </RequireAuth>
  );
}

function LogsContent() {
  const { user } = useAuth();
  const isAdmin = user?.role === "admin";

  const [pendingFilters, setPendingFilters] = useState<Filters>(EMPTY_FILTERS);
  const [appliedFilters, setAppliedFilters] = useState<Filters>(EMPTY_FILTERS);
  const [offset, setOffset] = useState(0);
  const [items, setItems] = useState<DecisionLogPublic[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  // Populated only for admins — the "filter by user" dropdown. Non-admins
  // never see it (and the backend would ignore user_id from them anyway).
  const [users, setUsers] = useState<UserPublic[]>([]);

  useEffect(() => {
    if (!isAdmin) return;
    let cancelled = false;
    void (async () => {
      try {
        const result = await listUsers();
        if (!cancelled) setUsers(result);
      } catch {
        // A failed user-list only costs the admin the convenience
        // dropdown; the log table itself still loads. Leave it empty
        // rather than surfacing an error over the whole page.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [isAdmin]);

  useEffect(() => {
    let cancelled = false;

    async function fetchLogs() {
      setLoading(true);
      setError(null);
      try {
        const result = await listLogs({
          session_id: appliedFilters.sessionId || undefined,
          action: appliedFilters.action === "any" ? undefined : appliedFilters.action,
          user_id:
            appliedFilters.userId === "all" ? undefined : Number(appliedFilters.userId),
          start_date: appliedFilters.startDate || undefined,
          end_date: appliedFilters.endDate || undefined,
          limit: PAGE_SIZE,
          offset,
        });
        if (!cancelled) {
          setItems(result.items);
          setTotal(result.total);
        }
      } catch (err) {
        if (cancelled) return;
        setError(
          err instanceof ApiError ? String(err.detail ?? err.message) : "Could not reach the server."
        );
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    void fetchLogs();
    return () => {
      cancelled = true;
    };
  }, [appliedFilters, offset]);

  function handleApply(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setOffset(0);
    setAppliedFilters(pendingFilters);
  }

  function handleClear() {
    setPendingFilters(EMPTY_FILTERS);
    setAppliedFilters(EMPTY_FILTERS);
    setOffset(0);
  }

  const rangeStart = total === 0 ? 0 : offset + 1;
  const rangeEnd = Math.min(offset + items.length, total);

  return (
    <div className="mx-auto flex w-full max-w-5xl flex-1 flex-col gap-6 p-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Audit Log</h1>
        <p className="text-muted-foreground text-sm">
          {isAdmin
            ? "Search and filter every decision the pipeline has made, across all users."
            : "Search and filter the decisions you have run."}
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Filters</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleApply} className="flex flex-wrap items-end gap-4">
            <div className="flex flex-col gap-2">
              <Label htmlFor="filter-session">Session ID</Label>
              <Input
                id="filter-session"
                placeholder="exact match"
                value={pendingFilters.sessionId}
                onChange={(event) =>
                  setPendingFilters((prev) => ({ ...prev, sessionId: event.target.value }))
                }
                className="w-48"
              />
            </div>

            <div className="flex flex-col gap-2">
              <Label htmlFor="filter-action">Action</Label>
              <Select
                value={pendingFilters.action}
                onValueChange={(value) =>
                  setPendingFilters((prev) => ({ ...prev, action: value as Filters["action"] }))
                }
              >
                <SelectTrigger id="filter-action" className="w-40">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="any">Any</SelectItem>
                  <SelectItem value="BLOCK">BLOCK</SelectItem>
                  <SelectItem value="SAFE_REWRITE">SAFE_REWRITE</SelectItem>
                  <SelectItem value="PASS">PASS</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {isAdmin && (
              <div className="flex flex-col gap-2">
                <Label htmlFor="filter-user">User</Label>
                <Select
                  value={pendingFilters.userId}
                  onValueChange={(value) =>
                    setPendingFilters((prev) => ({ ...prev, userId: value ?? "all" }))
                  }
                >
                  <SelectTrigger id="filter-user" className="w-56">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All users</SelectItem>
                    {users.map((u) => (
                      <SelectItem key={u.id} value={String(u.id)}>
                        {u.email}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            )}

            <div className="flex flex-col gap-2">
              <Label htmlFor="filter-start">From</Label>
              <Input
                id="filter-start"
                type="date"
                value={pendingFilters.startDate}
                onChange={(event) =>
                  setPendingFilters((prev) => ({ ...prev, startDate: event.target.value }))
                }
                className="w-40"
              />
            </div>

            <div className="flex flex-col gap-2">
              <Label htmlFor="filter-end">To</Label>
              <Input
                id="filter-end"
                type="date"
                value={pendingFilters.endDate}
                onChange={(event) =>
                  setPendingFilters((prev) => ({ ...prev, endDate: event.target.value }))
                }
                className="w-40"
              />
            </div>

            <div className="flex gap-2">
              <Button type="submit">Apply</Button>
              <Button type="button" variant="outline" onClick={handleClear}>
                Clear
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Decisions</CardTitle>
          <CardDescription>
            {total > 0
              ? `Showing ${rangeStart}–${rangeEnd} of ${total}`
              : loading
                ? "Loading…"
                : "No decisions match these filters."}
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          {error && (
            <Alert variant="destructive">
              <AlertTitle>Couldn&apos;t load logs</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {loading ? (
            <Skeleton className="h-64 w-full" />
          ) : items.length > 0 ? (
            <>
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Time</TableHead>
                      {isAdmin && <TableHead>User</TableHead>}
                      <TableHead>Session</TableHead>
                      <TableHead>Modality</TableHead>
                      <TableHead>Action</TableHead>
                      <TableHead>Matched rule</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {items.map((log) => (
                      <TableRow key={log.id}>
                        <TableCell className="text-muted-foreground whitespace-nowrap">
                          {new Date(log.created_at).toLocaleString()}
                        </TableCell>
                        {isAdmin && (
                          <TableCell className="max-w-40 truncate text-xs">
                            {log.user_email ?? (
                              <span className="text-muted-foreground italic">unattributed</span>
                            )}
                          </TableCell>
                        )}
                        <TableCell className="max-w-40 truncate font-mono text-xs">
                          {log.session_id}
                        </TableCell>
                        <TableCell>{log.input_modality}</TableCell>
                        <TableCell>
                          <Badge variant={ACTION_BADGE_VARIANT[log.policy_action]}>
                            {log.policy_action}
                          </Badge>
                        </TableCell>
                        <TableCell>
                          <Link
                            href={`/logs/${log.id}`}
                            className="underline underline-offset-4"
                          >
                            {log.matched_rule}
                          </Link>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>

              <div className="flex items-center justify-end gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  disabled={offset === 0}
                  onClick={() => setOffset((prev) => Math.max(0, prev - PAGE_SIZE))}
                >
                  Previous
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  disabled={offset + PAGE_SIZE >= total}
                  onClick={() => setOffset((prev) => prev + PAGE_SIZE)}
                >
                  Next
                </Button>
              </div>
            </>
          ) : (
            !error && (
              <p className="text-muted-foreground text-sm">No decisions match these filters.</p>
            )
          )}
        </CardContent>
      </Card>
    </div>
  );
}
