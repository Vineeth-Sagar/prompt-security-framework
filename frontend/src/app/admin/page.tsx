"use client";

import { useEffect, useState } from "react";
import { toast } from "sonner";

import { useAuth } from "@/components/auth-provider";
import { RequireRole } from "@/components/require-role";
import { ApiError, listUsers, updateUser } from "@/lib/api-client";
import type { UserPublic, UserRole } from "@/lib/types";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
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

const ROLES: UserRole[] = ["admin", "analyst", "viewer"];

export default function AdminPage() {
  return (
    <RequireRole roles={["admin"]}>
      <AdminContent />
    </RequireRole>
  );
}

function AdminContent() {
  const { user: currentUser } = useAuth();
  const [users, setUsers] = useState<UserPublic[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [pendingId, setPendingId] = useState<number | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function fetchUsers() {
      try {
        const data = await listUsers();
        if (!cancelled) setUsers(data);
      } catch (err) {
        if (cancelled) return;
        setError(
          err instanceof ApiError ? String(err.detail ?? err.message) : "Could not reach the server."
        );
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    void fetchUsers();
    return () => {
      cancelled = true;
    };
  }, []);

  async function applyUpdate(id: number, patch: { role?: UserRole; is_active?: boolean }) {
    setPendingId(id);
    try {
      const updated = await updateUser(id, patch);
      setUsers((prev) => prev.map((u) => (u.id === id ? updated : u)));
      toast.success(`Updated ${updated.email}.`);
    } catch (err) {
      // A toast, not the page-level Alert below — this is a transient
      // per-row action failure, not a reason the whole page's content
      // (the user list, which loaded fine) should look broken.
      toast.error(
        err instanceof ApiError ? String(err.detail ?? err.message) : "Update failed."
      );
    } finally {
      setPendingId(null);
    }
  }

  return (
    <div className="mx-auto flex w-full max-w-4xl flex-1 flex-col gap-6 p-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Admin</h1>
        <p className="text-muted-foreground text-sm">
          Manage user roles and access. You can&apos;t change your own account here — ask
          another admin.
        </p>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertTitle>Something went wrong</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <Card>
        <CardHeader>
          <CardTitle>Users</CardTitle>
          <CardDescription>
            {loading ? "Loading…" : `${users.length} user${users.length === 1 ? "" : "s"}`}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {loading ? (
            <Skeleton className="h-48 w-full" />
          ) : users.length === 0 ? (
            <p className="text-muted-foreground text-sm">No users found.</p>
          ) : (
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Email</TableHead>
                    <TableHead>Role</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Joined</TableHead>
                    <TableHead />
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {users.map((u) => {
                    const isSelf = u.id === currentUser?.id;
                    const isPending = pendingId === u.id;
                    return (
                      <TableRow key={u.id}>
                        <TableCell>
                          {u.email}
                          {isSelf && <span className="text-muted-foreground"> (you)</span>}
                        </TableCell>
                        <TableCell>
                          <Select
                            value={u.role}
                            disabled={isSelf || isPending}
                            onValueChange={(value) =>
                              void applyUpdate(u.id, { role: value as UserRole })
                            }
                          >
                            <SelectTrigger className="w-32">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              {ROLES.map((role) => (
                                <SelectItem key={role} value={role}>
                                  {role}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </TableCell>
                        <TableCell>
                          <Badge variant={u.is_active ? "default" : "secondary"}>
                            {u.is_active ? "active" : "deactivated"}
                          </Badge>
                        </TableCell>
                        <TableCell className="text-muted-foreground whitespace-nowrap">
                          {new Date(u.created_at).toLocaleDateString()}
                        </TableCell>
                        <TableCell>
                          <Button
                            variant="outline"
                            size="sm"
                            disabled={isSelf || isPending}
                            onClick={() => void applyUpdate(u.id, { is_active: !u.is_active })}
                          >
                            {u.is_active ? "Deactivate" : "Reactivate"}
                          </Button>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
