"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

import { RequireRole } from "@/components/require-role";
import { buildLiveDecisionsWsUrl } from "@/lib/api-client";
import type { LiveDecisionEvent, PolicyAction } from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

const MAX_EVENTS = 200;

const ACTION_BADGE_VARIANT: Record<PolicyAction, "destructive" | "secondary" | "default"> = {
  BLOCK: "destructive",
  SAFE_REWRITE: "secondary",
  PASS: "default",
};

type ConnectionState = "connecting" | "open" | "closed" | "error";

const CONNECTION_LABEL: Record<ConnectionState, string> = {
  connecting: "Connecting…",
  open: "Live",
  closed: "Reconnecting…",
  error: "Connection error",
};

const CONNECTION_VARIANT: Record<ConnectionState, "default" | "secondary" | "destructive"> = {
  connecting: "secondary",
  open: "default",
  closed: "secondary",
  error: "destructive",
};

export default function DashboardPage() {
  return (
    <RequireRole roles={["admin", "analyst"]}>
      <DashboardContent />
    </RequireRole>
  );
}

function DashboardContent() {
  const [events, setEvents] = useState<LiveDecisionEvent[]>([]);
  const [connection, setConnection] = useState<ConnectionState>("connecting");

  useEffect(() => {
    let socket: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let cancelled = false;

    function connect() {
      const url = buildLiveDecisionsWsUrl();
      if (!url) {
        // No access token to connect with — RequireRole already
        // guarantees a signed-in user by the time this component
        // renders, so this is only reachable in the brief window
        // before AuthProvider's initial check resolves.
        setConnection("error");
        return;
      }

      setConnection("connecting");
      socket = new WebSocket(url);

      socket.onopen = () => {
        if (!cancelled) setConnection("open");
      };

      socket.onmessage = (event) => {
        try {
          const parsed = JSON.parse(event.data) as LiveDecisionEvent;
          setEvents((prev) => [parsed, ...prev].slice(0, MAX_EVENTS));
        } catch {
          // Malformed frame — drop it rather than crash the feed.
        }
      };

      socket.onclose = () => {
        if (cancelled) return;
        setConnection("closed");
        reconnectTimer = setTimeout(connect, 3000);
      };

      socket.onerror = () => {
        if (!cancelled) setConnection("error");
      };
    }

    connect();

    return () => {
      cancelled = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      socket?.close();
    };
  }, []);

  return (
    <div className="mx-auto flex w-full max-w-5xl flex-1 flex-col gap-6 p-6">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Live Monitoring</h1>
          <p className="text-muted-foreground text-sm">
            Real-time feed of pipeline decisions across every session.
          </p>
        </div>
        <Badge variant={CONNECTION_VARIANT[connection]}>{CONNECTION_LABEL[connection]}</Badge>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Recent decisions</CardTitle>
          <CardDescription>
            Showing the last {events.length} of up to {MAX_EVENTS} events received this session.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {events.length === 0 ? (
            <p className="text-muted-foreground text-sm">
              No decisions yet — submit a prompt in the Playground to see it appear here live.
            </p>
          ) : (
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Time</TableHead>
                    <TableHead>Session</TableHead>
                    <TableHead>Modality</TableHead>
                    <TableHead>Action</TableHead>
                    <TableHead>Matched rule</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {events.map((event) => (
                    <TableRow key={event.id}>
                      <TableCell className="text-muted-foreground whitespace-nowrap">
                        {new Date(event.created_at).toLocaleTimeString()}
                      </TableCell>
                      <TableCell className="max-w-40 truncate font-mono text-xs">
                        {event.session_id}
                      </TableCell>
                      <TableCell>{event.input_modality}</TableCell>
                      <TableCell>
                        <Badge variant={ACTION_BADGE_VARIANT[event.policy_action]}>
                          {event.policy_action}
                        </Badge>
                      </TableCell>
                      <TableCell>
                        <Link
                          href={`/logs/${event.id}`}
                          className="underline underline-offset-4"
                        >
                          {event.matched_rule}
                        </Link>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
