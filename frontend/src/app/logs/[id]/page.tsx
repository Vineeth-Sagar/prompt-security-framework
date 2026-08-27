"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";

import { RequireAuth } from "@/components/require-auth";
import { useAuth } from "@/components/auth-provider";
import { ApiError, getLog } from "@/lib/api-client";
import type { DecisionLogPublic, DriftBreakdown, PolicyAction } from "@/lib/types";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";

const ACTION_BADGE_VARIANT: Record<PolicyAction, "destructive" | "secondary" | "default"> = {
  BLOCK: "destructive",
  SAFE_REWRITE: "secondary",
  PASS: "default",
};

// Matches app/config.py's DRIFT_THRESHOLD default. The API doesn't
// return the threshold actually in effect for a given decision, so
// this reference marker is best-effort — accurate for a default
// deployment, not guaranteed if DRIFT_THRESHOLD was overridden.
const DEFAULT_DRIFT_THRESHOLD = 0.8;

const DRIFT_SIGNALS: { key: keyof DriftBreakdown; label: string }[] = [
  { key: "semantic", label: "Semantic drift" },
  { key: "role_escalation", label: "Role escalation" },
  { key: "topic_entropy", label: "Topic entropy" },
  { key: "window_role_escalation", label: "Window role escalation" },
  { key: "drift_trend", label: "Drift trend" },
];

// Matches app/pipeline.py's actual stage order.
const STAGE_ORDER = [
  "input_layer",
  "preprocessing",
  "context_buffer_read",
  "swcsa",
  "ifsr",
  "policy",
  "llm_gateway",
  "output_governance",
  "context_buffer_write",
];

export default function LogDetailPage() {
  // Every authenticated user can reach this page — the backend already
  // 404s a non-admin fetching a log id they don't own (app/api/routes/
  // logs.py's get_log), so the actual access boundary is enforced
  // server-side, not by gating the route to admin/analyst here. That
  // also matters because the Playground's "View full trace" link points
  // here, and viewers use the Playground too.
  return (
    <RequireAuth>
      <LogDetailContent />
    </RequireAuth>
  );
}

function LogDetailContent() {
  const { user } = useAuth();
  const isAdmin = user?.role === "admin";
  const params = useParams<{ id: string }>();
  const [log, setLog] = useState<DecisionLogPublic | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    async function fetchLog() {
      const id = Number(params.id);
      if (!Number.isFinite(id)) {
        if (!cancelled) {
          setError("Invalid log id.");
          setLoading(false);
        }
        return;
      }
      try {
        const data = await getLog(id);
        if (!cancelled) setLog(data);
      } catch (err) {
        if (cancelled) return;
        setError(
          err instanceof ApiError ? String(err.detail ?? err.message) : "Could not reach the server."
        );
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    void fetchLog();
    return () => {
      cancelled = true;
    };
  }, [params.id]);

  return (
    <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-6 p-6">
      <div>
        <Link href="/logs" className="text-muted-foreground text-sm underline underline-offset-4">
          ← All logs
        </Link>
        <h1 className="mt-1 text-2xl font-semibold tracking-tight">
          Decision #{params.id}
        </h1>
      </div>

      {loading && (
        <div className="flex flex-col gap-4">
          <Skeleton className="h-24 w-full" />
          <Skeleton className="h-48 w-full" />
          <Skeleton className="h-32 w-full" />
        </div>
      )}

      {error && (
        <Alert variant="destructive">
          <AlertTitle>Couldn&apos;t load this decision</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {log && (
        <>
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between gap-4">
                <CardTitle>Overview</CardTitle>
                <Badge variant={ACTION_BADGE_VARIANT[log.policy_action]}>
                  {log.policy_action}
                </Badge>
              </div>
              <CardDescription>
                Session <code>{log.session_id}</code> · {log.input_modality} ·{" "}
                {new Date(log.created_at).toLocaleString()}
                {isAdmin && (
                  <>
                    {" "}
                    · {log.user_email ?? <span className="italic">unattributed</span>}
                  </>
                )}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-sm">
                Matched rule: <code>{log.matched_rule}</code>
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Drift breakdown</CardTitle>
              <CardDescription>
                Per-signal drift scores (0–1) that fed the weighted aggregate. Reference line
                at {DEFAULT_DRIFT_THRESHOLD} marks the default block threshold.
              </CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-4">
              <MeterRow
                label="Aggregate"
                value={log.drift_breakdown.aggregate}
                threshold={DEFAULT_DRIFT_THRESHOLD}
                emphasized
              />
              <Separator />
              {DRIFT_SIGNALS.map(({ key, label }) => (
                <MeterRow key={key} label={label} value={log.drift_breakdown[key] as number} />
              ))}
              {log.drift_breakdown.matched_patterns.length > 0 && (
                <div className="flex flex-wrap gap-2 pt-2">
                  {log.drift_breakdown.matched_patterns.map((pattern) => (
                    <Badge key={pattern} variant="secondary">
                      {pattern}
                    </Badge>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>IFS-R fragments</CardTitle>
              <CardDescription>
                {log.ifsr_result.blocked
                  ? "Every fragment was malicious, or nothing usable survived reconstruction."
                  : log.ifsr_result.modified
                    ? "One or more fragments were dropped; the rest were rebuilt into the prompt actually forwarded."
                    : "Every fragment was already safe or suspicious — nothing needed to change."}
              </CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-4">
              {log.ifsr_result.removed.length > 0 && (
                <FragmentList
                  title="Removed (malicious)"
                  fragments={log.ifsr_result.removed}
                  variant="destructive"
                />
              )}
              {log.ifsr_result.suspicious.length > 0 && (
                <FragmentList
                  title="Suspicious (kept)"
                  fragments={log.ifsr_result.suspicious}
                  variant="secondary"
                />
              )}
              {log.ifsr_result.removed.length === 0 && log.ifsr_result.suspicious.length === 0 && (
                <p className="text-muted-foreground text-sm">
                  No suspicious or removed fragments — every fragment was classified safe.
                </p>
              )}
              {log.ifsr_result.safe_text && (
                <div className="flex flex-col gap-2">
                  <span className="text-sm font-medium">Reconstructed prompt</span>
                  <p className="bg-muted rounded-md p-3 text-sm whitespace-pre-wrap">
                    {log.ifsr_result.safe_text}
                  </p>
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>PII found</CardTitle>
            </CardHeader>
            <CardContent>
              {log.pii_found.length === 0 ? (
                <p className="text-muted-foreground text-sm">No PII detected in the response.</p>
              ) : (
                <ul className="flex flex-col gap-1 text-sm">
                  {log.pii_found.map((match, index) => (
                    <li key={index} className="flex items-center gap-2">
                      <Badge variant="secondary">{match.type}</Badge>
                      <span className="text-muted-foreground">
                        chars {match.span[0]}–{match.span[1]}
                      </span>
                    </li>
                  ))}
                </ul>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Per-stage latency</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col gap-3">
              {(() => {
                const max = Math.max(...Object.values(log.latency_ms_per_stage), 1);
                return STAGE_ORDER.filter((stage) => stage in log.latency_ms_per_stage).map(
                  (stage) => (
                    <LatencyRow
                      key={stage}
                      label={stage}
                      ms={log.latency_ms_per_stage[stage]}
                      max={max}
                    />
                  )
                );
              })()}
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}

function MeterRow({
  label,
  value,
  threshold,
  emphasized,
}: {
  label: string;
  value: number;
  threshold?: number;
  emphasized?: boolean;
}) {
  const pct = Math.min(100, Math.max(0, value * 100));
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between text-sm">
        <span className={emphasized ? "font-medium" : "text-muted-foreground"}>{label}</span>
        <span className="font-mono">{value.toFixed(2)}</span>
      </div>
      <div className="bg-muted relative h-2 w-full overflow-hidden rounded-full">
        <div
          className={emphasized ? "bg-destructive h-full" : "bg-foreground/70 h-full"}
          style={{ width: `${pct}%` }}
        />
        {threshold !== undefined && (
          <div
            className="bg-destructive absolute top-0 h-full w-px"
            style={{ left: `${threshold * 100}%` }}
            title={`Default threshold: ${threshold}`}
          />
        )}
      </div>
    </div>
  );
}

function LatencyRow({ label, ms, max }: { label: string; ms: number; max: number }) {
  const pct = Math.min(100, (ms / max) * 100);
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between text-sm">
        <span className="text-muted-foreground font-mono">{label}</span>
        <span className="font-mono">{ms.toFixed(1)} ms</span>
      </div>
      <div className="bg-muted h-2 w-full overflow-hidden rounded-full">
        <div className="bg-foreground/70 h-full" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function FragmentList({
  title,
  fragments,
  variant,
}: {
  title: string;
  fragments: string[];
  variant: "destructive" | "secondary";
}) {
  return (
    <div className="flex flex-col gap-2">
      <span className="text-sm font-medium">
        {title} ({fragments.length})
      </span>
      <ul className="flex flex-col gap-1">
        {fragments.map((fragment, index) => (
          <li key={index}>
            <Badge variant={variant} className="h-auto max-w-full font-normal whitespace-normal">
              {fragment}
            </Badge>
          </li>
        ))}
      </ul>
    </div>
  );
}
