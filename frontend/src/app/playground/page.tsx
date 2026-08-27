"use client";

import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import Link from "next/link";

import { RequireAuth } from "@/components/require-auth";
import { ApiError, runPipeline, runPipelineText } from "@/lib/api-client";
import type { Modality, PipelineResult, PolicyAction } from "@/lib/types";
import type { PlaygroundSession } from "@/lib/playground-session";
import {
  formatRelativeTime,
  generateSessionId,
  recordSubmission,
  rememberSession,
  resolveInitialSessionId,
} from "@/lib/playground-session";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";

const ACTION_BADGE_VARIANT: Record<PolicyAction, "destructive" | "secondary" | "default"> = {
  BLOCK: "destructive",
  SAFE_REWRITE: "secondary",
  PASS: "default",
};

const MODALITY_ACCEPT: Record<Modality, string> = {
  text: "",
  image: "image/png,image/jpeg",
  audio: "audio/wav,audio/mpeg",
};

export default function PlaygroundPage() {
  return (
    <RequireAuth>
      <PlaygroundContent />
    </RequireAuth>
  );
}

function PlaygroundContent() {
  // Session id starts as "" — a deterministic value the server
  // prerender and the client's first render agree on — and is resolved
  // to the real persisted id inside the mount effect below. Reading
  // localStorage in a useState initializer would reintroduce the
  // hydration mismatch this page was fixed for once already (server
  // renders one value, client renders another, React discards the
  // tree); see lib/playground-session.ts's docstring for the full
  // history, including why useId() — the previous fix — was itself
  // wrong here.
  const [sessionId, setSessionId] = useState("");
  const [sessions, setSessions] = useState<PlaygroundSession[]>([]);
  const [modality, setModality] = useState<Modality>("text");
  const [text, setText] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PipelineResult | null>(null);

  // Resolve the persisted session on mount (see the sessionId state
  // comment above for why this can't happen during render). Written as
  // an async IIFE whose setState calls all follow an await, matching
  // React's documented resolve-on-mount shape rather than setting state
  // synchronously in the effect body.
  useEffect(() => {
    void (async () => {
      await Promise.resolve();
      const initial = resolveInitialSessionId();
      setSessionId(initial);
      setSessions(rememberSession(initial));
    })();
  }, []);

  // The Select must only ever be handed a value one of its items has,
  // otherwise a hand-typed id that isn't in history renders as a
  // silently empty trigger. "" falls back to the placeholder instead.
  const knownSessionValue = sessions.some((session) => session.id === sessionId)
    ? sessionId
    : "";

  function handleSessionChange(next: string) {
    setSessionId(next);
    setResult(null);
  }

  function handleSessionCommit(next: string) {
    const trimmed = next.trim();
    if (!trimmed) return;
    setSessions(rememberSession(trimmed));
  }

  function handleNewSession() {
    const fresh = generateSessionId();
    setSessionId(fresh);
    setSessions(rememberSession(fresh));
    setResult(null);
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);

    if (!sessionId.trim()) {
      setError("Enter a session id (or start a new session) before submitting.");
      return;
    }

    if (modality !== "text" && !file) {
      setError(`Choose a ${modality} file to upload.`);
      return;
    }

    setSubmitting(true);
    try {
      const response =
        modality === "text"
          ? await runPipelineText(text, sessionId)
          : await runPipeline({ file: file as File, filename: (file as File).name, modality, sessionId });
      setResult(response);
      setSessions(recordSubmission(sessionId));
    } catch (err) {
      if (err instanceof ApiError) {
        setError(String(err.detail ?? err.message));
      } else {
        // Deliberately hedged rather than asserting the server is down.
        // fetch() rejects identically whether the backend is genuinely
        // unreachable or it answered with a response the browser
        // refused to expose (a 500 that lost its CORS headers did
        // exactly this, and the flat "Could not reach the server."
        // this replaces sent debugging in the wrong direction for a
        // while). The prompt may well have been processed and logged.
        setError(
          "The request did not complete. The backend may be unreachable, or it " +
            "returned a response the browser could not read — check the Logs page " +
            "to see whether this prompt was in fact processed, then try again."
        );
      }
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-6 p-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Playground</h1>
        <p className="text-muted-foreground text-sm">
          Submit a prompt through the full pipeline — SWCSA drift scoring, IFS-R
          fragmentation, policy decision, target LLM, output governance — and see the
          decision live.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Submit</CardTitle>
          <CardDescription>
            Prompts submitted with the same session id share drift/context history with each
            other, the same way a real conversation would.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="flex flex-col gap-4">
            <div className="flex flex-col gap-2">
              <div className="flex items-center justify-between gap-2">
                <Label htmlFor="session-id">Session</Label>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={handleNewSession}
                  disabled={submitting}
                >
                  New session
                </Button>
              </div>
              <div className="flex flex-col gap-2 sm:flex-row">
                <Input
                  id="session-id"
                  className="sm:flex-1"
                  value={sessionId}
                  placeholder="Loading session…"
                  onChange={(event) => handleSessionChange(event.target.value)}
                  onBlur={(event) => handleSessionCommit(event.target.value)}
                />
                {sessions.length > 0 && (
                  <Select
                    value={knownSessionValue}
                    onValueChange={(value) => handleSessionChange(value ?? "")}
                  >
                    <SelectTrigger className="w-full sm:w-64" aria-label="Recent sessions">
                      <SelectValue placeholder="Recent sessions" />
                    </SelectTrigger>
                    <SelectContent>
                      {sessions.map((session) => (
                        <SelectItem key={session.id} value={session.id}>
                          {session.id} · {session.turnCount} turn
                          {session.turnCount === 1 ? "" : "s"} ·{" "}
                          {formatRelativeTime(session.lastUsedAt)}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                )}
              </div>
              <p className="text-muted-foreground text-xs">
                Kept across page navigation and reloads — leaving the Playground and coming
                back resumes this session. Switch with the dropdown, or type any session id
                (e.g. one copied from the audit log) to resume it.
              </p>
            </div>

            <div className="flex flex-col gap-2">
              <Label htmlFor="modality">Modality</Label>
              <Select
                value={modality}
                onValueChange={(value) => {
                  setModality(value as Modality);
                  setFile(null);
                }}
              >
                <SelectTrigger id="modality" className="w-full sm:w-48">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="text">Text</SelectItem>
                  <SelectItem value="image">Image</SelectItem>
                  <SelectItem value="audio">Audio</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {modality === "text" ? (
              <div className="flex flex-col gap-2">
                <Label htmlFor="prompt">Prompt</Label>
                <Textarea
                  id="prompt"
                  required
                  rows={5}
                  value={text}
                  onChange={(event) => setText(event.target.value)}
                  placeholder="Type a prompt to send through the pipeline…"
                />
              </div>
            ) : (
              <div className="flex flex-col gap-2">
                <Label htmlFor="file">{modality === "image" ? "Image file" : "Audio file"}</Label>
                <Input
                  id="file"
                  type="file"
                  accept={MODALITY_ACCEPT[modality]}
                  onChange={(event) => setFile(event.target.files?.[0] ?? null)}
                />
              </div>
            )}

            {error && (
              <Alert variant="destructive">
                <AlertTitle>Request failed</AlertTitle>
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            <Button type="submit" disabled={submitting}>
              {submitting ? "Running pipeline…" : "Run through pipeline"}
            </Button>
          </form>
        </CardContent>
      </Card>

      {result && <ResultCard result={result} />}
    </div>
  );
}

function ResultCard({ result }: { result: PipelineResult }) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between gap-4">
          <CardTitle>Decision</CardTitle>
          <Badge variant={ACTION_BADGE_VARIANT[result.policy.action]}>
            {result.policy.action}
          </Badge>
        </div>
        <CardDescription>
          Matched rule: <code>{result.policy.matched_rule}</code> · drift{" "}
          {result.drift.aggregate.toFixed(2)} · {result.total_duration_ms.toFixed(0)} ms
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        {result.policy.action === "BLOCK" ? (
          <Alert variant="destructive">
            <AlertTitle>Blocked</AlertTitle>
            <AlertDescription>{result.rejection_message}</AlertDescription>
          </Alert>
        ) : (
          <div className="flex flex-col gap-2">
            <Label>LLM response</Label>
            <p className="bg-muted rounded-md p-3 text-sm whitespace-pre-wrap">
              {result.final_response_text}
            </p>
          </div>
        )}

        {result.pii_scan && result.pii_scan.found.length > 0 && (
          <Alert>
            <AlertTitle>PII redacted</AlertTitle>
            <AlertDescription>
              {result.pii_scan.found.length} match
              {result.pii_scan.found.length === 1 ? "" : "es"} found and redacted from the
              response ({result.pii_scan.found.map((m) => m.type).join(", ")}).
            </AlertDescription>
          </Alert>
        )}

        <Separator />

        <div className="flex items-center justify-between text-sm">
          <span className="text-muted-foreground">Session: {result.session_id}</span>
          {result.log_id !== null && (
            <Link
              href={`/logs/${result.log_id}`}
              className="text-foreground underline underline-offset-4"
            >
              View full trace →
            </Link>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
