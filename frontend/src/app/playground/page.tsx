"use client";

import { useState } from "react";
import type { FormEvent } from "react";
import Link from "next/link";

import { RequireAuth } from "@/components/require-auth";
import { ApiError, runPipeline, runPipelineText } from "@/lib/api-client";
import type { Modality, PipelineResult, PolicyAction } from "@/lib/types";
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
  const [sessionId, setSessionId] = useState<string>(() => `playground-${crypto.randomUUID()}`);
  const [modality, setModality] = useState<Modality>("text");
  const [text, setText] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PipelineResult | null>(null);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);

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
    } catch (err) {
      if (err instanceof ApiError) {
        setError(String(err.detail ?? err.message));
      } else {
        setError("Could not reach the server.");
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
              <Label htmlFor="session-id">Session ID</Label>
              <Input
                id="session-id"
                value={sessionId}
                onChange={(event) => setSessionId(event.target.value)}
              />
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
