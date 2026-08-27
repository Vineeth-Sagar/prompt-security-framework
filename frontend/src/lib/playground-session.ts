/**
 * Playground session identity — persistence and human-readable naming.
 *
 * ## Why this module exists
 *
 * The Playground originally derived its session id from React's
 * `useId()`, producing values like `playground-_r_1k_`. That was chosen
 * to fix a *hydration* bug (a random id differs between the server
 * prerender and the client's first render, so React discards the tree),
 * and it did fix that — but `useId()` is only stable *within one mount*.
 * Navigating to /logs and back remounts the page, React hands out the
 * next id in tree order, and the Playground silently starts a brand-new
 * session: the drift/context history the user was deliberately building
 * up is orphaned, with no indication anything changed.
 *
 * Worse, `_r_1k_` is a React internal counter, not an identifier meant
 * for humans. Sessions accumulated in the decision log as
 * `playground-_r_0_`, `playground-_r_n_`, `playground-_r_1f_`,
 * `playground-_r_1k_` — impossible to tell apart when reviewing the
 * audit trail, and impossible to retype correctly from memory.
 *
 * ## What this does instead
 *
 * Session ids are generated as `adjective-noun-NNN` (e.g.
 * `calm-falcon-317`) — pronounceable, visually distinct from each other
 * at a glance, and short enough to retype. The current session and a
 * list of recent ones are persisted in `localStorage`, so leaving the
 * Playground and coming back resumes exactly where you were, and
 * switching sessions is an explicit choice rather than an accident of
 * component remounting.
 *
 * Persistence is `localStorage` (not `sessionStorage`) deliberately:
 * the point is to survive navigation *and* a page reload/browser
 * restart, which is when losing your session hurt most. Same
 * SSR-guard convention as auth-storage.ts — every accessor returns a
 * safe default when `window` is undefined, so this module is
 * importable from server-prerendered code without blowing up. Callers
 * must still avoid *reading* it during render (see the Playground
 * page's mount effect) or the hydration bug above comes back in a new
 * form.
 */

const CURRENT_SESSION_KEY = "psf.playground.current_session";
const SESSIONS_KEY = "psf.playground.sessions";

/** Cap on remembered sessions — enough to switch back to recent work,
 * bounded so localStorage can't grow without limit across months of use. */
const MAX_REMEMBERED_SESSIONS = 20;

export interface PlaygroundSession {
  id: string;
  createdAt: string;
  lastUsedAt: string;
  /** Submissions made from this browser under this id. Not authoritative
   * (the backend's decision log is) — just enough to tell a session you
   * actually used from one you created and abandoned. */
  turnCount: number;
}

// Short, common, unambiguous words — chosen to stay readable and
// distinct when several sessions sit next to each other in a dropdown.
// Deliberately no words that could read as a policy verdict
// ("blocked", "safe", "clean"), which would be confusing next to the
// pipeline's actual BLOCK/SAFE_REWRITE/PASS labels.
const ADJECTIVES = [
  "amber", "brisk", "calm", "clever", "cosmic", "crisp", "eager", "fair",
  "gentle", "jolly", "keen", "lively", "lucid", "mellow", "nimble", "noble",
  "polar", "quiet", "rapid", "royal", "sharp", "solar", "spry", "steady",
  "sunny", "swift", "tidy", "vivid", "warm", "witty",
];

const NOUNS = [
  "otter", "falcon", "cedar", "comet", "delta", "ember", "fern", "finch",
  "harbor", "heron", "ibex", "juniper", "kestrel", "lark", "lotus", "maple",
  "meadow", "onyx", "opal", "pine", "quartz", "raven", "river", "sable",
  "sparrow", "summit", "thistle", "willow", "wren", "zephyr",
];

function isBrowser(): boolean {
  return typeof window !== "undefined";
}

function pick<T>(items: T[]): T {
  return items[Math.floor(Math.random() * items.length)];
}

/**
 * A fresh `adjective-noun-NNN` id.
 *
 * ~30 x 30 x 900 = 810,000 combinations, and collisions only matter
 * within one browser's remembered list, which is capped at 20 — so the
 * 3-digit suffix is there to disambiguate a repeated word pair for a
 * human reader, not to provide cryptographic uniqueness. Session ids
 * are not a security boundary: the backend authenticates the *user* on
 * every request, and a session id only selects which drift/context
 * window a prompt joins.
 */
export function generateSessionId(): string {
  const suffix = String(Math.floor(Math.random() * 900) + 100);
  return `${pick(ADJECTIVES)}-${pick(NOUNS)}-${suffix}`;
}

export function loadSessions(): PlaygroundSession[] {
  if (!isBrowser()) return [];
  try {
    const raw = window.localStorage.getItem(SESSIONS_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    // Defensive: this is user-editable storage that may also hold data
    // written by an older version of this module. Drop anything that
    // isn't shaped right rather than rendering `undefined` in the UI.
    return parsed.filter(
      (item): item is PlaygroundSession =>
        typeof item === "object" &&
        item !== null &&
        typeof (item as PlaygroundSession).id === "string"
    );
  } catch {
    return [];
  }
}

function saveSessions(sessions: PlaygroundSession[]): void {
  if (!isBrowser()) return;
  try {
    window.localStorage.setItem(SESSIONS_KEY, JSON.stringify(sessions));
  } catch {
    // Quota exceeded / storage disabled (private windows). Losing
    // session *history* is a degraded experience, not a broken one —
    // the Playground still works with whatever id is in state.
  }
}

export function getCurrentSessionId(): string | null {
  if (!isBrowser()) return null;
  try {
    return window.localStorage.getItem(CURRENT_SESSION_KEY);
  } catch {
    return null;
  }
}

export function setCurrentSessionId(id: string): void {
  if (!isBrowser()) return;
  try {
    window.localStorage.setItem(CURRENT_SESSION_KEY, id);
  } catch {
    // See saveSessions().
  }
}

/**
 * Register `id` as the current session, creating its history entry if
 * it's new. Safe to call with an id the user typed by hand — that's
 * exactly how you resume a session started in another browser, or one
 * read off the audit log.
 */
export function rememberSession(id: string): PlaygroundSession[] {
  const trimmed = id.trim();
  if (!trimmed) return loadSessions();

  const now = new Date().toISOString();
  const existing = loadSessions();
  const match = existing.find((session) => session.id === trimmed);

  const updated: PlaygroundSession = match
    ? { ...match, lastUsedAt: now }
    : { id: trimmed, createdAt: now, lastUsedAt: now, turnCount: 0 };

  const next = [updated, ...existing.filter((session) => session.id !== trimmed)].slice(
    0,
    MAX_REMEMBERED_SESSIONS
  );

  saveSessions(next);
  setCurrentSessionId(trimmed);
  return next;
}

/** Bump `id`'s turn count and recency after a successful submission. */
export function recordSubmission(id: string): PlaygroundSession[] {
  const trimmed = id.trim();
  if (!trimmed) return loadSessions();

  const now = new Date().toISOString();
  const existing = loadSessions();
  const match = existing.find((session) => session.id === trimmed);

  const updated: PlaygroundSession = match
    ? { ...match, lastUsedAt: now, turnCount: match.turnCount + 1 }
    : { id: trimmed, createdAt: now, lastUsedAt: now, turnCount: 1 };

  const next = [updated, ...existing.filter((session) => session.id !== trimmed)].slice(
    0,
    MAX_REMEMBERED_SESSIONS
  );

  saveSessions(next);
  return next;
}

/**
 * The session to open the Playground with: the last one used, or a
 * newly generated one on a first visit. Call from a mount effect, never
 * during render — it touches localStorage, which the server prerender
 * can't see.
 */
export function resolveInitialSessionId(): string {
  const current = getCurrentSessionId();
  if (current && current.trim()) return current;
  return generateSessionId();
}

/** Short relative age ("just now", "12m ago", "3d ago") for the session
 * switcher — an absolute timestamp is more precision than "which of
 * these was I just working in" needs. */
export function formatRelativeTime(iso: string): string {
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "";

  const seconds = Math.max(0, Math.floor((Date.now() - then) / 1000));
  if (seconds < 60) return "just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}
