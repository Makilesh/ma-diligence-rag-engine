/**
 * HTTP client for the M&A Due Diligence API.
 *
 * Every function here is a thin transport wrapper. No pipeline decision is made
 * on this side — refusal, confidence and validation status all arrive from the
 * server already decided, and the UI only renders them.
 */

import type {
  BudgetStatus,
  Deal,
  DocumentRecord,
  QueryResponse,
  RiskSignal,
} from "./types";

export const API_BASE = (
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
).replace(/\/+$/, "");

const API = `${API_BASE}/api/v1`;

/**
 * GETs a JSON endpoint, returning a fallback on any failure.
 *
 * Panels are independent: the sources rail must still render when the risk
 * endpoint is down, so transport errors degrade to empty rather than throwing
 * into the render tree.
 */
async function getJson<T>(path: string, fallback: T, timeoutMs = 10_000): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${API}${path}`, { signal: controller.signal });
    if (!res.ok) return fallback;
    return (await res.json()) as T;
  } catch {
    return fallback;
  } finally {
    clearTimeout(timer);
  }
}

export const listDeals = () => getJson<Deal[]>("/deals", []);

export const listDocuments = (dealId: string) =>
  getJson<DocumentRecord[]>(`/deals/${encodeURIComponent(dealId)}/documents`, []);

export const listRiskSignals = (dealId: string) =>
  getJson<RiskSignal[]>(`/deals/${encodeURIComponent(dealId)}/risk-signals`, []);

export const getBudget = () => getJson<BudgetStatus>("/budget", {});

/**
 * Creates a deal.
 *
 * @param dealName Human-readable name.
 * @param isSandbox Mark the deal ephemeral — purged on tab close, swept on TTL.
 */
export async function createDeal(
  dealName: string,
  isSandbox = false,
): Promise<Deal> {
  const res = await fetch(`${API}/deals`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      deal_name: dealName,
      description: isSandbox ? "Temporary sandbox deal" : "",
      is_sandbox: isSandbox,
    }),
  });
  if (!res.ok) throw new Error(`Could not create deal (${res.status})`);
  return res.json();
}

/**
 * Uploads and ingests one document into a deal.
 *
 * @param dealId Target deal.
 * @param file The document.
 */
export async function ingestDocument(dealId: string, file: File) {
  const form = new FormData();
  form.append("deal_id", dealId);
  form.append("file", file);

  const res = await fetch(`${API}/ingest`, { method: "POST", body: form });
  if (!res.ok) {
    throw new Error(`Ingestion failed (${res.status}): ${await res.text()}`);
  }
  return res.json();
}

/**
 * Deletes a sandbox deal and everything indexed under it.
 *
 * Uses `sendBeacon` when the page is unloading, because a normal `fetch` is
 * cancelled the moment the browser tears the document down — which is precisely
 * when this needs to run. The beacon is queued by the browser and delivered
 * independently of the page's lifetime. It can only issue POST, hence the
 * dedicated `/purge` alias on the server.
 *
 * @param dealId Deal to delete.
 * @param unloading True when called from a `pagehide` handler.
 */
export function purgeDeal(dealId: string, unloading = false): void {
  const url = `${API}/deals/${encodeURIComponent(dealId)}/purge`;

  if (unloading && typeof navigator !== "undefined" && navigator.sendBeacon) {
    navigator.sendBeacon(url, new Blob([], { type: "text/plain" }));
    return;
  }

  // Not unloading: a normal request is fine, and `keepalive` covers the case
  // where the tab closes mid-flight anyway.
  void fetch(url, { method: "POST", keepalive: true }).catch(() => {
    // Deliberately silent. The TTL sweeper is the guarantee; this is the
    // fast path, and a user should never see an error for cleanup they did
    // not ask for.
  });
}

/**
 * Server-Sent Event names emitted by `/query/stream`.
 *
 * `token` frames carry the answer as the model writes it — an unverified
 * draft; `answer_reset` tells the client to discard that draft (the model
 * stream broke and another model is taking over, or a failed validation is
 * being revised). The checked answer always arrives in `result`.
 */
export type StreamEvent =
  | { event: "start"; data: { session_id: string; planned_stages: unknown[] } }
  | { event: "stage"; data: Record<string, unknown> }
  | { event: "token"; data: { text: string } }
  | { event: "answer_reset"; data: { reason: string } }
  | { event: "result"; data: QueryResponse }
  | { event: "error"; data: { detail: string } };

/**
 * Turns a failed query response into a message a visitor can act on.
 *
 * 429 is the per-visitor rate limit or the daily cap; 503 with Retry-After is
 * the server's concurrency cap. Both are "wait and retry", not a fault, so they
 * say how long to wait instead of surfacing a bare status code. The request id
 * is kept for anything else, so a report can be matched to server logs.
 */
async function describeQueryFailure(res: Response): Promise<string> {
  const retryAfter = Number(res.headers.get("Retry-After"));
  const wait =
    Number.isFinite(retryAfter) && retryAfter > 0
      ? `in about ${Math.ceil(retryAfter)} second${Math.ceil(retryAfter) === 1 ? "" : "s"}`
      : "in a little while";

  if (res.status === 429) {
    return `You've reached the query limit for now. Please try again ${wait}.`;
  }
  if (res.status === 503) {
    return `The engine is busy answering other questions. Please try again ${wait}.`;
  }

  let detail = "";
  try {
    const body = await res.json();
    if (body && typeof body.detail === "string") detail = body.detail;
  } catch {
    // Non-JSON error body — the status line is all there is.
  }
  const requestId = res.headers.get("X-Request-ID");
  return [
    `Query failed (${res.status})${detail ? `: ${detail}` : ""}`,
    requestId ? `Request ID ${requestId}` : "",
  ]
    .filter(Boolean)
    .join(" · ");
}

export interface StreamQueryArgs {
  query: string;
  dealId: string;
  includePii: boolean;
  signal?: AbortSignal;
  onEvent: (event: StreamEvent) => void;
}

/**
 * Runs a query against the streaming endpoint, invoking `onEvent` per frame.
 *
 * Uses `fetch` + a ReadableStream reader rather than `EventSource`, for two
 * reasons: EventSource is GET-only (the query would have to go in the URL) and
 * cannot be aborted cleanly mid-flight. The cost is parsing SSE framing by hand,
 * which is the loop below.
 *
 * @param args Query parameters and the per-event callback.
 */
export async function streamQuery({
  query,
  dealId,
  includePii,
  signal,
  onEvent,
}: StreamQueryArgs): Promise<void> {
  const res = await fetch(`${API}/query/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      deal_id: dealId,
      include_pii: includePii,
    }),
    signal,
  });

  if (!res.ok || !res.body) {
    throw new Error(
      res.ok ? "The engine returned an empty stream." : await describeQueryFailure(res),
    );
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by a blank line. A chunk boundary can fall
    // anywhere, so only complete frames are consumed and the remainder stays
    // buffered for the next read.
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";

    for (const frame of frames) {
      if (!frame.trim()) continue;

      let name = "message";
      const dataLines: string[] = [];

      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) name = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
      }

      if (!dataLines.length) continue;

      try {
        onEvent({
          event: name,
          data: JSON.parse(dataLines.join("\n")),
        } as StreamEvent);
      } catch {
        // A frame that will not parse is not worth tearing the stream down for —
        // the remaining events, including the final result, are still coming.
      }
    }
  }
}

/** Non-streaming fallback, used when the stream endpoint is unavailable. */
export async function runQuery(
  query: string,
  dealId: string,
  includePii: boolean,
): Promise<QueryResponse> {
  const res = await fetch(`${API}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, deal_id: dealId, include_pii: includePii }),
  });
  if (!res.ok) throw new Error(await describeQueryFailure(res));
  return res.json();
}
