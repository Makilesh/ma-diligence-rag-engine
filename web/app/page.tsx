"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { CornerDownLeft, Layers, Sparkles } from "lucide-react";

import AnswerPanel from "@/components/AnswerPanel";
import AskBar from "@/components/AskBar";
import DataRoomPanel from "@/components/DataRoomPanel";
import Masthead from "@/components/Masthead";
import PipelineTimeline from "@/components/PipelineTimeline";
import RefusalPanel from "@/components/RefusalPanel";
import SourcesRail from "@/components/SourcesRail";
import UploadPanel from "@/components/UploadPanel";
import { BRAND } from "@/lib/brand";
import {
  getBudget,
  listDeals,
  listDocuments,
  listRiskSignals,
  streamQuery,
} from "@/lib/api";
import { parseAnswer } from "@/lib/citations";
import { useSandbox } from "@/lib/useSandbox";
import type {
  BudgetStatus,
  Deal,
  DocumentRecord,
  PipelineStage,
  PlannedStage,
  QueryResponse,
  RiskSignal,
} from "@/lib/types";

type Phase = "idle" | "running" | "done";

/**
 * Folds one streamed stage event into the timeline.
 *
 * Three cases, and the conditional nodes are why this is not a simple index
 * bump. The graph's shape is decided at runtime: the financial verifier only
 * runs for financial queries, and the rewriter re-enters retrieval on a loop,
 * so a stage can arrive that the planned list never mentioned, or arrive twice.
 */
function applyStage(
  prev: PipelineStage[],
  incoming: Omit<PipelineStage, "status">,
): PipelineStage[] {
  const next = [...prev];

  // A planned row still waiting for this agent — the ordinary case.
  const slot = next.findIndex(
    (s) => s.agent === incoming.agent && s.status !== "done",
  );

  if (slot !== -1) {
    next[slot] = { ...next[slot], ...incoming, status: "done" };
  } else {
    // Either a conditional node or a second pass through the rewrite loop.
    // Insert it after the last completed row so the timeline reads in the
    // order things actually happened.
    let lastDone = -1;
    next.forEach((s, i) => {
      if (s.status === "done") lastDone = i;
    });
    next.splice(lastDone + 1, 0, { ...incoming, status: "done" });
  }

  // Whatever is next in line is now the one running.
  const upcoming = next.findIndex((s) => s.status !== "done");
  if (upcoming !== -1) {
    next[upcoming] = { ...next[upcoming], status: "running" };
  }

  return next;
}

export default function Home() {
  const [deals, setDeals] = useState<Deal[]>([]);
  const [activeDealId, setActiveDealId] = useState<string | null>(null);
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [risks, setRisks] = useState<RiskSignal[]>([]);
  const [budget, setBudget] = useState<BudgetStatus>({});

  const [phase, setPhase] = useState<Phase>("idle");
  const [question, setQuestion] = useState("");
  const [stages, setStages] = useState<PipelineStage[]>([]);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [highlighted, setHighlighted] = useState<number | null>(null);

  const abortRef = useRef<AbortController | null>(null);

  const refreshDeals = useCallback(async (preferDealId?: string) => {
    const fetched = await listDeals();
    setDeals(fetched);
    setActiveDealId((current) => {
      if (preferDealId) return preferDealId;
      if (current && fetched.some((d) => d.deal_id === current)) return current;
      // Default to whichever deal has the most indexed documents — on a public
      // demo that is the pre-loaded data room, which is what a first-time
      // visitor should land on.
      const best = [...fetched].sort(
        (a, b) => b.document_count - a.document_count,
      )[0];
      return best?.deal_id ?? null;
    });
  }, []);

  const sandbox = useSandbox(
    useCallback(
      (deal: Deal) => {
        void refreshDeals(deal.deal_id);
      },
      [refreshDeals],
    ),
    useCallback(() => {
      // Clear the selection before refetching so `refreshDeals` falls back to
      // the pre-indexed demo deal rather than keeping the purged id.
      setActiveDealId(null);
      void refreshDeals();
    }, [refreshDeals]),
  );

  useEffect(() => {
    void refreshDeals();
    void getBudget().then(setBudget);
  }, [refreshDeals]);

  // Deal-scoped panels. Refetched whenever the deal changes, and after an
  // upload lands (the sandbox hook bumps `deals`, which changes the id).
  useEffect(() => {
    if (!activeDealId) {
      setDocuments([]);
      setRisks([]);
      return;
    }
    void listDocuments(activeDealId).then(setDocuments);
    void listRiskSignals(activeDealId).then(setRisks);
  }, [activeDealId, sandbox.uploading]);

  // Wall clock for the running query. Driven by an interval rather than the
  // stage events, so the counter keeps moving during a long retrieval pass.
  useEffect(() => {
    if (phase !== "running") return;
    const started = performance.now();
    const id = setInterval(() => setElapsed(performance.now() - started), 100);
    return () => clearInterval(id);
  }, [phase]);

  const handleSubmit = useCallback(
    async (query: string, includePii: boolean) => {
      if (!activeDealId) return;

      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      setQuestion(query);
      setPhase("running");
      setResult(null);
      setError(null);
      setStages([]);
      setElapsed(0);
      setHighlighted(null);

      try {
        await streamQuery({
          query,
          dealId: activeDealId,
          includePii,
          signal: controller.signal,
          onEvent: (evt) => {
            if (evt.event === "start") {
              const planned = evt.data.planned_stages as PlannedStage[];
              setStages(
                planned.map((p, i) => ({
                  ...p,
                  seq: 0,
                  summary: "",
                  detail: {},
                  duration_ms: 0,
                  elapsed_ms: 0,
                  // The first agent is already running by the time this frame
                  // arrives — the server emits `start` immediately before
                  // invoking the graph.
                  status: i === 0 ? "running" : "pending",
                })),
              );
            } else if (evt.event === "stage") {
              setStages((prev) =>
                applyStage(prev, evt.data as unknown as Omit<PipelineStage, "status">),
              );
            } else if (evt.event === "result") {
              setResult(evt.data);
              setStages((prev) =>
                prev.map((s) =>
                  s.status === "done" ? s : { ...s, status: "done" },
                ),
              );
              setPhase("done");
            } else if (evt.event === "error") {
              setError(evt.data.detail);
              setPhase("done");
            }
          },
        });
      } catch (e) {
        // An abort is the user pressing stop, not a failure worth reporting.
        if (controller.signal.aborted) return;
        setError(
          e instanceof Error
            ? e.message
            : "Could not reach the engine. Is the API running?",
        );
        setPhase("done");
      } finally {
        void getBudget().then(setBudget);
      }
    },
    [activeDealId],
  );

  const handleCancel = useCallback(() => {
    abortRef.current?.abort();
    setPhase("idle");
    setStages([]);
  }, []);

  const handleSelectCitation = useCallback((index: number) => {
    setHighlighted(index);
    document
      .getElementById(`cite-${index}`)
      ?.scrollIntoView({ behavior: "smooth", block: "center" });
    // Clear the flash so re-clicking the same pill highlights it again.
    setTimeout(() => setHighlighted(null), 1600);
  }, []);

  // Memoised: `parseAnswer` walks the whole answer body, and this component
  // re-renders on every tick of the elapsed-time interval.
  const parsedCitations = useMemo(
    () => (result ? parseAnswer(result.answer, result.citations).citations : []),
    [result],
  );

  const isLanding = phase === "idle" && !result;
  const activeDeal = deals.find((d) => d.deal_id === activeDealId) ?? null;

  return (
    <div className="min-h-screen">
      <Masthead
        deals={deals}
        activeDealId={activeDealId}
        onSelectDeal={setActiveDealId}
        budget={budget}
      />

      <main className="mx-auto max-w-[1400px] px-5 pb-24">
        <AnimatePresence mode="wait">
          {isLanding ? (
            /* ---------------- Landing ---------------- */
            <motion.div
              key="landing"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -12 }}
              transition={{ duration: 0.3 }}
              className="mx-auto flex min-h-[calc(100vh-3.5rem)] max-w-3xl flex-col justify-center py-16"
            >
              <div className="mb-9 text-center">
                <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-line bg-ink-850/70 px-3 py-1.5">
                  <Sparkles size={11} className="text-gold-400" />
                  <span className="text-[0.7rem] font-medium tracking-wide text-ash-300">
                    8 agents · hybrid retrieval · grounded or refused
                  </span>
                </div>

                <h1 className="text-balance bg-gradient-to-br from-ash-50 via-ash-50 to-gold-400 bg-clip-text text-[2.6rem] font-semibold leading-[1.08] tracking-tight text-transparent sm:text-[3.1rem]">
                  Ask the data room
                </h1>

                <p className="mx-auto mt-4 max-w-[54ch] text-pretty text-[0.95rem] leading-relaxed text-ash-300">
                  {BRAND.promise}
                </p>
              </div>

              <AskBar
                variant="hero"
                onSubmit={handleSubmit}
                onCancel={handleCancel}
                running={false}
                disabled={!activeDealId}
              />

              {activeDeal && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.25 }}
                  className="mt-10 flex flex-wrap items-center justify-center gap-x-6 gap-y-2 text-[0.75rem] text-ash-500"
                >
                  <span className="inline-flex items-center gap-1.5">
                    <Layers size={11} />
                    {activeDeal.document_count} document
                    {activeDeal.document_count === 1 ? "" : "s"} indexed
                  </span>
                  {risks.length > 0 && (
                    <span>
                      {risks.length} risk signal{risks.length === 1 ? "" : "s"}{" "}
                      detected
                    </span>
                  )}
                  <span className="inline-flex items-center gap-1.5">
                    <CornerDownLeft size={11} />
                    Enter to ask
                  </span>
                </motion.div>
              )}
            </motion.div>
          ) : (
            /* ---------------- Result ---------------- */
            <motion.div
              key="result"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
              className="pt-8"
            >
              <h1 className="mb-6 max-w-[46ch] text-balance text-[1.55rem] font-semibold leading-snug tracking-tight text-ash-50">
                {question}
              </h1>

              <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_20rem]">
                <div className="min-w-0 space-y-5">
                  {error && (
                    <div className="panel border-bad/30 px-5 py-4">
                      <div className="text-[0.78rem] font-semibold uppercase tracking-widest text-bad">
                        Request failed
                      </div>
                      <p className="mt-1.5 text-[0.86rem] leading-relaxed text-ash-300">
                        {error}
                      </p>
                    </div>
                  )}

                  {result &&
                    (result.is_refusal ? (
                      <RefusalPanel result={result} />
                    ) : (
                      <AnswerPanel
                        result={result}
                        onSelectCitation={handleSelectCitation}
                      />
                    ))}

                  {/* While running, the timeline is the main event and belongs
                      in the primary column. Once the answer lands it moves to
                      the rail, where it becomes supporting evidence. */}
                  {phase === "running" && (
                    <PipelineTimeline
                      stages={stages}
                      running
                      elapsedMs={elapsed}
                    />
                  )}
                </div>

                <aside className="space-y-4 lg:sticky lg:top-20 lg:self-start">
                  {result && !result.is_refusal && (
                    <SourcesRail
                      citations={parsedCitations}
                      highlighted={highlighted}
                    />
                  )}

                  {phase === "done" && stages.length > 0 && (
                    <PipelineTimeline
                      stages={stages}
                      running={false}
                      elapsedMs={result?.total_latency_ms ?? elapsed}
                    />
                  )}

                  <DataRoomPanel documents={documents} risks={risks} />
                </aside>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Compact ask bar, docked once the conversation has started. */}
      {!isLanding && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="fixed inset-x-0 bottom-0 z-20 border-t border-line/70 bg-ink-950/78 backdrop-blur-xl"
        >
          <div className="mx-auto max-w-[1400px] px-5 py-3">
            <div className="lg:max-w-[calc(100%-21.25rem)]">
              <AskBar
                variant="compact"
                onSubmit={handleSubmit}
                onCancel={handleCancel}
                running={phase === "running"}
                disabled={!activeDealId}
              />
            </div>
          </div>
        </motion.div>
      )}

      {/* Upload lives on the landing state only — it is an invitation, not a
          control the reader needs while reading an answer. */}
      {isLanding && (
        <div className="pointer-events-none fixed bottom-5 right-5 z-20 hidden w-72 xl:block">
          <div className="pointer-events-auto">
            <UploadPanel
              sandboxDeal={sandbox.sandboxDeal}
              uploading={sandbox.uploading}
              error={sandbox.error}
              onUpload={sandbox.upload}
              onDiscard={sandbox.discard}
            />
          </div>
        </div>
      )}
    </div>
  );
}
