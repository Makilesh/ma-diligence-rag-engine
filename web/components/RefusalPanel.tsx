"use client";

import { motion } from "framer-motion";
import { Hand, Lightbulb } from "lucide-react";

import type { QueryResponse } from "@/lib/types";

interface RefusalPanelProps {
  result: QueryResponse;
}

/**
 * The refusal surface.
 *
 * A refusal is a designed outcome here, not a failure — the engine's premise is
 * that a confident wrong number can misprice a deal, so declining is the
 * correct behaviour when retrieval comes back thin. It is presented as a
 * deliberate decision with its evidence (the quality score that triggered it,
 * the number of attempts made) rather than as an error, because styling it like
 * one would teach the reader to distrust the very behaviour that protects them.
 */
export default function RefusalPanel({ result }: RefusalPanelProps) {
  const score = result.context_quality_score;
  const attempts = result.rewrite_iterations + 1;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.32, ease: [0.22, 1, 0.36, 1] }}
      className="panel overflow-hidden"
    >
      <div className="h-px bg-gradient-to-r from-warn via-warn/25 to-transparent" />

      <div className="px-6 py-6">
        <div className="flex items-start gap-4">
          <div className="mt-0.5 shrink-0 rounded-xl border border-warn/25 bg-warn/10 p-2.5 text-warn">
            <Hand size={18} />
          </div>

          <div className="min-w-0 flex-1">
            <h2 className="text-[1.02rem] font-semibold tracking-tight text-ash-50">
              Declined — the data room does not support an answer
            </h2>
            <p className="mt-2 max-w-[62ch] text-[0.9rem] leading-relaxed text-ash-300">
              Retrieval ran {attempts} time{attempts === 1 ? "" : "s"}, reformulating
              the question between attempts, and no passage scored high enough to
              ground an answer. Rather than assemble a plausible one from weak
              context, the pipeline stopped.
            </p>

            <div className="mt-5 flex flex-wrap gap-2.5">
              <div className="rounded-lg border border-line bg-ink-800 px-3 py-2">
                <div className="text-[0.62rem] font-semibold uppercase tracking-widest text-ash-500">
                  Best context quality
                </div>
                <div className="mt-0.5 text-[0.95rem] font-semibold tabular-nums text-warn">
                  {score.toFixed(2)}
                </div>
              </div>
              <div className="rounded-lg border border-line bg-ink-800 px-3 py-2">
                <div className="text-[0.62rem] font-semibold uppercase tracking-widest text-ash-500">
                  Retrieval attempts
                </div>
                <div className="mt-0.5 text-[0.95rem] font-semibold tabular-nums text-ash-50">
                  {attempts}
                </div>
              </div>
              <div className="rounded-lg border border-line bg-ink-800 px-3 py-2">
                <div className="text-[0.62rem] font-semibold uppercase tracking-widest text-ash-500">
                  Elapsed
                </div>
                <div className="mt-0.5 text-[0.95rem] font-semibold tabular-nums text-ash-50">
                  {(result.total_latency_ms / 1000).toFixed(1)}s
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="border-t border-line bg-ink-850/60 px-6 py-4">
        <div className="mb-2 flex items-center gap-2 text-[0.7rem] font-semibold uppercase tracking-widest text-ash-500">
          <Lightbulb size={12} />
          What usually fixes this
        </div>
        <ul className="grid gap-1.5 text-[0.84rem] leading-relaxed text-ash-300 sm:grid-cols-2">
          <li>Name the document or section you expect the answer to be in.</li>
          <li>Ask for one fact at a time rather than a broad summary.</li>
          <li>Check that the relevant document has been indexed for this deal.</li>
          <li>Use the language of the filing — &ldquo;indemnification cap&rdquo;, not &ldquo;liability limit&rdquo;.</li>
        </ul>
      </div>
    </motion.div>
  );
}
