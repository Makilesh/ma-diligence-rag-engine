"use client";

import { useMemo } from "react";
import { motion } from "framer-motion";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  BadgeCheck,
  Clock,
  Gauge,
  RefreshCw,
  ShieldAlert,
  ShieldX,
} from "lucide-react";

import CitationPill from "./CitationPill";
import { parseAnswer, type ParsedCitation } from "@/lib/citations";
import type { QueryResponse } from "@/lib/types";

interface AnswerPanelProps {
  result: QueryResponse;
  onSelectCitation?: (index: number) => void;
}

/** Visual treatment per validation outcome. */
const VERDICT = {
  passed: {
    icon: BadgeCheck,
    label: "Validated",
    detail: "Every claim was matched back to retrieved text",
    className: "text-ok bg-ok/12 border-ok/30",
  },
  warning: {
    icon: ShieldAlert,
    label: "Partially grounded",
    detail: "Some claims could not be matched to a source",
    className: "text-warn bg-warn/12 border-warn/30",
  },
  failed: {
    icon: ShieldX,
    label: "Validation failed",
    detail: "Unsupported claims were detected in this answer",
    className: "text-bad bg-bad/12 border-bad/30",
  },
} as const;

function Metric({
  icon: Icon,
  label,
  value,
  tone = "default",
}: {
  icon: React.ElementType;
  label: string;
  value: string;
  tone?: "default" | "good" | "warn" | "bad";
}) {
  const toneClass = {
    default: "text-ash-50",
    good: "text-ok",
    warn: "text-warn",
    bad: "text-bad",
  }[tone];

  return (
    <div className="flex items-center gap-2">
      <Icon size={13} className="shrink-0 text-ash-500" />
      <span className="text-[0.7rem] uppercase tracking-wider text-ash-500">
        {label}
      </span>
      <span className={`text-[0.82rem] font-semibold tabular-nums ${toneClass}`}>
        {value}
      </span>
    </div>
  );
}

export default function AnswerPanel({
  result,
  onSelectCitation,
}: AnswerPanelProps) {
  // Parsing rewrites every bracket citation into a numbered link, so it must
  // run before the markdown is handed to the renderer — and only when the
  // answer changes, since it walks the whole body.
  const { markdown, citations } = useMemo(
    () => parseAnswer(result.answer, result.citations),
    [result.answer, result.citations],
  );

  const byIndex = useMemo(() => {
    const map = new Map<number, ParsedCitation>();
    citations.forEach((c) => map.set(c.index, c));
    return map;
  }, [citations]);

  const verdict = VERDICT[result.validation_status] ?? VERDICT.passed;
  const VerdictIcon = verdict.icon;

  const confidence = Math.round(result.confidence_score * 100);
  const confidenceTone =
    confidence >= 75 ? "good" : confidence >= 45 ? "warn" : "bad";

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.32, ease: [0.22, 1, 0.36, 1] }}
      className="panel overflow-hidden"
    >
      {/* Gold hairline along the top edge — groups the answer as the primary
          surface without adding another border weight. */}
      <div className="h-px bg-gradient-to-r from-gold-500 via-gold-500/25 to-transparent" />

      <div className="flex flex-wrap items-center justify-between gap-x-6 gap-y-3 border-b border-line px-6 py-3.5">
        <div
          className={`inline-flex items-center gap-2 rounded-lg border px-2.5 py-1.5 ${verdict.className}`}
          title={verdict.detail}
        >
          <VerdictIcon size={14} />
          <span className="text-[0.74rem] font-semibold tracking-wide">
            {verdict.label}
          </span>
        </div>

        <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
          <Metric
            icon={Gauge}
            label="Confidence"
            value={`${confidence}%`}
            tone={confidenceTone}
          />
          <Metric
            icon={Clock}
            label="Latency"
            value={`${(result.total_latency_ms / 1000).toFixed(1)}s`}
          />
          {result.rewrite_iterations > 0 && (
            <Metric
              icon={RefreshCw}
              label="Rewrites"
              value={String(result.rewrite_iterations)}
              tone="warn"
            />
          )}
        </div>
      </div>

      <div className="answer-body px-6 py-6">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            // Citations arrive as `[n](#cite-n)` links from parseAnswer. Any
            // other link is a genuine link the model wrote and is left alone.
            a({ href, children, ...props }) {
              const match = /^#cite-(\d+)$/.exec(href ?? "");
              if (match) {
                const citation = byIndex.get(Number(match[1]));
                if (citation) {
                  return (
                    <CitationPill
                      citation={citation}
                      onSelect={onSelectCitation}
                    />
                  );
                }
              }
              return (
                <a
                  href={href}
                  className="text-gold-400 underline underline-offset-2 hover:text-gold-300"
                  target="_blank"
                  rel="noreferrer noopener"
                  {...props}
                >
                  {children}
                </a>
              );
            },
            // Wide financial tables must scroll inside their own container —
            // otherwise a ten-column comparison scrolls the whole page sideways.
            table({ children, ...props }) {
              return (
                <div className="table-scroll">
                  <table {...props}>{children}</table>
                </div>
              );
            },
          }}
        >
          {markdown}
        </ReactMarkdown>
      </div>

      {result.hallucination_flags.length > 0 && (
        <div className="border-t border-line bg-bad/[0.05] px-6 py-4">
          <div className="mb-2.5 flex items-center gap-2 text-[0.72rem] font-semibold uppercase tracking-wider text-bad">
            <ShieldAlert size={13} />
            {result.hallucination_flags.length} unsupported claim
            {result.hallucination_flags.length === 1 ? "" : "s"} flagged
          </div>
          <ul className="space-y-1.5">
            {result.hallucination_flags.map((flag, i) => (
              <li
                key={i}
                className="border-l-2 border-bad/40 pl-3 text-[0.82rem] leading-relaxed text-ash-300"
              >
                {flag}
              </li>
            ))}
          </ul>
        </div>
      )}
    </motion.div>
  );
}
