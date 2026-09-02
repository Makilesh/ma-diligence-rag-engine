"use client";

import { motion } from "framer-motion";
import { AlertTriangle, Calculator, FileSpreadsheet, FileText, Presentation } from "lucide-react";

import { citationLabel, prettyFilename, type ParsedCitation } from "@/lib/citations";

interface SourcesRailProps {
  citations: ParsedCitation[];
  /** Index most recently clicked in the answer body; flashes that row. */
  highlighted: number | null;
}

/** Picks an icon from the filename extension, falling back to a document. */
function iconFor(citation: ParsedCitation): React.ElementType {
  if (citation.isComputed) return Calculator;
  const name = citation.sourceFile.toLowerCase();
  if (/\.(xlsx?|csv)$/.test(name) || citation.sheet) return FileSpreadsheet;
  if (/\.pptx?$/.test(name) || citation.slide !== null) return Presentation;
  return FileText;
}

export default function SourcesRail({ citations, highlighted }: SourcesRailProps) {
  if (!citations.length) {
    return (
      <div className="panel px-4 py-5">
        <div className="text-[0.72rem] font-semibold uppercase tracking-widest text-ash-500">
          Sources
        </div>
        <p className="mt-2 text-[0.8rem] leading-relaxed text-ash-500">
          No inline citations were parsed from this answer.
        </p>
      </div>
    );
  }

  return (
    <div className="panel overflow-hidden">
      <div className="flex items-center justify-between border-b border-line px-4 py-3">
        <span className="text-[0.72rem] font-semibold uppercase tracking-widest text-ash-500">
          Sources
        </span>
        <span className="font-mono text-[0.7rem] tabular-nums text-ash-500">
          {citations.length}
        </span>
      </div>

      <ol className="divide-y divide-line">
        {citations.map((citation) => {
          const Icon = iconFor(citation);
          const label = citationLabel(citation);
          const isHighlighted = highlighted === citation.index;

          return (
            <motion.li
              key={citation.index}
              id={`cite-${citation.index}`}
              animate={
                isHighlighted
                  ? { backgroundColor: "rgba(224,179,106,0.10)" }
                  : { backgroundColor: "rgba(0,0,0,0)" }
              }
              transition={{ duration: 0.35 }}
              // Offsets the sticky masthead when the answer links here.
              className="scroll-mt-28 px-4 py-3 transition-colors hover:bg-white/[0.018]"
            >
              <div className="flex items-start gap-2.5">
                <span
                  className={`mt-px flex h-5 w-5 shrink-0 items-center justify-center rounded text-[0.66rem] font-bold tabular-nums ${
                    citation.isStale
                      ? "bg-bad/15 text-bad"
                      : "bg-gold-500/15 text-gold-400"
                  }`}
                >
                  {citation.index}
                </span>

                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-1.5">
                    <Icon size={12} className="shrink-0 text-ash-500" />
                    <span className="truncate text-[0.8rem] font-medium text-ash-50">
                      {prettyFilename(citation.sourceFile)}
                    </span>
                  </div>

                  {label && (
                    <div className="mt-0.5 truncate text-[0.73rem] text-ash-500">
                      {label}
                    </div>
                  )}

                  {citation.isStale && (
                    <div className="mt-1.5 inline-flex items-center gap-1 rounded bg-bad/10 px-1.5 py-0.5 text-[0.63rem] font-semibold uppercase tracking-wider text-bad">
                      <AlertTriangle size={9} />
                      Superseded
                    </div>
                  )}
                </div>
              </div>
            </motion.li>
          );
        })}
      </ol>
    </div>
  );
}
