"use client";

import { useCallback, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { AlertTriangle, Calculator, FileText } from "lucide-react";

import { citationLabel, prettyFilename, type ParsedCitation } from "@/lib/citations";

interface CitationPillProps {
  citation: ParsedCitation;
  /** Scrolls the matching row in the sources rail into view and flashes it. */
  onSelect?: (index: number) => void;
}

/**
 * An inline superscript citation marker with a hover card.
 *
 * The card is positioned `fixed` from the pill's measured rect rather than
 * absolutely inside a relative wrapper. Absolute positioning is simpler but
 * gets clipped: citations land inside table cells and list items, and the
 * answer's table wrapper sets `overflow-x: auto`, which would crop the card to
 * the cell it opened from.
 */
export default function CitationPill({ citation, onSelect }: CitationPillProps) {
  const ref = useRef<HTMLAnchorElement>(null);
  const [coords, setCoords] = useState<{ x: number; y: number } | null>(null);

  const show = useCallback(() => {
    const rect = ref.current?.getBoundingClientRect();
    if (!rect) return;
    setCoords({ x: rect.left + rect.width / 2, y: rect.top });
  }, []);

  const hide = useCallback(() => setCoords(null), []);

  const label = citationLabel(citation);

  return (
    <>
      <a
        ref={ref}
        href={`#cite-${citation.index}`}
        className={`cite-pill ${citation.isStale ? "cite-pill-stale" : ""}`}
        onMouseEnter={show}
        onMouseLeave={hide}
        onFocus={show}
        onBlur={hide}
        onClick={(e) => {
          e.preventDefault();
          onSelect?.(citation.index);
        }}
        aria-label={`Source ${citation.index}: ${citation.sourceFile}${
          label ? `, ${label}` : ""
        }`}
      >
        {citation.index}
      </a>

      <AnimatePresence>
        {coords && (
          <motion.div
            initial={{ opacity: 0, y: 4, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 4, scale: 0.97 }}
            transition={{ duration: 0.14, ease: "easeOut" }}
            role="tooltip"
            className="pointer-events-none fixed z-50 w-[19rem] -translate-x-1/2 -translate-y-full"
            style={{ left: coords.x, top: coords.y - 10 }}
          >
            <div className="panel px-3.5 py-3 shadow-2xl">
              <div className="flex items-start gap-2.5">
                <div
                  className={`mt-0.5 shrink-0 rounded-md p-1.5 ${
                    citation.isStale
                      ? "bg-bad/15 text-bad"
                      : "bg-gold-500/15 text-gold-400"
                  }`}
                >
                  {citation.isStale ? (
                    <AlertTriangle size={13} />
                  ) : citation.isComputed ? (
                    <Calculator size={13} />
                  ) : (
                    <FileText size={13} />
                  )}
                </div>

                <div className="min-w-0 flex-1">
                  <div className="truncate text-[0.82rem] font-semibold text-ash-50">
                    {prettyFilename(citation.sourceFile)}
                  </div>
                  {label && (
                    <div className="mt-0.5 text-[0.74rem] leading-snug text-ash-300">
                      {label}
                    </div>
                  )}

                  <div className="mt-2 flex flex-wrap gap-1.5">
                    {citation.isComputed && (
                      <span className="rounded bg-info/12 px-1.5 py-0.5 text-[0.62rem] font-semibold uppercase tracking-wider text-info">
                        Computed
                      </span>
                    )}
                    {citation.isStale && (
                      <span className="rounded bg-bad/12 px-1.5 py-0.5 text-[0.62rem] font-semibold uppercase tracking-wider text-bad">
                        Superseded version
                      </span>
                    )}
                    {citation.record?.content_type === "table" && (
                      <span className="rounded bg-violet/12 px-1.5 py-0.5 text-[0.62rem] font-semibold uppercase tracking-wider text-violet">
                        Table
                      </span>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* Arrow. Drawn as a rotated square so it inherits the panel's
                border on the two edges that actually show. */}
            <div className="mx-auto -mt-[5px] h-2.5 w-2.5 rotate-45 border-r border-b border-line bg-ink-900" />
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
