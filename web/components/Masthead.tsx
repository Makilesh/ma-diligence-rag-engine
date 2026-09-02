"use client";

import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Check, ChevronDown, Clock, Database, Zap } from "lucide-react";

import { BRAND } from "@/lib/brand";
import type { BudgetStatus, Deal } from "@/lib/types";

interface MastheadProps {
  deals: Deal[];
  activeDealId: string | null;
  onSelectDeal: (dealId: string) => void;
  budget: BudgetStatus;
}

/**
 * The engine's daily model quota, at a glance.
 *
 * Worth the masthead space because quota exhaustion is silent otherwise:
 * synthesis degrades to a smaller model and the only visible symptom is a
 * subtly worse answer. Surfacing the pressure explains the degradation before
 * it is mistaken for a model failure.
 */
function BudgetMeter({ budget }: { budget: BudgetStatus }) {
  const entries = Object.entries(budget);
  if (!entries.length) return null;

  const used = entries.reduce((sum, [, v]) => sum + (v.used ?? 0), 0);
  const limit = entries.reduce((sum, [, v]) => sum + (v.limit ?? 0), 0);
  if (!limit) return null;

  const pct = used / limit;
  const tone = pct < 0.7 ? "text-ok" : pct < 0.95 ? "text-warn" : "text-bad";
  const barTone = pct < 0.7 ? "bg-ok" : pct < 0.95 ? "bg-warn" : "bg-bad";

  return (
    <div
      className="hidden items-center gap-2.5 md:flex"
      title={entries
        .map(([k, v]) => `${k.replace(/_/g, " ")}: ${v.remaining}/${v.limit} left`)
        .join("\n")}
    >
      <Zap size={12} className="text-ash-500" />
      <div className="h-1 w-16 overflow-hidden rounded-full bg-ink-700">
        <motion.div
          className={`h-full rounded-full ${barTone}`}
          initial={{ width: 0 }}
          animate={{ width: `${Math.min(pct, 1) * 100}%` }}
          transition={{ duration: 0.5 }}
        />
      </div>
      <span className={`font-mono text-[0.7rem] tabular-nums ${tone}`}>
        {limit - used}
      </span>
    </div>
  );
}

function DealSwitcher({
  deals,
  activeDealId,
  onSelectDeal,
}: Omit<MastheadProps, "budget">) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  // Close on any click outside, and on Escape. A dropdown that traps the user
  // is worse than no dropdown.
  useEffect(() => {
    if (!open) return;
    const onPointerDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const active = deals.find((d) => d.deal_id === activeDealId);

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="listbox"
        aria-expanded={open}
        className="inline-flex max-w-[15rem] items-center gap-2 rounded-lg border border-line bg-ink-850 px-2.5 py-1.5 text-left transition-colors hover:border-line-strong"
      >
        <Database size={12} className="shrink-0 text-gold-500" />
        <span className="truncate text-[0.78rem] font-medium text-ash-50">
          {active?.deal_name ?? "Select a deal"}
        </span>
        {active && (
          <span className="shrink-0 font-mono text-[0.68rem] tabular-nums text-ash-500">
            {active.document_count}
          </span>
        )}
        <ChevronDown
          size={12}
          className={`shrink-0 text-ash-500 transition-transform ${open ? "rotate-180" : ""}`}
        />
      </button>

      <AnimatePresence>
        {open && (
          <motion.ul
            initial={{ opacity: 0, y: -4, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -4, scale: 0.98 }}
            transition={{ duration: 0.14 }}
            role="listbox"
            className="panel absolute left-0 top-full z-40 mt-1.5 max-h-80 w-72 overflow-y-auto p-1"
          >
            {deals.length === 0 && (
              <li className="px-3 py-3 text-[0.78rem] text-ash-500">
                No deals are indexed yet.
              </li>
            )}
            {deals.map((deal) => (
              <li key={deal.deal_id}>
                <button
                  type="button"
                  role="option"
                  aria-selected={deal.deal_id === activeDealId}
                  onClick={() => {
                    onSelectDeal(deal.deal_id);
                    setOpen(false);
                  }}
                  className="flex w-full items-center gap-2.5 rounded-md px-2.5 py-2 text-left transition-colors hover:bg-ink-750"
                >
                  <Check
                    size={13}
                    className={`shrink-0 ${
                      deal.deal_id === activeDealId
                        ? "text-gold-400"
                        : "text-transparent"
                    }`}
                  />
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-1.5">
                      <span className="truncate text-[0.8rem] font-medium text-ash-50">
                        {deal.deal_name}
                      </span>
                      {deal.is_sandbox && (
                        <span className="inline-flex shrink-0 items-center gap-0.5 rounded bg-violet/12 px-1.5 py-px text-[0.58rem] font-semibold uppercase tracking-wider text-violet">
                          <Clock size={8} />
                          Temp
                        </span>
                      )}
                    </div>
                    <div className="truncate text-[0.7rem] text-ash-500">
                      {deal.document_count} document
                      {deal.document_count === 1 ? "" : "s"} indexed
                    </div>
                  </div>
                </button>
              </li>
            ))}
          </motion.ul>
        )}
      </AnimatePresence>
    </div>
  );
}

export default function Masthead({
  deals,
  activeDealId,
  onSelectDeal,
  budget,
}: MastheadProps) {
  return (
    <header className="sticky top-0 z-30 border-b border-line/70 bg-ink-950/72 backdrop-blur-xl">
      <div className="mx-auto flex h-14 max-w-[1400px] items-center justify-between gap-4 px-5">
        <div className="flex items-center gap-3.5">
          <a href="/" className="group flex items-center gap-2.5">
            {/* Wordmark. The diamond is a nod to the gold accent and reads at
                favicon size, which a lettermark would not. */}
            <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-gradient-to-br from-gold-400 to-gold-600 shadow-[0_4px_14px_-6px_rgba(224,179,106,0.9)]">
              <span className="block h-2.5 w-2.5 rotate-45 rounded-[2px] bg-ink-950" />
            </span>
            <span className="text-[0.95rem] font-semibold tracking-tight text-ash-50">
              {BRAND.name}
            </span>
          </a>

          <span className="hidden h-4 w-px bg-line sm:block" />

          <span className="hidden text-[0.72rem] tracking-wide text-ash-500 sm:block">
            {BRAND.tagline}
          </span>
        </div>

        <div className="flex items-center gap-3.5">
          <BudgetMeter budget={budget} />
          <DealSwitcher
            deals={deals}
            activeDealId={activeDealId}
            onSelectDeal={onSelectDeal}
          />
        </div>
      </div>
    </header>
  );
}
