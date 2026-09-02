"use client";

import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { ArrowUp, EyeOff, Loader2, Square } from "lucide-react";

/**
 * One example per query type the classifier routes differently, so the examples
 * double as documentation of what the engine actually does with a question.
 * Mirrors EXAMPLE_QUERIES in `app/components/query_interface.py`.
 */
export const EXAMPLES: { type: string; query: string }[] = [
  {
    type: "Multi-hop",
    query:
      "What is the Section 280G excise tax exposure and which executives trigger it?",
  },
  {
    type: "Financial",
    query:
      "What was the total revenue in FY2023 and how does it compare to FY2022?",
  },
  {
    type: "Legal",
    query:
      "What are the key change of control provisions in the merger agreement?",
  },
  {
    type: "Comparative",
    query: "Compare the EBITDA margins across the last three fiscal years.",
  },
  {
    type: "Summary",
    query: "Summarize the key findings from the most recent board minutes.",
  },
];

interface AskBarProps {
  onSubmit: (query: string, includePii: boolean) => void;
  onCancel: () => void;
  running: boolean;
  disabled: boolean;
  /** Renders the tall hero treatment on the landing state, compact once answered. */
  variant: "hero" | "compact";
  initialValue?: string;
}

export default function AskBar({
  onSubmit,
  onCancel,
  running,
  disabled,
  variant,
  initialValue = "",
}: AskBarProps) {
  const [value, setValue] = useState(initialValue);
  const [includePii, setIncludePii] = useState(false);
  const [focused, setFocused] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Grow with the content instead of scrolling inside a fixed box. A
  // multi-fact due-diligence question routinely runs past one line, and hiding
  // half of it behind a scrollbar while the user is still composing is worse
  // than the extra height.
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 240)}px`;
  }, [value]);

  const submit = () => {
    const trimmed = value.trim();
    if (!trimmed || running || disabled) return;
    onSubmit(trimmed, includePii);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Enter sends, Shift+Enter breaks the line — the convention every chat
    // surface has trained users into.
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  const isHero = variant === "hero";

  return (
    <div className="w-full">
      <motion.div
        animate={{
          borderColor: focused ? "rgba(224,179,106,0.55)" : "rgb(33,39,57)",
          boxShadow: focused
            ? "0 0 0 3px rgba(224,179,106,0.10), 0 18px 44px -26px rgba(0,0,0,0.9)"
            : "0 12px 32px -22px rgba(0,0,0,0.8)",
        }}
        transition={{ duration: 0.18 }}
        className="rounded-2xl border bg-ink-850/80 backdrop-blur-xl"
      >
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          disabled={disabled}
          rows={1}
          placeholder={
            disabled
              ? "Select a deal to begin…"
              : "Ask anything about the data room…"
          }
          aria-label="Ask a question about the data room"
          className={`w-full resize-none bg-transparent px-5 text-ash-50 placeholder:text-ash-600 focus:outline-none disabled:cursor-not-allowed ${
            isHero ? "pt-5 text-[1.05rem]" : "pt-4 text-[0.95rem]"
          }`}
        />

        <div className="flex items-center justify-between gap-3 px-3.5 pb-3 pt-2">
          <button
            type="button"
            onClick={() => setIncludePii((v) => !v)}
            aria-pressed={includePii}
            title="Include PII-flagged content (HR records, salary data). Excluded by default; every authorized use is written to the audit log."
            className={`inline-flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-[0.72rem] font-medium transition-colors ${
              includePii
                ? "border-warn/35 bg-warn/10 text-warn"
                : "border-line bg-ink-800 text-ash-500 hover:text-ash-300"
            }`}
          >
            <EyeOff size={12} />
            {includePii ? "PII included" : "PII excluded"}
          </button>

          <div className="flex items-center gap-2.5">
            <span className="hidden text-[0.7rem] text-ash-600 sm:inline">
              {running ? "Running…" : "Enter to send"}
            </span>

            {running ? (
              <button
                type="button"
                onClick={onCancel}
                aria-label="Stop the running query"
                className="inline-flex h-9 w-9 items-center justify-center rounded-xl border border-line bg-ink-800 text-ash-300 transition-colors hover:border-bad/50 hover:text-bad"
              >
                <Square size={13} className="fill-current" />
              </button>
            ) : (
              <button
                type="button"
                onClick={submit}
                disabled={!value.trim() || disabled}
                aria-label="Run the query"
                className="inline-flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-b from-gold-400 to-gold-500 text-ink-950 transition-all hover:shadow-[0_6px_20px_-8px_rgba(224,179,106,0.85)] disabled:cursor-not-allowed disabled:from-ink-700 disabled:to-ink-700 disabled:text-ash-600 disabled:shadow-none"
              >
                {disabled ? <Loader2 size={15} className="animate-spin" /> : <ArrowUp size={16} />}
              </button>
            )}
          </div>
        </div>
      </motion.div>

      {isHero && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.12, duration: 0.35 }}
          className="mt-5 flex flex-wrap justify-center gap-2"
        >
          {EXAMPLES.map((ex) => (
            <button
              key={ex.type}
              type="button"
              onClick={() => {
                setValue(ex.query);
                textareaRef.current?.focus();
              }}
              disabled={disabled}
              className="group inline-flex items-center gap-2 rounded-full border border-line bg-ink-850/70 py-1.5 pl-2.5 pr-3.5 text-left transition-all hover:border-gold-600/45 hover:bg-ink-800 disabled:opacity-40"
            >
              <span className="rounded-full bg-gold-500/12 px-1.5 py-0.5 text-[0.62rem] font-semibold uppercase tracking-wider text-gold-400">
                {ex.type}
              </span>
              <span className="max-w-[26ch] truncate text-[0.78rem] text-ash-300 group-hover:text-ash-50">
                {ex.query}
              </span>
            </button>
          ))}
        </motion.div>
      )}
    </div>
  );
}
