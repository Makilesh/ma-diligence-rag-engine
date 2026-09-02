"use client";

import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  AlertTriangle,
  FileText,
  History,
  ShieldAlert,
  TriangleAlert,
} from "lucide-react";

import { prettyFilename } from "@/lib/citations";
import type { DocumentRecord, RiskSignal, Severity } from "@/lib/types";

interface DataRoomPanelProps {
  documents: DocumentRecord[];
  risks: RiskSignal[];
}

const SEVERITY_STYLE: Record<Severity, { dot: string; text: string; bg: string }> = {
  high: { dot: "bg-bad", text: "text-bad", bg: "bg-bad/10" },
  medium: { dot: "bg-warn", text: "text-warn", bg: "bg-warn/10" },
  low: { dot: "bg-ok", text: "text-ok", bg: "bg-ok/10" },
};

/** Turns `change_of_control` into `Change of control`. */
function humanise(signalType: string): string {
  const spaced = signalType.replace(/_/g, " ");
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

function RiskList({ risks }: { risks: RiskSignal[] }) {
  if (!risks.length) {
    return (
      <p className="px-4 py-5 text-[0.8rem] leading-relaxed text-ash-500">
        No risk signals were detected in this deal&apos;s documents.
      </p>
    );
  }

  // Group by type so ten hits of one clause read as one finding rather than
  // ten rows that bury everything else.
  const grouped = new Map<string, RiskSignal[]>();
  for (const risk of risks) {
    const list = grouped.get(risk.signal_type) ?? [];
    list.push(risk);
    grouped.set(risk.signal_type, list);
  }

  return (
    <ul className="divide-y divide-line">
      {[...grouped.entries()].map(([type, group]) => {
        const severity = group[0].severity;
        const style = SEVERITY_STYLE[severity] ?? SEVERITY_STYLE.low;
        const files = [...new Set(group.map((g) => g.source_file))];

        return (
          <li key={type} className="px-4 py-3">
            <div className="flex items-center justify-between gap-3">
              <div className="flex min-w-0 items-center gap-2">
                <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${style.dot}`} />
                <span className="truncate text-[0.82rem] font-medium text-ash-50">
                  {humanise(type)}
                </span>
              </div>
              <span
                className={`shrink-0 rounded px-1.5 py-0.5 text-[0.6rem] font-semibold uppercase tracking-wider ${style.bg} ${style.text}`}
              >
                {severity}
              </span>
            </div>
            <div className="mt-1 pl-3.5 text-[0.72rem] text-ash-500">
              {group.length} occurrence{group.length === 1 ? "" : "s"} across{" "}
              {files.length} document{files.length === 1 ? "" : "s"}
            </div>
          </li>
        );
      })}
    </ul>
  );
}

function DocumentList({ documents }: { documents: DocumentRecord[] }) {
  if (!documents.length) {
    return (
      <p className="px-4 py-5 text-[0.8rem] leading-relaxed text-ash-500">
        Nothing is indexed for this deal yet — every question will be refused
        until a document is uploaded.
      </p>
    );
  }

  return (
    <ul className="divide-y divide-line">
      {documents.map((doc) => (
        <li key={doc.doc_id} className="px-4 py-3">
          <div className="flex items-start gap-2.5">
            <FileText
              size={13}
              className={`mt-0.5 shrink-0 ${
                doc.is_current_version ? "text-ash-500" : "text-bad"
              }`}
            />
            <div className="min-w-0 flex-1">
              <div className="truncate text-[0.82rem] font-medium text-ash-50">
                {prettyFilename(doc.filename)}
              </div>
              <div className="mt-0.5 flex flex-wrap items-center gap-x-2.5 gap-y-1 text-[0.7rem] text-ash-500">
                {doc.document_category && (
                  <span className="capitalize">{doc.document_category}</span>
                )}
                <span className="tabular-nums">{doc.chunks_created} chunks</span>
                {doc.has_redline && (
                  <span className="text-violet">redlined</span>
                )}
              </div>

              {!doc.is_current_version && (
                <div className="mt-1.5 inline-flex items-center gap-1 rounded bg-bad/10 px-1.5 py-0.5 text-[0.62rem] font-semibold uppercase tracking-wider text-bad">
                  <AlertTriangle size={9} />
                  Superseded
                </div>
              )}
            </div>
          </div>
        </li>
      ))}
    </ul>
  );
}

export default function DataRoomPanel({ documents, risks }: DataRoomPanelProps) {
  const [tab, setTab] = useState<"documents" | "risk">("documents");
  const highRisk = risks.filter((r) => r.severity === "high").length;

  const tabs = [
    {
      id: "documents" as const,
      label: "Documents",
      icon: History,
      count: documents.length,
    },
    {
      id: "risk" as const,
      label: "Risk",
      icon: ShieldAlert,
      count: risks.length,
      alert: highRisk > 0,
    },
  ];

  return (
    <div className="panel overflow-hidden">
      <div className="flex border-b border-line p-1">
        {tabs.map((t) => {
          const Icon = t.icon;
          const active = tab === t.id;
          return (
            <button
              key={t.id}
              type="button"
              onClick={() => setTab(t.id)}
              aria-selected={active}
              role="tab"
              className={`relative flex flex-1 items-center justify-center gap-1.5 rounded-lg px-3 py-2 text-[0.75rem] font-medium transition-colors ${
                active ? "text-gold-400" : "text-ash-500 hover:text-ash-300"
              }`}
            >
              {active && (
                <motion.span
                  layoutId="dataroom-tab"
                  className="absolute inset-0 rounded-lg bg-ink-750"
                  transition={{ duration: 0.18 }}
                />
              )}
              <span className="relative flex items-center gap-1.5">
                <Icon size={12} />
                {t.label}
                <span className="font-mono text-[0.68rem] tabular-nums opacity-70">
                  {t.count}
                </span>
                {t.alert && (
                  <TriangleAlert size={10} className="text-bad" aria-label="high severity present" />
                )}
              </span>
            </button>
          );
        })}
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={tab}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.13 }}
          className="max-h-[22rem] overflow-y-auto"
        >
          {tab === "documents" ? (
            <DocumentList documents={documents} />
          ) : (
            <RiskList risks={risks} />
          )}
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
