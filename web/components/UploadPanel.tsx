"use client";

import { useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Loader2, ShieldCheck, Trash2, Upload } from "lucide-react";

import type { Deal } from "@/lib/types";

interface UploadPanelProps {
  sandboxDeal: Deal | null;
  uploading: boolean;
  error: string | null;
  onUpload: (file: File) => void;
  onDiscard: () => void;
}

const ACCEPTED = ".pdf,.docx,.xlsx,.xls,.pptx,.txt";

export default function UploadPanel({
  sandboxDeal,
  uploading,
  error,
  onUpload,
  onDiscard,
}: UploadPanelProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const handleFiles = (files: FileList | null) => {
    const file = files?.[0];
    if (file) onUpload(file);
  };

  return (
    <div className="panel overflow-hidden">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          handleFiles(e.dataTransfer.files);
        }}
        className={`px-4 py-5 text-center transition-colors ${
          dragging ? "bg-gold-500/[0.07]" : ""
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED}
          className="sr-only"
          onChange={(e) => {
            handleFiles(e.target.files);
            // Reset so re-uploading the same filename fires a change event.
            e.target.value = "";
          }}
        />

        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          disabled={uploading}
          className="mx-auto flex flex-col items-center gap-2 disabled:opacity-60"
        >
          <span
            className={`flex h-9 w-9 items-center justify-center rounded-xl border transition-colors ${
              dragging
                ? "border-gold-500/60 bg-gold-500/12 text-gold-400"
                : "border-line bg-ink-800 text-ash-500"
            }`}
          >
            {uploading ? (
              <Loader2 size={15} className="animate-spin" />
            ) : (
              <Upload size={15} />
            )}
          </span>
          <span className="text-[0.8rem] font-medium text-ash-50">
            {uploading ? "Indexing…" : "Test it on your own document"}
          </span>
          <span className="max-w-[24ch] text-[0.7rem] leading-snug text-ash-500">
            PDF, DOCX, XLSX or PPTX. Drop it here or click to browse.
          </span>
        </button>

        {uploading && (
          <p className="mt-3 text-[0.7rem] leading-snug text-ash-500">
            Chunking, embedding and indexing runs on CPU — expect roughly a
            minute for a long document.
          </p>
        )}
      </div>

      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden border-t border-line bg-bad/[0.06]"
          >
            <p className="px-4 py-2.5 text-[0.75rem] leading-snug text-bad">
              {error}
            </p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* The privacy promise, stated where the decision to upload is made
          rather than buried in a footer. */}
      <div className="border-t border-line bg-ink-850/60 px-4 py-3">
        <div className="flex items-start gap-2">
          <ShieldCheck size={12} className="mt-0.5 shrink-0 text-ok" />
          <p className="text-[0.7rem] leading-relaxed text-ash-500">
            Uploads go to a temporary deal that is deleted when you close this
            tab, and swept from the server on a timer if that never arrives.
          </p>
        </div>

        {sandboxDeal && (
          <button
            type="button"
            onClick={onDiscard}
            className="mt-2.5 inline-flex items-center gap-1.5 text-[0.72rem] font-medium text-ash-500 transition-colors hover:text-bad"
          >
            <Trash2 size={11} />
            Delete my documents now
          </button>
        )}
      </div>
    </div>
  );
}
