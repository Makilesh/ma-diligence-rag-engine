/**
 * Wire types for the FastAPI backend.
 *
 * These mirror `api/models/response_models.py` field for field. When a model
 * changes there, change it here — nothing enforces the correspondence at build
 * time, so the only protection is keeping them adjacent in review.
 */

export interface Citation {
  chunk_id: string;
  source_file: string;
  page_number: number | null;
  section_heading: string;
  is_current_version: boolean;
  content_type: string;
  is_redline: boolean;
  superseded_by: string;
}

export type ValidationStatus = "passed" | "warning" | "failed";

export interface QueryResponse {
  answer: string;
  query_type: string;
  confidence_score: number;
  validation_status: ValidationStatus;
  citations: Citation[];
  hallucination_flags: string[];
  total_latency_ms: number;
  session_id: string;
  rewrite_iterations: number;
  agent_trace: Record<string, unknown>[];
  is_refusal: boolean;
  context_quality_score: number;
}

export interface Deal {
  deal_id: string;
  deal_name: string;
  description: string;
  document_count: number;
  status: string;
  is_sandbox: boolean;
  expires_at: string;
}

export interface DocumentRecord {
  doc_id: string;
  filename: string;
  document_category: string;
  chunks_created: number;
  version_label: string;
  upload_date: string;
  is_current_version: boolean;
  supersedes_doc_id: string;
  superseded_by: string;
  has_redline: boolean;
}

export type Severity = "high" | "medium" | "low";

export interface RiskSignal {
  signal_type: string;
  severity: Severity;
  source_file: string;
  description: string;
  page_number: number | null;
}

export interface BudgetEntry {
  used: number;
  limit: number;
  remaining: number;
}

export type BudgetStatus = Record<string, BudgetEntry>;

/**
 * One row of the live pipeline timeline.
 *
 * `status` is client-side only: the backend emits a stage only once it has
 * completed, so "running" is inferred as "the next planned stage after the last
 * one that reported".
 */
export interface PipelineStage {
  seq: number;
  agent: string;
  label: string;
  description: string;
  summary: string;
  detail: Record<string, unknown>;
  duration_ms: number;
  elapsed_ms: number;
  status: "pending" | "running" | "done";
}

export interface PlannedStage {
  agent: string;
  label: string;
  description: string;
}
