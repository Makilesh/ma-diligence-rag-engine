/**
 * Parses the inline citations the synthesizer writes into the answer body.
 *
 * The prompt (`src/llm/prompt_templates/answer_synthesizer.py`) instructs the
 * model to cite in pipe-delimited brackets:
 *
 *   [📄 FileName | FY2023 | p.14 | Section | Version]
 *   [📊 FileName | Sheet "Q4" | Row 12 | COMPUTED: YoY growth FY22–FY23]
 *   [⚠ FileName | Section | NOT CURRENT VERSION → superseded by v3]
 *
 * Rendered literally those are unreadable — a full bracket mid-sentence breaks
 * the line and buries the claim. So each is replaced by a numbered marker and
 * the detail moves to a hover card and the sources rail, which is the pattern
 * every reader already knows from academic footnotes.
 */

import type { Citation } from "./types";

export interface ParsedCitation {
  /** 1-based display number, stable across repeat references. */
  index: number;
  sourceFile: string;
  page: number | null;
  slide: number | null;
  sheet: string;
  section: string;
  /** True when the cited document has been superseded by a newer version. */
  isStale: boolean;
  /** True when the figure was computed deterministically, not read verbatim. */
  isComputed: boolean;
  /** The original bracket text, kept for the hover card's "raw" line. */
  raw: string;
  /** Server-side citation record, when one matches this source. */
  record?: Citation;
}

export interface ParsedAnswer {
  /** Answer body with each bracket replaced by a `[n](#cite-n)` link. */
  markdown: string;
  citations: ParsedCitation[];
}

// Bracket groups containing at least one pipe. Requiring the pipe is what keeps
// ordinary markdown links and array indices — `[see below](#x)`, `[0]` — from
// being swallowed as citations.
const CITATION_PATTERN = /\[([^[\]]*\|[^[\]]*)\]/g;

const PAGE_PATTERN = /\bp\.?\s*(\d+)/i;
const SLIDE_PATTERN = /\bslide\s*(\d+)/i;
const SHEET_PATTERN = /\bsheet\s*"?([^"|]+)"?/i;

// The synthesizer prefixes citations with a type emoji. Stripped for display —
// the pill carries the type through colour and icon instead.
const LEADING_ICON = /^[\s\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}\u{FE0F}⚠]+/u;

/** Fields that are structural markers rather than a section name. */
function isNoiseField(field: string): boolean {
  return (
    !field ||
    PAGE_PATTERN.test(field) ||
    SLIDE_PATTERN.test(field) ||
    /^sheet\b/i.test(field) ||
    /^row\b/i.test(field) ||
    /^fy\s*\d{2,4}$/i.test(field) ||
    /not current version/i.test(field) ||
    /^v\d+/i.test(field)
  );
}

/**
 * Parses one bracket's inner text into structured fields.
 *
 * @param inner Text between the brackets, pipes included.
 * @param raw The full original bracket, for the hover card.
 */
function parseOne(inner: string, raw: string): Omit<ParsedCitation, "index"> {
  const fields = inner.split("|").map((f) => f.trim());
  const sourceFile = (fields[0] ?? "").replace(LEADING_ICON, "").trim();

  const pageMatch = inner.match(PAGE_PATTERN);
  const slideMatch = inner.match(SLIDE_PATTERN);
  const sheetMatch = inner.match(SHEET_PATTERN);

  // The section is whichever trailing field is not one of the structural
  // markers — the model's field order is not fixed, so position is unreliable.
  const section =
    fields
      .slice(1)
      .map((f) => f.trim())
      .filter((f) => !isNoiseField(f) && !/^computed:/i.test(f))
      .pop() ?? "";

  return {
    sourceFile,
    page: pageMatch ? Number(pageMatch[1]) : null,
    slide: slideMatch ? Number(slideMatch[1]) : null,
    sheet: sheetMatch ? sheetMatch[1].trim() : "",
    section,
    isStale: /not current version|superseded/i.test(inner),
    isComputed: /computed:/i.test(inner),
    raw,
  };
}

/**
 * Identity for deduplication.
 *
 * Two references to the same page of the same file share a number, the way a
 * footnote would. Section is included because a long document cited at two
 * sections is genuinely two references for the reader.
 */
function dedupeKey(c: Omit<ParsedCitation, "index">): string {
  return [c.sourceFile, c.page ?? "", c.slide ?? "", c.sheet, c.section]
    .join("::")
    .toLowerCase();
}

/**
 * Rewrites an answer's inline citations into numbered links.
 *
 * @param answer Raw markdown from the synthesizer.
 * @param records Server-side citation list, used to enrich each parsed marker.
 *
 * @returns The rewritten markdown plus the ordered citation list.
 */
export function parseAnswer(answer: string, records: Citation[] = []): ParsedAnswer {
  if (!answer) return { markdown: "", citations: [] };

  const byKey = new Map<string, ParsedCitation>();
  const ordered: ParsedCitation[] = [];

  const markdown = answer.replace(CITATION_PATTERN, (raw, inner: string) => {
    const parsed = parseOne(inner, raw);

    // A bracket with pipes but no plausible filename is not a citation — most
    // often it is a table fragment the model wrapped in brackets. Leave it alone.
    if (!parsed.sourceFile) return raw;

    const key = dedupeKey(parsed);
    let citation = byKey.get(key);

    if (!citation) {
      citation = {
        ...parsed,
        index: ordered.length + 1,
        record: matchRecord(parsed, records),
      };
      byKey.set(key, citation);
      ordered.push(citation);
    }

    return `[${citation.index}](#cite-${citation.index})`;
  });

  return { markdown, citations: ordered };
}

/**
 * Finds the server-side citation record backing a parsed marker.
 *
 * Matched on filename first, then narrowed by page when the server knows one.
 * The model writes the filename from the retrieved chunk's metadata, so exact
 * matches are the norm; the `includes` fallback covers the case where it
 * shortens a long path.
 */
function matchRecord(
  parsed: Omit<ParsedCitation, "index">,
  records: Citation[],
): Citation | undefined {
  const name = parsed.sourceFile.toLowerCase();
  if (!name) return undefined;

  const candidates = records.filter((r) => {
    const rn = (r.source_file ?? "").toLowerCase();
    return rn === name || rn.includes(name) || name.includes(rn);
  });

  if (!candidates.length) return undefined;
  if (parsed.page !== null) {
    const exact = candidates.find((r) => r.page_number === parsed.page);
    if (exact) return exact;
  }
  return candidates[0];
}

/**
 * Builds the one-line label shown on a citation pill's hover card and rail row.
 */
export function citationLabel(c: ParsedCitation): string {
  const parts: string[] = [];
  if (c.page !== null) parts.push(`p. ${c.page}`);
  if (c.slide !== null) parts.push(`slide ${c.slide}`);
  if (c.sheet) parts.push(`sheet ${c.sheet}`);
  if (c.section) parts.push(c.section);
  return parts.join(" · ");
}

/**
 * Strips the extension and separators from a filename for compact display.
 */
export function prettyFilename(filename: string): string {
  return filename
    .replace(/\.[a-z0-9]+$/i, "")
    .replace(/[_-]+/g, " ")
    .trim();
}
