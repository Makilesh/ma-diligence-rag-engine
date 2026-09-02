/**
 * Product naming, in one place.
 *
 * "Redline" is the term of art for a marked-up contract draft, and the engine
 * already tracks redlines as first-class metadata (`is_redline` on every chunk
 * payload). Change these three constants to rename the product everywhere.
 */

export const BRAND = {
  name: "Redline",
  tagline: "M&A Due Diligence Intelligence",
  /** The thesis, stated in one line. Used on the landing state. */
  promise:
    "Agentic retrieval across the data room — every claim traced to a source, every unsupported claim refused.",
} as const;
