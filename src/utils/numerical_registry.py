"""
Numerical registry — collects labelled financial figures across documents and
detects cross-document inconsistencies deterministically.

Used by Agent 4 (Financial Verifier). Figures come from two places:

- structured payloads written at ingestion (`structured_rows` on row-by-row
  table chunks, or a chunk-level `metric_name` / `normalized_value`);
- fixed-width tables in text chunks, parsed line by line: a period header
  ("FY2023   FY2022") followed by label + value rows ("Revenue  $452.8  $387.1").

An inconsistency is the same labelled metric, for the same period, reported
with different values by DIFFERENT sources, after unit normalisation and
allowing for the printed rounding. Two sources disagreeing is a fact about the
data room that a reviewer must see; a single source restating a figure is not.

CRITICAL: Compare normalised values only. Raw values across documents with
different scale factors produce false inconsistencies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.utils.logger import setup_logger
from src.verification.numeric_grounding import extract_numbers

logger = setup_logger(__name__)

# Period tokens recognised in table headers and structured-row columns.
_PERIOD = re.compile(
    r"\b(?:FY\s?'?(?:19|20)?\d{2}|(?:Q[1-4]|H[12])\s?(?:FY)?\s?'?(?:19|20)?\d{2}|"
    r"(?:19|20)\d{2}[EA]?|LTM|TTM)\b",
    re.IGNORECASE,
)
_SCALE_HINT = re.compile(
    r"\(\s*(?:\$|USD|US\$|EUR|GBP)?\s*(?:in\s+)?(thousands|millions|billions)\b"
    r"|\bin\s+(thousands|millions|billions)\s+of\b"
    r"|\b(?:USD|\$)\s*(thousands|millions|billions|000s|mm|m)\b",
    re.IGNORECASE,
)
_SCALE_WORD = {
    "thousands": 1e3, "000s": 1e3,
    "millions": 1e6, "mm": 1e6, "m": 1e6,
    "billions": 1e9,
}
_RULE_LINE = re.compile(r"^\s*[=\-_*~]{4,}\s*$")
_LABEL_PREFIX = re.compile(r"^(?:add(?:\s+back)?|less|plus|minus)\s*:\s*", re.IGNORECASE)
# Parentheticals that annotate a figure without changing what it measures.
# "(Non-Current)" or "(excluding restructuring)" DO change it, so they stay.
_ANNOTATION = re.compile(
    r"\(\s*(?:note\s*\d+[a-z]?|as\s+reported|reported|audited|unaudited|restated)\s*\)",
    re.IGNORECASE,
)
_SCALES = (1.0, 1e3, 1e6, 1e9)


def normalize_metric_label(label: str) -> str:
    """
    Canonical form of a line-item label, for matching across documents.

    Drops add/less prefixes, annotation parentheticals ("(Note 1)", "(as
    reported)") and the word "reported", so the QoE report's "Reported EBITDA"
    and the financials' "EBITDA" compare as one metric — which they are. Other
    parentheticals are kept: "Deferred Revenue (Non-Current)" is not "Deferred
    Revenue".

    Args:
        label: Raw label text.

    Returns:
        Lower-case, punctuation-free label.
    """
    text = _LABEL_PREFIX.sub("", label.strip())
    text = _ANNOTATION.sub(" ", text)
    text = re.sub(r"\breported\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-z0-9%/&+ ]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def normalize_period(token: str) -> str:
    """Canonical period form: "FY 2023", "FY'23", "fy23" -> "FY2023"."""
    t = re.sub(r"[\s']", "", token.upper())
    m = re.fullmatch(r"FY(\d{2})", t)
    if m:
        return f"FY20{m.group(1)}"
    m = re.fullmatch(r"(19|20)(\d{2})[EA]?", t)
    if m:
        return f"FY{m.group(1)}{m.group(2)}"
    return t


@dataclass
class NumericalEntry:
    """A single labelled figure from a source document."""

    metric_name: str
    raw_value: float
    normalized_value: float
    currency: str = "USD"
    scale_factor: float = 1.0
    source_file: str = ""
    page_number: int = 0
    fiscal_year: str = ""
    is_computed: bool = False
    citation_chain: str = ""
    kind: str = "currency"
    decimals: int = 1
    scale_known: bool = True
    raw_text: str = ""

    @property
    def tolerance(self) -> float:
        """Half of the last printed digit, in normalised units."""
        return 0.5 * (10 ** -self.decimals) * self.scale_factor


@dataclass
class MetricComparison:
    """Comparison result for a metric/period across sources."""

    metric_name: str
    entries: list[NumericalEntry] = field(default_factory=list)
    is_consistent: bool = True
    max_deviation_pct: float = 0.0
    discrepancy_detail: str = ""
    fiscal_year: str = ""


def _values_agree(a: NumericalEntry, b: NumericalEntry) -> bool:
    """
    True when two entries state the same figure, allowing for rounding and — if
    either side's unit is unknown — for a power-of-1,000 scale difference.
    """
    if a.kind != b.kind and {a.kind, b.kind} & {"percent", "multiple"}:
        return False
    tol = max(a.tolerance, b.tolerance, 1e-9)
    if a.scale_known and b.scale_known:
        return abs(a.normalized_value - b.normalized_value) <= tol
    for s in _SCALES:
        for x, y in ((a, b), (b, a)):
            if abs(x.normalized_value - y.normalized_value * s) <= max(tol * s, tol):
                return True
    return False


class NumericalRegistry:
    """
    Collects labelled figures and compares them across documents.

    Entries are keyed by (normalised metric label, period). Only entries with a
    period take part in consistency checks: "Revenue $452.8" and "Revenue
    $387.1" are not a contradiction if one is FY2023 and the other FY2022, and a
    figure whose period is unknown cannot be compared safely.
    """

    def __init__(self):
        self._entries: dict[tuple[str, str], list[NumericalEntry]] = {}

    def __len__(self) -> int:
        return sum(len(v) for v in self._entries.values())

    def register(self, entry: NumericalEntry) -> None:
        """
        Registers a figure, ignoring exact duplicates from the same source.

        Args:
            entry: NumericalEntry to register.
        """
        key = (normalize_metric_label(entry.metric_name), entry.fiscal_year)
        bucket = self._entries.setdefault(key, [])
        for existing in bucket:
            if existing.source_file == entry.source_file and _values_agree(existing, entry):
                return
        bucket.append(entry)

    # ── Building from chunks ───────────────────────────────────────────────

    @classmethod
    def from_chunks(cls, chunks: list[dict]) -> "NumericalRegistry":
        """
        Builds a registry from retrieved chunks.

        Args:
            chunks: Context chunk payloads.

        Returns:
            Populated NumericalRegistry.
        """
        registry = cls()
        for chunk in chunks:
            source = chunk.get("source_file", "") or "unknown"
            page = chunk.get("page_number") or 0
            registry._register_structured(chunk, source, page)
            text = chunk.get("text", "") or ""
            parent = chunk.get("parent_text", "") or ""
            for body in (text, parent):
                if body:
                    registry._register_text_tables(body, chunk, source, page)
        return registry

    def _register_structured(self, chunk: dict, source: str, page) -> None:
        """Registers figures from structured payload fields, when present."""
        for row in chunk.get("structured_rows") or []:
            label = str(row.get("line_item", "")).strip()
            if not label:
                continue
            for col, cell in (row.get("values") or {}).items():
                if not isinstance(cell, dict) or "normalized_value" not in cell:
                    continue
                try:
                    value = abs(float(cell["normalized_value"]))
                    raw = float(cell.get("raw_value", value))
                except (TypeError, ValueError):
                    continue
                period = _PERIOD.search(str(col))
                raw_str = repr(raw)
                decimals = len(raw_str.split(".")[1]) if "." in raw_str and not raw_str.endswith(".0") else 0
                self.register(NumericalEntry(
                    metric_name=label,
                    raw_value=raw,
                    normalized_value=value,
                    currency=cell.get("currency", "USD") or "USD",
                    scale_factor=float(cell.get("scale_factor", 1.0) or 1.0),
                    source_file=source,
                    page_number=page,
                    fiscal_year=normalize_period(period.group(0)) if period else "",
                    decimals=decimals,
                    scale_known=True,
                    raw_text=f"{label} {col}={raw}",
                ))

        if chunk.get("metric_name") and chunk.get("normalized_value") is not None:
            try:
                value = abs(float(chunk["normalized_value"]))
            except (TypeError, ValueError):
                return
            fy = chunk.get("fiscal_year") or ""
            self.register(NumericalEntry(
                metric_name=str(chunk["metric_name"]),
                raw_value=float(chunk.get("raw_value") or value),
                normalized_value=value,
                currency=chunk.get("currency", "USD") or "USD",
                scale_factor=float(chunk.get("scale_factor", 1.0) or 1.0),
                source_file=source,
                page_number=page,
                fiscal_year=normalize_period(str(fy)) if fy else "",
                is_computed=chunk.get("content_type") == "computed_metric",
                scale_known=True,
            ))

    def _register_text_tables(self, text: str, chunk: dict, source: str, page) -> None:
        """
        Parses fixed-width tables in text: a period header, then label/value rows.

        A row is registered only when its value count matches the header's
        period count, so a prose line that happens to end in a number is never
        mistaken for a table row.
        """
        hint = _SCALE_HINT.search(text)
        chunk_scale = None
        if hint:
            word = next(g for g in hint.groups() if g).lower()
            chunk_scale = _SCALE_WORD.get(word)
        if chunk_scale is None and chunk.get("scale_factor"):
            try:
                chunk_scale = float(chunk["scale_factor"])
            except (TypeError, ValueError):
                chunk_scale = None

        periods: list[str] = []
        for line in text.splitlines():
            if not line.strip() or _RULE_LINE.match(line):
                continue

            header_periods = _PERIOD.findall(line)
            if header_periods:
                remainder = _PERIOD.sub(" ", line)
                if not extract_numbers(remainder, claims_only=False):
                    periods = [normalize_period(p) for p in header_periods]
                    continue

            numbers = extract_numbers(line, claims_only=False)
            if not numbers or not periods or len(numbers) != len(periods):
                continue
            # "($8.2)" leaves its opening parenthesis on the label side.
            label = line[: numbers[0].start].strip().rstrip("(").strip().rstrip(":").strip()
            # Values must be the tail of the line — a table row, not prose.
            tail = line[numbers[-1].end:].strip(" )")
            if tail or len(re.findall(r"[A-Za-z]", label)) < 3:
                continue

            for period, num in zip(periods, numbers):
                unit_scaled = num.kind in ("percent", "pp", "bps", "multiple")
                scale_known = num.explicit_scale or unit_scaled or chunk_scale is not None
                scale = 1.0 if (num.explicit_scale or unit_scaled) else (chunk_scale or 1.0)
                self.register(NumericalEntry(
                    metric_name=label,
                    raw_value=num.value / (num.scale or 1.0),
                    normalized_value=num.value * scale,
                    scale_factor=num.scale * scale,
                    source_file=source,
                    page_number=page,
                    fiscal_year=period,
                    kind="percent" if num.kind in ("percent", "pp") else (
                        "multiple" if num.kind == "multiple" else "currency"
                    ),
                    decimals=num.decimals,
                    scale_known=scale_known,
                    raw_text=f"{label}  {num.raw}",
                ))

    # ── Comparison ─────────────────────────────────────────────────────────

    def check_consistency(self) -> list[MetricComparison]:
        """
        Compares every metric/period reported by more than one source.

        Returns:
            One MetricComparison per (metric, period) with 2+ distinct sources.
        """
        results = []
        for (_metric, period), entries in self._entries.items():
            if not period:
                continue
            sources = {e.source_file for e in entries}
            if len(sources) < 2:
                continue

            disagreements = []
            max_dev = 0.0
            for i, a in enumerate(entries):
                for b in entries[i + 1:]:
                    if a.source_file == b.source_file or _values_agree(a, b):
                        continue
                    base = max(abs(a.normalized_value), 1e-9)
                    dev = abs(a.normalized_value - b.normalized_value) / base * 100
                    max_dev = max(max_dev, dev)
                    disagreements.append(
                        f"{a.source_file}: {a.raw_text.split('  ')[-1] or a.normalized_value} vs "
                        f"{b.source_file}: {b.raw_text.split('  ')[-1] or b.normalized_value}"
                    )
            results.append(MetricComparison(
                metric_name=entries[0].metric_name,
                entries=entries,
                is_consistent=not disagreements,
                max_deviation_pct=round(max_dev, 2),
                discrepancy_detail="; ".join(disagreements),
                fiscal_year=period,
            ))
        return results

    def find_inconsistencies(self) -> list[dict]:
        """
        Cross-document disagreements, in the shape the synthesizer consumes.

        Severity is by relative size of the gap: >5% high, >1% medium, else low.
        Every item carries `method: "deterministic"` — these are measured facts
        about the documents, not a model's opinion.

        Returns:
            List of inconsistency dicts.
        """
        out = []
        for comp in self.check_consistency():
            if comp.is_consistent:
                continue
            dev = comp.max_deviation_pct
            severity = "high" if dev > 5 else "medium" if dev > 1 else "low"
            values = [
                {
                    "source": e.source_file,
                    "value": e.normalized_value,
                    "as_printed": e.raw_text.split("  ")[-1],
                    "page": e.page_number or None,
                }
                for e in comp.entries
            ]
            out.append({
                "metric": comp.metric_name,
                "fiscal_year": comp.fiscal_year,
                "values_found": values,
                "discrepancy_type": "value_mismatch",
                "severity": severity,
                "max_deviation_pct": dev,
                "explanation": (
                    f"{comp.metric_name} for {comp.fiscal_year} differs across sources: "
                    f"{comp.discrepancy_detail} ({dev:.1f}% apart)"
                ),
                "method": "deterministic",
            })
        return out

    def cross_checked_count(self) -> int:
        """Number of metric/period pairs reported by more than one source."""
        return len(self.check_consistency())

    def to_dict(self) -> dict:
        """
        Serialises the registry for AgentState.

        Returns:
            {"<metric> | <period>": {metric, fiscal_year, values, is_consistent}}.
        """
        consistency = {
            (normalize_metric_label(c.metric_name), c.fiscal_year): c.is_consistent
            for c in self.check_consistency()
        }
        out = {}
        for (metric, period), entries in self._entries.items():
            out[f"{metric} | {period or 'unknown period'}"] = {
                "metric": entries[0].metric_name,
                "fiscal_year": period,
                "values": [
                    {
                        "source": e.source_file,
                        "raw_value": e.raw_value,
                        "normalized_value": e.normalized_value,
                        "currency": e.currency,
                        "fiscal_year": e.fiscal_year,
                        "page": e.page_number or None,
                    }
                    for e in entries
                ],
                "is_consistent": consistency.get((metric, period), True),
            }
        return out
