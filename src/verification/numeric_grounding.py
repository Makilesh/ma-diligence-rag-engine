"""
Deterministic numeric grounding.

Every figure an answer states is extracted, normalised to a plain magnitude
(`$452.8 million` -> 452_800_000) and looked up among the figures present in
the retrieved context, normalised the same way. A figure is:

- grounded            — the same value appears in the context (within rounding);
- derived             — not verbatim, but reproducible by one arithmetic step
                        (difference, sum, ratio, percentage change) from two
                        figures the answer itself states AND that are grounded;
- derived_unverified  — not reproducible, but the sentence presents it as a
                        computation ("an increase of", "combined", "approximately"),
                        so it is reported as a warning rather than a fabrication;
- unsupported         — none of the above.

Why deterministic: the previous validator asked a lite model whether a stronger
model's numbers were right. That check cost an LLM call per query and was
unreliable in exactly the place it matters — a lite model will wave through a
transposed digit. String-level grounding cannot be talked out of a mismatch.

Why scale-agnostic matching: data-room tables state their unit once in a header
("(in millions of USD)") and print bare figures ("$452.8"), while answers write
"$452.8 million" or "$452,800,000". When one side of a comparison carries no
explicit unit, the other is allowed to differ by a power of 1,000.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, asdict
from typing import Iterable

# ─── Extraction ────────────────────────────────────────────────────────────────

_SCALE_WORDS = {
    "thousand": 1e3,
    "k": 1e3,
    "million": 1e6,
    "mn": 1e6,
    "mm": 1e6,
    "m": 1e6,
    "billion": 1e9,
    "bn": 1e9,
    "b": 1e9,
    "trillion": 1e12,
    "tn": 1e12,
}

# One pattern for every figure shape found in the sample data room and in the
# answers recorded in RESULTS.md. Group order matters: the scale must be tried
# before the unit so "12.0M" is a scaled count, not "12.0" followed by noise.
_NUMBER_RE = re.compile(
    r"""
    (?<![\w.])                                   # not the tail of FY2023, p.3, v2
    (?P<cur>US\$|USD\s?|EUR\s?|GBP\s?|\$|€|£)?
    (?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)
    (?:
        (?P<scale_attached>MM|mm|[Bb]n|BN|mn|M|B|K|k|m|b)(?![A-Za-z])
      | \s?(?P<scale_word>thousand|million|billion|trillion|mn|bn|tn)s?\b
    )?
    (?P<cur_suffix>\s?(?:USD|EUR|GBP)\b)?
    (?P<unit>
        \s?%
      | \s?(?:percentage\s+points?|ppts?|pp)\b
      | \s?(?:percent|per\s?cent)\b
      | \s?(?:bps|basis\s+points?)\b
      | [x×](?![A-Za-z])
    )?
    (?![\w])                                      # rejects 280G, 1st, 3rd
    """,
    re.VERBOSE,
)

# Citation markers carry page numbers, fiscal years and — in live output — whole
# section titles such as "Revenue Growth: 17.0% YoY". None of that is a claim.
_CITATION_MARKER_RE = re.compile(r"\[[^\[\]]*\|[^\[\]]*\]")

# Figures that are references, not quantities: "Section 4.3", "Note 5", "p. 3".
# A leading reference word disqualifies the number that follows it.
_REFERENCE_PREFIX_RE = re.compile(
    r"(?:\b(?:section|sections|§|article|note|notes|schedule|exhibit|item|clause|"
    r"rule|page|pages|pp?\.|appendix|annex|tab|row|sheet|slide|step|tier|level|"
    r"phase|series|form|reg\.?|regulation)\s*$)",
    re.IGNORECASE,
)

# Month-name dates: the day number is not a figure.
_MONTH_BEFORE_RE = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s*$",
    re.IGNORECASE,
)

_NUMERIC_DATE_RE = re.compile(r"\b\d{1,4}[/-]\d{1,2}[/-]\d{1,4}\b")


@dataclass(frozen=True)
class NumericMention:
    """
    One figure found in text, normalised.

    Attributes:
        raw: The matched text, e.g. "$452.8 million".
        value: Magnitude with any explicit scale applied (sign dropped — answers
            say "declined by $1.8M" where tables print "($1.8)").
        kind: currency | percent | pp | bps | multiple | count | plain.
        explicit_scale: True when the text itself stated the unit (M, million).
        scale: The explicit scale factor (1.0 when none).
        decimals: Digits after the decimal point as printed — drives the
            rounding tolerance.
        start: Offset in the source text.
        end: End offset in the source text.
    """

    raw: str
    value: float
    kind: str
    explicit_scale: bool
    scale: float
    decimals: int
    start: int
    end: int

    @property
    def half_unit(self) -> float:
        """Half of the last printed digit, in normalised units — rounding slack."""
        return 0.5 * (10 ** -self.decimals) * self.scale

    @property
    def is_claim(self) -> bool:
        """
        True for figures an answer asserts as a quantity.

        Plain integers without separators or units ("284", "5", "12") are
        excluded: in prose they are overwhelmingly list numbers, clause numbers,
        day counts and headcounts, and treating them as financial claims
        produces noise that trains reviewers to ignore the check.
        """
        return self.kind != "plain"


def strip_citations(text: str) -> str:
    """
    Removes inline citation markers, preserving offsets' meaning for the rest.

    Args:
        text: Answer text.

    Returns:
        Text with every `[... | ...]` marker replaced by a space.
    """
    return _CITATION_MARKER_RE.sub(" ", text or "")


def _is_year(num: str, has_unit: bool) -> bool:
    """True for a bare four-digit 19xx/20xx integer used as a year."""
    return (
        not has_unit
        and len(num) == 4
        and num.isdigit()
        and num[:2] in ("19", "20")
    )


def extract_numbers(text: str, claims_only: bool = True) -> list[NumericMention]:
    """
    Extracts every figure from text.

    Args:
        text: Text to scan. Citation markers should already be stripped when
            scanning an answer (see strip_citations).
        claims_only: When True, drop "plain" numbers (no unit, scale, or
            thousands separator). Answers are scanned with True; context is
            scanned with False, because tables print bare figures.

    Returns:
        NumericMention list in text order.
    """
    if not text:
        return []

    # Blank out numeric dates first so "12/31/2023" does not yield 12, 31, 2023.
    scrubbed = _NUMERIC_DATE_RE.sub(lambda m: " " * len(m.group(0)), text)

    mentions: list[NumericMention] = []
    for m in _NUMBER_RE.finditer(scrubbed):
        num = m.group("num")
        cur = (m.group("cur") or "").strip() or (m.group("cur_suffix") or "").strip()
        scale_token = m.group("scale_attached") or m.group("scale_word")
        unit = (m.group("unit") or "").strip().lower()

        # Lower-case single-letter scales are only a scale next to a currency:
        # "5m" is as likely minutes or metres; "$5m" is five million.
        if scale_token in ("m", "b", "k") and not cur:
            # Re-read without the scale: still a figure, just unscaled.
            scale_token = None

        has_unit = bool(cur or scale_token or unit)
        if _is_year(num, has_unit):
            continue

        prefix = scrubbed[max(0, m.start() - 24): m.start()]
        if _REFERENCE_PREFIX_RE.search(prefix) and not (cur or unit):
            continue
        if _MONTH_BEFORE_RE.search(prefix) and not has_unit:
            continue

        mantissa = float(num.replace(",", ""))
        decimals = len(num.split(".")[1]) if "." in num else 0
        scale = _SCALE_WORDS.get(scale_token.lower(), 1.0) if scale_token else 1.0

        # "percentage points" must be tested before "percent", which prefixes it.
        if unit.startswith("percentage") or unit.startswith("pp"):
            kind = "pp"
        elif unit.startswith("%") or unit.startswith("per"):
            kind = "percent"
        elif unit.startswith("bps") or unit.startswith("basis"):
            kind = "bps"
        elif unit in ("x", "×"):
            kind = "multiple"
        elif cur:
            kind = "currency"
        elif scale_token or "," in num:
            kind = "count"
        else:
            kind = "plain"

        mention = NumericMention(
            raw=m.group(0).strip(),
            value=abs(mantissa) * scale,
            kind=kind,
            explicit_scale=scale_token is not None,
            scale=scale,
            decimals=decimals,
            start=m.start(),
            end=m.end(),
        )
        if claims_only and not mention.is_claim:
            continue
        mentions.append(mention)

    return mentions


# ─── Context index ─────────────────────────────────────────────────────────────

# Unit-bearing kinds conflict with each other; count/plain are wildcards because
# tables routinely print a currency or a ratio without its symbol.
_UNIT_KINDS = {"currency", "percent", "pp", "bps", "multiple"}
_SCALES = (1.0, 1e3, 1e6, 1e9)

# Floor on relative tolerance, for a claim printed with more precision than the
# context rounds to ("$92,800,000" against a thousands table's "$92,812").
_REL_TOLERANCE = 5e-4


def _kinds_compatible(claim_kind: str, ctx_kind: str) -> bool:
    """True when a claim of one kind may be grounded by a context figure of another."""
    if claim_kind == ctx_kind:
        return True
    if claim_kind in ("pp",) and ctx_kind == "percent":
        return True
    if claim_kind in _UNIT_KINDS and ctx_kind in _UNIT_KINDS:
        return False
    return True


@dataclass
class ContextFigure:
    """A figure found in a context chunk, with where it came from."""

    mention: NumericMention
    chunk_index: int
    source_file: str
    snippet: str


class ContextIndex:
    """
    Sorted index of every figure in the retrieved context.

    Each figure is inserted once per plausible scale: its explicit scale when it
    has one, otherwise every power of 1,000 up to a billion. A lookup is then a
    range query, so grounding a whole answer is O(n log m) rather than a nested
    scan.
    """

    def __init__(self, chunks: Iterable[dict]):
        self.figures: list[ContextFigure] = []
        keyed: list[tuple[float, int]] = []

        for ci, chunk in enumerate(chunks):
            text = chunk.get("text", "") or ""
            parent = chunk.get("parent_text", "") or ""
            body = text if not parent or parent in text else f"{text}\n{parent}"
            for mention in extract_numbers(body, claims_only=False):
                lo = max(0, mention.start - 60)
                snippet = body[lo: mention.end + 20].replace("\n", " ").strip()
                fig = ContextFigure(
                    mention=mention,
                    chunk_index=ci,
                    source_file=chunk.get("source_file", "") or "",
                    snippet=snippet,
                )
                idx = len(self.figures)
                self.figures.append(fig)
                # A bare integer ("30", "45") is usually a day count, clause or
                # list number, so it may only ground the same bare value — never
                # "$30 million" via an assumed unit header. Scale-agnostic
                # matching is kept for figures that look like amounts.
                bare_integer = mention.kind == "plain" and mention.decimals == 0
                scales = (1.0,) if mention.explicit_scale or bare_integer else _SCALES
                for s in scales:
                    keyed.append((mention.value * s, idx))

        keyed.sort()
        self._values = [k for k, _ in keyed]
        self._ids = [i for _, i in keyed]

    def __len__(self) -> int:
        return len(self.figures)

    def lookup(self, mention: NumericMention) -> ContextFigure | None:
        """
        Finds a context figure matching a claimed figure, within rounding.

        Args:
            mention: Figure from the answer.

        Returns:
            The first compatible matching ContextFigure, or None.
        """
        scales = (1.0,) if mention.explicit_scale else _SCALES
        for s in scales:
            target = mention.value * s
            tol = max(mention.half_unit * s, _REL_TOLERANCE * target, 1e-9)
            lo = bisect.bisect_left(self._values, target - tol)
            hi = bisect.bisect_right(self._values, target + tol)
            for k in range(lo, hi):
                fig = self.figures[self._ids[k]]
                if not _kinds_compatible(mention.kind, fig.mention.kind):
                    continue
                return fig
        return None


# ─── Grounding ────────────────────────────────────────────────────────────────

# Wording that presents a figure as a computation or approximation. A figure in
# such a sentence that cannot be reproduced is flagged for review, not failed.
_COMPUTATION_CUE_RE = re.compile(
    r"\b(?:increas\w*|decreas\w*|grew|grow\w*|declin\w*|rose|fell|drop\w*|"
    r"change[ds]?|differen\w*|combined|aggregate|sum|impl(?:ied|ies|y)|"
    r"approximately|approx\.?|roughly|nearly|almost|exceed\w*|higher|lower|"
    r"more than|less than|delta|cagr|average|equat\w*|net of|reduction|"
    r"improv\w*|expan\w*|contract(?:ed|ion)|yoy|year-over-year)\b|[~≈+=−]",
    re.IGNORECASE,
)


@dataclass
class NumericCheck:
    """Grounding verdict for one figure in the answer."""

    raw: str
    value: float
    kind: str
    status: str  # grounded | derived | derived_unverified | unsupported
    evidence_source: str = ""
    evidence_snippet: str = ""
    formula: str = ""
    text_index: int = 0

    def to_dict(self) -> dict:
        """Serialisable form for AgentState."""
        return asdict(self)


def _fmt(v: float) -> str:
    """Compact human-readable number for formulas."""
    if v >= 1e9:
        return f"{v / 1e9:,.4g}B"
    if v >= 1e6:
        return f"{v / 1e6:,.4g}M"
    if v >= 1e3:
        return f"{v:,.0f}"
    return f"{v:,.4g}"


def _derive(
    mention: NumericMention,
    operands: list[tuple[float, str]],
) -> str | None:
    """
    Tries to reproduce a figure from two grounded figures stated alongside it.

    Deliberately restricted to grounded operands from the SAME claim — the
    sentence or table row the figure sits in. Searching pairs of context
    numbers, or even of the whole answer's figures, finds a coincidental match
    for almost any target: tried against the full RESULTS.md answer set, a
    whole-answer operand pool "derived" 86.3% as (24.8M - 181.1M) / 181.1M.
    Analysts state their inputs next to their result ("+$65.7M (+17.0%)" in the
    row holding $452.8M and $387.1M; "exceeds the $3.48M deductible by $4.52
    million" beside the $8.0M exposure), so same-claim operands keep the real
    derivations while making coincidences rare.

    Args:
        mention: Ungrounded figure from the answer.
        operands: (normalised value, kind) of grounded figures in the same claim.

    Returns:
        A formula string when reproducible, else None.
    """
    kind = mention.kind
    scales = (
        (1.0,)
        if mention.explicit_scale or kind in ("percent", "pp", "bps", "multiple")
        else _SCALES
    )

    def close(target: float, got: float, s: float) -> bool:
        tol = max(mention.half_unit * s, _REL_TOLERANCE * abs(target), 1e-9)
        return abs(target - got) <= tol

    amounts = [(v, k) for v, k in operands if k in ("currency", "count")]
    percents = [v for v, k in operands if k == "percent"]
    multiples = [v for v, k in operands if k == "multiple"]
    basis_points = [v for v, k in operands if k == "bps"]

    def amount_pairs():
        """Ordered pairs of distinct amounts of the same kind."""
        for i, (a, ka) in enumerate(amounts):
            for j, (b, kb) in enumerate(amounts):
                if i != j and ka == kb and b:
                    yield a, b

    for s in scales:
        target = mention.value * s
        if kind in ("currency", "count"):
            for a, b in amount_pairs():
                if close(target, abs(a - b), s):
                    return f"|{_fmt(a)} - {_fmt(b)}|"
                if close(target, a + b, s):
                    return f"{_fmt(a)} + {_fmt(b)}"
        elif kind == "percent":
            for a, b in amount_pairs():
                if close(target, abs(a - b) / b * 100, s):
                    return f"({_fmt(a)} - {_fmt(b)}) / {_fmt(b)}"
                if a < b and close(target, a / b * 100, s):
                    return f"{_fmt(a)} / {_fmt(b)}"
            for i, a in enumerate(percents):
                for b in percents[i + 1:]:
                    if close(target, abs(a - b), s):
                        return f"|{a:g}% - {b:g}%|"
            for a in basis_points:
                if close(target, a / 100, s):
                    return f"{a:g} bps / 100"
        elif kind == "pp":
            for i, a in enumerate(percents):
                for b in percents[i + 1:]:
                    if close(target, abs(a - b), s):
                        return f"|{a:g}% - {b:g}%|"
        elif kind == "bps":
            for i, a in enumerate(percents):
                for b in percents[i + 1:]:
                    if close(target, abs(a - b) * 100, s):
                        return f"|{a:g}% - {b:g}%| x 100"
            for a in percents:
                if close(target, a * 100, s):
                    return f"{a:g}% x 100"
        elif kind == "multiple":
            for a, b in amount_pairs():
                if close(target, a / b, s):
                    return f"{_fmt(a)} / {_fmt(b)}"
            for i, a in enumerate(multiples):
                for b in multiples[i + 1:]:
                    if close(target, abs(a - b), s):
                        return f"|{a:g}x - {b:g}x|"
    return None


def ground_texts(texts: list[str], index: ContextIndex) -> list[list[NumericCheck]]:
    """
    Grounds every figure in a list of answer segments (claims or sentences).

    Two passes per segment: every figure is first looked up verbatim; the ones
    not found are then tried as derivations from the segment's own grounded
    figures (see _derive), and failing that are classed by wording.

    Args:
        texts: Answer segments, citation markers already stripped.
        index: ContextIndex over the retrieved chunks.

    Returns:
        One list of NumericCheck per input text, in order.
    """
    results: list[list[NumericCheck]] = []

    for ti, text in enumerate(texts):
        checks: list[NumericCheck] = []
        pending: list[tuple[NumericMention, NumericCheck]] = []
        operands: list[tuple[float, str]] = []

        for mention in extract_numbers(text, claims_only=True):
            fig = index.lookup(mention)
            if fig is not None:
                # Operand value in the context's normalisation, so a claim
                # printed as "$452.8" (unscaled) still computes in dollars when
                # the context says "$452.8 million".
                value = mention.value
                if not mention.explicit_scale and fig.mention.explicit_scale:
                    value = fig.mention.value
                operands.append((value, mention.kind))
                checks.append(NumericCheck(
                    raw=mention.raw,
                    value=mention.value,
                    kind=mention.kind,
                    status="grounded",
                    evidence_source=fig.source_file,
                    evidence_snippet=fig.snippet[:160],
                    text_index=ti,
                ))
            else:
                check = NumericCheck(
                    raw=mention.raw,
                    value=mention.value,
                    kind=mention.kind,
                    status="unsupported",
                    text_index=ti,
                )
                checks.append(check)
                pending.append((mention, check))

        # Two rounds, so a figure computed from a derived one still resolves:
        # "margin rose 1.74pp" from two margins the sentence itself computed.
        for _ in range(2):
            progressed = False
            for mention, check in pending:
                if check.status == "derived":
                    continue
                formula = _derive(mention, operands)
                if formula:
                    check.status = "derived"
                    check.formula = formula
                    operands.append((mention.value, mention.kind))
                    progressed = True
            if not progressed:
                break

        for mention, check in pending:
            if check.status != "derived" and _COMPUTATION_CUE_RE.search(text):
                check.status = "derived_unverified"

        results.append(checks)

    return results


def ground_answer_numbers(answer: str, chunks: list[dict]) -> list[dict]:
    """
    Grounds every figure in an answer against the context — the answer's
    `numerical_claims`.

    Args:
        answer: Generated answer text (citation markers are stripped here).
        chunks: Retrieved context chunks.

    Returns:
        List of NumericCheck dicts, in answer order.
    """
    from src.verification.claims import split_claims  # deferred: avoids a cycle

    claims = split_claims(answer)
    if not claims:
        return []
    index = ContextIndex(chunks)
    grouped = ground_texts([c.text for c in claims], index)
    return [check.to_dict() for checks in grouped for check in checks]


def summarize_numeric(checks: Iterable[dict | NumericCheck]) -> dict[str, int]:
    """
    Counts numeric checks by status.

    Args:
        checks: NumericCheck objects or their dicts.

    Returns:
        {status: count} over the four statuses, zeros included.
    """
    counts = {"grounded": 0, "derived": 0, "derived_unverified": 0, "unsupported": 0}
    for c in checks:
        status = c["status"] if isinstance(c, dict) else c.status
        counts[status] = counts.get(status, 0) + 1
    return counts


__all__ = [
    "NumericMention",
    "NumericCheck",
    "ContextIndex",
    "extract_numbers",
    "strip_citations",
    "ground_texts",
    "ground_answer_numbers",
    "summarize_numeric",
]
