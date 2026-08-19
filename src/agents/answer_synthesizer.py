"""
Agent 7 — Answer Synthesis Agent.

Model: chosen at call time from BudgetTracker's SYNTHESIS_LADDER — best
available reasoning model, degrading through the ladder as daily quota is
spent and finally to the local model. See src/llm/model_registry.py.
Temp: 0.1 | Tokens: 3000 | JSON mode: OFF (prose answer)
"""

import json
import re

from src.llm.litellm_wrapper import (
    call_prose_agent,
    is_quota_error,
    is_auth_error,
    is_service_unavailable,
    is_model_unavailable_for_key,
)
from src.llm.budget_tracker import BudgetTracker
from src.llm.prompt_templates.answer_synthesizer import (
    ANSWER_SYNTHESIZER_SYSTEM_PROMPT,
    ANSWER_SYNTHESIZER_USER_TEMPLATE,
)
from src.workflow.state_definitions import AgentState
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# How many ladder rungs to try before giving up on a query. Bounded so a
# provider-wide outage cannot walk the entire ladder on every request.
MAX_LADDER_FALLBACKS = 6


def _format_context_for_synthesis(chunks: list[dict]) -> str:
    """
    Formats expanded context chunks for the synthesis prompt.
    Includes metadata for citation generation.

    Args:
        chunks: List of expanded context chunk dicts.

    Returns:
        Formatted string for inclusion in the synthesis prompt.
    """
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = []
        if chunk.get("source_file"):
            meta.append(f"Source: {chunk['source_file']}")
        if chunk.get("page_number"):
            meta.append(f"Page: {chunk['page_number']}")
        if chunk.get("section_heading"):
            meta.append(f"Section: {chunk['section_heading']}")
        if chunk.get("fiscal_year"):
            meta.append(f"FY: {chunk['fiscal_year']}")
        if chunk.get("is_current_version") == 0:
            meta.append("⚠ NOT CURRENT VERSION")
        if chunk.get("content_type") == "computed_metric":
            meta.append("COMPUTED METRIC")
        if chunk.get("is_redline"):
            meta.append("REDLINE VERSION")

        meta_str = " | ".join(meta)
        text = chunk.get("text", "")
        parent_text = chunk.get("parent_text", "")

        part = f"--- Chunk {i} [{meta_str}] ---\n{text}"
        if parent_text:
            part += f"\n[Parent context]: {parent_text[:500]}"
        parts.append(part)

    return "\n\n".join(parts)


# A citation marker is any bracketed group containing a pipe, which is the
# format the synthesis prompt specifies:
#   [📄 FileName | FiscalYear | p.PageNum | Section | Version]
#   [📊 FileName | Sheet "Name" | COMPUTED: description]
# Matching on the pipe avoids colliding with ordinary markdown links.
_CITATION_MARKER = re.compile(r"\[([^\[\]]*\|[^\[\]]*)\]")
_PAGE_IN_MARKER = re.compile(r"(?:p\.|pg\.|page\s*)(\d+)", re.IGNORECASE)


# Fragments that mean the model emitted its own working notes instead of an
# answer. Seen live during a provider incident: one response opened mid-thought
# with "Wu's patents: [...] Let's double-check all details to ensure accuracy",
# which shipped to the user as a finished answer with 0.85 confidence.
_SCRATCHPAD_MARKERS = (
    "let's double-check",
    "let me double-check",
    "let's verify",
    "let me verify",
    "wait, i need to",
    "actually, let me",
)

# Below this, a response is a fragment rather than a due-diligence answer.
# Genuine answers in this pipeline run 2,000-4,500 characters; the truncated
# generations during the incident came back at 349-575.
_MIN_ANSWER_CHARS = 220

# An answer may legitimately carry no citations when it is saying the corpus
# does not support the question. That is a designed outcome, not a failure —
# and often the *better* one, since naming the specific gap beats a blanket
# refusal (see DECISIONS_LOG Decision 22).
#
# This list exists because the citation guard originally had no exception for
# it, and the result was worse than the bug it was added to fix: asked for a
# multiple the corpus cannot support, gemini-3.6-flash, gemini-3.5-flash and
# gemini-3-flash-preview each correctly replied that the transaction value was
# not stated — 285-438 characters, no markers to cite — and each was rejected
# as a failed generation. The ladder burned three scarce 20-RPD reasoning slots
# in a row, took 183 seconds, and then accepted the *weakest* model's answer.
# A guard that spends the good models to reach the worst one is worse than no
# guard.
_DECLINES_TO_ANSWER = re.compile(
    r"do(?:es)? not contain|not contain(?:ed)?"
    r"|insufficient|not sufficient|unable to (?:find|determine|calculate|compute)"
    r"|cannot be (?:calculated|computed|determined|established)"
    r"|no (?:information|evidence|mention|disclosure|reference|record|data)"
    r"|not (?:disclosed|provided|available|specified|present|stated|included)"
    r"|is absent|are absent",
    re.IGNORECASE,
)


def _is_usable_answer(answer: str | None, chunks: list[dict]) -> bool:
    """
    True when a generation is worth returning rather than retrying.

    The synthesis prompt requires an inline `[file | page | section]` marker on
    every claim, so a response carrying none of them has not followed the
    contract. That normally cannot happen — but under a provider incident,
    gemini-3.6-flash returned answers at 10-15% of their usual length with zero
    citations, and the pipeline shipped them: one scored `validation=passed`
    with `confidence=1.0` and no sources at all, in a tool whose entire premise
    is that every claim is traced to a document.

    An uncited answer is therefore treated as a failed generation, not a weak
    one. Retrying costs a few seconds; publishing an unsourced figure in a due
    diligence report is the failure this project exists to prevent.

    Deliberately narrow, in two directions. It only rejects when there is
    evidence to cite, so a genuine refusal with no usable context passes
    through. And it accepts an uncited answer that *declines* — one saying the
    corpus does not support the question — because that is a designed outcome
    with nothing to cite, not a failed generation.

    Args:
        answer: Raw model output.
        chunks: Context the model was given.

    Returns:
        True if the answer should be accepted.
    """
    if answer is None or not answer.strip():
        return False

    text = answer.strip()

    # No evidence was supplied, so an uncited answer is the correct output.
    if not chunks:
        return True

    if len(text) < _MIN_ANSWER_CHARS:
        return False

    lowered = text.lower()
    if any(marker in lowered for marker in _SCRATCHPAD_MARKERS):
        return False

    if _CITATION_MARKER.search(text):
        return True

    # No citations. Acceptable only when the answer is declining rather than
    # asserting — an answer that states facts with nothing to trace them to is
    # the failure this guard exists for; an answer that says the corpus does not
    # support the question has nothing to trace in the first place.
    return bool(_DECLINES_TO_ANSWER.search(text))


def _select_cited_chunks(answer: str, chunks: list[dict]) -> list[dict]:
    """
    Returns the chunks the answer actually cited, in the order they were used.

    The naive approach — treat a chunk as cited if its filename appears anywhere
    in the answer — is document-level, not passage-level. When several retrieved
    chunks come from one document, all of them are returned, so the citation
    panel names sections the answer never touched. Observed on a live run: an
    answer citing only "SECTION 8. INDEMNIFICATION" listed citations labelled
    SECTION 7 and SECTION 5.

    Narrowing by `section_heading` does not fix it either: that heading describes
    the *chunk*, and a chunk routinely spans several sections, so the section the
    model names is often absent from the chunk's metadata.

    What does work is reading the markers the model already emits. It states the
    file and, for paginated sources, the page it used. Matching on file plus page
    resolves to the passage rather than the document.

    Falls back to document-level matching when the model emitted no parseable
    marker, because an over-broad citation list is still far better than none.

    Args:
        answer: Generated answer text containing inline citation markers.
        chunks: Candidate context chunks with source_file / page_number.

    Returns:
        Cited chunks, de-duplicated, preserving retrieval order.
    """
    markers = []
    for body in _CITATION_MARKER.findall(answer or ""):
        page_match = _PAGE_IN_MARKER.search(body)
        markers.append((body.lower(), int(page_match.group(1)) if page_match else None))

    def cited_by_marker(chunk: dict) -> bool:
        source = (chunk.get("source_file") or "").lower()
        if not source:
            return False
        for body, page in markers:
            if source not in body:
                continue
            chunk_page = chunk.get("page_number")
            # A page in the marker narrows to that page; its absence means the
            # model did not scope the citation, so the document match stands.
            if page is not None and chunk_page is not None and int(chunk_page) != page:
                continue
            return True
        return False

    selected = [c for c in chunks if cited_by_marker(c)]

    if not selected:
        answer_text = answer or ""
        selected = [
            c for c in chunks
            if c.get("source_file") and c["source_file"] in answer_text
        ]

    seen: set = set()
    unique = []
    for c in selected:
        key = c.get("chunk_id") or id(c)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


async def answer_synthesizer_node(state: AgentState) -> dict:
    """
    LangGraph node — generates prose answer with citations.
    Populates: generated_answer, citations, numerical_claims, agent_trace.

    Model comes from the synthesis ladder: the best reasoning model with
    quota left, degrading through the ladder and finally to the local model.

    Args:
        state: Current AgentState with expanded_context and query.

    Returns:
        Partial state dict with answer and citations.
    """
    logger.info("Agent 7: Answer Synthesizer starting")

    # Check for forced refusal
    if state.get("force_refusal"):
        logger.info("Agent 7: Forced refusal — insufficient context")
        return {
            "generated_answer": (
                "I don't have sufficient information in the data room to answer "
                "this question accurately. The retrieved documents did not contain "
                "relevant content for your query. Please try rephrasing your "
                "question or check that the relevant documents have been uploaded."
            ),
            "citations": [],
            "numerical_claims": [],
            "confidence_score": 0.0,
            "agent_trace": [
                {"agent": "answer_synthesizer", "forced_refusal": True}
            ],
        }

    chunks = state.get("expanded_context", state.get("reranked_results", []))
    context = _format_context_for_synthesis(chunks)

    # Financial verification results
    financial_verification = ""
    if state.get("numerical_registry"):
        financial_verification = json.dumps(
            state["numerical_registry"], indent=2, default=str
        )

    inconsistencies = ""
    if state.get("inconsistencies"):
        inconsistencies = json.dumps(
            state["inconsistencies"], indent=2, default=str
        )

    user_prompt = ANSWER_SYNTHESIZER_USER_TEMPLATE.format(
        query=state["current_query"],
        query_type=state["query_type"],
        context=context,
        financial_verification=financial_verification or "N/A",
        inconsistencies=inconsistencies or "None found",
    )

    tracker = await BudgetTracker.get_instance()

    # Walk down the ladder on quota refusals.
    #
    # Selecting a model up front and failing the query when it 429s wastes the
    # whole point of having a ladder. Local counters drift from the provider's
    # (process restarts reset the in-memory fallback, other processes share the
    # key), so the tracker will sometimes hand back a rung the provider has
    # already closed. Measured: a golden-set run lost its first 19 answers this
    # way — one per remaining unit of a quota that was in fact already spent —
    # with context quality scores as high as 0.945.
    #
    # Each refusal marks that rung spent and re-selects, so the request lands on
    # the next model rather than failing.
    answer = None
    last_error: Exception | None = None
    model = ""

    for _ in range(MAX_LADDER_FALLBACKS):
        choice = await tracker.get_model_for_synthesis()
        model = choice.model
        try:
            candidate = await call_prose_agent(
                system_prompt=ANSWER_SYNTHESIZER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                model=model,
                temperature=0.1,
                max_tokens=3000,
                api_key=choice.api_key,
            )
            if not _is_usable_answer(candidate, chunks):
                # A response that violates the prompt's citation contract is a
                # failed generation, not an answer. Retrying on another rung is
                # the same treatment a transport error gets, and for the same
                # reason: the model did not do what was asked.
                logger.warning(
                    "Synthesis returned an uncited or truncated answer; retrying",
                    extra={
                        "model": model,
                        "chars": len(candidate or ""),
                        "context_chunks": len(chunks),
                    },
                )
                last_error = RuntimeError("synthesis produced an uncited answer")
                tracker.skip_model_for_request(model)
                continue
            answer = candidate
            break
        except Exception as e:
            last_error = e
            if choice.key_index >= 0 and is_auth_error(e):
                # Bad credential — retire the key entirely and try another.
                tracker.mark_key_unusable(choice.key_index)
                continue
            if is_quota_error(e) and choice.key_index >= 0:
                await tracker.mark_slot_exhausted(choice.key_index, model)
                logger.warning(
                    "Synthesis rung refused by provider, descending the ladder",
                    extra={"model": model, "key_index": choice.key_index},
                )
                continue
            if is_model_unavailable_for_key(e) and choice.key_index >= 0:
                # This key may never use this model — retire the pair only.
                tracker.mark_slot_unavailable(choice.key_index, model)
                logger.warning(
                    "Model not available for this credential; retiring the slot",
                    extra={"model": model, "key_index": choice.key_index},
                )
                continue
            if is_service_unavailable(e) and choice.key_index >= 0:
                # 503 means the model is down for every key, so rotating keys
                # just repeats the failure. Skip the model briefly instead, and
                # do not debit quota — this clears in minutes and should not
                # cost the day's capacity on the best synthesis model.
                tracker.skip_model_for_request(model)
                logger.warning(
                    "Synthesis rung unavailable provider-side, trying another model",
                    extra={"model": model},
                )
                continue
            break

    if answer is None:
        e = last_error or RuntimeError("synthesis produced no answer")
        # Synthesis is the one place where an upstream failure would otherwise
        # take down the whole request. Degrade to an explicit refusal instead:
        # a reviewer who is told the engine could not answer is strictly better
        # off than one who gets a 500 and no explanation.
        logger.error(
            "Agent 7: synthesis failed, degrading to refusal",
            extra={"model": model, "error": str(e)},
        )
        return {
            "generated_answer": (
                "I could not generate an answer for this question because the "
                "language model did not return a usable response. The retrieved "
                "context may still be relevant — please retry the query."
            ),
            "citations": [],
            "numerical_claims": [],
            "confidence_score": 0.0,
            "agent_trace": [
                {
                    "agent": "answer_synthesizer",
                    "model": model,
                    "synthesis_failed": True,
                    "error": str(e),
                }
            ],
        }

    # Citations are resolved from the inline markers the model emits, not from
    # whether a filename appears anywhere in the answer. See _select_cited_chunks.
    #
    # The full chunk payload is carried through — not just chunk_id/source_file —
    # so the API can surface page, section, version and computed-metric provenance
    # without a second round-trip to Qdrant. Anything the ingestion pipeline does
    # not yet write (is_redline, superseded_by) simply falls back to its default.
    citations = [
        {
            "chunk_id": c.get("chunk_id", ""),
            "source_file": c.get("source_file", ""),
            "page_number": c.get("page_number"),
            "section_heading": c.get("section_heading", "") or "",
            "is_current_version": c.get("is_current_version", 1),
            "content_type": c.get("content_type", "text") or "text",
            "is_redline": bool(c.get("is_redline", 0)),
            "superseded_by": c.get("superseded_by", "") or "",
        }
        for c in _select_cited_chunks(answer, chunks)
    ]

    logger.info(
        "Agent 7: Answer Synthesizer complete",
        extra={
            "model": model,
            "answer_length": len(answer),
            "citations_count": len(citations),
        },
    )

    return {
        "generated_answer": answer,
        "citations": citations,
        "numerical_claims": state.get("inconsistencies", []),
        "agent_trace": [
            {
                "agent": "answer_synthesizer",
                "model": model,
                "answer_length": len(answer),
            }
        ],
    }
