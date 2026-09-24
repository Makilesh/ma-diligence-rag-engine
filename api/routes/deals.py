"""
Deal management routes.
"""

import asyncio
import os
import re
import secrets
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Request

from api.models.request_models import DealCreateRequest, DealIdPath
from api.models.response_models import DealResponse, DocumentRecord, RiskSignal
from api.security import get_request_id, is_admin, public_error_detail
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter()

# In-memory deal store (in production, use Postgres)
_deals: dict[str, dict] = {}

# In-memory document registry, keyed by deal_id -> list of document records.
# Mirrors the _deals pattern: it is process-local and lost on restart, which is
# the same single-worker constraint the rest of the API already carries. Qdrant
# remains the source of truth for chunk content; this only tracks document-level
# provenance (version chain, risk signals) that has no home in the vector store.
_documents: dict[str, list[dict]] = {}

# Severity ranking for the risk signal types emitted by RiskSignalExtractor.
# Defined here rather than in the extractor because severity is a presentation
# concern — the extractor reports what it matched, the dashboard ranks it.
_RISK_SEVERITY: dict[str, str] = {
    "change_of_control": "high",
    "material_adverse_change": "high",
    "financial_distress": "high",
    "litigation": "medium",
    "regulatory_risk": "medium",
    "environmental_liability": "medium",
    "ip_risk": "medium",
    "customer_concentration": "low",
    "key_person": "low",
    "indemnification": "low",
}


@router.post("/deals", response_model=DealResponse)
async def create_deal(request: DealCreateRequest, http_request: Request):
    """
    Creates a new deal.

    Anyone may create a sandbox deal — that is how a visitor uploads. A
    permanent deal is an owner operation and needs the admin key.

    Args:
        request: Deal name, description and sandbox flag.
        http_request: Raw request, for the admin check.

    Returns:
        The created deal.
    """
    if not request.is_sandbox and not is_admin(http_request):
        raise HTTPException(
            status_code=403, detail="Only sandbox deals can be created without the admin key."
        )

    now = datetime.now(timezone.utc)
    deal_id = new_sandbox_id(now) if request.is_sandbox else str(uuid.uuid4())
    # Derived from the id rather than `now` so the advertised deadline is exactly
    # the one the sweeper will apply, including after a restart.
    sandbox_deadline = sandbox_expires_at(deal_id)
    expires_at = sandbox_deadline.isoformat() if sandbox_deadline else ""
    _deals[deal_id] = {
        "deal_id": deal_id,
        "deal_name": request.deal_name,
        "description": request.description,
        "document_count": 0,
        "status": "active",
        "is_sandbox": request.is_sandbox,
        "created_at": now.isoformat(),
        "expires_at": expires_at,
    }

    logger.info(
        "Deal created",
        extra={"deal_id": deal_id, "deal_name": request.deal_name},
    )

    return DealResponse(**_deals[deal_id])


async def _facet_deal_ids() -> list[str]:
    """
    Lists every deal_id present in the vector store.

    Returns:
        Deal ids. Empty on any failure.
    """
    from src.vector_db.qdrant_client import get_qdrant_client
    from src.vector_db.constants import COLLECTION_NAME

    client = get_qdrant_client()
    try:
        deal_facet = await client.facet(
            collection_name=COLLECTION_NAME, key="deal_id", limit=1000
        )
    except Exception as e:
        logger.warning(f"Could not enumerate deals from the vector store: {e}")
        return []
    return [str(hit.value) for hit in deal_facet.hits]


async def _indexed_document_count(deal_id: str) -> int:
    """
    Counts distinct source files indexed under one deal.

    Args:
        deal_id: Deal to count.

    Returns:
        Number of distinct documents; 0 when the deal has nothing indexed or
        the store cannot be read.
    """
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    from src.vector_db.qdrant_client import get_qdrant_client
    from src.vector_db.constants import COLLECTION_NAME

    try:
        file_facet = await get_qdrant_client().facet(
            collection_name=COLLECTION_NAME,
            key="source_file",
            facet_filter=Filter(
                must=[FieldCondition(key="deal_id", match=MatchValue(value=deal_id))]
            ),
            limit=1000,
        )
    except Exception as e:
        logger.warning(f"Could not count documents for {deal_id}: {e}")
        return 0
    return len(file_facet.hits)


async def _discover_indexed_deals(include_sandboxes: bool = True) -> dict[str, int]:
    """
    Finds deals that exist in the vector store, with their document counts.

    `_deals` only knows about deals created through `POST /deals` in *this*
    process. That leaves two ways for real, queryable data to be invisible in
    the UI: anything ingested directly against a deal_id (which is how the
    evaluation harness loads the corpus), and everything at all after a restart,
    since the registry is in-memory while Qdrant is on disk. Both produced the
    same dead end — 131 indexed chunks, and a sidebar reading "No deals found."

    Qdrant is the source of truth for what is actually searchable, so ask it.

    Args:
        include_sandboxes: False skips sandbox ids before the per-deal count,
            which is one Qdrant round-trip each.

    Returns:
        Mapping of deal_id to distinct document count. Empty on any failure —
        the endpoint still returns the in-memory deals.
    """
    discovered: dict[str, int] = {}
    for deal_id in await _facet_deal_ids():
        if not include_sandboxes and is_sandbox_id(deal_id):
            continue
        # A deal listed with a zero count is far better than one that is
        # missing entirely, so a failed count does not drop the deal.
        discovered[deal_id] = await _indexed_document_count(deal_id)
    return discovered


def _discovered_deal_record(deal_id: str, doc_count: int) -> dict:
    """Builds the listing record for a deal known only from the vector store."""
    expires_at = sandbox_expires_at(deal_id)
    return {
        "deal_id": deal_id,
        "deal_name": deal_id,
        "description": "Discovered in the vector store",
        "document_count": doc_count,
        "status": "active",
        "is_sandbox": expires_at is not None,
        "expires_at": expires_at.isoformat() if expires_at else "",
    }


@router.get("/deals", response_model=list[DealResponse])
async def list_deals(http_request: Request):
    """
    Lists every deal that is queryable — registered in this process or indexed.

    Deals created via `POST /deals` keep their name and description; deals found
    only in the vector store are listed under their deal_id so they can still be
    selected.

    Sandbox deals are omitted for public callers. Listing them let any visitor
    open, and query, what another visitor had uploaded; a visitor's own sandbox
    id is already known to its browser, which is the only place it is needed.
    Admin callers see everything, which is how orphaned sandboxes are found.
    """
    include_sandboxes = is_admin(http_request)
    deals = {
        d["deal_id"]: dict(d)
        for d in _deals.values()
        if include_sandboxes or not (d.get("is_sandbox") or is_sandbox_id(d["deal_id"]))
    }

    for deal_id, doc_count in (
        await _discover_indexed_deals(include_sandboxes=include_sandboxes)
    ).items():
        if deal_id in deals:
            # Prefer the live index count over the registry's, which drifts on
            # restart while the vector store does not.
            if doc_count:
                deals[deal_id]["document_count"] = doc_count
            continue
        deals[deal_id] = _discovered_deal_record(deal_id, doc_count)

    return [DealResponse(**d) for d in deals.values()]


@router.get("/deals/{deal_id}", response_model=DealResponse)
async def get_deal(deal_id: DealIdPath):
    """
    Gets a specific deal by ID.

    Resolves against the same two sources `GET /deals` merges — the in-memory
    registry, then the vector store — so a deal the listing shows is never a
    404 here, which it used to be after every restart.

    Args:
        deal_id: Deal to fetch.

    Returns:
        The deal.
    """
    if deal_id in _deals:
        return DealResponse(**_deals[deal_id])

    doc_count = await _indexed_document_count(deal_id)
    if not doc_count:
        raise HTTPException(status_code=404, detail="Deal not found.")
    return DealResponse(**_discovered_deal_record(deal_id, doc_count))


# ==============================================================================
# Document registry — populated by the ingestion route
# ==============================================================================


def register_document(
    deal_id: str,
    doc_id: str,
    filename: str,
    document_category: str,
    chunks_created: int,
    is_current_version: bool,
    supersedes_doc_id: str | None,
    risk_signals: list[dict] | None = None,
) -> None:
    """
    Records an ingested document against its deal.

    Also maintains the version chain: when this document supersedes another,
    the superseded record is flipped to is_current_version=False and stamped
    with superseded_by, so the version browser and citation version warnings
    have a consistent view without re-reading Qdrant.

    Args:
        deal_id: Owning deal.
        doc_id: Newly assigned document ID.
        filename: Original uploaded filename.
        document_category: Detected or overridden category.
        chunks_created: Number of chunks indexed for this document.
        is_current_version: Whether this upload is the current version.
        supersedes_doc_id: Doc ID this version replaces, if any.
        risk_signals: Risk signal dicts detected during ingestion.
    """
    records = _documents.setdefault(deal_id, [])

    # doc_id is derived from the file's content, so re-uploading identical bytes
    # yields the same id and ingestion replaces the points in place. The registry
    # must do the same, or the deal lists the document twice.
    records[:] = [r for r in records if r["doc_id"] != doc_id]

    records.append(
        {
            "doc_id": doc_id,
            "deal_id": deal_id,
            "filename": filename,
            "document_category": document_category,
            "chunks_created": chunks_created,
            "is_current_version": is_current_version,
            "supersedes_doc_id": supersedes_doc_id or "",
            "superseded_by": "",
            "upload_date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "risk_signals": risk_signals or [],
            "has_redline": False,  # set once redline indexing is wired in ingest
        }
    )

    if supersedes_doc_id:
        for prior in records:
            if prior["doc_id"] == supersedes_doc_id:
                prior["is_current_version"] = False
                prior["superseded_by"] = doc_id
                break

    # document_count previously never moved off its initial 0
    if deal_id in _deals:
        _deals[deal_id]["document_count"] = len(records)


async def _scroll_deal_payloads(deal_id: str, fields: list[str]) -> list[dict]:
    """
    Reads selected payload fields for every chunk in a deal.

    Only the named fields are fetched — pulling `text` back for a whole deal
    would move megabytes to count documents.

    Args:
        deal_id: Deal to scan.
        fields: Payload keys to return.

    Returns:
        List of payload dicts. Empty on any failure.
    """
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    from src.vector_db.qdrant_client import get_qdrant_client
    from src.vector_db.constants import COLLECTION_NAME

    client = get_qdrant_client()
    payloads: list[dict] = []
    offset = None
    try:
        while True:
            points, offset = await client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=Filter(
                    must=[FieldCondition(key="deal_id", match=MatchValue(value=deal_id))]
                ),
                with_payload=fields,
                with_vectors=False,
                limit=512,
                offset=offset,
            )
            payloads.extend(p.payload or {} for p in points)
            if offset is None:
                break
    except Exception as e:
        logger.warning(f"Could not scroll deal payloads for {deal_id}: {e}")
        return []
    return payloads


async def _reconstruct_documents(deal_id: str) -> list[dict]:
    """
    Rebuilds document records for a deal from the vector store.

    `_documents` is process-local, so after any restart the version browser and
    the risk dashboard went blank while the deal was still fully queryable — the
    UI reported "no documents" for a deal that answered questions correctly.
    Qdrant holds one payload per chunk carrying the document-level fields, so
    the records can be reconstructed rather than lost.

    Upload dates are not recoverable this way; they are left empty rather than
    invented, and the version label falls back to chunk ordering.

    Args:
        deal_id: Deal to rebuild.

    Returns:
        Document records in the same shape `_documents` holds.
    """
    payloads = await _scroll_deal_payloads(
        deal_id,
        [
            "doc_id",
            "source_file",
            "document_category",
            "is_current_version",
            "supersedes_doc_id",
            "superseded_by",
            "risk_signals",
        ],
    )

    by_doc: dict[str, dict] = {}
    for p in payloads:
        doc_id = p.get("doc_id") or p.get("source_file", "unknown")
        record = by_doc.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "deal_id": deal_id,
                "filename": p.get("source_file", "unknown"),
                "document_category": p.get("document_category", "other"),
                "chunks_created": 0,
                "is_current_version": bool(p.get("is_current_version", 1)),
                "supersedes_doc_id": p.get("supersedes_doc_id") or "",
                "superseded_by": p.get("superseded_by") or "",
                "upload_date": "",
                "risk_signals": [],
                "has_redline": False,
            },
        )
        record["chunks_created"] += 1

        # Risk signals are stored per chunk; collapse them to one entry per
        # (document, signal type) so the dashboard counts documents at risk
        # rather than chunks mentioning risk.
        #
        # Two shapes exist. The chunk payload holds bare type strings
        # (["change_of_control"]) because that is all the retrieval filter needs,
        # while the in-memory registry holds the extractor's full dicts with
        # match counts and samples. Both have to normalise to the same record or
        # the dashboard silently shows nothing — which is precisely what happened
        # when only the dict shape was handled.
        for signal in p.get("risk_signals") or []:
            if isinstance(signal, str):
                signal = {"signal_type": signal, "match_count": 1, "sample_matches": []}
            elif not isinstance(signal, dict):
                continue

            sig_type = signal.get("signal_type", "other")
            existing = next(
                (s for s in record["risk_signals"] if s.get("signal_type") == sig_type),
                None,
            )
            if existing is None:
                record["risk_signals"].append(dict(signal))
            else:
                existing["match_count"] = existing.get("match_count", 0) + signal.get(
                    "match_count", 0
                )

    return list(by_doc.values())


async def _deal_documents(deal_id: str) -> list[dict]:
    """Returns in-memory records when present, else rebuilds them from Qdrant."""
    records = _documents.get(deal_id)
    if records:
        return records
    return await _reconstruct_documents(deal_id)


@router.get("/deals/{deal_id}/documents", response_model=list[DocumentRecord])
async def list_deal_documents(deal_id: DealIdPath):
    """
    Lists ingested documents for a deal, newest first.

    Feeds the version browser: each record carries its position in the version
    chain (supersedes_doc_id / superseded_by / is_current_version).
    """
    records = await _deal_documents(deal_id)
    # Reconstructed records carry no upload_date, so fall back to filename for a
    # stable order rather than letting an empty string shuffle the list.
    ordered = sorted(
        records, key=lambda r: (r.get("upload_date") or "", r["filename"]), reverse=True
    )
    return [
        DocumentRecord(
            doc_id=r["doc_id"],
            filename=r["filename"],
            document_category=r["document_category"],
            chunks_created=r["chunks_created"],
            version_label=f"v{len(records) - i}",
            upload_date=r["upload_date"],
            is_current_version=r["is_current_version"],
            supersedes_doc_id=r["supersedes_doc_id"],
            superseded_by=r["superseded_by"],
            has_redline=r["has_redline"],
        )
        for i, r in enumerate(ordered)
    ]


@router.get("/deals/{deal_id}/risk-signals", response_model=list[RiskSignal])
async def list_deal_risk_signals(deal_id: DealIdPath):
    """
    Returns risk signals detected across all documents in a deal.

    Signals are produced by RiskSignalExtractor at ingestion time; severity is
    assigned here from _RISK_SEVERITY. Sorted high → low so the dashboard's
    most important categories expand first.
    """
    signals: list[RiskSignal] = []

    for record in await _deal_documents(deal_id):
        for signal in record.get("risk_signals", []):
            signal_type = signal.get("signal_type", "other")
            match_count = signal.get("match_count", 0)
            samples = signal.get("sample_matches", [])
            sample_str = ", ".join(str(s) for s in samples if s)

            # Wording differs by source, deliberately. Records held in memory
            # carry the extractor's own match count and sample text; records
            # rebuilt from chunk payloads only know how many chunks carried the
            # signal, so they must not claim to be counting regex matches.
            if sample_str:
                description = f"{match_count} match(es) — e.g. \"{sample_str[:120]}\""
            else:
                description = f"{match_count} chunk(s) in this document matched"

            signals.append(
                RiskSignal(
                    signal_type=signal_type,
                    severity=_RISK_SEVERITY.get(signal_type, "low"),
                    source_file=record["filename"],
                    description=description,
                    page_number=signal.get("page_number"),
                )
            )

    severity_rank = {"high": 0, "medium": 1, "low": 2}
    signals.sort(key=lambda s: severity_rank.get(s.severity, 3))
    return signals


# ==============================================================================
# Ephemeral sandbox deals
# ==============================================================================
# A public demo has to let visitors upload their own documents without becoming
# a permanent store of other people's files. Sandbox deals are deleted on two
# independent triggers, because neither is sufficient alone:
#
#   1. The tab closes. The browser fires `navigator.sendBeacon` at the purge
#      endpoint below, which deletes immediately. This covers the ordinary case
#      and is the only one fast enough to feel like "closed it, it's gone".
#
#   2. The TTL sweeper. A beacon is best-effort by specification — a crashed
#      tab, a backgrounded mobile browser killed by the OS, or a dropped network
#      all lose it silently. The sweeper is what makes deletion a guarantee
#      rather than a hope, so it is the load-bearing one.
#
# Deletion is by `deal_id` payload filter, which is an indexed KEYWORD field, so
# it is a single cheap operation rather than a scroll-and-delete.
#
# The sweeper must survive a restart, and `_deals` does not: a Space that sleeps
# or redeploys forgets every sandbox it created, and their vectors would then sit
# in Qdrant Cloud forever. So the sandbox marker and its creation time live in
# the deal_id itself — `sbx-<created unix time, hex>-<128 random bits>` — and the
# sweeper also scans the ids Qdrant holds. Nothing extra has to be written to
# the chunk payloads, the ingest path needs no change to stay sweepable, and the
# random part keeps the id unguessable, which is what scopes one visitor's
# uploads away from another's.

# Sandbox lifetime. Long enough that a visitor reading a long answer does not
# have their upload swept mid-session, short enough that an abandoned upload does
# not sit in a free-tier cluster overnight.
SANDBOX_TTL_SECONDS: int = int(os.getenv("SANDBOX_TTL_SECONDS", str(2 * 60 * 60)))

# How often the sweeper wakes. Independent of the TTL: checking every few minutes
# bounds how long an expired deal outlives its deadline without making the sweep
# itself a load source.
SANDBOX_SWEEP_INTERVAL_SECONDS: int = int(
    os.getenv("SANDBOX_SWEEP_INTERVAL_SECONDS", "300")
)

_sweeper_task: asyncio.Task | None = None

SANDBOX_ID_PREFIX = "sbx-"
_SANDBOX_ID_RE = re.compile(r"^sbx-([0-9a-f]{8})-[0-9a-f]{32}$")

# A creation time further in the future than this is treated as already
# expired. The id is client-visible and anyone can mint one in the right shape,
# so a forged far-future timestamp must not buy an upload a longer life.
_SANDBOX_CLOCK_SKEW = timedelta(minutes=5)


def new_sandbox_id(now: datetime) -> str:
    """
    Mints a sandbox deal id carrying its own creation time.

    Args:
        now: Creation time (UTC).

    Returns:
        An id of the form `sbx-<8 hex>-<32 hex>`.
    """
    return f"{SANDBOX_ID_PREFIX}{int(now.timestamp()):08x}-{secrets.token_hex(16)}"


def is_sandbox_id(deal_id: str) -> bool:
    """True if `deal_id` has the sandbox shape."""
    return bool(_SANDBOX_ID_RE.fullmatch(deal_id or ""))


def sandbox_expires_at(deal_id: str) -> datetime | None:
    """
    Derives a sandbox's TTL deadline from its id.

    Args:
        deal_id: Deal id.

    Returns:
        The expiry time, or None if `deal_id` is not a sandbox id.
    """
    match = _SANDBOX_ID_RE.fullmatch(deal_id or "")
    if not match:
        return None
    created = datetime.fromtimestamp(int(match.group(1), 16), tz=timezone.utc)
    return created + timedelta(seconds=SANDBOX_TTL_SECONDS)


def sandbox_is_expired(deal_id: str, now: datetime | None = None) -> bool:
    """
    Whether a sandbox deal is past its TTL (or claims an impossible future birth).

    Args:
        deal_id: Deal id. Non-sandbox ids never expire.
        now: Reference time; defaults to the current UTC time.

    Returns:
        True if the deal should be purged.
    """
    expires_at = sandbox_expires_at(deal_id)
    if expires_at is None:
        return False
    now = now or datetime.now(timezone.utc)
    created = expires_at - timedelta(seconds=SANDBOX_TTL_SECONDS)
    return expires_at <= now or created > now + _SANDBOX_CLOCK_SKEW


async def _delete_deal_vectors(deal_id: str) -> None:
    """
    Removes every point belonging to a deal from both Qdrant collections.

    Args:
        deal_id: Deal whose vectors should be deleted.

    Raises:
        Exception: Propagated from the Qdrant client so callers can decide
            whether a failed purge is fatal (explicit delete) or worth retrying
            on the next pass (sweeper).
    """
    from qdrant_client.models import FilterSelector, FieldCondition, Filter, MatchValue

    from src.vector_db.qdrant_client import get_qdrant_client
    from src.vector_db.constants import COLLECTION_NAME, PARENT_COLLECTION_NAME

    client = get_qdrant_client()
    selector = FilterSelector(
        filter=Filter(
            must=[FieldCondition(key="deal_id", match=MatchValue(value=deal_id))]
        )
    )

    for collection in (COLLECTION_NAME, PARENT_COLLECTION_NAME):
        try:
            await client.delete(collection_name=collection, points_selector=selector)
        except Exception as e:
            # The parent collection is not written by the current ingest path, so
            # it is frequently absent. Missing it must not abort the delete of
            # the main collection, which is the one that actually holds content.
            if collection == PARENT_COLLECTION_NAME:
                logger.debug(
                    "Parent collection purge skipped",
                    extra={"deal_id": deal_id, "error": str(e)},
                )
                continue
            raise


async def purge_deal(deal_id: str) -> dict:
    """
    Deletes a deal's vectors and forgets its in-memory records.

    Args:
        deal_id: Deal to purge.

    Returns:
        Summary dict with the deal_id and how many document records were dropped.
    """
    await _delete_deal_vectors(deal_id)

    documents_dropped = len(_documents.pop(deal_id, []))
    _deals.pop(deal_id, None)

    logger.info(
        "Deal purged",
        extra={"deal_id": deal_id, "documents_dropped": documents_dropped},
    )
    return {"deal_id": deal_id, "documents_dropped": documents_dropped, "status": "purged"}


def _authorize_purge(deal_id: str, http_request: Request) -> None:
    """
    Lets anyone purge a sandbox deal, and only admin purge anything else.

    Sandbox purges must stay open: the tab-close beacon cannot carry a header,
    and the visitor who owns the sandbox has no credentials anyway. Knowing a
    sandbox id is the capability — it is 128 random bits, never listed publicly.
    Everything else, the demo corpus above all, needs the admin key; before this
    check a single anonymous DELETE could wipe the demo data room.

    Args:
        deal_id: Deal to purge.
        http_request: Raw request, for the admin check.

    Raises:
        HTTPException: 403 for a non-sandbox deal without the admin key.
    """
    if is_sandbox_id(deal_id) or is_admin(http_request):
        return
    logger.warning("Rejected non-admin purge of a non-sandbox deal", extra={"deal_id": deal_id})
    raise HTTPException(
        status_code=403, detail="Only sandbox deals can be deleted without the admin key."
    )


@router.delete("/deals/{deal_id}")
async def delete_deal(deal_id: DealIdPath, http_request: Request):
    """
    Deletes a deal and everything indexed under it.

    Args:
        deal_id: Deal to delete.
        http_request: Raw request, for the admin check.

    Returns:
        Purge summary.
    """
    _authorize_purge(deal_id, http_request)
    try:
        return await purge_deal(deal_id)
    except Exception as e:
        request_id = get_request_id(http_request)
        logger.error(
            "Deal purge failed",
            extra={"deal_id": deal_id, "error": str(e), "request_id": request_id},
        )
        raise HTTPException(status_code=500, detail=public_error_detail("Purge", request_id))


@router.post("/deals/{deal_id}/purge")
async def purge_deal_endpoint(deal_id: DealIdPath, http_request: Request):
    """
    POST-shaped alias of DELETE, for `navigator.sendBeacon`.

    This exists purely because the Beacon API can only issue POST — and a beacon
    is the only request that reliably survives the tab that fired it. Routing the
    unload path through `fetch(..., {method: "DELETE"})` instead loses the
    request whenever the browser tears the page down first, which is most of the
    time and is exactly the case the endpoint is for.

    A permitted purge always reports success: the caller is a beacon whose
    response nothing will ever read, and a failure here is recovered by the TTL
    sweeper anyway. An unpermitted one is still a 403.

    Args:
        deal_id: Deal to purge.
        http_request: Raw request, for the admin check.

    Returns:
        Purge summary, or a status of "deferred" if the delete failed.
    """
    _authorize_purge(deal_id, http_request)
    try:
        return await purge_deal(deal_id)
    except Exception as e:
        logger.warning(
            "Beacon purge failed — deferring to the TTL sweeper",
            extra={"deal_id": deal_id, "error": str(e)},
        )
        return {"deal_id": deal_id, "status": "deferred"}


async def _sweep_expired_sandboxes() -> int:
    """
    Purges every sandbox deal past its TTL.

    Candidates come from both the in-memory registry and the vector store, so
    sandboxes created before a restart — which the registry no longer knows
    about — are still reclaimed.

    Returns:
        Number of deals purged on this pass.
    """
    now = datetime.now(timezone.utc)

    # Materialise the candidate list before awaiting anything: `purge_deal`
    # mutates `_deals`, and mutating a dict while iterating it raises.
    candidates = {deal_id for deal_id, deal in list(_deals.items()) if deal.get("is_sandbox")}
    candidates.update(d for d in await _facet_deal_ids() if is_sandbox_id(d))
    expired = sorted(d for d in candidates if sandbox_is_expired(d, now))

    purged = 0
    for deal_id in expired:
        try:
            await purge_deal(deal_id)
            purged += 1
        except Exception as e:
            # Leave the record in place so the next pass retries it.
            logger.warning(
                "Sandbox sweep failed for deal",
                extra={"deal_id": deal_id, "error": str(e)},
            )

    if purged:
        logger.info("Sandbox sweep complete", extra={"deals_purged": purged})
    return purged


async def _sweeper_loop() -> None:
    """
    Runs `_sweep_expired_sandboxes` forever, on the configured interval.

    Sweeps once immediately: a Space that slept through a sandbox's TTL should
    reclaim it on wake, not one interval later.
    """
    while True:
        try:
            await _sweep_expired_sandboxes()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # A sweep that raises must not kill the loop — that would silently
            # disable the guarantee the sweeper exists to provide.
            logger.error("Sandbox sweeper iteration failed", extra={"error": str(e)})
        # Outside the try, so a sweep that fails every time still waits between
        # attempts instead of spinning.
        await asyncio.sleep(SANDBOX_SWEEP_INTERVAL_SECONDS)


async def start_sandbox_sweeper() -> None:
    """Starts the background sweeper. Idempotent."""
    global _sweeper_task
    if _sweeper_task is not None and not _sweeper_task.done():
        return
    _sweeper_task = asyncio.create_task(_sweeper_loop())
    logger.info(
        "Sandbox sweeper started",
        extra={
            "ttl_seconds": SANDBOX_TTL_SECONDS,
            "interval_seconds": SANDBOX_SWEEP_INTERVAL_SECONDS,
        },
    )


async def stop_sandbox_sweeper() -> None:
    """Cancels the background sweeper and waits for it to unwind."""
    global _sweeper_task
    if _sweeper_task is None:
        return
    _sweeper_task.cancel()
    try:
        await _sweeper_task
    except asyncio.CancelledError:
        pass
    _sweeper_task = None
    logger.info("Sandbox sweeper stopped")
