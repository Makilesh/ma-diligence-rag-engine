"""
Deal management routes.
"""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from api.models.request_models import DealCreateRequest
from api.models.response_models import DealResponse, DocumentRecord, RiskSignal
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
async def create_deal(request: DealCreateRequest):
    """Creates a new deal."""
    deal_id = str(uuid.uuid4())
    _deals[deal_id] = {
        "deal_id": deal_id,
        "deal_name": request.deal_name,
        "description": request.description,
        "document_count": 0,
        "status": "active",
    }

    logger.info(
        "Deal created",
        extra={"deal_id": deal_id, "deal_name": request.deal_name},
    )

    return DealResponse(**_deals[deal_id])


async def _discover_indexed_deals() -> dict[str, int]:
    """
    Finds deals that exist in the vector store, with their document counts.

    `_deals` only knows about deals created through `POST /deals` in *this*
    process. That leaves two ways for real, queryable data to be invisible in
    the UI: anything ingested directly against a deal_id (which is how the
    evaluation harness loads the corpus), and everything at all after a restart,
    since the registry is in-memory while Qdrant is on disk. Both produced the
    same dead end — 131 indexed chunks, and a sidebar reading "No deals found."

    Qdrant is the source of truth for what is actually searchable, so ask it.

    Returns:
        Mapping of deal_id to distinct document count. Empty on any failure —
        the endpoint still returns the in-memory deals.
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
        return {}

    discovered: dict[str, int] = {}
    for hit in deal_facet.hits:
        deal_id = str(hit.value)
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            file_facet = await client.facet(
                collection_name=COLLECTION_NAME,
                key="source_file",
                facet_filter=Filter(
                    must=[FieldCondition(key="deal_id", match=MatchValue(value=deal_id))]
                ),
                limit=1000,
            )
            discovered[deal_id] = len(file_facet.hits)
        except Exception:
            # Chunk count is a poor stand-in for document count, but a deal that
            # is listed with the wrong count is far better than one that is
            # missing entirely.
            discovered[deal_id] = 0
    return discovered


@router.get("/deals", response_model=list[DealResponse])
async def list_deals():
    """
    Lists every deal that is queryable — registered in this process or indexed.

    Deals created via `POST /deals` keep their name and description; deals found
    only in the vector store are listed under their deal_id so they can still be
    selected.
    """
    deals = {d["deal_id"]: dict(d) for d in _deals.values()}

    for deal_id, doc_count in (await _discover_indexed_deals()).items():
        if deal_id in deals:
            # Prefer the live index count over the registry's, which drifts on
            # restart while the vector store does not.
            if doc_count:
                deals[deal_id]["document_count"] = doc_count
            continue
        deals[deal_id] = {
            "deal_id": deal_id,
            "deal_name": deal_id,
            "description": "Discovered in the vector store",
            "document_count": doc_count,
            "status": "active",
        }

    return [DealResponse(**d) for d in deals.values()]


@router.get("/deals/{deal_id}", response_model=DealResponse)
async def get_deal(deal_id: str):
    """Gets a specific deal by ID."""
    if deal_id not in _deals:
        raise HTTPException(status_code=404, detail=f"Deal not found: {deal_id}")
    return DealResponse(**_deals[deal_id])


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
async def list_deal_documents(deal_id: str):
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
async def list_deal_risk_signals(deal_id: str):
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
