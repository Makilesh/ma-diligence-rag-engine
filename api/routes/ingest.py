"""
Document ingestion routes.

The route owns the HTTP concerns — upload size cap, filename handling, error
mapping. Everything from parsing to upsert lives in
src/data_processing/ingest_pipeline.py so the evaluation harness indexes
through exactly the same code.
"""

import os
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from api.models.response_models import IngestResponse
from api.routes.deals import register_document
from src.data_processing.ingest_pipeline import (
    SUPPORTED_EXTENSIONS,
    IngestionError,
    index_document,
    mark_superseded,
    sanitize_filename,
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter()

_READ_CHUNK_BYTES = 1024 * 1024


def _max_upload_bytes() -> int:
    """Upload size cap in bytes (env MAX_UPLOAD_MB, default 25)."""
    return int(os.getenv("MAX_UPLOAD_MB", "25")) * 1024 * 1024


async def _save_upload(file: UploadFile, dest_path: str, max_bytes: int) -> int:
    """
    Streams an upload to disk, enforcing the size cap while reading.

    Reading in fixed-size chunks keeps memory flat regardless of what the
    client sends, and the cap is checked before each write rather than after
    the whole body is buffered.

    Args:
        file: The multipart upload.
        dest_path: Where to write it.
        max_bytes: Maximum accepted size.

    Returns:
        Number of bytes written.

    Raises:
        HTTPException: 413 when over the cap, 400 when empty.
    """
    too_large = HTTPException(
        status_code=413,
        detail=f"File exceeds the {max_bytes // (1024 * 1024)} MB upload limit",
    )
    if file.size is not None and file.size > max_bytes:
        raise too_large

    written = 0
    with open(dest_path, "wb") as out:
        while True:
            block = await file.read(_READ_CHUNK_BYTES)
            if not block:
                break
            written += len(block)
            if written > max_bytes:
                raise too_large
            out.write(block)

    if written == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    return written


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    deal_id: str = Form(...),
    document_category: str | None = Form(None),
    is_current_version: bool = Form(True),
    supersedes_doc_id: str | None = Form(None),
):
    """
    Ingests a document into the RAG pipeline.

    Pipeline:
    1. Validate filename and file type
    2. Stream to a temp file under a generated name (size-capped)
    3. index_document(): classify → process → chunk → embed → upsert
       (idempotent: identical bytes in the same deal keep the same doc_id)
    4. Retire the superseded document, if any
    5. Return ingestion summary

    Args:
        file: Uploaded file.
        deal_id: Deal identifier for data isolation.
        document_category: Override category (auto-detected if not provided).
        is_current_version: Whether this is the current version.
        supersedes_doc_id: Doc ID this version supersedes.

    Returns:
        IngestResponse with doc_id and chunk count.
    """
    try:
        filename = sanitize_filename(file.filename)
    except IngestionError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))

    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: {extension or '(none)'}. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            ),
        )

    logger.info(
        "Document ingestion started",
        extra={"deal_id": deal_id, "file_name": filename, "extension": extension},
    )

    # The client's filename never becomes part of a filesystem path; it is kept
    # only as metadata. The generated name preserves the extension, which is
    # all the processors need.
    temp_dir = tempfile.mkdtemp(prefix="manda_ingest_")
    temp_path = os.path.join(temp_dir, f"upload{extension}")

    try:
        await _save_upload(file, temp_path, _max_upload_bytes())
        result = await index_document(
            temp_path,
            filename,
            deal_id,
            document_category,
            is_current_version=is_current_version,
            supersedes_doc_id=supersedes_doc_id,
        )
    except HTTPException:
        raise
    except IngestionError as e:
        logger.warning(
            "Document rejected",
            extra={"deal_id": deal_id, "file_name": filename, "reason": str(e)},
        )
        raise HTTPException(status_code=e.status_code, detail=str(e))
    except Exception as e:
        # Internal detail (paths, Qdrant errors) stays in the log.
        logger.error(
            "Document ingestion failed",
            extra={"deal_id": deal_id, "file_name": filename, "error": str(e)},
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Ingestion failed due to an internal error",
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    doc_id = result["doc_id"]

    # Retire the superseded document's chunks so retrieval's is_current_version=1
    # filter stops returning them, and any citation that does surface them
    # carries the pointer forward.
    if supersedes_doc_id:
        await mark_superseded(
            deal_id=deal_id,
            superseded_doc_id=supersedes_doc_id,
            superseded_by=doc_id,
        )

    register_document(
        deal_id=deal_id,
        doc_id=doc_id,
        filename=filename,
        document_category=result["document_category"],
        chunks_created=result["chunks_created"],
        is_current_version=is_current_version,
        supersedes_doc_id=supersedes_doc_id,
        risk_signals=result["risk_signals"],
    )

    logger.info(
        "AUDIT_LOG",
        extra={
            "event": "document_ingested",
            "doc_id": doc_id,
            "deal_id": deal_id,
            "file_name": filename,
            "category": result["document_category"],
            "chunks_created": result["chunks_created"],
            "parent_chunks_created": result["parent_chunks_created"],
            "previously_indexed": result["previously_indexed"],
            "warnings": result["warnings"],
            "risk_signal_types": [s["signal_type"] for s in result["risk_signals"]],
            "supersedes_doc_id": supersedes_doc_id or "",
        },
    )

    return IngestResponse(
        doc_id=doc_id,
        deal_id=deal_id,
        document_category=result["document_category"],
        chunks_created=result["chunks_created"],
        # Some sheets failed but others indexed — say so rather than "success".
        status="partial" if result["warnings"] else "success",
    )
