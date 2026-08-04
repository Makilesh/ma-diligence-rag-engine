"""
Document ingestion routes.
"""

import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from api.models.response_models import IngestResponse
from api.routes.deals import register_document
from src.data_processing.document_classifier import DocumentClassifier
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter()

# Supported file types
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".xls", ".txt"}


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
    1. Validate file type
    2. Save to temp location
    3. Classify document (if category not provided)
    4. Process with appropriate processor
    5. Chunk (structural → semantic)
    6. Embed and index in Qdrant
    7. Return ingestion summary

    Args:
        file: Uploaded file.
        deal_id: Deal identifier for data isolation.
        document_category: Override category (auto-detected if not provided).
        is_current_version: Whether this is the current version.
        supersedes_doc_id: Doc ID this version supersedes.

    Returns:
        IngestResponse with doc_id and chunk count.
    """
    # Validate file type
    extension = Path(file.filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {extension}. Supported: {SUPPORTED_EXTENSIONS}",
        )

    doc_id = str(uuid.uuid4())

    logger.info(
        "Document ingestion started",
        extra={
            "doc_id": doc_id,
            "deal_id": deal_id,
            "file_name": file.filename,
            "extension": extension,
        },
    )

    # Save uploaded file temporarily
    import tempfile
    import os

    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)

    try:
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        # Auto-classify if category not provided
        if document_category is None:
            classifier = DocumentClassifier()
            document_category = classifier.classify(temp_path, file.filename)

        # Process based on file type
        chunks_created, risk_signals = await _process_and_index(
            file_path=temp_path,
            extension=extension,
            doc_id=doc_id,
            deal_id=deal_id,
            document_category=document_category,
            is_current_version=is_current_version,
            supersedes_doc_id=supersedes_doc_id,
            filename=file.filename,
        )

        # Retire the superseded document's chunks so retrieval's default
        # is_current_version=1 filter stops returning them, and any citation that
        # does surface them (include_stale queries) carries the pointer forward.
        if supersedes_doc_id and chunks_created:
            await _mark_superseded(
                deal_id=deal_id,
                superseded_doc_id=supersedes_doc_id,
                superseded_by=doc_id,
            )

        register_document(
            deal_id=deal_id,
            doc_id=doc_id,
            filename=file.filename,
            document_category=document_category,
            chunks_created=chunks_created,
            is_current_version=is_current_version,
            supersedes_doc_id=supersedes_doc_id,
            risk_signals=risk_signals,
        )

        logger.info(
            "AUDIT_LOG",
            extra={
                "event": "document_ingested",
                "doc_id": doc_id,
                "deal_id": deal_id,
                "file_name": file.filename,
                "category": document_category,
                "chunks_created": chunks_created,
                "risk_signal_types": [s["signal_type"] for s in risk_signals],
                "supersedes_doc_id": supersedes_doc_id or "",
            },
        )

        return IngestResponse(
            doc_id=doc_id,
            deal_id=deal_id,
            document_category=document_category,
            chunks_created=chunks_created,
            status="success",
        )

    except Exception as e:
        logger.error(
            "Document ingestion failed",
            extra={"doc_id": doc_id, "error": str(e)},
        )
        raise HTTPException(status_code=500, detail=f"Ingestion error: {str(e)}")

    finally:
        # Cleanup temp file
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


async def _process_and_index(
    file_path: str,
    extension: str,
    doc_id: str,
    deal_id: str,
    document_category: str,
    is_current_version: bool,
    supersedes_doc_id: str | None,
    filename: str,
) -> tuple[int, list[dict]]:
    """
    Processes document and indexes chunks in Qdrant.

    Args:
        file_path: Path to the saved file.
        extension: File extension.
        doc_id: Document identifier.
        deal_id: Deal identifier.
        document_category: Document category.
        is_current_version: Whether this is the current version.
        supersedes_doc_id: Doc ID this version supersedes.
        filename: Original filename.

    Returns:
        Tuple of (number of chunks created, document-level risk signal dicts).
    """
    from src.data_processing.structural_chunker import StructuralChunker
    from src.data_processing.semantic_chunker import SemanticChunker
    from src.data_processing.pii_detector import PIIDetector
    from src.data_processing.risk_signal_extractor import RiskSignalExtractor
    from src.vector_db.reranker import embed_texts_async
    from src.vector_db.hybrid_search import compute_sparse_bm25
    from src.vector_db.qdrant_client import get_qdrant_client
    from src.vector_db.constants import COLLECTION_NAME
    from qdrant_client.models import PointStruct, NamedVector, NamedSparseVector

    # Step 1: Extract sections based on file type
    sections = []

    if extension == ".pdf":
        from src.data_processing.pdf_processor import PDFProcessor
        is_legal = document_category == "legal"
        processor = PDFProcessor(legal_mode=is_legal)
        pdf_sections = processor.process(file_path, doc_id)
        sections = [
            {
                "text": s.text,
                "section_heading": s.section_heading,
                "page_number": s.page_number,
                "section_type": s.section_type,
                "is_table": s.is_table,
                "clause_id": s.clause_id,
            }
            for s in pdf_sections
        ]

    elif extension == ".docx":
        from src.data_processing.docx_processor import process_docx_with_versions
        clean_chunks, redline_chunks = process_docx_with_versions(file_path, doc_id)
        for chunk in clean_chunks:
            sections.append({
                "text": chunk.text,
                "section_heading": chunk.section_heading,
                "section_type": "text",
            })
        # TODO: Index redline chunks separately with is_redline=1

    elif extension == ".pptx":
        from src.data_processing.pptx_processor import PPTXProcessor
        processor = PPTXProcessor()
        slides = processor.process(file_path, doc_id)
        sections = processor.to_text_chunks(slides)

    elif extension in (".xlsx", ".xls"):
        from src.data_processing.excel_processor import ExcelProcessor
        processor = ExcelProcessor()
        excel_sheets = processor.process(file_path, doc_id)
        sections = processor.to_chunks(excel_sheets)

    elif extension == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        paragraphs = text.split("\n\n")
        for i, para in enumerate(paragraphs):
            para = para.strip()
            if not para:
                continue
            lines = para.split("\n")
            is_table = "|" in para or ("$" in para and "------" in text[max(0, text.index(para)-100):text.index(para)])
            sections.append({
                "text": para,
                "section_heading": lines[0][:100] if len(lines) > 0 else "",
                "page_number": i // 10 + 1,
                "section_type": "table" if is_table else "text",
                "is_table": is_table,
            })

    if not sections:
        return 0, []

    # Step 2: Structural chunking
    structural_chunker = StructuralChunker()
    structural_chunks = structural_chunker.chunk(sections)

    # Step 3: Semantic chunking
    semantic_chunker = SemanticChunker()
    semantic_chunks = semantic_chunker.chunk_batch(structural_chunks)

    if not semantic_chunks:
        return 0, []

    # Step 4: PII detection
    pii_detector = PIIDetector()

    # Step 4b: Risk signal extraction (regex only — no LLM, no added latency).
    # Signals are written per-chunk into the payload AND aggregated to document
    # level for the risk dashboard, which reports per-document not per-chunk.
    risk_extractor = RiskSignalExtractor()
    chunk_risk_signals: dict[int, list[str]] = {}
    aggregated_risk: dict[str, dict] = {}

    for i, chunk in enumerate(semantic_chunks):
        result = risk_extractor.extract(
            chunk.text,
            file_name=filename,
            document_category=document_category,
        )
        if not result.signals:
            continue

        chunk_risk_signals[i] = result.signals

        for detail in result.signal_details:
            signal_type = detail["signal_type"]
            entry = aggregated_risk.setdefault(
                signal_type,
                {
                    "signal_type": signal_type,
                    "match_count": 0,
                    "sample_matches": [],
                    "page_number": chunk.page_number,
                },
            )
            entry["match_count"] += detail["match_count"]
            for sample in detail["sample_matches"]:
                if sample and len(entry["sample_matches"]) < 3:
                    entry["sample_matches"].append(sample)

    risk_signals = list(aggregated_risk.values())

    # Step 5: Embed and index
    texts = [c.text for c in semantic_chunks]
    embeddings = await embed_texts_async(texts)

    client = get_qdrant_client()
    points = []

    import asyncio
    from src.vector_db.reranker import get_embed_executor
    loop = asyncio.get_running_loop()

    for i, chunk in enumerate(semantic_chunks):
        chunk_id = f"{deal_id}_{doc_id}_{i:04d}"
        contains_pii = pii_detector.detect(chunk.text).contains_pii

        # Compute sparse BM25 vector
        sparse_vector = await loop.run_in_executor(
            get_embed_executor(),
            lambda text=chunk.text: compute_sparse_bm25(text),
        )

        point = PointStruct(
            id=hash(chunk_id) % (2**63),
            vector={
                "dense": embeddings[i].tolist(),
                "sparse": sparse_vector,
            },
            payload={
                "chunk_id": chunk_id,
                "deal_id": deal_id,
                "doc_id": doc_id,
                "text": chunk.text,
                "source_file": filename,
                "document_category": document_category,
                "section_heading": chunk.section_heading,
                "page_number": chunk.page_number,
                "clause_id": chunk.clause_id,
                "is_table": chunk.metadata.get("is_table", 0) if hasattr(chunk, "metadata") else 0,
                "is_current_version": 1 if is_current_version else 0,
                "contains_pii": contains_pii,
                "content_type": chunk.metadata.get("content_type", "text") if hasattr(chunk, "metadata") else "text",
                "token_count": chunk.token_count,
                "risk_signals": chunk_risk_signals.get(i, []),
                "supersedes_doc_id": supersedes_doc_id or "",
                "superseded_by": "",  # stamped later if a newer version replaces this doc
                "is_redline": 0,
            },
        )
        points.append(point)

    # Batch upsert
    batch_size = 100
    for batch_start in range(0, len(points), batch_size):
        batch = points[batch_start:batch_start + batch_size]
        await client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch,
        )

    return len(points)
