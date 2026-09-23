"""
Ingestion correctness per file format, plus chunker and identity guarantees.

Each test builds its document programmatically and runs the real code path the
/ingest route uses (extract_and_chunk), stopping short of embeddings. No
network, no LLM, no embedding model.
"""

from __future__ import annotations

import threading
import zipfile

import pytest

from src.data_processing.ingest_pipeline import (
    NoExtractableContentError,
    SUPPORTED_EXTENSIONS,
    UnsafeDocumentError,
    UnsupportedFileTypeError,
    compute_doc_id,
    extract_and_chunk,
    point_id_for,
    sanitize_filename,
    validate_office_archive,
)
from tests.fixtures import ingestion_docs as fx

DEAL = "deal-test"
MAX_TOKENS = 800


def _text_chunks(doc):
    return [c for c in doc.chunks if not c["is_table"] and not c["is_redline"]]


def _table_groups(doc) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for c in doc.chunks:
        if c.get("table_id"):
            groups.setdefault(c["table_id"], []).append(c)
    return groups


def _assert_common_invariants(doc):
    assert doc.chunks, "document produced no chunks"
    ids = [c["chunk_id"] for c in doc.chunks]
    assert len(ids) == len(set(ids))
    for c in doc.chunks:
        assert c["deal_id"] == DEAL
        assert c["doc_id"] == doc.doc_id
        assert c["text"].strip()
        assert c["token_count"] <= MAX_TOKENS
        assert c["is_current_version"] == 1
        assert c["contains_pii"] in (0, 1)
    # Every prose chunk points at a parent that actually exists.
    parent_ids = {p["chunk_id"] for p in doc.parents}
    for c in _text_chunks(doc):
        assert c["parent_chunk_id"] in parent_ids
    for p in doc.parents:
        assert p["deal_id"] == DEAL and p["text"].strip()


# ==============================================================================
# Formats
# ==============================================================================


class TestPdf:
    def test_pdf_has_real_pages_and_table_representations(self, tmp_path):
        path = fx.write_pdf(tmp_path)
        doc = extract_and_chunk(str(path), path.name, DEAL, "financial")
        _assert_common_invariants(doc)

        pages = {c["page_number"] for c in doc.chunks}
        assert pages <= {1, 2}
        assert 1 in pages and 2 in pages, "text on page 2 must be cited as page 2"
        page_one = [c for c in _text_chunks(doc) if "seventeen percent" in c["text"]]
        assert page_one and page_one[0]["page_number"] == 1

        groups = _table_groups(doc)
        assert len(groups) == 1
        reps = next(iter(groups.values()))
        assert {r["table_representation"] for r in reps} >= {"narrative", "row_by_row", "markdown"}
        assert all(r["page_number"] == 2 and r["is_table"] == 1 for r in reps)
        # Amounts are carried exactly — no rounding of 387.1 to 387.
        row_by_row = next(r for r in reps if r["table_representation"] == "row_by_row")
        assert "387.1" in row_by_row["text"] and "452.8" in row_by_row["text"]

    def test_legal_pdf_uses_clause_segmentation(self, tmp_path):
        path = fx.write_pdf(tmp_path, "agreement.pdf")
        doc = extract_and_chunk(str(path), path.name, DEAL, "legal")
        _assert_common_invariants(doc)
        assert all(c["page_number"] in (1, 2) for c in doc.chunks)


class TestDocx:
    def test_tracked_insertion_is_in_clean_text_and_deletion_is_not(self, tmp_path):
        path = fx.write_docx(tmp_path)
        doc = extract_and_chunk(str(path), path.name, DEAL)
        _assert_common_invariants(doc)

        clean = " ".join(c["text"] for c in _text_chunks(doc))
        assert "twenty-five percent of the purchase price" in clean
        assert "ten percent" not in clean
        assert all(c["page_number"] is None for c in doc.chunks)
        assert all(c["section_heading"] == "Article VIII Indemnification" for c in _text_chunks(doc))

    def test_only_changed_paragraphs_are_redlines(self, tmp_path):
        path = fx.write_docx(tmp_path)
        doc = extract_and_chunk(str(path), path.name, DEAL)
        assert doc.has_redline

        redlines = [c for c in doc.chunks if c["is_redline"]]
        assert len(redlines) == 1
        assert redlines[0]["content_type"] == "redline"
        assert "(+twenty-five percent" in redlines[0]["text"]
        assert "(~~ten percent" in redlines[0]["text"]
        assert "parent_chunk_id" not in redlines[0]

    def test_tables_are_extracted(self, tmp_path):
        path = fx.write_docx(tmp_path)
        doc = extract_and_chunk(str(path), path.name, DEAL)
        groups = _table_groups(doc)
        assert len(groups) == 1
        reps = next(iter(groups.values()))
        assert any("General Escrow" in r["text"] for r in reps)
        assert all(r["section_heading"] == "Schedule of Escrow" for r in reps)


class TestXlsx:
    def test_sheet_produces_linked_representations(self, tmp_path):
        path = fx.write_xlsx(tmp_path)
        doc = extract_and_chunk(str(path), path.name, DEAL)
        _assert_common_invariants(doc)
        assert doc.document_category == "financial"

        income = [c for c in doc.chunks if c.get("sheet_name") == "Income Statement"]
        assert len({c["table_id"] for c in income}) == 1
        by_rep = {c["table_representation"]: c for c in income}
        assert set(by_rep) == {"narrative", "row_by_row", "metrics_summary", "markdown"}

        assert by_rep["narrative"]["content_type"] == "table_narrative"
        assert by_rep["metrics_summary"]["content_type"] == "computed_metric"
        assert "Revenue CAGR" in by_rep["metrics_summary"]["text"]
        assert by_rep["row_by_row"]["scale_label"] == "thousands"
        assert by_rep["row_by_row"]["currency"] == "USD"
        # Normalised with the detected scale: 452,800 thousand.
        assert "452,800,000" in by_rep["row_by_row"]["text"]
        assert all(c["page_number"] is None for c in income)

    def test_text_only_sheet_is_indexed_verbatim(self, tmp_path):
        path = fx.write_xlsx(tmp_path)
        doc = extract_and_chunk(str(path), path.name, DEAL)
        notes = [c for c in doc.chunks if c.get("sheet_name") == "Notes"]
        assert len(notes) == 1
        assert notes[0]["content_type"] == "table_text"
        assert "Deloitte" in notes[0]["text"]

    def test_all_sheets_failing_is_an_error_not_an_empty_success(self, tmp_path, monkeypatch):
        from src.data_processing import excel_processor
        from src.data_processing.ingest_pipeline import DocumentExtractionError

        def boom(self, *args, **kwargs):
            raise RuntimeError("converter exploded")

        monkeypatch.setattr(excel_processor.ExcelProcessor, "_process_sheet", boom)
        path = fx.write_xlsx(tmp_path)
        with pytest.raises(DocumentExtractionError):
            extract_and_chunk(str(path), path.name, DEAL)

    def test_xls_is_not_advertised(self):
        assert ".xls" not in SUPPORTED_EXTENSIONS
        with pytest.raises(UnsupportedFileTypeError):
            extract_and_chunk(__file__, "legacy.xls", DEAL)


class TestPptx:
    def test_slide_text_and_table(self, tmp_path):
        path = fx.write_pptx(tmp_path)
        doc = extract_and_chunk(str(path), path.name, DEAL)
        _assert_common_invariants(doc)

        slide = [c for c in doc.chunks if c["content_type"] == "slide"]
        assert slide and "Three bidders" in slide[0]["text"]
        # The slide number is the deck's page locator.
        assert all(c["page_number"] == 1 and c["slide_number"] == 1 for c in doc.chunks)

        reps = next(iter(_table_groups(doc).values()))
        assert len(reps) >= 2
        assert any("Vertex" in r["text"] for r in reps)


class TestTxt:
    def test_no_fabricated_pages_and_real_headings(self, tmp_path):
        path = fx.write_txt(tmp_path)
        doc = extract_and_chunk(str(path), path.name, DEAL)
        _assert_common_invariants(doc)

        assert all(c["page_number"] is None for c in doc.chunks)
        headings = {c["section_heading"] for c in doc.chunks}
        assert "Section 8.2 — Indemnification Cap" in headings
        cap = next(c for c in doc.chunks if "$174 million" in c["text"])
        assert cap["section_heading"] == "Section 8.2 — Indemnification Cap"

    def test_text_table_keeps_heading_and_scale(self, tmp_path):
        path = fx.write_txt(tmp_path)
        doc = extract_and_chunk(str(path), path.name, DEAL)
        tables = [c for c in doc.chunks if c["is_table"]]
        assert len(tables) == 1
        table = tables[0]
        assert table["content_type"] == "table_text"
        assert table["table_id"]
        assert table["section_heading"] == "CONSOLIDATED INCOME STATEMENT (in millions of USD)"
        # Rows split by blank lines stay one table, with the unit attached.
        assert "(in millions of USD)" in table["text"]
        assert "$452.8" in table["text"] and "60.0%" in table["text"]

    def test_repeated_paragraphs_are_all_kept(self, tmp_path):
        """The old text.index(para) heuristic mis-handled repeated paragraphs."""
        para = "The Buyer shall deliver the closing certificate on the Closing Date."
        text = f"SECTION 1: CLOSING\n\n{para}\n\nSECTION 2: DELIVERIES\n\n{para}\n"
        path = fx.write_txt(tmp_path, "repeat.txt", text)
        doc = extract_and_chunk(str(path), path.name, DEAL)
        hits = [c for c in doc.chunks if para in c["text"]]
        assert {c["section_heading"] for c in hits} == {"SECTION 1: CLOSING", "SECTION 2: DELIVERIES"}

    def test_empty_document_is_rejected(self, tmp_path):
        path = fx.write_txt(tmp_path, "empty.txt", "   \n\n  ")
        with pytest.raises(NoExtractableContentError):
            extract_and_chunk(str(path), path.name, DEAL)


# ==============================================================================
# Identity
# ==============================================================================


class TestDeterministicIds:
    def test_same_content_same_ids(self, tmp_path):
        path = fx.write_txt(tmp_path)
        first = extract_and_chunk(str(path), path.name, DEAL)
        second = extract_and_chunk(str(path), path.name, DEAL)
        assert first.doc_id == second.doc_id
        assert [c["chunk_id"] for c in first.chunks] == [c["chunk_id"] for c in second.chunks]
        assert [p["chunk_id"] for p in first.parents] == [p["chunk_id"] for p in second.parents]

    def test_doc_id_depends_on_deal_and_content(self, tmp_path):
        a = fx.write_txt(tmp_path, "a.txt")
        b = fx.write_txt(tmp_path, "b.txt", fx.SAMPLE_TXT + "\nAddendum.\n")
        doc_a = extract_and_chunk(str(a), a.name, DEAL)
        assert extract_and_chunk(str(a), a.name, "other-deal").doc_id != doc_a.doc_id
        assert extract_and_chunk(str(b), b.name, DEAL).doc_id != doc_a.doc_id

    def test_point_ids_are_stable_across_processes(self):
        # A literal, not a recomputation: hash() is salted per process, uuid5 is not.
        assert point_id_for("deal_doc_0000") == "5b8d02cf-f2d8-5b4f-97c7-d5b4d49a6bd2"
        assert compute_doc_id("deal", "0" * 64) == "2a6ffc14-4cda-5c8b-9f9e-2b7cb2d1a3c4"


# ==============================================================================
# Input validation
# ==============================================================================


class TestFilenameSanitisation:
    @pytest.mark.parametrize("raw, expected", [
        ("report.pdf", "report.pdf"),
        ("../../etc/passwd.txt", "passwd.txt"),
        ("..\\..\\Windows\\evil.docx", "evil.docx"),
        ("C:\\Users\\a\\deck.pptx", "deck.pptx"),
        ("/abs/path/model.xlsx", "model.xlsx"),
        ("we<ird>:name?.txt", "we_ird__name_.txt"),
        ("tab\tand\nnewline.txt", "tabandnewline.txt"),
    ])
    def test_reduced_to_safe_basename(self, raw, expected):
        assert sanitize_filename(raw) == expected

    @pytest.mark.parametrize("raw", ["", None, ".", "..", ".env", "dir/.bashrc", "a/b/", "  "])
    def test_rejected(self, raw):
        with pytest.raises(UnsupportedFileTypeError):
            sanitize_filename(raw)


class TestZipBombGuard:
    def test_oversize_archive_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MAX_UNCOMPRESSED_MB", "1")
        path = tmp_path / "bomb.docx"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("word/document.xml", b"\0" * (3 * 1024 * 1024))
        with pytest.raises(UnsafeDocumentError):
            validate_office_archive(str(path), ".docx")

    def test_high_ratio_entry_rejected(self, tmp_path):
        path = tmp_path / "bomb.xlsx"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("xl/sheet1.xml", b"\0" * (20 * 1024 * 1024))
        with pytest.raises(UnsafeDocumentError):
            validate_office_archive(str(path), ".xlsx")

    def test_not_a_zip_is_an_extraction_error(self, tmp_path):
        from src.data_processing.ingest_pipeline import DocumentExtractionError

        path = tmp_path / "fake.pptx"
        path.write_bytes(b"not a zip")
        with pytest.raises(DocumentExtractionError):
            validate_office_archive(str(path), ".pptx")

    def test_real_office_file_passes(self, tmp_path):
        validate_office_archive(str(fx.write_docx(tmp_path)), ".docx")


# ==============================================================================
# Chunker guarantees
# ==============================================================================


def _run_with_timeout(fn, seconds: float = 20.0):
    """Runs fn in a daemon thread; fails instead of hanging if it never returns."""
    result: dict = {}

    def target():
        try:
            result["value"] = fn()
        except BaseException as e:  # surface errors from the worker
            result["error"] = e

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        pytest.fail(f"chunker did not terminate within {seconds}s (infinite loop)")
    if "error" in result:
        raise result["error"]
    return result["value"]


class TestSemanticChunkerTermination:
    def test_single_sentence_chunk_before_the_end_terminates(self):
        """
        A chunk made of one sentence that is not the last used to step the
        overlap back onto that same sentence forever.
        """
        from src.data_processing.semantic_chunker import SemanticChunker

        chunker = SemanticChunker(target_tokens=40, min_tokens=1, max_tokens=60, overlap_ratio=0.5)
        long_sentence = "The indemnification cap applies to " + "every covered claim " * 10 + "here."
        text = " ".join([long_sentence, "Short one.", long_sentence, "Another short one."])

        chunks = _run_with_timeout(lambda: chunker.chunk(text))
        assert chunks
        assert all(c.token_count <= 60 for c in chunks)
        assert "Another short one." in chunks[-1].text

    def test_oversize_sentence_is_hard_split(self):
        from src.data_processing.semantic_chunker import SemanticChunker

        chunker = SemanticChunker(target_tokens=100, min_tokens=10, max_tokens=120)
        run_on = " ".join(f"item{i}" for i in range(2000))  # one "sentence", no punctuation
        chunks = _run_with_timeout(lambda: chunker.chunk(run_on))
        assert len(chunks) > 1
        assert all(c.token_count <= 120 for c in chunks)

    def test_oversize_table_is_split_on_rows(self):
        from src.data_processing.semantic_chunker import SemanticChunker

        chunker = SemanticChunker(target_tokens=100, min_tokens=10, max_tokens=120)
        table = "\n".join(f"Line item {i}    ${i}.5    ${i}.7" for i in range(300))
        chunks = _run_with_timeout(lambda: chunker.chunk_batch([
            {"text": table, "is_table": True, "metadata": {"table_id": "t1"}},
        ]))
        assert len(chunks) > 1
        assert all(c.token_count <= 120 for c in chunks)
        assert all(c.metadata["table_id"] == "t1" for c in chunks)
        assert all(c.metadata["content_type"] == "table_text" for c in chunks)
        # Rows are never cut mid-line.
        for c in chunks:
            for line in c.text.split("\n"):
                assert line.startswith("Line item ")


class TestStructuralMetadata:
    def test_top_level_section_keys_survive_chunking(self):
        from src.data_processing.structural_chunker import StructuralChunker

        chunks = StructuralChunker().chunk([{
            "text": "Revenue: FY2023=452,800,000",
            "section_heading": "Income Statement",
            "page_number": None,
            "is_table": 1,
            "sheet_name": "Income Statement",
            "table_id": "deal_doc_t000",
            "content_type": "table_row_by_row",
            "currency": "USD",
        }])
        meta = chunks[0].metadata
        assert meta["sheet_name"] == "Income Statement"
        assert meta["table_id"] == "deal_doc_t000"
        assert meta["content_type"] == "table_row_by_row"
        assert meta["currency"] == "USD"

    def test_small_sections_do_not_merge_across_headings(self):
        from src.data_processing.structural_chunker import StructuralChunker

        chunks = StructuralChunker().chunk([
            {"text": "Short clause A.", "section_heading": "Section 1"},
            {"text": "Short clause B.", "section_heading": "Section 2"},
        ])
        assert [c.section_heading for c in chunks] == ["Section 1", "Section 2"]


class TestTableStitcher:
    def test_continuation_requires_the_next_page(self):
        from src.data_processing.multi_page_table_stitcher import (
            ExtractedTable,
            MultiPageTableStitcher,
            StitchedTable,
        )

        stitcher = MultiPageTableStitcher()
        current = StitchedTable(rows=[["Revenue", "1"]], headers=["Item", "FY"], page_range=[1, 1])
        same_page = ExtractedTable(rows=[["Cost", "2"]], headers=[], page_number=1, col_count=2, has_header=False)
        next_page = ExtractedTable(rows=[["Cost", "2"]], headers=[], page_number=2, col_count=2, has_header=False)
        far_page = ExtractedTable(rows=[["Cost", "2"]], headers=[], page_number=5, col_count=2, has_header=False)

        assert stitcher._is_continuation(current, next_page)
        assert not stitcher._is_continuation(current, same_page)
        assert not stitcher._is_continuation(current, far_page)
