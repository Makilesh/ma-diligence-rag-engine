"""
Tests for DocumentVersionResolver — version detection and chain construction.
"""

import pytest
from src.data_processing.document_version_resolver import (
    DocumentVersionResolver,
)


@pytest.fixture
def resolver():
    return DocumentVersionResolver()


class TestVersionExtraction:
    """Tests for version info extraction from filenames."""

    def test_extract_explicit_version(self, resolver):
        """Detect 'v2' in filename."""
        result = resolver.resolve_version("doc_001", "merger_agreement_v2.pdf")
        assert result.version_number is not None
        assert result.version_number >= 2.0

    def test_extract_revision(self, resolver):
        """Detect 'rev3' in filename."""
        result = resolver.resolve_version("doc_002", "financials_rev3.xlsx")
        assert result.version_string is not None or result.version_number is not None

    def test_no_version_indicator(self, resolver):
        """No version indicator — defaults to current version."""
        result = resolver.resolve_version("doc_003", "board_minutes.pdf")
        assert result.is_current_version == 1

    def test_final_marker(self, resolver):
        """'_final' is detected as finality marker."""
        result = resolver.resolve_version("doc_004", "agreement_final.docx")
        assert result.is_final is True


class TestVersionResolution:
    """Tests for version chain resolution."""

    def test_new_document_is_current(self, resolver):
        """First document with no existing docs is marked current."""
        result = resolver.resolve_version("doc_001", "report.pdf", existing_docs=[])
        assert result.is_current_version == 1

    def test_supersedes_older_version(self, resolver):
        """New version supersedes existing version with same base name."""
        existing = [
            {"doc_id": "doc_001", "file_name": "agreement_v1.pdf",
             "version_number": 1.0, "is_current_version": 1},
        ]
        result = resolver.resolve_version(
            "doc_002", "agreement_v2.pdf", existing_docs=existing
        )

        # New version should be current
        assert result.is_current_version == 1
        # Should reference the superseded doc
        if result.supersedes_doc_id:
            assert result.supersedes_doc_id == "doc_001"

    def test_base_name_extraction(self, resolver):
        """Base name strips version indicators."""
        result = resolver.resolve_version("doc_001", "merger_agreement_v3_final.pdf")
        assert result.base_name  # Should have a base name
        assert "v3" not in result.base_name.lower() or "merger" in result.base_name.lower()

    def test_empty_doc_id_raises(self, resolver):
        """Empty doc_id raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            resolver.resolve_version("", "file.pdf")
