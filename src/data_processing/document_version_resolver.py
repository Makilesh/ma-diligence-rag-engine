# src/data_processing/document_version_resolver.py
"""
Document version lifecycle management for M&A due diligence.

Handles version resolution at ingestion time:
- is_current_version assignment (0 or 1)
- supersedes_doc_id tracking
- Soft-delete of superseded versions (sets is_current_version=0)

Version detection uses filename patterns (v2, _rev3, _final, _amended)
and content-based heuristics (date comparison, explicit version statements).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# ─── Version extraction patterns ─────────────────────────────────────────────
VERSION_PATTERNS = [
    # Explicit version numbers: v1, v2.1, ver3, version4
    re.compile(r'[_\-\s.]v(?:er(?:sion)?)?\s*(\d+(?:\.\d+)*)', re.IGNORECASE),
    # Revision markers: rev1, rev_2, revision3
    re.compile(r'[_\-\s.]rev(?:ision)?\s*[_\-]?\s*(\d+)', re.IGNORECASE),
    # Draft markers: draft1, draft_2
    re.compile(r'[_\-\s.]draft\s*[_\-]?\s*(\d+)', re.IGNORECASE),
    # Date-based versions: 2024-01-15, 20240115, Jan2024
    re.compile(r'(\d{4}[-_]?\d{2}[-_]?\d{2})'),
]

# ─── Finality markers — documents with these are typically the latest ─────────
FINALITY_PATTERNS = [
    re.compile(r'[_\-\s.]final\b', re.IGNORECASE),
    re.compile(r'[_\-\s.]executed\b', re.IGNORECASE),
    re.compile(r'[_\-\s.]signed\b', re.IGNORECASE),
    re.compile(r'[_\-\s.]amended\b', re.IGNORECASE),
    re.compile(r'[_\-\s.]restated\b', re.IGNORECASE),
]


@dataclass
class DocumentVersion:
    """Represents the version metadata for a single document."""
    doc_id: str
    file_name: str
    version_string: str | None = None
    version_number: float | None = None
    version_date: str | None = None
    is_final: bool = False
    is_current_version: int = 1  # 0 or 1 — integer for Qdrant payload
    supersedes_doc_id: str | None = None
    base_name: str = ""  # Filename with version indicators stripped


@dataclass
class VersionGroup:
    """A group of documents that are versions of the same base document."""
    base_name: str
    versions: list[DocumentVersion] = field(default_factory=list)


class DocumentVersionResolver:
    """
    Resolves document versions at ingestion time.

    Detects version relationships between documents by:
    1. Stripping version indicators from filenames to find common base names
    2. Comparing version numbers, dates, and finality markers
    3. Assigning is_current_version=1 only to the latest version
    4. Setting supersedes_doc_id to link version chains

    Soft-delete: sets is_current_version=0, does NOT remove from Qdrant.
    Old versions remain searchable when explicitly requested via filter override.
    """

    def __init__(self) -> None:
        """Initialize the version resolver."""
        self._version_groups: dict[str, VersionGroup] = {}

    def resolve_version(
        self,
        doc_id: str,
        file_name: str,
        existing_docs: list[dict] | None = None,
    ) -> DocumentVersion:
        """
        Resolve the version status of a newly ingested document.

        Checks the document against existing documents to determine:
        - Whether this is a new version of an existing document
        - Which document it supersedes (if any)
        - Whether it should be the current version

        Args:
            doc_id: Unique document identifier.
            file_name: Original filename of the document.
            existing_docs: List of already-ingested document metadata dicts
                           with keys: doc_id, file_name, version_number, is_current_version.

        Returns:
            DocumentVersion with resolved version metadata.

        Raises:
            ValueError: If doc_id is empty.
        """
        if not doc_id:
            raise ValueError("doc_id cannot be empty")

        logger.info(
            "Resolving document version",
            extra={"doc_id": doc_id, "file_name": file_name},
        )

        # Extract version info from filename
        version_info = self._extract_version_info(file_name)
        base_name = self._extract_base_name(file_name)

        doc_version = DocumentVersion(
            doc_id=doc_id,
            file_name=file_name,
            version_string=version_info.get("version_string"),
            version_number=version_info.get("version_number"),
            version_date=version_info.get("version_date"),
            is_final=version_info.get("is_final", False),
            base_name=base_name,
        )

        # Check against existing documents
        if existing_docs:
            superseded = self._find_superseded(doc_version, existing_docs)
            if superseded:
                doc_version.supersedes_doc_id = superseded["doc_id"]
                doc_version.is_current_version = 1
                logger.info(
                    "New version supersedes existing document",
                    extra={
                        "doc_id": doc_id,
                        "supersedes": superseded["doc_id"],
                    },
                )
            else:
                doc_version.is_current_version = 1
        else:
            # First document — it's current by default
            doc_version.is_current_version = 1

        logger.info(
            "Version resolved",
            extra={
                "doc_id": doc_id,
                "is_current_version": doc_version.is_current_version,
                "supersedes_doc_id": doc_version.supersedes_doc_id,
                "version_number": doc_version.version_number,
            },
        )

        return doc_version

    def get_superseded_doc_ids(
        self,
        new_doc: DocumentVersion,
        existing_docs: list[dict],
    ) -> list[str]:
        """
        Get the doc_ids of all documents superseded by the new document.

        These documents should have their is_current_version set to 0.

        Args:
            new_doc: The newly resolved document version.
            existing_docs: Existing document metadata.

        Returns:
            List of doc_ids that should be marked as superseded.
        """
        superseded_ids: list[str] = []

        for existing in existing_docs:
            existing_base = self._extract_base_name(existing["file_name"])

            if existing_base == new_doc.base_name:
                existing_version = existing.get("version_number", 0)
                if (
                    new_doc.version_number is not None
                    and existing_version is not None
                    and new_doc.version_number > existing_version
                ):
                    superseded_ids.append(existing["doc_id"])
                elif new_doc.is_final and not existing.get("is_final", False):
                    superseded_ids.append(existing["doc_id"])

        return superseded_ids

    def _extract_version_info(self, file_name: str) -> dict[str, Any]:
        """
        Extract version information from a filename.

        Args:
            file_name: Document filename.

        Returns:
            Dict with version_string, version_number, version_date, is_final.
        """
        info: dict[str, Any] = {
            "version_string": None,
            "version_number": None,
            "version_date": None,
            "is_final": False,
        }

        # Check for finality markers
        for pattern in FINALITY_PATTERNS:
            if pattern.search(file_name):
                info["is_final"] = True
                break

        # Extract version number
        for pattern in VERSION_PATTERNS[:3]:  # Skip date pattern initially
            match = pattern.search(file_name)
            if match:
                version_str = match.group(1)
                info["version_string"] = version_str
                try:
                    info["version_number"] = float(version_str)
                except ValueError:
                    info["version_number"] = None
                break

        # Extract date-based version
        date_pattern = VERSION_PATTERNS[3]
        date_match = date_pattern.search(file_name)
        if date_match:
            info["version_date"] = date_match.group(1)
            # If no explicit version number, use date as version
            if info["version_number"] is None:
                date_str = date_match.group(1).replace("-", "").replace("_", "")
                try:
                    info["version_number"] = float(date_str)
                except ValueError:
                    pass

        return info

    def _extract_base_name(self, file_name: str) -> str:
        """
        Strip version indicators and extension from filename to get base name.

        This enables matching different versions of the same document:
        "Agreement_v1.pdf" and "Agreement_v2.pdf" → "agreement"

        Args:
            file_name: Original filename.

        Returns:
            Normalized base name without version indicators.
        """
        # Remove extension
        name = re.sub(r'\.[^.]+$', '', file_name)

        # Remove version indicators
        for pattern in VERSION_PATTERNS:
            name = pattern.sub('', name)

        for pattern in FINALITY_PATTERNS:
            name = pattern.sub('', name)

        # Normalize separators and whitespace
        name = re.sub(r'[_\-\s.]+', '_', name).strip('_').lower()

        return name

    def _find_superseded(
        self,
        new_doc: DocumentVersion,
        existing_docs: list[dict],
    ) -> dict | None:
        """
        Find the existing document that the new document supersedes.

        Args:
            new_doc: The new document version.
            existing_docs: List of existing document metadata.

        Returns:
            The superseded document's metadata dict, or None.
        """
        candidates: list[dict] = []

        for existing in existing_docs:
            existing_base = self._extract_base_name(existing["file_name"])

            if existing_base == new_doc.base_name:
                candidates.append(existing)

        if not candidates:
            return None

        # Find the current version among candidates
        current = next(
            (c for c in candidates if c.get("is_current_version", 0) == 1),
            None,
        )

        if current is None:
            return None

        # Determine if new doc is newer
        existing_version = current.get("version_number", 0) or 0
        new_version = new_doc.version_number or 0

        if new_version > existing_version:
            return current

        if new_doc.is_final and not current.get("is_final", False):
            return current

        # If same version but new doc has a later date
        if (
            new_doc.version_date
            and current.get("version_date")
            and new_doc.version_date > current["version_date"]
        ):
            return current

        return None

    def build_version_metadata(self, doc_version: DocumentVersion) -> dict:
        """
        Build the version-related payload fields for Qdrant ingestion.

        Args:
            doc_version: Resolved DocumentVersion.

        Returns:
            Dict with is_current_version, supersedes_doc_id, document_version.
        """
        return {
            "is_current_version": doc_version.is_current_version,
            "supersedes_doc_id": doc_version.supersedes_doc_id,
            "document_version": doc_version.version_string,
        }
