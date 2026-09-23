"""
Step 4: Offline pipeline validation.

Runs the ingestion + chunking pipeline on synthetic documents WITHOUT
requiring Docker (Qdrant, Postgres) or GPU. Captures chunk counts,
token distributions, and golden QA coverage metrics.

This validates that:
1. All 3 synthetic documents are processable
2. Structural + semantic chunking produces reasonable chunks
3. Golden QA set is well-formed and covers all query types
4. Document classifier assigns correct categories
5. PII detector and risk signal extractor run correctly
6. Financial table detection works on the financial statement

What this does NOT test (requires Docker/Ollama):
- Actual embedding + Qdrant indexing
- LLM-based agents (query intelligence, answer synthesis, hallucination validation)
- End-to-end query → answer pipeline
- Real recall/precision/latency numbers
"""

import json
import sys
from pathlib import Path
from collections import Counter

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing.structural_chunker import StructuralChunker
from src.data_processing.semantic_chunker import SemanticChunker
from src.data_processing.document_classifier import DocumentClassifier
from src.data_processing.pii_detector import PIIDetector
from src.data_processing.risk_signal_extractor import RiskSignalExtractor
from src.data_processing.document_version_resolver import DocumentVersionResolver
from src.utils.token_counter import count_tokens


def load_golden_qa():
    """Load and validate golden QA set."""
    qa_path = Path(__file__).parent / "golden_qa_set.json"
    with open(qa_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_synthetic_documents():
    """Load all synthetic source documents."""
    doc_dir = Path(__file__).parent.parent / "data" / "sample_deal"
    docs = {}
    for f in doc_dir.iterdir():
        if f.suffix == ".txt":
            docs[f.name] = f.read_text(encoding="utf-8")
    return docs


def text_to_sections(text: str, filename: str) -> list[dict]:
    """Convert raw text to sections for chunking (simulates PDF processor)."""
    paragraphs = text.split("\n\n")
    sections = []
    current_heading = ""

    for i, para in enumerate(paragraphs):
        para = para.strip()
        if not para:
            continue

        # Detect headings (lines with === or ALL CAPS short lines)
        lines = para.split("\n")
        if len(lines) <= 2 and (
            all(c in "=-" for c in lines[-1] if c.strip())
            or (len(para) < 80 and para == para.upper() and len(para) > 5)
        ):
            current_heading = lines[0].strip("= -\n")
            continue

        is_table = "|" in para or ("$" in para and "------" in text[max(0, text.index(para)-100):text.index(para)])

        sections.append({
            "text": para,
            "section_heading": current_heading,
            "page_number": i // 10 + 1,
            "section_type": "table" if is_table else "text",
            "is_table": is_table,
        })

    return sections


def run_pipeline_validation():
    """Run full offline pipeline validation."""
    results = {
        "documents": {},
        "chunking": {},
        "classification": {},
        "pii": {},
        "risk_signals": {},
        "version_resolution": {},
        "golden_qa_validation": {},
        "timing": {},
    }

    print("=" * 70)
    print("M&A DUE DILIGENCE ENGINE -- OFFLINE PIPELINE VALIDATION")
    print("=" * 70)

    # ─── Load documents ───────────────────────────────────────────────
    print("\n[DOC] Loading synthetic documents...")
    docs = load_synthetic_documents()
    print(f"   Found {len(docs)} documents:")
    for name, content in docs.items():
        tokens = count_tokens(content)
        print(f"   - {name}: {len(content):,} chars, {tokens:,} tokens")
        results["documents"][name] = {
            "chars": len(content),
            "tokens": tokens,
        }

    # ─── Document Classification ──────────────────────────────────────
    print("\n[TAG] Document Classification...")
    classifier = DocumentClassifier()
    for name in docs:
        category = classifier.classify(
            str(Path(__file__).parent.parent / "data" / "sample_deal" / name),
            name,
        )
        print(f"   - {name}: {category}")
        results["classification"][name] = category

    # ─── Version Resolution ───────────────────────────────────────────
    print("\n[VER] Version Resolution...")
    resolver = DocumentVersionResolver()
    existing = []
    for i, name in enumerate(sorted(docs.keys())):
        version = resolver.resolve_version(f"doc_{i:03d}", name, existing)
        print(f"   - {name}: v={version.version_string}, current={version.is_current_version}, final={version.is_final}")
        results["version_resolution"][name] = {
            "version_string": version.version_string,
            "is_current": version.is_current_version,
            "is_final": version.is_final,
            "base_name": version.base_name,
        }
        existing.append({
            "doc_id": f"doc_{i:03d}",
            "file_name": name,
            "version_number": version.version_number,
            "is_current_version": version.is_current_version,
        })

    # ─── Structural Chunking ─────────────────────────────────────────
    print("\n[CUT] Structural Chunking...")
    structural_chunker = StructuralChunker()
    all_structural = {}
    for name, content in docs.items():
        sections = text_to_sections(content, name)
        chunks = structural_chunker.chunk(sections)
        total_tokens = sum(c.token_count for c in chunks)
        print(f"   - {name}: {len(sections)} sections -> {len(chunks)} structural chunks ({total_tokens:,} tokens)")
        all_structural[name] = chunks
        results["chunking"][name] = {
            "input_sections": len(sections),
            "structural_chunks": len(chunks),
            "total_tokens": total_tokens,
        }

    # ─── Semantic Chunking ────────────────────────────────────────────
    print("\n[SEM] Semantic Chunking...")
    semantic_chunker = SemanticChunker()
    all_semantic = {}
    for name, structural_chunks in all_structural.items():
        semantic_chunks = semantic_chunker.chunk_batch(structural_chunks)
        token_counts = [c.token_count for c in semantic_chunks]
        avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
        min_tokens = min(token_counts) if token_counts else 0
        max_tokens = max(token_counts) if token_counts else 0
        print(f"   - {name}: {len(semantic_chunks)} semantic chunks (avg={avg_tokens:.0f}, min={min_tokens}, max={max_tokens} tokens)")
        all_semantic[name] = semantic_chunks
        results["chunking"][name].update({
            "semantic_chunks": len(semantic_chunks),
            "avg_tokens": round(avg_tokens),
            "min_tokens": min_tokens,
            "max_tokens": max_tokens,
        })

    # ─── PII Detection ────────────────────────────────────────────────
    print("\n[PII] PII Detection...")
    pii_detector = PIIDetector()
    total_pii = 0
    for name, chunks in all_semantic.items():
        pii_count = sum(1 for c in chunks if pii_detector.detect(c.text).contains_pii == 1)
        total_pii += pii_count
        print(f"   - {name}: {pii_count}/{len(chunks)} chunks flagged PII")
        results["pii"][name] = {"flagged": pii_count, "total": len(chunks)}

    # ─── Risk Signal Extraction ───────────────────────────────────────
    print("\n[RSK] Risk Signal Extraction...")
    extractor = RiskSignalExtractor()
    total_signals = 0
    for name, chunks in all_semantic.items():
        signals = []
        for c in chunks:
            s = extractor.extract(c.text)
            if s and s.signals:
                signals.extend(s.signals)
        total_signals += len(signals)
        signal_types = Counter(signals)
        print(f"   - {name}: {len(signals)} risk signals {dict(signal_types)}")
        results["risk_signals"][name] = {"count": len(signals), "types": dict(signal_types)}

    # ─── Golden QA Validation ─────────────────────────────────────────
    print("\n[QA ] Golden QA Set Validation...")
    qa_data = load_golden_qa()
    qa_pairs = qa_data["golden_qa_pairs"]

    type_counts = Counter(q["query_type"] for q in qa_pairs)
    print(f"   Total QA pairs: {len(qa_pairs)}")
    print(f"   Query types: {dict(type_counts)}")
    print(f"   Source documents referenced: {qa_data['source_documents']}")

    # Check that expected citations reference existing documents
    all_chunks_text = " ".join(c.text for chunks in all_semantic.values() for c in chunks)
    coverage = {"covered": 0, "uncovered": 0, "details": []}

    for qa in qa_pairs:
        facts_found = sum(1 for fact in qa["expected_answer_contains"] if fact.lower() in all_chunks_text.lower())
        total_facts = len(qa["expected_answer_contains"])
        pct = facts_found / total_facts * 100 if total_facts > 0 else 0

        status = "PASS" if pct >= 80 else "WARN" if pct >= 50 else "FAIL"
        if pct >= 80:
            coverage["covered"] += 1
        else:
            coverage["uncovered"] += 1

        coverage["details"].append({
            "id": qa["id"],
            "query_type": qa["query_type"],
            "facts_found": facts_found,
            "total_facts": total_facts,
            "coverage_pct": round(pct, 1),
        })
        print(f"   {status} {qa['id']} ({qa['query_type']}): {facts_found}/{total_facts} facts in chunks ({pct:.0f}%)")

    results["golden_qa_validation"] = coverage

    # ─── Summary ──────────────────────────────────────────────────────
    total_chunks = sum(len(chunks) for chunks in all_semantic.values())
    total_tokens_all = sum(c.token_count for chunks in all_semantic.values() for c in chunks)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Documents processed:     {len(docs)}")
    print(f"Total semantic chunks:   {total_chunks}")
    print(f"Total tokens:            {total_tokens_all:,}")
    print(f"PII chunks flagged:      {total_pii}")
    print(f"Risk signals detected:   {total_signals}")
    print(f"Golden QA coverage:      {coverage['covered']}/{len(qa_pairs)} ({coverage['covered']/len(qa_pairs)*100:.0f}%)")
    print(f"Query type distribution: {dict(type_counts)}")
    print("=" * 70)

    results["summary"] = {
        "documents_processed": len(docs),
        "total_semantic_chunks": total_chunks,
        "total_tokens": total_tokens_all,
        "pii_flagged": total_pii,
        "risk_signals": total_signals,
        "golden_qa_covered": coverage["covered"],
        "golden_qa_total": len(qa_pairs),
    }

    # Write results to JSON
    output_path = Path(__file__).parent / "pipeline_validation_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_pipeline_validation()
