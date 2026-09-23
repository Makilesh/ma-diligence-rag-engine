"""
Retrieval evaluation harness — golden QA set against the real retrieval path.

    python -m eval.run_retrieval_eval
    python -m eval.run_retrieval_eval --k 5 10 --ablations production production_decomp
    python -m eval.run_retrieval_eval --baseline eval/baseline.json
    python -m eval.run_retrieval_eval --update-baseline

Indexes data/sample_deal into an in-memory Qdrant with the production models,
then runs every golden question through five retrieval configurations:

    dense              bge-m3 dense list from hybrid_search
    sparse             BM25 sparse list from hybrid_search
    hybrid_rrf         weighted RRF of the two (the reranker's candidate pool)
    production         retrieval_executor_node: RRF -> cross-encoder -> threshold
    production_decomp  the same node with Agent 1's cached sub-questions

Makes ZERO LLM calls. Sub-questions come from eval/sub_questions.json, produced
once by eval/generate_sub_questions.py; without that file the decomposition
ablation is reported as not measured.

The retrieval config is chosen from the golden query_type with an empty
parsed_intent and no metadata filters, so both production ablations differ only
in the sub-questions. Agent 1's own classification and category filter — both
LLM guesses — are deliberately not part of what is measured here.

Writes eval/results/retrieval_<date>.json (per-question detail) and
eval/results/latest.md (summary, also printed). Exit status: 0 ok, 1 a gated
metric regressed against --baseline, 2 the run could not complete.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from eval import metrics as m
from eval.corpus import (
    DEAL_ID,
    GOLDEN_SET_PATH,
    PROJECT_ROOT,
    all_chunks,
    build_index,
    is_retrievable,
    load_golden_set,
    use_client,
)

EVAL_DIR = PROJECT_ROOT / "eval"
RESULTS_DIR = EVAL_DIR / "results"
DEFAULT_SUB_QUESTIONS = EVAL_DIR / "sub_questions.json"
DEFAULT_BASELINE = EVAL_DIR / "baseline.json"

ABLATIONS = ("dense", "sparse", "hybrid_rrf", "production", "production_decomp")
ABLATION_LABELS = {
    "dense": "dense-only",
    "sparse": "sparse-only (BM25)",
    "hybrid_rrf": "hybrid RRF (no rerank)",
    "production": "production (hybrid + rerank)",
    "production_decomp": "production + decomposition",
}
# Ablations that run the real node and therefore have a final context and a
# reranker score for the refusal gate.
NODE_ABLATIONS = ("production", "production_decomp")

# Rank depth for MRR and nDCG. Fixed, so ablations whose lists differ in length
# (40 dense candidates vs a thresholded rerank) are compared on equal terms.
RANK_DEPTH = 10

# The synthesizer sees each context chunk's text plus the first 500 characters
# of its parent (answer_synthesizer._format_context_for_synthesis). Fact
# coverage of the final context is measured over exactly that view.
PARENT_CONTEXT_CHARS = 500

DEFAULT_TOLERANCE = 0.02


# ==============================================================================
# Retrieval runs
# ==============================================================================


def _node_state(question: dict, sub_questions: list[str]) -> dict:
    """AgentState for Agent 3, as Agent 1 would leave it minus its LLM guesses."""
    return {
        "original_query": question["query"],
        "current_query": question["query"],
        "deal_id": DEAL_ID,
        "query_type": question["query_type"],
        "parsed_intent": {},
        "extracted_filters": {},
        "sub_questions": list(sub_questions),
        "include_pii": False,
        "rewrite_iteration": 0,
    }


def _gate_decision(state: dict, reranked: list[dict]) -> dict:
    """
    What the Quality Assessor's heuristic would do with this context.

    Calls _heuristic_assessment exactly as quality_assessor_node does before it
    ever considers the LLM. None from the heuristic means the ambiguous band,
    where the node would ask an LLM — reported as such, not guessed.
    """
    from src.agents.quality_assessor import _heuristic_assessment

    verdict = _heuristic_assessment({**state, "reranked_results": reranked})
    scores = [float(c.get("reranker_score", 0.0)) for c in reranked]
    if verdict is None:
        decision = "llm_fallback"
    elif verdict.get("force_refusal"):
        decision = "refused"
    else:
        decision = "admitted"
    return {
        "decision": decision,
        "max_reranker_score": round(max(scores), 4) if scores else None,
        "context_quality_score": None if verdict is None else verdict["context_quality_score"],
    }


async def _candidate_runs(question: dict, config: dict) -> dict:
    """
    Dense-only, sparse-only and hybrid-RRF rankings from one hybrid search.

    Uses the production functions in the production order (embed, BM25,
    hybrid_search, reciprocal_rank_fusion, fetch_chunks_by_ids). The dense and
    sparse lists come out of the same hybrid_search call, so they share its
    latency; RRF adds fusion and the payload fetch.
    """
    from src.vector_db.hybrid_search import compute_sparse_bm25, fetch_chunks_by_ids, hybrid_search
    from src.vector_db.reranker import embed_texts_async, get_embed_executor
    from src.vector_db.rrf_fusion import reciprocal_rank_fusion

    query = question["query"]
    start = time.perf_counter()
    query_vector = (await embed_texts_async([query]))[0].tolist()
    loop = asyncio.get_running_loop()
    query_sparse = await loop.run_in_executor(
        get_embed_executor(), lambda: compute_sparse_bm25(query)
    )
    dense_results, sparse_results = await hybrid_search(
        query_text=query,
        query_vector=query_vector,
        query_sparse=query_sparse,
        deal_id=DEAL_ID,
        metadata_filters={"include_pii": False},
        top_k_dense=config.get("top_k_dense", 40),
        top_k_sparse=config.get("top_k_sparse", 40),
    )
    search_ms = (time.perf_counter() - start) * 1000

    fuse_start = time.perf_counter()
    fused = reciprocal_rank_fusion(
        dense_results=dense_results,
        sparse_results=sparse_results,
        k=60,
        dense_weight=config.get("dense_weight", 0.6),
        sparse_weight=config.get("sparse_weight", 0.4),
    )
    fused_ids = [cid for cid, _ in fused[: config.get("reranker_top_k", 20)]]
    fused_chunks = await fetch_chunks_by_ids(fused_ids)
    fuse_ms = (time.perf_counter() - fuse_start) * 1000

    return {
        "dense": {"ranking": [p.payload for p in dense_results], "latency_ms": search_ms},
        "sparse": {"ranking": [p.payload for p in sparse_results], "latency_ms": search_ms},
        "hybrid_rrf": {"ranking": fused_chunks, "latency_ms": search_ms + fuse_ms},
    }


async def _node_run(question: dict, sub_questions: list[str]) -> dict:
    """One call of the real retrieval_executor_node, plus the refusal gate."""
    from src.agents.retrieval_executor import retrieval_executor_node

    state = _node_state(question, sub_questions)
    start = time.perf_counter()
    out = await retrieval_executor_node(state)
    latency_ms = (time.perf_counter() - start) * 1000
    return {
        "ranking": out["reranked_results"],
        "context": out["expanded_context"],
        "latency_ms": latency_ms,
        "gate": _gate_decision(state, out["reranked_results"]),
        "sub_questions": list(sub_questions),
    }


# ==============================================================================
# Scoring
# ==============================================================================


def _context_text(chunk: dict) -> str:
    """The text of one context chunk as the synthesizer sees it."""
    parent = (chunk.get("parent_text") or "")[:PARENT_CONTEXT_CHARS]
    return f"{chunk.get('text', '')}\n{parent}" if parent else chunk.get("text", "")


def score_run(run: dict, question: dict, qrels: dict[str, int], ks: list[int]) -> dict:
    """
    Metrics for one ranking (and final context, when the run has one).

    Args:
        run: {"ranking": [payload...], "context"?: [payload...]}.
        question: Golden entry.
        qrels: chunk_id -> grade for this question.
        ks: Cut-offs for coverage / recall / source hit.

    Returns:
        Flat metric dict plus the top-RANK_DEPTH ranking for inspection.
    """
    facts = question.get("expected_answer_contains") or []
    patterns = [c["source_pattern"] for c in question.get("expected_citations") or []]
    ranking = run["ranking"]
    ids = [c["chunk_id"] for c in ranking]
    texts = [c.get("text", "") for c in ranking]
    sources = [c.get("source_file") for c in ranking]

    scores: dict = {}
    for k in ks:
        coverage, found = m.fact_coverage(texts[:k], facts)
        scores[f"fact_coverage@{k}"] = coverage
        scores[f"recall@{k}"] = m.recall_at_k(ids, qrels, k)
        scores[f"source_hit@{k}"] = m.source_hit_at_k(sources, patterns, k)
    scores[f"mrr@{RANK_DEPTH}"] = m.mrr_at_k(ids, qrels, RANK_DEPTH)
    scores[f"ndcg@{RANK_DEPTH}"] = m.ndcg_at_k(ids, qrels, RANK_DEPTH)

    detail: dict = {"metrics": scores, "latency_ms": round(run["latency_ms"], 1)}
    if "context" in run:
        coverage, found = m.fact_coverage([_context_text(c) for c in run["context"]], facts)
        scores["fact_coverage@context"] = coverage
        detail["context_size"] = len(run["context"])
        detail["facts_missing_from_context"] = [
            m.fact_label(f) for i, f in enumerate(facts) if i not in found
        ]
        detail["gate"] = run["gate"]
        detail["sub_questions"] = run["sub_questions"]
    if "context" not in run:
        # Candidate-only ablations: the grade vector is enough to audit the
        # metrics and keeps the committed results file small.
        detail["top_grades"] = [qrels.get(cid, 0) for cid in ids[:RANK_DEPTH]]
        return _rounded(detail)
    detail["top"] = [
        {
            "chunk_id": c["chunk_id"],
            "source_file": c.get("source_file"),
            "section_heading": c.get("section_heading"),
            "grade": qrels.get(c["chunk_id"], 0),
            **({"reranker_score": round(float(c["reranker_score"]), 4)}
               if "reranker_score" in c else {}),
        }
        for c in ranking[:RANK_DEPTH]
    ]
    return _rounded(detail)


def _rounded(detail: dict) -> dict:
    """Rounds per-question metrics to 4 places (aggregates use the rounded values)."""
    detail["metrics"] = {k: None if v is None else round(v, 4)
                         for k, v in detail["metrics"].items()}
    return detail


def metric_names(ks: list[int], with_context: bool) -> list[str]:
    """Aggregate metric names in display order."""
    names = [f"fact_coverage@{k}" for k in ks]
    if with_context:
        names.append("fact_coverage@context")
    names += [f"recall@{k}" for k in ks]
    names += [f"mrr@{RANK_DEPTH}", f"ndcg@{RANK_DEPTH}"]
    names += [f"source_hit@{k}" for k in ks]
    return names


def aggregate(questions: list[dict], ablation: str, ks: list[int]) -> dict:
    """Overall and per-type means for one ablation over the answerable questions."""
    answerable = [q for q in questions if not q["is_control"] and ablation in q["runs"]]
    names = metric_names(ks, ablation in NODE_ABLATIONS)

    def _means(group: list[dict]) -> dict:
        out = {
            name: m.mean(q["runs"][ablation]["metrics"].get(name) for q in group)
            for name in names
        }
        out["n"] = len(group)
        if ablation == "production_decomp":
            out["n_decomposed"] = sum(
                1 for q in group if q["runs"][ablation].get("sub_questions"))
        return out

    by_type: dict[str, dict] = {}
    for qtype in sorted({q["query_type"] for q in answerable}):
        by_type[qtype] = _means([q for q in answerable if q["query_type"] == qtype])

    latencies = [q["runs"][ablation]["latency_ms"] for q in questions if ablation in q["runs"]]
    result = {
        "overall": _means(answerable),
        "by_type": by_type,
        "latency_ms": {
            "p50": m.percentile(latencies, 50),
            "p95": m.percentile(latencies, 95),
            "n": len(latencies),
        },
    }
    if ablation in NODE_ABLATIONS:
        result["gate"] = {
            group: {
                decision: sum(
                    1 for q in questions
                    if q["is_control"] == is_control and ablation in q["runs"]
                    and q["runs"][ablation]["gate"]["decision"] == decision
                )
                for decision in ("admitted", "llm_fallback", "refused")
            }
            for group, is_control in (("controls", True), ("answerable", False))
        }
    return result


def decomposition_delta(questions: list[dict], metric: str = "fact_coverage@context") -> dict:
    """Per-question improved / regressed / unchanged counts, production -> decomp."""
    improved, regressed, unchanged, decomposed = [], [], 0, 0
    for q in questions:
        runs = q["runs"]
        if q["is_control"] or "production" not in runs or "production_decomp" not in runs:
            continue
        before = runs["production"]["metrics"].get(metric)
        after = runs["production_decomp"]["metrics"].get(metric)
        if before is None or after is None:
            continue
        decomposed += bool(runs["production_decomp"].get("sub_questions"))
        if after > before + 1e-9:
            improved.append({"id": q["id"], "before": before, "after": after})
        elif after < before - 1e-9:
            regressed.append({"id": q["id"], "before": before, "after": after})
        else:
            unchanged += 1
    return {"metric": metric, "improved": improved, "regressed": regressed,
            "unchanged": unchanged, "decomposed": decomposed,
            "compared": len(improved) + len(regressed) + unchanged}


# ==============================================================================
# Sub-question cache
# ==============================================================================


def load_sub_questions(path: Path, golden: dict) -> tuple[dict[str, list[str]] | None, dict]:
    """
    Reads the Agent 1 sub-question cache, ignoring entries for edited questions.

    Returns:
        (id -> sub_questions, or None when the file is absent; info dict).
    """
    if not path.exists():
        return None, {"path": str(path), "present": False}
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("questions", {})
    queries = {q["id"]: q["query"] for q in golden["golden_qa_pairs"]}
    subs: dict[str, list[str]] = {}
    stale, missing = [], []
    for qid, query in queries.items():
        entry = entries.get(qid)
        if entry is None:
            missing.append(qid)
        elif entry.get("query") != query:
            stale.append(qid)
        else:
            subs[qid] = [s for s in entry.get("sub_questions") or [] if s.strip()]
    info = {
        "path": str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT)
        else str(path),
        "present": True,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "provenance": data.get("_provenance", {}),
        "decomposed_questions": sum(1 for s in subs.values() if s),
        "stale_ids": stale,
        "missing_ids": missing,
    }
    return subs, info


# ==============================================================================
# Reporting
# ==============================================================================


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}"


def _ms(value: float | None) -> str:
    return "—" if value is None else f"{value:.0f}"


def render_markdown(report: dict) -> str:
    """Concise summary: ablation x metric, per-type tables, gate, latency."""
    ks = report["config"]["k"]
    aggs = report["aggregates"]
    ablations = [a for a in ABLATIONS if a in aggs]
    lines = [
        f"# Retrieval eval — {report['generated_at'][:10]}",
        "",
        f"Golden set: {report['golden_set']['answerable']} answerable + "
        f"{report['golden_set']['controls']} control questions · corpus: "
        f"{len(report['index']['documents'])} documents, {report['index']['chunks']} chunks "
        f"({report['index']['retrievable_chunks']} retrievable) · device: "
        f"{report['environment']['device']} · LLM calls: 0 · fact-coverage ceiling "
        f"(facts present in any retrievable chunk): {_pct(report['fact_coverage_ceiling'])}%",
        "",
        "Percentages. fact_cov = share of expected facts present in the top-k chunk "
        "texts (any file); ctx = the final context the synthesizer receives "
        "(reranked + parent/sibling expansion). recall/MRR/nDCG use substring-derived "
        "chunk labels — see eval/README.md.",
        "",
    ]

    cols = metric_names(ks, with_context=True)
    header = "| ablation | " + " | ".join(c.replace("fact_coverage", "fact_cov") for c in cols)
    lines += [header + " | p50 ms | p95 ms |",
              "|---|" + "---|" * (len(cols) + 2)]
    for ablation in ablations:
        overall = aggs[ablation]["overall"]
        lat = aggs[ablation]["latency_ms"]
        cells = [_pct(overall.get(c)) for c in cols]
        lines.append(f"| {ABLATION_LABELS[ablation]} | " + " | ".join(cells)
                     + f" | {_ms(lat['p50'])} | {_ms(lat['p95'])} |")
    lines.append("")

    for ablation in (a for a in NODE_ABLATIONS if a in aggs):
        lines += [f"## {ABLATION_LABELS[ablation]} — by query type", ""]
        type_cols = [f"fact_coverage@{k}" for k in ks] + [
            "fact_coverage@context", f"recall@{ks[-1]}", f"mrr@{RANK_DEPTH}",
            f"ndcg@{RANK_DEPTH}"]
        lines += ["| type | n | " + " | ".join(
            c.replace("fact_coverage", "fact_cov") for c in type_cols) + " |",
            "|---|---|" + "---|" * len(type_cols)]
        rows = [*aggs[ablation]["by_type"].items(), ("**all**", aggs[ablation]["overall"])]
        for qtype, row in rows:
            n = row["n"] if "n_decomposed" not in row else f"{row['n']} ({row['n_decomposed']} dec.)"
            lines.append(f"| {qtype} | {n} | "
                         + " | ".join(_pct(row.get(c)) for c in type_cols) + " |")
        lines.append("")

    delta = report.get("decomposition_delta")
    if delta:
        lines += [
            f"Decomposition, per question on {delta['metric']}: "
            f"{len(delta['improved'])} improved, {len(delta['regressed'])} regressed, "
            f"{delta['unchanged']} unchanged "
            f"({delta['decomposed']} of the {delta['compared']} answerable questions "
            f"were decomposed by Agent 1).",
            "",
        ]
        if delta["improved"] or delta["regressed"]:
            lines.append("Changed: " + ", ".join(
                f"{d['id']} {_pct(d['before'])}→{_pct(d['after'])}"
                for d in delta["improved"] + delta["regressed"]
            ))
            lines.append("")
    elif not report["sub_questions"].get("present"):
        lines += ["Decomposition NOT measured: no sub-question cache "
                  f"({report['sub_questions']['path']}).", ""]

    lines += ["## Refusal gate (Quality Assessor heuristic, no LLM)", "",
              "| ablation | group | admitted | ambiguous → LLM | refused |",
              "|---|---|---|---|---|"]
    for ablation in (a for a in NODE_ABLATIONS if a in aggs):
        for group, counts in aggs[ablation]["gate"].items():
            lines.append(f"| {ABLATION_LABELS[ablation]} | {group} | {counts['admitted']} | "
                         f"{counts['llm_fallback']} | {counts['refused']} |")
    lines.append("")
    controls = [q for q in report["questions"] if q["is_control"] and "production" in q["runs"]]
    if controls:
        lines.append("Controls (production): " + ", ".join(
            f"{q['id']} max={q['runs']['production']['gate']['max_reranker_score']} "
            f"{q['runs']['production']['gate']['decision']}"
            for q in controls
        ))
        lines.append("")

    gate = report.get("baseline_check")
    if gate:
        status = "PASS" if not gate["failures"] else "FAIL"
        lines += [f"Baseline check ({gate['baseline']}, tolerance "
                  f"{gate['tolerance'] * 100:.1f}pp): **{status}**", ""]
        lines += [f"- {f}" for f in gate["failures"]]
        if gate["failures"]:
            lines.append("")
    return "\n".join(lines)


def baseline_metrics(report: dict) -> dict:
    """The gated aggregates: overall metrics of the two node ablations."""
    ks = report["config"]["k"]
    gated = [f"fact_coverage@{k}" for k in ks] + ["fact_coverage@context"]
    gated += [f"recall@{ks[-1]}", f"mrr@{RANK_DEPTH}", f"ndcg@{RANK_DEPTH}"]
    out = {}
    for ablation in NODE_ABLATIONS:
        overall = report["aggregates"].get(ablation, {}).get("overall")
        if overall:
            out[ablation] = {name: overall.get(name) for name in gated}
    return out


# ==============================================================================
# Main
# ==============================================================================


def _device() -> str:
    import torch

    return f"cuda ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "cpu"


async def run(args: argparse.Namespace) -> dict:
    """Builds the index, runs every ablation, returns the full report."""
    from src.agents.retrieval_strategy import get_retrieval_config
    from src.vector_db.reranker import (
        EMBEDDING_MODEL_NAME,
        RERANKER_MODEL_NAME,
        warm_models,
    )

    golden = load_golden_set()
    pairs = golden["golden_qa_pairs"]
    if args.questions:
        pairs = [q for q in pairs if q["id"] in set(args.questions)]
    ablations = list(args.ablations)

    subs, sub_info = load_sub_questions(Path(args.sub_questions), golden)
    if subs is None and "production_decomp" in ablations:
        print(f"[eval] no sub-question cache at {args.sub_questions}: "
              "decomposition will not be measured", file=sys.stderr)
        ablations.remove("production_decomp")
    if sub_info.get("stale_ids") or sub_info.get("missing_ids"):
        print(f"[eval] sub-question cache: stale {sub_info.get('stale_ids')}, "
              f"missing {sub_info.get('missing_ids')} (treated as undecomposed)",
              file=sys.stderr)

    started = time.perf_counter()
    print("[eval] indexing corpus into in-memory Qdrant ...", file=sys.stderr)
    client, documents = await build_index()
    index_s = time.perf_counter() - started
    chunks = await all_chunks(client)
    retrievable = [c for c in chunks if is_retrievable(c)]

    report_questions = []
    with use_client(client):
        await warm_models()
        for n, question in enumerate(pairs, start=1):
            is_control = bool(question.get("expect_refusal"))
            facts = question.get("expected_answer_contains") or []
            patterns = [c["source_pattern"] for c in question.get("expected_citations") or []]
            qrels = m.build_qrels(retrievable, facts, patterns)
            unreachable = len(m.build_qrels(
                [c for c in chunks if not is_retrievable(c)], facts, patterns))
            config = get_retrieval_config(question["query_type"], {})

            runs: dict[str, dict] = {}
            if {"dense", "sparse", "hybrid_rrf"} & set(ablations):
                candidate_runs = await _candidate_runs(question, config)
                runs.update({a: r for a, r in candidate_runs.items() if a in ablations})
            if "production" in ablations or "production_decomp" in ablations:
                production = await _node_run(question, [])
                if "production" in ablations:
                    runs["production"] = production
                if "production_decomp" in ablations:
                    question_subs = (subs or {}).get(question["id"]) or []
                    # An undecomposed question takes the identical path; re-running
                    # it would only re-measure latency noise.
                    runs["production_decomp"] = (
                        await _node_run(question, question_subs) if question_subs
                        else production
                    )

            report_questions.append({
                "id": question["id"],
                "query": question["query"],
                "query_type": question["query_type"],
                "is_control": is_control,
                "n_facts": len(facts),
                "n_relevant_chunks": len(qrels),
                "n_relevant_chunks_unreachable": unreachable,
                # Best fact coverage any retrieval could reach: facts that occur
                # in no retrievable chunk (e.g. a computed 7.5x) cap it below 1.
                "fact_coverage_ceiling": m.fact_coverage(
                    [c.get("text", "") for c in retrievable], facts)[0],
                "runs": {a: score_run(r, question, qrels, args.k) for a, r in runs.items()},
            })
            print(f"[eval] {n}/{len(pairs)} {question['id']}", file=sys.stderr)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": "python -m eval.run_retrieval_eval " + " ".join(sys.argv[1:]),
        "golden_set": {
            "path": str(GOLDEN_SET_PATH.relative_to(PROJECT_ROOT)),
            "sha256": hashlib.sha256(GOLDEN_SET_PATH.read_bytes()).hexdigest(),
            "answerable": sum(1 for q in pairs if not q.get("expect_refusal")),
            "controls": sum(1 for q in pairs if q.get("expect_refusal")),
        },
        "environment": {
            "device": _device(),
            "embedding_model": EMBEDDING_MODEL_NAME,
            "reranker_model": RERANKER_MODEL_NAME,
            "sparse_model": "Qdrant/bm25 (FastEmbed)",
            "vector_store": "qdrant-client local mode (:memory:) — exact search, "
                            "no HNSW/int8 quantization",
            "python": sys.version.split()[0],
        },
        "config": {
            "k": args.k,
            "rank_depth": RANK_DEPTH,
            "ablations": ablations,
            "deal_id": DEAL_ID,
            "retrieval_config_source": "RETRIEVAL_CONFIGS[golden query_type], "
                                       "parsed_intent={}, no metadata filters",
        },
        "index": {
            "documents": [
                {k: d[k] for k in ("filename", "document_category", "chunks_created",
                                   "parent_chunks_created", "table_count")}
                for d in documents
            ],
            "chunks": len(chunks),
            "retrievable_chunks": len(retrievable),
            "index_seconds": round(index_s, 1),
        },
        "sub_questions": sub_info,
        "questions": report_questions,
    }
    report["aggregates"] = {a: aggregate(report_questions, a, args.k) for a in ablations}
    report["fact_coverage_ceiling"] = m.mean(
        q["fact_coverage_ceiling"] for q in report_questions if not q["is_control"])
    if "production" in ablations and "production_decomp" in ablations:
        report["decomposition_delta"] = decomposition_delta(report_questions)
    report["total_seconds"] = round(time.perf_counter() - started, 1)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m eval.run_retrieval_eval",
        description="Retrieval evaluation over tests/golden_qa_set.json (no LLM calls).",
    )
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10],
                        help="cut-offs for fact coverage / recall / source hit (default 5 10)")
    parser.add_argument("--ablations", nargs="+", choices=ABLATIONS, default=list(ABLATIONS))
    parser.add_argument("--questions", nargs="+", metavar="ID",
                        help="only these golden ids (debugging; not for baselines)")
    parser.add_argument("--sub-questions", default=str(DEFAULT_SUB_QUESTIONS),
                        help="Agent 1 sub-question cache (default eval/sub_questions.json)")
    parser.add_argument("--baseline", help="fail (exit 1) if a gated metric regressed vs this file")
    parser.add_argument("--tolerance", type=float,
                        help="allowed absolute drop, e.g. 0.02 = 2pp "
                             "(default: the baseline file's, else 0.02)")
    parser.add_argument("--update-baseline", action="store_true",
                        help=f"write this run's gated metrics to {DEFAULT_BASELINE.name}")
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    parser.add_argument("--verbose", action="store_true", help="keep INFO logs from src.*")
    args = parser.parse_args(argv)
    args.k = sorted(set(args.k))
    if args.update_baseline and args.questions:
        parser.error("--update-baseline needs the full golden set (drop --questions)")
    return args


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    if not args.verbose:
        logging.disable(logging.INFO)

    try:
        report = asyncio.run(run(args))
    except Exception as e:  # any failure to run is exit 2, not a regression
        logging.disable(logging.NOTSET)
        logging.getLogger(__name__).exception("Retrieval eval failed to run")
        print(f"[eval] FAILED to run: {e}", file=sys.stderr)
        return 2

    exit_code = 0
    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        tolerance = (args.tolerance if args.tolerance is not None
                     else baseline.get("tolerance", DEFAULT_TOLERANCE))
        failures = m.compare_to_baseline(baseline_metrics(report), baseline["metrics"], tolerance)
        report["baseline_check"] = {"baseline": args.baseline, "tolerance": tolerance,
                                    "failures": failures}
        exit_code = 1 if failures else 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = report["generated_at"][:10]
    json_path = output_dir / f"retrieval_{stamp}.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown = render_markdown(report)
    (output_dir / "latest.md").write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    print(f"\n[eval] wrote {json_path} and {output_dir / 'latest.md'} "
          f"({report['total_seconds']}s)", file=sys.stderr)

    if args.update_baseline:
        baseline = {
            "_about": "Gated retrieval metrics for `python -m eval.run_retrieval_eval "
                      "--baseline eval/baseline.json`. Fractions in [0, 1]; a drop larger "
                      "than `tolerance` on any of them fails the run. Regenerate with "
                      "--update-baseline after an intentional retrieval change.",
            "generated_at": report["generated_at"],
            "device": report["environment"]["device"],
            "golden_set_sha256": report["golden_set"]["sha256"],
            "sub_questions_sha256": report["sub_questions"].get("sha256"),
            "tolerance": args.tolerance if args.tolerance is not None else DEFAULT_TOLERANCE,
            "metrics": {
                ablation: {k: (None if v is None else round(v, 4)) for k, v in values.items()}
                for ablation, values in baseline_metrics(report).items()
            },
        }
        DEFAULT_BASELINE.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
        print(f"[eval] baseline written to {DEFAULT_BASELINE}", file=sys.stderr)

    if exit_code:
        print("[eval] REGRESSION against baseline:", file=sys.stderr)
        for failure in report["baseline_check"]["failures"]:
            print(f"  - {failure}", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
