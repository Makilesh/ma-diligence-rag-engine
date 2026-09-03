import asyncio
import json
import re
import os
import sys
import time
from pathlib import Path
import httpx

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

API_URL = "http://127.0.0.1:8000"
DEAL_ID = "aurora_vertex_2024"

# Matches the engine's refusal templates (insufficient_context_node,
# answer_synthesizer forced refusal, and the synthesis-failure degrade path).
FULL_REFUSAL_PATTERN = re.compile(
    r"not have sufficient|unable to find sufficient|insufficient relevant"
    r"|did not return a usable",
    re.IGNORECASE,
)

# Matches an explicit acknowledgement, inside an otherwise normal answer, that
# the requested fact is not in the corpus.
#
# This distinction matters and a binary refused/answered flag gets it wrong. On
# the control questions the engine did not refuse wholesale — it answered the
# answerable part and named the gap: "The provided documents do not contain the
# revenue figures for Q1 FY2024", and for the churn question it reported the NRR
# and competitor data that *are* present while stating that churn is not. That
# partial answer is better due-diligence behaviour than a blanket refusal, but a
# naive metric scores it as a hallucination. What actually matters is whether the
# engine invented the missing figure, so that is what gets measured.
# LIMITATION, stated plainly: this enumerates phrasings, so it will keep missing
# constructions nobody thought of. Every miss so far has been in the same
# direction — a correct decline scored as a fabrication — which understates the
# engine rather than flattering it, so the bias is at least the safe one. Two
# were caught by reading answers the metric had failed: "contains **no
# information** regarding" (markdown broke the match) and "contains **no
# evidence, disclosure, or mention** of" (the noun was not in the list).
# Anything measuring generated prose with a regex has this property; the guard is
# to read the answers behind a control failure rather than trust the count.
ABSENCE_ACKNOWLEDGEMENT_PATTERN = re.compile(
    r"do(?:es)? not contain|not contain(?:ed)?"
    r"|no (?:information|evidence|mention|mentions|disclosure|disclosures"
    r"|reference|references|record|records|data|indication|details)"
    r"|not (?:disclosed|provided|available|specified|present|included|addressed|mentioned)"
    r"|not found in the (?:provided )?document"
    r"|do(?:es)? not (?:disclose|mention|address|specify|provide)"
    r"|cannot be determined|entirely absent|are absent|is absent",
    re.IGNORECASE,
)

# Markdown emphasis has to come out before matching. The model bolds the
# operative phrase, so "contains **no information** regarding" is not a
# contiguous substring anything a plain pattern can match — and a control
# question the engine had correctly declined got scored as a fabrication purely
# because of the asterisks. That is the third time a metric here measured
# formatting rather than behaviour, so normalisation now lives in one place.
_MARKDOWN_EMPHASIS = re.compile(r"[*_`]+")


def normalise_for_matching(text: str) -> str:
    """Strips markdown emphasis and collapses whitespace for substring matching."""
    return " ".join(_MARKDOWN_EMPHASIS.sub("", text or "").split())


def fact_present(fact, answer: str) -> bool:
    """
    True if an expected fact appears in the answer.

    A fact may be given as a list of equivalent surface forms, any one of which
    counts — e.g. ["thirty-six months", "36 months"]. Without this, the golden
    set measures phrasing rather than grounding: an answer saying "thirty-six
    months" was scored as missing "36 months" and lost a sixth of its recall for
    spelling a number in words. Same for a survival period written "3 years".

    Whitespace is normalised so a fact that happens to wrap across a line in the
    model's markdown still matches.

    Args:
        fact: An expected string, or a list of acceptable variants.
        answer: The generated answer.

    Returns:
        True if the fact (in any accepted form) is present.
    """
    haystack = normalise_for_matching(answer).lower()
    variants = [fact] if isinstance(fact, str) else list(fact)
    return any(normalise_for_matching(str(v)).lower() in haystack for v in variants)


def declines_to_fabricate(answer: str) -> bool:
    """True if the answer either refuses outright or names the missing data."""
    clean = normalise_for_matching(answer)
    return bool(
        FULL_REFUSAL_PATTERN.search(clean)
        or ABSENCE_ACKNOWLEDGEMENT_PATTERN.search(clean)
    )


# Retained for the answerable set, where a full refusal is the meaningful signal.
REFUSAL_PATTERN = FULL_REFUSAL_PATTERN

async def check_api_health(client: httpx.AsyncClient) -> bool:
    try:
        response = await client.get(f"{API_URL}/health", timeout=2.0)
        return response.status_code == 200 and response.json().get("status") == "healthy"
    except Exception:
        return False

async def ingest_files(client: httpx.AsyncClient, files_info: list[dict]) -> dict[str, str]:
    doc_mappings = {}
    for file_info in files_info:
        file_path = file_info["path"]
        category = file_info["category"]
        filename = file_path.name
        
        print(f"Ingesting {filename} (Category: {category or 'Auto-detect'})...")
        
        with open(file_path, "rb") as f:
            files = {"file": (filename, f, "text/plain")}
            data = {
                "deal_id": DEAL_ID,
                "is_current_version": "true",
            }
            if category:
                data["document_category"] = category
                
            response = await client.post(
                f"{API_URL}/api/v1/ingest",
                files=files,
                data=data,
                timeout=300.0
            )
            
            if response.status_code != 200:
                print(f"Failed to ingest {filename}: {response.text}")
                continue
                
            res_json = response.json()
            doc_id = res_json.get("doc_id")
            doc_mappings[filename] = doc_id
            print(f"   Success! Doc ID: {doc_id}, Chunks created: {res_json.get('chunks_created')}")
            
    return doc_mappings

async def run_queries(client: httpx.AsyncClient, golden_qa_pairs: list[dict]) -> list[dict]:
    results = []
    
    for idx, qa in enumerate(golden_qa_pairs):
        qa_id = qa["id"]
        query = qa["query"]
        query_type = qa["query_type"]
        expected_contains = qa["expected_answer_contains"]
        expected_citations = qa["expected_citations"]
        
        print(f"\n[{idx+1}/{len(golden_qa_pairs)}] Running {qa_id} ({query_type})...")
        print(f"   Query: {query}")
        
        start_time = time.monotonic()
        try:
            response = await client.post(
                f"{API_URL}/api/v1/query",
                json={
                    "query": query,
                    "deal_id": DEAL_ID
                },
                timeout=300.0
            )
            elapsed = (time.monotonic() - start_time) * 1000
            
            if response.status_code != 200:
                print(f"   Error: {response.text}")
                results.append({
                    "id": qa_id,
                    "query": query,
                    "query_type": query_type,
                    "status": "failed",
                    "error": response.text,
                    "latency_ms": elapsed
                })
                continue
                
            res_json = response.json()
            answer = res_json.get("answer", "")
            citations = res_json.get("citations", [])
            confidence = res_json.get("confidence_score", 0.0)
            validation_status = res_json.get("validation_status", "passed")
            agent_trace = res_json.get("agent_trace", [])
            hallucination_flags = res_json.get("hallucination_flags", [])
            
            # Refusal detection. Control questions (expect_refusal) are
            # unanswerable by construction: for those, refusing IS the correct
            # answer, and producing content is a hallucination.
            refused = bool(FULL_REFUSAL_PATTERN.search(answer))
            expect_refusal = qa.get("expect_refusal", False)
            if expect_refusal:
                # Correct = did not fabricate the missing fact, whether by
                # refusing outright or by naming the gap inside the answer.
                refusal_correct = declines_to_fabricate(answer)
            else:
                refusal_correct = not refused

            # Evaluate facts recalled
            recalled_facts = []
            missing_facts = []
            for fact in expected_contains:
                if fact_present(fact, answer):
                    recalled_facts.append(fact if isinstance(fact, str) else fact[0])
                else:
                    missing_facts.append(fact if isinstance(fact, str) else fact[0])

            if expect_refusal:
                # No facts to recall; score the behaviour instead.
                recall_score = 1.0 if refusal_correct else 0.0
            else:
                recall_score = len(recalled_facts) / len(expected_contains) if expected_contains else 0.0
            
            # Evaluate citations
            citation_match = False
            for cite in citations:
                for exp_cite in expected_citations:
                    pat = exp_cite["source_pattern"].lower()
                    if pat in cite.get("source_file", "").lower():
                        citation_match = True
                        break
            
            print(f"   Answer: {answer[:120]}...")
            print(f"   Confidence: {confidence:.2f}, Recall: {recall_score*100:.1f}%, Citations Checked: {citation_match}")
            
            results.append({
                "id": qa_id,
                "query": query,
                "query_type": query_type,
                "status": "passed",
                "answer": answer,
                "confidence_score": confidence,
                "validation_status": validation_status,
                "hallucination_flags": hallucination_flags,
                "citations": citations,
                "latency_ms": elapsed,
                "recall_score": recall_score,
                "recalled_facts": recalled_facts,
                "missing_facts": missing_facts,
                "citation_match": citation_match,
                "refused": refused,
                "expect_refusal": expect_refusal,
                "refusal_correct": refusal_correct,
                "agent_trace": agent_trace
            })
            
        except Exception as e:
            elapsed = (time.monotonic() - start_time) * 1000
            print(f"   Exceptions: {str(e)}")
            results.append({
                "id": qa_id,
                "query": query,
                "query_type": query_type,
                "status": "exception",
                "error": str(e),
                "latency_ms": elapsed
            })
            
    return results

async def main():
    print("M&A DUE DILIGENCE ENGINE -- END-TO-END VALIDATION RUN")
    print("=====================================================")
    
    async with httpx.AsyncClient() as client:
        # Step 1: Wait for API to be healthy
        print("Checking API connection...")
        healthy = False
        for i in range(10):
            if await check_api_health(client):
                healthy = True
                break
            print(f"   Waiting for API to be ready (attempt {i+1}/10)...")
            await asyncio.sleep(2.0)
            
        if not healthy:
            print("API is not reachable. Ensure the server is running.")
            return
            
        print("API is healthy! Proceeding with document ingestion.")
        
        # Step 2: Define files to ingest
        # Ingest every document in the data room rather than a hardcoded list.
        # A fixed list silently desynchronises from the corpus: documents added
        # for new golden questions were never indexed, so those questions were
        # scored as retrieval failures when the answer had simply never been
        # loaded. Globbing keeps the two in step by construction.
        deal_dir = Path(__file__).parent.parent / "data" / "sample_deal"

        # Explicit categories where the filename heuristic would mislabel.
        # Anything unlisted is auto-detected by the ingestion classifier.
        CATEGORY_OVERRIDES = {
            "aurora_financials_fy2023.txt": "financial",
            "quality_of_earnings_report_fy2023.txt": "financial",
            "credit_agreement_summary.txt": "financial",
            "merger_agreement_v2_final.txt": "legal",
            "customer_contracts_schedule.txt": "legal",
            "employment_and_retention_agreements.txt": "legal",
            "ip_portfolio_and_litigation_schedule.txt": "legal",
            "regulatory_and_data_privacy_memo.txt": "regulatory",
            "board_deck_strategic_review_mar2024.txt": "board",
        }

        files_info = [
            {"path": p, "category": CATEGORY_OVERRIDES.get(p.name)}
            for p in sorted(deal_dir.glob("*.txt"))
        ]
        print(f"Corpus: {len(files_info)} documents in {deal_dir}")

        doc_mappings = await ingest_files(client, files_info)
        print(f"\nIngestion completed. {len(doc_mappings)} documents indexed.")

        # Abort rather than measure nothing. A Qdrant client/server version skew
        # made every upsert fail while the API stayed healthy; the harness
        # printed "0 documents indexed" and then ran all 41 questions against an
        # empty index, producing a full results file of 0% recall that looked
        # like a catastrophic regression in the engine. An evaluation whose
        # corpus failed to load has no result to report — including a bad one.
        if len(doc_mappings) < len(files_info):
            print(
                f"\nABORTING: only {len(doc_mappings)}/{len(files_info)} documents "
                f"ingested. Every query would be scored against an incomplete "
                f"index, so the run would measure the ingestion failure rather "
                f"than the engine. Fix ingestion and re-run."
            )
            return

        # Step 3: Load golden QA set
        qa_path = Path(__file__).parent / "golden_qa_set.json"
        with open(qa_path, "r", encoding="utf-8") as f:
            qa_data = json.load(f)
            
        golden_qa_pairs = qa_data["golden_qa_pairs"]
        print(f"Loaded {len(golden_qa_pairs)} golden Q&A pairs. Starting query evaluations...")
        
        # Step 4: Run queries
        results = await run_queries(client, golden_qa_pairs)
        
        # Step 5: Save results
        output_path = Path(__file__).parent / "e2e_validation_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")
        
        # Step 6: Generate RESULTS.md
        write_results_md(results)


def write_results_md(results: list[dict]) -> None:
    """
    Renders RESULTS.md from a completed results list.

    Separated from the run loop so the report can be regenerated from a saved
    e2e_validation_results.json — re-running the live eval costs ~30 minutes and
    a chunk of daily quota, which is far too expensive to pay for a formatting
    change to the report.
    """
    if True:
        results_md_path = Path(__file__).parent.parent / "RESULTS.md"
        
        total_queries = len(results)
        passed = sum(1 for r in results if r.get("status") == "passed")
        avg_latency = sum(r.get("latency_ms", 0.0) for r in results) / total_queries if total_queries else 0.0

        # Controls are scored on behaviour (did it avoid fabricating?), answerable
        # questions on fact recall. Averaging the two together inflates recall,
        # because every correctly-declined control contributes a 1.0 that has
        # nothing to do with retrieval quality. Reported separately.
        answerable = [r for r in results
                      if r.get("status") == "passed" and not r.get("expect_refusal")]
        controls = [r for r in results
                    if r.get("status") == "passed" and r.get("expect_refusal")]

        avg_recall = (sum(r.get("recall_score", 0.0) for r in answerable) / len(answerable)
                      if answerable else 0.0)
        citation_matches = sum(1 for r in answerable if r.get("citation_match") is True)
        answered = sum(1 for r in answerable if not r.get("refused"))
        perfect = sum(1 for r in answerable if r.get("recall_score") == 1.0)
        controls_held = sum(1 for r in controls if r.get("refusal_correct"))

        # Run conditions. Recorded because this metric has twice been misread
        # without them: once when recall swung 15 points between identical runs
        # purely on which synthesis model quota allowed (DECISIONS_LOG 25), and
        # once when a provider incident produced truncated, uncited answers that
        # looked like a retrieval regression. The mix and the failure count are
        # the two facts needed to tell "the system got worse" from "the provider
        # was having a bad day", so they now ship with every report.
        synth_models: dict[str, int] = {}
        for r in results:
            for t in (r.get("agent_trace") or []):
                if t.get("agent") == "answer_synthesizer" and t.get("model"):
                    synth_models[t["model"]] = synth_models.get(t["model"], 0) + 1
        mix = ", ".join(
            f"`{m}` ×{n}" for m, n in sorted(synth_models.items(), key=lambda kv: -kv[1])
        ) or "not recorded"

        synthesis_failures = sum(
            1 for r in results
            if "did not return a usable response" in (r.get("answer") or "")
        )

        # Retrieval profile. Recorded for the same reason as the synthesis mix:
        # it is a run condition, not a property of the engine, and two runs are
        # not comparable without it. The deployed profile substitutes a much
        # smaller cross-encoder to fit a CPU-only host, so a report that does not
        # name its reranker cannot be told apart from one that used the default —
        # which is precisely the confusion that would let a deployed number be
        # read as the local one.
        from src.vector_db.reranker import (
            EMBEDDING_MODEL_NAME,
            RERANKER_MAX_LENGTH,
            RERANKER_MODEL_NAME,
        )

        is_default_profile = (
            EMBEDDING_MODEL_NAME == "BAAI/bge-m3"
            and RERANKER_MODEL_NAME == "BAAI/bge-reranker-v2-m3"
            and RERANKER_MAX_LENGTH == 1024
        )
        profile_label = "local (default)" if is_default_profile else "deployed (CPU)"
        decomposed = sum(
            1 for r in results
            if any((t.get("sub_questions") or [])
                   for t in (r.get("agent_trace") or [])
                   if t.get("agent") == "retrieval_executor")
        )

        md_content = f"""# E2E Validation Results

Generated by `tests/run_end_to_end_validation.py` against the golden Q&A set in
`tests/golden_qa_set.json`, on a freshly wiped index. Every number here is
computed from `tests/e2e_validation_results.json` — nothing is hand-entered.

The set is **{len(answerable) + len(controls)} questions**: {len(answerable)} answerable, plus {len(controls)} unanswerable
**control** questions whose answers are absent from the corpus by construction.
Controls are scored on whether the engine avoided inventing the missing figure,
not on fact recall — so the two are reported separately. Without them, "never
refuses" and "always finds the answer" would be indistinguishable.

## Run Summary
- **Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S')}
- **Deal ID**: `{DEAL_ID}`
- **Completed without an unhandled exception**: {passed}/{total_queries}
- **Average E2E latency**: {avg_latency/1000:.1f}s

### Answerable questions ({len(answerable)})
- **Answered** (not refused): {answered}/{len(answerable)}
- **Mean fact recall**: {avg_recall*100:.1f}%
- **Answers with every expected fact present**: {perfect}/{len(answerable)}
- **Citation-source match**: {citation_matches}/{len(answerable)}

### Control questions ({len(controls)})
- **Declined to fabricate the missing figure**: {controls_held}/{len(controls)}

### Run conditions

Read these before comparing this run to another one. Recall on this set moves
with the synthesis model, and the synthesis model moves with daily quota and
provider health — neither of which is a property of the engine.

- **Retrieval profile**: {profile_label}
- **Embedding model**: `{EMBEDDING_MODEL_NAME}`
- **Reranker**: `{RERANKER_MODEL_NAME}` (max_length {RERANKER_MAX_LENGTH})
- **Synthesis model mix**: {mix}
- **Queries decomposed into sub-questions**: {decomposed}/{total_queries}
- **Answers lost to upstream synthesis failure**: {synthesis_failures}/{total_queries}

A non-zero synthesis-failure count means the provider refused or truncated
generations during the run. Those questions score 0% recall regardless of how
good retrieval was, so the headline figure understates the engine by roughly
`failures / answerable` and the comparison to a clean run is not like-for-like.

## Metrics by Query Type

Answerable questions only; controls are excluded so per-type recall reflects
retrieval quality rather than refusal behaviour.

| Query Type | Count | Answered | Avg Recall | Avg Latency (ms) |
| --- | --- | --- | --- | --- |
"""

        types = ["financial", "legal", "comparative", "summary", "multi_hop"]
        for t in types:
            t_res = [r for r in answerable if r["query_type"] == t]
            if not t_res:
                continue
            t_passed = [r for r in t_res if not r.get("refused")]
            t_avg_recall = sum(r.get("recall_score", 0.0) for r in t_res) / len(t_res) if t_res else 0.0
            t_avg_latency = sum(r.get("latency_ms", 0.0) for r in t_res) / len(t_res)
            md_content += f"| {t.capitalize()} | {len(t_res)} | {len(t_passed)}/{len(t_res)} | {t_avg_recall*100:.1f}% | {t_avg_latency:.2f} |\n"
            
        md_content += """
## Detailed Query Output Reports

"""
        for r in results:
            md_content += f"### {r['id']} ({r['query_type'].capitalize()})\n"
            md_content += f"**Query**: {r['query']}\n\n"
            if r.get("status") == "passed":
                md_content += f"- **Status**: ✅ PASS\n"
                md_content += f"- **Confidence Score**: {r['confidence_score']:.2f}\n"
                md_content += f"- **Validation Status**: {r['validation_status']}\n"
                md_content += f"- **Facts Recalled**: {len(r['recalled_facts'])}/{len(r['recalled_facts']) + len(r['missing_facts'])} ({r['recall_score']*100:.1f}%)\n"
                md_content += f"  - *Recalled*: {r['recalled_facts']}\n"
                if r['missing_facts']:
                    md_content += f"  - *Missing*: {r['missing_facts']}\n"
                md_content += f"- **Citations Match**: {'✅ Yes' if r['citation_match'] else '❌ No'}\n"
                md_content += f"- **Total Latency**: {r['latency_ms']:.2f} ms\n"
                md_content += f"- **Answer**:\n```\n{r['answer']}\n```\n"
                
                # Show key agent trace steps
                if r.get("agent_trace"):
                    md_content += "- **Agent Trace Summary**:\n"
                    for step in r["agent_trace"][:5]:
                        agent = step.get("agent", "System")
                        action = step.get("action", "")
                        # Truncate detail for readability
                        detail = step.get("detail", "")
                        if len(detail) > 100:
                            detail = detail[:100] + "..."
                        md_content += f"  - **{agent}**: {action} ({detail})\n"
            else:
                md_content += f"- **Status**: ❌ FAIL\n"
                md_content += f"- **Error**: `{r.get('error')}`\n"
                md_content += f"- **Latency**: {r['latency_ms']:.2f} ms\n"
            md_content += "\n---\n\n"
            
        with open(results_md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"RESULTS.md generated at {results_md_path}")

if __name__ == "__main__":
    asyncio.run(main())
