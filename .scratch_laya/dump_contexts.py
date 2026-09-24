"""Scratch: dump reranked contexts for golden (prod + decomp) and dev questions."""
import asyncio, json, logging, sys
from pathlib import Path
logging.disable(logging.INFO)
from eval.corpus import build_index, use_client, load_golden_set
from eval.run_retrieval_eval import _node_state, load_sub_questions, DEFAULT_SUB_QUESTIONS

async def main():
    from src.agents.retrieval_executor import retrieval_executor_node
    from src.agents.quality_assessor import _heuristic_assessment
    from src.vector_db.reranker import warm_models
    golden = load_golden_set()
    subs, _ = load_sub_questions(DEFAULT_SUB_QUESTIONS, golden)
    dev = json.loads(Path("eval/answerability_dev.json").read_text(encoding="utf-8"))["questions"]
    client, _ = await build_index()
    out = []
    with use_client(client):
        await warm_models()
        items = []
        for q in golden["golden_qa_pairs"]:
            items.append(("golden", q["id"], q["query"], q["query_type"], not q.get("expect_refusal"), []))
            if subs.get(q["id"]):
                items.append(("golden_decomp", q["id"], q["query"], q["query_type"], not q.get("expect_refusal"), subs[q["id"]]))
        for q in dev:
            items.append(("dev", q["id"], q["query"], q["query_type"], q["answerable"], []))
        for set_, qid, query, qtype, ans, sq in items:
            state = _node_state({"query": query, "query_type": qtype}, sq)
            r = await retrieval_executor_node(state)
            h = _heuristic_assessment({**state, "reranked_results": r["reranked_results"]})
            dec = "llm_fallback" if h is None else ("refused" if h["force_refusal"] else "admitted")
            out.append({"set": set_, "id": qid, "query": query, "query_type": qtype, "answerable": ans,
                        "sub_questions": sq, "heuristic": dec,
                        "reranked": [{k: c.get(k) for k in ("chunk_id", "text", "reranker_score", "source_file", "section_heading", "parent_id", "parent_chunk_id")} for c in r["reranked_results"]],
                        "expanded": [{k: c.get(k) for k in ("chunk_id", "text", "source_file", "is_parent", "chunk_type")} for c in r["expanded_context"]]})
            print(set_, qid, dec, len(r["reranked_results"]), file=sys.stderr)
    Path(".scratch_laya/contexts.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

asyncio.run(main())
