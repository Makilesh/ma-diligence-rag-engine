"""
Generates eval/sub_questions.json by running the real Agent 1 once per question.

    python -m eval.generate_sub_questions            # fill in missing ids only
    python -m eval.generate_sub_questions --force    # regenerate everything

This is the ONLY part of the eval that calls an LLM. It runs
query_intelligence_node — the production Agent 1, prompt and all — on each
golden question, one call each, on the agent ladder (lite models only; the
ladder excludes the 20-RPD reasoning models by design). The retrieval harness
then reads the cache and makes no LLM calls, so its results are reproducible.

Regenerate only when Agent 1's prompt or the golden set changes. Progress is
saved after every question, so a quota failure part-way keeps what was done and
a re-run resumes.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from eval.corpus import DEAL_ID, GOLDEN_SET_PATH, PROJECT_ROOT, load_golden_set

DEFAULT_OUTPUT = PROJECT_ROOT / "eval" / "sub_questions.json"

# The lite agent models allow 10-15 RPM; one call every 5s stays under both.
DEFAULT_DELAY_S = 5.0


def _save(path: Path, provenance: dict, questions: dict) -> None:
    ordered = dict(sorted(questions.items()))
    models = sorted({q["model"] for q in ordered.values() if q.get("model")})
    body = {"_provenance": {**provenance, "models": models}, "questions": ordered}
    path.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


async def generate(args: argparse.Namespace) -> int:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    if not (os.getenv("GEMINI_API_KEYS") or os.getenv("GEMINI_API_KEY")):
        print("No GEMINI_API_KEYS / GEMINI_API_KEY configured — refusing to run: "
              "Agent 1 would silently fall back to the local model.", file=sys.stderr)
        return 2

    from src.agents.query_intelligence import query_intelligence_node
    from src.llm.budget_tracker import BudgetTracker

    # Postgres when it is up, so these calls count against the same daily
    # budget the API sees; otherwise the tracker's in-memory fallback.
    await BudgetTracker.get_instance(os.getenv(
        "POSTGRES_URL", "postgresql://manda_user:password@localhost:5432/manda_rag"))

    output = Path(args.output)
    existing = {}
    if output.exists() and not args.force:
        existing = json.loads(output.read_text(encoding="utf-8")).get("questions", {})

    golden = load_golden_set()
    provenance = {
        "about": "Agent 1 (src/agents/query_intelligence.py) output for each golden "
                 "question, cached so the retrieval eval is deterministic and makes "
                 "no LLM calls. Consumed by eval/run_retrieval_eval.py "
                 "(production_decomp ablation). An entry whose `query` no longer "
                 "matches the golden set is ignored.",
        "generator": "python -m eval.generate_sub_questions",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "golden_set_sha256": hashlib.sha256(GOLDEN_SET_PATH.read_bytes()).hexdigest(),
    }

    questions = dict(existing)
    calls = 0
    for q in golden["golden_qa_pairs"]:
        cached = questions.get(q["id"])
        if cached and cached.get("query") == q["query"]:
            continue
        if calls:
            await asyncio.sleep(args.delay)
        try:
            out = await query_intelligence_node(
                {"original_query": q["query"], "deal_id": DEAL_ID}
            )
        except Exception as e:
            print(f"{q['id']}: Agent 1 failed ({type(e).__name__}: {e}); stopping — "
                  "re-run later to resume.", file=sys.stderr)
            _save(output, provenance, questions)
            return 1
        calls += 1
        questions[q["id"]] = {
            "query": q["query"],
            "golden_query_type": q["query_type"],
            "query_type": out["query_type"],
            "sub_questions": out["sub_questions"],
            "reformulated_query": out["current_query"],
            "model": out["agent_trace"][0]["model"],
        }
        _save(output, provenance, questions)
        print(f"{q['id']}: {out['query_type']}, {len(out['sub_questions'])} sub-questions "
              f"({questions[q['id']]['model']})", file=sys.stderr)

    _save(output, provenance, questions)
    print(f"{calls} Agent 1 call(s); cache at {output}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.generate_sub_questions",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--force", action="store_true", help="regenerate every entry")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY_S,
                        help="seconds between calls (default 5)")
    args = parser.parse_args(argv)
    logging.disable(logging.INFO)
    return asyncio.run(generate(args))


if __name__ == "__main__":
    sys.exit(main())
