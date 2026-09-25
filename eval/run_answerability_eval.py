"""
Answerability gate calibration on the dev set — where the Laya thresholds come from.

    python -m eval.run_answerability_eval
    python -m eval.run_answerability_eval --compare-phrasings
    LAYA_DEVICE=cpu python -m eval.run_answerability_eval --threads 2 --limit 8   # CPU timing

Indexes data/sample_deal exactly as eval/run_retrieval_eval.py does, runs every
question in eval/answerability_dev.json through the production retrieval node
(no sub-questions: the dev set is not decomposed), then through the Quality
Assessor's heuristic and its Laya answerability check. Reports:

  * heuristic gate vs heuristic + Laya on the dev set;
  * a sweep of the veto threshold: false vetoes of answerable questions against
    unanswerable questions caught, split by what the heuristic did with them;
  * with --compare-phrasings, the same for each candidate instruction.

The golden set is NOT touched here: it is the held-out test set, reported by
run_retrieval_eval.py. An answerable question whose evidence retrieval failed to
put in the context is reported separately — the gate cannot be blamed for it.

Makes zero LLM calls. Writes eval/results/answerability_dev.json and
eval/results/answerability_dev.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from eval import metrics as m
from eval.corpus import PROJECT_ROOT, build_index, use_client
from eval.run_retrieval_eval import (
    RESULTS_DIR,
    _node_state,
    add_laya_gate,
    gate_decision,
    release_retrieval_models,
)

DEV_SET_PATH = PROJECT_ROOT / "eval" / "answerability_dev.json"

# The instructions compared on the dev set. P1 is the one adopted.
PHRASINGS = {
    "P1": "Does `passage` state the information needed to answer `question`?",
    "P2": "Does `passage` contain the answer to `question`?",
    "P3": "Can `question` be answered using only the facts stated in `passage`?",
}
SWEEP = (0.25, 0.30, 0.325, 0.35, 0.375, 0.40, 0.425, 0.45)


def _normalise(text: str) -> str:
    return " ".join(text.split())


def load_dev_set(path: Path = DEV_SET_PATH) -> list[dict]:
    """Dev questions; `note`-carrying entries (debatable labels) are kept but flagged."""
    return json.loads(path.read_text(encoding="utf-8"))["questions"]


def label(question: dict, reranked: list[dict]) -> str:
    """
    answerable / unanswerable / retrieval_miss / excluded, judged on the context.

    retrieval_miss: answerable in the corpus, but the verbatim evidence is in none
    of the reranked chunks, so the context genuinely lacks it.
    """
    if question.get("note"):
        return "excluded"
    if not question["answerable"]:
        return "unanswerable"
    evidence = _normalise(question["evidence"])
    if any(evidence in _normalise(c.get("text", "")) for c in reranked):
        return "answerable"
    return "retrieval_miss"


def sweep(rows: list[dict], key: str = "answerability") -> list[dict]:
    """False vetoes vs catches for each candidate threshold (Laya-consulted rows)."""
    def vetoed(r: dict, t: float) -> bool:
        return r[key] is not None and r[key] < t

    ans = [r for r in rows if r["label"] == "answerable"]
    una = [r for r in rows if r["label"] == "unanswerable"]
    out = []
    for t in SWEEP:
        out.append({
            "threshold": t,
            "false_vetoes": sum(vetoed(r, t) for r in ans),
            "answerable": len(ans),
            "caught": sum(vetoed(r, t) for r in una),
            "caught_heuristic_admitted": sum(
                vetoed(r, t) for r in una if r["heuristic"] == "admitted"),
            "caught_ambiguous": sum(vetoed(r, t) for r in una if r["heuristic"] == "llm_fallback"),
            "unanswerable": len(una),
            "unanswerable_heuristic_admitted": sum(r["heuristic"] == "admitted" for r in una),
            "retrieval_miss_vetoed": sum(
                vetoed(r, t) for r in rows if r["label"] == "retrieval_miss"),
        })
    return out


def auc(rows: list[dict], key: str = "answerability") -> float | None:
    """P(answerable scores above unanswerable) over consulted rows."""
    pos = [r[key] for r in rows if r["label"] == "answerable" and r[key] is not None]
    neg = [r[key] for r in rows if r["label"] == "unanswerable" and r[key] is not None]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def counts(rows: list[dict], field: str) -> dict:
    """admitted / llm_fallback / refused per label for one gate."""
    return {
        lab: {d: sum(1 for r in rows if r["label"] == lab and r[field] == d)
              for d in ("admitted", "llm_fallback", "refused")}
        for lab in ("answerable", "unanswerable", "retrieval_miss", "excluded")
    }


async def run(args: argparse.Namespace) -> dict:
    """Retrieves once per dev question, then gates it with each phrasing."""
    import src.decisions.answerability as ans
    from src.agents.retrieval_executor import retrieval_executor_node
    from src.vector_db.reranker import warm_models

    if args.threads:
        import torch

        torch.set_num_threads(args.threads)

    questions = load_dev_set()
    if args.limit:
        questions = questions[: args.limit]
    phrasings = PHRASINGS if args.compare_phrasings else {"P1": PHRASINGS["P1"]}

    started = time.perf_counter()
    client, _ = await build_index()
    rows: list[dict] = []
    contexts: list[tuple[dict, list[dict]]] = []
    with use_client(client):
        await warm_models()
        for q in questions:
            state = _node_state({"query": q["query"], "query_type": q["query_type"]}, [])
            reranked = (await retrieval_executor_node(state))["reranked_results"]
            gate = await gate_decision(state, reranked, laya=False)
            rows.append({"id": q["id"], "query": q["query"], "label": label(q, reranked),
                         "heuristic": gate["decision"],
                         "max_reranker_score": gate["max_reranker_score"]})
            contexts.append((state, reranked))

    # Retrieval is done: free its models before Laya loads (memory-tight hosts).
    release_retrieval_models()
    for n, (row, (state, reranked)) in enumerate(zip(rows, contexts), start=1):
        for name, instruction in phrasings.items():
            ans.ANSWERABILITY_INSTRUCTION = instruction
            gate: dict = {}
            await add_laya_gate(gate, state, reranked)
            laya = gate["laya"]
            if "unavailable" in laya:
                raise RuntimeError(f"Laya unavailable: {laya['unavailable']}")
            suffix = "" if name == "P1" else f"_{name}"
            row[f"laya{suffix}"] = laya.get("decision")
            row[f"answerability{suffix}"] = laya.get("answerability")
            row[f"laya_ms{suffix}"] = laya.get("latency_ms")
        ans.ANSWERABILITY_INSTRUCTION = PHRASINGS["P1"]
        print(f"[dev] {n}/{len(rows)} {row['id']} {row['label']} heuristic={row['heuristic']} "
              f"laya={row['laya']} P={row['answerability']}", file=sys.stderr)

    laya_ms = [r["laya_ms"] for r in rows if r["laya_ms"] is not None]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": "python -m eval.run_answerability_eval " + " ".join(sys.argv[1:]),
        "dev_set": str(DEV_SET_PATH.relative_to(PROJECT_ROOT)),
        "config": {
            "instruction": PHRASINGS["P1"],
            "top_k_passages": ans.TOP_K_PASSAGES,
            "veto_threshold": ans.VETO_THRESHOLD,
            "threads": args.threads,
        },
        "environment": {"laya_device": _laya_device()},
        "labels": {lab: sum(r["label"] == lab for r in rows)
                   for lab in ("answerable", "unanswerable", "retrieval_miss", "excluded")},
        "gates": {"heuristic": counts(rows, "heuristic"), "laya": counts(rows, "laya")},
        "phrasings": {
            name: {"instruction": text,
                   "auc": auc(rows, "answerability" if name == "P1" else f"answerability_{name}"),
                   "sweep": sweep(rows, "answerability" if name == "P1" else f"answerability_{name}")}
            for name, text in phrasings.items()
        },
        "laya_latency_ms": {"p50": m.percentile(laya_ms, 50), "p95": m.percentile(laya_ms, 95),
                            "n": len(laya_ms)},
        "questions": rows,
        "total_seconds": round(time.perf_counter() - started, 1),
    }
    return report


def _laya_device() -> str:
    import os

    import torch

    return os.getenv("LAYA_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")


def render_markdown(report: dict) -> str:
    """Summary: gate counts, AUC and threshold sweep per phrasing, latency."""
    cfg = report["config"]
    lab = report["labels"]
    lat = report["laya_latency_ms"]
    lines = [
        f"# Answerability gate — dev set, {report['generated_at'][:10]}", "",
        f"{report['dev_set']}: {lab['answerable']} answerable (evidence in context), "
        f"{lab['unanswerable']} unanswerable, {lab['retrieval_miss']} answerable-but-retrieval-missed, "
        f"{lab['excluded']} excluded (debatable label). Adopted: `{cfg['instruction']}` over the top "
        f"{cfg['top_k_passages']} reranked chunks, veto below {cfg['veto_threshold']}. LLM calls: 0.", "",
        "| label | gate | admitted | ambiguous → LLM | refused |", "|---|---|---|---|---|",
    ]
    for label_ in ("answerable", "unanswerable", "retrieval_miss"):
        for gate in ("heuristic", "laya"):
            c = report["gates"][gate][label_]
            lines.append(f"| {label_} | {'heuristic' if gate == 'heuristic' else '+ Laya'} | "
                         f"{c['admitted']} | {c['llm_fallback']} | {c['refused']} |")
    lines.append("")
    for name, data in report["phrasings"].items():
        auc_ = "—" if data["auc"] is None else f"{data['auc']:.3f}"
        lines += [f"## {name}: `{data['instruction']}` — AUC {auc_}", "",
                  "| veto below | false vetoes (answerable) | unanswerable caught | "
                  "of which heuristic-admitted | of which ambiguous | retrieval misses vetoed |",
                  "|---|---|---|---|---|---|"]
        for s in data["sweep"]:
            lines.append(
                f"| {s['threshold']} | {s['false_vetoes']}/{s['answerable']} | "
                f"{s['caught']}/{s['unanswerable']} | {s['caught_heuristic_admitted']}/"
                f"{s['unanswerable_heuristic_admitted']} | {s['caught_ambiguous']} | "
                f"{s['retrieval_miss_vetoed']} |")
        lines.append("")
    lines += [f"Laya latency per assessment ({report['environment']['laya_device']}"
              f"{', ' + str(cfg['threads']) + ' threads' if cfg['threads'] else ''}): "
              f"p50 {lat['p50']:.0f} ms, p95 {lat['p95']:.0f} ms (n={lat['n']}).", ""]
    lowest = sorted((r["answerability"], r["id"]) for r in report["questions"]
                    if r["label"] == "answerable" and r["answerability"] is not None)[:3]
    lines.append("Lowest-scoring answerable questions: "
                 + ", ".join(f"{i} {p}" for p, i in lowest))
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m eval.run_answerability_eval",
        description="Laya answerability gate on eval/answerability_dev.json (no LLM calls).")
    parser.add_argument("--compare-phrasings", action="store_true",
                        help="also score the alternative instructions P2, P3")
    parser.add_argument("--limit", type=int, help="first N dev questions only (timing runs)")
    parser.add_argument("--threads", type=int, help="torch.set_num_threads, e.g. 2 to mimic the host")
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    if not args.verbose:
        logging.disable(logging.INFO)
    try:
        report = asyncio.run(run(args))
    except Exception as e:
        print(f"[dev] FAILED to run: {e}", file=sys.stderr)
        return 2
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    suffix = "" if not (args.limit or args.threads) else "_timing"
    (out / f"answerability_dev{suffix}.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown = render_markdown(report)
    (out / f"answerability_dev{suffix}.md").write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
