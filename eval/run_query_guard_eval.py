"""
Public query guard measurement — precision and recall on eval/query_guard_set.json.

    python -m eval.run_query_guard_eval
    LAYA_DEVICE=cpu python -m eval.run_query_guard_eval --threads 2   # CPU latency

Scores every labelled prompt with the guard's three Laya questions
(src/decisions/query_guard.py), then reports: the guard at its configured
thresholds (precision, recall, false blocks by name), a sweep of each threshold
with the other two held at their configured values, and single-query latency.
Negatives are the golden and dev-set questions plus the set's extra on-topic
variants; a false block is a genuine question turned away.

Makes zero LLM calls. Writes eval/results/query_guard.json and
eval/results/query_guard.md.
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
from eval.corpus import GOLDEN_SET_PATH, PROJECT_ROOT
from eval.run_retrieval_eval import RESULTS_DIR

GUARD_SET_PATH = PROJECT_ROOT / "eval" / "query_guard_set.json"
DEV_SET_PATH = PROJECT_ROOT / "eval" / "answerability_dev.json"

TOPIC_SWEEP = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50)
FLAG_SWEEP = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80)


def load_prompts() -> tuple[list[dict], list[str]]:
    """(labelled rows, borderline prompts)."""
    guard = json.loads(GUARD_SET_PATH.read_text(encoding="utf-8"))
    golden = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))["golden_qa_pairs"]
    dev = json.loads(DEV_SET_PATH.read_text(encoding="utf-8"))["questions"]
    rows = [{"prompt": q["query"], "positive": False, "source": "golden"} for q in golden]
    rows += [{"prompt": q["query"], "positive": False, "source": "dev"} for q in dev]
    rows += [{"prompt": p, "positive": False, "source": "extra"} for p in guard["extra_negatives"]]
    rows += [{"prompt": p["prompt"], "positive": True, "source": p["category"]}
             for p in guard["positives"]]
    return rows, guard["borderline"]


def confusion(rows: list[dict], blocked_key: str = "blocked") -> dict:
    """Precision / recall of blocking, plus the false blocks by prompt."""
    tp = sum(r["positive"] and r[blocked_key] for r in rows)
    fp = sum((not r["positive"]) and r[blocked_key] for r in rows)
    fn = sum(r["positive"] and not r[blocked_key] for r in rows)
    return {
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "blocked_positives": tp, "positives": tp + fn,
        "false_blocks": fp, "negatives": sum(not r["positive"] for r in rows),
        "false_block_prompts": [r["prompt"] for r in rows if not r["positive"] and r[blocked_key]],
        "missed_prompts": [r["prompt"] for r in rows if r["positive"] and not r[blocked_key]],
    }


async def run(args: argparse.Namespace) -> dict:
    """Scores every prompt once, then evaluates configured and swept thresholds."""
    from src.decisions import query_guard as qg

    if args.threads:
        import torch

        torch.set_num_threads(args.threads)

    rows, borderline = load_prompts()
    scores = await qg.score_queries([r["prompt"] for r in rows] + borderline)
    for r, s in zip(rows, scores):
        r["scores"] = {k: round(v, 4) for k, v in s.items()}
        r["reasons"] = qg.decide(s)
        r["blocked"] = bool(r["reasons"])
    border = [{"prompt": p, "scores": {k: round(v, 4) for k, v in s.items()},
               "reasons": qg.decide(s)} for p, s in zip(borderline, scores[len(rows):])]

    def swept(**overrides) -> dict:
        for r in rows:
            r["_b"] = bool(qg.decide(r["scores"], **overrides))
        return confusion(rows, "_b")

    defaults = {"on_topic_min": qg.ON_TOPIC_MIN, "jailbreak_max": qg.JAILBREAK_MAX,
                "injection_max": qg.INJECTION_MAX}
    sweeps = {
        "on_topic_min": [{"value": t, **swept(**{**defaults, "on_topic_min": t})}
                         for t in TOPIC_SWEEP],
        "jailbreak_max": [{"value": t, **swept(**{**defaults, "jailbreak_max": t})}
                          for t in FLAG_SWEEP],
        "injection_max": [{"value": t, **swept(**{**defaults, "injection_max": t})}
                          for t in FLAG_SWEEP],
    }
    for r in rows:
        r.pop("_b", None)

    # Latency as the route sees it: one query per call, after warmup.
    timings = []
    for r in rows[:: max(1, len(rows) // args.timing_samples)][: args.timing_samples]:
        start = time.perf_counter()
        await qg.check_query(r["prompt"])
        timings.append((time.perf_counter() - start) * 1000)

    import os

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": "python -m eval.run_query_guard_eval " + " ".join(sys.argv[1:]),
        "config": {**defaults, "on_topic_question": qg.ON_TOPIC_QUESTION,
                   "threads": args.threads, "laya_device": os.getenv("LAYA_DEVICE", "default")},
        "configured": confusion(rows),
        "by_source": {
            src: {"n": sum(r["source"] == src for r in rows),
                  "blocked": sum(r["source"] == src and r["blocked"] for r in rows)}
            for src in sorted({r["source"] for r in rows})
        },
        "sweeps": sweeps,
        "borderline": border,
        "latency_ms": {"p50": m.percentile(timings, 50), "p95": m.percentile(timings, 95),
                       "n": len(timings)},
        "rows": rows,
    }


def _pct(v: float | None) -> str:
    return "—" if v is None else f"{v * 100:.1f}%"


def render_markdown(report: dict) -> str:
    """Configured result, per-source counts, sweeps, borderline prompts, latency."""
    c, cfg = report["configured"], report["config"]
    lines = [
        f"# Query guard — {report['generated_at'][:10]}", "",
        f"Configured: block if on_topic < {cfg['on_topic_min']} or jailbreak >= "
        f"{cfg['jailbreak_max']} or prompt_injection >= {cfg['injection_max']}. "
        f"on_topic = P(`{qg_option()}`) from the choice "
        f"`{cfg['on_topic_question']['instructions']}` over "
        f"{', '.join(cfg['on_topic_question']['criteria'])}.", "",
        f"**Precision {_pct(c['precision'])}, recall {_pct(c['recall'])}** — blocked "
        f"{c['blocked_positives']}/{c['positives']} junk prompts, false blocks "
        f"{c['false_blocks']}/{c['negatives']} genuine questions.", "",
        "| source | n | blocked |", "|---|---|---|",
    ]
    lines += [f"| {s} | {v['n']} | {v['blocked']} |" for s, v in report["by_source"].items()]
    lines.append("")
    if c["false_block_prompts"]:
        lines += ["False blocks: " + "; ".join(c["false_block_prompts"]), ""]
    if c["missed_prompts"]:
        lines += ["Missed: " + "; ".join(c["missed_prompts"]), ""]
    for name, rows in report["sweeps"].items():
        lines += [f"## Sweep: {name} (others at configured values)", "",
                  "| value | precision | recall | false blocks |", "|---|---|---|---|"]
        lines += [f"| {r['value']} | {_pct(r['precision'])} | {_pct(r['recall'])} | "
                  f"{r['false_blocks']}/{r['negatives']} |" for r in rows]
        lines.append("")
    lines += ["## Borderline (not scored)", "", "| prompt | on_topic | jailbreak | injection | blocked |",
              "|---|---|---|---|---|"]
    for b in report["borderline"]:
        s = b["scores"]
        lines.append(f"| {b['prompt']} | {s['on_topic']:.2f} | {s['jailbreak']:.2f} | "
                     f"{s['prompt_injection']:.2f} | {', '.join(b['reasons']) or 'no'} |")
    lat = report["latency_ms"]
    lines += ["", f"Single-query latency ({cfg['laya_device']}"
              f"{', ' + str(cfg['threads']) + ' threads' if cfg['threads'] else ''}): "
              f"p50 {lat['p50']:.0f} ms, p95 {lat['p95']:.0f} ms (n={lat['n']})."]
    negs = sorted((r["scores"]["on_topic"], r["prompt"]) for r in report["rows"] if not r["positive"])
    lines.append("Lowest on-topic genuine questions: "
                 + "; ".join(f"{p:.3f} {q}" for p, q in negs[:3]))
    return "\n".join(lines)


def qg_option() -> str:
    from src.decisions.query_guard import ON_TOPIC_OPTION

    return ON_TOPIC_OPTION


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m eval.run_query_guard_eval",
                                     description="Query guard precision/recall (no LLM calls).")
    parser.add_argument("--threads", type=int, help="torch.set_num_threads (mimic a small host)")
    parser.add_argument("--timing-samples", type=int, default=10)
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    logging.disable(logging.INFO)
    try:
        report = asyncio.run(run(args))
    except Exception as e:
        print(f"[guard] FAILED to run: {e}", file=sys.stderr)
        return 2
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = "query_guard" if not args.threads else "query_guard_timing"
    (out / f"{name}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    markdown = render_markdown(report)
    (out / f"{name}.md").write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
