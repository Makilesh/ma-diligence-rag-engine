"""
Times extract_and_chunk over the sample deal with Laya off and on.

    python -m eval.decisions.timing --device cuda --mode union
    python -m eval.decisions.timing --device cpu --threads 2 --mode confirm

Run as its own process (the runner shells out to it) so the device, the torch
thread count and the risk mode are fixed before anything is loaded. Laya is
warmed first, so the "on" time is the steady-state cost an upload pays, not
the one-off model load. Prints one JSON line.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="")
    parser.add_argument("--mode", choices=("union", "confirm"), default="union")
    parser.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    parser.add_argument("--limit-docs", type=int, default=0, help="time only the first N documents")
    args = parser.parse_args()

    if args.device:
        os.environ["LAYA_DEVICE"] = args.device
    os.environ["LAYA_RISK_MODE"] = args.mode
    os.environ["LAYA_RISK"] = "1"
    os.environ["LAYA_CATEGORY"] = "1"
    os.environ["LAYA_PII"] = "0"
    logging.disable(logging.CRITICAL)
    sys.path.insert(0, str(PROJECT_ROOT))

    import torch

    if args.threads:
        torch.set_num_threads(args.threads)

    from src.data_processing.ingest_pipeline import extract_and_chunk
    from src.decisions.laya_client import loaded_model, warm_laya

    files = sorted((PROJECT_ROOT / "data" / "sample_deal").glob("*.txt"))
    if args.limit_docs:
        files = files[: args.limit_docs]

    def run() -> tuple[float, int, dict]:
        start = time.perf_counter()
        chunks, sources = 0, Counter()
        for path in files:
            doc = extract_and_chunk(str(path), path.name, "timing")
            chunks += len(doc.chunks)
            sources[f"category:{doc.category_source}"] += 1
            for c in doc.chunks:
                for decision in c["risk_decisions"].values():
                    sources[f"risk:{decision['source']}"] += 1
        return time.perf_counter() - start, chunks, dict(sources)

    os.environ["LAYA_ENABLED"] = "0"
    run()  # warm the tokenizer and parsers
    off, chunks, sources_off = run()
    os.environ["LAYA_ENABLED"] = "1"
    asyncio.run(warm_laya())
    on, _, sources_on = run()
    print(json.dumps({
        "device": args.device or "default", "threads": args.threads or torch.get_num_threads(),
        "mode": args.mode, "documents": len(files), "chunks": chunks,
        "off_s": round(off, 2), "on_s": round(on, 2), "added_s": round(on - off, 2),
        "sources_off": sources_off, "sources_on": sources_on, "model": loaded_model(),
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
