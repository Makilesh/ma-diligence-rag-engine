"""
Ingest-decision evaluation — regex vs Laya vs hybrid for risk signals, PII and
document category, on the hand-labelled sets in eval/decisions/data/.

    python -m eval.decisions.run_ingest_decisions_eval
    python -m eval.decisions.run_ingest_decisions_eval --timing          # + ingest cost, GPU and CPU
    python -m eval.decisions.run_ingest_decisions_eval --no-cache

For each task it compares:

    regex_legacy   the detectors as of LEGACY_COMMIT (loaded from git, unmodified)
    regex_fixed    the current detectors with Laya off (known regex bugs fixed)
    laya           Laya alone, one threshold
    hybrid_*       regex and Laya combined (see each task)

Every threshold and phrasing is chosen on the DEV split and reported on TEST;
DEV numbers are printed alongside only so over-fitting is visible. Corpus rows
pin their chunk text by sha1: if chunking changes, the run stops rather than
scoring stale labels.

PII also gets a retrieval-damage metric: PII-flagged chunks are excluded from
retrieval, so for each design it counts the corpus chunks it would exclude and
which golden-set facts (tests/golden_qa_set.json) those chunks carry.

Laya answers are cached on disk (keyed by model, question and text), so
re-running with different thresholds costs nothing; --no-cache recomputes.
Writes eval/decisions/results.md and eval/decisions/results.json.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval import metrics as m  # noqa: E402
from src.data_processing.document_classifier import DocumentClassifier  # noqa: E402
from src.data_processing.ingest_pipeline import _content_sample, extract_and_chunk  # noqa: E402
from src.data_processing.pii_detector import PIIDetector  # noqa: E402
from src.data_processing.risk_signal_extractor import RiskSignalExtractor  # noqa: E402
from src.decisions import ingest_signals as sig  # noqa: E402
from src.decisions.laya_client import decide_batch, laya_model, noul  # noqa: E402

DECISIONS_DIR = PROJECT_ROOT / "eval" / "decisions"
DATA_DIR = DECISIONS_DIR / "data"
CORPUS_DIR = PROJECT_ROOT / "data" / "sample_deal"
GOLDEN_PATH = PROJECT_ROOT / "tests" / "golden_qa_set.json"
DEFAULT_CACHE = Path(tempfile.gettempdir()) / "redline_laya_eval_cache.json"

# The detectors before this work. Loaded with `git show`, never edited.
LEGACY_COMMIT = "b2b4ee3"

THRESHOLDS = [round(0.05 + 0.025 * i, 3) for i in range(37)]  # 0.05 .. 0.95
DEAL = "eval-decisions"


# ==============================================================================
# Data
# ==============================================================================


def load_json(name: str) -> dict:
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


def corpus_texts() -> dict[tuple[str, int], str]:
    """Every chunk the real ingest path produces for the sample deal."""
    texts = {}
    for path in sorted(CORPUS_DIR.glob("*.txt")):
        doc = extract_and_chunk(str(path), path.name, DEAL)
        for chunk in doc.chunks:
            texts[(path.name, chunk["chunk_index"])] = chunk["text"]
    return texts


def attach_texts(rows: list[dict], texts: dict[tuple[str, int], str]) -> list[dict]:
    """Adds `text` to corpus rows, refusing labels whose chunk text has changed."""
    stale = []
    for row in rows:
        text = texts.get((row["file"], row["chunk_index"]))
        if text is None or hashlib.sha1(text.encode("utf-8")).hexdigest() != row["sha1"]:
            stale.append(row["id"])
        row["text"] = text or ""
    if stale or len(rows) != len(texts):
        raise SystemExit(
            f"Labels are stale: {len(stale)} rows no longer match the chunker output "
            f"({stale[:5]}...), {len(rows)} rows vs {len(texts)} chunks. Re-label before scoring."
        )
    return rows


def load_legacy(module_path: str, alias: str):
    """Imports a module as it was at LEGACY_COMMIT."""
    source = subprocess.run(
        ["git", "show", f"{LEGACY_COMMIT}:{module_path}"],
        cwd=PROJECT_ROOT, capture_output=True, check=True,
    ).stdout.decode("utf-8")
    spec = importlib.util.spec_from_loader(alias, loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module  # dataclasses resolve annotations through sys.modules
    exec(compile(source, f"{LEGACY_COMMIT}:{module_path}", "exec"), module.__dict__)
    return module


# ==============================================================================
# Laya with a disk cache
# ==============================================================================


class Scorer:
    """decide_batch with a per-(model, question, state) disk cache."""

    def __init__(self, cache_path: Path | None):
        self.cache_path = cache_path
        self.cache: dict[str, dict] = {}
        if cache_path and cache_path.exists():
            self.cache = json.loads(cache_path.read_text(encoding="utf-8"))
        self.calls = 0

    def _key(self, state: dict, question: dict) -> str:
        blob = json.dumps([laya_model(), state, question], sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()

    def answers(self, states: list[dict], questions: dict[str, dict]) -> list[dict[str, dict]]:
        """One {qname: answer} per state; only uncached (state, question) pairs hit the model."""
        out: list[dict[str, dict]] = [{} for _ in states]
        missing: dict[int, dict[str, dict]] = {}
        for i, state in enumerate(states):
            for name, q in questions.items():
                hit = self.cache.get(self._key(state, q))
                if hit is None:
                    missing.setdefault(i, {})[name] = q
                else:
                    out[i][name] = hit
        if missing:
            order = sorted(missing)
            self.calls += len(order)
            results = decide_batch([(states[i], missing[i]) for i in order])
            for i, answer in zip(order, results):
                for name, value in answer.items():
                    slim = {k: value[k] for k in ("noul", "choice", "probabilities", "confidence")
                            if k in value}
                    out[i][name] = slim
                    self.cache[self._key(states[i], missing[i][name])] = slim
        return out

    def save(self) -> None:
        if self.cache_path:
            self.cache_path.write_text(json.dumps(self.cache), encoding="utf-8")


# ==============================================================================
# Metrics
# ==============================================================================


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 3), "recall": round(r, 3),
            "f1": round(f, 3)}


def score_multilabel(rows: list[dict], predictions: list[set[str]], categories: list[str]) -> dict:
    """Micro and per-category P/R/F1; (row, category) pairs marked borderline are skipped."""
    per = {c: Counter() for c in categories}
    for row, pred in zip(rows, predictions):
        gold, skip = set(row["labels"]), set(row.get("borderline", []))
        for c in categories:
            if c in skip:
                continue
            if c in pred and c in gold:
                per[c]["tp"] += 1
            elif c in pred:
                per[c]["fp"] += 1
            elif c in gold:
                per[c]["fn"] += 1
    total = sum(per.values(), Counter())
    return {"micro": prf(total["tp"], total["fp"], total["fn"]),
            "per_category": {c: prf(per[c]["tp"], per[c]["fp"], per[c]["fn"]) for c in categories}}


def score_binary(rows: list[dict], predictions: list[bool]) -> dict:
    """P/R/F1 for the PII flag; borderline rows are skipped."""
    c = Counter()
    for row, pred in zip(rows, predictions):
        if row.get("borderline"):
            continue
        gold = bool(row["pii"])
        c["tp" if pred and gold else "fp" if pred else "fn" if gold else "tn"] += 1
    out = prf(c["tp"], c["fp"], c["fn"])
    out["tn"] = c["tn"]
    return out


def split(rows: list[dict], name: str) -> list[int]:
    return [i for i, r in enumerate(rows) if r["split"] == name]


def pick(idx: list[int], seq: list) -> list:
    return [seq[i] for i in idx]


def best_threshold(evaluate, thresholds=THRESHOLDS) -> tuple[float, dict]:
    """Threshold maximising DEV F1; ties go to the higher threshold (fewer flags)."""
    best_t, best = thresholds[0], None
    for t in thresholds:
        s = evaluate(t)
        if best is None or s["f1"] >= best["f1"]:
            best_t, best = t, s
    return best_t, best


# ==============================================================================
# Task 1 — risk signals
# ==============================================================================


def run_risk(scorer: Scorer, texts: dict) -> dict:
    data = load_json("risk_labels.json")
    categories = data["_meta"]["categories"]
    rows = attach_texts(data["corpus"], texts) + data["synthetic"]
    dev, test = split(rows, "dev"), split(rows, "test")

    legacy = load_legacy("src/data_processing/risk_signal_extractor.py", "legacy_risk")
    regex_legacy = [set(legacy.RiskSignalExtractor().extract(r["text"]).signals) for r in rows]
    fixed_extractor = RiskSignalExtractor()
    regex_fixed = [set(fixed_extractor.extract(r["text"]).signals) for r in rows]

    states = [{"passage": r["text"]} for r in rows]
    probs_by_phrasing: dict[str, list[dict[str, float]]] = {}
    for phrasing in sig.RISK_PHRASINGS:
        answers = scorer.answers(states, sig.risk_questions(phrasing))
        probs_by_phrasing[phrasing] = [
            {name.split(":", 1)[1]: noul(a) for name, a in ans.items()} for ans in answers
        ]

    def micro(idx, preds):
        return score_multilabel(pick(idx, rows), pick(idx, preds), categories)["micro"]

    def laya_preds(probs, t):
        return [{c for c, p in pr.items() if p >= t} for pr in probs]

    def confirm_preds(probs, t):
        return [{c for c in rx if pr.get(c, 0) >= t} for rx, pr in zip(regex_fixed, probs)]

    def union_preds(probs, hi, lo):
        return [{c for c, p in pr.items() if p >= hi or (c in rx and p >= lo)}
                for rx, pr in zip(regex_fixed, probs)]

    # Phrasing search (DEV only), Laya alone.
    phrasing_dev = {}
    for phrasing, probs in probs_by_phrasing.items():
        t, s = best_threshold(lambda t, probs=probs: micro(dev, laya_preds(probs, t)))
        phrasing_dev[phrasing] = {"threshold": t, **s}
    chosen = max(phrasing_dev, key=lambda k: (phrasing_dev[k]["f1"], k == sig.RISK_PHRASING))
    probs = probs_by_phrasing[chosen]

    designs: dict[str, dict] = {}

    def record(name, preds, params):
        designs[name] = {
            "params": params,
            "dev": micro(dev, preds),
            "test": score_multilabel(pick(test, rows), pick(test, preds), categories),
        }

    record("regex_legacy", regex_legacy, {})
    record("regex_fixed", regex_fixed, {})
    t_laya, _ = best_threshold(lambda t: micro(dev, laya_preds(probs, t)))
    record("laya", laya_preds(probs, t_laya), {"phrasing": chosen, "threshold": t_laya})
    t_conf, _ = best_threshold(lambda t: micro(dev, confirm_preds(probs, t)))
    record("hybrid_confirm", confirm_preds(probs, t_conf),
           {"phrasing": chosen, "confirm_threshold": t_conf,
            "rule": "regex_fixed proposes, Laya confirms (P >= t)"})
    best = None
    for hi in THRESHOLDS:
        for lo in THRESHOLDS:
            if lo > hi:
                continue
            s = micro(dev, union_preds(probs, hi, lo))
            if best is None or s["f1"] > best[2]["f1"]:
                best = (hi, lo, s)
    hi, lo, _ = best
    record("hybrid_union", union_preds(probs, hi, lo),
           {"phrasing": chosen, "threshold": hi, "confirm_threshold": lo,
            "rule": "Laya P >= t, or regex_fixed match with Laya P >= t_confirm"})

    # The code path ingest actually runs, with the module's constants — so the
    # table shows what ships, not a re-implementation of it.
    wired = probs_by_phrasing[sig.RISK_PHRASING]
    base = [fixed_extractor.extract(r["text"]) for r in rows]
    record("wired_union", [set(fixed_extractor.apply_laya(b, pr).signals) for b, pr in zip(base, wired)],
           {"phrasing": sig.RISK_PHRASING, "threshold": sig.RISK_THRESHOLD,
            "confirm_threshold": sig.RISK_CONFIRM_THRESHOLD, "rule": "LAYA_RISK_MODE=union"})
    record("wired_confirm",
           [set(fixed_extractor.apply_laya(b, {c: pr[c] for c in b.signals}).signals)
            for b, pr in zip(base, wired)],
           {"phrasing": sig.RISK_PHRASING, "confirm_threshold": sig.RISK_CONFIRM_THRESHOLD,
            "rule": "LAYA_RISK_MODE=confirm (Laya asked only about regex matches)"})

    # Rows each design gets wrong on TEST, for the write-up.
    final = union_preds(probs, hi, lo)
    errors = []
    for i in test:
        gold = set(rows[i]["labels"]) - set(rows[i].get("borderline", []))
        pred = final[i] - set(rows[i].get("borderline", []))
        if gold != pred:
            errors.append({"id": rows[i]["id"], "missed": sorted(gold - pred),
                           "extra": sorted(pred - gold)})

    return {
        "rows": {"dev": len(dev), "test": len(test)},
        "positives": {
            s: dict(Counter(c for i in split(rows, s) for c in rows[i]["labels"])) for s in ("dev", "test")
        },
        "borderline_pairs": sum(len(r.get("borderline", [])) for r in rows),
        "phrasing_dev": phrasing_dev,
        "chosen_phrasing": chosen,
        "designs": designs,
        "hybrid_union_test_errors": errors,
    }


# ==============================================================================
# Task 2 — PII
# ==============================================================================


def run_pii(scorer: Scorer, texts: dict) -> dict:
    data = load_json("pii_labels.json")
    corpus_rows = attach_texts(data["corpus"], texts)
    rows = corpus_rows + data["synthetic"]
    dev, test = split(rows, "dev"), split(rows, "test")

    legacy = load_legacy("src/data_processing/pii_detector.py", "legacy_pii")
    legacy_detector = legacy.PIIDetector()
    fixed_detector = PIIDetector()
    # Ingest calls detect(chunk.text) without a filename; so does this.
    legacy_results = [legacy_detector.detect(r["text"]) for r in rows]
    fixed_results = [fixed_detector.detect(r["text"]) for r in rows]
    regex_legacy = [bool(x.contains_pii) for x in legacy_results]
    regex_fixed = [bool(x.contains_pii) for x in fixed_results]
    strong = [bool(set(x.pii_types) & PIIDetector.STRONG_TYPES) for x in fixed_results]

    states = [{"passage": r["text"]} for r in rows]
    probs_by_phrasing = {}
    for phrasing in sig.PII_PHRASINGS:
        answers = scorer.answers(states, sig.pii_questions(phrasing))
        probs_by_phrasing[phrasing] = [max(noul(a) for a in ans.values()) for ans in answers]

    def binary(idx, preds):
        return score_binary(pick(idx, rows), pick(idx, preds))

    phrasing_dev = {}
    for phrasing, probs in probs_by_phrasing.items():
        t, s = best_threshold(lambda t, probs=probs: binary(dev, [p >= t for p in probs]))
        phrasing_dev[phrasing] = {"threshold": t, **s}
    chosen = max(phrasing_dev, key=lambda k: (phrasing_dev[k]["f1"], k == sig.PII_PHRASING))
    probs = probs_by_phrasing[chosen]

    def confirm_preds(t):
        return [rx and p >= t for rx, p in zip(regex_fixed, probs)]

    def hybrid_preds(hi, lo):
        # Strong identifiers (SSN, DOB, passport) stand on their own; other regex
        # hits need Laya's agreement; Laya alone flags only when confident.
        return [st or (rx and p >= lo) or p >= hi
                for st, rx, p in zip(strong, regex_fixed, probs)]

    designs: dict[str, dict] = {}
    preds_by_design: dict[str, list[bool]] = {}

    def record(name, preds, params):
        preds_by_design[name] = preds
        designs[name] = {"params": params, "dev": binary(dev, preds), "test": binary(test, preds)}

    record("regex_legacy", regex_legacy, {})
    record("regex_fixed", regex_fixed, {})
    t_laya, _ = best_threshold(lambda t: binary(dev, [p >= t for p in probs]))
    record("laya", [p >= t_laya for p in probs], {"phrasing": chosen, "threshold": t_laya})
    t_conf, _ = best_threshold(lambda t: binary(dev, confirm_preds(t)))
    record("hybrid_confirm", confirm_preds(t_conf),
           {"phrasing": chosen, "confirm_threshold": t_conf,
            "rule": "regex_fixed proposes, Laya confirms"})
    best = None
    for hi in THRESHOLDS:
        for lo in THRESHOLDS:
            if lo > hi:
                continue
            s = binary(dev, hybrid_preds(hi, lo))
            if best is None or s["f1"] > best[2]["f1"]:
                best = (hi, lo, s)
    hi, lo, _ = best
    record("hybrid_strong", hybrid_preds(hi, lo),
           {"phrasing": chosen, "threshold": hi, "confirm_threshold": lo,
            "rule": "strong regex identifier, or regex hit with Laya P >= t_confirm, or Laya P >= t"})
    # Laya may only veto weak regex hits; it never adds a flag of its own.
    t_veto, _ = best_threshold(lambda t: binary(dev, hybrid_preds(2.0, t)))
    record("hybrid_veto", hybrid_preds(2.0, t_veto),
           {"phrasing": chosen, "confirm_threshold": t_veto,
            "rule": "strong regex identifier, or other regex hit with Laya P >= t_confirm"})
    wired = probs_by_phrasing[sig.PII_PHRASING]
    record("wired_laya_pii", [bool(fixed_detector.detect(r["text"], laya_probability=p).contains_pii)
                              for r, p in zip(rows, wired)],
           {"phrasing": sig.PII_PHRASING, "threshold": sig.PII_THRESHOLD,
            "confirm_threshold": sig.PII_CONFIRM_THRESHOLD, "rule": "LAYA_PII=1 (off by default)"})

    # Per-kind recall on TEST (identifier vs individual compensation).
    for name, preds in preds_by_design.items():
        kinds = Counter()
        for i in test:
            r = rows[i]
            if r["pii"] and not r.get("borderline"):
                kinds[(r.get("pii_kind", "?"), "total")] += 1
                kinds[(r.get("pii_kind", "?"), "found")] += int(preds[i])
        designs[name]["test_recall_by_kind"] = {
            k: f"{kinds[(k, 'found')]}/{kinds[(k, 'total')]}"
            for k in sorted({k for k, _ in kinds})
        }

    damage = retrieval_damage(corpus_rows, {n: p[: len(corpus_rows)] for n, p in preds_by_design.items()})
    return {
        "rows": {"dev": len(dev), "test": len(test)},
        "positives": {s: sum(rows[i]["pii"] for i in split(rows, s)) for s in ("dev", "test")},
        "borderline_rows": sum(1 for r in rows if r.get("borderline")),
        "phrasing_dev": phrasing_dev,
        "chosen_phrasing": chosen,
        "designs": designs,
        "retrieval_damage": damage,
        "corpus_probabilities": {
            r["id"]: round(p, 3) for r, p in zip(corpus_rows, probs) if p >= 0.3 or r["pii"]
        },
    }


def retrieval_damage(corpus_rows: list[dict], flags_by_design: dict[str, list[bool]]) -> dict:
    """
    What each design's PII exclusions cost retrieval on the golden set.

    A chunk is relevant to a golden question when it comes from a cited file and
    contains one of the question's expected facts (eval.metrics' matching). A
    fact is lost when every chunk carrying it in a cited file is excluded.
    """
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))["golden_qa_pairs"]
    out = {}
    for name, flags in flags_by_design.items():
        excluded = [r for r, f in zip(corpus_rows, flags) if f]
        kept = [r for r, f in zip(corpus_rows, flags) if not f]
        relevant_excluded: set[str] = set()
        lost: list[str] = []
        for qa in golden:
            facts = qa["expected_answer_contains"]
            patterns = [c["source_pattern"] for c in qa.get("expected_citations", [])]
            if not facts or not patterns:
                continue
            for r in excluded:
                if m.source_matches(r["file"], patterns) and m.facts_in_text(facts, r["text"]):
                    relevant_excluded.add(r["id"])
            for fact in facts:
                was = any(m.source_matches(r["file"], patterns) and m.fact_present(fact, r["text"])
                          for r in corpus_rows)
                now = any(m.source_matches(r["file"], patterns) and m.fact_present(fact, r["text"])
                          for r in kept)
                if was and not now:
                    lost.append(f"{qa['id']}:{m.fact_label(fact)}")
        out[name] = {
            "excluded_chunks": len(excluded),
            "excluded_ids": [r["id"] for r in excluded],
            "excluded_with_golden_fact": sorted(relevant_excluded),
            "golden_facts_lost": lost,
            "golden_questions_hit": sorted({x.split(":", 1)[0] for x in lost}),
        }
    return out


# ==============================================================================
# Task 3 — document category
# ==============================================================================


def category_inputs(rows: list[dict], workdir: Path) -> None:
    """Adds `sample` (the text the classifier sees) and `file_type` to each row."""
    from tests.fixtures import ingestion_docs as fx

    for row in rows:
        if row["source"] == "corpus":
            path = CORPUS_DIR / row["filename"]
        elif row["source"] == "fixture":
            path = getattr(fx, row["writer"])(workdir)
        else:
            row["sample"] = row["text"][:2000]
            row["file_type"] = Path(row["filename"]).suffix.lstrip(".")
            continue
        suffix = path.suffix.lower()
        row["sample"] = _content_sample(str(path), suffix)
        row["file_type"] = suffix.lstrip(".")


def run_category(scorer: Scorer) -> dict:
    data = load_json("category_labels.json")
    rows = data["documents"]
    with tempfile.TemporaryDirectory() as tmp:
        category_inputs(rows, Path(tmp))
    dev, test = split(rows, "dev"), split(rows, "test")

    legacy = load_legacy("src/data_processing/document_classifier.py", "legacy_classifier").DocumentClassifier()
    fixed = DocumentClassifier()
    regex_legacy = [legacy.classify(r["filename"], r["file_type"], r["sample"]) for r in rows]
    regex_fixed = [fixed.classify(r["filename"], r["file_type"], r["sample"]) for r in rows]

    laya_by_phrasing: dict[str, list[tuple[str, float] | None]] = {}
    for phrasing in sig.CATEGORY_PHRASINGS:
        states, idx = [], []
        for i, r in enumerate(rows):
            if r["sample"].strip():
                states.append(sig.category_state(r["filename"], r["sample"], phrasing))
                idx.append(i)
        answers = scorer.answers(states, sig.category_question(phrasing))
        results: list[tuple[str, float] | None] = [None] * len(rows)
        for i, ans in zip(idx, answers):
            a = ans["category"]
            results[i] = (a["choice"], float(a["probabilities"][a["choice"]]))
        laya_by_phrasing[phrasing] = results

    def accuracy(idx, preds, lenient=False):
        ok = sum(1 for i in idx if (preds[i] in rows[i]["acceptable"] if lenient
                                    else preds[i] == rows[i]["category"]))
        return round(ok / len(idx), 3) if idx else 0.0

    def laya_only(results, fallback):
        return [res[0] if res else fb for res, fb in zip(results, fallback)]

    def override(results, fallback, c):
        return [res[0] if res and res[1] >= c else fb for res, fb in zip(results, fallback)]

    phrasing_dev = {p: accuracy(dev, laya_only(res, regex_fixed)) for p, res in laya_by_phrasing.items()}
    chosen = max(phrasing_dev, key=lambda k: (phrasing_dev[k], k == sig.CATEGORY_PHRASING))
    results = laya_by_phrasing[chosen]

    best_c, best_acc = 1.0, -1.0
    for c in THRESHOLDS:
        acc = accuracy(dev, override(results, regex_fixed, c))
        if acc >= best_acc:
            best_c, best_acc = c, acc

    preds = {
        "regex_legacy": (regex_legacy, {}),
        "regex_fixed": (regex_fixed, {}),
        "laya": (laya_only(results, regex_fixed),
                 {"phrasing": chosen, "note": "regex_fixed only where there is no text sample (.xlsx)"}),
        "hybrid_override": (override(results, regex_fixed, best_c),
                            {"phrasing": chosen, "min_confidence": best_c,
                             "rule": "Laya's choice when its confidence >= c, else regex_fixed"}),
    }
    groups = {
        "test_all": test,
        "test_corpus_and_fixtures": [i for i in test if rows[i]["source"] != "synthetic"],
        "test_synthetic": [i for i in test if rows[i]["source"] == "synthetic"],
    }
    designs = {}
    for name, (p, params) in preds.items():
        designs[name] = {
            "params": params,
            "dev_accuracy": accuracy(dev, p),
            **{f"{g}_accuracy": accuracy(idx, p) for g, idx in groups.items()},
            **{f"{g}_lenient_accuracy": accuracy(idx, p, lenient=True) for g, idx in groups.items()},
            "test_predictions": {rows[i]["id"]: p[i] for i in test},
        }
    confusion = {name: dict(Counter(f"{rows[i]['category']}->{p[i]}" for i in test if p[i] != rows[i]["category"]))
                 for name, (p, _) in preds.items()}
    return {
        "rows": {"dev": len(dev), "test": len(test),
                 **{g: len(idx) for g, idx in groups.items()}},
        "phrasing_dev": phrasing_dev,
        "chosen_phrasing": chosen,
        "laya_test": {rows[i]["id"]: results[i] for i in test},
        "designs": designs,
        "test_confusion_errors": confusion,
    }


# ==============================================================================
# Ingest cost
# ==============================================================================


# (device, torch threads, risk mode, documents) combinations timed with --timing.
# The deployed backend is CPU-only on 2 cores, hence threads=2 for the CPU runs;
# CPU union mode is timed on two documents and reported per chunk.
TIMING_RUNS = [
    ("cuda", 0, "union", 0),
    ("cuda", 0, "confirm", 0),
    ("cpu", 2, "confirm", 0),
    ("cpu", 2, "union", 2),
]


def time_ingest(device: str, threads: int, mode: str, limit_docs: int) -> dict:
    """Runs eval.decisions.timing in a fresh process and returns its JSON line."""
    cmd = [sys.executable, "-m", "eval.decisions.timing", "--device", device, "--mode", mode,
           "--threads", str(threads), "--limit-docs", str(limit_docs)]
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=3600)
    lines = [line for line in proc.stdout.splitlines() if line.startswith('{"device"')]
    if proc.returncode or not lines:
        return {"device": device, "mode": mode, "error": (proc.stderr or proc.stdout)[-400:]}
    return json.loads(lines[-1])


# ==============================================================================
# Report
# ==============================================================================


def fmt(s: dict) -> str:
    return f"{s['precision']:.3f} | {s['recall']:.3f} | {s['f1']:.3f} | {s['tp']}/{s['fp']}/{s['fn']}"


def adoption_section(results: dict) -> list[str]:
    """
    Applies the adoption rule to the TEST numbers and states what ships.

    Rule: a Laya design is adopted only if it beats regex_fixed on TEST (F1 for
    risk, precision for PII, accuracy for category) without lowering recall.
    """
    risk, pii, cat = results["risk"]["designs"], results["pii"]["designs"], results["category"]["designs"]
    base_r = risk["regex_fixed"]["test"]["micro"]
    lines = ["", "## Adoption check (TEST, against regex_fixed)", ""]
    for name in ("wired_union", "wired_confirm"):
        s = risk[name]["test"]["micro"]
        ok = s["f1"] > base_r["f1"] and s["recall"] >= base_r["recall"]
        lines.append(f"- risk `{name}`: F1 {base_r['f1']:.3f} -> {s['f1']:.3f}, recall "
                     f"{base_r['recall']:.3f} -> {s['recall']:.3f}: {'PASS' if ok else 'FAIL'}")
    base_p, s = pii["regex_fixed"]["test"], pii["wired_laya_pii"]["test"]
    ok = s["precision"] > base_p["precision"] and s["recall"] >= base_p["recall"]
    lines.append(f"- PII `wired_laya_pii`: precision {base_p['precision']:.3f} -> {s['precision']:.3f}, "
                 f"recall {base_p['recall']:.3f} -> {s['recall']:.3f}, corpus chunks excluded "
                 f"{results['pii']['retrieval_damage']['regex_fixed']['excluded_chunks']} -> "
                 f"{results['pii']['retrieval_damage']['wired_laya_pii']['excluded_chunks']}: "
                 f"{'PASS' if ok else 'FAIL'}")
    b, s = cat["regex_fixed"]["test_all_accuracy"], cat["laya"]["test_all_accuracy"]
    lines.append(f"- category `laya`: accuracy {b:.3f} -> {s:.3f}: {'PASS' if s > b else 'FAIL'}")
    defaults = ", ".join(f"{k}={v}" for k, v in sig._FLAG_DEFAULTS.items())
    lines += ["", f"Shipped defaults (src/decisions/ingest_signals.py): {defaults}, LAYA_RISK_MODE=auto "
              "(union on CUDA, confirm on CPU).", "",
              "Caveats: one annotator; small TEST positive counts (see the rows above), so a "
              "difference of one or two chunks moves F1 by several points; the regex fixes were "
              "written after reading the corpus, so regex_fixed is optimistic on corpus rows "
              "(most visibly for category filenames); synthetic rows were written by the same "
              "agent that wrote the questions."]
    return lines


def write_report(results: dict) -> str:
    risk, pii, cat = results["risk"], results["pii"], results["category"]
    lines = [
        "# Ingest decisions: regex vs Laya",
        "",
        f"Model: `{results['model']}` (no fine-tuning). Legacy detectors: commit `{LEGACY_COMMIT}`. "
        "Thresholds and phrasings chosen on DEV, reported on TEST. "
        "Generated by `python -m eval.decisions.run_ingest_decisions_eval`.",
        "",
        "Datasets (`eval/decisions/data/`, provenance in each file's `_meta`): corpus chunks from "
        "`extract_and_chunk` over `data/sample_deal/*.txt`, split by document (DEV: board deck, "
        "credit agreement, customer contracts, QoE; TEST: the other five), plus hand-written "
        "synthetic hard cases alternating DEV/TEST. One annotator (the implementing agent).",
        "",
        "## Risk signals (chunk x 10 categories)",
        "",
        f"Rows: DEV {risk['rows']['dev']}, TEST {risk['rows']['test']}; "
        f"{risk['borderline_pairs']} borderline (row, category) pairs unscored. "
        f"TEST positives: {risk['positives']['test']}.",
        "",
        "Phrasing search on DEV (Laya alone, best threshold): "
        + ", ".join(f"`{k}` F1 {v['f1']:.3f} @ {v['threshold']}" for k, v in risk["phrasing_dev"].items())
        + f" -> `{risk['chosen_phrasing']}`.",
        "",
        "| design | params | DEV F1 | TEST P | TEST R | TEST F1 | TEST tp/fp/fn |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, d in risk["designs"].items():
        params = ", ".join(f"{k}={v}" for k, v in d["params"].items() if k != "rule")
        lines.append(f"| {name} | {params} | {d['dev']['f1']:.3f} | {fmt(d['test']['micro'])} |")
    lines += ["", "Per-category TEST F1:", "",
              "| category | " + " | ".join(risk["designs"]) + " |",
              "|---|" + "---|" * len(risk["designs"])]
    for c in next(iter(risk["designs"].values()))["test"]["per_category"]:
        cells = []
        for d in risk["designs"].values():
            s = d["test"]["per_category"][c]
            cells.append(f"{s['f1']:.2f} ({s['tp']}/{s['fp']}/{s['fn']})")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## PII (chunk flag; flagged chunks are excluded from retrieval)",
        "",
        f"Rows: DEV {pii['rows']['dev']}, TEST {pii['rows']['test']}; positives DEV "
        f"{pii['positives']['dev']}, TEST {pii['positives']['test']}; {pii['borderline_rows']} "
        "borderline rows unscored.",
        "",
        "Phrasing search on DEV (Laya alone): "
        + ", ".join(f"`{k}` F1 {v['f1']:.3f} @ {v['threshold']}" for k, v in pii["phrasing_dev"].items())
        + f" -> `{pii['chosen_phrasing']}`.",
        "",
        "| design | params | DEV F1 | TEST P | TEST R | TEST F1 | TEST tp/fp/fn | TEST recall by kind |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, d in pii["designs"].items():
        params = ", ".join(f"{k}={v}" for k, v in d["params"].items() if k != "rule")
        kinds = ", ".join(f"{k} {v}" for k, v in d["test_recall_by_kind"].items())
        lines.append(f"| {name} | {params} | {d['dev']['f1']:.3f} | {fmt(d['test'])} | {kinds} |")
    lines += ["", "Retrieval damage over all 112 corpus chunks (both splits):", "",
              "| design | chunks excluded | excluded chunks holding a golden fact | golden facts lost | questions hit |",
              "|---|---|---|---|---|"]
    for name, d in pii["retrieval_damage"].items():
        lines.append(f"| {name} | {d['excluded_chunks']} | {len(d['excluded_with_golden_fact'])} "
                     f"{d['excluded_with_golden_fact'] or ''} | {len(d['golden_facts_lost'])} | "
                     f"{', '.join(d['golden_questions_hit']) or '-'} |")

    lines += [
        "",
        "## Document category",
        "",
        f"Rows: DEV {cat['rows']['dev']} (synthetic), TEST {cat['rows']['test']} "
        f"({cat['rows']['test_corpus_and_fixtures']} corpus + fixtures, {cat['rows']['test_synthetic']} synthetic). "
        "Lenient accuracy accepts any category in the row's `acceptable` list.",
        "",
        "Phrasing search on DEV (accuracy): "
        + ", ".join(f"`{k}` {v:.3f}" for k, v in cat["phrasing_dev"].items())
        + f" -> `{cat['chosen_phrasing']}`.",
        "",
        "| design | params | DEV acc | TEST acc (strict / lenient) | corpus+fixtures | synthetic |",
        "|---|---|---|---|---|---|",
    ]
    for name, d in cat["designs"].items():
        params = ", ".join(f"{k}={v}" for k, v in d["params"].items() if k not in ("rule", "note"))
        lines.append(
            f"| {name} | {params} | {d['dev_accuracy']:.3f} | {d['test_all_accuracy']:.3f} / "
            f"{d['test_all_lenient_accuracy']:.3f} | {d['test_corpus_and_fixtures_accuracy']:.3f} / "
            f"{d['test_corpus_and_fixtures_lenient_accuracy']:.3f} | {d['test_synthetic_accuracy']:.3f} / "
            f"{d['test_synthetic_lenient_accuracy']:.3f} |")
    lines += ["", "TEST errors (gold->predicted):", ""]
    for name, errs in cat["test_confusion_errors"].items():
        lines.append(f"- {name}: " + (", ".join(f"{k} x{v}" for k, v in sorted(errs.items())) or "none"))

    lines += adoption_section(results)

    if results.get("timing"):
        lines += ["", "## Ingest cost (extract_and_chunk, Laya warmed, LAYA_PII off)", "",
                  "Environment-dependent (a shared GPU and a loaded host); not part of the "
                  "deterministic metrics.", "",
                  "| device | threads | risk mode | docs | chunks | Laya off (s) | Laya on (s) | added (s) "
                  "| added per chunk (s) | decisions |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for t in results["timing"]:
            if "error" in t:
                lines.append(f"| {t['device']} | | {t['mode']} | error: {t['error'][-160:]!r} | | | | | | |")
                continue
            per_chunk = round(t["added_s"] / t["chunks"], 3) if t["chunks"] else 0
            lines.append(f"| {t['device']} | {t['threads']} | {t['mode']} | {t['documents']} | {t['chunks']} "
                         f"| {t['off_s']} | {t['on_s']} | {t['added_s']} | {per_chunk} | {t['sources_on']} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--no-cache", action="store_true", help="recompute every Laya answer")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--timing", action="store_true", help="also time ingest on GPU and CPU")
    args = parser.parse_args()

    logging.disable(logging.CRITICAL)
    # The evaluation asks its own questions; the pipeline's Laya path must not run
    # inside extract_and_chunk while the corpus is being rebuilt.
    os.environ["LAYA_RISK"] = os.environ["LAYA_PII"] = os.environ["LAYA_CATEGORY"] = "0"

    scorer = Scorer(None if args.no_cache else args.cache)
    texts = corpus_texts()
    start = time.perf_counter()
    results = {
        "model": laya_model(),
        "legacy_commit": LEGACY_COMMIT,
        "risk": run_risk(scorer, texts),
        "pii": run_pii(scorer, texts),
        "category": run_category(scorer),
    }
    scorer.save()
    results["laya_requests"] = scorer.calls
    results["seconds"] = round(time.perf_counter() - start, 1)
    for key in ("LAYA_RISK", "LAYA_PII", "LAYA_CATEGORY"):
        os.environ.pop(key, None)
    if args.timing:
        results["timing"] = [time_ingest(*run) for run in TIMING_RUNS]
    elif (DECISIONS_DIR / "results.json").exists():
        # Timing is slow and environment-bound: keep the last measured table
        # rather than dropping it from the report on a metrics-only rerun.
        previous = json.loads((DECISIONS_DIR / "results.json").read_text(encoding="utf-8"))
        results["timing"] = previous.get("timing")

    report = write_report(results)
    (DECISIONS_DIR / "results.md").write_text(report, encoding="utf-8")
    stable = {k: v for k, v in results.items() if k not in ("seconds", "laya_requests", "timing")}
    (DECISIONS_DIR / "results.json").write_text(
        json.dumps({**stable, "timing": results.get("timing")}, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
