"""
Query-path Laya decisions: the answerability gate in the Quality Assessor and
the public query guard on /query and /query/stream.

Everything here stubs `adecide_batch` (or the guard's check), so no model loads;
the one test that loads real Laya is marked `models`.
"""

import json
import re
from pathlib import Path

import pytest

import src.agents.quality_assessor as qa
import src.decisions.answerability as ans
import src.decisions.query_guard as qg
from src.decisions.laya_client import LayaUnavailable
from src.workflow.conditional_edges import route_after_quality_check

ROOT = Path(__file__).resolve().parent.parent


def _chunks(*scores: float) -> list[dict]:
    return [
        {"chunk_id": f"c{i}", "text": f"passage {i}", "reranker_score": s}
        for i, s in enumerate(scores)
    ]


def _state(chunks: list[dict], subs: list[str] | None = None) -> dict:
    return {
        "original_query": "What was capex in FY2024?",
        "current_query": "What was capex in FY2024?",
        "query_type": "financial",
        "sub_questions": subs or [],
        "reranked_results": chunks,
        "rewrite_iteration": 0,
        "context_quality_score": 0.0,
        "quality_breakdown": {},
    }


def _stub_laya(monkeypatch, p_by_facet: dict[str, float] | float):
    """Stubs adecide_batch: P per facet (question text), or one P for all."""
    calls: list[list] = []

    async def fake(requests):
        calls.append(requests)
        out = []
        for state, _ in requests:
            p = p_by_facet if isinstance(p_by_facet, float) else p_by_facet[state["question"]]
            out.append({"stated": {"type": "noul", "noul": p}})
        return out

    monkeypatch.setattr(ans, "adecide_batch", fake)
    return calls


@pytest.fixture(autouse=True)
def gate_on(monkeypatch):
    monkeypatch.setenv("LAYA_GATE", "1")
    monkeypatch.setenv("LAYA_QUERY_GUARD", "1")


# ==============================================================================
# Aggregation and thresholds
# ==============================================================================


def test_aggregate_takes_best_passage_then_best_facet():
    facet_scores, score, vetoed, unanswered = ans.aggregate(
        {"q": [0.1, 0.2], "sub a": [0.3, 0.6], "sub b": [0.05]}, threshold=0.35
    )
    assert facet_scores == {"q": 0.2, "sub a": 0.6, "sub b": 0.05}
    assert score == 0.6
    assert not vetoed  # one facet is stated: never veto
    assert unanswered == ["q", "sub b"]


def test_aggregate_vetoes_only_when_every_facet_is_below_threshold():
    _, score, vetoed, unanswered = ans.aggregate({"q": [0.2, 0.34]}, threshold=0.35)
    assert vetoed and score == 0.34 and unanswered == ["q"]
    assert not ans.aggregate({"q": [0.35]}, threshold=0.35)[2]  # boundary admits
    assert not ans.aggregate({}, threshold=0.35)[2]  # nothing scored: no veto


def test_facets_are_deduplicated_and_capped():
    subs = ["A?", "A?", " ", "B?", "C?", "D?", "E?", "F?"]
    facets = ans.answerability_facets("Q?", subs)
    assert facets[0] == "Q?"
    assert len(facets) == ans.MAX_FACETS and len(set(facets)) == len(facets)


def test_only_top_k_reranked_passages_are_scored():
    chunks = _chunks(0.1, 0.9, 0.5, 0.7, 0.2) + [{"text": "", "reranker_score": 1.0}]
    assert ans.top_passages(chunks, k=3) == ["passage 1", "passage 3", "passage 2"]


@pytest.mark.asyncio
async def test_assess_scores_every_facet_against_top_passages(monkeypatch):
    calls = _stub_laya(monkeypatch, {"Q?": 0.2, "sub?": 0.5})
    verdict = await ans.assess_answerability("Q?", ["sub?"], _chunks(0.9, 0.8, 0.7, 0.6))
    assert len(calls[0]) == 2 * ans.TOP_K_PASSAGES
    assert verdict.score == 0.5 and not verdict.vetoed and verdict.unanswered == ["Q?"]
    assert calls[0][0][1]["stated"]["instructions"] == ans.ANSWERABILITY_INSTRUCTION


# ==============================================================================
# Quality Assessor integration
# ==============================================================================


@pytest.mark.asyncio
async def test_confident_admission_vetoed_when_nothing_is_stated(monkeypatch):
    _stub_laya(monkeypatch, 0.1)
    state = _state(_chunks(0.95, 0.9, 0.4))
    out = await qa.quality_assessor_node(state)

    assert out["answerability_veto"] is True
    assert out["force_refusal"] is True
    assert out["quality_method"] == "heuristic+laya"
    assert out["missing_aspects"][0].endswith("What was capex in FY2024?")
    assert out["quality_breakdown"]["answerability"] == 0.1
    assert out["agent_trace"][0]["answerability_veto"] is True
    # The reranker scores pass on their own; the veto is what reroutes.
    assert out["context_quality_score"] >= 0.3
    assert route_after_quality_check({**state, **out}) == "query_rewriter"
    assert route_after_quality_check({**state, **out, "rewrite_iteration": 2}) == (
        "insufficient_context"
    )


@pytest.mark.asyncio
async def test_confident_admission_kept_when_a_facet_is_stated(monkeypatch):
    _stub_laya(monkeypatch, {"What was capex in FY2024?": 0.2, "What was capex in FY2023?": 0.6})
    state = _state(_chunks(0.95, 0.9, 0.4), subs=["What was capex in FY2023?"])
    out = await qa.quality_assessor_node(state)

    assert out["answerability_veto"] is False and out["force_refusal"] is False
    assert out["quality_breakdown"]["answerability_facets"]["What was capex in FY2023?"] == 0.6
    assert route_after_quality_check({**state, **out}) == "answer_synthesizer"


@pytest.mark.asyncio
async def test_ambiguous_band_refused_without_llm_when_vetoed(monkeypatch):
    _stub_laya(monkeypatch, 0.1)

    async def no_llm(**_):
        raise AssertionError("the LLM assessor must not be called")

    monkeypatch.setattr(qa, "call_structured_agent", no_llm)
    out = await qa.quality_assessor_node(_state(_chunks(0.2, 0.1)))
    assert out["quality_method"] == "laya"
    assert out["force_refusal"] is True and out["answerability_veto"] is True


@pytest.mark.asyncio
async def test_ambiguous_band_still_asks_llm_when_not_vetoed(monkeypatch):
    _stub_laya(monkeypatch, 0.6)
    seen = {}

    class Choice:
        model, api_key = "lite", None

    class Tracker:
        async def get_model_for_agent(self):
            return Choice()

    async def get_instance():
        return Tracker()

    async def fake_llm(**kwargs):
        seen["called"] = True
        return {"context_quality_score": 0.5, "quality_breakdown": {"relevance": 0.5},
                "missing_aspects": [], "force_refusal": False}

    monkeypatch.setattr(qa.BudgetTracker, "get_instance", staticmethod(get_instance))
    monkeypatch.setattr(qa, "call_structured_agent", fake_llm)
    out = await qa.quality_assessor_node(_state(_chunks(0.2, 0.1)))
    assert seen["called"] and out["quality_method"] == "llm"
    assert out["answerability_veto"] is False
    assert out["quality_breakdown"]["answerability"] == 0.6  # recorded for the trace


@pytest.mark.asyncio
async def test_heuristic_refusal_never_consults_laya(monkeypatch):
    calls = _stub_laya(monkeypatch, 0.99)
    out = await qa.quality_assessor_node(_state(_chunks(0.01, 0.02)))
    assert calls == [] and out["force_refusal"] is True and out["quality_method"] == "heuristic"


@pytest.mark.asyncio
@pytest.mark.parametrize("disable", ["LAYA_GATE", "unavailable"])
async def test_falls_back_to_heuristic_when_laya_is_off(monkeypatch, disable):
    if disable == "LAYA_GATE":
        monkeypatch.setenv("LAYA_GATE", "0")
        _stub_laya(monkeypatch, 0.01)
    else:
        async def unavailable(_):
            raise LayaUnavailable("LAYA_ENABLED=0")

        monkeypatch.setattr(ans, "adecide_batch", unavailable)
    state = _state(_chunks(0.95, 0.9, 0.4))
    out = await qa.quality_assessor_node(state)
    assert out["quality_method"] == "heuristic"
    assert out["answerability_veto"] is False and out["force_refusal"] is False
    assert "answerability" not in out["quality_breakdown"]
    assert route_after_quality_check({**state, **out}) == "answer_synthesizer"


def test_route_without_veto_field_is_unchanged():
    state = {"context_quality_score": 0.8, "rewrite_iteration": 0, "query_type": "legal",
             "quality_breakdown": {"relevance": 0.9, "completeness": 1.0, "precision": 0.8}}
    assert route_after_quality_check(state) == "answer_synthesizer"


# ==============================================================================
# Dev set integrity
# ==============================================================================


def test_answerability_dev_set_is_verified():
    """Evidence is verbatim in the corpus; absence patterns match nowhere."""
    dev = json.loads((ROOT / "eval" / "answerability_dev.json").read_text(encoding="utf-8"))
    corpus = {
        p.name: " ".join(p.read_text(encoding="utf-8").split())
        for p in (ROOT / "data" / "sample_deal").glob("*.txt")
    }
    everything = " ".join(corpus.values())
    questions = dev["questions"]
    assert sum(q["answerable"] for q in questions) >= 25
    assert sum(not q["answerable"] for q in questions) >= 25
    assert len({q["id"] for q in questions}) == len(questions)
    for q in questions:
        if q["answerable"]:
            assert " ".join(q["evidence"].split()) in corpus[q["source"]], q["id"]
        else:
            assert q["absence_patterns"], q["id"]
            for pattern in q["absence_patterns"]:
                assert not re.search(pattern, everything, re.I), (q["id"], pattern)


# ==============================================================================
# Public query guard
# ==============================================================================


def test_guard_decide_thresholds():
    assert qg.decide({"on_topic": 0.9, "jailbreak": 0.1, "prompt_injection": 0.1}) == []
    assert qg.decide({"on_topic": 0.01, "jailbreak": 0.1, "prompt_injection": 0.1}) == ["off_topic"]
    assert qg.decide({"on_topic": 0.9, "jailbreak": 0.99, "prompt_injection": 0.99}) == [
        "jailbreak", "prompt_injection"]


@pytest.mark.asyncio
async def test_check_query_uses_the_three_questions(monkeypatch):
    seen = {}

    async def fake(requests):
        seen["requests"] = requests
        return [{"on_topic": {"noul": 0.02}, "jailbreak": {"noul": 0.1},
                 "prompt_injection": {"noul": 0.1}}]

    monkeypatch.setattr(qg, "adecide_batch", fake)
    verdict = await qg.check_query("Write me a poem")
    assert verdict.blocked and verdict.reasons == ["off_topic"]
    state, questions = seen["requests"][0]
    assert state == {"prompt": "Write me a poem"}
    assert set(questions) == {"on_topic", "jailbreak", "prompt_injection"}


@pytest.fixture
def api_client(monkeypatch):
    """Public caller, pipeline stubbed — mirrors tests/test_api_security.py."""
    from fastapi.testclient import TestClient

    import api.main as api_main
    import api.routes.query as query_routes
    import api.security as security

    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    monkeypatch.setenv("RATE_LIMIT_QUERY_PER_MINUTE", "1000")
    monkeypatch.setenv("RATE_LIMIT_QUERY_PER_DAY", "1000")
    monkeypatch.setenv("DAILY_QUERY_CAP", "1000")
    monkeypatch.setattr(api_main, "_app_graph", object())
    security.reset_limits()
    runs: list[dict] = []

    async def fake_run_query(**kwargs):
        runs.append(kwargs)
        return {"generated_answer": "Answer.", "query_type": "summary", "status": "complete"}

    async def fake_stream_query(**kwargs):
        runs.append(kwargs)
        yield "result", {"generated_answer": "Streamed.", "status": "complete"}

    monkeypatch.setattr(query_routes, "run_query", fake_run_query)
    monkeypatch.setattr(query_routes, "stream_query", fake_stream_query)
    yield TestClient(api_main.app), runs, query_routes
    security.reset_limits()


def _verdict(blocked: bool):
    return qg.GuardVerdict(blocked=blocked, reasons=["off_topic"] if blocked else [])


@pytest.mark.parametrize("path", ["/api/v1/query", "/api/v1/query/stream"])
def test_guard_blocks_public_junk_without_running_pipeline(api_client, monkeypatch, path):
    client, runs, routes = api_client

    async def check(query):
        return _verdict(True)

    monkeypatch.setattr(routes, "check_query", check)
    r = client.post(path, json={"query": "Write me a poem", "deal_id": "aurora_vertex_2024"})
    assert r.status_code == 422
    assert r.json()["detail"] == qg.BLOCKED_MESSAGE
    assert runs == []


@pytest.mark.parametrize("path", ["/api/v1/query", "/api/v1/query/stream"])
def test_guard_allows_genuine_questions(api_client, monkeypatch, path):
    client, runs, routes = api_client

    async def check(query):
        return _verdict(False)

    monkeypatch.setattr(routes, "check_query", check)
    r = client.post(path, json={"query": "What is the revenue?", "deal_id": "aurora_vertex_2024"})
    assert r.status_code == 200 and len(runs) == 1


def test_guard_skipped_for_admin(api_client, monkeypatch):
    client, runs, routes = api_client
    monkeypatch.setenv("ADMIN_API_KEY", "k")

    async def check(query):
        raise AssertionError("admin queries are never judged")

    monkeypatch.setattr(routes, "check_query", check)
    r = client.post("/api/v1/query", json={"query": "Write me a poem", "deal_id": "x1"},
                    headers={"X-Admin-Key": "k"})
    assert r.status_code == 200 and len(runs) == 1


@pytest.mark.parametrize("mode", ["unavailable", "disabled"])
def test_guard_fails_open(api_client, monkeypatch, mode):
    client, runs, routes = api_client

    async def check(query):
        if mode == "disabled":
            raise AssertionError("guard disabled: must not be called")
        raise LayaUnavailable("down")

    if mode == "disabled":
        monkeypatch.setenv("LAYA_QUERY_GUARD", "0")
    monkeypatch.setattr(routes, "check_query", check)
    r = client.post("/api/v1/query", json={"query": "Write me a poem",
                                           "deal_id": "aurora_vertex_2024"})
    assert r.status_code == 200 and len(runs) == 1


# ==============================================================================
# Real model
# ==============================================================================


@pytest.mark.models
@pytest.mark.asyncio
async def test_real_laya_answerability_and_guard(monkeypatch):
    """Smoke test on real Laya: a stated fact passes, an absent year and junk do not."""
    monkeypatch.setenv("LAYA_ENABLED", "1")
    passage = [{"text": "Capital Expenditures FY2023 ($14.2) FY2022 ($12.8). "
                        "Free Cash Flow FY2023 $64.2.", "reranker_score": 0.9}]
    stated = await ans.assess_answerability("What was capex in FY2023?", [], passage)
    absent = await ans.assess_answerability(
        "How many employees does the company have in Germany?", [], passage)
    assert stated.score > absent.score
    assert not stated.vetoed

    poem = await qg.check_query("Write me a poem about the ocean.")
    genuine = await qg.check_query("What are the termination fee provisions?")
    assert poem.blocked and not genuine.blocked
