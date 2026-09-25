"""
Ingest-path Laya decisions: risk signals, PII and document category.

Every test except the last class stubs `decide_batch` (as imported into
src.decisions.ingest_signals), so no model loads. The regex tests pin the
known-bug fixes made alongside (negations and keyword collisions are Laya's
job; these are the patterns that could never match or matched the wrong word).
The `models` class loads the real checkpoint.
"""

from __future__ import annotations

import pytest

import src.data_processing.ingest_pipeline as pipeline
import src.decisions.ingest_signals as sig
from src.data_processing.document_classifier import DocumentClassifier
from src.data_processing.pii_detector import PIIDetector
from src.data_processing.risk_signal_extractor import RISK_PATTERNS, RiskSignalExtractor
from src.decisions.laya_client import LayaUnavailable

DEAL = "deal-decisions"


@pytest.fixture(autouse=True)
def _laya_env(monkeypatch):
    """Pins every flag so a developer's .env cannot change what is tested."""
    for name in ("LAYA_RISK", "LAYA_PII", "LAYA_CATEGORY", "LAYA_RISK_MODE", "LAYA_DEVICE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LAYA_ENABLED", "1")


def _stub_decide(monkeypatch, risk_p=None, pii_p=0.0, category=None):
    """
    Stubs decide_batch. risk_p maps a phrase in the passage to {category: P};
    unlisted categories get 0.05. Returns the list of calls (each a request list).
    """
    calls: list[list] = []
    risk_p = risk_p or {}

    def fake(requests):
        calls.append(requests)
        out = []
        for state, questions in requests:
            answer = {}
            for name in questions:
                kind, _, key = name.partition(":")
                if kind == "risk":
                    p = 0.05
                    for phrase, probs in risk_p.items():
                        if phrase in state["passage"] and key in probs:
                            p = probs[key]
                    answer[name] = {"type": "noul", "noul": p}
                elif kind == "pii":
                    answer[name] = {"type": "noul", "noul": pii_p}
                elif name == "category":
                    choice, conf = category
                    answer[name] = {"type": "choice", "choice": choice,
                                    "probabilities": {choice: conf}, "confidence": conf}
            out.append(answer)
        return out

    monkeypatch.setattr(sig, "decide_batch", fake)
    return calls


def _unavailable(monkeypatch):
    def fail(requests):
        raise LayaUnavailable("stubbed outage")

    monkeypatch.setattr(sig, "decide_batch", fail)


# ==============================================================================
# Question definitions
# ==============================================================================


def test_risk_descriptions_cover_every_pattern_family():
    assert set(sig.RISK_DESCRIPTIONS) == set(RISK_PATTERNS)


def test_questions_reference_the_passage_field():
    for q in {**sig.risk_questions(), **sig.pii_questions()}.values():
        assert "`passage`" in q["instructions"]
    assert set(sig.category_question()["category"]["criteria"]) == set(sig.CATEGORY_CRITERIA)


def test_flag_defaults_match_the_evaluation_outcome():
    assert sig.laya_risk_enabled() is True
    assert sig.laya_category_enabled() is True
    assert sig.laya_pii_enabled() is False  # lost precision on TEST — not adopted


@pytest.mark.parametrize("env,device,expected", [
    ("union", "cpu", "union"),
    ("confirm", "cuda", "confirm"),
    ("auto", "cuda", "union"),
    ("auto", "cpu", "confirm"),
    ("bogus", "cpu", "confirm"),
])
def test_risk_mode(monkeypatch, env, device, expected):
    monkeypatch.setenv("LAYA_RISK_MODE", env)
    monkeypatch.setenv("LAYA_DEVICE", device)
    assert sig.risk_mode() == expected


# ==============================================================================
# Regex fixes (the fallback path)
# ==============================================================================


class TestRegexFixes:
    extractor = RiskSignalExtractor()

    def signals(self, text):
        return self.extractor.extract(text).signals

    def test_mac_is_case_sensitive(self):
        assert "material_adverse_change" not in self.signals("Staff use Mac laptops running macOS.")
        assert "material_adverse_change" in self.signals("Buyer may terminate if a MAC occurs.")

    def test_by_default_is_not_distress(self):
        assert "financial_distress" not in self.signals("By default, data is kept for 90 days.")
        assert "financial_distress" in self.signals("Lenders issued a notice of default in May.")

    def test_patent_expiry_matches(self):
        assert "ip_risk" in self.signals("Patent expiry: the core patent expires in 2025.")

    def test_customer_share_of_revenue_matches(self):
        text = "Customer concentration aside, the largest customer represents 38% of revenue."
        assert "customer_concentration" in self.signals(text)
        text = "The single largest customer, Northstar, represents 12.0% of total revenue."
        assert "customer_concentration" in self.signals(text)

    def test_top_n_customer_contracts_is_not_concentration(self):
        assert "customer_concentration" not in self.signals(
            "The top 5 customer contracts were renegotiated at higher prices."
        )

    def test_invoice_number_is_not_an_ssn(self):
        result = PIIDetector().detect("Transaction Bonus 2024 - Invoice 123456789 - net 30.")
        assert result.contains_pii == 0

    def test_labelled_and_dashed_ssns_still_match(self):
        assert "ssn" in PIIDetector().detect("SSN 123-45-6789").pii_types
        assert "ssn" in PIIDetector().detect("Social Security Number: 123456789").pii_types

    def test_salary_requires_an_amount(self):
        assert "salary_data" not in PIIDetector().detect("Transaction Bonus 2024").pii_types
        assert "salary_data" in PIIDetector().detect("Salary: $142,500").pii_types

    @pytest.mark.parametrize("name,expected", [
        ("quality_of_earnings_report_fy2023.txt", "financial"),
        ("employment_and_retention_agreements.txt", "legal"),
        ("credit_agreement_summary.txt", "legal"),
        ("ip_portfolio_and_litigation_schedule.txt", "legal"),
        ("aurora_financials_fy2023.txt", "financial"),
        ("IT_infrastructure.pdf", "operational"),
    ])
    def test_filename_patterns_are_anchored(self, name, expected):
        assert DocumentClassifier()._classify_by_filename(name) == expected


# ==============================================================================
# Combination rules
# ==============================================================================


class TestRiskHybrid:
    extractor = RiskSignalExtractor()

    def test_regex_match_confirmed_by_laya(self):
        base = self.extractor.extract("There is pending litigation against the Company.")
        result = self.extractor.apply_laya(base, {"litigation": sig.RISK_CONFIRM_THRESHOLD})
        (detail,) = result.signal_details
        assert result.signals == ["litigation"]
        assert detail["source"] == "regex+laya"
        assert detail["confidence"] == sig.RISK_CONFIRM_THRESHOLD
        assert detail["sample_matches"]  # regex evidence is kept

    def test_negated_regex_match_is_rejected_not_dropped_silently(self):
        base = self.extractor.extract("The Company is not a party to any pending litigation.")
        assert base.signals == ["litigation"]
        result = self.extractor.apply_laya(base, {"litigation": 0.1})
        assert result.signals == []
        assert result.rejected[0]["signal_type"] == "litigation"
        assert result.rejected[0]["confidence"] == 0.1

    def test_laya_alone_needs_the_high_threshold(self):
        base = self.extractor.extract("The largest customer buys most of our output.")
        below = self.extractor.apply_laya(base, {"customer_concentration": sig.RISK_THRESHOLD - 0.01})
        at = self.extractor.apply_laya(base, {"customer_concentration": sig.RISK_THRESHOLD})
        assert below.signals == []
        assert at.signals == ["customer_concentration"]
        assert at.signal_details[0] == {
            "signal_type": "customer_concentration", "match_count": 1, "sample_matches": [],
            "source": "laya", "confidence": sig.RISK_THRESHOLD,
        }

    def test_unasked_category_keeps_its_regex_signal(self):
        base = self.extractor.extract("Seller shall indemnify Buyer.")
        result = self.extractor.apply_laya(base, {})
        assert result.signals == ["indemnification"]
        assert result.signal_details[0]["source"] == "regex"

    def test_extract_accepts_scores_directly(self):
        text = "No pending litigation. Seller shall indemnify Buyer."
        result = self.extractor.extract(text, laya_scores={"litigation": 0.1, "indemnification": 0.9})
        assert result.signals == ["indemnification"]


class TestPiiHybrid:
    detector = PIIDetector()

    def test_regex_only_when_no_probability(self):
        result = self.detector.detect("Salary: $142,500")
        assert result.contains_pii == 1
        assert result.source == "regex"
        assert result.laya_probability is None

    def test_strong_identifier_survives_a_low_probability(self):
        result = self.detector.detect("Employee SSN 123-45-6789", laya_probability=0.0)
        assert result.contains_pii == 1
        assert result.source == "regex+laya"

    def test_weak_hit_needs_confirmation(self):
        text = "Loan account number 9876543210 relates to the revolver."
        assert self.detector.detect(text).contains_pii == 1
        assert self.detector.detect(text, laya_probability=0.1).contains_pii == 0
        assert self.detector.detect(text, laya_probability=sig.PII_CONFIRM_THRESHOLD).contains_pii == 1

    def test_laya_alone_can_flag(self):
        text = "Elena Marsh  Chief Executive Officer  625,000  437,500"
        assert self.detector.detect(text).contains_pii == 0
        assert self.detector.detect(text, laya_probability=sig.PII_THRESHOLD).contains_pii == 1


# ==============================================================================
# Batching
# ==============================================================================


class TestScoreChunks:
    def test_one_request_per_chunk_with_all_questions(self, monkeypatch):
        calls = _stub_decide(monkeypatch)
        scores = sig.score_chunks(["a", "b"], risk_categories=[None, None], pii=True)
        requests = [r for call in calls for r in call]
        assert len(requests) == 2
        assert len(requests[0][1]) == len(sig.RISK_DESCRIPTIONS) + 1
        assert set(scores[0].risk) == set(sig.RISK_DESCRIPTIONS)
        assert scores[0].pii_probability == 0.0

    def test_confirm_mode_asks_only_regex_categories(self, monkeypatch):
        calls = _stub_decide(monkeypatch)
        scores = sig.score_chunks(["a", "b", "c"], risk_categories=[["litigation"], [], None])
        sent = [r for call in calls for r in call]
        assert [set(q) for _, q in sent] == [
            {"risk:litigation"}, {f"risk:{c}" for c in sig.RISK_DESCRIPTIONS},
        ]
        assert scores[1].risk == {} and scores[1].pii_probability is None

    def test_nothing_to_ask_makes_no_call(self, monkeypatch):
        calls = _stub_decide(monkeypatch)
        assert len(sig.score_chunks(["a"], risk_categories=[[]])) == 1
        assert calls == []

    def test_calls_are_bounded_by_rows(self, monkeypatch):
        calls = _stub_decide(monkeypatch)
        n = 20
        sig.score_chunks(["x"] * n, risk_categories=[None] * n, pii=True)
        per_request = len(sig.RISK_DESCRIPTIONS) + 1
        for call in calls:
            assert len(call) * per_request <= max(sig.MAX_ROWS_PER_CALL, per_request)
        assert sum(len(c) for c in calls) == n

    def test_outage_raises_for_the_caller_to_handle(self, monkeypatch):
        _unavailable(monkeypatch)
        with pytest.raises(LayaUnavailable):
            sig.score_chunks(["a"], risk_categories=[None])


# ==============================================================================
# Pipeline wiring
# ==============================================================================

DOC = """\
LITIGATION
The Company is not a party to any pending or threatened litigation.

CUSTOMERS
Our largest customer buys most of what we make, far ahead of the rest.

INDEMNIFICATION
Seller shall indemnify and hold harmless Buyer from any Losses arising from taxes.
"""


def _write(tmp_path, text=DOC, name="schedule.txt"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _by_heading(doc, heading):
    return next(c for c in doc.chunks if heading in c["text"])


class TestPipeline:
    RISK_P = {
        "not a party": {"litigation": 0.1},
        "largest customer": {"customer_concentration": 0.9},
        "indemnify": {"indemnification": 0.8},
    }

    def test_union_mode_payload(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LAYA_RISK_MODE", "union")
        _stub_decide(monkeypatch, self.RISK_P, category=("legal", 0.7))
        path = _write(tmp_path)
        doc = pipeline.extract_and_chunk(str(path), path.name, DEAL)

        assert doc.document_category == "legal"
        assert (doc.category_source, doc.category_confidence) == ("laya", 0.7)

        lit = _by_heading(doc, "LITIGATION")
        assert lit["risk_signals"] == []
        assert lit["risk_decisions"]["litigation"] == {
            "source": "regex", "confidence": 0.1, "accepted": False,
        }
        cust = _by_heading(doc, "CUSTOMERS")
        assert cust["risk_signals"] == ["customer_concentration"]
        assert cust["risk_decisions"]["customer_concentration"]["source"] == "laya"
        ind = _by_heading(doc, "INDEMNIFICATION")
        assert ind["risk_signals"] == ["indemnification"]
        assert ind["risk_decisions"]["indemnification"]["source"] == "regex+laya"

        # PII stays regex-only by default, but the fields are always present.
        assert all(c["pii_source"] == "regex" and c["pii_confidence"] is None for c in doc.chunks)

        agg = {s["signal_type"]: s for s in doc.risk_signals}
        assert set(agg) == {"customer_concentration", "indemnification"}
        assert agg["customer_concentration"]["source"] == "laya"
        assert agg["customer_concentration"]["confidence"] == 0.9
        assert agg["indemnification"]["sample_matches"]  # regex evidence survives

    def test_confirm_mode_cannot_add_signals(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LAYA_RISK_MODE", "confirm")
        calls = _stub_decide(monkeypatch, self.RISK_P, category=("legal", 0.7))
        path = _write(tmp_path)
        doc = pipeline.extract_and_chunk(str(path), path.name, DEAL)
        assert _by_heading(doc, "CUSTOMERS")["risk_signals"] == []
        assert _by_heading(doc, "INDEMNIFICATION")["risk_signals"] == ["indemnification"]
        asked = {name for call in calls for _, qs in call for name in qs if name.startswith("risk:")}
        assert asked <= {"risk:litigation", "risk:indemnification"}

    def test_pii_joins_the_same_request_when_enabled(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LAYA_RISK_MODE", "union")
        monkeypatch.setenv("LAYA_PII", "1")
        calls = _stub_decide(monkeypatch, self.RISK_P, pii_p=0.9, category=("legal", 0.7))
        path = _write(tmp_path)
        doc = pipeline.extract_and_chunk(str(path), path.name, DEAL)
        chunk_requests = [qs for call in calls for _, qs in call if "category" not in qs]
        assert all("pii:pii" in qs and "risk:litigation" in qs for qs in chunk_requests)
        assert len(chunk_requests) == len(doc.chunks)
        assert all(c["contains_pii"] == 1 and c["pii_source"] == "regex+laya" for c in doc.chunks)
        assert all(p["contains_pii"] == 1 for p in doc.parents)

    def test_outage_falls_back_to_regex(self, monkeypatch, tmp_path):
        _unavailable(monkeypatch)
        path = _write(tmp_path)
        doc = pipeline.extract_and_chunk(str(path), path.name, DEAL)
        assert doc.category_source == "rules"
        assert _by_heading(doc, "LITIGATION")["risk_signals"] == ["litigation"]
        decision = _by_heading(doc, "LITIGATION")["risk_decisions"]["litigation"]
        assert decision == {"source": "regex", "confidence": None, "accepted": True}

    def test_disabled_laya_is_never_called(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LAYA_ENABLED", "0")
        calls = _stub_decide(monkeypatch, self.RISK_P, category=("legal", 0.7))
        path = _write(tmp_path)
        doc = pipeline.extract_and_chunk(str(path), path.name, DEAL)
        assert calls == []
        assert doc.category_source == "rules"
        assert _by_heading(doc, "LITIGATION")["risk_signals"] == ["litigation"]

    def test_per_task_flags(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LAYA_RISK", "0")
        monkeypatch.setenv("LAYA_CATEGORY", "0")
        calls = _stub_decide(monkeypatch, self.RISK_P, category=("legal", 0.7))
        path = _write(tmp_path)
        doc = pipeline.extract_and_chunk(str(path), path.name, DEAL)
        assert calls == []
        assert doc.category_source == "rules"

    def test_category_override_skips_classification(self, monkeypatch, tmp_path):
        calls = _stub_decide(monkeypatch, category=("board", 0.9))
        path = _write(tmp_path)
        doc = pipeline.extract_and_chunk(str(path), path.name, DEAL, document_category="audit")
        assert doc.document_category == "audit"
        assert doc.category_source == "override"
        assert not any("category" in qs for call in calls for _, qs in call)

    def test_chunk_payload_stays_readable_by_the_dashboard(self, monkeypatch, tmp_path):
        """deals.py rebuilds records from chunk payloads: risk_signals must stay list[str]."""
        _stub_decide(monkeypatch, self.RISK_P, category=("legal", 0.7))
        path = _write(tmp_path)
        doc = pipeline.extract_and_chunk(str(path), path.name, DEAL)
        for c in doc.chunks:
            assert all(isinstance(s, str) for s in c["risk_signals"])
        for s in doc.risk_signals:
            assert {"signal_type", "match_count", "sample_matches", "page_number",
                    "source", "confidence"} <= set(s)


class TestCategory:
    def test_laya_choice(self, monkeypatch):
        _stub_decide(monkeypatch, category=("audit", 0.61))
        got = DocumentClassifier().classify_with_source("doc_0147.pdf", "pdf", "INDEPENDENT AUDITOR'S REPORT")
        assert got == ("audit", "laya", 0.61)

    def test_no_sample_uses_rules(self, monkeypatch):
        calls = _stub_decide(monkeypatch, category=("audit", 0.61))
        got = DocumentClassifier().classify_with_source("income_model.xlsx", "xlsx", "")
        assert got == ("financial", "rules", None)
        assert calls == []

    def test_outage_uses_rules(self, monkeypatch):
        _unavailable(monkeypatch)
        got = DocumentClassifier().classify_with_source("board_deck.pptx", "pptx", "Agenda")
        assert got == ("board", "rules", None)

    def test_unknown_choice_is_treated_as_unavailable(self, monkeypatch):
        _stub_decide(monkeypatch, category=("marketing", 0.9))
        got = DocumentClassifier().classify_with_source("board_deck.pptx", "pptx", "Agenda")
        assert got == ("board", "rules", None)

    def test_filename_only_in_state_when_the_phrasing_uses_it(self):
        assert "filename" in sig.category_state("a.pdf", "text", "kind")
        assert "filename" not in sig.category_state("a.pdf", "text", "kind_content")


# ==============================================================================
# Real model
# ==============================================================================


@pytest.mark.models
class TestRealLaya:
    """Loads the typed-decisions checkpoint; cases have wide margins in the eval."""

    @pytest.fixture(autouse=True)
    def _need_laya(self):
        try:
            sig.score_chunks(["probe"], risk_categories=[["litigation"]])
        except LayaUnavailable as e:
            pytest.skip(f"Laya unavailable: {e}")

    def test_negated_litigation_is_rejected_and_a_real_suit_is_kept(self):
        extractor = RiskSignalExtractor()
        texts = [
            # Not every negation clears the bar: the longer "not a party to any pending
            # or, to its knowledge, threatened litigation, arbitration or governmental
            # proceeding" scores ~0.39 and is still flagged (confirm threshold 0.375).
            "The Company is not subject to any pending litigation.",
            "On March 3, 2024, Helix Corp filed suit against the Company in the U.S. District "
            "Court for the District of Delaware alleging breach of a supply agreement and "
            "seeking damages of $12 million.",
        ]
        base = [extractor.extract(t) for t in texts]
        scores = sig.score_chunks(texts, risk_categories=[None, None])  # union mode
        negated, suit = (extractor.apply_laya(b, s.risk) for b, s in zip(base, scores))
        assert "litigation" in base[0].signals and "litigation" not in negated.signals
        assert "litigation" not in base[1].signals  # "filed suit" escapes the regex
        assert "litigation" in suit.signals

    def test_quality_of_earnings_opening_is_financial(self):
        opening = (
            "AURORA TECHNOLOGIES INC.\nQUALITY OF EARNINGS REPORT\nPrepared for: Vertex "
            "Capital Partners LLC\nPrepared by: Grant Thornton LLP, Transaction Advisory "
            "Services\nThis report presents a quality of earnings analysis, including "
            "normalization of reported EBITDA and a working capital peg."
        )
        category, confidence = sig.classify_document("quality_of_earnings_report_fy2023.txt", opening)
        assert category == "financial"
        assert 0.0 < confidence <= 1.0
