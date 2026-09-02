"""
Tests for the configurable embedding and reranker model selection.

The reranker is overridable because the default does not fit a CPU-only host —
measured on 2 vCPU, bge-reranker-v2-m3 scores 40 passages in 111s against 3.2s
for a MiniLM cross-encoder. These tests pin the two properties that make that
override safe to rely on:

  1. The defaults are exactly the models every number in RESULTS.md was measured
     with, so an unset environment reproduces the recorded behaviour.
  2. The override reaches the model constructor, rather than being read into a
     module constant that nothing consumes.

The second is the one worth a test. A config value that is parsed but never
passed through fails silently and in the most expensive way possible: the
deployment quietly runs the slow model it was configured to avoid.
"""

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest


def _reload_reranker(monkeypatch, **env):
    """
    Reimports the reranker module with the given environment.

    The model names are read at import time, so `monkeypatch.setenv` alone has
    no effect on an already-imported module — the reload is what makes the new
    environment visible.
    """
    for key, value in env.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)

    sys.modules.pop("src.vector_db.reranker", None)
    return importlib.import_module("src.vector_db.reranker")


@pytest.fixture(autouse=True)
def _restore_module():
    """Leaves the module in its default state for every other test in the run."""
    yield
    sys.modules.pop("src.vector_db.reranker", None)
    importlib.import_module("src.vector_db.reranker")


class TestDefaults:
    """An unset environment must reproduce the measured configuration."""

    def test_defaults_are_the_measured_models(self, monkeypatch):
        mod = _reload_reranker(
            monkeypatch,
            EMBEDDING_MODEL=None,
            RERANKER_MODEL=None,
            RERANKER_MAX_LENGTH=None,
        )

        assert mod.EMBEDDING_MODEL_NAME == "BAAI/bge-m3"
        assert mod.RERANKER_MODEL_NAME == "BAAI/bge-reranker-v2-m3"
        assert mod.RERANKER_MAX_LENGTH == 1024

    def test_embedding_dimension_still_matches_the_collection(self, monkeypatch):
        """
        The default embedding model must keep producing 1024-dim vectors.

        VECTOR_SIZE is fixed at collection creation and cannot be changed
        without a full re-index, so a default that drifts away from bge-m3
        would not raise — it would write vectors Qdrant rejects at upsert.
        """
        from src.vector_db.constants import VECTOR_SIZE

        mod = _reload_reranker(monkeypatch, EMBEDDING_MODEL=None)

        assert VECTOR_SIZE == 1024
        assert mod.EMBEDDING_MODEL_NAME == "BAAI/bge-m3"


class TestOverride:
    """The override must reach the constructor, not just the module namespace."""

    def test_reranker_override_is_passed_to_cross_encoder(self, monkeypatch):
        mod = _reload_reranker(
            monkeypatch,
            RERANKER_MODEL="cross-encoder/ms-marco-MiniLM-L-6-v2",
            RERANKER_MAX_LENGTH="512",
        )

        with patch.object(mod, "CrossEncoder", return_value=MagicMock()) as ctor:
            mod._reranker_model = None
            mod._get_reranker_model()

        ctor.assert_called_once()
        args, kwargs = ctor.call_args
        assert args[0] == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert kwargs["max_length"] == 512

    def test_sigmoid_activation_survives_the_override(self, monkeypatch):
        """
        Every reranker_threshold in RETRIEVAL_CONFIGS (0.25-0.4) assumes scores
        in [0, 1]. A cross-encoder emits unbounded logits without an explicit
        activation, so dropping it while swapping models would leave every
        threshold comparing against the wrong scale — and nothing would raise.
        """
        import torch

        mod = _reload_reranker(
            monkeypatch, RERANKER_MODEL="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        with patch.object(mod, "CrossEncoder", return_value=MagicMock()) as ctor:
            mod._reranker_model = None
            mod._get_reranker_model()

        activation = ctor.call_args.kwargs["default_activation_function"]
        assert isinstance(activation, torch.nn.Sigmoid)

    def test_embedding_override_is_passed_to_sentence_transformer(self, monkeypatch):
        mod = _reload_reranker(monkeypatch, EMBEDDING_MODEL="BAAI/bge-large-en-v1.5")

        with patch.object(mod, "SentenceTransformer", return_value=MagicMock()) as ctor:
            mod._embedding_model = None
            mod._get_embedding_model()

        assert ctor.call_args.args[0] == "BAAI/bge-large-en-v1.5"
