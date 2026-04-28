"""Tests for SentenceEncoder.

Model.encode() is mocked to avoid HF download / GPU dependency in CI.
GPU integration tests are gated behind @pytest.mark.gpu.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pytest


def _make_mock_model(dim: int = 384) -> MagicMock:
    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = dim
    model.get_embedding_dimension.return_value = dim
    return model


def _make_encoder(dim: int = 384, model_name: str = "all-MiniLM-L6-v2"):
    from retrieval.dense.encoder import SentenceEncoder

    enc = SentenceEncoder.__new__(SentenceEncoder)
    enc.model_name = model_name
    enc.embedding_dim = dim
    enc._is_e5 = "e5" in model_name.lower()
    enc.model = _make_mock_model(dim)
    return enc


def test_encode_query_returns_float32_vector():
    enc = _make_encoder()
    enc.model.encode.return_value = np.ones(384, dtype=np.float32)

    result = enc.encode_query("what causes fever")

    assert result.dtype == np.float32
    assert result.shape == (384,)


def test_encode_query_calls_model_with_query_text():
    enc = _make_encoder()
    enc.model.encode.return_value = np.ones(384, dtype=np.float32)

    enc.encode_query("my query")

    enc.model.encode.assert_called_once()
    call_arg = enc.model.encode.call_args.args[0]
    assert "my query" in call_arg


def test_encode_query_e5_adds_prefix():
    enc = _make_encoder(model_name="intfloat/e5-small-v2")
    enc.model.encode.return_value = np.ones(384, dtype=np.float32)

    enc.encode_query("my query")

    assert enc.model.encode.call_args.args[0].startswith("query: ")


def test_encode_query_miniLM_no_prefix():
    enc = _make_encoder(model_name="all-MiniLM-L6-v2")
    enc.model.encode.return_value = np.ones(384, dtype=np.float32)

    enc.encode_query("my query")

    assert not enc.model.encode.call_args.args[0].startswith("query: ")


def test_encode_batch_shape():
    enc = _make_encoder(dim=384)
    enc.model.encode.return_value = np.random.rand(3, 384).astype(np.float32)

    result = enc.encode_batch(["text one", "text two", "text three"])

    assert result.shape == (3, 384)
    assert result.dtype == np.float32


def test_encode_batch_e5_passage_prefix():
    enc = _make_encoder(model_name="intfloat/e5-small-v2")
    enc.model.encode.return_value = np.ones((2, 384), dtype=np.float32)

    enc.encode_batch(["text a", "text b"], is_query=False)

    prefixed = enc.model.encode.call_args.args[0]
    assert all(t.startswith("passage: ") for t in prefixed)


def test_encode_batch_e5_query_prefix():
    enc = _make_encoder(model_name="intfloat/e5-small-v2")
    enc.model.encode.return_value = np.ones((2, 384), dtype=np.float32)

    enc.encode_batch(["q a", "q b"], is_query=True)

    prefixed = enc.model.encode.call_args.args[0]
    assert all(t.startswith("query: ") for t in prefixed)


def test_encode_corpus_writes_manifest(tmp_path):
    enc = _make_encoder(dim=4)
    passages = [(f"p{i}", f"text {i}") for i in range(10)]
    enc.model.encode.return_value = np.ones((10, 4), dtype=np.float32)

    enc.encode_corpus(passages, tmp_path, batch_size=5)

    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists()
    m = json.loads(manifest_path.read_text())
    assert m["num_passages"] == 10
    assert m["embedding_dim"] == 4
    assert m["model_name"] == "all-MiniLM-L6-v2"


def test_encode_corpus_writes_pids_json(tmp_path):
    enc = _make_encoder(dim=4)
    passages = [("pid_a", "text a"), ("pid_b", "text b")]
    enc.model.encode.return_value = np.ones((2, 4), dtype=np.float32)

    enc.encode_corpus(passages, tmp_path, batch_size=2)

    pids = json.loads((tmp_path / "pids.json").read_text())
    assert pids == ["pid_a", "pid_b"]


def test_encode_corpus_load_round_trip(tmp_path):
    """Written .npy file is loadable as the same array."""
    from retrieval.dense.encoder import SentenceEncoder

    enc = _make_encoder(dim=4)
    n = 6
    vecs = np.arange(n * 4, dtype=np.float32).reshape(n, 4)
    enc.model.encode.return_value = vecs
    passages = [(f"p{i}", f"t{i}") for i in range(n)]

    enc.encode_corpus(passages, tmp_path, batch_size=n)

    loaded, pids, manifest = SentenceEncoder.load_embeddings(tmp_path)
    assert loaded.shape == (n, 4)
    np.testing.assert_array_equal(np.asarray(loaded), vecs)
    assert pids == [f"p{i}" for i in range(n)]
    assert manifest["num_passages"] == n


@pytest.mark.gpu
def test_real_encode_query_shape():
    from retrieval.dense.encoder import SentenceEncoder

    enc = SentenceEncoder("all-MiniLM-L6-v2")
    vec = enc.encode_query("what is the speed of light?")
    assert vec.shape == (384,)
    assert vec.dtype == np.float32
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-5
