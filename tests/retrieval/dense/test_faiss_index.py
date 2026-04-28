"""Tests for FAISSIVFPQIndex.

Most tests use a tiny in-memory FAISS index so they run in CI without GPU.
GPU integration tests are gated behind @pytest.mark.gpu.
"""
from __future__ import annotations

import json

import numpy as np
import pytest


def _make_index(dim: int = 8, nlist: int = 4, m: int = 2, nbits: int = 8, nprobe: int = 2):
    from retrieval.dense.faiss_index import FAISSIVFPQIndex

    return FAISSIVFPQIndex(embedding_dim=dim, nlist=nlist, m=m, nbits=nbits, nprobe=nprobe)


def _random_vecs(n: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def test_default_params_match_spec():
    from retrieval.dense.faiss_index import (
        DEFAULT_M,
        DEFAULT_NBITS,
        DEFAULT_NLIST,
        DEFAULT_NPROBE,
        FAISSIVFPQIndex,
    )

    idx = FAISSIVFPQIndex()
    assert idx.nlist == DEFAULT_NLIST
    assert idx.m == DEFAULT_M
    assert idx.nbits == DEFAULT_NBITS
    assert idx.nprobe == DEFAULT_NPROBE


def test_add_before_train_raises():
    idx = _make_index()
    vecs = _random_vecs(10)
    with pytest.raises(RuntimeError, match="train"):
        idx.add(vecs, [f"p{i}" for i in range(10)])


def test_search_before_build_raises():
    idx = _make_index()
    q = _random_vecs(1)
    with pytest.raises(RuntimeError):
        idx.search(q)


@pytest.fixture
def built_index():
    pytest.importorskip("faiss")
    dim, nlist, m = 16, 4, 2
    # nbits=4 → 16 PQ centroids; safe with 500 train vecs.
    # nbits=8 (256 centroids) needs ≥256×n_subvec training points and segfaults otherwise.
    idx = _make_index(dim=dim, nlist=nlist, m=m, nbits=4, nprobe=2)

    n_train = 500
    n_corpus = 200
    train_vecs = _random_vecs(n_train, dim)
    corpus_vecs = _random_vecs(n_corpus, dim)
    pids = [f"pid_{i}" for i in range(n_corpus)]

    idx.train(train_vecs)
    idx.add(corpus_vecs, pids)
    return idx, pids


def test_ntotal_after_add(built_index):
    idx, pids = built_index
    assert idx._index.ntotal == len(pids)


def test_search_returns_correct_shape(built_index):
    idx, _ = built_index
    q = _random_vecs(3, dim=16)
    results, dists = idx.search(q, top_k=10)
    assert len(results) == 3
    assert all(len(r) == 10 for r in results)
    assert dists.shape == (3, 10)


def test_search_single_vector_1d(built_index):
    idx, _ = built_index
    q = _random_vecs(1, dim=16)[0]
    results, _ = idx.search(q, top_k=5)
    assert len(results) == 1
    assert len(results[0]) == 5


def test_search_results_are_valid_pids(built_index):
    idx, pids = built_index
    pid_set = set(pids)
    q = _random_vecs(2, dim=16)
    results, _ = idx.search(q, top_k=20)
    for row in results:
        for pid in row:
            assert pid in pid_set or pid == ""


def test_nprobe_override(built_index):
    idx, _ = built_index
    q = _random_vecs(1, dim=16)
    idx.search(q, top_k=5, nprobe=1)
    idx.search(q, top_k=5, nprobe=4)


def test_save_load_round_trip(tmp_path, built_index):
    idx, pids = built_index
    save_path = tmp_path / "faiss_idx"
    idx.save(save_path)

    assert (save_path / "index.faiss").exists()
    assert (save_path / "pids.json").exists()
    assert (save_path / "meta.json").exists()

    from retrieval.dense.faiss_index import FAISSIVFPQIndex

    loaded = FAISSIVFPQIndex.load(save_path)
    assert loaded._index.ntotal == idx._index.ntotal
    assert loaded._pids == pids


def test_meta_json_contents(tmp_path, built_index):
    idx, _ = built_index
    idx.save(tmp_path / "idx")
    meta = json.loads((tmp_path / "idx" / "meta.json").read_text())
    assert meta["nlist"] == idx.nlist
    assert meta["m"] == idx.m
    assert meta["nbits"] == idx.nbits
    assert meta["ntotal"] == idx._index.ntotal


@pytest.mark.gpu
def test_real_8m_scale_search():
    """Smoke test at 8.8M scale — run locally after building the full index."""
    from retrieval.dense.encoder import SentenceEncoder
    from retrieval.dense.faiss_index import FAISSIVFPQIndex

    idx = FAISSIVFPQIndex.load("data/faiss/all_minilm_l6_v2")
    enc = SentenceEncoder("all-MiniLM-L6-v2")
    q = enc.encode_query("what causes fever in adults")
    results, _ = idx.search(q, top_k=100)
    assert len(results[0]) == 100
