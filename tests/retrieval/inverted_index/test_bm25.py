"""Unit tests for BM25Scorer.

Doc IDs are integers to match the array-backed InvertedIndex storage.
"""
from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from retrieval.inverted_index.bm25 import BM25Scorer
from retrieval.inverted_index.index import InvertedIndex, tokenize


def _small_index() -> InvertedIndex:
    """3-doc index for deterministic scoring tests."""
    idx = InvertedIndex()
    idx.add_document(1, tokenize("the cat sat on the mat"))
    idx.add_document(2, tokenize("the dog lay on the floor"))
    idx.add_document(3, tokenize("cats and dogs are common pets"))
    return idx


def test_score_non_negative():
    s = BM25Scorer().score(tokenize("cat"), 1, _small_index())
    assert s >= 0.0


def test_score_zero_for_doc_not_containing_query_term():
    s = BM25Scorer().score(tokenize("cat"), 2, _small_index())
    assert s == 0.0


def test_score_zero_for_empty_query():
    assert BM25Scorer().score([], 1, _small_index()) == 0.0


def test_score_returns_float():
    s = BM25Scorer().score(tokenize("cat mat"), 1, _small_index())
    assert isinstance(s, float)


def test_score_higher_for_more_matching_terms():
    """Doc with both query terms scores higher than doc with one."""
    idx = InvertedIndex()
    idx.add_document(1, tokenize("cat sat on the mat"))
    idx.add_document(2, tokenize("the cat ran away"))
    scorer = BM25Scorer()
    s1 = scorer.score(tokenize("cat mat"), 1, idx)
    s2 = scorer.score(tokenize("cat mat"), 2, idx)
    assert s1 > s2


def test_score_higher_tf_increases_score():
    idx = InvertedIndex()
    idx.add_document(1, ["cat"])
    idx.add_document(2, ["cat", "cat", "cat"])
    scorer = BM25Scorer(k1=10.0)
    s1 = scorer.score(["cat"], 1, idx)
    s2 = scorer.score(["cat"], 2, idx)
    assert s2 > s1


def test_score_rare_term_higher_idf():
    idx = InvertedIndex()
    for i in range(10):
        idx.add_document(i, ["common", "rare" if i == 0 else "other"])
    scorer = BM25Scorer()
    s_rare = scorer.score(["rare"], 0, idx)
    s_common = scorer.score(["common"], 0, idx)
    assert s_rare > s_common


def test_idf_formula_matches_manual_calculation():
    """IDF(t) = log((N - df_t + 0.5) / (df_t + 0.5) + 1)"""
    idx = InvertedIndex()
    idx.add_document(1, ["cat"])
    idx.add_document(2, ["dog"])
    idx.add_document(3, ["cat"])

    N, df = 3, 2
    expected_idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
    scorer = BM25Scorer(k1=1.2, b=0.0)
    tf = 1
    expected_tf_norm = tf * (1.2 + 1) / (tf + 1.2)
    expected_score = expected_idf * expected_tf_norm

    actual = scorer.score(["cat"], 1, idx)
    assert abs(actual - expected_score) < 1e-9


def test_score_batch_matches_score_per_candidate():
    idx = _small_index()
    scorer = BM25Scorer()
    tokens = tokenize("cat mat")
    candidates = list({d for t in tokens for d, _ in idx.get_posting_list(t)})

    batch = scorer.score_batch(tokens, candidates, idx)
    for doc_id in candidates:
        assert abs(batch.get(doc_id, 0.0) - scorer.score(tokens, doc_id, idx)) < 1e-9


def test_score_batch_excludes_non_candidates():
    idx = _small_index()
    batch = BM25Scorer().score_batch(["cats"], [1], idx)
    assert 3 not in batch


def test_score_batch_empty_query_returns_empty():
    assert BM25Scorer().score_batch([], [1, 2], _small_index()) == {}


def test_score_batch_all_candidates_outside_posting_list_returns_empty():
    assert BM25Scorer().score_batch(["zebra"], [1, 2], _small_index()) == {}


def test_score_batch_returns_dict():
    result = BM25Scorer().score_batch(tokenize("cat"), [1], _small_index())
    assert isinstance(result, dict)


@given(
    k1=st.floats(min_value=0.1, max_value=3.0),
    b=st.floats(min_value=0.0, max_value=1.0),
)
@settings(max_examples=30)
def test_hypothesis_score_non_negative(k1, b):
    idx = InvertedIndex()
    idx.add_document(1, ["cat", "mat"])
    idx.add_document(2, ["dog"])
    s = BM25Scorer(k1=k1, b=b).score(["cat"], 1, idx)
    assert s >= 0.0


@given(st.integers(min_value=2, max_value=50))
@settings(max_examples=20)
def test_hypothesis_idf_decreases_with_df(n_docs):
    idx_rare = InvertedIndex()
    idx_common = InvertedIndex()
    for i in range(n_docs):
        idx_rare.add_document(i, ["target"] if i == 0 else ["other"])
    for i in range(n_docs):
        idx_common.add_document(i, ["target"])

    scorer = BM25Scorer(b=0.0)
    s_rare = scorer.score(["target"], 0, idx_rare)
    s_common = scorer.score(["target"], 0, idx_common)
    assert s_rare >= s_common
