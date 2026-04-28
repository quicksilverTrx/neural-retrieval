"""Tests for BM25Retriever wiring.

Mocks ``InvertedIndex.get_raw_posting`` (which now returns ``array.array('i')``)
and ``BM25Scorer.score_batch``. The retriever calls ``get_raw_posting(term)``
which must return an ``array.array('i')`` of interleaved (doc_id, tf) pairs.
"""
from __future__ import annotations

from array import array
from unittest.mock import MagicMock

import pytest

from retrieval.inverted_index.bm25 import BM25Scorer
from retrieval.inverted_index.index import InvertedIndex
from retrieval.inverted_index.retriever import BM25Retriever


def _arr(*pairs: tuple[int, int]) -> array:
    """Build an array.array('i') with interleaved (doc, tf) pairs."""
    out = array("i")
    for d, tf in pairs:
        out.extend((d, tf))
    return out


def _make_retriever() -> tuple[BM25Retriever, MagicMock, MagicMock]:
    index = MagicMock(spec=InvertedIndex)
    scorer = MagicMock(spec=BM25Scorer)
    retriever = BM25Retriever(index=index, scorer=scorer)
    return retriever, index, scorer


def test_empty_query_returns_empty():
    r, _, _ = _make_retriever()
    assert r.retrieve("") == []


def test_whitespace_only_query_returns_empty():
    r, _, _ = _make_retriever()
    assert r.retrieve("   \t\n") == []


def test_no_candidates_skips_scorer():
    r, index, scorer = _make_retriever()
    index.get_raw_posting.return_value = array("i")

    result = r.retrieve("rare_term_xyz", top_k=10)

    assert result == []
    scorer.score_batch.assert_not_called()


def test_calls_posting_list_for_each_query_token():
    r, index, scorer = _make_retriever()
    index.get_raw_posting.return_value = array("i")
    scorer.score_batch.return_value = {}

    r.retrieve("hello world", top_k=10)

    called_terms = {c.args[0] for c in index.get_raw_posting.call_args_list}
    assert called_terms == {"hello", "world"}


def test_union_of_posting_lists_sent_to_scorer():
    r, index, scorer = _make_retriever()
    index.get_raw_posting.side_effect = lambda t: {
        "foo": _arr((1, 2), (2, 1)),
        "bar": _arr((2, 3), (3, 1)),
    }.get(t, array("i"))
    scorer.score_batch.return_value = {1: 1.0, 2: 2.0, 3: 0.5}

    r.retrieve("foo bar", top_k=10)

    candidates_arg = set(scorer.score_batch.call_args.args[1])
    assert candidates_arg == {1, 2, 3}


def test_results_sorted_descending_by_score():
    r, index, scorer = _make_retriever()
    index.get_raw_posting.side_effect = lambda t: _arr((1, 1), (2, 1), (3, 1))
    scorer.score_batch.return_value = {1: 3.0, 2: 1.5, 3: 5.0}

    results = r.retrieve("query", top_k=10)

    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0] == (3, 5.0)


def test_top_k_truncates_results():
    r, index, scorer = _make_retriever()
    index.get_raw_posting.side_effect = lambda t: _arr(*[(i, 1) for i in range(20)])
    scorer.score_batch.return_value = {i: float(i) for i in range(20)}

    results = r.retrieve("query", top_k=5)
    assert len(results) == 5


def test_top_k_larger_than_candidates_returns_all():
    r, index, scorer = _make_retriever()
    index.get_raw_posting.return_value = _arr((1, 1), (2, 1))
    scorer.score_batch.return_value = {1: 1.0, 2: 2.0}

    results = r.retrieve("q", top_k=1000)
    assert len(results) == 2


def test_retrieve_timed_returns_non_negative_latency():
    r, index, scorer = _make_retriever()
    index.get_raw_posting.return_value = array("i")
    scorer.score_batch.return_value = {}

    _, latency_ms = r.retrieve_timed("test query")

    assert isinstance(latency_ms, float)
    assert latency_ms >= 0.0


def test_retrieve_timed_results_match_retrieve():
    r, index, scorer = _make_retriever()
    index.get_raw_posting.side_effect = lambda t: _arr((1, 1))
    scorer.score_batch.return_value = {1: 4.2}

    results_direct = r.retrieve("q", top_k=10)
    results_timed, _ = r.retrieve_timed("q", top_k=10)

    assert results_direct == results_timed


def test_default_scorer_created_with_given_params():
    index = MagicMock(spec=InvertedIndex)
    r = BM25Retriever(index=index, k1=1.5, b=0.8)
    assert r.scorer.k1 == 1.5
    assert r.scorer.b == 0.8
