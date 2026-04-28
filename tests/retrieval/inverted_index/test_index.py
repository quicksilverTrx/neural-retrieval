"""Unit tests for InvertedIndex.

Doc IDs are integers to match the array-backed storage format
(``array.array('i')`` internally).
"""
from __future__ import annotations

from array import array

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from retrieval.inverted_index.index import InvertedIndex, tokenize


# ---------------------------------------------------------------------------
# tokenize()
# ---------------------------------------------------------------------------

def test_tokenize_lowercase():
    assert tokenize("Hello World") == ["hello", "world"]


def test_tokenize_strips_punctuation():
    assert tokenize("cats, dogs! fish.") == ["cats", "dogs", "fish"]


def test_tokenize_empty():
    assert tokenize("") == []


def test_tokenize_numbers_kept():
    assert "42" in tokenize("room 42")


# ---------------------------------------------------------------------------
# add_document() + get_posting_list()
# ---------------------------------------------------------------------------

def test_posting_list_entry_is_tuple():
    """Posting list entries must be (int, int) tuples."""
    idx = InvertedIndex()
    idx.add_document(1, ["cat"])
    entry = idx.get_posting_list("cat")[0]
    assert isinstance(entry, tuple)
    assert entry == (1, 1)


def test_term_frequency_counted_correctly():
    idx = InvertedIndex()
    idx.add_document(1, ["cat", "cat", "dog"])
    assert (1, 2) in idx.get_posting_list("cat")
    assert (1, 1) in idx.get_posting_list("dog")


def test_multiple_docs_accumulate_in_posting_list():
    idx = InvertedIndex()
    idx.add_document(1, ["cat"])
    idx.add_document(2, ["cat", "cat"])
    pl = idx.get_posting_list("cat")
    doc_ids = [d for d, _ in pl]
    assert 1 in doc_ids
    assert 2 in doc_ids


def test_term_only_in_one_doc_not_in_other():
    idx = InvertedIndex()
    idx.add_document(1, ["cat"])
    idx.add_document(2, ["dog"])
    pl = idx.get_posting_list("cat")
    assert all(d == 1 for d, _ in pl)


def test_get_posting_list_unknown_term_returns_empty():
    idx = InvertedIndex()
    idx.add_document(1, ["cat"])
    assert idx.get_posting_list("elephant") == []


def test_get_posting_list_on_empty_index_returns_empty():
    assert InvertedIndex().get_posting_list("anything") == []


def test_add_document_new_term_and_existing_term():
    """Covers both branches of 'if posting is None'."""
    idx = InvertedIndex()
    idx.add_document(1, ["apple"])
    idx.add_document(2, ["apple"])
    pl = idx.get_posting_list("apple")
    assert len(pl) == 2


# ---------------------------------------------------------------------------
# get_raw_posting() — low-allocation hot-path API
# ---------------------------------------------------------------------------

def test_get_raw_posting_is_array_i():
    idx = InvertedIndex()
    idx.add_document(1, ["cat", "cat"])
    raw = idx.get_raw_posting("cat")
    assert isinstance(raw, array)
    assert raw.typecode == "i"
    assert len(raw) == 2
    assert raw[0] == 1
    assert raw[1] == 2


def test_get_raw_posting_unknown_term_returns_empty_array():
    idx = InvertedIndex()
    raw = idx.get_raw_posting("nothing-here")
    assert isinstance(raw, array)
    assert len(raw) == 0


# ---------------------------------------------------------------------------
# vocab
# ---------------------------------------------------------------------------

def test_vocab_contains_all_indexed_terms():
    idx = InvertedIndex()
    idx.add_document(1, ["alpha", "beta"])
    idx.add_document(2, ["beta", "gamma"])
    assert idx.vocab == {"alpha", "beta", "gamma"}


def test_vocab_empty_index():
    assert InvertedIndex().vocab == set()


# ---------------------------------------------------------------------------
# num_docs
# ---------------------------------------------------------------------------

def test_num_docs_zero_initially():
    assert InvertedIndex().num_docs == 0


def test_num_docs_increments_per_document():
    idx = InvertedIndex()
    idx.add_document(1, ["a"])
    idx.add_document(2, ["b"])
    assert idx.num_docs == 2


# ---------------------------------------------------------------------------
# doc_length + avg_doc_length
# ---------------------------------------------------------------------------

def test_doc_length_matches_tokens_passed():
    idx = InvertedIndex()
    idx.add_document(1, ["a", "b", "c"])
    assert idx.doc_length(1) == 3


def test_doc_length_unknown_doc_returns_zero():
    assert InvertedIndex().doc_length(9999) == 0


def test_avg_doc_length_zero_on_empty_index():
    assert InvertedIndex().avg_doc_length == 0.0


def test_avg_doc_length_correct():
    idx = InvertedIndex()
    idx.add_document(1, ["a", "b", "c"])
    idx.add_document(2, ["a", "b"])
    assert idx.avg_doc_length == 2.5


def test_doc_length_repeated_tokens_counts_all():
    """doc_length = len(tokens), not len(unique tokens)."""
    idx = InvertedIndex()
    idx.add_document(1, ["cat", "cat", "cat"])
    assert idx.doc_length(1) == 3


# ---------------------------------------------------------------------------
# Hypothesis round-trip
# ---------------------------------------------------------------------------

@given(
    doc_ids=st.lists(st.integers(min_value=1, max_value=10_000), min_size=1, max_size=20, unique=True),
    words=st.lists(st.from_regex(r"[a-z]{2,8}", fullmatch=True), min_size=1, max_size=10),
)
@settings(max_examples=50)
def test_hypothesis_round_trip(doc_ids, words):
    """Every (doc_id, term) pair added must appear in the posting list."""
    idx = InvertedIndex()
    for doc_id in doc_ids:
        idx.add_document(doc_id, words)
    for term in set(words):
        pl = idx.get_posting_list(term)
        pl_doc_ids = {d for d, _ in pl}
        for doc_id in doc_ids:
            assert doc_id in pl_doc_ids
