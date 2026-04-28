"""Unit tests for RRF fusion."""
from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from retrieval.fusion.rrf import DEFAULT_K, _rrf_scores, fuse


def test_single_list_preserves_order():
    assert fuse([["a", "b", "c"]]) == ["a", "b", "c"]


def test_identical_lists_preserve_order():
    """Both lists rank in the same order → merged order matches."""
    assert fuse([["a", "b", "c"], ["a", "b", "c"]]) == ["a", "b", "c"]


def test_document_in_one_list_only_still_scores():
    """A document absent from one list contributes 0 there but still appears."""
    result = fuse([["a", "b"], ["c", "a"]])
    assert "b" in result
    assert "c" in result


def test_top_rank_in_both_beats_top_rank_in_one():
    """d1 at rank 1 in both lists outranks d2 (rank 1 in one only)."""
    result = fuse([["d1", "d2"], ["d1", "d3"]])
    assert result.index("d1") < result.index("d2")


def test_sorted_descending_by_score():
    """Output sorted by RRF score descending."""
    result = fuse([["a", "b", "c"], ["c", "b", "a"]])
    scores = _rrf_scores([["a", "b", "c"], ["c", "b", "a"]])
    for i in range(len(result) - 1):
        assert scores[result[i]] >= scores[result[i + 1]]


def test_empty_lists_return_empty():
    assert fuse([[], []]) == []


def test_one_empty_one_populated():
    assert fuse([[], ["a", "b"]]) == ["a", "b"]


def test_custom_k_accepted():
    result = fuse([["a", "b"], ["b", "a"]], k=1)
    assert isinstance(result, list)
    assert set(result) == {"a", "b"}


def test_higher_k_equalises_scores():
    """At very high k, the score gap between rank-1 and rank-2 shrinks."""
    scores_low_k = _rrf_scores([["d1", "d2"]], k=1)
    scores_high_k = _rrf_scores([["d1", "d2"]], k=1000)
    gap_low = scores_low_k["d1"] - scores_low_k["d2"]
    gap_high = scores_high_k["d1"] - scores_high_k["d2"]
    assert gap_high < gap_low


def test_output_contains_all_unique_documents():
    result = fuse([["a", "b"], ["b", "c"], ["c", "d"]])
    assert set(result) == {"a", "b", "c", "d"}


def test_default_k_is_60():
    assert DEFAULT_K == 60


def test_rrf_formula_single_system_rank1():
    """1 system, doc at rank 1 → score = 1/(60+1) = 1/61."""
    scores = _rrf_scores([["d1"]])
    assert abs(scores["d1"] - 1 / 61) < 1e-9


def test_rrf_formula_two_systems_rank1_both():
    """2 systems, doc at rank 1 in both → score = 2/(60+1) = 2/61."""
    scores = _rrf_scores([["d1"], ["d1"]])
    assert abs(scores["d1"] - 2 / 61) < 1e-9


def test_rrf_formula_rank_2():
    """1 system, doc at rank 2 → score = 1/(60+2) = 1/62."""
    scores = _rrf_scores([["d_other", "d1"]])
    assert abs(scores["d1"] - 1 / 62) < 1e-9


_doc_strategy = st.from_regex(r"d[0-9]{1,3}", fullmatch=True)


@given(
    list1=st.lists(_doc_strategy, min_size=1, max_size=20),
    list2=st.lists(_doc_strategy, min_size=1, max_size=20),
)
@settings(max_examples=50)
def test_hypothesis_output_is_union(list1, list2):
    result = fuse([list1, list2])
    assert set(result) == set(list1) | set(list2)


@given(docs=st.lists(_doc_strategy, min_size=2, max_size=30, unique=True))
@settings(max_examples=30)
def test_hypothesis_single_list_preserves_all(docs):
    assert fuse([docs]) == docs
