"""
Unit + property tests for retrieval/chunker.py.

Coverage: 100%. Key invariants: coverage, chunk size bounded, unique IDs, passage_id consistent.
"""
from __future__ import annotations

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from retrieval.chunker import ChunkRecord, PassageChunker, _tokenize


# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_lowercases(self):
        assert _tokenize("Hello World") == ["hello", "world"]

    def test_strips_punctuation(self):
        assert _tokenize("foo, bar! baz.") == ["foo", "bar", "baz"]

    def test_numbers(self):
        tokens = _tokenize("abc 123 def")
        assert "123" in tokens

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_only_punctuation(self):
        assert _tokenize("!!! ???") == []


# ---------------------------------------------------------------------------
# PassageChunker — construction
# ---------------------------------------------------------------------------

class TestChunkerInit:
    def test_default_params(self):
        c = PassageChunker()
        assert c.window_size == 256
        assert c.stride == 32

    def test_custom_params(self):
        c = PassageChunker(window_size=128, stride=16)
        assert c.window_size == 128
        assert c.stride == 16

    def test_stride_ge_window_raises(self):
        with pytest.raises(ValueError):
            PassageChunker(window_size=64, stride=64)

    def test_stride_gt_window_raises(self):
        with pytest.raises(ValueError):
            PassageChunker(window_size=64, stride=128)


# ---------------------------------------------------------------------------
# PassageChunker — short passages (single-chunk path)
# ---------------------------------------------------------------------------

class TestShortPassage:
    def test_short_passage_is_one_chunk(self):
        c = PassageChunker(window_size=256, stride=32)
        chunks = c.chunk_passage("p0", "short text here")
        assert len(chunks) == 1

    def test_chunk_id_format(self):
        c = PassageChunker(window_size=256, stride=32)
        chunks = c.chunk_passage("pid_42", "hello world")
        assert chunks[0].chunk_id == "pid_42_0"

    def test_passage_id_preserved(self):
        c = PassageChunker(window_size=256, stride=32)
        chunks = c.chunk_passage("myid", "some text")
        assert chunks[0].passage_id == "myid"

    def test_exact_window_size_is_one_chunk(self):
        c = PassageChunker(window_size=5, stride=2)
        text = "a b c d e"
        chunks = c.chunk_passage("p", text)
        assert len(chunks) == 1

    def test_token_span_correct_for_short(self):
        c = PassageChunker(window_size=10, stride=2)
        chunks = c.chunk_passage("p", "one two three")
        assert chunks[0].token_start == 0
        assert chunks[0].token_end == 3

    def test_empty_passage(self):
        c = PassageChunker(window_size=256, stride=32)
        chunks = c.chunk_passage("p0", "")
        assert len(chunks) == 1
        assert chunks[0].text == ""
        assert chunks[0].token_end == 0


# ---------------------------------------------------------------------------
# PassageChunker — long passages (multi-chunk path)
# ---------------------------------------------------------------------------

class TestLongPassage:
    def _make_passage(self, n_tokens: int) -> str:
        return " ".join(f"tok{i}" for i in range(n_tokens))

    def test_long_passage_produces_multiple_chunks(self):
        c = PassageChunker(window_size=10, stride=5)
        chunks = c.chunk_passage("p", self._make_passage(30))
        assert len(chunks) > 1

    def test_first_chunk_starts_at_zero(self):
        c = PassageChunker(window_size=10, stride=5)
        chunks = c.chunk_passage("p", self._make_passage(25))
        assert chunks[0].token_start == 0

    def test_last_chunk_ends_at_passage_length(self):
        c = PassageChunker(window_size=10, stride=5)
        n = 25
        chunks = c.chunk_passage("p", self._make_passage(n))
        assert chunks[-1].token_end == n

    def test_chunk_ids_sequential(self):
        c = PassageChunker(window_size=10, stride=5)
        chunks = c.chunk_passage("p42", self._make_passage(30))
        for i, ch in enumerate(chunks):
            assert ch.chunk_id == f"p42_{i}"

    def test_passage_id_consistent(self):
        c = PassageChunker(window_size=10, stride=5)
        chunks = c.chunk_passage("mypassage", self._make_passage(30))
        assert all(ch.passage_id == "mypassage" for ch in chunks)

    def test_chunks_cover_all_tokens(self):
        c = PassageChunker(window_size=10, stride=5)
        n = 23
        chunks = c.chunk_passage("p", self._make_passage(n))
        covered = set()
        for ch in chunks:
            covered.update(range(ch.token_start, ch.token_end))
        assert covered == set(range(n))

    def test_overlap_correct(self):
        # stride=5 means chunks[1].start == chunks[0].start + stride
        c = PassageChunker(window_size=10, stride=5)
        chunks = c.chunk_passage("p", self._make_passage(20))
        if len(chunks) >= 2:
            assert chunks[1].token_start == chunks[0].token_start + 5

    def test_each_chunk_size_le_window(self):
        c = PassageChunker(window_size=10, stride=3)
        chunks = c.chunk_passage("p", self._make_passage(35))
        for ch in chunks:
            assert (ch.token_end - ch.token_start) <= 10


# ---------------------------------------------------------------------------
# chunk_corpus
# ---------------------------------------------------------------------------

class TestChunkCorpus:
    def test_returns_all_chunk_ids_in_mapping(self):
        c = PassageChunker(window_size=5, stride=2)
        passages = [("p0", "a b c d e f g"), ("p1", "x y z")]
        chunks, mapping = c.chunk_corpus(passages)
        for ch in chunks:
            assert ch.chunk_id in mapping
            assert mapping[ch.chunk_id] == ch.passage_id

    def test_mapping_round_trip(self):
        c = PassageChunker(window_size=256, stride=32)
        passages = [("id1", "short"), ("id2", "also short")]
        _, mapping = c.chunk_corpus(passages)
        assert mapping["id1_0"] == "id1"
        assert mapping["id2_0"] == "id2"

    def test_empty_corpus(self):
        c = PassageChunker()
        chunks, mapping = c.chunk_corpus([])
        assert chunks == []
        assert mapping == {}


# ---------------------------------------------------------------------------
# Property tests (hypothesis)
# ---------------------------------------------------------------------------

@given(
    window=st.integers(min_value=2, max_value=20),
    stride=st.integers(min_value=1, max_value=19),
    n_tokens=st.integers(min_value=0, max_value=50),
)
@settings(max_examples=200)
def test_property_coverage(window: int, stride: int, n_tokens: int):
    """Every token position in [0, n_tokens) appears in at least one chunk."""
    assume(stride < window)
    text = " ".join(f"w{i}" for i in range(n_tokens))
    c = PassageChunker(window_size=window, stride=stride)
    chunks = c.chunk_passage("p", text)
    if n_tokens == 0:
        return
    covered = set()
    for ch in chunks:
        covered.update(range(ch.token_start, ch.token_end))
    assert covered == set(range(n_tokens))


@given(
    window=st.integers(min_value=2, max_value=20),
    stride=st.integers(min_value=1, max_value=19),
    n_tokens=st.integers(min_value=0, max_value=50),
)
@settings(max_examples=200)
def test_property_chunk_size_bounded(window: int, stride: int, n_tokens: int):
    """No chunk ever exceeds window_size tokens."""
    assume(stride < window)
    text = " ".join(f"w{i}" for i in range(n_tokens))
    c = PassageChunker(window_size=window, stride=stride)
    chunks = c.chunk_passage("p", text)
    for ch in chunks:
        assert (ch.token_end - ch.token_start) <= window


@given(
    window=st.integers(min_value=2, max_value=20),
    stride=st.integers(min_value=1, max_value=19),
    n_tokens=st.integers(min_value=1, max_value=50),
)
@settings(max_examples=200)
def test_property_chunk_ids_unique(window: int, stride: int, n_tokens: int):
    """All chunk_ids within a passage are unique."""
    assume(stride < window)
    text = " ".join(f"w{i}" for i in range(n_tokens))
    c = PassageChunker(window_size=window, stride=stride)
    chunks = c.chunk_passage("p", text)
    ids = [ch.chunk_id for ch in chunks]
    assert len(ids) == len(set(ids))


@given(
    window=st.integers(min_value=2, max_value=20),
    stride=st.integers(min_value=1, max_value=19),
    n_tokens=st.integers(min_value=1, max_value=50),
)
@settings(max_examples=200)
def test_property_passage_id_invariant(window: int, stride: int, n_tokens: int):
    """All chunks from a passage share the same passage_id."""
    assume(stride < window)
    pid = "passage_xyz"
    text = " ".join(f"w{i}" for i in range(n_tokens))
    c = PassageChunker(window_size=window, stride=stride)
    chunks = c.chunk_passage(pid, text)
    assert all(ch.passage_id == pid for ch in chunks)
