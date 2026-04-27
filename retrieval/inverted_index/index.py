"""Inverted index: term → posting list over the MS MARCO 8.8M corpus.

Memory design
-------------
Posting lists are stored as ``array.array('i')`` — a contiguous C int32 buffer —
interleaved as [doc0, tf0, doc1, tf1, ...].  This is a ~14× memory reduction vs
the earlier ``list[tuple[int, int]]`` representation:

    Old: Python tuple(int, int)      = 112 bytes/entry  (56 tuple + 28+28 int)
    New: two int32 slots in array    =   8 bytes/entry

At 88M posting-list entries (8.8M docs × ~40 unique terms avg):

    Old total: ~10 GB Python heap
    New total: ~700 MB array.array  (fits comfortably in RAM)

``array.array`` supports O(1) amortised ``append`` and efficient ``extend`` on
another array of the same typecode, so the build-time API is unchanged.

Doc IDs must be integers.  The original MSMARCO passage IDs are numeric strings
("7132531"), parsed to ``int`` at JSONL export time.  For unit tests with
synthetic corpora, use integer doc IDs (1, 2, 3) rather than string ("d1", "d2").
"""
from __future__ import annotations

import re
from array import array
from collections import Counter


# ---------------------------------------------------------------------------
# Stopwords
#
# Without stopword removal, query terms like "the" pull 7.7 M postings (87% of
# the 8.8 M corpus) into the candidate set, inflating end-to-end query latency
# from ~50 ms to ~12 s on a profiled query. Matching the bm25s library's NLTK
# English stopword list keeps our retrieval comparable to the baseline.
#
# IMPORTANT: stopword removal must be applied at BOTH index build time and
# query time so the vocabularies match. ``tokenize()`` is the single shared
# entry point — change here propagates everywhere automatically.
# ---------------------------------------------------------------------------
def _load_stopwords() -> frozenset[str]:
    """Load NLTK English stopwords; trigger one-time download on first run."""
    from nltk.corpus import stopwords
    try:
        return frozenset(stopwords.words("english"))
    except LookupError:
        import nltk
        nltk.download("stopwords", quiet=True)
        return frozenset(stopwords.words("english"))


STOPWORDS: frozenset[str] = _load_stopwords()


def tokenize(text: str) -> list[str]:
    """Lowercase word-boundary tokenizer with English stopword removal.

    Consistent with chunker.py for the word-boundary split.
    """
    return [t for t in re.findall(r"\w+", text.lower()) if t not in STOPWORDS]


# array.array typecode for 32-bit signed int.
_PAIR_TYPECODE = "i"


class InvertedIndex:
    """Term → posting list inverted index.

    Internal storage: ``dict[str, array.array('i')]`` where each array holds
    interleaved (doc_id, term_freq) pairs.

    Two iteration APIs:

    - ``get_posting_list(term)``  → list of tuples. Backward-compatible for
      tests; allocates one tuple per entry — do not use in hot paths.
    - ``get_raw_posting(term)``   → the underlying ``array.array``. Callers
      iterate pairs by index. Used by BM25Scorer and BM25Retriever.
    """

    def __init__(self) -> None:
        self._index: dict[str, array] = {}
        self._doc_lengths: dict[int, int] = {}
        self._num_docs: int = 0
        self._total_tokens: int = 0

    def add_document(self, doc_id: int, tokens: list[str]) -> None:
        """Index one document: update posting lists and length stats."""
        freq_map = Counter(tokens)
        for term, freq in freq_map.items():
            posting = self._index.get(term)
            if posting is None:
                posting = array(_PAIR_TYPECODE)
                self._index[term] = posting
            posting.extend((doc_id, freq))

        self._doc_lengths[doc_id] = len(tokens)
        self._total_tokens += len(tokens)
        self._num_docs += 1

    def get_posting_list(self, term: str) -> list[tuple[int, int]]:
        """Return [(doc_id, term_freq), ...] for ``term``; [] if unknown.

        Backward-compatible shim. Allocates tuples; for large posting lists
        in hot paths, use :py:meth:`get_raw_posting`.
        """
        raw = self._index.get(term)
        if raw is None:
            return []
        return [(raw[i], raw[i + 1]) for i in range(0, len(raw), 2)]

    def get_raw_posting(self, term: str) -> array:
        """Return the raw ``array.array('i')`` for ``term``; empty array if unknown.

        Callers iterate pairs by index to avoid per-entry tuple allocation:

            raw = index.get_raw_posting("cat")
            for i in range(0, len(raw), 2):
                doc_id = raw[i]
                tf     = raw[i + 1]
                ...
        """
        raw = self._index.get(term)
        if raw is None:
            return array(_PAIR_TYPECODE)
        return raw

    @property
    def vocab(self) -> set[str]:
        return set(self._index.keys())

    @property
    def num_docs(self) -> int:
        return self._num_docs

    @property
    def avg_doc_length(self) -> float:
        if self._num_docs == 0:
            return 0.0
        return self._total_tokens / self._num_docs

    def doc_length(self, doc_id: int) -> int:
        """Token count for a document. Returns 0 for unknown doc_id."""
        return self._doc_lengths.get(doc_id, 0)
