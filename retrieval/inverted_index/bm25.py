"""BM25 scoring over an InvertedIndex.

``score_batch`` is the hot path; it is **numpy-vectorised** to avoid the
Python-loop bottleneck that put a single query on the 8.8 M-passage corpus
at ~12 s end-to-end. Combined with the stopword-removal change in
``index.tokenize`` to get time reduction.

``score`` (single-doc) is kept on the Python path — it is only called in
unit tests and is not on any hot serving path.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .index import InvertedIndex


class BM25Scorer:
    """BM25 scoring with configurable k1 and b.

    score(Q, D) = Σ_{t ∈ Q} IDF(t) × TF_norm(t, D)

    IDF(t)        = log((N - df_t + 0.5) / (df_t + 0.5) + 1)
    TF_norm(t, D) = tf × (k1 + 1) / (tf + k1 × (1 - b + b × |D| / avgdl))

    k1: TF saturation — higher means repeated terms keep gaining score longer.
    b:  length normalization — 0.0 = none, 1.0 = full normalization by avgdl.
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        # Lazy numpy caches built on first scoring call, invalidated when the
        # backing index changes identity (e.g. after a reload). Memory cost
        # at 8.8 M-doc scale: 35 MB int32 doc-length array + 70 MB float64
        # score buffer ≈ 105 MB, persistent on this scorer instance.
        self._cached_index_id: int | None = None
        self._dl_array: np.ndarray | None = None
        self._score_buffer: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Single-doc scoring — kept on the Python path (test-only callers)
    # ------------------------------------------------------------------

    def score(self, query_tokens: list[str], doc_id: int, index: InvertedIndex) -> float:
        """BM25 score for a single query-document pair.

        Iterates posting lists once per query token, O(|Q| × |posting|).
        Use score_batch() when scoring many documents for the same query.
        """
        num_docs = index.num_docs
        avg_dl = index.avg_doc_length
        dl = index.doc_length(doc_id)

        total = 0.0
        for token in query_tokens:
            posting = index.get_raw_posting(token)   # array.array('i') interleaved
            df_t = len(posting) // 2
            if df_t == 0:
                continue

            # Linear scan for tf of this specific doc; O(|posting|) per token.
            tf = 0
            for i in range(0, len(posting), 2):
                if posting[i] == doc_id:
                    tf = posting[i + 1]
                    break
            if tf == 0:
                continue

            idf = math.log((num_docs - df_t + 0.5) / (df_t + 0.5) + 1)
            tf_norm = tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / avg_dl))
            total += idf * tf_norm
        return total

    # ------------------------------------------------------------------
    # Vectorised batch scoring — production hot path
    # ------------------------------------------------------------------

    def _ensure_caches(self, index: InvertedIndex) -> None:
        """Lazily build per-index numpy caches.

        Caches are keyed on ``id(index)``. If the caller swaps to a different
        InvertedIndex instance (e.g. via a hot-reload), we rebuild on the next
        scoring call. Within a single index's lifetime the caches are reused
        for every query — no per-query allocation.
        """
        if id(index) == self._cached_index_id:
            return

        # Dense int32 array of doc lengths indexed by doc_id.
        max_did = max(index._doc_lengths.keys()) if index._doc_lengths else -1
        dl = np.zeros(max_did + 1, dtype=np.int32)
        for did, length in index._doc_lengths.items():
            dl[did] = length
        self._dl_array = dl

        # Pre-allocated float64 score accumulator. Reused via in-place
        # fill(0.0) — zeroing 70 MB at 20 GB/s mem bandwidth ≈ 3.5 ms per query.
        self._score_buffer = np.zeros(max_did + 1, dtype=np.float64)

        self._cached_index_id = id(index)

    def score_batch(
        self,
        query_tokens: list[str],
        candidate_doc_ids: list[int],
        index: InvertedIndex,
    ) -> dict[int, float]:
        """Score the union of posting-list docs in a single vectorised pass.

        Returns ``{doc_id: score}`` for every doc in ``candidate_doc_ids``
        that has a non-zero BM25 score.

        Algorithm:
          1. For each query term: zero-copy view of the posting array, slice
             into strided (doc_ids, tfs) views, compute IDF × TF_norm in
             vectorised float64.
          2. Concatenate per-term doc_ids and contributions.
          3. Single ``np.bincount`` scatter-add into the dense score buffer
             (vectorised C, handles collisions, dwarfs per-term np.add.at).
          4. Fancy-index the score buffer by ``candidate_doc_ids`` to produce
             the result dict.
        """
        if not query_tokens:
            return {}
        self._ensure_caches(index)

        num_docs = index.num_docs
        avg_dl = index.avg_doc_length
        if avg_dl == 0.0:
            return {}

        scores = self._score_buffer
        scores.fill(0.0)
        dl_arr = self._dl_array
        k1 = self.k1
        b = self.b

        # ── Per-term feature compute, deferred accumulation ──────────────
        per_term_doc_ids: list[np.ndarray] = []
        per_term_contributions: list[np.ndarray] = []

        for token in query_tokens:
            posting = index.get_raw_posting(token)   # array.array('i') interleaved
            df_t = len(posting) // 2
            if df_t == 0:
                continue
            idf = math.log((num_docs - df_t + 0.5) / (df_t + 0.5) + 1)

            # Zero-copy view of the array.array('i') buffer as int32.
            buf = np.frombuffer(posting, dtype=np.int32)
            doc_ids = buf[0::2]   # strided view, even slots
            tfs = buf[1::2]        # strided view, odd slots

            # Vectorised BM25 TF norm in float64 for numerical fidelity.
            dl = dl_arr[doc_ids].astype(np.float64, copy=False)
            tfs_f = tfs.astype(np.float64, copy=False)
            denom = tfs_f + k1 * (1.0 - b + b * dl / avg_dl)
            contribution = idf * (tfs_f * (k1 + 1.0) / denom)

            per_term_doc_ids.append(doc_ids)
            per_term_contributions.append(contribution)

        if not per_term_doc_ids:
            return {}

        # ── Single bincount-based scatter-add over concatenated arrays ──
        # bincount is the dense scatter-add primitive; it
        # handles collisions in one pass. np.add.at would serialise on
        # collisions and allocate per-term temps.
        flat_ids = np.concatenate(per_term_doc_ids)
        flat_contribs = np.concatenate(per_term_contributions)
        scores += np.bincount(
            flat_ids, weights=flat_contribs, minlength=len(scores)
        )

        # ── Filter to candidate set ─────────────────────────────────────
        if not candidate_doc_ids:
            return {}
        cand_arr = np.fromiter(candidate_doc_ids, dtype=np.int64,
                               count=len(candidate_doc_ids))
        cand_scores = scores[cand_arr]
        nonzero = cand_scores != 0.0
        return dict(zip(cand_arr[nonzero].tolist(), cand_scores[nonzero].tolist()))
