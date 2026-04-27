"""BM25 retriever: wires tokenizer → InvertedIndex → BM25Scorer → ranked results."""
from __future__ import annotations

import time

import numpy as np

from .bm25 import BM25Scorer
from .index import InvertedIndex, tokenize


class BM25Retriever:
    """End-to-end BM25 retrieval: query string → ranked (doc_id, score) list.

    Usage:
        index = InvertedIndex()
        for pid, text in corpus:
            index.add_document(pid, tokenize(text))   # pid is int

        retriever = BM25Retriever(index)
        results = retriever.retrieve("what causes fever", top_k=100)
        # → [(123, 18.4), (456, 16.1), ...]   (doc_ids are int)
    """

    def __init__(
        self,
        index: InvertedIndex,
        scorer: BM25Scorer | None = None,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        self.index = index
        self.scorer = scorer or BM25Scorer(k1=k1, b=b)

    def retrieve(self, query: str, top_k: int = 100) -> list[tuple[int, float]]:
        """Retrieve ranked (doc_id, score) list for a query string.

        Steps:
        1. Tokenize query (with stopword removal — see index.tokenize).
        2. Build candidate union via vectorised numpy concat.
        3. Score candidates via BM25Scorer.score_batch().
        4. Sort descending and return top_k.

        Returns [] for empty query or query with no indexed terms.
        """
        tokens = tokenize(query)
        if not tokens:
            return []

        # ── Vectorised candidate collection (with duplicates) ───────────
        # Per-term: zero-copy view of the array.array('i') buffer + strided
        # slice of even indices (doc_ids). np.concatenate stitches them
        # into one int32 array.
        per_term_ids: list[np.ndarray] = []
        for token in tokens:
            raw = self.index.get_raw_posting(token)
            if len(raw) == 0:
                continue
            buf = np.frombuffer(raw, dtype=np.int32)
            per_term_ids.append(buf[0::2])

        if not per_term_ids:
            return []

        candidates_arr = np.concatenate(per_term_ids)
        if candidates_arr.size == 0:
            return []

        scores = self.scorer.score_batch(tokens, candidates_arr.tolist(), self.index)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def retrieve_timed(
        self, query: str, top_k: int = 100
    ) -> tuple[list[tuple[int, float]], float]:
        """retrieve() with wall-clock latency_ms returned as second element."""
        t0 = time.perf_counter()
        results = self.retrieve(query, top_k)
        return results, (time.perf_counter() - t0) * 1000
