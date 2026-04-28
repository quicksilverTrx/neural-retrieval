"""Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009).

Combines ranked lists from heterogeneous retrievers without score
normalisation — only rank positions matter.

Formula
-------
    score(d) = Σ_i 1 / (k + rank_i(d))

    rank_i(d) = 1-based position of d in system i's list
                (absent documents contribute 0 for that system)
    k         = smoothing constant, default 60

Why k=60?
---------
    k=60 prevents a rank-1 document from monopolising the fused score when
    it is absent from another system's list. At k=60 the maximum per-system
    contribution is 1/61 ≈ 0.016.

    Boundary intuitions:
        k→∞ : all documents score equally → fusion degenerates to voting.
        k=0 : rank-1 contribution = 1.0, tail contribution → 0 sharply.
        k=60: smooth middle ground; rank differences still meaningful.

    Rank-based fusion also sidesteps BM25 (unbounded) vs cosine (bounded [0,1])
    score-distribution mismatch — only positions are used.
"""
from __future__ import annotations

DEFAULT_K = 60


def fuse(
    ranked_lists: list[list[str]],
    k: int = DEFAULT_K,
) -> list[str]:
    """Fuse ranked lists with RRF.

    Args:
        ranked_lists: Each element is a ranked list of doc_ids (rank 1 = index 0).
                      Documents absent from a list contribute 0 for that system.
        k: Smoothing constant (default 60 per Cormack et al. 2009).

    Returns:
        Merged list of doc_ids sorted by RRF score descending.
    """
    scores = _rrf_scores(ranked_lists, k=k)

    # Build (score, doc_id) pairs so we can sort by score.
    scores_list = []
    for doc_id, score in scores.items():
        scores_list.append([score, doc_id])

    scores_list = sorted(scores_list, key=lambda x: x[0], reverse=True)

    return [item[1] for item in scores_list]


def fuse_scored(
    bm25_results: list[tuple[str, float]],
    dense_results: list[tuple[str, float]],
    k: int = DEFAULT_K,
) -> list[tuple[str, float]]:
    """Convenience wrapper: accept (doc_id, score) tuples, return (doc_id, rrf_score).

    Scores from individual systems are discarded; only rank order is used.

    Args:
        bm25_results: Ranked list of (doc_id, bm25_score) tuples.
        dense_results: Ranked list of (doc_id, dense_score) tuples.
        k: RRF smoothing constant.

    Returns:
        List of (doc_id, rrf_score) sorted by rrf_score descending.
    """
    bm25_ids = [doc_id for doc_id, _ in bm25_results]
    dense_ids = [doc_id for doc_id, _ in dense_results]
    merged = fuse([bm25_ids, dense_ids], k=k)
    scores = _rrf_scores([bm25_ids, dense_ids], k=k)
    return [(doc_id, scores[doc_id]) for doc_id in merged]


def _rrf_scores(
    ranked_lists: list[list[str]],
    k: int = DEFAULT_K,
) -> dict[str, float]:
    """Return {doc_id: rrf_score} without sorting.

    For each ranked list, every document receives a contribution of
    1 / (k + rank) where rank is 1-based. A document absent from a
    list simply has no iteration entry for it, so it contributes 0
    from that list — no special casing needed.

    Args:
        ranked_lists: list of ranked doc_id lists (rank 1 = index 0).
        k: RRF smoothing constant.

    Returns:
        dict mapping doc_id → accumulated RRF score across all lists.
    """
    scores: dict[str, float] = {}

    for ranked_list in ranked_lists:
        for rank_0based, doc_id in enumerate(ranked_list):
            rank = rank_0based + 1
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    return scores
