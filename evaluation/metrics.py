"""
IR evaluation metrics: nDCG@k, MRR@k, Recall@k, and bootstrap confidence intervals.

Implemented from first principles — no pytrec_eval dependency.
Graded relevance follows TREC DL convention: 0=not relevant, 1=related,
2=highly relevant, 3=perfectly relevant.

Relevance threshold for binary metrics (MRR, Recall) defaults to 1, meaning
any grade >= 1 counts as relevant. Override per call when a stricter threshold
is needed (e.g. threshold=2 for highly-relevant-only analysis).
"""
from __future__ import annotations

import math
import random
from typing import Callable, List, Set


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _relevant_set(qrels_for_query: dict[str, int], threshold: int) -> set[str]:
    """Return passage IDs whose relevance grade >= threshold."""
    result_set: Set[str] = set()
    for pid, grade in qrels_for_query.items():
        if grade >= threshold:
            result_set.add(pid)
    return result_set


def _dcg(grades: list[int]) -> float:
    """Discounted Cumulative Gain for an ordered list of relevance grades.

    grades[0] is rank-1, grades[1] is rank-2, etc.
    Discount denominator is log2(rank + 1), so rank-1 contributes full gain.
    """
    if not grades:
        return 0.0
    present_rank: int = 1
    dcg: float = 0.0
    for grade in grades:
        gain = math.exp2(grade) - 1
        discount = math.log2(present_rank + 1)
        dcg += gain / discount
        present_rank += 1
    return dcg


def _idcg(qrels_for_query: dict[str, int], k: int) -> float:
    """Ideal DCG@k — DCG of the perfect ranking built from qrels grades."""
    grade_list: List[int] = sorted(qrels_for_query.values(), reverse=True)
    return _dcg(grade_list[:k])


def _dcg_at_k_for_query(
    retrieved: list[str],
    qrels_for_query: dict[str, int],
    k: int,
) -> float:
    """DCG@k for one query: map retrieved PIDs to grades, then call _dcg()."""
    filtered_grades: List[int] = [qrels_for_query.get(pid, 0) for pid in retrieved[:k]]
    return _dcg(filtered_grades)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ndcg_at_k(
    qrels: dict[str, dict[str, int]],
    run: dict[str, list[str]],
    k: int = 10,
) -> float:
    """Mean nDCG@k over all queries in run.

    Queries present in run but absent from qrels are skipped (unjudged).
    Queries where IDCG == 0 (all grades are 0) contribute 0.0 to the mean.
    """
    scores: List[float] = []
    for query_id, retrieved_passages in run.items():
        if query_id not in qrels:
            continue
        ideal_dcg = _idcg(qrels[query_id], k)
        if ideal_dcg == 0.0:
            scores.append(0.0)
            continue
        actual_dcg = _dcg_at_k_for_query(retrieved_passages, qrels[query_id], k)
        scores.append(actual_dcg / ideal_dcg)
    return sum(scores) / len(scores) if scores else 0.0


def mrr_at_k(
    qrels: dict[str, dict[str, int]],
    run: dict[str, list[str]],
    k: int = 10,
    relevance_threshold: int = 1,
) -> float:
    """Mean Reciprocal Rank @ k.

    Scores the rank of the first relevant hit per query, then averages.
    Queries with no relevant result in top-k contribute 0.0.
    """
    reciprocal_ranks: List[float] = []
    for query_id, retrieved_passages in run.items():
        if query_id not in qrels:
            continue
        relevant = _relevant_set(qrels[query_id], relevance_threshold)
        rr: float = 0.0
        for rank, pid in enumerate(retrieved_passages[:k], start=1):
            if pid in relevant:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


def recall_at_k(
    qrels: dict[str, dict[str, int]],
    run: dict[str, list[str]],
    k: int = 100,
    relevance_threshold: int = 1,
) -> float:
    """Mean Recall@k.

    Denominator is the total number of relevant passages in qrels for each
    query — not capped at k. Queries with zero relevant passages are skipped.
    """
    recall_scores: List[float] = []
    for query_id, retrieved_passages in run.items():
        if query_id not in qrels:
            continue
        relevant = _relevant_set(qrels[query_id], relevance_threshold)
        if not relevant:
            continue
        retrieved_set: Set[str] = set(retrieved_passages[:k])
        hits: int = len(retrieved_set & relevant)
        recall_scores.append(hits / len(relevant))
    return sum(recall_scores) / len(recall_scores) if recall_scores else 0.0


def _single_query_ndcg(
    qid: str,
    qrels: dict[str, dict[str, int]],
    run: dict[str, list[str]],
    k: int,
) -> float | None:
    """nDCG@k for one query. Returns None if qid is absent from qrels or run."""
    if qid not in qrels or qid not in run:
        return None
    ideal_dcg = _idcg(qrels[qid], k)
    if ideal_dcg == 0.0:
        return 0.0
    actual_dcg = _dcg_at_k_for_query(run[qid], qrels[qid], k)
    return actual_dcg / ideal_dcg


def per_query_metrics(
    qrels: dict[str, dict[str, int]],
    run: dict[str, list[str]],
    k: int = 10,
) -> list[dict]:
    """Per-query metric breakdown for JSON output and failure analysis.

    Returns [{"qid": str, "nDCG@k": float, "Recall@100": float}, ...]
    Used to populate the per_query field in result JSON and to drive
    per-query failure analysis.
    """
    results: List[dict] = []
    for qid in run:
        ndcg = _single_query_ndcg(qid, qrels, run, k)
        if ndcg is None:
            continue
        relevant = _relevant_set(qrels[qid], threshold=1)
        retrieved_set: Set[str] = set(run[qid][:100])
        hits: int = len(retrieved_set & relevant)
        recall: float = hits / len(relevant) if relevant else 0.0
        results.append({"qid": qid, f"nDCG@{k}": ndcg, "Recall@100": recall})
    return results


def bootstrap_ci(
    metric_fn: Callable,
    qrels: dict[str, dict[str, int]],
    run: dict[str, list[str]],
    k: int = 10,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Bootstrap (1 - alpha) confidence interval for a metric, resampled over queries.

    Resamples at the query level (not passage level) because the query is the
    unit of measurement. Each query contributes one scalar score; CI quantifies
    variance across the query distribution, not the corpus.

    Returns (lower_bound, upper_bound) at the alpha/2 and 1-alpha/2 percentiles.
    """
    per_query_scores: List[float] = []
    for qid in run:
        if qid not in qrels:
            continue
        single_qrels = {qid: qrels[qid]}
        single_run = {qid: run[qid]}
        per_query_scores.append(metric_fn(single_qrels, single_run, k))

    if not per_query_scores:
        return (0.0, 0.0)

    bootstrap_means: List[float] = []
    n: int = len(per_query_scores)
    for _ in range(n_bootstrap):
        sample = random.choices(per_query_scores, k=n)
        bootstrap_means.append(sum(sample) / n)

    bootstrap_means.sort()
    lo_idx: int = int(n_bootstrap * alpha / 2)
    hi_idx: int = int(n_bootstrap * (1 - alpha / 2))
    return (bootstrap_means[lo_idx], bootstrap_means[hi_idx])
