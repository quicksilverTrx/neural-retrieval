"""
Tests for evaluation/trec_eval.py.

Covers loader correctness against pre-downloaded JSON files.
Skips data-dependent tests when files are absent (CI).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evaluation.trec_eval import (
    TREC_DL_2019,
    TREC_DL_2020,
    combined_qrels,
    combined_queries,
    dataset_stats,
    load_qrels,
    load_queries,
    qrels_hash,
)

DATA_PRESENT = (
    Path("data/queries/trec_dl_2020_queries.json").exists()
    and Path("data/qrels/trec_dl_2020_qrels.json").exists()
)

skip_no_data = pytest.mark.skipif(not DATA_PRESENT, reason="TREC data not downloaded")


# ---------------------------------------------------------------------------
# Structure tests (data present)
# ---------------------------------------------------------------------------

@skip_no_data
class TestTrecDl2020:
    def test_query_count(self):
        q = load_queries(TREC_DL_2020)
        assert len(q) == 54, f"Expected 54 queries, got {len(q)}"

    def test_query_values_are_strings(self):
        q = load_queries(TREC_DL_2020)
        for qid, text in q.items():
            assert isinstance(qid, str)
            assert isinstance(text, str)
            assert len(text) > 0

    def test_qrels_structure(self):
        qr = load_qrels(TREC_DL_2020)
        for qid, pids in qr.items():
            assert isinstance(qid, str)
            assert isinstance(pids, dict)
            for pid, grade in pids.items():
                assert isinstance(pid, str)
                assert isinstance(grade, int)
                assert grade in {0, 1, 2, 3}

    def test_qrels_total_judgments(self):
        qr = load_qrels(TREC_DL_2020)
        total = sum(len(v) for v in qr.values())
        # TREC DL 2020 has ~11k judgments
        assert total > 5000, f"Suspiciously few judgments: {total}"

    def test_qrels_hash_is_deterministic(self):
        h1 = qrels_hash(TREC_DL_2020)
        h2 = qrels_hash(TREC_DL_2020)
        assert h1 == h2

    def test_qrels_hash_starts_with_sha256(self):
        h = qrels_hash(TREC_DL_2020)
        assert h.startswith("sha256:")


@skip_no_data
class TestTrecDl2019:
    def test_query_count(self):
        q = load_queries(TREC_DL_2019)
        assert len(q) == 43, f"Expected 43 queries, got {len(q)}"

    def test_qrels_total_judgments(self):
        qr = load_qrels(TREC_DL_2019)
        total = sum(len(v) for v in qr.values())
        assert total > 4000, f"Suspiciously few judgments: {total}"


@skip_no_data
class TestCombined:
    def test_combined_queries_count(self):
        q = combined_queries()
        assert len(q) == 97  # 54 + 43

    def test_no_qid_overlap(self):
        q19 = load_queries(TREC_DL_2019)
        q20 = load_queries(TREC_DL_2020)
        overlap = set(q19.keys()) & set(q20.keys())
        assert len(overlap) == 0, f"Unexpected qid overlap: {overlap}"

    def test_combined_qrels_superset(self):
        qr = combined_qrels()
        qr19 = load_qrels(TREC_DL_2019)
        qr20 = load_qrels(TREC_DL_2020)
        for qid in qr19:
            assert qid in qr
        for qid in qr20:
            assert qid in qr

    def test_dataset_stats(self):
        stats = dataset_stats(TREC_DL_2020)
        assert stats["num_queries"] == 54
        assert stats["avg_judgments_per_query"] > 0


# ---------------------------------------------------------------------------
# Error handling (no data)
# ---------------------------------------------------------------------------

class TestMissingData:
    def test_missing_queries_raises(self, tmp_path, monkeypatch):
        import evaluation.trec_eval as te

        monkeypatch.setitem(
            te._QUERY_FILES, TREC_DL_2020, tmp_path / "nonexistent.json"
        )
        with pytest.raises(FileNotFoundError):
            load_queries(TREC_DL_2020)

    def test_missing_qrels_raises(self, tmp_path, monkeypatch):
        import evaluation.trec_eval as te

        monkeypatch.setitem(
            te._QREL_FILES, TREC_DL_2020, tmp_path / "nonexistent.json"
        )
        with pytest.raises(FileNotFoundError):
            load_qrels(TREC_DL_2020)


# ---------------------------------------------------------------------------
# Metrics stub contract
# ---------------------------------------------------------------------------

class TestMetrics:
    """Behavioral tests for the implemented metrics."""

    # --- nDCG ---

    def test_ndcg_perfect_ranking_is_one(self):
        from evaluation.metrics import ndcg_at_k
        qrels = {"q1": {"p1": 3, "p2": 2, "p3": 1}}
        run   = {"q1": ["p1", "p2", "p3"]}
        assert ndcg_at_k(qrels, run, k=3) == pytest.approx(1.0)

    def test_ndcg_empty_run_returns_zero(self):
        from evaluation.metrics import ndcg_at_k
        assert ndcg_at_k({}, {}, k=10) == 0.0

    def test_ndcg_all_grades_zero_returns_zero(self):
        from evaluation.metrics import ndcg_at_k
        qrels = {"q1": {"p1": 0, "p2": 0}}
        run   = {"q1": ["p1", "p2"]}
        assert ndcg_at_k(qrels, run, k=10) == 0.0

    def test_ndcg_unjudged_query_skipped(self):
        from evaluation.metrics import ndcg_at_k
        qrels = {"q1": {"p1": 3}}
        run   = {"q1": ["p1"], "q_missing": ["p99"]}
        # q_missing has no qrels entry — should be silently skipped
        assert ndcg_at_k(qrels, run, k=10) == pytest.approx(1.0)

    def test_ndcg_reversed_ranking_below_one(self):
        from evaluation.metrics import ndcg_at_k
        qrels = {"q1": {"p1": 3, "p2": 1}}
        run   = {"q1": ["p2", "p1"]}   # worse order
        score = ndcg_at_k(qrels, run, k=2)
        assert 0.0 < score < 1.0

    # --- MRR ---

    def test_mrr_first_hit_is_one(self):
        from evaluation.metrics import mrr_at_k
        qrels = {"q1": {"p1": 2}}
        run   = {"q1": ["p1", "p2", "p3"]}
        assert mrr_at_k(qrels, run, k=10) == pytest.approx(1.0)

    def test_mrr_second_hit_is_half(self):
        from evaluation.metrics import mrr_at_k
        qrels = {"q1": {"p2": 2}}
        run   = {"q1": ["p_irrelevant", "p2", "p3"]}
        assert mrr_at_k(qrels, run, k=10) == pytest.approx(0.5)

    def test_mrr_no_hit_in_topk_is_zero(self):
        from evaluation.metrics import mrr_at_k
        qrels = {"q1": {"p_relevant": 2}}
        run   = {"q1": ["p1", "p2", "p3"]}
        assert mrr_at_k(qrels, run, k=3) == pytest.approx(0.0)

    def test_mrr_only_first_hit_counts(self):
        from evaluation.metrics import mrr_at_k
        # Both p1 (rank 2) and p2 (rank 3) are relevant — only rank 2 should count
        qrels = {"q1": {"p1": 2, "p2": 2}}
        run   = {"q1": ["p_irr", "p1", "p2"]}
        assert mrr_at_k(qrels, run, k=10) == pytest.approx(0.5)

    # --- Recall ---

    def test_recall_all_retrieved_is_one(self):
        from evaluation.metrics import recall_at_k
        qrels = {"q1": {"p1": 2, "p2": 1}}
        run   = {"q1": ["p1", "p2", "p3"]}
        assert recall_at_k(qrels, run, k=10) == pytest.approx(1.0)

    def test_recall_none_retrieved_is_zero(self):
        from evaluation.metrics import recall_at_k
        qrels = {"q1": {"p_relevant": 2}}
        run   = {"q1": ["p1", "p2", "p3"]}
        assert recall_at_k(qrels, run, k=3) == pytest.approx(0.0)

    def test_recall_partial(self):
        from evaluation.metrics import recall_at_k
        qrels = {"q1": {"p1": 2, "p2": 2, "p3": 2, "p4": 2}}
        run   = {"q1": ["p1", "p2"]}
        assert recall_at_k(qrels, run, k=10) == pytest.approx(0.5)

    def test_recall_denominator_is_full_qrels_not_capped(self):
        from evaluation.metrics import recall_at_k
        # 4 relevant passages, retrieve 2 at k=2 — recall should be 2/4 not 2/2
        qrels = {"q1": {"p1": 2, "p2": 2, "p3": 2, "p4": 2}}
        run   = {"q1": ["p1", "p2"]}
        assert recall_at_k(qrels, run, k=2) == pytest.approx(0.5)

    # --- per_query_metrics ---

    def test_per_query_metrics_structure(self):
        from evaluation.metrics import per_query_metrics
        qrels = {"q1": {"p1": 3}, "q2": {"p2": 2}}
        run   = {"q1": ["p1"], "q2": ["p2"]}
        rows = per_query_metrics(qrels, run, k=10)
        assert len(rows) == 2
        for row in rows:
            assert "qid" in row
            assert "nDCG@10" in row
            assert "Recall@100" in row

    def test_per_query_metrics_skips_unjudged(self):
        from evaluation.metrics import per_query_metrics
        qrels = {"q1": {"p1": 3}}
        run   = {"q1": ["p1"], "q_no_qrels": ["p99"]}
        rows = per_query_metrics(qrels, run, k=10)
        qids = [r["qid"] for r in rows]
        assert "q_no_qrels" not in qids


# ---------------------------------------------------------------------------
# Fixture integrity
# ---------------------------------------------------------------------------

class TestFixture:
    def test_fixture_loads(self):
        fixture_path = Path("tests/fixtures/trec_dl_2020_tiny.json")
        assert fixture_path.exists(), "Fixture file missing"
        with fixture_path.open() as f:
            data = json.load(f)
        assert "passages" in data
        assert "queries" in data
        assert "qrels" in data
        assert len(data["passages"]) == 100
        assert len(data["queries"]) == 10

    def test_fixture_qrel_grades_valid(self):
        with open("tests/fixtures/trec_dl_2020_tiny.json") as f:
            data = json.load(f)
        for qid, pids in data["qrels"].items():
            for pid, grade in pids.items():
                assert grade in {0, 1, 2, 3}
