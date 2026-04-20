"""
TREC DL 2019 + 2020 query and qrels loader.

Loads pre-downloaded JSON files (written by bootstrap_data.sh).
Returns queries and qrels in the canonical format used throughout
the evaluation pipeline.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

TREC_DL_2019 = "trec_dl_2019"
TREC_DL_2020 = "trec_dl_2020"

_QUERY_FILES = {
    TREC_DL_2019: DATA_DIR / "queries" / "trec_dl_2019_queries.json",
    TREC_DL_2020: DATA_DIR / "queries" / "trec_dl_2020_queries.json",
}
_QREL_FILES = {
    TREC_DL_2019: DATA_DIR / "qrels" / "trec_dl_2019_qrels.json",
    TREC_DL_2020: DATA_DIR / "qrels" / "trec_dl_2020_qrels.json",
}


def load_queries(dataset: str) -> dict[str, str]:
    """Load queries for a TREC DL dataset.

    Returns
    -------
    dict mapping qid (str) → query text (str)
    """
    path = _QUERY_FILES[dataset]
    if not path.exists():
        raise FileNotFoundError(
            f"Query file not found: {path}. Run scripts/bootstrap_data.sh first."
        )
    with path.open() as f:
        return json.load(f)


def load_qrels(dataset: str) -> dict[str, dict[str, int]]:
    """Load qrels for a TREC DL dataset.

    Returns
    -------
    dict mapping qid (str) → {pid (str) → relevance_grade (int)}
    Grades: 0=not relevant, 1=related, 2=highly relevant, 3=perfectly relevant
    """
    path = _QREL_FILES[dataset]
    if not path.exists():
        raise FileNotFoundError(
            f"Qrels file not found: {path}. Run scripts/bootstrap_data.sh first."
        )
    with path.open() as f:
        return json.load(f)


def qrels_hash(dataset: str) -> str:
    """SHA256 hash of the qrels file for result JSON integrity tracking."""
    path = _QREL_FILES[dataset]
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return f"sha256:{h.hexdigest()}"


def combined_queries() -> dict[str, str]:
    """Return all 97 queries (TREC DL 2019 + 2020) merged."""
    q = load_queries(TREC_DL_2019)
    q.update(load_queries(TREC_DL_2020))
    return q


def combined_qrels() -> dict[str, dict[str, int]]:
    """Return merged qrels for all 97 queries."""
    qr = load_qrels(TREC_DL_2019)
    qr.update(load_qrels(TREC_DL_2020))
    return qr


def dataset_stats(dataset: str) -> dict:
    """Return summary stats for a dataset (for sanity checking)."""
    queries = load_queries(dataset)
    qrels = load_qrels(dataset)
    total_judgments = sum(len(v) for v in qrels.values())
    return {
        "dataset": dataset,
        "num_queries": len(queries),
        "num_judged_passages": total_judgments,
        "avg_judgments_per_query": total_judgments / max(len(queries), 1),
    }
