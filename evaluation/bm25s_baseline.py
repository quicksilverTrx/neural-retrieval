"""
BM25s library baseline on MS MARCO v1 (8.8M passages).

Evaluates on TREC DL 2019 (43 queries) + 2020 (54 queries) = 97 total.
Writes results to benchmarks/results/{timestamp}_bm25s_{hash8}.json per the
canonical schema (benchmarks/SCHEMA.md).

The index is saved to data/bm25s_index/ after the first build and reloaded
on subsequent runs (~5s vs ~30min rebuild from scratch).

Usage:
    python evaluation/bm25s_baseline.py [--limit N] [--top-k 100]

    --limit N   : only index first N passages (smoke-testing; bypasses cache)
    --top-k     : passages to retrieve per query (default=100)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = REPO_ROOT / "data" / "bm25s_index"
sys.path.insert(0, str(REPO_ROOT))

from evaluation.trec_eval import (
    TREC_DL_2019,
    TREC_DL_2020,
    combined_qrels,
    combined_queries,
    load_qrels,
    load_queries,
    qrels_hash,
)


def _corpus_hash(passages: list[tuple[str, str]]) -> str:
    h = hashlib.sha256()
    h.update(str(len(passages)).encode())
    if passages:
        h.update(passages[0][1].encode())
        h.update(passages[-1][1].encode())
    return f"sha256:{h.hexdigest()}"


def load_corpus(limit: int | None) -> list[tuple[str, str]]:
    """Load passages from data/msmarco_passages.jsonl or HF cache."""
    jsonl_path = REPO_ROOT / "data" / "msmarco_passages.jsonl"

    if jsonl_path.exists():
        print(f"Loading corpus from {jsonl_path}")
        passages = []
        with jsonl_path.open() as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                row = json.loads(line)
                passages.append((row["pid"], row["text"]))
                if (i + 1) % 1_000_000 == 0:
                    print(f"  {i+1:,} passages loaded...")
        return passages

    print("msmarco_passages.jsonl not found — loading from HuggingFace cache...")
    from datasets import load_dataset

    ds = load_dataset(
        "Tevatron/msmarco-passage-corpus",
        cache_dir=str(REPO_ROOT / "data" / "corpus"),
        split="train",
    )
    end = limit if limit is not None else len(ds)
    passages = [(str(row["docid"]), row["text"]) for row in ds.select(range(end))]
    return passages


def build_bm25s_index(passages: list[tuple[str, str]], save: bool = True):
    """Build (or load cached) bm25s index. Returns (retriever, pid_list)."""
    import bm25s

    pids_cache = INDEX_DIR / "pids.json"

    if save and INDEX_DIR.exists() and pids_cache.exists():
        print(f"Loading cached BM25s index from {INDEX_DIR} ...")
        t0 = time.perf_counter()
        retriever = bm25s.BM25.load(str(INDEX_DIR), load_corpus=False)
        pids = json.loads(pids_cache.read_text())
        print(f"  Index loaded in {time.perf_counter() - t0:.1f}s")
        return retriever, pids

    pids = [p[0] for p in passages]
    texts = [p[1] for p in passages]

    print(f"Tokenizing {len(texts):,} passages...")
    t0 = time.perf_counter()
    corpus_tokens = bm25s.tokenize(texts, stopwords="en")
    print(f"  Tokenization: {time.perf_counter() - t0:.1f}s")

    print("Building BM25s index...")
    t0 = time.perf_counter()
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    print(f"  Index build: {time.perf_counter() - t0:.1f}s")

    if save:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        retriever.save(str(INDEX_DIR))
        pids_cache.write_text(json.dumps(pids))
        print(f"  Index saved to {INDEX_DIR}")

    return retriever, pids


def retrieve(
    retriever,
    pids: list[str],
    queries: dict[str, str],
    top_k: int,
) -> tuple[dict[str, list[str]], dict[str, float]]:
    """Retrieve top_k passages for each query.

    Returns
    -------
    run : dict[qid → list[pid]] ordered by decreasing BM25 score
    latencies : dict[qid → latency_ms]
    """
    import bm25s

    run: dict[str, list[str]] = {}
    latencies: dict[str, float] = {}

    qids = list(queries.keys())
    query_texts = [queries[qid] for qid in qids]

    print(f"Retrieving top-{top_k} for {len(qids)} queries...")
    for qid, qtext in zip(qids, query_texts):
        t0 = time.perf_counter()
        query_tokens = bm25s.tokenize([qtext], stopwords="en")
        results, _ = retriever.retrieve(query_tokens, k=min(top_k, len(pids)))
        run[qid] = [pids[doc_idx] for doc_idx in results[0]]
        latencies[qid] = (time.perf_counter() - t0) * 1000

    return run, latencies


def compute_latency_stats(latencies: dict[str, float]) -> dict:
    import statistics

    vals = sorted(latencies.values())
    n = len(vals)
    return {
        "p50": vals[int(n * 0.50)],
        "p95": vals[int(n * 0.95)],
        "p99": vals[min(int(n * 0.99), n - 1)],
        "mean": statistics.mean(vals),
        "stages": {"bm25": statistics.mean(vals)},
    }


def compute_metrics(qrels, run, k: int = 10) -> dict:
    from evaluation.metrics import mrr_at_k, ndcg_at_k, recall_at_k

    return {
        f"nDCG@{k}": ndcg_at_k(qrels, run, k=k),
        f"MRR@{k}": mrr_at_k(qrels, run, k=k),
        "Recall@100": recall_at_k(qrels, run, k=100),
    }


def write_results(
    experiment: str,
    config: dict,
    dataset: str,
    metrics: dict,
    latency_ms: dict,
    run: dict[str, list[str]],
    latencies: dict[str, float],
    notes: str = "",
) -> Path:
    """Write canonical result JSON per benchmarks/SCHEMA.md."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cfg_str = json.dumps(config, sort_keys=True)
    config_hash = hashlib.sha256(cfg_str.encode()).hexdigest()[:8]
    filename = f"{ts}_{experiment}_{config_hash}.json"

    results_dir = REPO_ROOT / "benchmarks" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        import subprocess

        commit_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        commit_sha = "unknown"

    try:
        import torch

        torch_ver = torch.__version__
    except Exception:
        torch_ver = "n/a"

    per_query = [
        {"qid": qid, "latency_ms": latencies.get(qid, 0)}
        for qid in run
    ]

    doc = {
        "schema_version": "1.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "experiment": experiment,
        "config": config,
        "dataset": dataset,
        "metrics": metrics,
        "latency_ms": latency_ms,
        "per_query": per_query,
        "environment": {
            "python": platform.python_version(),
            "torch": torch_ver,
            "faiss": "n/a",
            "gpu": "cpu",
            "commit_sha": commit_sha,
        },
        "notes": notes,
    }

    out_path = results_dir / filename
    with out_path.open("w") as f:
        json.dump(doc, f, indent=2)
    print(f"Results written to {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="BM25s baseline")
    parser.add_argument("--limit", type=int, default=None, help="Limit corpus size (testing)")
    parser.add_argument("--top-k", type=int, default=100, help="Top-k to retrieve per query")
    args = parser.parse_args()

    print("=" * 60)
    print("BM25s Library Baseline — MS MARCO v1")
    print("=" * 60)

    # Corpus
    passages = load_corpus(limit=args.limit)
    print(f"Corpus: {len(passages):,} passages")
    corpus_h = _corpus_hash(passages)

    # Index (cached after first build)
    use_cache = args.limit is None
    retriever, pids = build_bm25s_index(passages, save=use_cache)

    # Eval data
    queries_combined = combined_queries()
    qrels_2019 = load_qrels(TREC_DL_2019)
    qrels_2020 = load_qrels(TREC_DL_2020)
    qrels_combined = combined_qrels()
    q19_hash = qrels_hash(TREC_DL_2019)
    q20_hash = qrels_hash(TREC_DL_2020)

    print(f"Queries: {len(queries_combined)} (DL2019={len(qrels_2019)}, DL2020={len(qrels_2020)})")

    # Retrieve
    run, latencies = retrieve(retriever, pids, queries_combined, top_k=args.top_k)

    # Per-year and combined metrics
    run_2019 = {qid: pids_list for qid, pids_list in run.items() if qid in qrels_2019}
    run_2020 = {qid: pids_list for qid, pids_list in run.items() if qid in qrels_2020}

    metrics_2019 = compute_metrics(qrels_2019, run_2019)
    metrics_2020 = compute_metrics(qrels_2020, run_2020)
    metrics_combined = compute_metrics(qrels_combined, run)

    print("\n--- Metrics ---")
    print(f"DL2019 ({len(run_2019)}q): nDCG@10={metrics_2019['nDCG@10']:.4f}  MRR@10={metrics_2019['MRR@10']:.4f}  Recall@100={metrics_2019['Recall@100']:.4f}")
    print(f"DL2020 ({len(run_2020)}q): nDCG@10={metrics_2020['nDCG@10']:.4f}  MRR@10={metrics_2020['MRR@10']:.4f}  Recall@100={metrics_2020['Recall@100']:.4f}")
    print(f"Combined (97q):  nDCG@10={metrics_combined['nDCG@10']:.4f}  MRR@10={metrics_combined['MRR@10']:.4f}  Recall@100={metrics_combined['Recall@100']:.4f}")

    gate_2019 = metrics_2019["nDCG@10"] > 0.42
    gate_2020 = metrics_2020["nDCG@10"] > 0.40
    print(f"\nGate DL2019 (>0.42): {'PASS' if gate_2019 else 'FAIL'}")
    print(f"Gate DL2020 (>0.40): {'PASS' if gate_2020 else 'FAIL'}")

    latency_stats = compute_latency_stats(latencies)
    print(f"Latency — P50: {latency_stats['p50']:.1f}ms  P99: {latency_stats['p99']:.1f}ms")

    config = {
        "retriever": "bm25s",
        "bm25s_version": "library_baseline",
        "top_k": args.top_k,
        "corpus_size": len(passages),
        "corpus_hash": corpus_h,
        "qrels_hash": f"{q19_hash}+{q20_hash}",
    }

    metrics_to_write = {
        **{f"DL2019_{k}": v for k, v in metrics_2019.items()},
        **{f"DL2020_{k}": v for k, v in metrics_2020.items()},
        **{f"combined_{k}": v for k, v in metrics_combined.items()},
    }

    write_results(
        experiment="bm25s",
        config=config,
        dataset="trec_dl_2019+2020",
        metrics=metrics_to_write,
        latency_ms=latency_stats,
        run=run,
        latencies=latencies,
        notes="BM25s library baseline, full 8.8M corpus." + (f" Corpus limited to {args.limit:,}." if args.limit else ""),
    )


if __name__ == "__main__":
    main()
