"""Per-stage latency decomposition report.

Measures P50/P95/P99 latency for each pipeline stage across N sample queries:
    bm25_retrieval   Inverted index scan + BM25 scoring
    dense_encode     Query embedding with sentence encoder
    faiss_search     FAISS IVF-PQ approximate nearest-neighbour
    rrf_fusion       Reciprocal Rank Fusion
    acl_filter       Post-retrieval ACL permission filtering
    full_query       End-to-end (wall clock)

Runs with whatever indexes are present:
    - BM25 only: dense/faiss/rrf stages skipped
    - BM25 + FAISS: all stages measured

Usage:
    python evaluation/latency_report.py
    python evaluation/latency_report.py --queries 100 --nprobe 16
    python evaluation/latency_report.py --bm25-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evaluation.trec_eval import combined_queries


def _slug(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_").lower()


def _commit_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _lat_stats(latencies: list[float]) -> dict:
    if not latencies:
        return {"p50": None, "p95": None, "p99": None, "mean": None, "n": 0}
    vals = sorted(latencies)
    n = len(vals)
    return {
        "p50": round(vals[n // 2], 3),
        "p95": round(vals[int(n * 0.95)], 3),
        "p99": round(vals[min(int(n * 0.99), n - 1)], 3),
        "mean": round(statistics.mean(vals), 3),
        "n": n,
    }


def _load_bm25(bm25_path: Path):
    if not bm25_path.exists():
        return None
    from retrieval.inverted_index import BM25Retriever, BM25Scorer
    from retrieval.inverted_index.persistence import load_index

    print(f"Loading BM25 index from {bm25_path} ...")
    index, _, _ = load_index(bm25_path)
    print(f"  {index.num_docs:,} docs loaded")
    return BM25Retriever(index=index, scorer=BM25Scorer())


def _load_dense(model: str):
    from retrieval.dense.encoder import SentenceEncoder
    from retrieval.dense.faiss_index import FAISSIVFPQIndex

    slug = _slug(model)
    faiss_dir = REPO_ROOT / "data" / "faiss" / slug

    if not faiss_dir.exists():
        return None, None

    print(f"Loading encoder: {model} ...")
    enc = SentenceEncoder(model_name=model)

    print(f"Loading FAISS index from {faiss_dir} ...")
    idx = FAISSIVFPQIndex.load(faiss_dir)
    print(f"  {idx._index.ntotal:,} vectors indexed")
    return enc, idx


def _load_acl():
    from retrieval.acl import ACLFilter, PassageACL

    acl_path = REPO_ROOT / "data" / "acl" / "passage_acl.json"
    if not acl_path.exists():
        return None

    acl = PassageACL()
    acl.load(REPO_ROOT / "data" / "acl")
    return ACLFilter(acl)


def write_results(payload: dict) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cfg_hash = hashlib.sha256(
        json.dumps(payload["config"], sort_keys=True).encode()
    ).hexdigest()[:8]
    out = REPO_ROOT / "benchmarks" / "results" / f"{ts}_latency_{cfg_hash}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nResults written to {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-stage latency report")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--queries", type=int, default=100,
                        help="Number of queries to sample (default: 100)")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--nprobe", type=int, default=16)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--bm25-path", default=None,
                        help="Path to BM25 index (default: data/custom_bm25_8m.bin)")
    parser.add_argument("--bm25-only", action="store_true",
                        help="Skip dense/FAISS stages (BM25-only mode)")
    parser.add_argument("--user-role", default="engineer",
                        help="ACL role for filter stage (default: engineer)")
    args = parser.parse_args()

    bm25_path = (
        Path(args.bm25_path) if args.bm25_path
        else REPO_ROOT / "data" / "custom_bm25_8m.bin"
    )

    print("=" * 60)
    print("Per-Stage Latency Decomposition")
    print("=" * 60)

    bm25_retriever = _load_bm25(bm25_path)
    if bm25_retriever is None:
        print(f"ERROR: BM25 index not found at {bm25_path}")
        print("Run bm25_eval.py --index-path first.")
        sys.exit(1)

    encoder, faiss_idx = (None, None) if args.bm25_only else _load_dense(args.model)
    acl_filter = _load_acl()

    from retrieval.fusion.rrf import fuse_scored

    all_queries = combined_queries()
    query_items = list(all_queries.items())[: args.queries]
    print(f"\nRunning {len(query_items)} queries ...")

    bm25_lats: list[float] = []
    encode_lats: list[float] = []
    faiss_lats: list[float] = []
    rrf_lats: list[float] = []
    acl_lats: list[float] = []
    full_lats: list[float] = []

    query_vecs = None
    if encoder is not None:
        texts = [text for _, text in query_items]
        t0 = time.perf_counter()
        query_vecs = encoder.encode_batch(texts, is_query=True, show_progress=False)
        batch_ms = (time.perf_counter() - t0) * 1000
        per_query_encode_ms = batch_ms / len(query_items)
        print(f"  Batch encode: {batch_ms:.1f}ms total → {per_query_encode_ms:.2f}ms/query")

    for i, (qid, query_text) in enumerate(query_items):
        full_t0 = time.perf_counter()

        t0 = time.perf_counter()
        bm25_results, _ = bm25_retriever.retrieve_timed(query_text, top_k=args.top_k)
        # str() cast: BM25 returns int doc_ids, FAISS pids.json stores str.
        # Without normalising, RRF would treat the same passage as two
        # different ids and double-count the union.
        bm25_results = [(str(d), s) for d, s in bm25_results]
        bm25_lats.append((time.perf_counter() - t0) * 1000)

        dense_results: list[tuple[str, float]] = []

        if encoder is not None and query_vecs is not None and faiss_idx is not None:
            t0 = time.perf_counter()
            q_vec = query_vecs[i : i + 1]
            encode_lats.append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            pids_per_q, dists = faiss_idx.search(q_vec, top_k=args.top_k, nprobe=args.nprobe)
            faiss_lats.append((time.perf_counter() - t0) * 1000)
            dense_results = [(p, -float(d)) for p, d in zip(pids_per_q[0], dists[0].tolist())]

            t0 = time.perf_counter()
            fused = fuse_scored(bm25_results, dense_results, k=args.rrf_k)
            rrf_lats.append((time.perf_counter() - t0) * 1000)
            top_results = fused[: args.top_k]
        else:
            top_results = bm25_results[: args.top_k]

        if acl_filter is not None:
            oversample = list(top_results)
            t0 = time.perf_counter()
            acl_filter.filter(oversample, user_role=args.user_role, top_k=args.top_k)
            acl_lats.append((time.perf_counter() - t0) * 1000)

        full_lats.append((time.perf_counter() - full_t0) * 1000)

        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(query_items)} queries processed ...")

    stages = [
        ("bm25_retrieval", bm25_lats),
        ("dense_encode", encode_lats),
        ("faiss_search", faiss_lats),
        ("rrf_fusion", rrf_lats),
        ("acl_filter", acl_lats),
        ("full_query", full_lats),
    ]

    print("\n" + "─" * 70)
    print(f"{'Stage':<20} {'P50 (ms)':>10} {'P95 (ms)':>10} {'P99 (ms)':>10} {'N':>6}")
    print("─" * 70)

    stage_stats = {}
    for stage_name, lats in stages:
        stats = _lat_stats(lats)
        stage_stats[stage_name] = stats
        if stats["n"] == 0:
            print(f"  {stage_name:<18} {'—':>10} {'—':>10} {'—':>10} {'—':>6}  (skipped)")
        else:
            print(
                f"  {stage_name:<18} {stats['p50']:>10.2f} {stats['p95']:>10.2f}"
                f" {stats['p99']:>10.2f} {stats['n']:>6}"
            )
    print("─" * 70)

    full_p99 = stage_stats["full_query"]["p99"]
    gate_pass = full_p99 is not None and full_p99 < 20.0
    if full_p99 is None:
        print("\nShip gate (full_query P99 < 20ms): no data")
    else:
        print(f"\nShip gate (full_query P99 < 20ms): "
              f"{'PASS' if gate_pass else 'FAIL'} (P99 = {full_p99:.1f}ms)")

    config = {
        "num_queries": len(query_items),
        "top_k": args.top_k,
        "nprobe": args.nprobe,
        "rrf_k": args.rrf_k,
        "encoder": args.model if not args.bm25_only else None,
        "bm25_path": str(bm25_path),
        "user_role": args.user_role,
        "bm25_only": args.bm25_only,
    }
    payload = {
        "schema_version": "1.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "config": config,
        "stage_latency_ms": stage_stats,
        "ship_gate": {
            "pass": gate_pass,
            "metric": "full_query P99 < 20ms",
            "value": full_p99,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "commit_sha": _commit_sha(),
        },
    }
    write_results(payload)


if __name__ == "__main__":
    main()
