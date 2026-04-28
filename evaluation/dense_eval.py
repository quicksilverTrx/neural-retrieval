"""Evaluate dense retrieval on TREC DL 2019+2020.

Prerequisites: run encode_corpus.py + build_faiss.py.

Usage:
    python evaluation/dense_eval.py --model all-MiniLM-L6-v2
    python evaluation/dense_eval.py --model intfloat/e5-small-v2
    python evaluation/dense_eval.py --model all-MiniLM-L6-v2 --nprobe 32
    python evaluation/dense_eval.py --model all-MiniLM-L6-v2 --sweep
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

from evaluation.metrics import mrr_at_k, ndcg_at_k, per_query_metrics, recall_at_k
from evaluation.trec_eval import (
    TREC_DL_2019,
    TREC_DL_2020,
    combined_qrels,
    combined_queries,
    load_qrels,
    qrels_hash,
)
from retrieval.dense.encoder import SentenceEncoder
from retrieval.dense.faiss_index import FAISSIVFPQIndex


def _slug(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_").lower()


def _commit_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def encode_queries(enc: SentenceEncoder, queries: dict[str, str]) -> tuple[list[str], object]:
    qids = list(queries.keys())
    texts = [queries[qid] for qid in qids]
    vecs = enc.encode_batch(texts, is_query=True, show_progress=False)
    return qids, vecs


def retrieve_all(
    idx: FAISSIVFPQIndex,
    query_vecs,
    qids: list[str],
    top_k: int,
    nprobe: int,
) -> tuple[dict[str, list[str]], dict[str, float]]:
    run: dict[str, list[str]] = {}
    latencies: dict[str, float] = {}
    for i, qid in enumerate(qids):
        q = query_vecs[i : i + 1]
        t0 = time.perf_counter()
        pids_per_q, _ = idx.search(q, top_k=top_k, nprobe=nprobe)
        latencies[qid] = (time.perf_counter() - t0) * 1000
        run[qid] = pids_per_q[0]
    return run, latencies


def _lat_stats(latencies: dict[str, float]) -> dict:
    vals = sorted(latencies.values())
    n = len(vals)
    return {
        "p50": vals[n // 2],
        "p95": vals[int(n * 0.95)],
        "p99": vals[min(int(n * 0.99), n - 1)],
        "mean": statistics.mean(vals),
    }


def write_results(payload: dict) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cfg_hash = hashlib.sha256(
        json.dumps(payload["config"], sort_keys=True).encode()
    ).hexdigest()[:8]
    out = REPO_ROOT / "benchmarks" / "results" / f"{ts}_dense_{cfg_hash}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"Results written to {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Dense retrieval eval")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--nprobe", type=int, default=16)
    parser.add_argument("--sweep", action="store_true",
                        help="Run nprobe sweep (1,4,8,16,32,64) and write to JSON.")
    parser.add_argument("--faiss-dir-suffix", default="",
                        help="Optional suffix to faiss dir name (e.g. '_flat' for IVF-Flat).")
    args = parser.parse_args()

    slug = _slug(args.model)
    faiss_dir = REPO_ROOT / "data" / "faiss" / (slug + args.faiss_dir_suffix)

    if not faiss_dir.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {faiss_dir}. Run build_faiss.py first."
        )

    print("=" * 60)
    print(f"Dense Retrieval Eval ({args.model})")
    print("=" * 60)

    queries = combined_queries()
    qrels_2019 = load_qrels(TREC_DL_2019)
    qrels_2020 = load_qrels(TREC_DL_2020)
    all_qrels = combined_qrels()
    q19_hash = qrels_hash(TREC_DL_2019)
    q20_hash = qrels_hash(TREC_DL_2020)

    print(f"Loading encoder: {args.model}")
    enc = SentenceEncoder(model_name=args.model)

    print(f"Loading FAISS index from {faiss_dir}")
    idx = FAISSIVFPQIndex.load(faiss_dir)
    print(f"  {idx._index.ntotal:,} vectors indexed")

    print(f"Encoding {len(queries)} queries ...")
    qids, query_vecs = encode_queries(enc, queries)

    sweep_results = None
    if args.sweep:
        print("\n--- nprobe sweep ---")
        sweep_results = idx.nprobe_sweep(
            query_vecs, all_qrels, qids, top_k=args.top_k
        )

    print(f"\nRetrieving top-{args.top_k} at nprobe={args.nprobe} ...")
    run, latencies = retrieve_all(idx, query_vecs, qids, top_k=args.top_k, nprobe=args.nprobe)
    lat_stats = _lat_stats(latencies)
    print(f"  P50={lat_stats['p50']:.1f}ms  P99={lat_stats['p99']:.1f}ms")

    run_2019 = {qid: pids for qid, pids in run.items() if qid in qrels_2019}
    run_2020 = {qid: pids for qid, pids in run.items() if qid in qrels_2020}

    def _m(qr, r):
        return {
            "nDCG@10": ndcg_at_k(qr, r, k=10),
            "MRR@10": mrr_at_k(qr, r, k=10),
            "Recall@100": recall_at_k(qr, r, k=100),
        }

    m19 = _m(qrels_2019, run_2019)
    m20 = _m(qrels_2020, run_2020)

    print(f"\nDL2019 ({len(run_2019)}q): nDCG@10={m19['nDCG@10']:.4f} "
          f"MRR@10={m19['MRR@10']:.4f} Recall@100={m19['Recall@100']:.4f}")
    print(f"DL2020 ({len(run_2020)}q): nDCG@10={m20['nDCG@10']:.4f} "
          f"MRR@10={m20['MRR@10']:.4f} Recall@100={m20['Recall@100']:.4f}")

    payload = {
        "schema_version": "1.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "config": {
            "retriever": "dense_ivfpq",
            "encoder": args.model,
            "index": {
                "type": "IVFPQ",
                "nlist": idx.nlist,
                "m": idx.m,
                "nbits": idx.nbits,
                "nprobe": args.nprobe,
            },
            "top_k": args.top_k,
            "qrels_hash": f"{q19_hash}+{q20_hash}",
        },
        "dataset": "trec_dl_2019+2020",
        "metrics": {
            **{f"DL2019_{k}": v for k, v in m19.items()},
            **{f"DL2020_{k}": v for k, v in m20.items()},
        },
        "latency_ms": lat_stats,
        "nprobe_sweep": sweep_results,
        "per_query": per_query_metrics(qrels_2020, run_2020, k=10),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "commit_sha": _commit_sha(),
        },
    }
    write_results(payload)


if __name__ == "__main__":
    main()
