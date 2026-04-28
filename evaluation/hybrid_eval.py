"""Hybrid retrieval eval: BM25 + dense + RRF + alpha-sweep.

Steps:
    1. Load BM25 index + Dense FAISS index
    2. Retrieve top-100 with BM25 for all 97 TREC queries
    3. Retrieve top-100 with dense encoder for all 97 queries
    4. Fuse with RRF
    5. Alpha-sweep: for α in [0.0, 0.1, ..., 1.0], combine min-max-normalised
       scores as α*bm25_norm + (1-α)*dense_norm
    6. Compute nDCG@10 / MRR@10 / Recall@100 for each system
    7. Write ablation table JSON

Usage:
    python evaluation/hybrid_eval.py
    python evaluation/hybrid_eval.py --model intfloat/e5-small-v2
    python evaluation/hybrid_eval.py --no-alpha-sweep
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
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
    combined_queries,
    load_qrels,
    qrels_hash,
)
from retrieval.dense.encoder import SentenceEncoder
from retrieval.dense.faiss_index import FAISSIVFPQIndex
from retrieval.fusion.rrf import fuse_scored
from retrieval.inverted_index import BM25Retriever, BM25Scorer
from retrieval.inverted_index.persistence import load_index

BM25_INDEX_PATH = REPO_ROOT / "data" / "custom_bm25_8m.bin"
ALPHA_STEPS = [round(a * 0.1, 1) for a in range(11)]


def _slug(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_").lower()


def _commit_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _metrics(qrels: dict, run: dict[str, list[str]]) -> dict:
    return {
        "nDCG@10": ndcg_at_k(qrels, run, k=10),
        "MRR@10": mrr_at_k(qrels, run, k=10),
        "Recall@100": recall_at_k(qrels, run, k=100),
    }


def _normalise(scored: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Map scores to [0, 1] with min-max normalisation."""
    if not scored:
        return scored
    scores = [s for _, s in scored]
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [(d, 1.0) for d, _ in scored]
    return [(d, (s - lo) / (hi - lo)) for d, s in scored]


def alpha_fuse(
    bm25_norm: list[tuple[str, float]],
    dense_norm: list[tuple[str, float]],
    alpha: float,
    top_k: int = 100,
) -> list[str]:
    """α*bm25 + (1-α)*dense over normalised scores."""
    bm25_map = dict(bm25_norm)
    dense_map = dict(dense_norm)
    all_docs = set(bm25_map) | set(dense_map)
    combined = {
        d: alpha * bm25_map.get(d, 0.0) + (1 - alpha) * dense_map.get(d, 0.0)
        for d in all_docs
    }
    ranked = sorted(combined, key=combined.__getitem__, reverse=True)
    return ranked[:top_k]


def bm25_retrieve_all(
    retriever: BM25Retriever,
    queries: dict[str, str],
    top_k: int = 100,
) -> tuple[dict[str, list[tuple[str, float]]], dict[str, float]]:
    run_scored: dict[str, list[tuple[str, float]]] = {}
    latencies: dict[str, float] = {}
    for qid, text in queries.items():
        results, lat = retriever.retrieve_timed(text, top_k=top_k)
        # BM25Retriever returns int doc_ids; cast to str for FAISS/qrels parity
        run_scored[qid] = [(str(d), s) for d, s in results]
        latencies[qid] = lat
    return run_scored, latencies


def dense_retrieve_all(
    idx: FAISSIVFPQIndex,
    enc: SentenceEncoder,
    queries: dict[str, str],
    top_k: int = 100,
    nprobe: int = 16,
) -> tuple[dict[str, list[tuple[str, float]]], dict[str, float]]:
    qids = list(queries.keys())
    texts = [queries[q] for q in qids]
    vecs = enc.encode_batch(texts, is_query=True, show_progress=False)

    run_scored: dict[str, list[tuple[str, float]]] = {}
    latencies: dict[str, float] = {}
    for i, qid in enumerate(qids):
        q = vecs[i : i + 1]
        t0 = time.perf_counter()
        pids_per_q, dists = idx.search(q, top_k=top_k, nprobe=nprobe)
        latencies[qid] = (time.perf_counter() - t0) * 1000
        # FAISS returns L2 distances; smaller is better. Convert to similarity
        # so min-max normalisation aligns with BM25 (higher = better).
        run_scored[qid] = [(pid, -float(d)) for pid, d in zip(pids_per_q[0], dists[0].tolist())]
    return run_scored, latencies


def write_results(payload: dict) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cfg_hash = hashlib.sha256(
        json.dumps(payload["config"], sort_keys=True).encode()
    ).hexdigest()[:8]
    out = REPO_ROOT / "benchmarks" / "results" / f"{ts}_hybrid_{cfg_hash}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"Results written to {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid retrieval eval")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--nprobe", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--no-alpha-sweep", action="store_true",
                        help="Skip alpha sweep (RRF-only run)")
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--faiss-dir-suffix", default="",
                        help="Suffix to FAISS dir name (e.g. '_flat' for IVF-Flat).")
    args = parser.parse_args()

    slug = _slug(args.model)
    faiss_dir = REPO_ROOT / "data" / "faiss" / (slug + args.faiss_dir_suffix)

    for path, label in [(BM25_INDEX_PATH, "BM25 index"), (faiss_dir, "FAISS index")]:
        if not path.exists():
            print(f"ERROR: {label} not found at {path}")
            print("Run bm25_eval.py --index-path and build_faiss.py first.")
            sys.exit(1)

    print("=" * 60)
    print(f"Hybrid Fusion Eval  model={args.model}")
    print("=" * 60)

    queries = combined_queries()
    qrels_2019 = load_qrels(TREC_DL_2019)
    qrels_2020 = load_qrels(TREC_DL_2020)

    print(f"Loading BM25 index from {BM25_INDEX_PATH} ...")
    index, _, _ = load_index(BM25_INDEX_PATH)
    scorer = BM25Scorer(k1=args.k1, b=args.b)
    retriever = BM25Retriever(index=index, scorer=scorer)

    print(f"Loading dense encoder: {args.model}")
    enc = SentenceEncoder(model_name=args.model)
    print(f"Loading FAISS index from {faiss_dir}")
    faiss_idx = FAISSIVFPQIndex.load(faiss_dir)
    print(f"  {faiss_idx._index.ntotal:,} vectors indexed")

    print(f"\nBM25: Retrieving top-{args.top_k} for {len(queries)} queries ...")
    bm25_run, bm25_lat = bm25_retrieve_all(retriever, queries, top_k=args.top_k)

    print(f"\nDense: Retrieving top-{args.top_k} (nprobe={args.nprobe}) ...")
    dense_run, dense_lat = dense_retrieve_all(
        faiss_idx, enc, queries, top_k=args.top_k, nprobe=args.nprobe
    )

    print(f"\nRRF fusion (k={args.rrf_k}) ...")
    rrf_run: dict[str, list[str]] = {}
    for qid in queries:
        fused = fuse_scored(bm25_run.get(qid, []), dense_run.get(qid, []), k=args.rrf_k)
        rrf_run[qid] = [d for d, _ in fused[: args.top_k]]

    best_alpha = None
    best_alpha_ndcg = -1.0
    alpha_sweep_results = []

    if not args.no_alpha_sweep:
        print(f"\nAlpha sweep ({ALPHA_STEPS}) ...")
        for alpha in ALPHA_STEPS:
            alpha_run: dict[str, list[str]] = {}
            for qid in queries:
                bm25_norm = _normalise(bm25_run.get(qid, []))
                dense_norm = _normalise(dense_run.get(qid, []))
                alpha_run[qid] = alpha_fuse(bm25_norm, dense_norm, alpha, top_k=args.top_k)

            run_2020 = {qid: pids for qid, pids in alpha_run.items() if qid in qrels_2020}
            m = _metrics(qrels_2020, run_2020)
            alpha_sweep_results.append({"alpha": alpha, **m})
            print(f"  α={alpha:.1f}: nDCG@10={m['nDCG@10']:.4f}  MRR@10={m['MRR@10']:.4f}")

            if m["nDCG@10"] > best_alpha_ndcg:
                best_alpha_ndcg = m["nDCG@10"]
                best_alpha = alpha

        print(f"\n  Best α = {best_alpha} (DL2020 nDCG@10 = {best_alpha_ndcg:.4f})")

    def _split_run(run_scored):
        return {qid: [d for d, _ in scored] for qid, scored in run_scored.items()}

    bm25_2020 = {qid: v for qid, v in _split_run(bm25_run).items() if qid in qrels_2020}
    dense_2020 = {qid: v for qid, v in _split_run(dense_run).items() if qid in qrels_2020}
    rrf_2020 = {qid: v for qid, v in rrf_run.items() if qid in qrels_2020}

    m_bm25 = _metrics(qrels_2020, bm25_2020)
    m_dense = _metrics(qrels_2020, dense_2020)
    m_rrf = _metrics(qrels_2020, rrf_2020)

    gate_pass = m_rrf["nDCG@10"] > m_dense["nDCG@10"]

    print("\n" + "─" * 60)
    print(f"{'System':<20} {'nDCG@10':>9} {'MRR@10':>9} {'Recall@100':>12}")
    print("─" * 60)
    for name, m in [
        ("BM25 only", m_bm25),
        ("Dense only", m_dense),
        ("RRF hybrid", m_rrf),
    ]:
        print(f"{name:<20} {m['nDCG@10']:>9.4f} {m['MRR@10']:>9.4f} {m['Recall@100']:>12.4f}")

    if best_alpha is not None:
        best_run = {}
        for qid in queries:
            bm25_norm = _normalise(bm25_run.get(qid, []))
            dense_norm = _normalise(dense_run.get(qid, []))
            best_run[qid] = alpha_fuse(bm25_norm, dense_norm, best_alpha, top_k=args.top_k)
        best_2020 = {qid: v for qid, v in best_run.items() if qid in qrels_2020}
        m_best = _metrics(qrels_2020, best_2020)
        print(f"{'Best α='+str(best_alpha):<20} {m_best['nDCG@10']:>9.4f} "
              f"{m_best['MRR@10']:>9.4f} {m_best['Recall@100']:>12.4f}")

    print("─" * 60)
    print(f"\nShip gate (RRF nDCG@10 > Dense nDCG@10 on DL2020): "
          f"{'PASS' if gate_pass else 'FAIL'}")

    payload = {
        "schema_version": "1.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "config": {
            "encoder": args.model,
            "bm25": {"k1": args.k1, "b": args.b},
            "dense": {"nprobe": args.nprobe},
            "rrf_k": args.rrf_k,
            "top_k": args.top_k,
            "qrels_hash": f"{qrels_hash(TREC_DL_2019)}+{qrels_hash(TREC_DL_2020)}",
        },
        "dataset": "trec_dl_2020",
        "ablation": {
            "BM25_only": m_bm25,
            "Dense_only": m_dense,
            "RRF_hybrid": m_rrf,
        },
        "alpha_sweep": alpha_sweep_results,
        "best_alpha": best_alpha,
        "ship_gate": {
            "pass": gate_pass,
            "metric": "DL2020 nDCG@10 RRF > Dense",
            "rrf_ndcg": round(m_rrf["nDCG@10"], 4),
            "dense_ndcg": round(m_dense["nDCG@10"], 4),
        },
        "per_query_rrf": per_query_metrics(qrels_2020, rrf_2020, k=10),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "commit_sha": _commit_sha(),
        },
    }
    write_results(payload)


if __name__ == "__main__":
    main()
