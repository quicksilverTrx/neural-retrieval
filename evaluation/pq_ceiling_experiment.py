"""PQ-ceiling experiment: build IVF-PQ m=32 and IVF-SQ8 from existing embeddings,
evaluate Recall@100 on TREC DL 2020+2019, compare against the m=16 baseline.

Reads the same data/embeddings/all_minilm_l6_v2/ that the production index uses.
Writes new indexes to data/faiss/all_minilm_l6_v2_{m32,sq8}/ and result JSON to
benchmarks/results/{ts}_pq_ceiling_{variant}.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evaluation.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from evaluation.trec_eval import (
    TREC_DL_2019,
    TREC_DL_2020,
    combined_queries,
    load_qrels,
)
from retrieval.dense.encoder import SentenceEncoder

EMB_DIR = REPO_ROOT / "data" / "embeddings" / "all_minilm_l6_v2"
TRAIN_SAMPLE = 100_000
NLIST = 4096


def build_ivf_pq_m32(embeddings: np.ndarray, pids: list[str], out_dir: Path,
                     nprobe: int = 16) -> tuple:
    """IVF-PQ with m=32 sub-quantizers (12-dim sub-vectors). 32 bytes/vec."""
    import faiss

    n, dim = embeddings.shape
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, NLIST, 32, 8)
    index.nprobe = nprobe

    rng = np.random.default_rng(42)
    sample_idxs = rng.choice(n, size=min(TRAIN_SAMPLE, n), replace=False)
    sample = np.ascontiguousarray(embeddings[sorted(sample_idxs)], dtype=np.float32)

    print(f"  Training IVF-PQ m=32 on {len(sample):,} vectors ...")
    t0 = time.perf_counter()
    index.train(sample)
    print(f"  Train: {time.perf_counter() - t0:.1f}s")

    print(f"  Adding {n:,} vectors ...")
    t0 = time.perf_counter()
    chunk = 200_000
    for start in range(0, n, chunk):
        index.add(np.ascontiguousarray(embeddings[start:start+chunk], dtype=np.float32))
    add_time = time.perf_counter() - t0
    print(f"  Add: {add_time:.1f}s")

    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "index.faiss"))
    (out_dir / "pids.json").write_text(json.dumps(pids))
    meta = {"index_type": "IVFPQ", "nlist": NLIST, "m": 32, "nbits": 8,
            "nprobe": nprobe, "ntotal": index.ntotal}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return index, meta


def build_ivf_sq8(embeddings: np.ndarray, pids: list[str], out_dir: Path,
                  nprobe: int = 16) -> tuple:
    """IVF-SQ8: scalar quantisation, 1 byte per dim. 384 bytes/vec."""
    import faiss

    n, dim = embeddings.shape
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFScalarQuantizer(
        quantizer, dim, NLIST,
        faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_L2,
    )
    index.nprobe = nprobe

    rng = np.random.default_rng(42)
    sample_idxs = rng.choice(n, size=min(TRAIN_SAMPLE, n), replace=False)
    sample = np.ascontiguousarray(embeddings[sorted(sample_idxs)], dtype=np.float32)

    print(f"  Training IVF-SQ8 on {len(sample):,} vectors ...")
    t0 = time.perf_counter()
    index.train(sample)
    print(f"  Train: {time.perf_counter() - t0:.1f}s")

    print(f"  Adding {n:,} vectors ...")
    t0 = time.perf_counter()
    chunk = 200_000
    for start in range(0, n, chunk):
        index.add(np.ascontiguousarray(embeddings[start:start+chunk], dtype=np.float32))
    add_time = time.perf_counter() - t0
    print(f"  Add: {add_time:.1f}s")

    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "index.faiss"))
    (out_dir / "pids.json").write_text(json.dumps(pids))
    meta = {"index_type": "IVFScalarQuantizer", "nlist": NLIST,
            "nprobe": nprobe, "ntotal": index.ntotal, "qtype": "QT_8bit"}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return index, meta


def eval_index(index, pids: list[str], queries: dict, qrels_2019: dict, qrels_2020: dict,
               nprobe: int = 16, top_k: int = 100):
    """Encode queries, retrieve top-K, compute metrics."""
    print(f"  Loading encoder + encoding {len(queries)} queries ...")
    enc = SentenceEncoder("all-MiniLM-L6-v2")
    qids = list(queries.keys())
    texts = [queries[qid] for qid in qids]
    qvecs = enc.encode_batch(texts, is_query=True, show_progress=False)

    index.nprobe = nprobe

    run = {}
    latencies = []
    for i, qid in enumerate(qids):
        q = np.ascontiguousarray(qvecs[i:i+1], dtype=np.float32)
        t0 = time.perf_counter()
        dists, idxs = index.search(q, top_k)
        latencies.append((time.perf_counter() - t0) * 1000)
        run[qid] = [pids[j] if 0 <= j < len(pids) else "" for j in idxs[0]]

    run_2019 = {qid: r for qid, r in run.items() if qid in qrels_2019}
    run_2020 = {qid: r for qid, r in run.items() if qid in qrels_2020}

    def _m(qr, r):
        return {
            "nDCG@10": ndcg_at_k(qr, r, k=10),
            "MRR@10": mrr_at_k(qr, r, k=10),
            "Recall@100": recall_at_k(qr, r, k=100),
        }

    return {
        "DL2019": _m(qrels_2019, run_2019),
        "DL2020": _m(qrels_2020, run_2020),
        "latency_p50": round(statistics.median(latencies), 3),
        "latency_p99": round(sorted(latencies)[min(int(len(latencies)*0.99), len(latencies)-1)], 3),
    }


def write_results(payload: dict, variant: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = REPO_ROOT / "benchmarks" / "results" / f"{ts}_pq_ceiling_{variant}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="PQ ceiling experiment")
    parser.add_argument("--variant", choices=["m32", "sq8", "both"], default="both")
    parser.add_argument("--nprobe", type=int, default=16)
    args = parser.parse_args()

    if not EMB_DIR.exists():
        print(f"ERROR: {EMB_DIR} not found")
        sys.exit(1)

    print("Loading embeddings memmap ...")
    embeddings, pids, manifest = SentenceEncoder.load_embeddings(EMB_DIR)
    n, dim = embeddings.shape
    print(f"  {n:,} × {dim}-dim, {embeddings.nbytes/1e9:.2f} GB on disk")

    queries = combined_queries()
    qrels_2019 = load_qrels(TREC_DL_2019)
    qrels_2020 = load_qrels(TREC_DL_2020)

    variants = ["m32", "sq8"] if args.variant == "both" else [args.variant]

    summary = {}
    for variant in variants:
        print(f"\n=== {variant.upper()} ===")
        out_dir = REPO_ROOT / "data" / "faiss" / f"all_minilm_l6_v2_{variant}"

        if (out_dir / "index.faiss").exists():
            print(f"  Index exists at {out_dir}, loading ...")
            import faiss
            index = faiss.read_index(str(out_dir / "index.faiss"))
            meta = json.loads((out_dir / "meta.json").read_text())
            stored_pids = json.loads((out_dir / "pids.json").read_text())
        else:
            t_build = time.perf_counter()
            if variant == "m32":
                index, meta = build_ivf_pq_m32(embeddings, pids, out_dir, args.nprobe)
            else:
                index, meta = build_ivf_sq8(embeddings, pids, out_dir, args.nprobe)
            print(f"  Total build: {time.perf_counter() - t_build:.1f}s")
            stored_pids = pids

        size_mb = (out_dir / "index.faiss").stat().st_size / 1e6
        print(f"  Index size on disk: {size_mb:.1f} MB")

        metrics = eval_index(index, stored_pids, queries, qrels_2019, qrels_2020,
                             nprobe=args.nprobe)
        print(f"  DL2020 nDCG@10={metrics['DL2020']['nDCG@10']:.4f} "
              f"R@100={metrics['DL2020']['Recall@100']:.4f}")
        print(f"  DL2019 nDCG@10={metrics['DL2019']['nDCG@10']:.4f} "
              f"R@100={metrics['DL2019']['Recall@100']:.4f}")
        print(f"  Latency P50={metrics['latency_p50']}ms P99={metrics['latency_p99']}ms")

        payload = {
            "schema_version": "1.0",
            "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "variant": variant,
            "config": {"encoder": "all-MiniLM-L6-v2", **meta},
            "size_mb": round(size_mb, 1),
            "metrics": metrics,
        }
        out = write_results(payload, variant)
        print(f"  Written: {out.name}")
        summary[variant] = {"size_mb": size_mb, **metrics}

    # Comparison vs m=16 baseline
    print("\n" + "=" * 70)
    print("COMPARISON — DL2020")
    print("=" * 70)
    print(f"{'Variant':<10} {'Size MB':>8} {'nDCG@10':>9} {'R@100':>8} {'P50 ms':>8} {'P99 ms':>8}")
    print(f"{'m=16 base':<10} {209.0:>8.1f} {0.4547:>9.4f} {0.3388:>8.4f} {0.5:>8.1f} {0.7:>8.1f}")
    for v, s in summary.items():
        print(f"{v:<10} {s['size_mb']:>8.1f} {s['DL2020']['nDCG@10']:>9.4f} "
              f"{s['DL2020']['Recall@100']:>8.4f} {s['latency_p50']:>8.2f} {s['latency_p99']:>8.2f}")


if __name__ == "__main__":
    main()
