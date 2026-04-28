"""Build IVF-Flat (no PQ) FAISS index — falsification experiment.

If the IVF-PQ Recall@100 plateau at ~0.34 is caused by PQ quantisation
loss, then rebuilding with IndexIVFFlat (no PQ — full float32 vectors per
cluster) should produce a much higher recall.

Same IVF clustering (nlist=4096), same train sample (100K), same nprobe
sweep done by dense_eval.py — only the per-vector storage changes.

Output: data/faiss/<model_slug>_flat/
    index.faiss   ~13.5 GB on disk, full float32 per vector
    pids.json     same format as the PQ build
    meta.json     notes the index type: IVFFlat

Usage:
    python evaluation/build_faiss_flat.py --model all-MiniLM-L6-v2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from retrieval.dense.encoder import SentenceEncoder

TRAIN_SAMPLE_SIZE = 100_000


def _slug(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_").lower()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build IVF-Flat (no PQ) FAISS index")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--nlist", type=int, default=4096)
    parser.add_argument("--nprobe", type=int, default=16)
    parser.add_argument("--train-sample", type=int, default=TRAIN_SAMPLE_SIZE)
    args = parser.parse_args()

    import faiss

    slug = _slug(args.model)
    emb_dir = REPO_ROOT / "data" / "embeddings" / slug
    faiss_dir = REPO_ROOT / "data" / "faiss" / f"{slug}_flat"

    if not emb_dir.exists():
        raise FileNotFoundError(f"Embeddings not found at {emb_dir}")
    if (faiss_dir / "index.faiss").exists():
        print(f"IVF-Flat index already exists at {faiss_dir}. Delete to rebuild.")
        return

    faiss_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading embeddings from {emb_dir} ...")
    embeddings, pids, manifest = SentenceEncoder.load_embeddings(emb_dir)
    n, dim = embeddings.shape
    print(f"  {n:,} passages × {dim}-dim ({embeddings.nbytes / 1e9:.2f} GB memmap)")

    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, args.nlist, faiss.METRIC_L2)
    index.nprobe = args.nprobe

    rng = np.random.default_rng(42)
    sample_idxs = rng.choice(n, size=min(args.train_sample, n), replace=False)
    sample = np.ascontiguousarray(embeddings[sorted(sample_idxs)], dtype=np.float32)
    print(f"Training on {len(sample):,} random passages (k-means → {args.nlist} clusters) ...")
    t0 = time.perf_counter()
    index.train(sample)
    print(f"  Training: {time.perf_counter() - t0:.1f} s")

    print(f"Adding {n:,} vectors to IVF-Flat index ...")
    t0 = time.perf_counter()
    chunk = 200_000
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        index.add(np.ascontiguousarray(embeddings[start:stop], dtype=np.float32))
        if (start // chunk) % 5 == 0:
            print(f"  {stop:,}/{n:,} added ({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"  Add: {time.perf_counter() - t0:.1f} s")

    print(f"Writing index to {faiss_dir / 'index.faiss'} ...")
    t0 = time.perf_counter()
    faiss.write_index(index, str(faiss_dir / "index.faiss"))
    print(f"  Write: {time.perf_counter() - t0:.1f} s")

    (faiss_dir / "pids.json").write_text(json.dumps(pids))

    meta = {
        "embedding_dim": dim,
        "index_type": "IVFFlat",
        "nlist": args.nlist,
        "nprobe": args.nprobe,
        "ntotal": index.ntotal,
        "train_sample": args.train_sample,
        "encoder_model": args.model,
        "purpose": "Falsification experiment: tests whether IVF-PQ Recall@100 ~0.34 plateau is PQ-quantisation-driven (this index has no PQ).",
    }
    (faiss_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Done. IVF-Flat index at {faiss_dir}")


if __name__ == "__main__":
    main()
