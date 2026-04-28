"""Train + build FAISS IVF-PQ index from pre-encoded embeddings.

Prerequisites: run encode_corpus.py first.

Usage:
    python evaluation/build_faiss.py --model all-MiniLM-L6-v2
    python evaluation/build_faiss.py --model intfloat/e5-small-v2

Output: data/faiss/<model_slug>/
    index.faiss   FAISS IVF-PQ index
    pids.json     passage ID list (FAISS row → pid)
    meta.json     nlist, m, nbits, nprobe, ntotal
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from retrieval.dense.encoder import SentenceEncoder
from retrieval.dense.faiss_index import FAISSIVFPQIndex, TRAIN_SAMPLE_SIZE


def _slug(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_").lower()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS IVF-PQ index from embeddings")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--nlist", type=int, default=4096,
                        help="IVF cell count")
    parser.add_argument("--m", type=int, default=32,
                        help="PQ sub-quantizers (32 → 12-dim sub-vectors, 32 bytes/vec)")
    parser.add_argument("--nbits", type=int, default=8,
                        help="Bits per sub-quantizer")
    parser.add_argument("--nprobe", type=int, default=16,
                        help="Default nprobe (sweep in dense_eval to choose operating point)")
    parser.add_argument("--train-sample", type=int, default=TRAIN_SAMPLE_SIZE)
    args = parser.parse_args()

    slug = _slug(args.model)
    emb_dir = REPO_ROOT / "data" / "embeddings" / slug
    faiss_dir = REPO_ROOT / "data" / "faiss" / slug

    if not emb_dir.exists():
        raise FileNotFoundError(
            f"Embeddings not found at {emb_dir}. "
            "Run encode_corpus.py first."
        )

    if faiss_dir.exists() and (faiss_dir / "index.faiss").exists():
        print(f"FAISS index already exists at {faiss_dir}. Delete to rebuild.")
        return

    print(f"Loading embeddings from {emb_dir} ...")
    embeddings, pids, manifest = SentenceEncoder.load_embeddings(emb_dir)
    n, dim = embeddings.shape
    print(f"  {n:,} passages × {dim}-dim  ({embeddings.nbytes / 1e9:.2f} GB memmap)")

    idx = FAISSIVFPQIndex(
        embedding_dim=dim,
        nlist=args.nlist,
        m=args.m,
        nbits=args.nbits,
        nprobe=args.nprobe,
    )

    # Random sample for training (must NOT be the first N rows)
    rng = np.random.default_rng(42)
    sample_idxs = rng.choice(n, size=min(args.train_sample, n), replace=False)
    sample = np.ascontiguousarray(embeddings[sorted(sample_idxs)], dtype=np.float32)
    print(f"Training on {len(sample):,} random passages ...")
    idx.train(sample)

    idx.add(embeddings, pids)
    idx.save(faiss_dir)
    print(f"FAISS index built and saved to {faiss_dir}")


if __name__ == "__main__":
    main()
