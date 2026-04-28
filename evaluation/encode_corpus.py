"""Batch-encode 8.8M passages with a sentence encoder.

Memory design
-------------
Streams the HF Arrow dataset row-by-row instead of materialising the corpus
in RAM. Peak in-process memory:
    - Model weights: ~90 MB (MiniLM) / ~130 MB (E5-small)
    - One chunk buffer: batch_size × N texts × ~300 chars ≈ 7 MB
    - pids list: 8.8M × ~8 chars × overhead ≈ ~140 MB
    - Embeddings file (streamed write): 13.5 GB on disk, NOT RAM
Total: ~400-500 MB.

Run once per model:
    python evaluation/encode_corpus.py --model all-MiniLM-L6-v2 --device mps
    python evaluation/encode_corpus.py --model intfloat/e5-small-v2 --device mps

Output: data/embeddings/<model_slug>/
    embeddings.npy   float32 (n, dim)
    pids.json        ordered list of passage IDs
    manifest.json    model, dim, count, encoding time
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from retrieval.dense.encoder import SentenceEncoder


def _slug(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_").lower()


def _load_hf_dataset(limit: int | None = None):
    from datasets import load_dataset

    ds = load_dataset(
        "Tevatron/msmarco-passage-corpus",
        cache_dir=str(REPO_ROOT / "data" / "corpus"),
        split="train",
    )
    if limit is not None:
        ds = ds.select(range(limit))
    return ds


def _load_jsonl(limit: int | None = None):
    """Load from local JSONL if present (generator — no list)."""
    jsonl_path = REPO_ROOT / "data" / "msmarco_passages.jsonl"
    if not jsonl_path.exists():
        return None, None

    print(f"Counting passages in {jsonl_path} ...")
    n = 0
    with jsonl_path.open() as f:
        for line in f:
            n += 1
            if limit is not None and n >= limit:
                break
    if limit is not None:
        n = min(n, limit)

    def _gen():
        with jsonl_path.open() as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    return
                obj = json.loads(line)
                # JSONL stores int pids; eval expects str
                yield str(obj["pid"]), obj["text"]

    return _gen(), n


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode MS MARCO passage corpus")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="HF model name")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Per-device inference batch size.")
    parser.add_argument("--chunk-multiplier", type=int, default=8,
                        help="Rows per model.encode() call = batch_size × this. "
                             "Small values bound MPS accumulated state.")
    parser.add_argument("--limit", type=int, default=None, help="Cap corpus size (smoke test)")
    parser.add_argument("--device", default=None, help="cuda / cpu / mps (auto-detect if omitted)")
    args = parser.parse_args()

    output_dir = REPO_ROOT / "data" / "embeddings" / _slug(args.model)

    if output_dir.exists() and (output_dir / "manifest.json").exists():
        manifest = json.loads((output_dir / "manifest.json").read_text())
        print(
            f"Embeddings already exist at {output_dir} "
            f"({manifest['num_passages']:,} passages). "
            "Delete the directory to re-encode."
        )
        return

    print(f"Loading encoder: {args.model} (device={args.device or 'auto'})")
    enc = SentenceEncoder(model_name=args.model, device=args.device)

    jsonl_iter, jsonl_n = _load_jsonl(limit=args.limit)

    if jsonl_iter is not None:
        print(f"Streaming from local JSONL ({jsonl_n:,} passages — no list in RAM)")
        enc.encode_corpus(jsonl_iter, output_dir,
                          batch_size=args.batch_size, num_passages=jsonl_n,
                          chunk_multiplier=args.chunk_multiplier)
    else:
        print("Local JSONL not found — streaming from HuggingFace Arrow cache ...")
        ds = _load_hf_dataset(limit=args.limit)
        n = len(ds)
        print(f"  Dataset: {n:,} passages (Arrow memory-mapped, not loaded to RAM)")

        passage_iter = ((str(row["docid"]), row["text"]) for row in ds)
        enc.encode_corpus(passage_iter, output_dir,
                          batch_size=args.batch_size, num_passages=n,
                          chunk_multiplier=args.chunk_multiplier)

    print(f"Done. Embeddings at {output_dir}")


if __name__ == "__main__":
    main()
