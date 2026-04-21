"""
Prepare MS MARCO v1 passages for indexing.

Loads from HuggingFace cache (Tevatron/msmarco-passage-corpus),
streams passages, and writes them to data/msmarco_passages.jsonl
for use by the BM25 index builder and dense encoder.

Usage:
    python data/prepare_msmarco.py [--limit N]

The output file format (one JSON object per line):
    {"pid": "0", "text": "passage text..."}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_PATH = DATA_DIR / "msmarco_passages.jsonl"


def prepare(limit: int | None = None) -> None:
    from datasets import load_dataset

    print("Loading MS MARCO corpus from HuggingFace cache...")
    ds = load_dataset(
        "Tevatron/msmarco-passage-corpus",
        cache_dir=str(DATA_DIR / "corpus"),
        split="train",
    )

    total = len(ds) if limit is None else min(limit, len(ds))
    print(f"Writing {total:,} passages to {OUTPUT_PATH}")

    written = 0
    with OUTPUT_PATH.open("w", encoding="utf-8") as fout:
        for i, row in enumerate(ds):
            if limit is not None and i >= limit:
                break
            record = {"pid": str(row["docid"]), "text": row["text"]}
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            if written % 500_000 == 0:
                print(f"  {written:,} written...")

    print(f"Done. {written:,} passages written to {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit passages (for testing)")
    args = parser.parse_args()
    prepare(limit=args.limit)
