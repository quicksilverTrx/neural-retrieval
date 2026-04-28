"""FAISS index corruption recovery.

Recovery protocol
-----------------
1. On startup, validate the FAISS index checksum.
2. If validation fails → log warning, fall back to BM25-only mode.
3. Serve BM25-only until the index is rebuilt or repaired.
4. Recovery: rebuild the FAISS index from the raw embeddings .npy file
   (which is a separate artifact, validated independently).

This module provides:
    - validate_faiss_index(path): bool — checksum check
    - rebuild_index(emb_dir, faiss_dir, **kwargs): rebuild from raw embeddings
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


_CHECKSUM_KEY = "sha256"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def record_index_checksum(faiss_dir: Path) -> str:
    """Compute and store the sha256 of index.faiss in meta.json.

    Called after building a new index. Returns the checksum string.
    """
    faiss_dir = Path(faiss_dir)
    index_path = faiss_dir / "index.faiss"
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")

    checksum = _sha256_file(index_path)
    meta_path = faiss_dir / "meta.json"

    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        meta = {}
    meta[_CHECKSUM_KEY] = checksum
    meta_path.write_text(json.dumps(meta, indent=2))
    return checksum


def validate_faiss_index(faiss_dir: Path) -> bool:
    """Return True if the index.faiss checksum matches the recorded value.

    Returns False (without raising) on:
        - Missing meta.json (no checksum recorded → treat as corrupted)
        - Missing index.faiss
        - Checksum mismatch
    """
    faiss_dir = Path(faiss_dir)
    meta_path = faiss_dir / "meta.json"
    index_path = faiss_dir / "index.faiss"

    if not index_path.exists():
        return False
    if not meta_path.exists():
        return False

    meta = json.loads(meta_path.read_text())
    expected = meta.get(_CHECKSUM_KEY)
    if expected is None:
        return False

    actual = _sha256_file(index_path)
    return actual == expected


def rebuild_index(
    emb_dir: Path,
    faiss_dir: Path,
    nlist: int = 4096,
    m: int = 16,
    nbits: int = 8,
    nprobe: int = 16,
    train_sample_size: int = 100_000,
) -> None:
    """Rebuild a FAISS IVF-PQ index from raw embedding files.

    Recovery path: called when validate_faiss_index() returns False.
    Rebuilds in-place, overwriting the corrupted index.

    Args:
        emb_dir: Directory containing embeddings.npy, pids.json, manifest.json
        faiss_dir: Destination for rebuilt index (overwritten if exists)
        nlist, m, nbits, nprobe: FAISS parameters
        train_sample_size: Number of vectors to use for IVF training
    """
    import numpy as np

    from retrieval.dense.encoder import SentenceEncoder
    from retrieval.dense.faiss_index import FAISSIVFPQIndex

    emb_dir = Path(emb_dir)
    faiss_dir = Path(faiss_dir)

    print(f"Rebuilding FAISS index from {emb_dir} → {faiss_dir} ...")
    embeddings, pids, manifest = SentenceEncoder.load_embeddings(emb_dir)
    n, dim = embeddings.shape
    print(f"  {n:,} passages × {dim}-dim")

    idx = FAISSIVFPQIndex(embedding_dim=dim, nlist=nlist, m=m, nbits=nbits, nprobe=nprobe)

    rng = np.random.default_rng(42)
    sample_idxs = rng.choice(n, size=min(train_sample_size, n), replace=False)
    sample = np.ascontiguousarray(embeddings[sorted(sample_idxs)], dtype=np.float32)
    print(f"  Training on {len(sample):,} vectors ...")
    idx.train(sample)

    idx.add(embeddings, pids)
    idx.save(faiss_dir)
    record_index_checksum(faiss_dir)
    print(f"  Rebuild complete: {n:,} vectors indexed, checksum recorded.")
