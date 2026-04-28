"""FAISS IVF-PQ index for 8.8M-scale dense retrieval.

Index parameters:
    nlist  = 4096   √8.8M ≈ 2966;
    m      = 32     sub-quantizers; 384-dim / 32 = 12 dims per sub-quantizer
    nbits  = 8      256 centroids/sub-quantizer → 32 bytes/vector
    nprobe = 16     default probe count; sweep 1–64 to find operating point

Memory at 8.8M scale (m=32):
    8,841,823 vectors × 32 bytes = ~283 MB (PQ codes)
    IVF centroids: 4096 × 384 × 4 bytes = ~6 MB
    PQ codebooks: 32 × 256 × 12 × 4 bytes = ~393 KB
    Total: ~360 MB measured on disk.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

DEFAULT_NLIST = 4096
DEFAULT_M = 32
DEFAULT_NBITS = 8
DEFAULT_NPROBE = 16

NPROBE_SWEEP_VALUES = [1, 4, 8, 16, 32, 64]

# FAISS train heuristic: 39 × nlist minimum; 100K is ~24× nlist (a soft warning).
TRAIN_SAMPLE_SIZE = 100_000


class FAISSIVFPQIndex:
    """FAISS IVFFlat + PQ index with save/load and nprobe sweep support.

    Build flow:
        idx = FAISSIVFPQIndex()
        idx.train(sample_embeddings)         # 100K random sample
        idx.add(all_embeddings, pids)        # 8.8M in chunks
        idx.save(path)

    Serve flow:
        idx = FAISSIVFPQIndex.load(path)
        pids_per_q, dists = idx.search(query_vectors, top_k=100)

    nprobe sweep:
        results = idx.nprobe_sweep(query_vecs, qrels, qids, top_k=100)
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        nlist: int = DEFAULT_NLIST,
        m: int = DEFAULT_M,
        nbits: int = DEFAULT_NBITS,
        nprobe: int = DEFAULT_NPROBE,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.nlist = nlist
        self.m = m
        self.nbits = nbits
        self.nprobe = nprobe
        self._index = None              # faiss.IndexIVFPQ, built after train()
        self._pids: list[str] = []

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _make_index(self):
        import faiss

        quantizer = faiss.IndexFlatL2(self.embedding_dim)
        return faiss.IndexIVFPQ(
            quantizer, self.embedding_dim, self.nlist, self.m, self.nbits
        )

    def train(self, sample_embeddings: np.ndarray) -> None:
        """Train IVF coarse quantizer + PQ product quantizer on a random sample.

        Args:
            sample_embeddings: float32 (n_sample, dim).
                Must be a *random* sample — not the first N passages.
                Recommended size: TRAIN_SAMPLE_SIZE (100K).
                Minimum guideline: 39 × nlist for cluster quality.
        """
        if len(sample_embeddings) < 39 * self.nlist:
            import warnings
            warnings.warn(
                f"Training sample ({len(sample_embeddings)}) < 39×nlist "
                f"({39 * self.nlist}). Cluster quality may be poor.",
                stacklevel=2,
            )

        self._index = self._make_index()
        print(
            f"Training FAISS IVF-PQ "
            f"(nlist={self.nlist}, m={self.m}, nbits={self.nbits}) "
            f"on {len(sample_embeddings):,} vectors ..."
        )
        t0 = time.perf_counter()
        self._index.train(sample_embeddings.astype(np.float32))
        print(f"  Training complete: {time.perf_counter() - t0:.1f}s")

    def add(
        self,
        embeddings: np.ndarray,
        pids: list[str],
        chunk_size: int = 100_000,
    ) -> None:
        """Add embeddings to the trained index in memory-safe chunks."""
        if self._index is None:
            raise RuntimeError("Call train() before add()")

        n = len(embeddings)
        print(f"Adding {n:,} vectors to FAISS index in chunks of {chunk_size:,} ...")
        t0 = time.perf_counter()

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            self._index.add(np.ascontiguousarray(embeddings[start:end], dtype=np.float32))
            if (end // 1_000_000) > (start // 1_000_000):
                print(f"  {end:,}/{n:,} added ({time.perf_counter() - t0:.0f}s)")

        self._pids = list(pids)
        print(
            f"  Done: {self._index.ntotal:,} vectors indexed "
            f"in {time.perf_counter() - t0:.1f}s"
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_vectors: np.ndarray,
        top_k: int = 100,
        nprobe: int | None = None,
    ) -> tuple[list[list[str]], np.ndarray]:
        """Approximate nearest-neighbour search.

        Args:
            query_vectors: float32 (n_queries, dim) or (dim,) for one query
            top_k:         neighbours to return per query
            nprobe:        IVF cells to visit (overrides instance default)

        Returns:
            (pids_per_query, distances)
            pids_per_query[i]: list of top_k passage IDs for query i
            distances:         float32 (n_queries, top_k)
        """
        if self._index is None:
            raise RuntimeError("Index not built — call train() + add() or load() first")

        self._index.nprobe = nprobe if nprobe is not None else self.nprobe

        q = query_vectors
        if q.ndim == 1:
            q = q[np.newaxis, :]
        q = np.ascontiguousarray(q, dtype=np.float32)

        distances, indices = self._index.search(q, top_k)

        results = []
        for row in indices:
            results.append(
                [self._pids[i] if 0 <= i < len(self._pids) else "" for i in row]
            )
        return results, distances

    # ------------------------------------------------------------------
    # nprobe sweep
    # ------------------------------------------------------------------

    def nprobe_sweep(
        self,
        query_vectors: np.ndarray,
        qrels: dict[str, dict[str, int]],
        qids: list[str],
        top_k: int = 100,
        nprobe_values: list[int] | None = None,
    ) -> list[dict]:
        """Measure Recall@top_k and latency at each nprobe value.

        Returns:
            list of dicts: [{"nprobe": int, "Recall@100": float,
                             "latency_ms_mean": float, "latency_ms_p99": float}]
        """
        from evaluation.metrics import recall_at_k

        nprobe_values = nprobe_values or NPROBE_SWEEP_VALUES
        print(f"nprobe sweep: {nprobe_values} — measuring Recall@{top_k} and latency")

        sweep_results = []
        for np_ in nprobe_values:
            latencies_ms = []
            run: dict[str, list[str]] = {}

            for i, qid in enumerate(qids):
                q = query_vectors[i : i + 1]
                t0 = time.perf_counter()
                pids_per_q, _ = self.search(q, top_k=top_k, nprobe=np_)
                latencies_ms.append((time.perf_counter() - t0) * 1000)
                run[qid] = pids_per_q[0]

            recall = recall_at_k(qrels, run, k=top_k)
            latencies_ms.sort()
            n = len(latencies_ms)
            entry = {
                "nprobe": np_,
                f"Recall@{top_k}": round(recall, 4),
                "latency_ms_mean": round(sum(latencies_ms) / n, 2),
                "latency_ms_p50": round(latencies_ms[n // 2], 2),
                "latency_ms_p99": round(latencies_ms[min(int(n * 0.99), n - 1)], 2),
            }
            sweep_results.append(entry)
            print(
                f"  nprobe={np_:3d}: Recall@{top_k}={recall:.3f}, "
                f"p50={entry['latency_ms_p50']:.1f}ms, "
                f"p99={entry['latency_ms_p99']:.1f}ms"
            )

        return sweep_results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Save FAISS index + pid list to directory."""
        import faiss

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(path / "index.faiss"))
        (path / "pids.json").write_text(json.dumps(self._pids))
        meta = {
            "embedding_dim": self.embedding_dim,
            "nlist": self.nlist,
            "m": self.m,
            "nbits": self.nbits,
            "nprobe": self.nprobe,
            "ntotal": self._index.ntotal,
        }
        (path / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"FAISS index saved to {path} ({self._index.ntotal:,} vectors)")

    @classmethod
    def load(cls, path: Path) -> "FAISSIVFPQIndex":
        """Load saved index from directory."""
        import faiss

        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        inst = cls(
            embedding_dim=meta["embedding_dim"],
            nlist=meta["nlist"],
            m=meta["m"],
            nbits=meta["nbits"],
            nprobe=meta["nprobe"],
        )
        inst._index = faiss.read_index(str(path / "index.faiss"))
        inst._index.nprobe = meta["nprobe"]
        inst._pids = json.loads((path / "pids.json").read_text())
        return inst
