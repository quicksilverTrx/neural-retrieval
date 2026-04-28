"""Sentence encoder: batch-encode passages and queries with HF sentence-transformers.

Handles both MiniLM and E5 model families.

Supports:
    all-MiniLM-L6-v2       (22M params, 384-dim, no prefix needed)
    intfloat/e5-small-v2   (33M params, 384-dim, needs 'query:'/'passage:' prefix)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

# E5 models need input prefixes; detect by model name substring
_E5_SLUG = "e5"


class SentenceEncoder:
    """Thin wrapper around sentence-transformers for offline and online encoding.

    Offline (corpus): encode_corpus() streams 8.8M passages to a memory-mapped
    .npy file, avoiding OOM on large corpora.

    Online (query):  encode_query() returns a single normalized float32 vector.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        # get_embedding_dimension() is the current API; fall back for older versions
        _dim_fn = getattr(self.model, "get_embedding_dimension",
                          self.model.get_sentence_embedding_dimension)
        self.embedding_dim: int = _dim_fn()
        self._is_e5 = _E5_SLUG in model_name.lower()

    # ------------------------------------------------------------------
    # Prefix helpers for E5 models
    # ------------------------------------------------------------------

    def _prefix_query(self, text: str) -> str:
        return f"query: {text}" if self._is_e5 else text

    def _prefix_passage(self, text: str) -> str:
        return f"passage: {text}" if self._is_e5 else text

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_query(self, text: str) -> np.ndarray:
        """Encode a single query string. Returns normalized float32 (dim,)."""
        vec = self.model.encode(
            self._prefix_query(text),
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec.astype(np.float32)

    def encode_batch(
        self,
        texts: list[str],
        batch_size: int = 256,
        is_query: bool = False,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode a list of texts. Returns float32 (n, dim).

        Args:
            texts:         list of strings to encode
            batch_size:    inference batch size
            is_query:      True → apply query prefix (E5 only)
            show_progress: show tqdm bar
        """
        prefix_fn = self._prefix_query if is_query else self._prefix_passage
        prefixed = [prefix_fn(t) for t in texts]
        vecs = self.model.encode(
            prefixed,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return vecs.astype(np.float32)

    def encode_corpus(
        self,
        passages,
        output_dir: Path,
        batch_size: int = 512,
        num_passages: int | None = None,
        chunk_multiplier: int = 100,
        progress_every: int | None = None,
    ) -> Path:
        """Batch-encode corpus and stream to disk as a .npy file + pids.json.

        ``passages`` may be any iterable that yields ``(pid, text)`` pairs.

        Two layers of batching:
            * Chunk level — ``batch_size * chunk_multiplier`` rows buffered
              in Python RAM before each ``model.encode()`` call.
            * Device level — ``model.encode(batch_size=…)`` internally splits
              the chunk into ``batch_size``-row forward passes on GPU/MPS.

        Memory stays bounded: model weights (~90–130 MB) + one chunk buffer
        (~15 MB) + incremental ``pids`` list (~140 MB at 8.8M docs) +
        the on-disk file (streamed, not memmap).

        Writes:
            {output_dir}/embeddings.npy  — float32 (n, dim) on disk
            {output_dir}/pids.json       — list of passage IDs (same order)
            {output_dir}/manifest.json   — model name, dim, count, timing

        Args:
            passages:         Iterable of (pid, text) pairs.
            output_dir:       Directory for output files (created if missing).
            batch_size:       Device-level inference batch size.
            num_passages:     Total passage count. Required if ``passages`` has
                              no ``__len__``. Use ``len(ds)`` for HF datasets.
            chunk_multiplier: Rows per encode() call = batch_size × this.
            progress_every:   Print a progress line every N rows.

        Returns:
            output_dir Path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Resolve total count for the .npy header
        if num_passages is not None:
            n = num_passages
        elif hasattr(passages, "__len__"):
            n = len(passages)
        else:
            raise ValueError(
                "num_passages is required when passages is an iterable without __len__. "
                "Pass len(ds) for HuggingFace datasets."
            )

        emb_path = output_dir / "embeddings.npy"
        pids_path = output_dir / "pids.json"

        # Streaming write (not np.memmap): a 13.5 GB memmap on a 16 GB machine
        # faults the target page resident on every write; the kernel only evicts
        # when forced. Plain file.write() lets the kernel move bytes to the
        # page cache (then disk) without affecting the process's resident
        # memory. The on-disk .npy file is byte-identical and downstream
        # consumers (load_embeddings, FAISS build) can still mmap it read-only.
        emb_file = emb_path.open("wb")
        np.lib.format.write_array_header_1_0(
            emb_file,
            {
                "descr": np.lib.format.dtype_to_descr(np.dtype(np.float32)),
                "fortran_order": False,
                "shape": (n, self.embedding_dim),
            },
        )
        data_start = emb_file.tell()

        chunk_size = batch_size * chunk_multiplier
        if progress_every is None:
            progress_every = max(chunk_size, 100_000)

        device_str = str(getattr(self.model, "device", "?"))

        print(
            f"Encoding {n:,} passages with {self.model_name} "
            f"(device={device_str}, batch_size={batch_size}, "
            f"chunk={chunk_size:,}, streaming .npy write) ...",
            flush=True,
        )
        t0 = time.perf_counter()
        last_progress_at = 0

        pids: list[str] = []
        chunk_texts: list[str] = []
        write_ptr = 0

        # MPS cache clearer (optional). Without this, the MPS allocator caches
        # intermediate tensors across forward passes and the process footprint
        # grows monotonically (observed: 14 GB on a 16 GB machine after ~3 min).
        _empty_mps_cache = None
        if device_str.startswith("mps"):
            try:
                import torch
                _empty_mps_cache = torch.mps.empty_cache
            except (ImportError, AttributeError):
                pass

        def _flush_chunk() -> None:
            nonlocal write_ptr
            if not chunk_texts:
                return
            vecs = self.model.encode(
                chunk_texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            vecs_f32 = vecs.astype(np.float32, copy=False)
            if not vecs_f32.flags["C_CONTIGUOUS"]:
                vecs_f32 = np.ascontiguousarray(vecs_f32)
            emb_file.write(vecs_f32.tobytes())
            write_ptr += len(chunk_texts)
            chunk_texts.clear()
            if _empty_mps_cache is not None:
                _empty_mps_cache()

        def _fmt_duration(seconds: float) -> str:
            seconds = int(seconds)
            h, rem = divmod(seconds, 3600)
            m, s = divmod(rem, 60)
            if h:
                return f"{h}h{m:02d}m"
            if m:
                return f"{m}m{s:02d}s"
            return f"{s}s"

        def _print_progress() -> None:
            elapsed = time.perf_counter() - t0
            rate = write_ptr / elapsed if elapsed > 0 else 0.0
            remaining = n - write_ptr
            eta = remaining / rate if rate > 0 else 0.0
            pct = write_ptr / n * 100 if n else 0.0
            print(
                f"  {write_ptr:>9,}/{n:,} ({pct:4.1f}%) | "
                f"elapsed {_fmt_duration(elapsed)} | "
                f"{rate:>6,.0f} p/s | "
                f"ETA {_fmt_duration(eta)}",
                flush=True,
            )

        for pid, text in passages:
            pids.append(pid)
            chunk_texts.append(self._prefix_passage(text))

            if len(chunk_texts) >= chunk_size:
                _flush_chunk()
                if write_ptr - last_progress_at >= progress_every:
                    _print_progress()
                    last_progress_at = write_ptr

        _flush_chunk()   # remainder
        emb_file.flush()
        emb_file.close()

        # Verify we wrote exactly the number of rows we promised in the header.
        expected_bytes = data_start + n * self.embedding_dim * 4
        actual_bytes = emb_path.stat().st_size
        if actual_bytes != expected_bytes:
            raise RuntimeError(
                f"Embeddings file size mismatch: header declares "
                f"{n:,} × {self.embedding_dim} float32, expected {expected_bytes} "
                f"bytes on disk; got {actual_bytes} bytes. Iterator likely yielded "
                f"{write_ptr:,} rows instead of {n:,}."
            )

        elapsed = time.perf_counter() - t0
        pids_path.write_text(json.dumps(pids))

        manifest = {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "num_passages": write_ptr,
            "dtype": "float32",
            "encoding_time_s": round(elapsed, 1),
            "passages_per_second": round(write_ptr / elapsed) if elapsed > 0 else 0,
            "device": device_str,
            "batch_size": batch_size,
            "chunk_size": chunk_size,
        }
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        size_gb = write_ptr * self.embedding_dim * 4 / 1e9
        print(
            f"Encoding complete: {write_ptr:,} passages | "
            f"{_fmt_duration(elapsed)} | "
            f"{write_ptr / elapsed:,.0f} p/s | "
            f"{size_gb:.2f} GB → {emb_path}"
        )
        return output_dir

    @staticmethod
    def load_embeddings(output_dir: Path) -> tuple[np.ndarray, list[str], dict]:
        """Load saved embeddings as read-only memmap.

        Returns:
            (embeddings, pids, manifest)
            embeddings: float32 (n, dim) memory-mapped array
            pids:       list of passage IDs
            manifest:   dict with model_name, embedding_dim, etc.
        """
        output_dir = Path(output_dir)
        manifest = json.loads((output_dir / "manifest.json").read_text())
        pids = json.loads((output_dir / "pids.json").read_text())
        embeddings = np.lib.format.open_memmap(
            str(output_dir / "embeddings.npy"),
            mode="r",
            dtype=np.float32,
            shape=(manifest["num_passages"], manifest["embedding_dim"]),
        )
        return embeddings, pids, manifest
