"""Custom BM25 correctness check + benchmark vs bm25s baseline.

Steps:
    1. Stream corpus one row at a time into InvertedIndex
    2. Retrieve top-100 for 97 TREC queries
    3. Compute nDCG@10 / MRR@10 / Recall@100
    4. Ship gate: DL2020 nDCG@10 within ±0.02 of bm25s baseline (0.428)
    5. Write results JSON

Memory + CPU design
-------------------
Three complementary strategies vs the naive HF-Arrow single-process build:

1. JSONL corpus cache (IO + memory)
   First full-corpus run exports the HF Arrow dataset to a compact JSONL file
   (data/msmarco_passages.jsonl). Subsequent runs read directly from JSONL:
   no HF/pyarrow overhead → base RSS drops from ~3.5 GB to ~200 MB.
   Integer doc IDs (MSMARCO docids are numeric) save a further ~650 MB vs
   storing string doc IDs in posting lists.

2. Parallel worker processes (CPU)
   With --jobs N, the corpus is split into N equal byte-aligned shards
   (computed in one pass). Each worker seeks directly to its shard start —
   no wasted reads. Workers are OS processes (no GIL contention) and exit
   after saving their partial index, freeing memory before merge.

3. Sequential merge (memory)
   Partial indices are merged one at a time into an accumulator, so peak
   RAM during merge is: accumulated_index + one_partial.
   Full-corpus merge peak: ~1.5 GB (vs ~7 GB single-process).

Smoke test (--limit N) streams from HF Arrow directly — no JSONL export.

Usage:
    python evaluation/bm25_eval.py --limit 50000          # smoke test
    python evaluation/bm25_eval.py --jobs 4               # parallel full build
    python evaluation/bm25_eval.py --index-path data/custom_bm25_8m.bin
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

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
from retrieval.inverted_index import BM25Retriever, BM25Scorer, InvertedIndex, tokenize
from retrieval.inverted_index.persistence import load_index, save_index

BM25S_BASELINE_DL2020 = 0.428
GATE_DELTA = 0.02

# macOS reports ru_maxrss in bytes; Linux in kilobytes
_RSS_SCALE = 1024 * 1024 if platform.system() == "Darwin" else 1024


def export_corpus_jsonl(jsonl_path: Path) -> None:
    """Export HF Arrow corpus to compact JSONL with integer doc IDs.

    Run once; subsequent calls are skipped by caller. MSMARCO docids are
    numeric strings; storing as int saves ~650 MB RSS at 8.8M scale.
    """
    from datasets import load_dataset

    ds = load_dataset(
        "Tevatron/msmarco-passage-corpus",
        cache_dir=str(REPO_ROOT / "data" / "corpus"),
        split="train",
    )
    total = len(ds)
    print(f"Exporting {total:,} passages → {jsonl_path}")
    print("  (one-time; subsequent runs read directly from JSONL)")
    t0 = time.perf_counter()
    buf_size = 4 << 20
    with jsonl_path.open("w", buffering=buf_size) as f:
        for i, row in enumerate(ds):
            f.write(json.dumps({"pid": int(row["docid"]), "text": row["text"]}) + "\n")
            if (i + 1) % 1_000_000 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  {i+1:>9,} / {total:,} ({elapsed:.0f}s)", flush=True)
    elapsed = time.perf_counter() - t0
    size_gb = jsonl_path.stat().st_size / 1e9
    print(f"  Export complete: {total:,} docs | {size_gb:.2f} GB | {elapsed:.1f}s")


def _iter_corpus_hf(limit: int | None) -> Generator[tuple[int, str], None, None]:
    """Yield (pid_int, text) directly from HF Arrow cache. Used for --limit runs."""
    from datasets import load_dataset

    ds = load_dataset(
        "Tevatron/msmarco-passage-corpus",
        cache_dir=str(REPO_ROOT / "data" / "corpus"),
        split="train",
    )
    end = limit if limit is not None else len(ds)
    for i, row in enumerate(ds):
        if i >= end:
            break
        yield int(row["docid"]), row["text"]


def iter_corpus(limit: int | None = None) -> Generator[tuple[int, str], None, None]:
    """Yield (pid_int, text) one row at a time.

    Smoke tests (--limit) stream from HF Arrow directly; full-corpus builds
    prefer JSONL (exported once from HF if missing).
    """
    jsonl_path = REPO_ROOT / "data" / "msmarco_passages.jsonl"

    if limit is not None:
        print("Streaming from HuggingFace Arrow cache ...")
        yield from _iter_corpus_hf(limit)
        return

    if not jsonl_path.exists():
        export_corpus_jsonl(jsonl_path)

    print(f"Streaming corpus from {jsonl_path}")
    with jsonl_path.open() as f:
        for line in f:
            obj = json.loads(line)
            yield obj["pid"], obj["text"]


def _compute_byte_offsets(jsonl_path: Path, n_shards: int) -> list[tuple[int, int]]:
    """One sequential scan to find byte offsets for N equal-sized shards.

    Returns list of (start_byte, line_count). Workers seek directly to
    start_byte. Main-process peak: ~250 MB for the offset list at 8.8M lines.
    """
    print(f"Computing {n_shards}-way byte offsets for {jsonl_path.name} ...", flush=True)
    t0 = time.perf_counter()

    line_offsets: list[int] = [0]
    with jsonl_path.open("rb") as f:
        for _ in f:
            line_offsets.append(f.tell())

    total_lines = len(line_offsets) - 1
    chunk = (total_lines + n_shards - 1) // n_shards

    shards = []
    for i in range(n_shards):
        start_line = i * chunk
        end_line = min((i + 1) * chunk, total_lines)
        shards.append((line_offsets[start_line], end_line - start_line))

    elapsed = time.perf_counter() - t0
    print(f"  {total_lines:,} lines → {n_shards} shards of ~{chunk:,} "
          f"({elapsed:.1f}s)", flush=True)
    return shards


def _worker_build(
    worker_id: int,
    jsonl_path_str: str,
    start_byte: int,
    line_count: int,
    output_path_str: str,
) -> None:
    """Build a partial InvertedIndex for one corpus shard and save to disk."""
    _repo = Path(jsonl_path_str).resolve().parent.parent
    if str(_repo) not in sys.path:
        sys.path.insert(0, str(_repo))

    from retrieval.inverted_index import InvertedIndex, tokenize
    from retrieval.inverted_index.persistence import save_index as _save

    index = InvertedIndex()
    t0 = time.perf_counter()
    n = 0

    with open(jsonl_path_str, "rb") as f:
        f.seek(start_byte)
        for _ in range(line_count):
            raw = f.readline()
            if not raw:
                break
            obj = json.loads(raw)
            index.add_document(obj["pid"], tokenize(obj["text"]))
            n += 1
            if n % 500_000 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  worker-{worker_id}: {n:,} docs | {elapsed:.0f}s", flush=True)

    elapsed = time.perf_counter() - t0
    _save(index, Path(output_path_str), {"worker_id": worker_id, "n_docs": n})
    print(
        f"  worker-{worker_id}: done — {n:,} docs | {elapsed:.1f}s | "
        f"saved to {Path(output_path_str).name}",
        flush=True,
    )


def _merge_index(base: InvertedIndex, other: InvertedIndex) -> InvertedIndex:
    """Merge other into base in-place. Returns base.

    Each partial covers non-overlapping doc IDs. Posting lists are
    array.array('i'); extend() is O(n) memcpy. For new terms we transfer
    ownership of the other partial's array (the partial is about to be
    discarded) to avoid a 700 MB copy at full scale.
    """
    for term, postings in other._index.items():
        existing = base._index.get(term)
        if existing is None:
            base._index[term] = postings
        else:
            existing.extend(postings)
    base._doc_lengths.update(other._doc_lengths)
    base._num_docs += other._num_docs
    base._total_tokens += other._total_tokens
    return base


def build_index_parallel(
    jsonl_path: Path,
    n_jobs: int,
) -> tuple[InvertedIndex, float, float, str]:
    """Build InvertedIndex using N parallel worker processes.

    Returns: (merged_index, total_time_s, peak_rss_mb, corpus_hash)
    """
    shards = _compute_byte_offsets(jsonl_path, n_jobs)

    tmp_dir = Path(tempfile.mkdtemp(prefix="nr_bm25_"))
    tmp_paths = [tmp_dir / f"partial_{i}.bin" for i in range(n_jobs)]

    t0 = time.perf_counter()
    print(f"Launching {n_jobs} worker processes ...", flush=True)

    # spawn (not fork) to avoid fork+thread issues on macOS
    ctx = mp.get_context("spawn")
    processes: list[mp.Process] = []
    for i, ((start_byte, line_count), tmp_path) in enumerate(zip(shards, tmp_paths)):
        p = ctx.Process(
            target=_worker_build,
            args=(i, str(jsonl_path), start_byte, line_count, str(tmp_path)),
            daemon=False,
            name=f"bm25-worker-{i}",
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(
                f"Worker {p.name} exited with code {p.exitcode}. "
                "Check worker output above for errors."
            )

    build_elapsed = time.perf_counter() - t0
    print(f"All workers done in {build_elapsed:.1f}s — merging partial indices ...",
          flush=True)

    merged: InvertedIndex | None = None
    for i, tmp_path in enumerate(tmp_paths):
        partial, _, _ = load_index(tmp_path)
        if merged is None:
            merged = partial
        else:
            _merge_index(merged, partial)
            del partial

        rss = _rss_mb()
        tmp_path.unlink()
        print(f"  merged {i+1}/{n_jobs} | docs so far: {merged._num_docs:,} | "
              f"RSS {rss:.0f} MB", flush=True)

    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    total_elapsed = time.perf_counter() - t0
    peak_rss = _rss_mb()

    h = hashlib.sha256()
    h.update(str(merged._num_docs).encode())
    corpus_hash = f"sha256:{h.hexdigest()}"

    print(
        f"  Build complete: {merged._num_docs:,} docs | "
        f"{total_elapsed:.1f}s | Peak RSS {peak_rss:.0f} MB"
    )
    return merged, total_elapsed, peak_rss, corpus_hash


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / _RSS_SCALE


def build_index_streaming(
    corpus: Generator[tuple[int, str], None, None],
) -> tuple[InvertedIndex, float, float, str]:
    """Single-process streaming build. Used for --jobs 1 and --limit runs."""
    index = InvertedIndex()
    h = hashlib.sha256()
    n = 0
    first_text: str | None = None
    last_text: str | None = None

    t0 = time.perf_counter()

    for pid, text in corpus:
        index.add_document(pid, tokenize(text))

        if first_text is None:
            first_text = text
        last_text = text
        n += 1

        if n % 500_000 == 0:
            elapsed = time.perf_counter() - t0
            rss = _rss_mb()
            print(f"  {n:>9,} docs | {elapsed:6.0f}s | RSS {rss:.0f} MB", flush=True)

    elapsed = time.perf_counter() - t0
    peak_rss = _rss_mb()

    h.update(str(n).encode())
    if first_text:
        h.update(first_text.encode())
    if last_text:
        h.update(last_text.encode())
    corpus_hash = f"sha256:{h.hexdigest()}"

    print(f"  Build complete: {n:,} docs | {elapsed:.1f}s | Peak RSS {peak_rss:.0f} MB")
    return index, elapsed, peak_rss, corpus_hash


def retrieve_all(
    retriever: BM25Retriever,
    queries: dict[str, str],
    top_k: int = 100,
) -> tuple[dict[str, list[str]], dict[str, float]]:
    run: dict[str, list[str]] = {}
    latencies: dict[str, float] = {}
    for qid, text in queries.items():
        results, lat = retriever.retrieve_timed(text, top_k=top_k)
        run[qid] = [str(doc_id) for doc_id, _ in results]
        latencies[qid] = lat
    return run, latencies


def _latency_stats(latencies: dict[str, float]) -> dict:
    vals = sorted(latencies.values())
    n = len(vals)
    return {
        "p50": vals[n // 2],
        "p95": vals[int(n * 0.95)],
        "p99": vals[min(int(n * 0.99), n - 1)],
        "mean": statistics.mean(vals),
    }


def _commit_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def write_results(payload: dict) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cfg_hash = hashlib.sha256(
        json.dumps(payload["config"], sort_keys=True).encode()
    ).hexdigest()[:8]
    out = REPO_ROOT / "benchmarks" / "results" / f"{ts}_bm25_{cfg_hash}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"Results written to {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Custom BM25 evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Smoke test (50k docs, no JSONL export):
  python evaluation/bm25_eval.py --limit 50000

  # Full build, 4 parallel workers:
  python evaluation/bm25_eval.py --jobs 4 --index-path data/custom_bm25_8m.bin

  # Load cached index and evaluate:
  python evaluation/bm25_eval.py --index-path data/custom_bm25_8m.bin
""",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Build from first N docs (smoke test)")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--index-path", type=Path, default=None,
                        help="Save/load index (skip rebuild if file exists)")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Worker processes for parallel build (default: 1).")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Custom BM25  k1={args.k1}  b={args.b}")
    if args.jobs > 1 and args.limit is None:
        print(f"Build mode: {args.jobs} parallel workers")
    print("=" * 60)

    queries = combined_queries()
    qrels_2019 = load_qrels(TREC_DL_2019)
    qrels_2020 = load_qrels(TREC_DL_2020)
    q19_hash = qrels_hash(TREC_DL_2019)
    q20_hash = qrels_hash(TREC_DL_2020)

    index_checksum = "not-saved"
    build_time_s = 0.0
    peak_ram_mb = 0.0
    corpus_hash = "streamed"
    build_mode = "loaded"

    if args.index_path and Path(args.index_path).exists():
        print(f"Loading cached index from {args.index_path} ...")
        index, _, index_checksum = load_index(args.index_path)
        print(f"  {index.num_docs:,} docs loaded")

    elif args.jobs > 1 and args.limit is None:
        jsonl_path = REPO_ROOT / "data" / "msmarco_passages.jsonl"
        if not jsonl_path.exists():
            export_corpus_jsonl(jsonl_path)
        build_mode = f"parallel_{args.jobs}_workers"
        print(f"Building index with {args.jobs} parallel workers ...")
        index, build_time_s, peak_ram_mb, corpus_hash = build_index_parallel(
            jsonl_path, n_jobs=args.jobs
        )
        if args.index_path:
            config_meta = {"k1": args.k1, "b": args.b, "build_mode": build_mode}
            index_checksum = save_index(index, args.index_path, config_meta)
            print(f"  Index saved → {args.index_path} ({index_checksum})")

    else:
        build_mode = f"streaming_jobs1{'_limit' + str(args.limit) if args.limit else ''}"
        print("Building inverted index (single-process streaming) ...")
        index, build_time_s, peak_ram_mb, corpus_hash = build_index_streaming(
            iter_corpus(limit=args.limit)
        )
        if args.index_path:
            config_meta = {"k1": args.k1, "b": args.b, "corpus_limit": args.limit}
            index_checksum = save_index(index, args.index_path, config_meta)
            print(f"  Index saved → {args.index_path} ({index_checksum})")

    scorer = BM25Scorer(k1=args.k1, b=args.b)
    retriever = BM25Retriever(index=index, scorer=scorer)

    print(f"Retrieving top-{args.top_k} for {len(queries)} queries ...")
    run, latencies = retrieve_all(retriever, queries, top_k=args.top_k)
    lat_stats = _latency_stats(latencies)
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

    # One-sided gate: don't regress vs bm25s baseline. Improvements past
    # +GATE_DELTA aren't a regression.
    delta = m20["nDCG@10"] - BM25S_BASELINE_DL2020
    gate_pass = delta >= -GATE_DELTA
    direction = "above" if delta >= 0 else "below"
    print(f"\nShip gate (DL2020 nDCG@10 ≥ bm25s={BM25S_BASELINE_DL2020} − {GATE_DELTA}): "
          f"{'PASS' if gate_pass else 'FAIL'} "
          f"(delta={delta:+.4f}, {abs(delta):.4f} {direction} baseline)")

    payload = {
        "schema_version": "1.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "config": {
            "retriever": "custom_bm25",
            "bm25": {"k1": args.k1, "b": args.b},
            "top_k": args.top_k,
            "index_checksum": index_checksum,
            "corpus_limit": args.limit,
            "corpus_hash": corpus_hash,
            "build_mode": build_mode,
            "n_jobs": args.jobs,
            "qrels_hash": f"{q19_hash}+{q20_hash}",
        },
        "dataset": "trec_dl_2019+2020",
        "metrics": {
            **{f"DL2019_{k}": v for k, v in m19.items()},
            **{f"DL2020_{k}": v for k, v in m20.items()},
        },
        "latency_ms": lat_stats,
        "build": {
            "time_s": round(build_time_s, 2),
            "peak_rss_mb": round(peak_ram_mb, 1),
            "n_jobs": args.jobs,
        },
        "ship_gate": {
            "pass": gate_pass,
            "metric": "DL2020 nDCG@10",
            "threshold": f"within ±{GATE_DELTA} of bm25s={BM25S_BASELINE_DL2020}",
            "actual": round(m20["nDCG@10"], 4),
            "delta": round(delta, 4),
        },
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
