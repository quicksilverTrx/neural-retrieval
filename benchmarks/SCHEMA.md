# Results JSON Schema (canonical — v1.0)

Every evaluation run writes one file to `benchmarks/results/`. Never overwrite.

**Filename:** `{iso_timestamp}_{experiment}_{config_hash8}.json`

**Rules:**
- Never overwrite — always a new filename per run.
- `corpus_hash` + `qrels_hash` catch silent data drift between runs.
- `per_query` enables per-query failure analysis.
- `environment.commit_sha` eliminates mystery regressions.
- `schema_version` bumps require an explicit migration script — do not freely add top-level fields.

```json
{
  "schema_version": "1.0",
  "timestamp_utc": "2026-05-01T14:32:11Z",
  "experiment": "hybrid_rrf",
  "config": {
    "retriever": "hybrid_rrf",
    "encoder": "all-MiniLM-L6-v2",
    "index": {"type": "IVFPQ", "nlist": 4096, "m": 16, "nbits": 8, "nprobe": 16},
    "bm25": {"k1": 1.2, "b": 0.75},
    "rrf": {"k": 60},
    "chunk": {"size_tokens": 256, "stride_tokens": 32},
    "corpus_hash": "sha256:abc123...",
    "qrels_hash": "sha256:def456..."
  },
  "dataset": "trec_dl_2019+2020",
  "metrics": {
    "DL2020_nDCG@10": 0.641,
    "DL2020_MRR@10": 0.789,
    "DL2020_Recall@100": 0.873,
    "DL2019_nDCG@10": 0.598,
    "DL2019_MRR@10": 0.731,
    "DL2019_Recall@100": 0.821
  },
  "latency_ms": {
    "p50": 14.2, "p95": 19.8, "p99": 23.1, "mean": 14.9,
    "stages": {"bm25": 3.1, "encode_query": 2.4, "faiss": 8.2, "rrf": 0.3}
  },
  "per_query": [
    {"qid": "1037496", "latency_ms": 15.2}
  ],
  "environment": {
    "python": "3.11.9",
    "torch": "2.2.2",
    "faiss": "1.8.0",
    "gpu": "Apple M1 Pro",
    "commit_sha": "7f3a9c1"
  },
  "notes": "Optional free-text."
}
```
