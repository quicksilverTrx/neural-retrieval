# Neural Retrieval System

A multi-stage retrieval and ranking system built over MS MARCO v1 (8.8M passages). The core is a custom inverted index and dense retriever fused with Reciprocal Rank Fusion, re-ranked by a NanoLlama 127M cross-encoder fine-tuned on MS MARCO and a LambdaMART model trained on hand-crafted features. Retrieval quality is evaluated rigorously on TREC DL 2019 + 2020 using nDCG, MRR, and Recall implemented from scratch. The system includes production ingestion with WAL-based dual-index consistency, OpenTelemetry tracing, ACL-aware filtering, and a RAG layer with NLI-based faithfulness measurement.

---

## Results

| System | DL2020 nDCG@10 | DL2019 nDCG@10 | DL2020 Recall@100 | P99 (ms) |
|---|---|---|---|---|
| BM25s library | 0.428 | 0.363 | 0.424 | 565 |
| Custom BM25 | — | — | — | — |
| Dense (MiniLM + FAISS) | — | — | — | — |
| Hybrid RRF | — | — | — | — |
| Hybrid + cross-encoder | — | — | — | — |

---

## What's in the Repo

**Evaluation** — `evaluation/`

- `metrics.py` — nDCG@k, MRR@k, Recall@k, per-query breakdown, bootstrap CI. Implemented from Järvelin & Kekäläinen (2002). No pytrec_eval dependency.
- `trec_eval.py` — loader for TREC DL 2019 (43 queries, ~9.2K judgments) and TREC DL 2020 (54 queries, ~11.4K judgments).
- `bm25s_baseline.py` — BM25s library baseline on 8.8M passages. Index cached after first build (~30 min); subsequent runs load in ~5s.

**Retrieval** — `retrieval/`

- `chunker.py` — 256-token windows, 32-token stride, word-boundary tokenization matching the BM25 index tokenizer.

**Benchmarks** — `benchmarks/`

- `methodology.md` — evaluation design rationale for each system.
- `SCHEMA.md` — canonical result JSON schema.
- `results/` — one immutable JSON file per experiment run.

**Tests** — `tests/`

- `tests/evaluation/` — 31 tests: loader correctness + metric behavioural tests (nDCG, MRR, Recall edge cases).
- `tests/retrieval/` — 30 tests: chunker unit tests + Hypothesis property tests.
- `tests/fixtures/trec_dl_2020_tiny.json` — 10-query, 100-passage fixture with graded qrels (0–3).

**Data pipeline** — `data/prepare_msmarco.py`, `scripts/bootstrap_data.sh`

---

## Setup

```bash
conda activate foundry-llm
pip install -e ".[dev]"

# Download MS MARCO + TREC DL data (~42 GB)
bash scripts/bootstrap_data.sh

# Run BM25s baseline
python evaluation/bm25s_baseline.py --top-k 100

# Run tests
pytest tests/ -v
```

---

## Connection to foundry-llm

Extends [foundry-llm](https://github.com/quicksilverTrx/foundry-llm) (NanoLlama 127M pretrain + P3 serving):

- NanoLlama → cross-encoder re-ranker (fine-tuned with `[RELEVANCE_QUERY]` / `[RELEVANCE_PASSAGE]` tokens)
- `quant.py` → INT8 quantization × nDCG divergence experiment
- P3 serving patterns → `/search`, `/rag`, `/health`, `/metrics` endpoints
