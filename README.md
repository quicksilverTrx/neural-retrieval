# Neural Retrieval System

A multi-stage retrieval and ranking system built over MS MARCO v1 (8.8M passages). The core is a custom inverted index and dense retriever fused with Reciprocal Rank Fusion, re-ranked by a NanoLlama 127M cross-encoder fine-tuned on MS MARCO and a LambdaMART model trained on hand-crafted features. Retrieval quality is evaluated rigorously on TREC DL 2019 + 2020 using nDCG, MRR, and Recall implemented from scratch. The system includes production ingestion with WAL-based dual-index consistency, OpenTelemetry tracing, ACL-aware filtering, and a RAG layer with NLI-based faithfulness measurement.

---

## Results

| System | DL2020 nDCG@10 | DL2019 nDCG@10 | DL2020 Recall@100 | P99 (ms) |
|---|---|---|---|---|
| BM25s library | 0.428 | 0.363 | 0.424 | 565 |
| **Custom BM25** | **0.4622** | **0.3731** | **0.4635** | **720** |
| **Dense (MiniLM + FAISS IVF-PQ m=32)** | **0.5262** | **0.4791** | **0.4037** | **12** |
| Hybrid RRF (k=60) | 0.5240 | 0.4763 | 0.5488 | (BM25-bound) |
| **Hybrid α-fusion (α=0.4, MiniLM m=32)** | **0.5815** | **0.5108** | **0.5450** | (BM25-bound) |
| Hybrid + cross-encoder | — | — | — | — |

---

## What's in the Repo

**Evaluation** — `evaluation/`

- `metrics.py` — nDCG@k, MRR@k, Recall@k, per-query breakdown, bootstrap CI. Implemented from Järvelin & Kekäläinen (2002). No pytrec_eval dependency.
- `trec_eval.py` — loader for TREC DL 2019 (43 queries, ~9.2K judgments) and TREC DL 2020 (54 queries, ~11.4K judgments).
- `bm25s_baseline.py` — BM25s library baseline on 8.8M passages. Index cached after first build (~30 min); subsequent runs load in ~5s.

**Retrieval** — `retrieval/`

- `chunker.py` — 256-token windows, 32-token stride, word-boundary tokenization matching the BM25 index tokenizer.
- `inverted_index/` — custom BM25 over an `array.array('i')` posting-list store (~14× memory reduction vs `list[tuple]`), numpy-vectorised `score_batch`, NRIDX2 binary persistence with sha256 verification, and an in-progress VByte gap codec. See `docs/design_decisions.md` #6, #15, #17 for the rationale and the latency/memory traceback.
- `dense/` — `SentenceEncoder` for batch-encoding 8.8M passages with MPS streaming `.npy` writes, `FAISSIVFPQIndex` (production: `nlist=4096, m=32, nbits=8`), SQLite chunk → passage `lookup`, and sha256-verified index `recovery` with in-place rebuild.
- `fusion/` — Reciprocal Rank Fusion (`rrf.fuse`, `rrf.fuse_scored`) with the Cormack-Clarke-Buettcher k=60 default. Pure rank-based, no score normalisation; both rank-list and (doc_id, score)-list APIs.
- `acl.py` — `PassageACL` synthetic role-bitmap generator (admin / engineer / analyst / sales / viewer over 8.8M passages) plus `ACLFilter` for post-retrieval filtering. See `docs/design_decisions.md` #10 for the post-retrieval-vs-IDSelector trade-off.

**Evaluation drivers**

- `evaluation/bm25_eval.py` — builds the custom BM25 index in parallel from a JSONL corpus cache, then runs the TREC DL 2019 + 2020 evaluation. Supports incremental index reuse via `--index-path` and per-shard parallel builds via `--jobs N`.
- `evaluation/encode_corpus.py` — CLI for streaming-encoding the corpus into `data/embeddings/{model}/embeddings.npy` (one model at a time).
- `evaluation/build_faiss.py`, `build_faiss_flat.py` — train + add the production IVF-PQ index (or the IVF-Flat falsification variant) from existing embeddings.
- `evaluation/dense_eval.py` — TREC DL 2019+2020 dense-only eval; `--sweep` runs the nprobe sweep at 1/4/8/16/32/64.
- `evaluation/pq_ceiling_experiment.py` — PQ-ceiling sweep that exercised m∈{16,32}, IVF-SQ8, and IVF-Flat from the same embeddings; the comparison that picked m=32 for production.
- `evaluation/hybrid_eval.py` — runs BM25 + dense + RRF (k=60) + α-sweep ablation on the same 97 TREC DL queries; emits per-system + per-query metrics.
- `evaluation/acl_eval.py` — measures Recall@100 drop per role with the post-retrieval ACL filter at configurable oversample factors.

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

# Run BM25s library baseline
python evaluation/bm25s_baseline.py --top-k 100

# Build + evaluate the custom BM25 index (4-way parallel, ~2 min on 8.8M passages)
python evaluation/bm25_eval.py --jobs 4 --index-path data/custom_bm25_8m.bin

# Run tests
pytest tests/ -v
```

---

## Connection to foundry-llm

Extends [foundry-llm](https://github.com/quicksilverTrx/foundry-llm) (NanoLlama 127M pretrain + P3 serving):

- NanoLlama → cross-encoder re-ranker (fine-tuned with `[RELEVANCE_QUERY]` / `[RELEVANCE_PASSAGE]` tokens)
- `quant.py` → INT8 quantization × nDCG divergence experiment
- P3 serving patterns → `/search`, `/rag`, `/health`, `/metrics` endpoints
