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

## Notable findings

- **Custom BM25 beats the `bm25s` library reference by +0.034 nDCG@10 on TREC DL 2020** after a stopword + numpy-`bincount` rewrite of `score_batch` (47× P50 / 24× P99 latency improvement). Same data structure, same formula — the win is operational. See `docs/design_decisions.md` #15, #17.
- **PQ-ceiling falsification.** Rebuilt the FAISS index four ways (m=16 / m=32 / SQ8 / Flat) from the same MiniLM embeddings: doubling sub-quantisers (m=16 → m=32) recovered 96.5% of the IVF-Flat recall ceiling at 1/35× the size, P99 still inside 20 ms. The remaining gap to the spec's R@100 = 0.75 target is encoder-bound, not index-bound — the experiment quantified what was previously a guess. See `docs/design_decisions.md` #21.
- **Cross-stage memory contention surfaced via OTel boundaries.** The pre-optimisation latency report showed BM25 P99 = 25.7 s and ACL P99 = 21 ms. After the BM25 fix, ACL P99 dropped to 3.93 ms with no code change to ACL — the original 21 ms wasn't filter cost, it was GC pressure leaking from BM25's 7 GB transient candidate set. The misdiagnosis was caught only because every stage had its own span. See `docs/design_decisions.md` #17.
- **α=0.4 score-fusion beats RRF on this system; RRF kept as the conservative fallback.** When the dense leg is strong (m=32), explicit weighting captures more joint signal than rank-fusion. When the encoder distribution shifts, RRF still works without retuning. See `docs/design_decisions.md` #11.

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
- `observability/tracing.py` — OpenTelemetry SDK setup + `retrieval_span()` context manager that wraps each retrieval stage. Span names are constants (`SPAN_BM25`, `SPAN_DENSE_ENCODE`, `SPAN_FAISS_SEARCH`, `SPAN_RRF`, `SPAN_ACL`, `SPAN_QUERY`); falls back to a no-op tracer if the OTLP collector is unreachable so observability never blocks the request path.

**Serving** — `api/`

- `api/main.py` — FastAPI app with `POST /search` (modes: bm25 / dense / hybrid), `GET /health` (liveness), `GET /ready` (returns 503 until indexes are loaded), `GET /metrics` (Prometheus text). Lifespan loader handles BM25 + dense + FAISS validate/rebuild + ACL with graceful degradation. `CorrelationIDMiddleware` propagates `X-Request-ID` to OTel root spans and response headers.

**Evaluation drivers**

- `evaluation/bm25_eval.py` — builds the custom BM25 index in parallel from a JSONL corpus cache, then runs the TREC DL 2019 + 2020 evaluation. Supports incremental index reuse via `--index-path` and per-shard parallel builds via `--jobs N`.
- `evaluation/encode_corpus.py` — CLI for streaming-encoding the corpus into `data/embeddings/{model}/embeddings.npy` (one model at a time).
- `evaluation/build_faiss.py`, `build_faiss_flat.py` — train + add the production IVF-PQ index (or the IVF-Flat falsification variant) from existing embeddings.
- `evaluation/dense_eval.py` — TREC DL 2019+2020 dense-only eval; `--sweep` runs the nprobe sweep at 1/4/8/16/32/64.
- `evaluation/pq_ceiling_experiment.py` — PQ-ceiling sweep that exercised m∈{16,32}, IVF-SQ8, and IVF-Flat from the same embeddings; the comparison that picked m=32 for production.
- `evaluation/hybrid_eval.py` — runs BM25 + dense + RRF (k=60) + α-sweep ablation on the same 97 TREC DL queries; emits per-system + per-query metrics.
- `evaluation/acl_eval.py` — measures Recall@100 drop per role with the post-retrieval ACL filter at configurable oversample factors.
- `evaluation/latency_report.py` — runs 100 sample queries through each pipeline stage (BM25 / dense_encode / faiss_search / rrf / acl) and records per-stage P50/P95/P99. Used as the audit trail behind decision #17 (BM25 latency optimisation).

**Load testing** — `tests/load/`

- `tests/load/locustfile.py` — Locust harness with realistic 60% hybrid / 20% BM25 / 10% dense / 10% health task weights. Run as `locust -f tests/load/locustfile.py --host http://localhost:8000 --headless -u N -r N --run-time 60s`. Used to produce the 1/5/10/25/50-user throughput sweep (`benchmarks/methodology.md` §5).

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
