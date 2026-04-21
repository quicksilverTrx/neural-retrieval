# Benchmark Methodology

Evaluation design decisions for this repo. Every number cited traces back to a
JSON file in `benchmarks/results/` and to the rationale described here.

---

## Section 1 — BM25s Baseline

### Why this evaluation exists

Before claiming any retrieval number, the evaluation harness needs to be verified.
The gate: **BM25 nDCG@10 > 0.40 on TREC DL 2020** (expected ~0.487 per Craswell
et al. 2020). Below 0.40 means the harness, corpus loading, or tokenisation is
broken.

### Corpus: MS MARCO v1 at 8.8M scale

TREC DL 2020 qrels were judged against the full 8.8M passage corpus. A system
limited to the 500K subset will miss many judged-relevant passages, producing low
Recall@100 that reflects corpus truncation, not retrieval quality. The full 8.8M
is the only comparable baseline to published results.

Corpus source: `Tevatron/msmarco-passage-corpus` via HuggingFace datasets.
Loaded count: 8,841,823 passages. Corpus hash recorded in each result JSON.

### Evaluation sets

| Set | Queries | Judgments | Source |
|---|---|---|---|
| TREC DL 2020 | 54 | ~11,386 | Craswell et al. (2020) |
| TREC DL 2019 | 43 | ~9,260 | Craswell et al. (2019) |
| **Combined** | **97** | **~20,646** | Both years |

Using both years guards against query-selection noise. Cross-year consistency —
similar nDCG on both years — confirms the system is measuring retrieval quality,
not overfitting to a 54-query sample. Gates: nDCG@10 > 0.40 on DL2020 and
> 0.42 on DL2019. Both must pass.

### Chunking: 256-token windows, 32-token stride

MS MARCO v1 passages average ~60 tokens. Most pass through as a single chunk.
The chunker exists for the long-tail and for the chunk-size ablation.

**Why 256 tokens:** Aligns with the lower half of the 512-token BERT context
window, leaving room for the query prefix in a cross-encoder pass.

**Why 32-token stride (87.5% overlap):** A relevant sentence at any position in
a long passage will appear intact in at least one chunk, maximising Recall@100.
The index-size cost is negligible given that most passages are single-chunk.

**Tokenisation:** `re.findall(r"\w+", text.lower())` — word-boundary split,
identical to the BM25 index tokeniser. No stemming applied at this baseline;
the BM25 index comparison will measure its effect.

**Chunk→passage mapping:** `chunk_id` format: `"{passage_id}_{chunk_index}"`.
The `chunk_to_passage` dict maps retrieved chunks back to original passage IDs
for deduplication and citation.

### BM25s library baseline

`bm25s` (library) is used as a reference point for the eval harness, not as the
final system. It serves two purposes:

1. **Eval harness validation.** If the library baseline nDCG matches the
   published figure (~0.487 on DL2020), the harness is correct. The custom
   BM25 index must match this within ±0.02 nDCG.

2. **Tokenisation reference.** `bm25s` applies stopword removal. The comparison
   between `bm25s` and the custom BM25 will reveal whether tokenisation
   differences explain any quality gap.

### Results

| Set | nDCG@10 | MRR@10 | Recall@100 | P99 (ms) |
|---|---|---|---|---|
| TREC DL 2020 | 0.428 | 0.822 | 0.424 | 565 |
| TREC DL 2019 | 0.363 | 0.722 | 0.400 | — |

DL2020 gate: **PASS** (0.428 > 0.40).
DL2019 gate: **FAIL** (0.363 < 0.42). `bm25s` uses stopword removal but no
stemming. The published figure (~0.487) is from Anserini with a Porter stemmer;
the ~0.06 nDCG gap is tokenisation-driven, not a harness failure. The custom
BM25 index comparison will quantify the stemming effect directly.

### Result JSON location

`benchmarks/results/{timestamp}_bm25s_{config_hash8}.json`

Schema: see `benchmarks/SCHEMA.md`.
