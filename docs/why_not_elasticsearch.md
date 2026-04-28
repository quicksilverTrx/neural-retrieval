# Why Not Elasticsearch (or ColBERT or LangChain)?

This document explains the architectural decision to build a custom retrieval
stack rather than adopt established off-the-shelf tools.

---

## Why not Elasticsearch?

**1. Operational overhead doesn't fit the architecture.**
Elasticsearch requires a JVM (typically 8–16 GB heap for an 8.8M-passage index
in production, plus OS buffer cache overhead). For a system where the query path
also runs sentence encoder inference and FAISS search, JVM memory competes with
GPU memory and the embedding memmap. The custom index uses ~700 MB of Python
heap with `array.array('i')` posting-list storage (decision #15). VByte
gap-coding is implemented as a standalone codec but not yet wired into
persistence — see decision #9 for the integration trade-off.

**2. Plugging a neural reranker into ES is operationally painful.**
The production pipeline is:
```
BM25 top-100 → RRF fusion → cross-encoder reranker → top-10
```
ES is a black box for the retrieval stage. Intercepting its ranked list at the
right point, running cross-encoder inference, and returning reranked results
requires either (a) a custom plugin running inside the JVM or (b) a two-hop
round-trip ES→Python→ES. Both add latency and operational coupling.

With the custom index, the BM25 scorer, RRF, and cross-encoder are all in the
same Python process. The stage boundary is a function call.

**3. Direct control over the data structure.**
A custom inverted index makes the BM25 formula directly visible in the scoring
loop and exposes the posting list as a first-class data structure. Bug fixes,
performance work (vectorisation, compression, WAND-style pruning), and feature
work (per-term weighting, custom similarity functions) all happen at the source
rather than against an opaque library API.

### Where Elasticsearch wins

At >100M documents, or in a multi-tenant environment where indexing latency,
distributed search, and query routing are production concerns, ES's operational
investment pays off. At 8.8M passages on a single machine, it is unnecessary.

---

## Why not ColBERT?

ColBERT (Khattab & Zaharia, 2020) uses late interaction: instead of a single
query embedding (bi-encoder) or full cross-attention (cross-encoder), it encodes
query and passage into per-token embeddings and scores with MaxSim aggregation.

### The appeal

ColBERT achieves near-cross-encoder quality at bi-encoder latency. ColBERTv2
with PLAID compression is the SOTA dense retrieval baseline on MS MARCO.

### Why not here

1. **Index size.** ColBERT stores one embedding per token, not one per passage.
   At 8.8M passages × ~60 tokens × 128-dim × float32 ≈ 270 GB. The PLAID
   compression in ColBERTv2 reduces this to ~8 GB but requires custom indexing
   infrastructure not provided by vanilla FAISS.

2. **Architectural coupling.** ColBERT replaces the BM25 + dense + reranker
   stack with a single late-interaction model. That removes the ability to
   ablate retrieval components independently, swap in a different reranker, or
   degrade gracefully when one component is unavailable.

### The honest verdict

ColBERT would likely score higher nDCG@10 (literature ~0.72 on similar setups
vs **0.5815 measured** for this system's α=0.4 score-fusion hybrid on TREC DL 2020). The
decision to not use it is architectural, not quality-optimising. The measured
gap reflects that ColBERT's late-interaction representation captures
token-level matching that any single-vector bi-encoder cannot. The
PQ-ceiling experiment (`design_decisions.md` #21) confirmed our index
quality is at 96.5% of IVF-Flat — the residual gap to ColBERT is encoder
representation, not index compression.

---

## Why not LangChain (or LlamaIndex)?

LangChain and LlamaIndex are retrieval-augmented generation orchestration
frameworks. They abstract the retrieval + generation pipeline into configuration.

### The problem

These frameworks are optimised for getting something working fast, not for
understanding what's working or why. The abstraction layers make:
- Debugging a precision drop (BM25? chunking? embedding? reranking?) hard
  without reading framework source code.
- Latency attribution (where is the P99 coming from?) similarly difficult.
- Hallucination root-cause analysis (is it retrieval missing the passage, or
  generation ignoring it?) invisible.

The production reliability concerns (OpenTelemetry tracing, circuit breakers,
checksum validation, ACL filtering) are all below the LangChain abstraction
layer. A LangChain RAG system has none of these — they must be grafted on
after the fact.

### The architectural principle

Every component this system uses is a direct function call with measurable
latency. `BM25Retriever.retrieve_timed()` returns `(results, latency_ms)`. The
OTel span for `faiss_search` starts and ends around exactly
`FAISSIVFPQIndex.search()`.

When something breaks or gets slow, the trace immediately shows which
component. With LangChain, the trace shows `chain.run()`.

### Where LangChain wins

Rapid prototyping, demos, and use cases where time-to-first-result matters
more than production reliability.

---

## Summary

| Dimension | Elasticsearch | ColBERT | LangChain | This system |
|---|---|---|---|---|
| Build time | Fast | Medium | Very fast | Slow (deliberate) |
| Memory footprint | 8–16 GB JVM | 8 GB (PLAID) | varies | ~3 GB |
| Retrieval quality (nDCG@10 DL2020) | ~0.43 (BM25 only) | ~0.72 (lit) | depends on backing | **0.5815** α=0.4 score-fusion measured |
| Cross-encoder pluggability | Painful | N/A | Abstracted | Native function call |
| Observability | ES slow logs | None | Callback hooks | OTel spans, P99 per stage |
| Component-level ablation | Limited | None | Limited | Each stage independently swappable |
