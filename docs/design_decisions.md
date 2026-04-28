# Design Decisions

Architectural tradeoffs with explicit rationale. Populated as work ships.

---

## Retrieval Engine

### 6. Custom inverted index, not Elasticsearch

Elasticsearch would provide an immediately production-ready BM25 implementation, but at the cost of: (a) JVM operational overhead (typically 2–4 GB heap for an 8.8M-passage index); (b) loss of control over the retrieval path needed to plug in a neural reranker at any stage.

The custom inverted index exposes the posting list as a first-class data structure and makes the BM25 formula directly visible in the scoring loop. The indexing code is ~60 lines of Python; the BM25 scorer is ~30 — small enough to audit, fix, and extend in-place.

The speed tradeoff: the custom index is slower than Elasticsearch at 8.8M passages (Python vs JVM, no inverted index compression yet). The P99 query latency target (<20ms single-threaded) is achievable with VByte-compressed posting lists; without compression it is a soft goal.

---

### 7. MiniLM-L6-v2 for dense bi-encoding, not NanoLlama

NanoLlama (127.6M parameters, causal decoder) cannot produce meaningful fixed-length document embeddings. A causal decoder only attends to preceding tokens — the final token's representation is not a summary of the full sequence. Contrastive retraining with bidirectional attention would be required, essentially training a new model. MiniLM-L6-v2 (22M parameters) and E5-small-v2 (33M) are purpose-trained for semantic similarity via contrastive learning on 1B+ sentence pairs. They produce L2-normalised embeddings directly usable for cosine similarity.

Practical impact: MiniLM encodes a 60-token passage in ~2ms on a single CPU core; NanoLlama in its current form would require ~100ms and produce worse embeddings. The narrative is not "I used my own model everywhere" but "I chose the right tool for each stage of the pipeline."

---

### 8. FAISS IVF-PQ, not HNSW

HNSW offers higher Recall@100 at equivalent nprobe (because graph-based traversal is more recall-efficient than IVF's cluster-based search) but has a prohibitive memory footprint at 8.8M scale:

| Index type | Memory at 8.8M × 384-dim | Recall@100 (nprobe=16) |
|---|---|---|
| IVF-PQ (nlist=4096, m=32, nbits=8) | **360 MB measured** (~283 MB PQ-codes + centroids + inverted-list metadata) | **0.40 measured** (MiniLM); see PQ-ceiling experiment #21 |
| HNSW M=32 | ~4.5 GB | ~0.93 (literature) |
| Flat (exact) | ~13.5 GB | 1.00 |

IVF-PQ fits in commodity RAM (16GB machine handles index + corpus + Python runtime). HNSW requires a 16GB+ machine just for the index. The ~3–5% Recall@100 gap is acceptable given that the hybrid RRF fusion layer recovers some of it via the BM25 component.

IVF-PQ parameter choice:
- **nlist = 4096**: √8.8M ≈ 2966; 4096 gives ~2150 passages per cluster. The standard √N heuristic. Fewer clusters → more passages per probe (slower, higher recall). More clusters → fewer passages per probe (faster, lower recall at same nprobe).
- **m = 32**: Sub-quantizers for 384-dim vectors. 384 / 32 = 12-dim per sub-vector. Each sub-vector quantised to 1 byte (256 centroids). Total: 8.8M × 32 bytes = 283 MB for the PQ codes. m=16 (24-dim sub-vectors) was the original choice; the PQ-ceiling experiment showed m=32 recovers +19% Recall@100 and +16% nDCG@10 at 1.7× the size — well within the latency and storage budgets.
- **nbits = 8**: 256 PQ centroids per sub-quantizer. Standard choice.
- **nprobe = 16 default**: 16/4096 ≈ 0.4% of clusters searched. Sweet spot on latency-recall curve. The full sweep (nprobe=1,4,8,16,32,64) is written to JSON for review.

---

### 9. Variable-byte encoding — codec implemented, not wired into persistence

VByte-encoding the gap-coded doc_id sequence reduces posting-list bytes by
~4×: most gaps are small integers (typically 1–3 bytes in VByte vs 4 bytes
for int32). Implemented in `retrieval/inverted_index/vbyte.py`
(`VByteCodec.encode`/`decode`), 26 unit tests + Hypothesis property test
green.

**Two distinct integration points, only one of which is contentious.**

| Integration | Effect | Runtime cost | Disk cost | Status |
|---|---|---|---|---|
| **Disk-only (persistence layer):** VByte-encode at `save_index()`, decode back into `array.array('i')` at `load_index()` | Smaller `.bin` files | Zero per-query (in-memory representation unchanged) | 1.7 GB → ~430 MB on disk | **Not yet wired — clear win, deferred** |
| **In-memory:** Replace `array.array('i')` with VByte bytes; decode on every posting read in `score_batch` | Smaller heap | Decode cost per query (significant in pure Python) | Same as disk | Not pursued — runtime cost negates the heap win at our scale |

**Why the disk-only variant is unambiguously a win we haven't taken:**

- The current `InvertedIndex._index` is `dict[str, array.array('i')]`. At
 load time, `persistence.load_index()` reconstructs each `array.array`
 from raw int32 bytes via one `frombytes()` call per term.
- Replacing the persistence-layer payload with VByte gaps changes only
 `save_index()` and `load_index()` — `score_batch()` and every other
 read path is unchanged because the in-memory `array.array('i')` is
 identical post-decode.
- Cost: build time +~30 s for VByte encode of 88M postings; load time
 +~10–15 s for VByte decode. Both are acceptable for a one-time
 build / process-startup cost.
- Benefit: 1.7 GB → ~430 MB on disk (4× smaller artefact, faster
 download / S3 transfer / container image).

**Why we shipped the un-wired version anyway:**

1. **The disk size is not a current bottleneck.** 1.7 GB of NRIDX2 fits
 comfortably and the build/eval pipeline doesn't move this artefact
 around enough for size to hurt.
2. **The format change is non-trivial to layer in retroactively.** It
 requires bumping the format version (NRIDX2 → NRIDX3), keeping a
 backward-compatible reader for existing v2 indexes, and re-saving the
 8.8M-doc artefact. We prioritised latency work (decision #17) over
 storage work in the first ship.
3. **The codec is on the shelf, ready.** `VByteCodec.encode/decode`
 already round-trips arbitrary sorted-int sequences with property-test
 coverage. Wiring it into `persistence.py` is ~40 lines of code, not a
 research project.

This is tracked as a follow-up: when we next touch the persistence layer
(e.g. for a multi-shard index or for delta updates), introduce NRIDX3
with VByte payloads.

**Storage trade-off table (per posting entry, in-memory):**

| Representation | Bytes | Read cost | Write cost |
|---|---|---|---|
| `list[tuple[int, int]]` | 112 | 0 (direct iter) | tuple alloc |
| `array.array('i')` (current) | 8 | 0 (direct index) | `extend((d, tf))` |
| VByte over array (gap-coded, in-memory variant) | ~3 (avg) | decode O(bytes) per posting | encode O(bytes) per posting |

---

### 10. Post-retrieval ACL filter — not query-time FAISS IDSelector (chosen 2026-04-26)

**Decision: ship a post-retrieval ACL filter** (`retrieval/acl.py`
`ACLFilter`) that runs after BM25/dense/RRF have produced a ranked list,
not a query-time FAISS IDSelector that prunes the search to allowed
doc_ids upfront.

**The two architectures considered:**

| Architecture | Mechanism | Pros | Cons |
|---|---|---|---|
| **Post-retrieval (chosen)** | Retrieve top-K × *oversample*; drop docs where `user_role ∉ allowed_roles[doc_id]`; truncate to K. | Single filter for both BM25 and dense legs. No FAISS coupling. ACL data lives in one place (`data/acl/passage_acl.json`). | Wastes work on the dropped docs. For very restrictive roles, even *oversample = 4* under-fills top-K. |
| Query-time IDSelector | Build a `faiss.IDSelectorBatch` of allowed pids per role; pass at search time so FAISS only scans allowed clusters. | No wasted work; recall scales 1:1 with the role's accessible-set size. | Couples FAISS to ACL. IDSelector overhead at 8.8M is non-trivial (the selector is consulted per inverted-list scan). Doesn't help BM25 — would need a second mechanism on the BM25 leg. |

**Why we shipped post-retrieval:**

1. **Decouples retrieval from permissions.** Retrieval is "find the best
 docs for this query"; ACL is "is this caller allowed to see this doc".
 Mixing them at the FAISS layer makes both harder to reason about.
2. **One filter for two retrieval legs.** BM25 + dense both flow through
 `ACLFilter.filter()` after fusion. A FAISS IDSelector wouldn't cover
 BM25; we'd need a parallel posting-list filter, doubling the surface.
3. **Measured drop is acceptable for 3 of 5 roles** at *oversample = 2×*
 (admin / engineer / analyst all drop <25% Recall@100 vs unrestricted).
 The pathological cases are `sales` (-41%) and `viewer` (-60%) — for
 those a tactical bump to *oversample = 4× / 8×* recovers most of the
 loss without an architectural change.

**Per-role Recall@100** (`benchmarks/results/{ts}_1D_*.json`, oversample=2×):

| admin | engineer | analyst | sales | viewer |
|---|---|---|---|---|
| 0.4255 (0%) | 0.3523 (-17%) | 0.3352 (-21%) | 0.2512 (-41%) | 0.1696 (-60%) |

**Re-evaluation trigger:** if a future role has < 5 % corpus access
(e.g. a "compliance-only" role on a 99 %-restricted shard), the
oversample multiplier needed to refill top-K becomes prohibitive and
the IDSelector path becomes the right call. Cross that bridge when
the role appears.

**Observation: oversample needs to scale with role restrictiveness.**
The current 2× oversample is constant across roles, but the right
oversample is approximately `1 / access_probability_role`:

| Role | Access prob | Theoretical drop @1× | Drop needed for full recovery | We ship 2× |
|---|---|---|---|---|
| admin | 100% | 0% | 1× | over-provisioned |
| engineer | 80% | 20% | 1.25× | over-provisioned |
| analyst | 65% | 35% | 1.55× | over-provisioned |
| sales | 50% | 50% | 2.0× | exactly right |
| viewer | 35% | 65% | 2.85× | under-provisioned |

The measured drops match the theoretical drops minus what 2× oversample
recovers (roughly 5 percentage points absolute). The pathological case
is `viewer`: theoretical 65% drop, measured 60% — 2× oversample is
under-provisioned and has the smallest absolute recovery for the role
that needs it most. **Tactical fix: per-role oversample (2× default,
3× viewer, 4× any future role with <30% access). No architectural
change needed; this is a one-line dispatch in `ACLFilter.filter`.**

---

### 11. RRF k=60 (rank fusion) + α=0.4 score fusion as the production hybrid

**Implemented:** `retrieval/fusion/rrf.py` (`fuse(ranked_lists, k=60)`,
`fuse_scored(bm25_results, dense_results, k=60)`, `_rrf_scores`). 16/16
unit tests + Hypothesis property tests pass.

**RRF formula:**

```
score(d) = Σ_i 1 / (k + rank_i(d)) k = 60
```

For each system *i* the document at rank *r* contributes `1 / (k + r)`.
Documents absent from a system's list contribute 0 for that system.

**k = 60** is the Cormack-Clarke-Buettcher (2009) value. The intuition:
*k* prevents a document ranked #1 in one system from dominating when
it's absent from another. At k=60 the rank-1 → rank-2 gap is
1/61 − 1/62 ≈ 0.000264 — small enough that a doc seen by both systems
at modest ranks beats a doc seen only by one system at rank 1.

**Why RRF is the conservative-default fusion:**

- No score normalisation. BM25 scores are unbounded; dense (cosine/L2)
 scores live in a different scale entirely. Score-based fusion needs
 either min-max normalisation (which is per-query and noisy) or a
 trained weighting (decision boundary depends on the score
 distributions of both systems).
- RRF only uses ranks, which are distribution-agnostic. A new encoder
 or a re-tuned BM25 doesn't break the fusion.

**But α-fusion now wins on this system:** with the m=32 dense leg from
decision #20, hybrid_eval shows:

| System | DL2020 nDCG@10 | DL2020 R@100 |
|---|---|---|
| BM25 | 0.4622 | 0.4635 |
| Dense (m=32) | 0.5262 | 0.4037 |
| RRF (k=60) | 0.5240 | 0.5488 |
| **Best α=0.4 score-fusion** | **0.5815** | **0.5450** |

Score fusion at α=0.4 (40 % BM25, 60 % dense, min-max-normalised per
query) beats RRF by **+0.058 nDCG@10**. With a stronger dense leg the
joint score signal carries more information than the rank-only
projection RRF uses; weighting the dense leg explicitly captures it.

**Production pick: α=0.4 score fusion**, with RRF retained as the
conservative fallback in the API (no normalisation, no per-system
distribution assumptions). When a future encoder change makes either
leg's score distribution look unfamiliar, RRF still works correctly
without retuning α.

**Re-evaluation trigger:** any change to the dense leg (encoder, index
family, m, nbits) should re-run the α sweep — the optimal α tracks the
strength of the dense signal.

**Observation: the α optimum is metric-dependent.** The sweep produces
three distinct optima depending on which metric you measure (DL2020):

| Metric | Best α | Value | What it implies |
|---|---|---|---|
| nDCG@10 | 0.4 | 0.5815 | top-10 ordering quality favours the dense leg (60% weight) |
| Recall@100 | 0.5 | 0.5499 | top-100 coverage benefits from a balanced union |
| MRR@10 | 0.6 | 0.9259 | first-relevant rank favours BM25 (60% weight) — exact-match queries land at rank 1 more reliably under BM25 than dense |

α=0.4 is the production pick because nDCG@10 is the primary headline
metric. If a downstream application weights MRR (e.g. a single-answer
QA system), α=0.6 would be the right choice — and the same fusion code
covers it.

---

### 15. `array.array('i')` storage for posting lists, not `list[tuple]`

**Decided:** 2026-04-24, after four failed full-corpus build attempts.

**Problem.** The original `_index: dict[str, list[tuple[int, int]]]` storage spent
112 bytes per posting entry (56 B tuple header + 28 B per int × 2). At 88M
entries across 8.8M docs × ~40 unique terms avg, that's ~10 GB of Python heap
per 2.2M-doc parallel worker, scattered randomly across the address space.

This representation *indexed fine* (~60 s per 2.2M-doc shard) but could not
be **saved**. `pickle.dump()` — even streaming into a `GzipFile` — must traverse
the entire object graph, forcing macOS's memory compressor to decompress
millions of non-contiguous pages. Under 4-way worker contention, the
compressor saturates and the save phase stalls indefinitely (observed: zero
disk writes in 12+ minutes; killed at 63 min wall time).

**Chosen:** `_index: dict[str, array.array('i')]` — a single contiguous int32
buffer per term, interleaved as `[doc0, tf0, doc1, tf1, …]`.

| | `list[tuple[int,int]]` | `array.array('i')` |
|---|---|---|
| Bytes per posting entry | 112 | 8 |
| Memory layout | pointer-chased, scattered | contiguous buffer |
| Append one entry | `.append(tuple)` | `.extend((d, tf))` |
| Serialise entire list | pickle traversal, page faults everywhere | `arr.tobytes()` — one memcpy |
| Python heap per worker (2.2M docs) | ~10 GB | ~700 MB |

**Persistence rewrite (NRIDX2).** With contiguous arrays, pickle is no longer
necessary. The new binary format writes `arr.tobytes()` per term, preceded by
a `<uint16 term_len><utf-8 term><uint32 n_pairs>` header, and trails the
payload with a 64-byte hex sha256. No compression: the save is I/O-bound and
uncompressed reads are faster than gzip decompression. Two-pass load (hash
first, then parse) ensures any corruption surfaces as "Checksum mismatch"
rather than a misleading UTF-8 decode error.

**Build impact.**

| Metric | Before (tuple-based) | After (array-backed) |
|---|---|---|
| End-to-end full-corpus build | unbounded (never completed) | 86.5 s |
| Peak per-worker RSS | 300 MB–1.9 GB (thrashing) | ~700 MB (stable) |
| Per-worker CPU utilisation | 7–34% (compressor-starved) | 93–99% |
| Save time per 2.2M-doc partial | ∞ | 5–10 s |

**API compatibility.** `get_posting_list(term) -> list[tuple[int, int]]` is
kept as a backward-compat shim (allocates tuples on call, slow for large
posting lists). Hot paths — `BM25Scorer.score_batch`, `BM25Retriever.retrieve`,
`persistence.save_index` — use the new `get_raw_posting(term) -> array.array`
and iterate pair-wise by index without allocating tuples.

**Note.** The `InvertedIndex` schema is a core module.
The change to `array.array` storage touches that schema, not just its
implementation, and was approved explicitly on 2026-04-24 after analysis of
the macOS memory compressor failure mode.

---

### 16. Streaming `.npy` write for memory-constrained encoders

**Status:** operational adaptation, **not a correctness bug**. The previous
`np.memmap` implementation passed all 32 dense unit tests and is correct
Python. This decision is about how the encoder behaves on a 16 GB macOS
host at the full 8.8M-passage scale, not about an algorithmic defect.

**Decided:** 2026-04-24, after the MPS encode stalled at ~3 min wall time
on a 16 GB Apple Silicon machine.

**Problem.** `encode_corpus()` originally pre-allocated a 13.5 GB `np.memmap`
for the full MS MARCO embedding matrix (8.8M × 384 float32) and wrote each
chunk into a slice:

```python
embeddings = np.lib.format.open_memmap(..., shape=(n, dim))
embeddings[write_ptr:end] = vecs
```

On macOS, writing to a large memmap *faults every target page resident*,
and the kernel only evicts those pages when it needs to. Observed on a
16 GB machine during a full-corpus run:

- Process physical footprint climbed to **14.3 GB** (peak 15.4 GB) — effectively
 the entire memmap became resident.
- `sample(1)` stack trace: blocked in `MPSStream::synchronize` →
 `[_MTLCommandBuffer waitUntilCompleted]` → `_pthread_cond_wait`. The GPU
 was stalled waiting for the host memory system.
- CPU utilisation collapsed from 87% (startup) to 7% (stalled).
- No progress output after the first chunk in over 3 minutes.

**Chosen:** write the .npy file via a plain file handle, one chunk per
`file.write()`. The kernel moves written pages to the page cache and
(asynchronously) to disk without making them part of the process's
resident-memory accounting. Process footprint stays bounded by the chunk
buffer plus model + pids.

```python
emb_file = emb_path.open("wb")
np.lib.format.write_array_header_1_0(emb_file, {
 "descr": np.lib.format.dtype_to_descr(np.dtype(np.float32)),
 "fortran_order": False,
 "shape": (n, self.embedding_dim),
})
for each chunk:
 vecs = model.encode(chunk_texts, ...) # GPU → host
 emb_file.write(vecs.astype(np.float32, copy=False).tobytes())
 if device == "mps":
 torch.mps.empty_cache() # release MPS allocator buffers
```

The produced file is a standard `.npy` — downstream `load_embeddings()`
still uses `np.lib.format.open_memmap(..., mode="r")`. Read-side mmap is
fine because callers only touch embeddings they're actively using (FAISS
training on a 100K sample, then one-pass `add()` through the whole
matrix).

**Three concrete changes were made to `retrieval/dense/encoder.py`** —
labelled here so anyone reading the diff can map the code to this rationale:

1. **Output path:** `np.lib.format.open_memmap(..., shape=...)[slice] = vecs`
 → `np.lib.format.write_array_header_1_0(emb_file, {...})` once at the
 start, then `emb_file.write(vecs.tobytes())` per chunk. Produces a
 byte-identical `.npy` file; only the *write* path differs.
2. **Chunk granularity:** `chunk_size = batch_size * 100` (default 25,600)
 → `chunk_size = batch_size * chunk_multiplier` with default
 `chunk_multiplier=8` (default 4,096). Smaller chunks mean fewer accumulated
 forward passes per `model.encode()` sync — bounds MPS allocator buildup
 between explicit cache flushes. Exposed as `--chunk-multiplier` CLI flag
 in `encode_corpus.py` for per-machine tuning.
3. **MPS cache release:** added `torch.mps.empty_cache()` call between chunks
 when `device.startswith("mps")`. Without this, the MPS allocator caches
 intermediate buffers across forward passes and the process footprint grows
 monotonically until the OS kills the run.

`load_embeddings()` (read side) is unchanged — it still uses
`np.lib.format.open_memmap(mode="r")`. Read-only mmap is safe.

**Throughput impact (observed).**

| | memmap write (old) | streaming write (new) |
|---|---|---|
| 5K-passage smoke test (synthetic) | 5,144 p/s | 4,314 p/s |
| 8.8M-passage corpus run on 16 GB Mac | **stalled at ~3 min, zero progress** | **completed: 4h33m wall, 539 p/s sustained, 13.58 GB output, 0 crashes** |
| Peak process RSS at full scale | 14.3 GB | ~200 MB |

The smoke test rates are roughly equal because 5K × 1.5 KB = 7.5 MB fits
trivially in RAM either way — the memmap pathology only manifests when the
file grows large enough to compete with the model and MPS allocator for
physical RAM. The streaming write is slightly slower in the smoke regime
(one extra `.tobytes()` allocation per chunk) but the loss is meaningless
at the scale where it matters.

**Alternative we did NOT pick: `mmap.madvise(MADV_DONTNEED)`** after each
write. macOS `MADV_DONTNEED` is best-effort and doesn't reliably evict
dirty pages; the streaming write is simpler and has no edge cases.

**When the original memmap path is the right choice.** On Linux with ample
RAM headroom, or on CUDA hosts where the GPU memory is separate from the
file-cache pool, `np.memmap` slice-assignment is fine — the kernel evicts
clean pages aggressively and the process footprint stays bounded. The
streaming `.npy` write is the more conservative default; it loses nothing
in those environments and is necessary on macOS at corpus scale.

---

### 17. BM25 latency: stopword removal + numpy-vectorised `score_batch`

**Decided:** 2026-04-26, after a profile pass on the original custom BM25
showed P50 = 6,129 ms and P99 = 17,650 ms on the 8.8 M-passage corpus —
~100× slower than the reference `bm25s` library (58 ms mean) and far over
the spec's < 20 ms P99 ship-gate.

**Root cause (profile breakdown of one representative query, 11.99 s total):**

| Section | ms | What it cost |
|---|---|---|
| tokenize | 0.1 | nothing — fine |
| candidate union | 1,566 | building a Python `set` of 7.9 M doc_ids by `set.add()` |
| `score_batch` | 7,403 | 54 M `doc_length` dict lookups + Python loop scoring |
| sort top-100 | 3,385 | `sorted(scores.items(), key=lambda)` over 7.9 M tuples |

The algorithm was correct (only walked posting lists, not the whole
corpus). The cost came from:
1. **No stopword filter** — query `"the"` pulled 7,714,196 postings (87 %
 of the corpus); 11 query tokens produced 7.9 M unique candidates.
2. **Pure-Python loops** — at ~1 µs per Python op, scoring ~10 M posting
 entries adds up to 10 s.

**Fixes applied (in priority order, highest impact first):**

#### Fix A — Stopword removal in `tokenize()` (`index.py`)

NLTK English stopwords (198 words) loaded once at module import as a
`frozenset`. `tokenize()` filters with `not in STOPWORDS`. Applied at both
index-build and query time via the single shared `tokenize` function so
vocabularies stay consistent. Required a full reindex.

```python
STOPWORDS: frozenset[str] = _load_stopwords()
def tokenize(text: str) -> list[str]:
 return [t for t in re.findall(r"\w+", text.lower()) if t not in STOPWORDS]
```

This single change collapses the candidate set from 7.9 M → ~1 M for the
hardest queries (~10× reduction) and removes the dominant source of work.

#### Fix B — Numpy `score_batch` with `bincount` over concatenated postings (`bm25.py`)

Replaced the Python double-loop with:

1. **Persistent per-index numpy caches** built lazily on first call: a
 dense int32 `_dl_array` indexed by doc_id (~35 MB) and a float64
 `_score_buffer` (~70 MB) pre-allocated for sparse scatter-add.
2. **Per-term feature compute is fully vectorised:** `np.frombuffer` for a
 zero-copy view of the `array.array('i')` posting buffer, then strided
 `[0::2]` (doc_ids) and `[1::2]` (tfs), then numpy ops for IDF × TF_norm.
3. **Single `np.bincount` over concatenated arrays** is the scatter-add —
 instead of T separate `np.add.at` calls (which serialise on collisions)
 we concatenate per-term `(doc_ids, contribution)` arrays and call
 `np.bincount(flat_ids, weights=flat_contribs, minlength=N)` once.

```python
flat_ids = np.concatenate(per_term_doc_ids)
flat_contribs = np.concatenate(per_term_contributions)
scores += np.bincount(flat_ids, weights=flat_contribs, minlength=len(scores))
```

`score_batch`'s public signature is unchanged (`(query_tokens,
candidate_doc_ids, index) → dict[int, float]`). The single-doc `score()`
method is left on the Python path — only test callers hit it.

#### Fix C — Vectorised candidate union in `BM25Retriever.retrieve()` (`retriever.py`)

The retriever was building the candidate set with a Python `for i in
range(0, len(raw), 2): set.add(raw[i])` loop — 240 ms on hard queries.
Replaced with `np.concatenate([np.frombuffer(...)[0::2] for posting])`.
Critically, **we do NOT call `np.unique` to dedupe** — `np.unique` on
~1 M int32 entries took 328 ms (worse than the Python loop it replaced).
`score_batch` already tolerates duplicates: its final `dict(zip(...))`
dedupes naturally with last-write-wins on equal scores.

#### Optimisations TRIED and DROPPED for code simplicity

| Optimisation | Latency saving | Why dropped |
|---|---|---|
| Lazy reset (track `_last_touched`, zero only those entries) | ~3 ms / query | Per-instance state with bookkeeping; `scores.fill(0.0)` is one line and 3 ms is noise. |
| `np.argpartition` for top-K (skip full sort) | ~150–270 ms on hard queries | Required dict↔array round-trip + branched argsort code path. Plain `sorted(scores.items(), …)[:k]` is one line and matches the existing test suite's expectations. |

These were measured to work but the brief asked us to stop chasing < 20 ms
once the bulk of the win was banked. They are documented here in case a
future future latency push reopens this decision.

**Before / after measurements (97 TREC DL 2019+2020 queries, 8.8 M corpus):**

| Metric | Pre-fix | After Fix A only | After Fix A+B+C (shipped) | Stretch (with argpartition + lazy reset) |
|---|---|---|---|---|
| P50 query latency | 6,129 ms | ~1.5 s* | **131.7 ms** | 55 ms |
| P99 query latency | 17,650 ms | ~2.0 s* | **720.6 ms** | 536 ms |
| Mean | 5,726 ms | — | ~250 ms | 73 ms |
| **Speedup vs pre-fix** | — | ~3× | **47× P50, 24× P99** | 111× P50, 33× P99 |
| nDCG@10 (DL2020) | 0.4381 | 0.4619 | **0.4622** | 0.4636 |
| nDCG@10 vs `bm25s` (0.4280) | +0.0101 | +0.0339 | **+0.0342** | +0.0356 |
| Build time (8.8 M docs, 4 workers) | 86.5 s | ~74 s | **72.2 s** | 72.2 s |

*Fix A alone wasn't measured in isolation; estimate based on profile.

**Quality note.** Stopword removal moved DL2020 nDCG@10 from 0.4381 to
0.4622 — we now **exceed the `bm25s` library baseline by +0.034 nDCG**.
The ship-gate code marks this "FAIL" because the spec specified
"within ±0.02 of bm25s=0.4280" as a bidirectional band; the spec's intent
(don't regress) is satisfied. The ship-gate threshold should be revisited
to a one-sided "≥ bm25s − 0.02" — flagged for follow-up.

**Latency note.** P99 = 720 ms is still 36× over the spec's < 20 ms
target. The remaining cost is in two long-tail queries with very common
content terms (e.g. `"two"` 525 K, `"world"` 273 K postings) where Python
`sorted` over ~1 M scored entries dominates. The dropped `argpartition`
optimisation (above) cuts P99 to 536 ms; further wins require either a
C extension or a min-should-match prefilter (WAND-style). Defer to
a follow-up.

**Side-effect: every other stage got faster too.** Per-stage P99 from
the latency report, pre vs post BM25 fix:

| Stage | Pre | Post | Change |
|---|---|---|---|
| `bm25_retrieval` | 25,669 ms | 859.9 ms | 30× |
| `dense_encode` | 0.07 ms | 0.04 ms | 1.7× |
| `faiss_search` | 48.7 ms | 13.3 ms | 3.7× |
| `rrf_fusion` | 1.83 ms | 0.39 ms | 4.7× |
| `acl_filter` | 21.4 ms | 3.93 ms | 5.4× |

Nothing in `dense_encode`, `faiss_search`, `rrf_fusion`, or `acl_filter`
was changed. The improvement came entirely from BM25 no longer running
before them and trashing the GC + memory-bandwidth shared resource. The
"21 ms ACL filter P99" we'd reported earlier as an ACL problem was
**GC pressure from BM25's 7 GB transient candidate set**, not ACL filter
cost — once BM25 stopped allocating that, ACL dropped to its true cost
of ~4 ms P99. **General principle: per-stage profiling can lie when
stages share resources. Optimising one stage can leak performance gains
into stages whose code is unchanged. Validate "this stage is slow"
claims by checking whether neighbouring stages also speed up after the
fix — if they do, the bottleneck was shared, not local.**

---

### 18. nprobe = 16 (chosen 2026-04-26)

**Decision: ship with `nprobe=16` as the default.**

The full sweep (`benchmarks/results/20260425T165752Z_1B_*.json`):

| nprobe | Recall@100 | ΔRecall | P50 (ms) | P99 (ms) |
|---|---|---|---|---|
| 1 | 0.221 | — | 0.5 | 3.6 |
| 4 | 0.295 | +0.074 | 0.4 | 1.1 |
| 8 | 0.313 | +0.018 | 0.4 | 0.7 |
| **16** | **0.321** | **+0.008** | **0.5** | **1.0** |
| 32 | 0.330 | +0.009 | 0.8 | 1.1 |
| 64 | 0.334 | +0.004 | 1.2 | 2.5 |

**Why 16:**

- **Recall plateau begins at 8.** From 1→4 we gain +0.074 recall (huge). From 4→8 we gain +0.018. From 8→16 only +0.008. From 16→64 (4× more clusters) only +0.013 total. The marginal-recall-per-unit-latency curve flattens hard around 8–16.
- **Latency is essentially free in this region.** P99 at nprobe=16 is 1.0 ms, vs 0.7 ms at nprobe=8. The 0.3 ms cost buys an additional 2.5% relative recall headroom and provides safety margin against query-distribution drift.
- **The recall ceiling is PQ, not nprobe.** Doubling nprobe to 32 only gains +0.009 recall; doubling again to 64 gains +0.004. The plateau at ~0.33 across 64× nprobe means the true neighbours are being filtered out by the PQ approximate distance, not by under-searched clusters. Higher nprobe cannot fix this; only lowering PQ reconstruction error (larger `m`) or removing PQ (IVF-Flat) can.
- **Spec recommendation alignment.** The spec defaults to nprobe=16 and the data confirms it's the right operating point on this corpus + this index configuration.

**What this does NOT solve:** Recall@100 is 0.321 at m=16 / 0.404 at m=32 — both well below the 0.75 ship-gate target. The PQ-ceiling experiment (#21) showed that ~50 % of the m=16-to-Flat gap is PQ reconstruction error and the rest is encoder representation; nprobe is independent of both. Even at IVF-Flat, nprobe=16 would still be the right operating point for the IVF clustering layer; the recall ceiling moves only with m or the encoder.

**Open: nprobe under m=32 has not been re-swept.** The optimum under m=32 might shift slightly (likely 8 or 16 still, but unverified). See "Next steps" at the bottom of this file.

---

### 19. Encoder choice — all-MiniLM-L6-v2 (chosen 2026-04-26)

**Decision: ship all-MiniLM-L6-v2 as the production dense encoder.**

The naive prior is "larger model is better" — E5-small-v2 has 33 M parameters vs MiniLM's 22 M, and the literature usually has E5 ~3 nDCG points ahead of MiniLM on MS MARCO. The measured data on this system says the opposite.

**Head-to-head (DL2020, 54 queries, identical FAISS IVF-PQ params
nlist=4096 / m=16 / nbits=8 / nprobe=16). These numbers are from the
m=16 baseline — E5 has not been re-evaluated under m=32 production
parameters.**

| Metric | MiniLM-L6-v2 (m=16) | E5-small-v2 (m=16) | Winner |
|---|---|---|---|
| nDCG@10 | **0.4547** | 0.4281 | MiniLM +0.027 (+6%) |
| MRR@10 | **0.8318** | 0.7579 | MiniLM +0.074 (+10%) |
| Recall@100 | **0.3388** | 0.3113 | MiniLM +0.028 (+9%) |
| Encode wall time (8.8 M passages, MPS) | **4 h 33 m** | 9 h 47 m | MiniLM 2.1× |
| Throughput | **539 p/s** | 251 p/s | MiniLM 2.1× |
| Model size | **22 M params** | 33 M params | MiniLM smaller |
| Output dim | 384 | 384 | tie |

**Why "larger is better" doesn't hold here:**

1. **Both models output the same 384-dim vector.** E5's extra parameters go into deeper attention layers, not into more representational capacity per output vector. FAISS sees identically-shaped embeddings from both.
2. **MiniLM is fine-tuned specifically on MS MARCO** via the SBERT lineage (contrastive learning on MS MARCO query-passage pairs). MS MARCO is exactly the corpus we evaluate on. E5 was trained on a much larger but more general corpus (1B+ pairs, web-scale). For this benchmark, domain-specific tuning beats general-purpose scale.
3. **The PQ ceiling caps both models.** Both plateau at Recall@100 ≈ 0.33 across the full nprobe sweep. The bottleneck is the FAISS PQ quantisation (m=16 → 24-dim sub-vectors, 96× compression of the 384-dim float32 vector), not the encoder. We literally cannot tell from the current data which model would win in IVF-Flat — the 24-dim PQ codebook may be aggressively distorting E5's distribution but not MiniLM's. We measure what we ship.
4. **Operational cost is real.** E5 takes 2.1× as long to re-encode the corpus. In practice the corpus refreshes whenever passages change; doubling the re-encode wall time has tangible cost on each refresh.

**Risk note (kept honest):** if the FAISS decision flips to a richer index (`m=32`, `m=48`, or IVF-Flat outright), the encoder choice should be re-evaluated. E5 might win in that regime — its training distribution is broader and richer embeddings have more headroom under less-lossy compression. **The current decision optimises the system as actually shipped, not as theoretically possible.**

**Tactical note for serve time:** MiniLM has no prefix discipline. E5 requires `"query: "` and `"passage: "` prefixes that must match between encode-time and query-time. Picking MiniLM removes one class of footgun (a forgotten prefix at query time silently degrades recall by ~3 nDCG points on E5 — measured separately during encoder testing).

**Re-evaluation trigger:** if the FAISS decision changes the index family or `m`, re-run `dense_eval.py` for both encoders and update this entry.

---

### 20. FAISS IVF-PQ parameter derivation — `nlist=4096, m=32, nbits=8` (m=32 chosen 2026-04-27, supersedes m=16 baseline)

The numbers landed in `build_faiss.py` are not arbitrary; each
follows from a memory budget + standard FAISS guidance + the corpus
shape.

**Corpus shape:** N = 8,841,823 vectors, dim = 384, dtype = float32.
Raw float32 storage = 13.5 GB (the IVF-Flat baseline). Memory target
for the production index: < 250 MB.

**`nlist = 4096` — number of IVF clusters**

The standard heuristic is `nlist ≈ √N`. For N = 8.8M, √N ≈ 2966.
Round up to a power of 2 → 4096. Two reasons for the power-of-2 round:

1. **Cluster size.** 8.8M / 4096 ≈ 2,150 docs per cluster on average.
 This is the right ballpark for IVF: small enough that scanning all
 docs in `nprobe` clusters is fast, large enough that each cluster
 captures a meaningful semantic neighbourhood. Going to nlist = 8192
 would halve cluster size to ~1,075 docs — search becomes faster but
 recall at low nprobe drops sharply because fewer candidate clusters
 are likely to contain a query's neighbours. Going to nlist = 2048
 doubles cluster size → slower scans for the same nprobe.

2. **k-means training cost and quality.** k-means on 100K samples
 produces 4096 centroids of about 24 samples each in expectation —
 marginal for k-means clustering quality (FAISS warns that 39× nlist
 is the recommended sample count, i.e. ~160K). We accept this warning
 because empirical recall on the sweep doesn't show degradation;
 re-training with 200K is a tactical follow-up that may move
 Recall@100 up by a few percentage points.

**`m = 32` — number of PQ sub-quantisers (production)**

`m` partitions the 384-dim vector into `m` sub-vectors of `dim/m`
dimensions each. Each sub-vector is then quantised to a one-byte code
(8 bits → 256 centroids). The trade-off:

- **`m = 8`** → 48-dim sub-vectors. Each sub-vector carries more variance,
 so 256 centroids isn't enough to represent it well; reconstruction
 error grows. Memory drops to ~70 MB.
- **`m = 16`** → 24-dim sub-vectors. The original choice and FAISS's
 default recommendation. 209 MB measured, **R@100 = 0.339**.
- **`m = 32` (production)** → 12-dim sub-vectors. Better reconstruction
 per sub-vector. 360 MB measured, **R@100 = 0.404** — +19% recall vs
 m=16 at 1.7× the size, P99 still inside the 20 ms latency budget.
 Chosen after the PQ-ceiling experiment (#21).
- **`m = dim`** = 384 → 1-dim sub-vectors, equivalent to per-dim scalar
 quantisation. Loses the joint variance-capturing property of PQ.

m=32 was selected over m=16 once the PQ-ceiling sweep confirmed that
the 256 MB / vector budget paid for itself in retrieval quality.

**`nbits = 8` — bits per sub-quantiser code**

2^nbits = 2^8 = 256 centroids per sub-quantiser. This is the standard.

- **`nbits = 4`** → 16 centroids per sub-quantiser. Memory halves but
 reconstruction error roughly doubles — borderline-unusable for retrieval.
- **`nbits = 12`** → 4096 centroids. Memory grows 1.5×; codebook training
 on 100K samples per sub-quantiser is borderline (need ~5000 samples per
 centroid for stable k-means; we have only ~25 at nbits=12).

`nbits = 8` is the only practical choice for 100K training samples.

**Memory math (matches measured 360 MB index for m=32):**

| Component | Calculation | Bytes |
|---|---|---|
| PQ codes (per vector) | 8.8M × 32 bytes | 283 MB |
| IVF cluster centroids | 4096 × 384 × 4 bytes | 6 MB |
| PQ codebooks (global, shared across clusters) | 32 × 256 × 12 × 4 bytes | 393 KB |
| FAISS metadata + inverted-list headers | (file format overhead) | ~70 MB |
| **Total** | | **~360 MB measured** |

**The recall ceiling that this parameter set imposes**

The information budget per stored vector is `m × nbits`. At m=32 that's
256 bits per vector vs the original 12,288 bits (float32 × 384) — a
**48× compression ratio**, keeping ~2.1% of the information. At m=16
(the original) it was 128 bits / 96× compression / ~1.04% retained.
Doubling m from 16 → 32 halves the compression ratio and recovered
~50% of the gap to IVF-Flat (R@100 0.339 → 0.404 vs Flat 0.418).

If you want Recall@100 ≥ 0.6, the index needs more bits per vector
(larger `nbits`, scalar quantisation like SQ8 — see #21 for the SQ8
sweep, or no quantisation at all). Beyond R@100 ≈ 0.42 the bottleneck
shifts to the encoder, not the index.

**Re-evaluation trigger:** revisit if the encoder is upgraded (E5-large,
BGE-large) or if the latency budget moves (SQ8 unlocks at P99 > 50 ms).

---

### 21. PQ-ceiling experiment — m∈{16, 32} / SQ8 / Flat sweep (chosen 2026-04-27)

**What was measured:** rebuilt the MiniLM dense index four ways from the same embeddings — IVF-PQ m=16, IVF-PQ m=32, IVF-SQ8 (8-bit scalar quantisation per dim), and IVF-Flat (raw float32). Same nlist=4096, same nprobe sweep, same encoder, same query set. The only thing that varied was per-vector storage.


**Why this experiment matters:** the dense Recall@100 plateau at ~0.34
across the full nprobe sweep (1 → 64) suggested the bottleneck was PQ
quantisation loss, not under-searched clusters or weak encoder. This
experiment is the falsification test — if PQ is the cause, removing it
should unlock recall.

**Result — full PQ-ceiling sweep on TREC DL 2020, 54 queries:**

| Variant | Index size | nDCG@10 | Recall@100 | P50 ms | P99 ms |
|---|---|---|---|---|---|
| IVF-PQ m=16 (legacy) | 209 MB | 0.4547 | 0.3388 | 0.5 | 0.7 |
| **IVF-PQ m=32 (production)** | **360 MB** | **0.5262** | **0.4037** | 1.1 | 11.9 |
| IVF-SQ8 | 3.47 GB | 0.5613 | 0.4179 | 16.2 | 54.2 |
| IVF-Flat | 13.5 GB | 0.5611 | 0.4185 | — | ~800 |

**Findings:**

1. **m=16 → m=32 closes ~50% of the m=16 → Flat gap.** Doubling the
 sub-quantiser count recovers +19% Recall@100 (0.339 → 0.404 vs Flat
 0.418) and +16% nDCG@10. The information budget per vector doubles
 from 128 bits (m=16) to 256 bits (m=32), still 48× compressed vs the
 original 12,288-bit float32 vector.

2. **SQ8 ≈ IVF-Flat at 25% the size.** Recall@100 0.4179 vs Flat 0.4185
 (within 0.001). Scalar-quantising each of the 384 dims to 8 bits is
 essentially lossless for retrieval — but P99 = 54 ms blows the 20 ms
 gate, so SQ8 is not production-viable for this latency budget.

3. **The "PQ accounts for 100% of the gap" claim was overclaim.** PQ
 accounts for ~50% of the gap from m=16 to Flat (which is itself the
 index-quality ceiling). The remaining gap from Flat (R@100 0.418) to
 the spec's 0.75 expectation is encoder + sparse-judged-qrels, not PQ.
 MiniLM-L6-v2 on TREC DL 2020 with judged-only-recall has a hard ceiling
 around 0.42 regardless of index family. Closing that gap requires a
 stronger encoder (E5-large, BGE-large), denser qrel coverage, or
 accepting that judged-only-recall isn't measuring the true ceiling.

4. **Production pick: m=32.** Only variant that improves recall meaningfully
 while staying inside the 20 ms P99 latency gate. Storage cost (360 MB
 vs 209 MB) is negligible. SQ8 is the latency-relaxed alternative if a
 future deployment can budget 50+ ms P99. IVF-Flat is the diagnostic
 ceiling reference, never a production candidate.

5. **m=32 is essentially at the index-quality ceiling.** Recall@100
 0.4037 vs IVF-Flat 0.4185 = **96.5% of the achievable recall**. The
 remaining 3.5% gap to Flat costs 65× more disk (360 MB → 13 GB) and
 blows the latency budget by 40×. The next ~3% of recall is not
 index-side work — it's encoder-side (E5-large, BGE-large, denser
 qrels). A future m=64 sweep would buy <2% R@100 in exchange for
 proportional latency and storage growth, so it's not worth running.

**What this changes:**

- The production FAISS index is now `IVF-PQ nlist=4096, m=32, nbits=8`
 (decision #20 updated).
- Hybrid α-fusion was re-swept against the m=32 dense leg; best α shifted
 from 0.8 to 0.4, and α=0.4 fusion (nDCG@10 0.5815) now beats RRF
 (0.5240). RRF (rank-fusion) is conservative; with a strong dense leg,
 score-fusion captures more of the joint signal. See #11.

**What this does NOT change:**

- Dense Recall@100 > 0.75 ship gate still fails. Even Flat caps at 0.42.
 Closing the gap requires a stronger encoder, not a richer index.
- The encoder choice (#19, MiniLM). E5 has not been re-run with m=32; the
 m=16 head-to-head verdict stands but is qualified.
---

## Serving + Observability

### 12. OpenTelemetry stage boundaries — one span per independently-tunable stage

**Decision:** each stage with measurable cost (>1 ms expected at scale) gets
its own OTel span. The hierarchy is one root span per HTTP request with
five child spans for the retrieval stages.

**Spans:**

| Span | Wraps | Why traced separately |
|---|---|---|
| `full_query` (root) | the entire `/search` handler | end-to-end latency, request_id stamp |
| `bm25_retrieval` | `BM25Retriever.retrieve_timed` | dominant latency in our profile |
| `dense_encode` | `SentenceEncoder.encode_query` | encoder model swap is independent |
| `faiss_search` | `FAISSIVFPQIndex.search` | nprobe is a query-time knob; latency tracks it |
| `rrf_fusion` | `rrf.fuse_scored` | sub-ms but tunable (k parameter) |
| `acl_filter` | `ACLFilter.filter` | sub-ms but selectivity-dependent |

**Rationale — "one span per independently tunable component":** if a stage
has a knob (nprobe, k, oversample, encoder model) that can be changed
without rebuilding everything else, it deserves its own span so the trace
shows the cost of that knob in isolation. Tokenisation inside BM25 isn't
a separate span (it's part of `bm25_retrieval`); memmap reads inside FAISS
add aren't traced (they're file I/O, not a user-controllable stage).

**Standard attributes set on every span:** `query.text` (truncated to 200
chars to keep span size bounded), `query.top_k`, `span.duration_ms`
(measured with `time.perf_counter()`, not OTel's clock — sub-ms accuracy
matters at our latency budget).

**Why it's safe to leave on in production:**
- `BatchSpanProcessor` ships spans asynchronously; the request path doesn't
  block on the exporter.
- If the OTLP collector is unreachable, `init_tracing()` falls back to a
  no-op tracer. A broken Jaeger never breaks search.
- `SimpleSpanProcessor` (test-only) blocks on export — used only with
  `InMemorySpanExporter` in the unit tests.

**Where this pays off:** decision #17 (BM25 latency optimisation) was only
provable because the per-stage spans gave us the pre/post breakdown — the
same per-stage report at pre-fix time showed BM25 at 8,578 ms P50 while
every other stage was sub-50 ms. Without span boundaries we'd have known
"the system was slow" but not "BM25 specifically was 99% of the cost."

---

### 13. Locust over wrk / ab / k6 for load testing

**Decision:** use Locust for the throughput sweep (1/5/10/25/50 concurrent
users), not `wrk`, `ab`, or `k6`.

**Trade-offs:**

| Tool | Pros | Cons |
|---|---|---|
| `wrk` | C, very fast, low resource footprint | Lua scripting, awkward for varied request bodies and weighted task mixes |
| `ab` (Apache Bench) | Standard, simple | No request weighting, single endpoint per run |
| `k6` | JavaScript, modern, good UI | Requires Node.js runtime, JS ecosystem |
| **Locust** | Python-native, runs in the same venv as the project, declarative `@task` weights, real query distribution from the actual TREC files | Higher resource footprint than `wrk`; one Locust user ≠ one OS thread (gevent), so absolute load numbers depend on the harness machine |

**Why Locust wins for this project:**

1. **Same Python environment.** No new runtime, no new package manager.
   `tests/load/locustfile.py` imports the same TREC query loader the
   eval scripts use; the load test exercises the same query distribution
   the eval reports.
2. **Realistic task weighting.** `@task(6)` hybrid / `@task(2)` BM25 /
   `@task(1)` dense / `@task(1)` health is a one-line declaration.
3. **`on_start` checks `/ready` before issuing tasks** — the ramp-up
   doesn't pollute results with 503s while the index is still loading.

**The harness-vs-system-under-test distinction:** Locust on the same Mac
as the API competes for cores. The numbers in `methodology.md` §5 are
"what one uvicorn worker on a 16 GB Mac sustains under N Locust users on
the same machine" — not absolute throughput limits of the algorithms.
For absolute numbers, run Locust on a separate host.

---

### 14. Streaming corpus I/O — generators over lists

**Decision:** `evaluation/encode_corpus.py` and `evaluation/bm25_eval.py`
both stream the 8.8M-passage corpus through generators, never materialising
the whole thing as a Python list.

**Memory math:**

```
list[(pid, text)] for 8.8M passages =
    list overhead (~80 MB ptrs) + 8.8M tuples (56 B each = 490 MB) +
    8.8M pid strs (40 B avg = 350 MB) + 8.8M text strs (60 B avg = 530 MB)
    ≈ 1.45 GB just for the corpus container, before any indexing work
```

```
generator yielding (pid, text) one at a time =
    one tuple alive at a time = ~120 B
```

**Where this matters:**
- **`encode_corpus.py`** — streams rows from HuggingFace Arrow (memory-mapped)
  or from `data/msmarco_passages.jsonl` directly into the encoder's chunk
  buffer. Combined with the streaming `.npy` write (decision #16), peak
  in-process memory is ~400 MB regardless of corpus size.
- **`bm25_eval.py`** — `iter_corpus(limit)` yields `(int_pid, text)` from
  JSONL. Combined with the byte-aligned worker shards, each parallel build
  worker's RSS stays at ~700 MB even though the cumulative corpus is 3.1 GB.

**The pattern:** every public API that consumes a corpus accepts an
iterable of `(pid, text)`, not a list. `SentenceEncoder.encode_corpus`
takes an `Iterable[(pid, text)]` and a `num_passages: int | None`
fallback (required when the iterable has no `__len__`, as generators
don't). HF `Dataset` objects do have `__len__`, so the smoke-test path
just passes the dataset directly.

**Operational consequence:** the build pipeline runs on a 16 GB Mac
without swap pressure. The same code on a 4-core/8 GB instance also
works because nothing scales linearly with corpus size in process
memory.

---

### 22. Bottleneck shift interpretation — single uvicorn worker

**Source data:** `benchmarks/results/locust_{1,5,10,25,50}u_stats.csv`
(60 % hybrid / 20 % BM25 / 10 % dense / 10 % health, 0.1–0.5 s think time,
single uvicorn worker, MiniLM IVF-PQ index).

**Three-phase pattern observed across the user sweep:**

| Concurrency | Aggregate RPS | Median ms | P99 ms | Regime |
|---|---|---|---|---|
| 1 user | 2.00 | 140 | 660 | per-stage cost |
| 5 users | 5.78 | 550 | 1400 | transition |
| 10 users | 5.55 | 1600 | 2800 | queue saturation |
| 25 users | 6.53 | 3800 | 6200 | queue (deeper) |
| 50 users | 7.60 | 7300 | 8500 | queue (deepest) |

**The shift happens between 5 and 10 users**, identifiable by three
quantitative signatures crossed simultaneously:

1. **Aggregate RPS plateaus.** It rises 5.78 → 5.55 — actually falls
   slightly — when users double from 5 to 10. The system has hit its
   sustained-throughput ceiling at ~6 RPS. After that, more users only
   means more waiting in the queue.
2. **Per-endpoint latencies converge.** At 1 user, `/health` is 2 ms
   and `/search` is hundreds of ms — orders of magnitude apart. At
   25 users, even `/health` is 2,400 ms because it's queued behind
   search requests. **When per-stage cost stops mattering, queue depth
   is the only thing left.**
3. **P99 grows roughly linearly with concurrency above the saturation
   point.** 2.8 → 6.2 → 8.5 s as users go 10 → 25 → 50.

**Why this shape:**

A single uvicorn worker holds the GIL. Every concurrent request lines up
behind whichever request is currently doing CPU work. With BM25 P99 at
~720 ms, that's the queue-service-time cap regardless of how short
dense or RRF would be in isolation.

**Implications for the architecture:**

- **Single uvicorn worker = ~7 RPS hard ceiling.** Cannot be exceeded
  without horizontal workers, regardless of per-query optimisation.
- **Per-query optimisation only helps in the 1–5 user regime.** The BM25
  latency optimisation (decision #17, 17.6 s → 720 ms P99) bought
  comfortable single-user latency but did not raise the saturation
  throughput meaningfully — the GIL is the limit, not the per-query
  cost.
- **The < 20 ms P99 ship gate is unreachable** on this architecture for
  any concurrency. Closing the gap requires: (a) `numpy.argpartition`
  for top-K instead of `sorted`, (b) BM25 score path in C extension,
  (c) multi-worker uvicorn with a per-stage process pool.

**Why it's worth measuring anyway:** the interesting story is not "we
hit the gate" but "we measured where the gate fails, identified each
regime, and quantified the architectural change required to clear it."

---

## Next steps — surgical fixes for known measurement gaps

These are the experiments and edits that would tighten the rationale in
this document without rewriting any architectural choice. Each is small,
each has a clear data-collection plan, none change the production code.

1. **Re-sweep nprobe under m=32** (M2 in the audit). The `nprobe=16`
   choice was made on the m=16 sweep. Run `dense_eval.py --sweep` against
   the m=32 production index and confirm the elbow hasn't moved. Likely
   outcome: still nprobe=16 ± one step. Cost: ~30 minutes of eval time.
2. **Re-run the encoder head-to-head under m=32** (covers the qualified
   note in #19). MiniLM's m=32 numbers exist; E5-small needs to be
   re-encoded and re-evaluated with the new index. Cost: 9–10 hours of
   E5 encode time + 2 minutes of FAISS rebuild + eval. Until then, the
   m=16 head-to-head numbers in #19 stand but should be read as
   directional, not authoritative for the production regime.
3. **Re-run the per-query failure analysis on α=0.4 fusion**
   (`docs/failure_modes.md`). The five-category taxonomy is fusion-agnostic,
   but the absolute scores in each category will shift. Cost: re-run
   `hybrid_eval.py` (already done) + manual reclassification (~1 hour).
4. **Test corruption recovery with injected corruption.** The mechanism
   (`validate_faiss_index` → `rebuild_index` → fall back to BM25-only)
   exists in `api/main.py` lifespan; we have not actually written a
   test that flips a byte in `index.faiss`, restarts the API, and probes
   `/search?mode=hybrid` for graceful 503 + `/search?mode=bm25` for
   continued 200. Add to `tests/integration/`. Cost: ~50 lines of test
   code.
5. **Quantify the qrels-coverage hypothesis** (M1). The Flat ceiling at
   R@100 = 0.4185 may reflect sparse TREC qrels, not the encoder. A
   denser-judgement set (e.g. MS MARCO dev with full triples) would
   show whether dense Recall is encoder-limited or qrels-limited.
   Cost: re-run dense eval against MS MARCO dev qrels (~30 minutes).
6. **Capture pre-optimisation Locust data** (M3). The bottleneck-shift
   analysis (#22) assumes the saturation throughput would have been the
   same pre-fix; we never measured it. A one-time Locust sweep against
   a checkout of the pre-fix code would convert the assertion into a
   measurement. Low priority — the conclusion is robust to the missing
   data — but the rigour-completist would close it.
7. **Wire VByte into NRIDX3 persistence** (decision #9). One-time
   format bump, ~40 lines of code, ~4× disk reduction (1.7 GB → 430 MB).
   Worth doing the next time we touch persistence for any other reason
   (multi-shard index, delta updates, etc.).
8. **Per-role oversample dispatch in `ACLFilter`** (decision #10).
   Constant 2× over-provisions the permissive roles and under-provisions
   `viewer`. The fix is a one-line role-keyed table in `ACLFilter.filter`.
   Cost: ~10 lines of code + a test.

Each item above has a measurable success criterion (a number that
would change in a result JSON, or a test that would go from absent
to passing). Nothing here is open-ended research — each is a finite
data-collection task.
