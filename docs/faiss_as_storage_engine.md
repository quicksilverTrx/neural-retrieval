# FAISS as a Storage Engine — A DDIA Lens

A Designing Data-Intensive Applications (Kleppmann) analysis of FAISS IVF-PQ:
write path, read path, and nprobe as read amplification.

---

## The analogy

DDIA frames storage engines around two primitives: **append-only writes** and
**index reads**. An LSM-tree trades write amplification for read amplification;
a B-tree trades write amplification for space amplification. FAISS IVF-PQ
follows the same architectural logic, applied to approximate nearest-neighbour
search.

---

## Write path: training → adding → flushing

FAISS IVF-PQ is a **two-phase build**, not an append-friendly structure:

**Step 1 — Training:**
```
train_sample (100K random vectors)
    → k-means cluster to nlist=4096 centroids
    → for each sub-quantizer m: k-means to 256 PQ centroids
    → result: quantizer + PQ codebook (fixed after training)
```

Training costs ~30s for 100K vectors and produces ~6MB of centroid data. It
is expensive and cannot be done incrementally — new centroids would invalidate
existing PQ codes. This is the FAISS equivalent of schema migration: it requires
a full rebuild.

**Step 2 — Adding:**
```
for each vector v:
    assign to nearest IVF cluster (coarse quantizer): 1 FLOP per centroid = O(nlist × dim)
    encode v as PQ codes: residual from centroid → quantize each sub-vector
    append (cluster_id → pq_codes) to the inverted list for that cluster
```

Adding is incremental and O(nlist × dim) per vector. At 8.8M vectors, adding
takes ~30 minutes on CPU (dominated by PQ encoding, not IVF assignment).

**Write amplification:** each vector is stored once as 16 bytes of PQ codes.
Compare to 384 × 4 = 1536 bytes for the original float32 vector — 96× compression.
The memmap embeddings.npy (13.5GB) is kept separately as the recovery artifact;
only the PQ codes (~141MB) live in the FAISS index.

---

## Read path: nprobe as read amplification

A query against an IVF-PQ index executes:

```
1. Encode query: same coarse quantization + PQ encoding as write path
2. Identify top-nprobe clusters (by distance to their centroids)
3. For each of the nprobe clusters:
       scan all entries in that cluster's inverted list
       compute approximate distance using PQ asymmetric distance tables
4. Maintain a top-K heap across all scanned entries
5. Return top-K (cluster_id, pq_code) pairs → decode to passage IDs
```

**nprobe = read amplification factor:**

| nprobe | Clusters searched | Passages scanned (avg 2150/cluster) | Predicted Recall@100 | **Measured (MiniLM, 2026-04-25)** | Measured P99 latency |
|---|---|---|---|---|---|
| 1 | 1/4096 = 0.02% | ~2,150 | ~0.30 | **0.221** | 3.6 ms |
| 4 | 0.1% | ~8,600 | ~0.55 | **0.295** | 1.1 ms |
| 8 | 0.2% | ~17,200 | ~0.70 | **0.313** | 0.7 ms |
| **16** | **0.4%** | **~34,400** | **~0.82** | **0.321** 🚩 | 1.0 ms |
| 32 | 0.8% | ~68,800 | ~0.88 | **0.330** | 1.1 ms |
| 64 | 1.6% | ~137,600 | ~0.91 | **0.334** | 2.5 ms |

The recall plateau at ~0.33 across 64× nprobe (0.5% → 1.6% of clusters
searched) means **the predicted curve was wrong**: it assumed PQ
reconstruction error was small enough that scanning more clusters would
keep producing new relevant docs. In reality, the PQ encoding (m=16,
24-dim sub-vectors) is lossy enough that ~67% of true neighbours are
filtered out by the approximate distance computation regardless of how
many clusters we search. Fix is on the index side (larger `m` or no PQ),
not the query side. See `status.md` Known Issues.

The **recall-latency curve** is the key FAISS operating-point tradeoff. At
nprobe=16, we search 0.4% of the index — that's the same read amplification
as reading 0.4% of a database table on every query. The DDIA analogy: nprobe
is the FAISS equivalent of a B-tree's level count.

---

## Persistence model: write-once, read-many

FAISS has no WAL, no transaction log, no partial-write recovery. It is write-once:

```python
faiss.write_index(index, "index.faiss")   # atomic from Python's perspective
```

The file is written by serializing the index object. A crash mid-write produces
a corrupt file. This is why `record_index_checksum()` (in retrieval/dense/recovery.py)
computes the sha256 of the written file and stores it in `meta.json`. On every
load, `validate_faiss_index()` recomputes the sha256 and compares — a mismatch
triggers the fallback to BM25-only mode.

**Recovery protocol:**
1. `validate_faiss_index(path)` → False
2. Log warning: "FAISS index corrupted — falling back to BM25-only"
3. Serve BM25-only (latency increases, recall decreases, but no outage)
4. Background job: `rebuild_index(emb_dir, faiss_dir)` — reconstructs from
   the raw embeddings.npy (which is separately checksummed)
5. Hot-reload: swap the validated index into the serving path without restart

---

## Space amplification

| Artifact | Size | Ratio to original |
|---|---|---|
| Raw embeddings (float32, 8.8M × 384-dim) | 13.5 GB | 1.0× |
| PQ codes in FAISS index | ~141 MB | 0.010× |
| IVF centroids (4096 × 384-dim × float32) | ~6 MB | 0.0004× |
| Passage lookup (SQLite) | ~1–2 GB | ~0.1× |

The FAISS index is a lossy compression of the embedding space. The precision
lost (PQ reconstruction error → lower Recall@100 vs flat exact search) is the
price for the 96× compression ratio.

---

## The fundamental tradeoff stated as DDIA-style invariants

1. **Recall at nprobe k** is monotonically non-decreasing in k.
2. **Latency at nprobe k** is monotonically non-decreasing in k.
3. **Recall at nprobe=all** = Recall of exact flat search ≈ 1.0 (modulo PQ error).
4. **Index build is not incremental**: new vectors can be added but the quantizer
   (cluster centroids + PQ codebook) is fixed at training time. Quality degrades
   if the data distribution shifts significantly from the training sample.
5. **nprobe is a query-time parameter**: changing it requires no reindex.
   It is a pure operating-point slider between latency and recall.

The choice of nprobe=16 as the default (item #5 on the nprobe sweep) is an
operating-point decision: see the measured curve in `dense_eval.py --sweep`
output and pick the latency/recall trade that matches deployment SLOs.
