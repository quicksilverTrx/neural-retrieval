# Retrieval Failure Mode Taxonomy

Per-query failure analysis for the **TREC DL 2020** 54-query benchmark.
Classifications and patterns derived from `benchmarks/results/20260425T172427Z_1C_*.json`
(RRF hybrid, MiniLM dense leg + custom BM25, post-Bug-1-fix).

**System under analysis:** RRF hybrid (k=60), nDCG@10 mean = 0.525.

---

## Failure taxonomy (5 categories)

The 54 queries cluster into five distinct types based on what kind of
retrieval signal would be needed to answer them well. Mean nDCG@10 by
category, sorted from worst to best:

| Category | n | Mean nDCG@10 | Median nDCG@10 |
|---|---|---|---|
| **Abbreviation / acronym lookup** | 3 | **0.187** 🚩 | 0.208 |
| Compositional / multi-constraint | 16 | 0.446 | 0.450 |
| Entity-fact (named entity + attribute) | 16 | 0.577 | 0.589 |
| Process / how-it-works | 9 | 0.595 | 0.664 |
| **Definition / vocabulary lookup** | 10 | **0.608** ✅ | 0.692 |
| **All 54 queries** | **54** | **0.525** | **0.522** |

The taxonomy explains the variance: definition queries are easy (clear
keyword anchor + corpus full of definitional passages); abbreviation
queries are nearly broken (3-letter token = no signal for either system).

---

### 1. Abbreviation / acronym lookup

**Definition:** Query contains a short, ambiguous abbreviation that needs
expansion to match the corpus vocabulary. BM25 fails because the corpus
text uses the expanded form; dense fails because rare-token embeddings
aren't well-trained.

**Expected pattern:** Both BM25 and dense underperform; RRF can't recover.

**Examples (all from DL2020):**

| qid | Query | nDCG@10 |
|---|---|---|
| 673670 | what is a alm | **0.000** |
| 118440 | define bmt medical | 0.208 |
| 1110678 | what is the un fao | 0.354 |

**Why these fail:**
- "alm" is too short — could mean Application Lifecycle Management, an actor's surname, or many other things. The corpus has no canonical passage answering "define alm" because the query is ambiguous.
- "bmt" expands to Bone Marrow Transplant. Passages about bone marrow transplants don't contain the literal token "bmt"; BM25 misses entirely. Dense retrieval may pick up "transplant" from the qualifier "medical" but not enough to land relevant passages in top-10.
- "un fao" needs to be expanded to "United Nations Food and Agriculture Organization." Passages routinely mention "FAO" but rarely both "UN" and "FAO" together as the query expects.

**Mitigation paths:**
- Acronym expansion at query-rewrite time (synonym dict).
- Cross-encoder reranking — a BERT-style reranker can resolve "bmt medical" → "bone marrow transplant" via context.
- Future: query expansion via LLM ("what is a alm" → "alm could mean Application Lifecycle Management").

---

### 2. Compositional / multi-constraint

**Definition:** Query has two or more constraints that must all be satisfied
simultaneously (location + topic, comparison between two things, "best X
for Y", etc.). Passages tend to satisfy each constraint individually but
not the conjunction.

**Expected pattern:** BM25 and dense both find candidate passages; the
conjunction-filtering happens at the rank-aggregation step, but neither
system natively understands "AND" semantics.

**Examples:**

| qid | Query | nDCG@10 |
|---|---|---|
| 67316 | can fever cause miscarriage early pregnancy | 0.089 |
| 1116380 | what is a nonconformity? earth science | 0.108 |
| 555530 | what are best foods to lower cholesterol | 0.152 |
| 1136047 | difference between a company's strategy and business model is | 0.426 |
| 1121353 | what can you do about discrimination in the workplace in oklahoma city | 0.543 |
| 1136043 | difference between a hotel and motel | 0.538 |
| 42255 | average salary for dental hygienist in nebraska | 0.517 |

**Why these fail:**
- "discrimination in the workplace **in oklahoma city**" — the location filter is the rare constraint. Most passages discuss workplace discrimination generally; few specifically mention Oklahoma City. RRF brings the two legs together, but neither retrieved passage addresses the conjunction.
- "difference between hotel and motel" — passages explain *what a hotel is* or *what a motel is* but rarely contrast them. The retrieval target is a comparison passage that may not exist in MS MARCO at all.
- "best foods to lower cholesterol" — "best foods" is a recommendation framing. Most relevant passages enumerate foods individually; ranking depends on which passage happens to use the word "best" or to enumerate them in a list format.

**Mitigation paths:**
- Query decomposition — split compositional queries into sub-queries, retrieve per sub-query, intersect the result sets.
- LLM-based query rewriting — turn "discrimination in workplace in oklahoma city" into ["workplace discrimination", "oklahoma city laws", "employment law oklahoma"] and fuse.
- Two-stage retrieval with a cross-encoder reranker that can attend to the conjunction at inference time.

---

### 3. Entity-fact (named entity + attribute)

**Definition:** Query asks a specific factual attribute about a named entity
(person, organisation, work, place, chemical, etc.). The mean is moderate
(0.577) but the variance is enormous: clear well-known-entity queries score
near-perfect, rare or ambiguous-entity queries collapse to zero.

**Expected pattern:** BM25 + dense both win when the entity has many
passages in the corpus. They both fail when the entity is rare or the
attribute requires structured knowledge (a date, an age, a count) that
passages don't always state explicitly.

**Examples:**

| qid | Query | nDCG@10 |
|---|---|---|
| 336901 | how old is vanessa redgrave | **0.000** |
| 938400 | when did family feud come out? | 0.183 |
| 940547 | when did rock n roll begin? | 0.201 |
| 1071750 | why is pete rose banned from hall of fame | 0.300 |
| 1043135 | who killed nicholas ii of russia | 0.374 |
| 47210 | average wedding dress alteration cost | 0.878 |
| 1113256 | what is reba mcentire's net worth | **1.000** |
| 1131069 | how many sons robert kraft has | **1.000** |
| 583468 | what carvedilol used for | **1.000** |

**Why the variance:**

The high scorers all share two properties: the entity is well-represented
in the corpus AND the attribute has a canonical passage that states it
plainly ("Reba McEntire's net worth is $X").

The low scorers have one of two issues:
- **Temporal attributes** ("how old", "when did X come out", "when did Y begin") — the answer is a date or duration that requires either structured data or a passage that explicitly states the date next to the entity name. Many MS MARCO passages discuss the entity but in a way that doesn't pair entity + date directly.
- **Causal/explanatory attributes** ("why X was banned", "who killed Y") — the answer requires a passage that links cause and effect. BM25 retrieves passages mentioning the entity; dense retrieves topically-similar passages; neither prefers passages that contain the explanation.

**Mitigation paths:**
- Knowledge-base augmentation (Wikidata) for high-value attributes (dates, counts, ages).
- LLM-based answer generation from retrieved passages — even an imperfect retrieval set may be enough for an LLM to compose the answer.
- Multi-vector representations (ColBERT-style) that retain token-level information for matching attribute-value pairs.

---

### 4. Process / how-it-works

**Definition:** Query asks for a procedure, mechanism, or step-by-step
explanation. Answers are usually multi-sentence paragraphs.

**Expected pattern:** Both systems retrieve relevant material; ranking
depends on which paragraph happens to contain the most query terms or the
most semantically similar phrasing.

**Examples:**

| qid | Query | nDCG@10 |
|---|---|---|
| 1133579 | how does granulation tissue start | 0.215 |
| 332593 | how often to button quail lay eggs | 0.482 |
| 1109707 | what medium do radio waves travel through | 0.599 |
| 156498 | do google docs auto save | 0.664 |
| 1064670 | why do hunters pattern their shotguns? | 0.681 |
| 141630 | describe how muscles and bones work together to produce movement | 0.684 |
| 258062 | how long does it take to remove wisdom tooth | 0.684 |
| 640502 | what does it mean if your tsh is low | 0.723 |

**Why these mostly work:**
- Mid-range mean (0.595) reflects that process passages are abundant in MS MARCO and dense retrieval handles topical similarity well.
- The outlier "how does granulation tissue start" (0.215) fails because "granulation tissue" is medical jargon and "start" is too generic — the relevant passages may exist but don't use "start" as the verb.

**Mitigation paths:**
- Stemming would help "start" / "starts" / "starting" / "began" matching.
- Context-aware reranking from a cross-encoder.

---

### 5. Definition / vocabulary lookup

**Definition:** Query is "define X", "what is X", "meaning of X" with X
being a single word or short phrase. The retrieval target is a passage
that defines or explains X.

**Expected pattern:** RRF wins consistently because:
- BM25 catches passages with literal "X" in them.
- Dense catches passages topically related to X.
- Both signals concentrate on definitional passages; RRF amplifies the consensus.

**Examples:**

| qid | Query | nDCG@10 |
|---|---|---|
| 121171 | define etruscans | 0.312 |
| 768208 | what is mamey | 0.342 |
| 390360 | ia suffix meaning | 0.488 |
| 174463 | dog day afternoon meaning | 0.576 |
| 1106979 | define pareto chart in statistics | 0.757 |
| 1105792 | define: geon | 0.803 |
| 135802 | definition of laudable | 0.829 |
| 1127540 | meaning of shebang | 0.872 |
| 701453 | what is a statutory deed | 0.942 |

**Why these mostly work:**
- The query and the target passage share the literal term X — perfect for BM25.
- The lower scorers ("etruscans", "mamey") have rare-term issues — the entity exists but in few passages, so top-10 has limited material to choose from.

**No mitigation needed for the head; tail-term issues addressed by patterns 1 and 3.**

---

## 12 specific failure patterns observed

These are concrete patterns identified from the per-query analysis above.
Each pattern is a recurring shape that shows up in multiple bottom-quartile
queries.

### Pattern 1: Acronym ↔ expanded-form mismatch
The query uses an acronym (`bmt`, `fao`, `tsh`) and the relevant passage
uses the expanded form (or vice versa). Without query-side expansion or
acronym dictionary, both BM25 and dense miss the connection.

*Example:* qid 118440 "define bmt medical" — relevant passages discuss
"bone marrow transplant" but contain no "bmt" token.

### Pattern 2: Ambiguous short token, no clarifier
A 2–3-letter query token that has no canonical referent. The corpus has
many possible matches, none clearly relevant to the user's actual intent.

*Example:* qid 673670 "what is a alm" — "alm" matches many unrelated
passages; no signal favours any one interpretation.

### Pattern 3: Yes/no question with no consolidation
The user is asking for a binary answer that requires aggregating evidence
across multiple passages. Retrieval returns relevant passages but none
states the answer in yes/no form.

*Example:* qid 405163 "is caffeine an narcotic" — passages discuss
caffeine's effects and the legal definition of narcotics, but rarely make
the conjunction explicit.

### Pattern 4: Comparison query (X vs Y)
The user wants a contrast between two things. Most passages discuss either
X or Y individually; comparison passages are rare in MS MARCO.

*Example:* qid 1136043 "difference between a hotel and motel" — passages
defining one or the other are common; passages contrasting both are rare.

### Pattern 5: Compound location/role filter
A factual query gated by a specific location or role qualifier (e.g.,
"in Nebraska", "in Oklahoma City"). The qualifier is the rare constraint;
most passages match the topic but not the location.

*Example:* qid 42255 "average salary for dental hygienist in nebraska" —
salary passages are easy to find; Nebraska-specific passages much rarer.

### Pattern 6: Temporal attribute about a named entity
Query asks "when did X happen" or "how old is X". Passages discuss X but
don't always pair X with the date / age in retrievable form.

*Example:* qid 938400 "when did family feud come out?" — relevant passage
must contain both "Family Feud" and the year 1976 in close proximity.

### Pattern 7: Causal / explanatory query about an entity
"Why did X happen", "who caused Y". The answer is one specific passage
that explains the cause-effect chain; many passages mention X superficially.

*Example:* qid 1071750 "why is pete rose banned from hall of fame" — many
passages mention Pete Rose; the gambling-explanation passage is what's
needed but not consistently top-ranked.

### Pattern 8: Best-of / recommendation query
"Best X for Y", "what is the best way to Z". Retrieval target is a passage
that explicitly recommends or enumerates "best" choices. Recommendation
framing is rare in encyclopaedic passages.

*Example:* qid 555530 "what are best foods to lower cholesterol" — passages
list foods that lower cholesterol but rarely use the word "best".

### Pattern 9: Domain-qualifier query
Short query with a clarifying domain ("X? earth science", "X medical").
The clarifier is supposed to disambiguate but often doesn't help retrieval
because most candidate passages don't contain both.

*Example:* qid 1116380 "what is a nonconformity? earth science" —
"nonconformity" matches many psychology / sociology passages; the "earth
science" qualifier is too short to filter.

### Pattern 10: Multi-step medical / scientific query
Query bridges two medical concepts with a causal link. Passages discuss
each concept individually; passages explicitly stating the link are rarer.

*Example:* qid 67316 "can fever cause miscarriage early pregnancy" —
fever passages and miscarriage passages exist; passages linking them
specifically in early pregnancy are rare.

### Pattern 11: Compound entity treated as single concept
Query treats two related terms as a single entity ("chaff and flare", "X
and Y"). Most passages mention the two terms separately, not together.

*Example:* qid 1115210 "what is chaff and flare" — passages mention chaff
or flares as countermeasures individually; passages defining them together
are rarer.

### Pattern 12: Process query with non-canonical verb
Query uses one verb ("start", "begin") but relevant passages use a
synonym ("forms", "develops", "originates"). Stemming would close some
of this gap; semantic dense retrieval helps but doesn't fully bridge it.

*Example:* qid 1133579 "how does granulation tissue start" — passages
describe how granulation tissue *forms* or *develops*, not *starts*.

---

## What this means for the system

The bottom-quartile queries cluster around **three actionable opportunities**:

1. **Query expansion / rewriting** — patterns 1, 2, 9, 12. Acronym expansion + synonym normalisation + stemming would lift the lower tail significantly without changing the retrieval architecture.

2. **Query decomposition for compositional queries** — patterns 3, 4, 5, 8, 10. An LLM-based decomposition step (split the query into sub-queries, retrieve per sub-query, intersect) is a future architectural addition.

3. **Cross-encoder reranking** — patterns 6, 7, 11. A BERT-style cross-encoder over the top-100 from RRF would resolve the temporal/causal/compound patterns by attending to query-passage tokens jointly. This is the standard "hybrid retrieval + cross-encoder rerank" architecture and a strong future candidate.

The middle-quartile (entity-fact, process) is already at 0.58–0.60 nDCG@10 — close to the upper bound the current architecture can reach without external knowledge or rewriting. The path to 0.70+ goes through query understanding, not retrieval.

---

## Methodological note

Categories were assigned manually by reading each query and judging the
intent. Some queries fit multiple categories (e.g., "difference between
hotel and motel" is both Comparison and Definition); the assignment goes
to the most discriminating category for failure-mode analysis. The
classification log is in this document; the per-query nDCG values are
in `benchmarks/results/20260425T172427Z_1C_5133355b.json` under
`per_query_rrf`.

For DL2019 (the other 43 queries), the same taxonomy could be applied;
deferred to a future audit.
