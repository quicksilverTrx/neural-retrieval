#!/usr/bin/env bash
# One-time data bootstrap script.
# Run from repo root: bash scripts/bootstrap_data.sh
# Requires ~42 GB free disk. Uses foundry-llm conda env.
set -euo pipefail

PYTHON="/Users/ron/miniconda3/envs/foundry-llm/bin/python"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> Bootstrap starting in $REPO_ROOT"
mkdir -p "$REPO_ROOT/data"/{corpus,qrels,queries,beir,hotpotqa,nq,squad2,models}

# MS MARCO v1 passages (8.8M, ~3 GB compressed, ~13 GB uncompressed)
echo "==> Downloading MS MARCO corpus..."
$PYTHON -c "
from datasets import load_dataset
load_dataset('Tevatron/msmarco-passage-corpus', cache_dir='$REPO_ROOT/data/corpus')
print('MS MARCO corpus: OK')
"

# TREC DL 2020 (54 queries, graded relevance)
echo "==> Downloading TREC DL 2020..."
$PYTHON -c "
import ir_datasets
ds = ir_datasets.load('msmarco-passage/trec-dl-2020/judged')
_ = list(ds.queries_iter())
_ = list(ds.qrels_iter())
print('TREC DL 2020: OK')
"

# TREC DL 2019 (43 queries)
echo "==> Downloading TREC DL 2019..."
$PYTHON -c "
import ir_datasets
ds = ir_datasets.load('msmarco-passage/trec-dl-2019/judged')
_ = list(ds.queries_iter())
_ = list(ds.qrels_iter())
print('TREC DL 2019: OK')
"

# BEIR zero-shot datasets
echo "==> Downloading BEIR datasets..."
for ds in trec-covid fiqa scifact nfcorpus; do
  $PYTHON -c "
import ir_datasets
ir_datasets.load('beir/$ds')
print('BEIR $ds: OK')
" || echo "  WARNING: beir/$ds failed — may need manual download"
done

# HotpotQA (multi-hop eval)
echo "==> Downloading HotpotQA..."
$PYTHON -c "
from datasets import load_dataset
load_dataset('hotpot_qa', 'distractor', cache_dir='$REPO_ROOT/data/hotpotqa')
print('HotpotQA: OK')
"

# NaturalQuestions (answer correctness eval)
echo "==> Downloading NaturalQuestions (dev)..."
$PYTHON -c "
from datasets import load_dataset
load_dataset('google-research-datasets/natural_questions', 'default', cache_dir='$REPO_ROOT/data/nq')
print('NaturalQuestions: OK')
" || echo "  WARNING: NQ failed — may require HF token"

# SQuAD 2.0 unanswerable (abstention eval)
echo "==> Downloading SQuAD v2..."
$PYTHON -c "
from datasets import load_dataset
load_dataset('rajpurkar/squad_v2', cache_dir='$REPO_ROOT/data/squad2')
print('SQuAD v2: OK')
"

# DeBERTa NLI model (faithfulness scorer)
echo "==> Caching deberta-large-mnli..."
$PYTHON -c "
from transformers import AutoModel, AutoTokenizer
AutoModel.from_pretrained('microsoft/deberta-large-mnli', cache_dir='$REPO_ROOT/data/models')
AutoTokenizer.from_pretrained('microsoft/deberta-large-mnli', cache_dir='$REPO_ROOT/data/models')
print('DeBERTa NLI: OK')
"

echo "==> Bootstrap complete."
