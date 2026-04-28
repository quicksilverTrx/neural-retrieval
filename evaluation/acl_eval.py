"""ACL-aware retrieval quality eval.

Steps:
    1. Load BM25 index
    2. Generate / load synthetic ACL data for all 8.8M passages
    3. Measure unrestricted Recall@100 baseline
    4. Measure ACL-filtered Recall@100 for each role
       (retrieve top-K * oversample, filter, truncate to K)
    5. Report Recall@100 drop per role

Usage:
    python evaluation/acl_eval.py
    python evaluation/acl_eval.py --role engineer
    python evaluation/acl_eval.py --top-k 10 --oversample 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evaluation.metrics import recall_at_k
from evaluation.trec_eval import (
    TREC_DL_2020,
    combined_queries,
    load_qrels,
    qrels_hash,
)
from retrieval.acl import ACLFilter, PassageACL, ROLES
from retrieval.inverted_index import BM25Retriever, BM25Scorer
from retrieval.inverted_index.persistence import load_index

BM25_INDEX_PATH = REPO_ROOT / "data" / "custom_bm25_8m.bin"
ACL_DATA_DIR = REPO_ROOT / "data" / "acl"


def _commit_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def write_results(payload: dict) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cfg_hash = hashlib.sha256(
        json.dumps(payload["config"], sort_keys=True).encode()
    ).hexdigest()[:8]
    out = REPO_ROOT / "benchmarks" / "results" / f"{ts}_acl_{cfg_hash}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"Results written to {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="ACL-aware retrieval eval")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--oversample", type=int, default=2,
                        help="Retrieve top_k * oversample before ACL filter")
    parser.add_argument("--role", default=None,
                        help="Single role to evaluate (default: all roles)")
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    args = parser.parse_args()

    if not BM25_INDEX_PATH.exists():
        print(f"ERROR: BM25 index not found at {BM25_INDEX_PATH}")
        print("Run bm25_eval.py --index-path first.")
        sys.exit(1)

    print("=" * 60)
    print("ACL-Aware Retrieval Eval")
    print("=" * 60)

    queries = combined_queries()
    qrels_2020 = load_qrels(TREC_DL_2020)

    print(f"Loading BM25 index from {BM25_INDEX_PATH} ...")
    index, _, _ = load_index(BM25_INDEX_PATH)
    scorer = BM25Scorer(k1=args.k1, b=args.b)
    retriever = BM25Retriever(index=index, scorer=scorer)

    if (ACL_DATA_DIR / "passage_acl.json").exists():
        print("Loading existing ACL data ...")
        acl = PassageACL()
        acl.load(ACL_DATA_DIR)
    else:
        print(f"Generating synthetic ACL data for {index.num_docs:,} passages ...")
        all_pids = list(index._doc_lengths.keys())
        acl = PassageACL()
        acl.generate(all_pids)
        acl.save(ACL_DATA_DIR)

    acl_filter = ACLFilter(acl)
    print(f"  ACL loaded: {acl.num_passages:,} passages")

    print(f"\nUnrestricted retrieval (top-{args.top_k}) ...")
    unrestricted_run: dict[str, list[str]] = {}
    for qid, text in queries.items():
        results, _ = retriever.retrieve_timed(text, top_k=args.top_k)
        # BM25Retriever returns int doc_ids; qrels use str keys
        unrestricted_run[qid] = [str(d) for d, _ in results]

    run_2020 = {qid: v for qid, v in unrestricted_run.items() if qid in qrels_2020}
    baseline_recall = recall_at_k(qrels_2020, run_2020, k=100)
    print(f"  Baseline Recall@100 = {baseline_recall:.4f}")

    roles_to_test = [args.role] if args.role else ROLES
    oversample_k = args.top_k * args.oversample

    role_results = {}
    print(f"\nACL-filtered retrieval (oversample={args.oversample}×) ...")
    for role in roles_to_test:
        role_run: dict[str, list[str]] = {}
        for qid, text in queries.items():
            results, _ = retriever.retrieve_timed(text, top_k=oversample_k)
            results_str = [(str(d), s) for d, s in results]
            filtered = acl_filter.filter(results_str, user_role=role, top_k=args.top_k)
            role_run[qid] = [d for d, _ in filtered]

        run_2020_role = {qid: v for qid, v in role_run.items() if qid in qrels_2020}
        recall = recall_at_k(qrels_2020, run_2020_role, k=100)
        drop = baseline_recall - recall
        role_results[role] = {"Recall@100": round(recall, 4), "drop": round(drop, 4)}
        print(f"  {role:<10} Recall@100={recall:.4f}  drop={drop:.4f}")

    payload = {
        "schema_version": "1.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "config": {
            "retriever": "bm25_acl_filtered",
            "bm25": {"k1": args.k1, "b": args.b},
            "top_k": args.top_k,
            "oversample": args.oversample,
            "qrels_hash": qrels_hash(TREC_DL_2020),
        },
        "dataset": "trec_dl_2020",
        "unrestricted_recall_100": round(baseline_recall, 4),
        "acl_role_results": role_results,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "commit_sha": _commit_sha(),
        },
    }
    write_results(payload)


if __name__ == "__main__":
    main()
