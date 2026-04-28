"""Locust load test for the FastAPI serving layer.

Usage
-----
Start the API first (in a separate terminal):
    uvicorn api.main:app --workers 1 --port 8000

Then run the load test:
    # Interactive UI (http://localhost:8089):
    locust -f tests/load/locustfile.py --host http://localhost:8000

    # Headless: 10 users, ramp 1/s, run 120s, save results:
    locust -f tests/load/locustfile.py --host http://localhost:8000 \
        --headless -u 10 -r 1 --run-time 120s \
        --csv benchmarks/results/locust_10u

    # Full sweep (1, 5, 10, 25, 50 users), 60s each:
    for U in 1 5 10 25 50; do
      locust -f tests/load/locustfile.py --host http://localhost:8000 \
        --headless -u $U -r $U --run-time 60s \
        --csv benchmarks/results/locust_${U}u
    done

After each run, P50/P95/P99 are in benchmarks/results/locust_{U}u_stats.csv.

Query pool: TREC DL 2019+2020 (97 queries) with built-in fallback.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from locust import HttpUser, between, task

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


_FALLBACK_QUERIES = [
    "what is the capital of France",
    "how does photosynthesis work",
    "symptoms of appendicitis",
    "difference between RAM and ROM",
    "what causes earthquakes",
    "how to train a dog to sit",
    "what are the causes of the French Revolution",
    "how does the immune system fight viruses",
    "what is machine learning",
    "how to make sourdough bread",
    "what is the boiling point of water at altitude",
    "causes of World War 1",
    "how to treat a sunburn",
    "what is a blockchain",
    "how does GPS work",
    "what are the symptoms of diabetes",
    "how to write a cover letter",
    "what is quantum entanglement",
    "best way to learn a foreign language",
    "how does the stock market work",
]


def _load_query_pool() -> list[str]:
    """Load TREC DL queries; fall back to built-in sample if unavailable."""
    q_files = [
        REPO_ROOT / "data" / "queries" / "trec_dl_2019_queries.json",
        REPO_ROOT / "data" / "queries" / "trec_dl_2020_queries.json",
    ]
    queries: list[str] = []
    for f in q_files:
        if f.exists():
            data = json.loads(f.read_text())
            queries.extend(data.values())
    if not queries:
        print(
            "[locust] TREC query files not found — using built-in fallback. "
            "Run scripts/bootstrap_data.sh to load TREC queries."
        )
        return _FALLBACK_QUERIES
    print(f"[locust] Loaded {len(queries)} TREC DL queries")
    return queries


_QUERY_POOL: list[str] = _load_query_pool()


class SearchUser(HttpUser):
    """Simulates a user issuing search queries."""

    wait_time = between(0.1, 0.5)

    @task(6)
    def hybrid_search(self):
        query = random.choice(_QUERY_POOL)
        with self.client.post(
            "/search",
            json={"query": query, "mode": "hybrid", "top_k": 10},
            catch_response=True,
            name="/search [hybrid]",
        ) as resp:
            if resp.status_code == 200:
                body = resp.json()
                if "results" not in body:
                    resp.failure(f"Missing 'results' key: {body!r:.200}")
                else:
                    resp.success()
            elif resp.status_code == 503:
                resp.failure("Service not ready (503) — is the index loaded?")
            else:
                resp.failure(f"Unexpected status {resp.status_code}")

    @task(2)
    def bm25_search(self):
        query = random.choice(_QUERY_POOL)
        with self.client.post(
            "/search",
            json={"query": query, "mode": "bm25", "top_k": 10},
            catch_response=True,
            name="/search [bm25]",
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"status={resp.status_code}")

    @task(1)
    def dense_search(self):
        query = random.choice(_QUERY_POOL)
        with self.client.post(
            "/search",
            json={"query": query, "mode": "dense", "top_k": 10},
            catch_response=True,
            name="/search [dense]",
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"status={resp.status_code}")

    @task(1)
    def health_check(self):
        with self.client.get("/health", catch_response=True, name="/health") as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"health check failed: {resp.status_code}")

    def on_start(self):
        resp = self.client.get("/ready")
        if resp.status_code != 200:
            print(
                f"[locust] /ready returned {resp.status_code} — "
                "index may not be loaded. Proceeding anyway."
            )


class ACLSearchUser(SearchUser):
    """Same as SearchUser but exercises ACL filtering by sending user_role."""

    _ROLES = ["admin", "engineer", "analyst", "sales", "viewer"]
    _WEIGHTS = [0.05, 0.35, 0.25, 0.20, 0.15]

    @task(6)
    def hybrid_search_with_role(self):
        query = random.choice(_QUERY_POOL)
        role = random.choices(self._ROLES, weights=self._WEIGHTS, k=1)[0]
        with self.client.post(
            "/search",
            json={"query": query, "mode": "hybrid", "top_k": 10, "user_role": role},
            catch_response=True,
            name="/search [hybrid+acl]",
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"status={resp.status_code}")
