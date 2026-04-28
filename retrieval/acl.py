"""ACL-aware post-retrieval filter.

Two architectures were considered:

1. Post-retrieval filter: retrieve top-K*N, filter by ACL, truncate to K.
   + Simple: ACL check is a set intersection in Python.
   - Wasteful: oversample N× for filtering.

2. Query-time FAISS metadata filter: pass allowed_doc_ids as a FAISS
   IDSelector at search time, so FAISS only evaluates documents the caller
   can see.
   + No wasted computation for heavily-restricted roles.
   - Couples FAISS with ACL logic; IDSelector has non-trivial overhead at 8.8M.

Implementation here is post-retrieval (option 1) — the simpler default that
decouples retrieval from permissions. See `docs/design_decisions.md` for the
full rationale.

Usage
-----
    from retrieval.acl import ACLFilter, PassageACL

    acl = PassageACL()
    acl.load(data_dir)   # load synthetic ACL data

    f = ACLFilter(acl)
    filtered = f.filter(results, user_role="engineer", top_k=10)
"""
from __future__ import annotations

import json
import random
from pathlib import Path

# Five synthetic roles
ROLES = ["admin", "engineer", "analyst", "sales", "viewer"]

# Probability that a passage is accessible to a non-admin role
_ROLE_ACCESS_PROB = {
    "admin": 1.0,        # admin sees everything
    "engineer": 0.80,
    "analyst": 0.65,
    "sales": 0.50,
    "viewer": 0.35,
}


class PassageACL:
    """Stores allowed_roles for each passage.

    Backed by a dict for in-memory lookups during retrieval.
    Larger deployments would use an external ACL service.
    """

    def __init__(self) -> None:
        # passage_id → frozenset of allowed roles
        self._acl: dict[str, frozenset[str]] = {}

    # ------------------------------------------------------------------
    # Build (generation)
    # ------------------------------------------------------------------

    def generate(
        self,
        passage_ids: list[str],
        seed: int = 42,
    ) -> None:
        """Generate synthetic ACL data for a list of passage IDs.

        For each passage, each non-admin role has a fixed probability of
        access (see _ROLE_ACCESS_PROB). Admin always has access.
        """
        rng = random.Random(seed)
        for pid in passage_ids:
            allowed = {"admin"}   # admin always permitted
            for role in ROLES:
                if role == "admin":
                    continue
                if rng.random() < _ROLE_ACCESS_PROB[role]:
                    allowed.add(role)
            self._acl[pid] = frozenset(allowed)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, data_dir: Path) -> None:
        """Save ACL data as JSON (list of {pid, roles} objects)."""
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        out = data_dir / "passage_acl.json"
        records = [
            {"pid": pid, "roles": sorted(roles)}
            for pid, roles in self._acl.items()
        ]
        out.write_text(json.dumps(records))
        print(f"  ACL data written to {out} ({len(records):,} passages)")

    def load(self, data_dir: Path) -> None:
        """Load ACL data from JSON."""
        path = Path(data_dir) / "passage_acl.json"
        records = json.loads(path.read_text())
        self._acl = {r["pid"]: frozenset(r["roles"]) for r in records}

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def can_access(self, passage_id: str, role: str) -> bool:
        """True if role is allowed to see passage_id.

        Unknown passages default to admin-only (deny by default).
        """
        allowed = self._acl.get(passage_id, frozenset({"admin"}))
        return role in allowed

    def accessible_set(self, role: str) -> set[str]:
        """Return the set of all passage IDs accessible to role.

        O(n) scan — for production, maintain a role → set[pid] inverted index.
        """
        return {pid for pid, roles in self._acl.items() if role in roles}

    @property
    def num_passages(self) -> int:
        return len(self._acl)


class ACLFilter:
    """Post-retrieval ACL filter.

    Filters a ranked list of (doc_id, score) tuples to those accessible
    by the requesting user's role, then returns the top-k.
    """

    def __init__(self, acl: PassageACL) -> None:
        self._acl = acl

    def filter(
        self,
        results: list[tuple[str, float]],
        user_role: str,
        top_k: int | None = None,
    ) -> list[tuple[str, float]]:
        """Filter results to passages accessible by user_role.

        Args:
            results:   Ranked list of (doc_id, score). Caller should oversample
                       to survive filtering.
            user_role: Role of the requesting user.
            top_k:     Truncate to this many results after filtering
                       (None = return all accessible results).

        Returns:
            Filtered, optionally truncated list of (doc_id, score).
        """
        filtered = [
            (doc_id, score)
            for doc_id, score in results
            if self._acl.can_access(doc_id, user_role)
        ]
        if top_k is not None:
            filtered = filtered[:top_k]
        return filtered
