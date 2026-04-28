"""Tests for the ACL filter."""
from __future__ import annotations

import pytest

from retrieval.acl import ACLFilter, PassageACL, ROLES


@pytest.fixture
def small_acl() -> PassageACL:
    acl = PassageACL()
    acl.generate(["p0", "p1", "p2", "p3", "p4"], seed=42)
    return acl


def test_generate_admin_always_has_access(small_acl):
    for pid in ["p0", "p1", "p2", "p3", "p4"]:
        assert small_acl.can_access(pid, "admin"), f"admin must have access to {pid}"


def test_generate_known_roles_only():
    acl = PassageACL()
    acl.generate(["p0"])
    allowed = acl._acl["p0"]
    assert all(r in ROLES for r in allowed)


def test_can_access_unknown_passage_denies_non_admin():
    acl = PassageACL()
    acl.generate(["p0"])
    assert acl.can_access("unknown_pid", "admin")
    assert not acl.can_access("unknown_pid", "viewer")


def test_num_passages_matches_generate_input():
    acl = PassageACL()
    acl.generate([f"p{i}" for i in range(100)])
    assert acl.num_passages == 100


def test_accessible_set_includes_admin_passages():
    acl = PassageACL()
    acl.generate(["p0", "p1", "p2"])
    assert acl.accessible_set("admin") == {"p0", "p1", "p2"}


def test_accessible_set_subset_of_admin_for_restricted_role():
    acl = PassageACL()
    acl.generate([f"p{i}" for i in range(50)], seed=7)
    assert acl.accessible_set("viewer").issubset(acl.accessible_set("admin"))


def test_save_load_round_trip(tmp_path):
    acl = PassageACL()
    acl.generate(["p0", "p1", "p2"])
    acl.save(tmp_path)

    acl2 = PassageACL()
    acl2.load(tmp_path)
    assert acl2.num_passages == 3
    for pid in ["p0", "p1", "p2"]:
        assert acl2.can_access(pid, "admin")
        assert acl2._acl[pid] == acl._acl[pid]


@pytest.fixture
def filter_with_acl() -> ACLFilter:
    """ACL where p0 and p1 accessible to 'engineer', p2 is admin-only."""
    acl = PassageACL()
    acl._acl = {
        "p0": frozenset({"admin", "engineer"}),
        "p1": frozenset({"admin", "engineer", "viewer"}),
        "p2": frozenset({"admin"}),
    }
    return ACLFilter(acl)


def test_filter_removes_inaccessible(filter_with_acl):
    results = [("p0", 1.0), ("p1", 0.9), ("p2", 0.8)]
    filtered = filter_with_acl.filter(results, user_role="engineer")
    ids = [d for d, _ in filtered]
    assert "p2" not in ids


def test_filter_preserves_accessible(filter_with_acl):
    results = [("p0", 1.0), ("p1", 0.9), ("p2", 0.8)]
    filtered = filter_with_acl.filter(results, user_role="engineer", top_k=10)
    ids = [d for d, _ in filtered]
    assert "p0" in ids
    assert "p1" in ids


def test_filter_admin_gets_all(filter_with_acl):
    results = [("p0", 1.0), ("p1", 0.9), ("p2", 0.8)]
    assert len(filter_with_acl.filter(results, user_role="admin")) == 3


def test_filter_truncates_to_top_k(filter_with_acl):
    results = [("p0", 1.0), ("p1", 0.9), ("p2", 0.8)]
    assert len(filter_with_acl.filter(results, user_role="admin", top_k=2)) == 2


def test_filter_no_top_k_returns_all_accessible(filter_with_acl):
    results = [("p0", 1.0), ("p1", 0.9), ("p2", 0.8)]
    filtered = filter_with_acl.filter(results, user_role="engineer")
    assert len(filtered) == 2  # p0, p1 (p2 is admin-only)


def test_filter_preserves_score_order(filter_with_acl):
    results = [("p1", 0.9), ("p0", 0.5)]
    filtered = filter_with_acl.filter(results, user_role="engineer", top_k=5)
    assert [d for d, _ in filtered] == ["p1", "p0"]


def test_filter_empty_results(filter_with_acl):
    assert filter_with_acl.filter([], user_role="engineer") == []


def test_filter_all_inaccessible(filter_with_acl):
    results = [("p2", 1.0)]
    assert filter_with_acl.filter(results, user_role="viewer") == []
