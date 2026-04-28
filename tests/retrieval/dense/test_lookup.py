"""Tests for PassageLookup (SQLite passage store)."""
from __future__ import annotations

import pytest

from retrieval.dense.lookup import PassageLookup


@pytest.fixture
def sample_chunks() -> list[tuple[str, str, str]]:
    return [
        ("c0", "p0", "The quick brown fox"),
        ("c1", "p1", "jumped over the lazy dog"),
        ("c2", "p0", "second chunk of passage zero"),
    ]


@pytest.fixture
def populated_db(tmp_path, sample_chunks):
    db_path = tmp_path / "passages.db"
    with PassageLookup(db_path) as lk:
        lk.build(sample_chunks)
    return db_path


def test_get_existing_chunk(populated_db):
    with PassageLookup(populated_db) as lk:
        assert lk.get("c0") == ("p0", "The quick brown fox")


def test_get_missing_chunk_returns_none(populated_db):
    with PassageLookup(populated_db) as lk:
        assert lk.get("nonexistent_chunk") is None


def test_get_batch_returns_all_present(populated_db):
    with PassageLookup(populated_db) as lk:
        result = lk.get_batch(["c0", "c2"])
    assert set(result.keys()) == {"c0", "c2"}
    assert result["c0"] == ("p0", "The quick brown fox")
    assert result["c2"] == ("p0", "second chunk of passage zero")


def test_get_batch_omits_missing(populated_db):
    with PassageLookup(populated_db) as lk:
        result = lk.get_batch(["c0", "missing_key"])
    assert "missing_key" not in result
    assert "c0" in result


def test_get_batch_empty_input(populated_db):
    with PassageLookup(populated_db) as lk:
        assert lk.get_batch([]) == {}


def test_count_matches_inserted_rows(tmp_path, sample_chunks):
    with PassageLookup(tmp_path / "db") as lk:
        lk.build(sample_chunks)
        assert lk.count() == len(sample_chunks)


def test_count_empty_db(tmp_path):
    with PassageLookup(tmp_path / "db") as lk:
        assert lk.count() == 0


def test_rebuild_replaces_existing_rows(tmp_path):
    path = tmp_path / "db"
    with PassageLookup(path) as lk:
        lk.build([("c0", "p0", "original text")])

    with PassageLookup(path) as lk:
        lk.build([("c0", "p0", "updated text")])
        assert lk.get("c0") == ("p0", "updated text")


def test_from_corpus_chunk_id_equals_passage_id(tmp_path):
    passages = [("pid_1", "text one"), ("pid_2", "text two")]
    PassageLookup.from_corpus(passages, tmp_path / "db")

    with PassageLookup(tmp_path / "db") as lk:
        passage_id, text = lk.get("pid_1")

    assert passage_id == "pid_1"
    assert text == "text one"


def test_context_manager_closes_connection(tmp_path):
    path = tmp_path / "db"
    lk = PassageLookup(path)

    with lk:
        lk.build([("c0", "p0", "text")])

    assert lk._conn is None


def test_large_batch_build_and_retrieve(tmp_path):
    n = 5_000
    passages = [(f"c{i}", f"p{i}", f"text for passage {i}") for i in range(n)]
    with PassageLookup(tmp_path / "db") as lk:
        lk.build(passages, batch_size=1_000)
        assert lk.count() == n
        result = lk.get("c4999")

    assert result == ("p4999", "text for passage 4999")
