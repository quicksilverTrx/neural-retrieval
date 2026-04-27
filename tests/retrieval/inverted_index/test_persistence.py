"""Tests for retrieval/inverted_index/persistence.py.

The NRIDX2 format serialises an ``InvertedIndex`` directly — its ``_index``
is a ``dict[str, array.array('i')]`` and ``_doc_lengths`` is ``dict[int, int]``.
These tests build real InvertedIndex instances and verify the round-trip
preserves structure, checksums, and detects corruption.
"""
from __future__ import annotations

import json

import pytest

from retrieval.inverted_index.index import InvertedIndex
from retrieval.inverted_index.persistence import load_index, save_index


def _sample_index() -> InvertedIndex:
    """Small InvertedIndex with a few terms and doc-length entries."""
    idx = InvertedIndex()
    idx.add_document(0, ["hello", "hello", "world"])   # hello×2 world×1
    idx.add_document(1, ["hello"])                      # hello×1
    return idx


def _indexes_equal(a: InvertedIndex, b: InvertedIndex) -> bool:
    """Structural equality on the fields the format persists."""
    if a.num_docs != b.num_docs:
        return False
    if a._total_tokens != b._total_tokens:
        return False
    if a._doc_lengths != b._doc_lengths:
        return False
    if set(a._index.keys()) != set(b._index.keys()):
        return False
    for term in a._index:
        if list(a._index[term]) != list(b._index[term]):
            return False
    return True


def test_save_returns_sha256_string(tmp_path):
    idx = _sample_index()
    checksum = save_index(idx, tmp_path / "idx.bin", {"k1": 1.2, "b": 0.75})
    assert checksum.startswith("sha256:")
    assert len(checksum) == len("sha256:") + 64


def test_round_trip_preserves_index(tmp_path):
    idx = _sample_index()
    config = {"k1": 1.2, "b": 0.75}
    path = tmp_path / "idx.bin"
    checksum = save_index(idx, path, config)

    loaded, loaded_config, loaded_checksum = load_index(path)
    assert _indexes_equal(loaded, idx)
    assert loaded_config == config
    assert loaded_checksum == checksum


def test_round_trip_no_config(tmp_path):
    idx = _sample_index()
    path = tmp_path / "idx.bin"
    save_index(idx, path)
    loaded, config, _ = load_index(path)
    assert _indexes_equal(loaded, idx)
    assert config == {}


def test_round_trip_empty_index(tmp_path):
    path = tmp_path / "idx.bin"
    empty = InvertedIndex()
    save_index(empty, path)
    loaded, _, _ = load_index(path)
    assert _indexes_equal(loaded, empty)
    assert loaded.num_docs == 0


def test_creates_parent_directories(tmp_path):
    nested = tmp_path / "a" / "b" / "c" / "idx.bin"
    save_index(_sample_index(), nested)
    assert nested.exists()


def test_meta_sidecar_written(tmp_path):
    path = tmp_path / "idx.bin"
    save_index(_sample_index(), path, config={"version": 2})

    meta_path = path.with_suffix(".meta.json")
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert "checksum_sha256" in meta
    assert meta["format"] == "NRIDX2"
    assert meta["num_docs"] == 2
    assert meta["config"] == {"version": 2}
    assert meta["bytes"] > 0


def test_meta_checksum_matches_return_value(tmp_path):
    path = tmp_path / "idx.bin"
    checksum = save_index(_sample_index(), path)
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    assert f"sha256:{meta['checksum_sha256']}" == checksum


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_index(tmp_path / "nonexistent.bin")


def test_load_bad_magic_raises(tmp_path):
    path = tmp_path / "bad.bin"
    path.write_bytes(b"NOTANINDEX\n" + b"\x00" * 200)
    with pytest.raises(ValueError, match="Bad magic"):
        load_index(path)


def test_load_too_small_raises(tmp_path):
    path = tmp_path / "tiny.bin"
    path.write_bytes(b"abc")
    with pytest.raises(ValueError, match="too small"):
        load_index(path)


def test_load_corrupted_data_raises(tmp_path):
    path = tmp_path / "idx.bin"
    save_index(_sample_index(), path)

    # Flip a byte in the middle of the payload (well past size fields, before
    # the 64-byte trailer) so the stored checksum disagrees.
    raw = bytearray(path.read_bytes())
    middle = (len(raw) - 64) // 2
    raw[middle] ^= 0xFF
    path.write_bytes(bytes(raw))

    with pytest.raises(ValueError, match="Checksum mismatch"):
        load_index(path)


def test_same_data_same_checksum(tmp_path):
    idx = _sample_index()
    config = {"k1": 1.2, "b": 0.75}
    c1 = save_index(idx, tmp_path / "a.bin", config)
    c2 = save_index(idx, tmp_path / "b.bin", config)
    assert c1 == c2
