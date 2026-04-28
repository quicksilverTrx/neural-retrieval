"""Tests for FAISS index corruption recovery."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from retrieval.dense.recovery import (
    record_index_checksum,
    validate_faiss_index,
)


def _make_fake_index(path: Path) -> Path:
    """Write a fake index.faiss + pids.json + meta.json for testing."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "index.faiss").write_bytes(b"fake faiss bytes")
    (path / "pids.json").write_text(json.dumps(["p0", "p1"]))
    (path / "meta.json").write_text(json.dumps({"nlist": 4, "m": 2}))
    return path


def test_record_checksum_writes_to_meta(tmp_path):
    _make_fake_index(tmp_path)
    checksum = record_index_checksum(tmp_path)

    meta = json.loads((tmp_path / "meta.json").read_text())
    assert "sha256" in meta
    assert meta["sha256"] == checksum


def test_record_checksum_is_hex_string(tmp_path):
    _make_fake_index(tmp_path)
    checksum = record_index_checksum(tmp_path)
    assert len(checksum) == 64
    assert all(c in "0123456789abcdef" for c in checksum)


def test_record_checksum_missing_index_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        record_index_checksum(tmp_path)


def test_validate_returns_true_for_valid_index(tmp_path):
    _make_fake_index(tmp_path)
    record_index_checksum(tmp_path)
    assert validate_faiss_index(tmp_path) is True


def test_validate_returns_false_after_corruption(tmp_path):
    _make_fake_index(tmp_path)
    record_index_checksum(tmp_path)
    (tmp_path / "index.faiss").write_bytes(b"corrupted bytes")
    assert validate_faiss_index(tmp_path) is False


def test_validate_returns_false_missing_index(tmp_path):
    assert validate_faiss_index(tmp_path) is False


def test_validate_returns_false_missing_meta(tmp_path):
    _make_fake_index(tmp_path)
    (tmp_path / "meta.json").unlink()
    assert validate_faiss_index(tmp_path) is False


def test_validate_returns_false_missing_checksum_key(tmp_path):
    _make_fake_index(tmp_path)
    assert validate_faiss_index(tmp_path) is False


def test_validate_returns_false_after_append(tmp_path):
    _make_fake_index(tmp_path)
    record_index_checksum(tmp_path)
    with open(tmp_path / "index.faiss", "ab") as f:
        f.write(b"\x00")
    assert validate_faiss_index(tmp_path) is False


@pytest.fixture
def tiny_emb_dir(tmp_path):
    """Minimal embeddings directory for rebuild testing."""
    pytest.importorskip("faiss")
    dim, n = 16, 800
    emb_dir = tmp_path / "embeddings"
    emb_dir.mkdir()

    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    mm = np.lib.format.open_memmap(
        str(emb_dir / "embeddings.npy"),
        mode="w+",
        dtype=np.float32,
        shape=(n, dim),
    )
    mm[:] = vecs
    mm.flush()

    (emb_dir / "pids.json").write_text(json.dumps([f"p{i}" for i in range(n)]))
    manifest = {"model_name": "test", "embedding_dim": dim, "num_passages": n, "dtype": "float32"}
    (emb_dir / "manifest.json").write_text(json.dumps(manifest))
    return emb_dir


def test_rebuild_index_creates_valid_index(tmp_path, tiny_emb_dir):
    from retrieval.dense.recovery import rebuild_index

    faiss_dir = tmp_path / "faiss"
    rebuild_index(tiny_emb_dir, faiss_dir, nlist=4, m=2, nbits=4, train_sample_size=500)

    assert (faiss_dir / "index.faiss").exists()
    assert (faiss_dir / "meta.json").exists()
    assert (faiss_dir / "pids.json").exists()


def test_rebuild_index_passes_validation(tmp_path, tiny_emb_dir):
    from retrieval.dense.recovery import rebuild_index

    faiss_dir = tmp_path / "faiss"
    rebuild_index(tiny_emb_dir, faiss_dir, nlist=4, m=2, nbits=4, train_sample_size=500)
    assert validate_faiss_index(faiss_dir) is True
