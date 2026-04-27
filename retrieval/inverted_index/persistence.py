"""Persistence layer for InvertedIndex: save/load with checksum verification.

File format — NRIDX2
--------------------
Forward-only binary stream. No pickle, no gzip, no random heap access.
Each posting list is already a contiguous ``array.array('i')`` in memory;
saving is one ``array.tobytes()`` call per term followed by ``file.write()``.

    magic         = b"NRIDX2\\n"
    version       = uint32 (= 2)
    reserved      = uint32 (= 0)
    num_docs      = uint64
    total_tokens  = uint64
    num_terms     = uint64
    num_doc_lens  = uint64
    config_len    = uint32
    config_bytes  = utf-8 JSON (length = config_len)

    repeat num_terms times:
        term_len       = uint16
        term_bytes     = utf-8 (length = term_len)
        posting_pairs  = uint32
        posting_data   = int32 × (2 × posting_pairs)

    doc_len_bytes     = int32 × (2 × num_doc_lens)
    sha256_hex        = 64 ASCII bytes (hex over everything above)

Two-pass load: pass 1 stream-hashes the payload to verify the checksum
*before* any parsing; pass 2 parses the now-trusted payload. Corruption
surfaces as ``ValueError("Checksum mismatch ...")`` rather than misleading
mid-parse ``UnicodeDecodeError`` / ``struct.error``.
"""
from __future__ import annotations

import hashlib
import json
import struct
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

_MAGIC = b"NRIDX2\n"
_VERSION = 2
_PAIR_TYPECODE = "i"          # int32
_TRAILER_BYTES = 64           # ascii hex of sha256
_CHUNK = 1 << 16              # 64 KB I/O chunks for streaming hash


# ---------------------------------------------------------------------------
# Header — fixed counts + variable-length JSON config
# ---------------------------------------------------------------------------

@dataclass
class NRIDXHeader:
    """The NRIDX2 header that follows the magic + version bytes.

    Wire layout:  ``<QQQQI>`` (4 × uint64 counts + uint32 config-len) followed
    by ``config_len`` UTF-8 bytes of JSON config.
    """
    num_docs: int
    total_tokens: int
    num_terms: int
    num_doc_lens: int
    config: dict

    _COUNTS_FMT = "<QQQQI"

    def pack(self) -> bytes:
        cfg = json.dumps(self.config, sort_keys=True).encode("utf-8")
        return struct.pack(
            self._COUNTS_FMT,
            self.num_docs, self.total_tokens, self.num_terms,
            self.num_doc_lens, len(cfg),
        ) + cfg

    @classmethod
    def counts_size(cls) -> int:
        return struct.calcsize(cls._COUNTS_FMT)

    @classmethod
    def unpack(cls, counts_bytes: bytes, config_bytes: bytes) -> "NRIDXHeader":
        nd, tt, nt, ndl, _cfg_len = struct.unpack(cls._COUNTS_FMT, counts_bytes)
        config = json.loads(config_bytes.decode("utf-8")) if config_bytes else {}
        return cls(num_docs=nd, total_tokens=tt, num_terms=nt,
                   num_doc_lens=ndl, config=config)


# ---------------------------------------------------------------------------
# HashingStream — file + sha256 wrapper
# ---------------------------------------------------------------------------

class HashingStream:
    """Wraps a binary file with a running sha256.

    ``write()`` and ``read_exact()`` each forward to the file *and* update the
    hash. Use ``hexdigest`` after the last hashed byte. The trailer (which
    holds the digest) is read or written via the underlying file directly to
    bypass hashing.
    """

    def __init__(self, file: BinaryIO) -> None:
        self._file = file
        self._sha = hashlib.sha256()

    def write(self, data: bytes) -> None:
        self._sha.update(data)
        self._file.write(data)

    def read_exact(self, n: int) -> bytes:
        data = self._file.read(n)
        if len(data) != n:
            raise ValueError(f"Unexpected EOF after reading {len(data)}/{n} bytes")
        self._sha.update(data)
        return data

    @property
    def hexdigest(self) -> str:
        return self._sha.hexdigest()


# ---------------------------------------------------------------------------
# Public API — save_index / load_index
# ---------------------------------------------------------------------------

def save_index(index_data: Any, path: Path, config: dict | None = None) -> str:
    """Serialise an ``InvertedIndex`` to ``path`` in NRIDX2 binary format.

    Peak RAM: one ``array.array('i')`` (already in memory as part of the index).

    Returns:
        Checksum string ``'sha256:<hex>'``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp_nridx")

    header = NRIDXHeader(
        num_docs=index_data._num_docs,
        total_tokens=index_data._total_tokens,
        num_terms=len(index_data._index),
        num_doc_lens=len(index_data._doc_lengths),
        config=config or {},
    )

    with tmp.open("wb") as f:
        stream = HashingStream(f)

        stream.write(_MAGIC)
        stream.write(struct.pack("<II", _VERSION, 0))
        stream.write(header.pack())

        for term, posting in index_data._index.items():
            _write_term_record(stream, term, posting)

        _write_doc_lengths(stream, index_data._doc_lengths)

        # Trailer: sha256 of everything written above (raw write, not hashed)
        f.write(stream.hexdigest.encode("ascii"))

    tmp.replace(path)
    checksum = stream.hexdigest
    _write_meta_sidecar(path, header, checksum)
    return f"sha256:{checksum}"


def load_index(path: Path) -> tuple[Any, dict, str]:
    """Load and verify an NRIDX2 index from ``path``.

    Two-pass: pass 1 stream-hashes and compares to the trailer; pass 2 parses
    the now-trusted bytes.

    Returns:
        ``(index_data, config, 'sha256:<hex>')``.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        ValueError: magic mismatch, version mismatch, or checksum mismatch.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Index file not found: {path}")
    total = path.stat().st_size
    if total < len(_MAGIC) + _TRAILER_BYTES:
        raise ValueError(f"File too small to be an NRIDX2 index: {path}")

    checksum = _verify_checksum(path, payload_size=total - _TRAILER_BYTES)
    index, config = _parse_payload(path)
    return index, config, f"sha256:{checksum}"


# ---------------------------------------------------------------------------
# Per-record write helpers
# ---------------------------------------------------------------------------

def _write_term_record(stream: HashingStream, term: str, posting: array) -> None:
    term_bytes = term.encode("utf-8")
    if len(term_bytes) > 0xFFFF:
        raise ValueError(f"term too long to encode (>64 KB): {term!r}")
    stream.write(struct.pack("<H", len(term_bytes)))
    stream.write(term_bytes)
    stream.write(struct.pack("<I", len(posting) // 2))
    stream.write(posting.tobytes())


def _write_doc_lengths(stream: HashingStream, doc_lengths: dict[int, int]) -> None:
    flat = array(_PAIR_TYPECODE)
    for did, dlen in doc_lengths.items():
        flat.extend((did, dlen))
    stream.write(flat.tobytes())


def _write_meta_sidecar(path: Path, header: NRIDXHeader, checksum: str) -> None:
    meta = {
        "checksum_sha256": checksum,
        "format": "NRIDX2",
        "version": _VERSION,
        "num_docs": header.num_docs,
        "num_terms": header.num_terms,
        "num_doc_lens": header.num_doc_lens,
        "total_tokens": header.total_tokens,
        "bytes": path.stat().st_size,
        "config": header.config,
    }
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))


# ---------------------------------------------------------------------------
# Pass 1 — verify checksum
# ---------------------------------------------------------------------------

def _verify_checksum(path: Path, payload_size: int) -> str:
    """Stream-hash ``payload_size`` bytes, compare to the 64-byte hex trailer.

    Returns the verified checksum on success; raises ``ValueError`` on
    magic/EOF/mismatch.
    """
    sha = hashlib.sha256()
    with path.open("rb") as f:
        magic = f.read(len(_MAGIC))
        if magic != _MAGIC:
            raise ValueError(
                f"Bad magic bytes — not an NRIDX2 index (got {magic!r}): {path}"
            )
        sha.update(magic)

        remaining = payload_size - len(_MAGIC)
        while remaining:
            chunk = f.read(min(_CHUNK, remaining))
            if not chunk:
                raise ValueError(f"Unexpected EOF during checksum pass: {path}")
            sha.update(chunk)
            remaining -= len(chunk)

        stored = f.read(_TRAILER_BYTES).decode("ascii")

    actual = sha.hexdigest()
    if actual != stored:
        raise ValueError(
            f"Checksum mismatch for {path}: stored={stored}, actual={actual}"
        )
    return actual


# ---------------------------------------------------------------------------
# Pass 2 — parse trusted payload
# ---------------------------------------------------------------------------

def _parse_payload(path: Path) -> tuple[Any, dict]:
    """Parse a checksum-verified NRIDX2 file into an InvertedIndex + config."""
    from .index import InvertedIndex

    with path.open("rb") as f:
        f.read(len(_MAGIC))   # magic already validated in pass 1
        version, _reserved = struct.unpack("<II", _read_exact(f, 8))
        if version != _VERSION:
            raise ValueError(
                f"Unsupported index version {version} (expected {_VERSION}): {path}"
            )

        counts_bytes = _read_exact(f, NRIDXHeader.counts_size())
        # Peek the config-len field so we know how many bytes to read next.
        cfg_len = struct.unpack(NRIDXHeader._COUNTS_FMT, counts_bytes)[-1]
        config_bytes = _read_exact(f, cfg_len)
        header = NRIDXHeader.unpack(counts_bytes, config_bytes)

        index = InvertedIndex()
        index._num_docs = header.num_docs
        index._total_tokens = header.total_tokens

        for _ in range(header.num_terms):
            term, posting = _read_term_record(f)
            index._index[term] = posting

        index._doc_lengths = _read_doc_lengths(f, header.num_doc_lens)

    return index, header.config


def _read_term_record(f: BinaryIO) -> tuple[str, array]:
    (term_len,) = struct.unpack("<H", _read_exact(f, 2))
    term = _read_exact(f, term_len).decode("utf-8")
    (n_pairs,) = struct.unpack("<I", _read_exact(f, 4))
    posting = array(_PAIR_TYPECODE)
    if n_pairs:
        posting.frombytes(_read_exact(f, n_pairs * 2 * posting.itemsize))
    return term, posting


def _read_doc_lengths(f: BinaryIO, num_doc_lens: int) -> dict[int, int]:
    flat = array(_PAIR_TYPECODE)
    if num_doc_lens:
        flat.frombytes(_read_exact(f, num_doc_lens * 2 * flat.itemsize))
    it = iter(flat)
    return dict(zip(it, it))


def _read_exact(f: BinaryIO, n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise ValueError(f"Unexpected EOF after reading {len(data)}/{n} bytes")
    return data
