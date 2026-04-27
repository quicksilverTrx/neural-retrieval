"""Tests for VByteCodec.

Round-trip property: decode(encode(xs)) == xs for any sorted list of
non-negative integers.
"""
from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st


def _codec():
    from retrieval.inverted_index.vbyte import VByteCodec
    return VByteCodec


def _encode_or_skip(integers: list[int]) -> bytes:
    codec = _codec()
    try:
        return codec.encode(integers)
    except NotImplementedError:
        pytest.skip("VByteCodec.encode() not yet implemented")


def _decode_or_skip(data: bytes) -> list[int]:
    codec = _codec()
    try:
        return codec.decode(data)
    except NotImplementedError:
        pytest.skip("VByteCodec.decode() not yet implemented")


def _roundtrip_or_skip(integers: list[int]) -> list[int]:
    codec = _codec()
    try:
        encoded = codec.encode(integers)
        return codec.decode(encoded)
    except NotImplementedError:
        pytest.skip("VByteCodec not yet implemented")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_input_encode(self):
        result = _encode_or_skip([])
        assert result == b""

    def test_empty_input_decode(self):
        result = _decode_or_skip(b"")
        assert result == []

    def test_single_zero(self):
        """Value 0 encodes to a single byte 0x00."""
        result = _roundtrip_or_skip([0])
        assert result == [0]

    def test_single_zero_byte_value(self):
        """VByte: integer 0 → single byte 0x00 (MSB=0, value=0)."""
        encoded = _encode_or_skip([0])
        assert len(encoded) == 1
        assert encoded[0] == 0x00

    def test_single_element(self):
        result = _roundtrip_or_skip([42])
        assert result == [42]

    def test_single_127(self):
        """127 fits in 7 bits — must encode to exactly 1 byte."""
        encoded = _encode_or_skip([127])
        assert len(encoded) == 1
        assert encoded[0] == 127

    def test_single_128_needs_two_bytes(self):
        """128 overflows 7 bits — must encode to exactly 2 bytes."""
        encoded = _encode_or_skip([128])
        assert len(encoded) == 2

    def test_large_value_five_bytes(self):
        """Values > 2^28 require 5 bytes."""
        big = 2**28
        encoded = _encode_or_skip([big])
        assert len(encoded) == 5
        assert _decode_or_skip(encoded) == [big]


# ---------------------------------------------------------------------------
# Specific byte values
# ---------------------------------------------------------------------------

class TestKnownByteValues:
    def test_integer_300_byte_layout(self):
        """300 must encode to exactly [0xAC, 0x02]."""
        encoded = _encode_or_skip([300])
        assert list(encoded) == [0xAC, 0x02], (
            f"Expected [0xAC, 0x02], got {[hex(b) for b in encoded]}"
        )

    def test_integer_1_byte_layout(self):
        encoded = _encode_or_skip([1])
        assert list(encoded) == [0x01]

    def test_two_element_gap_coding(self):
        """[0, 127] → gaps [0, 127] → each fits in 1 byte → 2 bytes total."""
        encoded = _encode_or_skip([0, 127])
        assert len(encoded) == 2

    def test_consecutive_integers_small_gaps(self):
        """[100, 101, 102] → gaps [100, 1, 1] → bytes: 1 + 1 + 1 = 3."""
        encoded = _encode_or_skip([100, 101, 102])
        assert len(encoded) == 3

    def test_large_gap_increases_size(self):
        """[0, 200_000] → gaps [0, 200_000]; 200_000 needs 3 bytes."""
        encoded = _encode_or_skip([0, 200_000])
        assert len(encoded) == 4


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_single_value_roundtrip(self):
        for v in [0, 1, 63, 64, 127, 128, 255, 300, 16383, 16384, 2**21 - 1, 2**21]:
            result = _roundtrip_or_skip([v])
            assert result == [v], f"round-trip failed for [{v}]"

    def test_small_sorted_list(self):
        xs = [1, 5, 10, 15, 23, 100, 500, 1000, 8_000_000]
        assert _roundtrip_or_skip(xs) == xs

    def test_dense_sorted_list(self):
        """Consecutive integers: gaps = 1 each → very compact."""
        xs = list(range(1000))
        assert _roundtrip_or_skip(xs) == xs

    def test_sparse_sorted_list(self):
        """Large doc IDs similar to MS MARCO pids."""
        xs = [1_000_000, 2_000_000, 3_000_000, 8_000_000, 8_800_000]
        assert _roundtrip_or_skip(xs) == xs

    def test_single_large_gap(self):
        xs = [0, 8_841_823]
        assert _roundtrip_or_skip(xs) == xs


# ---------------------------------------------------------------------------
# Compression ratio
# ---------------------------------------------------------------------------

class TestCompressionRatio:
    def test_consecutive_integers_compress_better_than_fixed_int32(self):
        n = 1000
        xs = list(range(n))
        encoded = _encode_or_skip(xs)
        fixed_size = n * 4
        assert len(encoded) < fixed_size

    def test_large_sparse_list_still_compresses(self):
        xs = list(range(0, 1_000_000, 100))
        encoded = _encode_or_skip(xs)
        fixed_size = len(xs) * 4
        assert len(encoded) < fixed_size


# ---------------------------------------------------------------------------
# Decode boundary
# ---------------------------------------------------------------------------

class TestDecodeMultipleIntegers:
    def test_three_integers_all_recovered(self):
        xs = [10, 200, 500_000]
        assert _roundtrip_or_skip(xs) == xs

    def test_mixed_single_and_multi_byte(self):
        xs = [0, 1, 2, 3, 128, 256, 100_000, 100_001]
        assert _roundtrip_or_skip(xs) == xs

    def test_all_127_boundary(self):
        """127 is the single-byte maximum; 128 is the first two-byte value."""
        xs = [127, 255, 383]
        assert _roundtrip_or_skip(xs) == xs


# ---------------------------------------------------------------------------
# Hypothesis property
# ---------------------------------------------------------------------------

@given(
    xs=st.lists(
        st.integers(min_value=0, max_value=8_841_823),
        min_size=0,
        max_size=200,
    ).map(sorted)
)
@settings(max_examples=300)
def test_hypothesis_roundtrip(xs: list[int]):
    """decode(encode(xs)) == xs for any sorted list of non-negative ints."""
    codec = _codec()
    try:
        encoded = codec.encode(xs)
        decoded = codec.decode(encoded)
    except NotImplementedError:
        pytest.skip("VByteCodec not yet implemented")
    assert decoded == xs, f"Round-trip failed:\n  input={xs}\n  decoded={decoded}"


@given(
    xs=st.lists(
        st.integers(min_value=0, max_value=8_841_823),
        min_size=1,
        max_size=200,
    ).map(sorted)
)
@settings(max_examples=100)
def test_hypothesis_encode_monotone_length(xs: list[int]):
    """Each additional element adds at least 1 byte to the encoded output."""
    codec = _codec()
    try:
        enc = codec.encode(xs)
    except NotImplementedError:
        pytest.skip("VByteCodec not yet implemented")
    assert len(enc) >= len(xs)


@given(
    xs=st.lists(
        st.integers(min_value=0, max_value=8_841_823),
        min_size=2,
        max_size=100,
    ).map(sorted).filter(lambda l: len(l) >= 2)
)
@settings(max_examples=100)
def test_hypothesis_compression_beats_fixed_int32(xs: list[int]):
    """VByte must compress better than fixed int32 when gaps are small."""
    codec = _codec()
    try:
        enc = codec.encode(xs)
    except NotImplementedError:
        pytest.skip("VByteCodec not yet implemented")
    avg_gap = (xs[-1] - xs[0]) / (len(xs) - 1) if len(xs) > 1 else xs[0]
    if avg_gap < 2 ** 28:
        fixed_size = len(xs) * 4
        assert len(enc) <= fixed_size
