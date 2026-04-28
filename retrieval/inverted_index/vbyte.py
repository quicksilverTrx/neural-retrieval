"""Variable-byte (VByte) encoding for gap-coded posting list doc_id sequences.

A canonical IR data structure: gap-code sorted doc IDs to small ints, then
encode each gap as a variable-length byte sequence.

Bit layout per byte: 7 data bits + 1 continuation bit (MSB).
- MSB = 1 → more bytes follow
- MSB = 0 → terminal byte
- Bytes carry the least-significant 7 bits first.

Example: encode integer 300:
    300 in binary: 1_0010_1100
    First 7 bits (LSB):  010_1100 = 44   → with MSB=1 (continuation): 0xAC
    Next  7 bits:        000_0010 = 2    → with MSB=0 (terminal):     0x02
    Result: bytes([0xAC, 0x02])

Round-trip property: decode(encode(xs)) == xs for any sorted list of
non-negative integers.
"""
from __future__ import annotations


class VByteCodec:
    """VByte encoding for sorted integer sequences (gap-coded doc IDs)."""

    @staticmethod
    def VByteEncode(gap: int) -> bytes:
        """Encode a single non-negative integer as a VByte sequence.
        """
        assert gap >= 0
        result: list[int] = []
        while True:
            chunk = gap & 0x7F
            gap >>= 7
            if gap == 0:
                # Terminal byte — MSB cleared.
                result.append(chunk)
                break
            # Continuation byte — MSB set.
            result.append(chunk | 0x80)
        return bytes(result)

    @staticmethod
    def VByteDecode(data: bytes, offset: int) -> tuple[int, int]:
        """Decode one VByte integer starting at ``data[offset]``.

        Returns ``(value, next_offset)`` where ``next_offset`` points at the
        first byte AFTER the consumed sequence.
        """
        shift = 0
        output = 0

        while True:
            byte = data[offset]
            offset += 1
            output |= (byte & 0x7F) << shift
            # Note parens: ``byte & 0x80 == 0`` parses as ``byte & (0x80 == 0)``
            # = ``byte & 0`` (always 0). Need explicit parens around the mask.
            if (byte & 0x80) == 0:
                break
            shift += 7

        return output, offset

    @staticmethod
    def encode(integers: list[int]) -> bytes:
        """Gap-code then VByte-encode a sorted list of non-negative integers.

        Returns a single concatenated ``bytes`` object so callers can
        slice/hash/write it directly.
        """
        previous = 0
        parts: list[bytes] = []
        for value in integers:
            # Static methods called from within the same class need the
            # class qualifier.
            parts.append(VByteCodec.VByteEncode(value - previous))
            previous = value
        # One allocation, O(n).
        return b"".join(parts)

    @staticmethod
    def decode(data: bytes) -> list[int]:
        """VByte-decode and prefix-sum reconstruct a sorted integer sequence."""
        result: list[int] = []
        offset = 0
        prefix = 0
        while offset < len(data):
            gap, offset = VByteCodec.VByteDecode(data, offset)
            prefix += gap
            result.append(prefix)
        return result
