"""Signed int8 two's-complement bit codec."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

INT8_BIT_WIDTH = 8
INT8_MIN = -128
INT8_MAX = 127
UINT8_MAX = 255


@dataclass(frozen=True)
class BitMetadata:
    """Metadata for one LSB-first int8 bit position."""

    bit_index: int
    mask: int
    role: str
    is_lsb: bool
    is_msb: bool
    is_sign_bit: bool


@dataclass(frozen=True)
class SignedInt8TwoComplementCodec:
    """Encode signed int8 model values and flip their two's-complement bits."""

    scale: float = 1.0

    def __post_init__(self) -> None:
        if not isfinite(self.scale) or self.scale <= 0:
            msg = "scale must be a finite positive number"
            raise ValueError(msg)

    def encode(self, value: int) -> int:
        """Encode a signed int8 value as an unsigned two's-complement byte."""
        signed_value = _validate_int8(value)
        return signed_value & UINT8_MAX

    def decode(self, encoded: int) -> int:
        """Decode an unsigned two's-complement byte as a signed int8 value."""
        byte = _validate_uint8(encoded)
        if byte & 0b1000_0000:
            return byte - 0b1_0000_0000
        return byte

    def quantize(self, value: float) -> int:
        """Quantize a model value to signed int8 using the per-tensor scale."""
        if not isfinite(value):
            msg = "value must be finite"
            raise ValueError(msg)

        quantized = round(value / self.scale)
        return min(max(int(quantized), INT8_MIN), INT8_MAX)

    def dequantize(self, value: int) -> float:
        """Convert a signed int8 quantized value back to model-value units."""
        return _validate_int8(value) * self.scale

    def flip_bit(self, encoded: int, bit_index: int) -> int:
        """Flip a bit in an encoded int8 byte using LSB-first indexing."""
        byte = _validate_uint8(encoded)
        return byte ^ self.bit_metadata(bit_index).mask

    def flip_value_bit(self, value: int, bit_index: int) -> int:
        """Flip a bit in a signed int8 value and return the signed result."""
        return self.decode(self.flip_bit(self.encode(value), bit_index))

    def bit_metadata(self, bit_index: int) -> BitMetadata:
        """Return metadata for a bit position using LSB-first indexing."""
        index = _validate_bit_index(bit_index)
        is_lsb = index == 0
        is_msb = index == INT8_BIT_WIDTH - 1
        is_sign_bit = is_msb

        role = "value"
        if is_lsb:
            role = "lsb"
        if is_sign_bit:
            role = "sign_msb"

        return BitMetadata(
            bit_index=index,
            mask=1 << index,
            role=role,
            is_lsb=is_lsb,
            is_msb=is_msb,
            is_sign_bit=is_sign_bit,
        )


def _validate_int8(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        msg = "value must be an integer"
        raise TypeError(msg)
    if value < INT8_MIN or value > INT8_MAX:
        msg = f"value must be in signed int8 range [{INT8_MIN}, {INT8_MAX}]"
        raise ValueError(msg)
    return value


def _validate_uint8(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        msg = "encoded value must be an integer"
        raise TypeError(msg)
    if value < 0 or value > UINT8_MAX:
        msg = f"encoded value must be in byte range [0, {UINT8_MAX}]"
        raise ValueError(msg)
    return value


def _validate_bit_index(bit_index: int) -> int:
    if isinstance(bit_index, bool) or not isinstance(bit_index, int):
        msg = "bit_index must be an integer"
        raise TypeError(msg)
    if bit_index < 0 or bit_index >= INT8_BIT_WIDTH:
        msg = f"bit_index must be in range [0, {INT8_BIT_WIDTH - 1}]"
        raise ValueError(msg)
    return bit_index
