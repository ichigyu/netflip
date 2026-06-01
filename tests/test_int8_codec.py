from __future__ import annotations

import pytest

from netflip import SignedInt8TwoComplementCodec
from netflip.int8_codec import INT8_MAX, INT8_MIN


@pytest.mark.parametrize(
    ("value", "encoded"),
    [
        (INT8_MIN, 0b1000_0000),
        (-127, 0b1000_0001),
        (-2, 0b1111_1110),
        (-1, 0b1111_1111),
        (0, 0b0000_0000),
        (1, 0b0000_0001),
        (2, 0b0000_0010),
        (INT8_MAX, 0b0111_1111),
    ],
)
def test_signed_int8_round_trip(value: int, encoded: int) -> None:
    codec = SignedInt8TwoComplementCodec()

    assert codec.encode(value) == encoded
    assert codec.decode(encoded) == value
    assert codec.decode(codec.encode(value)) == value


def test_quantize_and_dequantize_use_per_tensor_scale() -> None:
    codec = SignedInt8TwoComplementCodec(scale=0.25)

    assert codec.quantize(1.0) == 4
    assert codec.dequantize(-6) == -1.5


@pytest.mark.parametrize(
    ("value", "bit_index", "expected"),
    [
        (0, 0, 1),
        (1, 0, 0),
        (0, 7, -128),
        (-1, 7, 127),
        (127, 7, -1),
        (-128, 0, -127),
    ],
)
def test_bit_flips_use_lsb_first_two_complement_indexing(
    value: int,
    bit_index: int,
    expected: int,
) -> None:
    codec = SignedInt8TwoComplementCodec()

    assert codec.flip_value_bit(value, bit_index) == expected


def test_flip_bit_returns_encoded_byte() -> None:
    codec = SignedInt8TwoComplementCodec()

    assert codec.flip_bit(0b0000_0010, 1) == 0b0000_0000
    assert codec.flip_bit(0b0000_0010, 2) == 0b0000_0110


@pytest.mark.parametrize(
    ("bit_index", "mask", "role", "is_lsb", "is_msb", "is_sign_bit"),
    [
        (0, 0b0000_0001, "lsb", True, False, False),
        (3, 0b0000_1000, "value", False, False, False),
        (7, 0b1000_0000, "sign_msb", False, True, True),
    ],
)
def test_bit_metadata_marks_lsb_msb_and_sign_positions(
    bit_index: int,
    mask: int,
    role: str,
    is_lsb: bool,
    is_msb: bool,
    is_sign_bit: bool,
) -> None:
    codec = SignedInt8TwoComplementCodec()

    metadata = codec.bit_metadata(bit_index)

    assert metadata.bit_index == bit_index
    assert metadata.mask == mask
    assert metadata.role == role
    assert metadata.is_lsb is is_lsb
    assert metadata.is_msb is is_msb
    assert metadata.is_sign_bit is is_sign_bit


@pytest.mark.parametrize("value", [INT8_MIN - 1, INT8_MAX + 1])
def test_encode_rejects_values_outside_signed_int8_range(value: int) -> None:
    codec = SignedInt8TwoComplementCodec()

    with pytest.raises(ValueError, match="signed int8 range"):
        codec.encode(value)


@pytest.mark.parametrize("encoded", [-1, 256])
def test_decode_rejects_values_outside_byte_range(encoded: int) -> None:
    codec = SignedInt8TwoComplementCodec()

    with pytest.raises(ValueError, match="byte range"):
        codec.decode(encoded)


@pytest.mark.parametrize("bit_index", [-1, 8])
def test_bit_metadata_rejects_invalid_bit_indexes(bit_index: int) -> None:
    codec = SignedInt8TwoComplementCodec()

    with pytest.raises(ValueError, match="bit_index"):
        codec.bit_metadata(bit_index)


def test_scale_must_be_positive_and_finite() -> None:
    with pytest.raises(ValueError, match="scale"):
        SignedInt8TwoComplementCodec(scale=0)
