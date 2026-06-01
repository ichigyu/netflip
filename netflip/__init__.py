"""NetFlip package."""

from importlib.metadata import PackageNotFoundError, version

from netflip.int8_codec import (
    INT8_BIT_WIDTH,
    INT8_MAX,
    INT8_MIN,
    INT8_MODULUS,
    INT8_SIGN_BIT_MASK,
    UINT8_MAX,
    BitMetadata,
    BitRole,
    SignedInt8TwoComplementCodec,
)

try:
    __version__ = version("netflip")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "INT8_BIT_WIDTH",
    "INT8_MAX",
    "INT8_MIN",
    "INT8_MODULUS",
    "INT8_SIGN_BIT_MASK",
    "UINT8_MAX",
    "BitMetadata",
    "BitRole",
    "SignedInt8TwoComplementCodec",
    "__version__",
]
