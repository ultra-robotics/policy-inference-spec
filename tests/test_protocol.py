from __future__ import annotations

from typing import Any, cast

import msgspec
import numpy as np
import pytest
import simplejpeg
from beartype.roar import BeartypeCallHintParamViolation

from policy_inference_spec.protocol import (
    FloatArray,
    NdarrayField,
    ProtocolPayload,
    deserialize_from_msgpack,
    encode_image,
    serialize_to_msgpack,
)
from policy_inference_spec.hardware_model import validate_wire_inference_response


def test_serialize_to_msgpack_uses_flat_ndarray_tags() -> None:
    expected = cast(FloatArray, np.arange(6, dtype=np.float32).reshape(2, 3))
    payload: ProtocolPayload = {"actions": expected}
    encoded = serialize_to_msgpack(payload)
    raw = msgspec.msgpack.decode(encoded)

    assert raw["actions"]["__ndarray__"] is True
    assert raw["actions"]["dtype"] == "float32"
    assert raw["actions"]["shape"] == [2, 3]
    assert isinstance(raw["actions"]["data"], bytes)


def test_deserialize_from_msgpack_accepts_flat_byte_key_ndarray_tags() -> None:
    expected = cast(FloatArray, np.arange(6, dtype=np.float32).reshape(2, 3))
    payload = {
        "actions": {
            b"__ndarray__": True,
            b"data": expected.tobytes(),
            b"dtype": "float32",
            b"shape": [2, 3],
        }
    }

    decoded = deserialize_from_msgpack(msgspec.msgpack.encode(payload))

    assert isinstance(decoded["actions"], np.ndarray)
    assert decoded["actions"].dtype == np.float32
    assert decoded["actions"].shape == (2, 3)
    assert np.array_equal(decoded["actions"], expected)


def test_serialize_to_msgpack_rejects_float64_ndarray() -> None:
    payload: Any = {"actions": np.arange(6, dtype=np.float64).reshape(2, 3)}

    with pytest.raises(BeartypeCallHintParamViolation):
        serialize_to_msgpack(cast(Any, payload))


def test_deserialize_from_msgpack_rejects_float64_ndarray_tag() -> None:
    payload = {
        "actions": {
            b"__ndarray__": True,
            b"data": np.arange(6, dtype=np.float64).reshape(2, 3).tobytes(),
            b"dtype": "float64",
            b"shape": [2, 3],
        }
    }

    with pytest.raises(AssertionError, match="Unsupported ndarray dtype"):
        deserialize_from_msgpack(msgspec.msgpack.encode(payload))


def test_validate_wire_inference_response_summarizes_binary_like_payloads() -> None:
    with pytest.raises(AssertionError) as exc_info:
        validate_wire_inference_response(
            {
                "actions": {
                    "__ndarray__": True,
                    "data": b"\x00" * 32,
                    "dtype": "float64",
                    "shape": [2, 2],
                }
            }
        )

    message = str(exc_info.value)
    assert "summary=" in message
    assert "dict(keys=['__ndarray__', 'data', 'dtype', 'shape'])" in message
    assert "\\x00" not in message


def test_encode_image_preserves_original_shape_metadata() -> None:
    expected = np.zeros((1, 12, 16, 3), dtype=np.uint8)

    encoded = encode_image(expected, height=12, width=16, jpeg_quality=75)
    decoded = simplejpeg.decode_jpeg(encoded.data)

    assert isinstance(encoded, NdarrayField)
    assert encoded.codec == "jpeg"
    assert encoded.shape == expected.shape
    assert encoded.dtype == str(expected.dtype)
    assert decoded.shape == (12, 16, 3)
    assert decoded.dtype == expected.dtype
