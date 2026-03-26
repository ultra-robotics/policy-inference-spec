from __future__ import annotations

import msgspec
import numpy as np
import pytest

from policy_inference_spec.protocol import msgpack_decode, msgpack_encode
from policy_inference_spec.schema import validate_wire_inference_response


def test_msgpack_encode_uses_flat_ndarray_tags() -> None:
    expected = np.arange(6, dtype=np.float64).reshape(2, 3)
    payload = {"actions": expected}
    encoded = msgpack_encode(payload)
    raw = msgspec.msgpack.decode(encoded)

    assert raw["actions"]["__ndarray__"] is True
    assert raw["actions"]["dtype"] == "float64"
    assert raw["actions"]["shape"] == [2, 3]
    assert isinstance(raw["actions"]["data"], bytes)


def test_msgpack_decode_accepts_flat_byte_key_ndarray_tags() -> None:
    expected = np.arange(6, dtype=np.float64).reshape(2, 3)
    payload = {
        "actions": {
            b"__ndarray__": True,
            b"data": expected.tobytes(),
            b"dtype": "float64",
            b"shape": [2, 3],
        }
    }

    decoded = msgpack_decode(msgspec.msgpack.encode(payload))

    assert isinstance(decoded["actions"], np.ndarray)
    assert decoded["actions"].dtype == np.float64
    assert decoded["actions"].shape == (2, 3)
    assert np.array_equal(decoded["actions"], expected)


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
