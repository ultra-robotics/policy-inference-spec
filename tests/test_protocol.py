from __future__ import annotations

from typing import Any, cast

import msgspec
import numpy as np
import pytest
import simplejpeg
from beartype.roar import BeartypeCallHintParamViolation

from policy_inference_spec.codec import NdarrayField, deserialize_from_msgpack, encode_image, serialize_to_msgpack
from policy_inference_spec.protocol import (
    ACTION_KEY,
    CONTEXT_EMBEDDINGS_KEY,
    CONTEXT_EMBEDDING_TOKENS,
    CONTEXT_EMBEDDING_WIDTH,
    ENDPOINT_KEY,
    ENDPOINT_REWARD,
    FloatArray,
    ProtocolPayload,
    REWARD_DESCRIPTION_KEY,
    REWARD_KEY,
    RewardSignal,
    ServerFeature,
    ServerHandshake,
)
from policy_inference_spec.hardware_model import validate_wire_inference_response


def test_serialize_to_msgpack_uses_flat_ndarray_tags() -> None:
    expected = cast(FloatArray, np.arange(6, dtype=np.float32).reshape(2, 3))
    payload: ProtocolPayload = {"action": expected}
    encoded = serialize_to_msgpack(payload)
    raw = msgspec.msgpack.decode(encoded)

    assert raw["action"]["__ndarray__"] is True
    assert raw["action"]["dtype"] == "float32"
    assert raw["action"]["shape"] == [2, 3]
    assert isinstance(raw["action"]["data"], bytes)


def test_deserialize_from_msgpack_accepts_flat_byte_key_ndarray_tags() -> None:
    expected = cast(FloatArray, np.arange(6, dtype=np.float32).reshape(2, 3))
    payload = {
        "action": {
            b"__ndarray__": True,
            b"data": expected.tobytes(),
            b"dtype": "float32",
            b"shape": [2, 3],
        }
    }

    decoded = deserialize_from_msgpack(msgspec.msgpack.encode(payload))

    assert isinstance(decoded["action"], np.ndarray)
    assert decoded["action"].dtype == np.float32
    assert decoded["action"].shape == (2, 3)
    assert np.array_equal(decoded["action"], expected)


def test_serialize_to_msgpack_rejects_float64_ndarray() -> None:
    payload: Any = {"action": np.arange(6, dtype=np.float64).reshape(2, 3)}

    with pytest.raises(BeartypeCallHintParamViolation):
        serialize_to_msgpack(cast(Any, payload))


def test_deserialize_from_msgpack_rejects_float64_ndarray_tag() -> None:
    payload = {
        "action": {
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
                ACTION_KEY: {
                    "__ndarray__": True,
                    "data": b"\x00" * 32,
                    "dtype": "float64",
                    "shape": [2, 2],
                },
                CONTEXT_EMBEDDINGS_KEY: np.zeros((CONTEXT_EMBEDDING_TOKENS, CONTEXT_EMBEDDING_WIDTH), dtype=np.float32),
            }
        )

    message = str(exc_info.value)
    assert "summary=" in message
    assert "dict(keys=['__ndarray__', 'data', 'dtype', 'shape'])" in message
    assert "\\x00" not in message


def test_validate_wire_inference_response_accepts_context_embeddings() -> None:
    validate_wire_inference_response(
        {
            ACTION_KEY: np.zeros((2, 25), dtype=np.float32),
            CONTEXT_EMBEDDINGS_KEY: np.zeros(
                (CONTEXT_EMBEDDING_TOKENS, CONTEXT_EMBEDDING_WIDTH),
                dtype=np.float32,
            ),
        }
    )


def test_encode_image_preserves_original_shape_metadata() -> None:
    expected = np.zeros((1, 12, 16, 3), dtype=np.uint8)

    encoded = encode_image(expected, jpeg_quality=75)
    decoded = simplejpeg.decode_jpeg(encoded.data)

    assert isinstance(encoded, NdarrayField)
    assert encoded.codec == "jpeg"
    assert encoded.shape == expected.shape
    assert encoded.dtype == str(expected.dtype)
    assert decoded.shape == (12, 16, 3)
    assert decoded.dtype == expected.dtype


def test_server_handshake_round_trip_preserves_server_features() -> None:
    handshake = ServerHandshake(
        camera_names=("images/main_image", "images/left_wrist_image"),
        image_resolution=(360, 640),
        server_features=(ServerFeature.REWARDS.value,),
    )

    decoded = ServerHandshake.from_payload(handshake.to_payload())

    assert decoded == handshake
    assert decoded.supports(ServerFeature.REWARDS)


def test_reward_signal_round_trip_with_optional_description() -> None:
    reward_signal = RewardSignal(1.5, "The box was successfully sealed")

    assert RewardSignal.from_payload(reward_signal.to_payload()) == reward_signal
    assert reward_signal.to_payload() == {
        ENDPOINT_KEY: ENDPOINT_REWARD,
        REWARD_KEY: 1.5,
        REWARD_DESCRIPTION_KEY: "The box was successfully sealed",
    }
