from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any

import numpy as np
import pytest
import simplejpeg
from websockets.exceptions import ConnectionClosedError
from websockets.frames import Close

from policy_inference_spec.client import (
    DEFAULT_PREDICT_URL,
    InferenceServiceRestartedError,
    RemotePolicyClient,
    policy_ws_url,
)
from policy_inference_spec.hardware_model import (
    DEFAULT_HARDWARE_MODEL,
    server_handshake_for_hardware_model,
    validate_wire_inference_request_frame,
)
from policy_inference_spec.codec import deserialize_from_msgpack, serialize_to_msgpack
from policy_inference_spec.protocol import (
    ACTION_KEY,
    ACTION_PREFIX_KEY,
    CHUNK_ID_KEY,
    CONTEXT_EMBEDDINGS_KEY,
    CONTEXT_EMBEDDING_TOKENS,
    CONTEXT_EMBEDDING_WIDTH,
    DUMB_REWARD_GOAL_ACTION_CHUNK_KEY,
    DUMB_REWARD_THRESHOLD_KEY,
    ENDPOINT_KEY,
    ENDPOINT_REWARD,
    INFERENCE_TIME_KEY,
    JOINT_STATE_KEY,
    MODEL_ID_KEY,
    POLICY_ID_KEY,
    PREFIX_CHANGE_START_KEY,
    SUBTASK_KEY,
    TASK_KEY,
    REWARD_DESCRIPTION_KEY,
    REWARDS_H_KEY,
    STATUS_KEY,
    ServerFeature,
)


def _minimal_jpeg() -> bytes:
    return simplejpeg.encode_jpeg(np.zeros((16, 16, 3), dtype=np.uint8), quality=75)


def _valid_wire_frame() -> dict[str, Any]:
    jpeg = _minimal_jpeg()
    frame: dict[str, str | np.ndarray | bytes] = {
        JOINT_STATE_KEY: np.zeros(DEFAULT_HARDWARE_MODEL.state_dim, dtype=np.float32),
        TASK_KEY: "test",
        SUBTASK_KEY: "subtask",
        MODEL_ID_KEY: "",
    }
    for camera in DEFAULT_HARDWARE_MODEL.cameras:
        frame[f"observation/{camera}"] = jpeg
    validate_wire_inference_request_frame(frame)
    return frame


def _server_handshake_payload(*, rewards_enabled: bool = False) -> dict[str, Any]:
    features = (ServerFeature.REWARDS,) if rewards_enabled else ()
    return server_handshake_for_hardware_model(
        DEFAULT_HARDWARE_MODEL,
        include_image_resolution=False,
        server_features=features,
    ).to_payload()


def _server_handshake_payload_with_resolution() -> dict[str, Any]:
    return server_handshake_for_hardware_model(
        DEFAULT_HARDWARE_MODEL,
        include_image_resolution=True,
    ).to_payload()


@pytest.mark.parametrize(
    "raw",
    [
        "ws://host/ws",
        "wss://host/ws",
        "  ws://host:18090/ws  ",
    ],
)
def test_policy_ws_url_accepts_ws_urls(raw: str) -> None:
    assert policy_ws_url(raw) == raw.strip()


def test_policy_ws_url_rejects_http() -> None:
    with pytest.raises(AssertionError):
        policy_ws_url("http://host/predict")


def test_default_predict_url_is_ws() -> None:
    assert DEFAULT_PREDICT_URL.startswith("ws://")
    assert DEFAULT_PREDICT_URL.endswith("/ws")


@pytest.mark.asyncio
async def test_predict_round_trip_with_mock_websocket() -> None:
    cfg = serialize_to_msgpack(_server_handshake_payload())
    actions = np.zeros((4, DEFAULT_HARDWARE_MODEL.action_dim), dtype=np.float32)
    context_embeddings = np.zeros((CONTEXT_EMBEDDING_TOKENS, CONTEXT_EMBEDDING_WIDTH), dtype=np.float32)
    context_embeddings[-1, -1] = 1.0
    resp = serialize_to_msgpack(
        {
            ACTION_KEY: actions,
            CONTEXT_EMBEDDINGS_KEY: context_embeddings,
            INFERENCE_TIME_KEY: 3.5,
            POLICY_ID_KEY: "policy-1",
            CHUNK_ID_KEY: "chunk-xyz",
        }
    )
    ws_mock = MagicMock()
    ws_mock.recv = AsyncMock(side_effect=[cfg, resp])
    ws_mock.send = AsyncMock()
    ws_mock.close = AsyncMock()

    async def fake_connect(*_a: object, **_kw: object) -> MagicMock:
        return ws_mock

    frame = _valid_wire_frame()
    with patch("policy_inference_spec.client.websockets.connect", side_effect=fake_connect):
        client = RemotePolicyClient("ws://127.0.0.1:9/ws")
        pred = await client.predict(frame)
        await client.aclose()

    assert pred.policy_id == "policy-1"
    assert pred.chunk_id == "chunk-xyz"
    assert pred.actions_d.shape == (4, DEFAULT_HARDWARE_MODEL.action_dim)
    assert pred.actions_d.dtype == np.float32
    assert pred.context_embeddings.shape == (CONTEXT_EMBEDDING_TOKENS, CONTEXT_EMBEDDING_WIDTH)
    assert pred.context_embeddings.dtype == np.float32
    assert pred.total_latency_ms >= 0.0
    ws_mock.send.assert_called_once()
    assert ws_mock.recv.call_count == 2


@pytest.mark.asyncio
async def test_predict_accepts_response_without_chunk_id() -> None:
    cfg = serialize_to_msgpack(_server_handshake_payload())
    actions = np.zeros((4, DEFAULT_HARDWARE_MODEL.action_dim), dtype=np.float32)
    context_embeddings = np.zeros((CONTEXT_EMBEDDING_TOKENS, CONTEXT_EMBEDDING_WIDTH), dtype=np.float32)
    resp = serialize_to_msgpack(
        {
            ACTION_KEY: actions,
            CONTEXT_EMBEDDINGS_KEY: context_embeddings,
            POLICY_ID_KEY: "policy-old",
        }
    )
    ws_mock = MagicMock()
    ws_mock.recv = AsyncMock(side_effect=[cfg, resp])
    ws_mock.send = AsyncMock()
    ws_mock.close = AsyncMock()

    async def fake_connect(*_a: object, **_kw: object) -> MagicMock:
        return ws_mock

    frame = _valid_wire_frame()
    with patch("policy_inference_spec.client.websockets.connect", side_effect=fake_connect):
        client = RemotePolicyClient("ws://127.0.0.1:9/ws")
        pred = await client.predict(frame)
        await client.aclose()

    assert pred.chunk_id is None
    assert pred.policy_id == "policy-old"


@pytest.mark.asyncio
async def test_predict_preserves_optional_dumb_reward_goal_chunk_and_threshold() -> None:
    cfg = serialize_to_msgpack(_server_handshake_payload())
    resp = serialize_to_msgpack(
        {
            ACTION_KEY: np.zeros((1, DEFAULT_HARDWARE_MODEL.action_dim), dtype=np.float32),
            CONTEXT_EMBEDDINGS_KEY: np.zeros((CONTEXT_EMBEDDING_TOKENS, CONTEXT_EMBEDDING_WIDTH), dtype=np.float32),
            POLICY_ID_KEY: "policy-1",
        }
    )
    ws_mock = MagicMock()
    ws_mock.recv = AsyncMock(side_effect=[cfg, resp])
    ws_mock.send = AsyncMock()
    ws_mock.close = AsyncMock()

    async def fake_connect(*_a: object, **_kw: object) -> MagicMock:
        return ws_mock

    frame = _valid_wire_frame()
    frame[DUMB_REWARD_GOAL_ACTION_CHUNK_KEY] = np.full((2, DEFAULT_HARDWARE_MODEL.action_dim), 0.75, dtype=np.float32)
    frame[DUMB_REWARD_THRESHOLD_KEY] = 0.5
    with patch("policy_inference_spec.client.websockets.connect", side_effect=fake_connect):
        client = RemotePolicyClient("ws://127.0.0.1:9/ws")
        await client.predict(frame)
        await client.aclose()

    await_args = ws_mock.send.await_args
    assert await_args is not None
    sent_payload = deserialize_from_msgpack(await_args.args[0])
    sent_goal_action_chunk_hd = np.asarray(sent_payload[DUMB_REWARD_GOAL_ACTION_CHUNK_KEY], dtype=np.float32)
    np.testing.assert_allclose(
        sent_goal_action_chunk_hd,
        np.full((2, DEFAULT_HARDWARE_MODEL.action_dim), 0.75, dtype=np.float32),
    )
    assert sent_payload[DUMB_REWARD_THRESHOLD_KEY] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_predict_jpeg_encodes_ndarray_images_without_resizing() -> None:
    cfg = serialize_to_msgpack(_server_handshake_payload_with_resolution())
    context_embeddings = np.zeros((CONTEXT_EMBEDDING_TOKENS, CONTEXT_EMBEDDING_WIDTH), dtype=np.float32)
    resp = serialize_to_msgpack(
        {
            ACTION_KEY: np.zeros((1, DEFAULT_HARDWARE_MODEL.action_dim), dtype=np.float32),
            CONTEXT_EMBEDDINGS_KEY: context_embeddings,
            INFERENCE_TIME_KEY: 1.0,
            POLICY_ID_KEY: "policy-1",
        }
    )
    ws_mock = MagicMock()
    ws_mock.recv = AsyncMock(side_effect=[cfg, resp])
    ws_mock.send = AsyncMock()
    ws_mock.close = AsyncMock()

    async def fake_connect(*_a: object, **_kw: object) -> MagicMock:
        return ws_mock

    frame: dict[str, Any] = {
        JOINT_STATE_KEY: np.zeros(DEFAULT_HARDWARE_MODEL.state_dim, dtype=np.float32),
        TASK_KEY: "test",
        SUBTASK_KEY: "subtask",
        MODEL_ID_KEY: "",
        "observation/images/main_image": np.zeros((23, 37, 3), dtype=np.uint8),
        "observation/images/left_wrist_image": np.zeros((19, 29, 3), dtype=np.uint8),
        "observation/images/right_wrist_image": np.zeros((17, 31, 3), dtype=np.uint8),
    }

    with patch("policy_inference_spec.client.websockets.connect", side_effect=fake_connect):
        client = RemotePolicyClient("ws://127.0.0.1:9/ws")
        await client.predict(frame)
        await client.aclose()

    await_args = ws_mock.send.await_args
    assert await_args is not None
    sent_payload = deserialize_from_msgpack(await_args.args[0])
    for key, expected_shape in (
        ("observation/images/main_image", (23, 37, 3)),
        ("observation/images/left_wrist_image", (19, 29, 3)),
        ("observation/images/right_wrist_image", (17, 31, 3)),
    ):
        value = sent_payload[key]
        assert isinstance(value, bytes), f"{key} should be JPEG bytes"
        assert simplejpeg.decode_jpeg(value).shape == expected_shape


@pytest.mark.asyncio
async def test_predict_rejects_invalid_response() -> None:
    cfg = serialize_to_msgpack(_server_handshake_payload())
    bad_actions = np.zeros((1, 7), dtype=np.float32)
    context_embeddings = np.zeros((CONTEXT_EMBEDDING_TOKENS, CONTEXT_EMBEDDING_WIDTH), dtype=np.float32)
    resp = serialize_to_msgpack(
        {
            ACTION_KEY: bad_actions,
            CONTEXT_EMBEDDINGS_KEY: context_embeddings,
            INFERENCE_TIME_KEY: 1.0,
            POLICY_ID_KEY: "",
        }
    )
    ws_mock = MagicMock()
    ws_mock.recv = AsyncMock(side_effect=[cfg, resp])
    ws_mock.send = AsyncMock()
    ws_mock.close = AsyncMock()

    async def fake_connect(*_a: object, **_kw: object) -> MagicMock:
        return ws_mock

    frame = _valid_wire_frame()
    with patch("policy_inference_spec.client.websockets.connect", side_effect=fake_connect):
        client = RemotePolicyClient("ws://127.0.0.1:9/ws")
        with pytest.raises(AssertionError):
            await client.predict(frame)
        await client.aclose()


@pytest.mark.asyncio
async def test_predict_raises_clear_restart_signal_on_service_restart() -> None:
    cfg = serialize_to_msgpack(_server_handshake_payload())
    ws_mock = MagicMock()
    ws_mock.recv = AsyncMock(side_effect=[cfg])
    ws_mock.send = AsyncMock(
        side_effect=ConnectionClosedError(
            Close(1012, "service restart"),
            Close(1012, "service restart"),
            True,
        )
    )
    ws_mock.close = AsyncMock()

    async def fake_connect(*_a: object, **_kw: object) -> MagicMock:
        return ws_mock

    frame = _valid_wire_frame()
    with patch("policy_inference_spec.client.websockets.connect", side_effect=fake_connect):
        client = RemotePolicyClient("ws://127.0.0.1:9/ws")
        with pytest.raises(InferenceServiceRestartedError, match="service restarted"):
            await client.predict(frame)

    ws_mock.close.assert_called_once()

def test_validate_wire_inference_request_frame_rejects_hardware_model_field() -> None:
    jpeg = _minimal_jpeg()
    frame: dict[str, str | np.ndarray | bytes] = {
        "hardware_model": "gen2",
        JOINT_STATE_KEY: np.zeros(DEFAULT_HARDWARE_MODEL.state_dim, dtype=np.float32),
        TASK_KEY: "test",
        SUBTASK_KEY: "subtask",
        MODEL_ID_KEY: "",
    }
    for camera in DEFAULT_HARDWARE_MODEL.cameras:
        frame[f"observation/{camera}"] = jpeg

    with pytest.raises(AssertionError, match="wire inference keys"):
        validate_wire_inference_request_frame(frame)


def test_validate_wire_inference_request_frame_accepts_unpadded_action_prefix() -> None:
    frame = _valid_wire_frame()
    frame[ACTION_PREFIX_KEY] = np.zeros((3, DEFAULT_HARDWARE_MODEL.action_dim), dtype=np.float32)
    frame[PREFIX_CHANGE_START_KEY] = 7

    validate_wire_inference_request_frame(frame)


def test_validate_wire_inference_request_frame_requires_action_prefix_pair() -> None:
    frame = _valid_wire_frame()
    frame[ACTION_PREFIX_KEY] = np.zeros((3, DEFAULT_HARDWARE_MODEL.action_dim), dtype=np.float32)

    with pytest.raises(AssertionError, match=PREFIX_CHANGE_START_KEY):
        validate_wire_inference_request_frame(frame)


def test_validate_wire_inference_request_frame_rejects_bad_action_prefix_shape() -> None:
    frame = _valid_wire_frame()
    frame[ACTION_PREFIX_KEY] = np.zeros((3, DEFAULT_HARDWARE_MODEL.action_dim + 1), dtype=np.float32)
    frame[PREFIX_CHANGE_START_KEY] = 3

    with pytest.raises(AssertionError, match=ACTION_PREFIX_KEY):
        validate_wire_inference_request_frame(frame)


def test_validate_wire_inference_request_frame_rejects_nonfloating_action_prefix() -> None:
    frame = _valid_wire_frame()
    frame[ACTION_PREFIX_KEY] = np.zeros((3, DEFAULT_HARDWARE_MODEL.action_dim), dtype=np.int32)
    frame[PREFIX_CHANGE_START_KEY] = 3

    with pytest.raises(AssertionError, match=ACTION_PREFIX_KEY):
        validate_wire_inference_request_frame(frame)


@pytest.mark.asyncio
async def test_reward_sends_default_value_when_server_supports_rewards() -> None:
    cfg = serialize_to_msgpack(_server_handshake_payload(rewards_enabled=True))
    reward_ack = serialize_to_msgpack({ENDPOINT_KEY: ENDPOINT_REWARD, STATUS_KEY: "ok", REWARDS_H_KEY: [1.0]})
    ws_mock = MagicMock()
    ws_mock.recv = AsyncMock(side_effect=[cfg, reward_ack])
    ws_mock.send = AsyncMock()
    ws_mock.close = AsyncMock()

    async def fake_connect(*_a: object, **_kw: object) -> MagicMock:
        return ws_mock

    with patch("policy_inference_spec.client.websockets.connect", side_effect=fake_connect):
        client = RemotePolicyClient("ws://127.0.0.1:9/ws")
        await client.reward(chunk_id="chunk-1")
        await client.aclose()

    ws_mock.send.assert_called_once()
    await_args = ws_mock.send.await_args
    assert await_args is not None
    sent_payload = deserialize_from_msgpack(await_args.args[0])
    assert sent_payload == {ENDPOINT_KEY: ENDPOINT_REWARD, CHUNK_ID_KEY: "chunk-1", REWARDS_H_KEY: [1.0]}


@pytest.mark.asyncio
async def test_reward_includes_description_only_when_provided() -> None:
    cfg = serialize_to_msgpack(_server_handshake_payload(rewards_enabled=True))
    reward_ack = serialize_to_msgpack(
        {
            ENDPOINT_KEY: ENDPOINT_REWARD,
            STATUS_KEY: "ok",
            REWARDS_H_KEY: [2.5],
            REWARD_DESCRIPTION_KEY: "The box was successfully sealed",
        }
    )
    ws_mock = MagicMock()
    ws_mock.recv = AsyncMock(side_effect=[cfg, reward_ack])
    ws_mock.send = AsyncMock()
    ws_mock.close = AsyncMock()

    async def fake_connect(*_a: object, **_kw: object) -> MagicMock:
        return ws_mock

    with patch("policy_inference_spec.client.websockets.connect", side_effect=fake_connect):
        client = RemotePolicyClient("ws://127.0.0.1:9/ws")
        await client.reward([2.5], "The box was successfully sealed", chunk_id="chunk-2")
        await client.aclose()

    await_args = ws_mock.send.await_args
    assert await_args is not None
    sent_payload = deserialize_from_msgpack(await_args.args[0])
    assert sent_payload == {
        ENDPOINT_KEY: ENDPOINT_REWARD,
        CHUNK_ID_KEY: "chunk-2",
        REWARDS_H_KEY: [2.5],
        REWARD_DESCRIPTION_KEY: "The box was successfully sealed",
    }


@pytest.mark.asyncio
async def test_reward_drops_and_warns_when_server_lacks_reward_support(caplog: pytest.LogCaptureFixture) -> None:
    cfg = serialize_to_msgpack(_server_handshake_payload(rewards_enabled=False))
    ws_mock = MagicMock()
    ws_mock.recv = AsyncMock(side_effect=[cfg])
    ws_mock.send = AsyncMock()
    ws_mock.close = AsyncMock()

    async def fake_connect(*_a: object, **_kw: object) -> MagicMock:
        return ws_mock

    with patch("policy_inference_spec.client.websockets.connect", side_effect=fake_connect):
        client = RemotePolicyClient("ws://127.0.0.1:9/ws")
        with caplog.at_level("WARNING", logger="policy_inference_spec.client"):
            await client.reward([3.0], "ignored", chunk_id="chunk-3")
        await client.aclose()

    ws_mock.send.assert_not_called()
    assert "Dropping reward because server does not advertise rewards support" in caplog.text
