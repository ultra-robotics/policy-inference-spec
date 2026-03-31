from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any

import numpy as np
import pytest
import simplejpeg
from websockets.exceptions import ConnectionClosedError
from websockets.frames import Close

from policy_inference_spec.constants import (
    ACTIONS_KEY,
    INFERENCE_TIME_KEY,
    MODEL_ID_KEY,
    OBS_JOINT_POSITION_KEY,
    PROMPT_KEY,
)
from policy_inference_spec.client import (
    DEFAULT_PREDICT_URL,
    InferenceServiceRestartedError,
    RemotePolicyClient,
    policy_ws_url,
)
from policy_inference_spec.protocol import msgpack_encode
from policy_inference_spec.hardware_model import (
    DEFAULT_HARDWARE_MODEL,
    HardwareModel,
    validate_wire_inference_request_frame,
    wire_joint_position_array,
)


def _minimal_jpeg() -> bytes:
    return simplejpeg.encode_jpeg(np.zeros((16, 16, 3), dtype=np.uint8), quality=75)


def _valid_wire_frame() -> dict[str, Any]:
    jpeg = _minimal_jpeg()
    frame = {
        OBS_JOINT_POSITION_KEY: np.zeros(DEFAULT_HARDWARE_MODEL.state_dim, dtype=np.float32),
        PROMPT_KEY: "test",
        MODEL_ID_KEY: "",
    }
    for camera in DEFAULT_HARDWARE_MODEL.cameras:
        frame[f"observation/{camera}"] = jpeg
    validate_wire_inference_request_frame(frame)
    return frame


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
    cfg = msgpack_encode({"camera_names": [DEFAULT_HARDWARE_MODEL.cameras[0]]})
    actions = np.zeros((4, DEFAULT_HARDWARE_MODEL.action_dim), dtype=np.float32)
    resp = msgpack_encode(
        {
            ACTIONS_KEY: actions,
            INFERENCE_TIME_KEY: 3.5,
            "policy_id": "policy-1",
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
    assert pred.actions_d.shape == (4, DEFAULT_HARDWARE_MODEL.action_dim)
    assert pred.actions_d.dtype == np.float32
    assert pred.total_latency_ms >= 0.0
    ws_mock.send.assert_called_once()
    assert ws_mock.recv.call_count == 2


@pytest.mark.asyncio
async def test_predict_rejects_invalid_response() -> None:
    cfg = msgpack_encode({})
    bad_actions = np.zeros((1, 7), dtype=np.float32)
    resp = msgpack_encode({ACTIONS_KEY: bad_actions, INFERENCE_TIME_KEY: 1.0, "policy_id": ""})
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
    cfg = msgpack_encode({"camera_names": [DEFAULT_HARDWARE_MODEL.cameras[0]]})
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


def test_warmup_swallows_connection_errors() -> None:
    with patch("policy_inference_spec.client.ws_connect_sync") as m:
        m.side_effect = OSError("no server")
        client = RemotePolicyClient("ws://127.0.0.1:9/ws")
        assert client.warmup() is False


def test_validate_wire_inference_request_frame_rejects_hardware_model_field() -> None:
    jpeg = _minimal_jpeg()
    frame = {
        "hardware_model": "gen2",
        OBS_JOINT_POSITION_KEY: np.zeros(DEFAULT_HARDWARE_MODEL.state_dim, dtype=np.float32),
        PROMPT_KEY: "test",
        MODEL_ID_KEY: "",
    }
    for camera in DEFAULT_HARDWARE_MODEL.cameras:
        frame[f"observation/{camera}"] = jpeg

    with pytest.raises(AssertionError, match="wire inference keys"):
        validate_wire_inference_request_frame(frame)


def test_wire_joint_position_array_accepts_hardware_model_string() -> None:
    joint = wire_joint_position_array(np.zeros(DEFAULT_HARDWARE_MODEL.state_dim, dtype=np.float32), "gen2")
    assert joint.shape == (DEFAULT_HARDWARE_MODEL.state_dim,)


def test_wire_joint_position_array_accepts_explicit_hardware_model_enum() -> None:
    joint = wire_joint_position_array(np.zeros(DEFAULT_HARDWARE_MODEL.state_dim, dtype=np.float32), HardwareModel.GEN2)
    assert joint.shape == (DEFAULT_HARDWARE_MODEL.state_dim,)


def test_wire_joint_position_array_rejects_invalid_hardware_model() -> None:
    with pytest.raises(ValueError):
        wire_joint_position_array(np.zeros(DEFAULT_HARDWARE_MODEL.state_dim, dtype=np.float32), "gen3")
