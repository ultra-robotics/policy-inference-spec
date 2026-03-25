from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import numpy as np
import websockets
from websockets.asyncio.server import ServerConnection

from policy_inference_spec.hardware_model import HardwareModel
from policy_inference_spec.protocol import msgpack_decode, msgpack_encode
from policy_inference_spec.schema import (
    ENDPOINT_RESET,
    ENDPOINT_TELEMETRY,
    GEN2_GATEWAY_CAMERAS,
    GEN2_SHAPES,
    KEY_ACTIONS,
    KEY_ENDPOINT,
    KEY_INFERENCE_TIME,
    KEY_MODEL_ID,
    KEY_PROMPT,
    validate_wire_inference_request_frame,
    validate_wire_inference_response,
)

DEFAULT_ACTION_HORIZON = 4
EXAMPLE_POLICY_ID = "example-dummy"


def server_handshake_config() -> dict[str, Any]:
    h, w = int(GEN2_SHAPES["observation.images.head"][2]), int(GEN2_SHAPES["observation.images.head"][3])
    return {
        "camera_names": list(GEN2_GATEWAY_CAMERAS),
        "image_resolution": (h, w),
        "action_space": "joint_position",
        "needs_wrist_camera": True,
        "n_external_cameras": 1,
    }


def structured_dummy_actions(*, horizon: int = DEFAULT_ACTION_HORIZON) -> np.ndarray:
    action_dim = 25
    stripe = np.arange(action_dim, dtype=np.float32) % 3.0
    time_axis = np.arange(horizon, dtype=np.float32)[:, None] * 10.0
    return (time_axis + stripe[None, :]).astype(np.float32)


def _inference_response() -> dict[str, Any]:
    actions = structured_dummy_actions()
    resp = {
        KEY_ACTIONS: actions,
        KEY_INFERENCE_TIME: 0.25,
        "policy_id": EXAMPLE_POLICY_ID,
    }
    validate_wire_inference_response(resp)
    return resp


async def handle_inference_connection(connection: ServerConnection) -> None:
    cfg = server_handshake_config()
    await connection.send(msgpack_encode(cfg))
    async for message in connection:
        assert isinstance(message, bytes), type(message)
        frame = msgpack_decode(message)
        if not isinstance(frame, dict):
            await connection.send(msgpack_encode({"error": "expected dict frame"}))
            continue
        if frame.get(KEY_ENDPOINT) == ENDPOINT_RESET:
            await connection.send(msgpack_encode({"status": "ok"}))
            continue
        if frame.get(KEY_ENDPOINT) == ENDPOINT_TELEMETRY:
            await connection.send(msgpack_encode({"status": "ok"}))
            continue
        hm = validate_wire_inference_request_frame(frame)
        assert hm == HardwareModel.GEN2, f"example server is gen2-only, got {hm.value}"
        _ = frame[KEY_PROMPT]
        _ = frame[KEY_MODEL_ID]
        resp = _inference_response()
        await connection.send(msgpack_encode(resp))


@asynccontextmanager
async def run_example_server() -> AsyncIterator[str]:
    async def handler(connection: ServerConnection) -> None:
        await handle_inference_connection(connection)

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        sock = next(iter(server.sockets))
        port = sock.getsockname()[1]
        yield f"ws://127.0.0.1:{port}/ws"
