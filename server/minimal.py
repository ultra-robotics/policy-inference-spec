from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import numpy as np
import torch
import websockets
from websockets.asyncio.server import ServerConnection

from policy_inference_spec.constants import (
    ACTIONS_KEY,
    ENDPOINT_KEY,
    ENDPOINT_RESET,
    ENDPOINT_TELEMETRY,
    INFERENCE_TIME_KEY,
    MODEL_ID_KEY,
    OBS_JOINT_POSITION_KEY,
    PROMPT_KEY,
)
from policy_inference_spec.protocol import deserialize_from_msgpack, serialize_to_msgpack
from policy_inference_spec.hardware_model import (
    DEFAULT_HARDWARE_MODEL,
    validate_wire_inference_request_frame,
    validate_wire_inference_response,
)

DEFAULT_ACTION_HORIZON = 4
EXAMPLE_POLICY_ID = "example-linear"
EXAMPLE_IMAGE_RESOLUTION = DEFAULT_HARDWARE_MODEL.image_resolution
EXAMPLE_ACTION_DIM = DEFAULT_HARDWARE_MODEL.action_dim


class ExampleLinearPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(DEFAULT_HARDWARE_MODEL.state_dim, EXAMPLE_ACTION_DIM)
        with torch.no_grad():
            weight = torch.arange(
                EXAMPLE_ACTION_DIM * DEFAULT_HARDWARE_MODEL.state_dim,
                dtype=torch.float32,
            ).reshape(EXAMPLE_ACTION_DIM, DEFAULT_HARDWARE_MODEL.state_dim)
            self.linear.weight.copy_(weight / 1000.0)
            self.linear.bias.copy_(torch.arange(EXAMPLE_ACTION_DIM, dtype=torch.float32) / 100.0)

    def forward(self, joint_position: torch.Tensor) -> torch.Tensor:
        return self.linear(joint_position)


EXAMPLE_POLICY = ExampleLinearPolicy()


def server_handshake_config() -> dict[str, Any]:
    return {
        "camera_names": list(DEFAULT_HARDWARE_MODEL.cameras),
        "image_resolution": EXAMPLE_IMAGE_RESOLUTION,
        "action_space": "joint_position",
        "needs_wrist_camera": True,
        "n_external_cameras": 1,
    }


def example_policy_actions(
    joint_position: np.ndarray,
    *,
    horizon: int = DEFAULT_ACTION_HORIZON,
) -> np.ndarray:
    assert joint_position.shape == (DEFAULT_HARDWARE_MODEL.state_dim,), (
        f"joint_position must have shape ({DEFAULT_HARDWARE_MODEL.state_dim},), got {joint_position.shape}"
    )
    with torch.no_grad():
        joint_tensor = torch.from_numpy(joint_position.astype(np.float32, copy=True))
        action = EXAMPLE_POLICY(joint_tensor).cpu().numpy().astype(np.float32, copy=False)
    return np.repeat(action[None, :], horizon, axis=0)


def _inference_response(frame: dict[str, Any]) -> dict[str, Any]:
    joint_position = frame[OBS_JOINT_POSITION_KEY]
    assert isinstance(joint_position, np.ndarray), type(joint_position)
    actions = example_policy_actions(joint_position)
    resp = {
        ACTIONS_KEY: actions,
        INFERENCE_TIME_KEY: 0.25,
        "policy_id": EXAMPLE_POLICY_ID,
    }
    validate_wire_inference_response(resp)
    return resp


async def handle_inference_connection(connection: ServerConnection) -> None:
    cfg = server_handshake_config()
    await connection.send(serialize_to_msgpack(cfg))
    async for message in connection:
        assert isinstance(message, bytes), type(message)
        frame = deserialize_from_msgpack(message)
        if not isinstance(frame, dict):
            await connection.send(serialize_to_msgpack({"error": "expected dict frame"}))
            continue
        if frame.get(ENDPOINT_KEY) == ENDPOINT_RESET:
            await connection.send(serialize_to_msgpack({"status": "ok"}))
            continue
        if frame.get(ENDPOINT_KEY) == ENDPOINT_TELEMETRY:
            await connection.send(serialize_to_msgpack({"status": "ok"}))
            continue
        validate_wire_inference_request_frame(frame)
        _ = frame[PROMPT_KEY]
        _ = frame[MODEL_ID_KEY]
        resp = _inference_response(frame)
        await connection.send(serialize_to_msgpack(resp))


@asynccontextmanager
async def run_example_server() -> AsyncIterator[str]:
    async def handler(connection: ServerConnection) -> None:
        await handle_inference_connection(connection)

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        sock = next(iter(server.sockets))
        port = sock.getsockname()[1]
        yield f"ws://127.0.0.1:{port}/ws"
