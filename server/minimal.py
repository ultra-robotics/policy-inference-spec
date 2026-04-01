from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Iterable

import numpy as np
import torch
import websockets
from websockets.asyncio.server import ServerConnection

from policy_inference_spec.codec import deserialize_from_msgpack, serialize_to_msgpack
from policy_inference_spec.hardware_model import (
    DEFAULT_HARDWARE_MODEL,
    server_handshake_for_hardware_model,
    validate_wire_inference_request_frame,
    validate_wire_inference_response,
)
from policy_inference_spec.protocol import (
    ACTION_KEY,
    ENDPOINT_KEY,
    ENDPOINT_RESET,
    ENDPOINT_REWARD,
    ENDPOINT_TELEMETRY,
    INFERENCE_TIME_KEY,
    JOINT_STATE_KEY,
    MODEL_ID_KEY,
    POLICY_ID_KEY,
    PROMPT_KEY,
    REWARD_DESCRIPTION_KEY,
    REWARD_KEY,
    STATUS_KEY,
    RewardSignal,
    ServerFeature,
    ServerHandshake,
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


def server_handshake_config(
    *, server_features: Iterable[str | ServerFeature] = ()
) -> ServerHandshake:
    return server_handshake_for_hardware_model(
        DEFAULT_HARDWARE_MODEL,
        include_image_resolution=True,
        server_features=server_features,
    )


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
    joint_position = frame[JOINT_STATE_KEY]
    assert isinstance(joint_position, np.ndarray), type(joint_position)
    actions = example_policy_actions(joint_position)
    resp = {
        ACTION_KEY: actions,
        INFERENCE_TIME_KEY: 0.25,
        POLICY_ID_KEY: EXAMPLE_POLICY_ID,
    }
    validate_wire_inference_response(resp)
    return resp


async def handle_inference_connection(
    connection: ServerConnection,
    *,
    server_features: Iterable[str | ServerFeature] = (),
) -> None:
    cfg = server_handshake_config(server_features=server_features)
    await connection.send(serialize_to_msgpack(cfg.to_payload()))
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
        if frame.get(ENDPOINT_KEY) == ENDPOINT_REWARD:
            reward_signal = RewardSignal.from_payload(frame)
            await connection.send(
                serialize_to_msgpack(
                    {
                        ENDPOINT_KEY: ENDPOINT_REWARD,
                        STATUS_KEY: "ok",
                        REWARD_KEY: reward_signal.reward,
                        **(
                            {REWARD_DESCRIPTION_KEY: reward_signal.description}
                            if reward_signal.description is not None
                            else {}
                        ),
                    }
                )
            )
            continue
        validate_wire_inference_request_frame(frame)
        _ = frame[PROMPT_KEY]
        _ = frame[MODEL_ID_KEY]
        resp = _inference_response(frame)
        await connection.send(serialize_to_msgpack(resp))


@asynccontextmanager
async def run_example_server(
    *, server_features: Iterable[str | ServerFeature] = (ServerFeature.REWARDS,)
) -> AsyncIterator[str]:
    async def handler(connection: ServerConnection) -> None:
        await handle_inference_connection(connection, server_features=server_features)

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        sock = next(iter(server.sockets))
        port = sock.getsockname()[1]
        yield f"ws://127.0.0.1:{port}/ws"
