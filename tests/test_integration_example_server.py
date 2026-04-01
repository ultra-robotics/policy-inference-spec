from __future__ import annotations

import numpy as np
import pytest
import simplejpeg  # type: ignore[import-untyped]
from typing import Any, cast

from policy_inference_spec.client import RemotePolicyClient
from policy_inference_spec.hardware_model import DEFAULT_HARDWARE_MODEL
from policy_inference_spec.protocol import JOINT_STATE_KEY, MODEL_ID_KEY, PROMPT_KEY, ServerFeature
from server.minimal import EXAMPLE_POLICY_ID, example_policy_actions, run_example_server, server_handshake_config

pytestmark = pytest.mark.asyncio


def _random_predict_frame() -> dict[str, object]:
    rng = np.random.default_rng()
    height, width = DEFAULT_HARDWARE_MODEL.image_resolution
    frame: dict[str, object] = {
        JOINT_STATE_KEY: rng.standard_normal(DEFAULT_HARDWARE_MODEL.state_dim, dtype=np.float32),
        PROMPT_KEY: "",
        MODEL_ID_KEY: "",
    }
    for cam in DEFAULT_HARDWARE_MODEL.cameras:
        rgb = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        frame[f"observation/{cam}"] = simplejpeg.encode_jpeg(rgb, quality=75)
    return frame


async def test_client_predict_against_example_server() -> None:
    frame = _random_predict_frame()
    async with run_example_server() as url:
        client = RemotePolicyClient(url)
        async with client:
            pred = await client.predict(frame)
            assert client._server_config == server_handshake_config(server_features=(ServerFeature.REWARDS,))

    expected = example_policy_actions(cast(np.ndarray[Any, Any], frame[JOINT_STATE_KEY]))
    assert pred.actions_d.shape == expected.shape
    assert pred.actions_d.dtype == np.float32
    assert np.allclose(pred.actions_d, expected)
    assert pred.policy_id == EXAMPLE_POLICY_ID

    for i in range(pred.actions_d.shape[0] - 1):
        assert np.allclose(pred.actions_d[i + 1, :], pred.actions_d[i, :]), "linear demo policy repeats each action row"


async def test_client_reward_round_trip_against_example_server() -> None:
    async with run_example_server() as url:
        client = RemotePolicyClient(url)
        async with client:
            await client.reward(1.5, "The box was successfully sealed")
            assert client._server_config == server_handshake_config(server_features=(ServerFeature.REWARDS,))


async def test_client_reward_is_dropped_when_example_server_does_not_advertise_rewards(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async with run_example_server(server_features=()) as url:
        client = RemotePolicyClient(url)
        async with client:
            with caplog.at_level("WARNING", logger="policy_inference_spec.client"):
                await client.reward(1.5, "ignored")

    assert "Dropping reward because server does not advertise rewards support" in caplog.text
