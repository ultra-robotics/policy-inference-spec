from __future__ import annotations

import numpy as np
import pytest

from policy_inference_spec.client import RemotePolicyClient, _random_warmup_wire_frame
from policy_inference_spec.hardware_model import HardwareModel
from server.minimal import EXAMPLE_POLICY_ID, example_policy_actions, run_example_server, server_handshake_config

pytestmark = pytest.mark.asyncio


async def test_client_predict_against_example_server_gen2() -> None:
    frame = _random_warmup_wire_frame(HardwareModel.GEN2)
    async with run_example_server() as url:
        client = RemotePolicyClient(url)
        async with client:
            pred = await client.predict(frame)
            assert client._server_config == {
                **server_handshake_config(),
                "image_resolution": [360, 640],
            }

    expected = example_policy_actions(frame["observation/joint_position"])
    assert pred.actions_d.shape == expected.shape
    assert pred.actions_d.dtype == np.float32
    assert np.allclose(pred.actions_d, expected)
    assert pred.policy_id == EXAMPLE_POLICY_ID

    for i in range(pred.actions_d.shape[0] - 1):
        assert np.allclose(pred.actions_d[i + 1, :], pred.actions_d[i, :]), "linear demo policy repeats each action row"
