from __future__ import annotations

import numpy as np
import pytest

from policy_inference_spec.client import RemotePolicyClient, _random_warmup_wire_frame
from policy_inference_spec.hardware_model import HardwareModel
from server.minimal import EXAMPLE_POLICY_ID, run_example_server, structured_dummy_actions

pytestmark = pytest.mark.asyncio


async def test_client_predict_against_example_server_gen2() -> None:
    async with run_example_server() as url:
        client = RemotePolicyClient(url)
        frame = _random_warmup_wire_frame(HardwareModel.GEN2)
        async with client:
            pred = await client.predict(frame)

    expected = structured_dummy_actions()
    assert pred.actions_d.shape == expected.shape
    assert pred.actions_d.dtype == np.float32
    assert np.allclose(pred.actions_d, expected)
    assert pred.policy_id == EXAMPLE_POLICY_ID

    for i in range(pred.actions_d.shape[0] - 1):
        assert np.all(pred.actions_d[i + 1, :] > pred.actions_d[i, :]), "time axis should increase row-wise"

    row0 = pred.actions_d[0]
    assert np.allclose(row0[:3], np.array([0.0, 1.0, 2.0], dtype=np.float32)), "stripe pattern along action columns"
