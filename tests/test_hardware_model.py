from __future__ import annotations

import pytest

from policy_inference_spec.hardware_model import HardwareModel, as_hardware_model


def test_str_enum_values() -> None:
    assert HardwareModel.GEN1.value == "gen1"
    assert HardwareModel.GEN2.value == "gen2"


def test_as_hardware_model_accepts_enum() -> None:
    assert as_hardware_model(HardwareModel.GEN2) == HardwareModel.GEN2


def test_as_hardware_model_parses_string() -> None:
    assert as_hardware_model(" gen1 ") == HardwareModel.GEN1


def test_as_hardware_model_rejects_invalid() -> None:
    with pytest.raises(AssertionError):
        as_hardware_model("gen3")
