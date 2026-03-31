from __future__ import annotations

import pytest

from policy_inference_spec.hardware_model import HardwareModel


def test_str_enum_values() -> None:
    assert HardwareModel.GEN1.value == "gen1"
    assert HardwareModel.GEN2.value == "gen2"


def test_constructor_accepts_enum() -> None:
    assert HardwareModel(HardwareModel.GEN2) == HardwareModel.GEN2


def test_constructor_parses_string() -> None:
    assert HardwareModel(" gen1 ") == HardwareModel.GEN1


def test_constructor_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        HardwareModel("gen3")
