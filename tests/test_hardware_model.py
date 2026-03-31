from __future__ import annotations

import pytest

from policy_inference_spec.schema import DEFAULT_HARDWARE_MODEL, HardwareModel


def test_str_enum_values() -> None:
    assert HardwareModel.GEN2.value == "gen2"


def test_constructor_accepts_enum() -> None:
    assert HardwareModel(HardwareModel.GEN2) == HardwareModel.GEN2


def test_constructor_parses_string() -> None:
    assert HardwareModel(" gen2 ") == HardwareModel.GEN2


def test_hardware_model_properties() -> None:
    assert DEFAULT_HARDWARE_MODEL.state_dim == 97
    assert DEFAULT_HARDWARE_MODEL.action_dim == 25
    assert DEFAULT_HARDWARE_MODEL.image_resolution == (360, 640)
    assert DEFAULT_HARDWARE_MODEL.gateway_cameras == (
        "images/main_image_left",
        "images/left_wrist_image_left",
        "images/right_wrist_image_left",
    )
    assert dict(DEFAULT_HARDWARE_MODEL.ultra_to_gateway_image) == {
        "observation.images.head": "images/main_image_left",
        "observation.images.left_wrist": "images/left_wrist_image_left",
        "observation.images.right_wrist": "images/right_wrist_image_left",
    }
    assert dict(DEFAULT_HARDWARE_MODEL.gateway_to_ultra_image) == {
        "images/main_image_left": "observation.images.head",
        "images/left_wrist_image_left": "observation.images.left_wrist",
        "images/right_wrist_image_left": "observation.images.right_wrist",
    }


def test_hardware_model_image_maps_are_defensive_copies() -> None:
    ultra_to_gateway = DEFAULT_HARDWARE_MODEL.ultra_to_gateway_image
    gateway_to_ultra = DEFAULT_HARDWARE_MODEL.gateway_to_ultra_image

    ultra_to_gateway["extra"] = "mutated"
    gateway_to_ultra["extra"] = "mutated"

    assert "extra" not in DEFAULT_HARDWARE_MODEL.ultra_to_gateway_image
    assert "extra" not in DEFAULT_HARDWARE_MODEL.gateway_to_ultra_image


def test_constructor_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        HardwareModel("gen3")
