from __future__ import annotations

import numpy as np
import pytest

from policy_inference_spec.constants import OBS_JOINT_POSITION_KEY
from policy_inference_spec.hardware_model import (
    DEFAULT_HARDWARE_MODEL,
    HardwareModel,
    validate_ultra_arrays_for_hardware_model,
)


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
    assert DEFAULT_HARDWARE_MODEL.cameras == (
        "images/main_image",
        "images/left_wrist_image",
        "images/right_wrist_image",
    )


def test_validate_ultra_arrays_accepts_gateway_camera_names() -> None:
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    arrays = {
        OBS_JOINT_POSITION_KEY: np.zeros((1, DEFAULT_HARDWARE_MODEL.state_dim), dtype=np.float32),
    }
    for camera in DEFAULT_HARDWARE_MODEL.cameras:
        arrays[f"observation/{camera}"] = image

    validate_ultra_arrays_for_hardware_model(arrays)


def test_validate_ultra_arrays_rejects_legacy_ultra_camera_names() -> None:
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    arrays = {
        OBS_JOINT_POSITION_KEY: np.zeros((1, DEFAULT_HARDWARE_MODEL.state_dim), dtype=np.float32),
        "observation.images.head": image,
        "observation.images.left_wrist": image,
        "observation.images.right_wrist": image,
    }

    with pytest.raises(AssertionError, match="request keys"):
        validate_ultra_arrays_for_hardware_model(arrays)


def test_constructor_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        HardwareModel("gen3")
