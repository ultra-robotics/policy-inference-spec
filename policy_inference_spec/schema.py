from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import numpy.typing as npt

KEY_OBS_JOINT_POSITION = "observation/joint_position"
KEY_PROMPT = "prompt"
KEY_ACTIONS = "actions"
KEY_INFERENCE_TIME = "inference_time"
KEY_ENDPOINT = "endpoint"
ENDPOINT_RESET = "reset"
ENDPOINT_TELEMETRY = "telemetry"
KEY_MODEL_ID = "model_id"


@dataclass(frozen=True)
class _HardwareModelSpec:
    state_dim: int
    action_dim: int
    image_resolution: tuple[int, int]
    gateway_cameras: tuple[str, ...]
    ultra_to_gateway_image: dict[str, str]
    gateway_to_ultra_image: dict[str, str]


class HardwareModel(StrEnum):
    GEN2 = "gen2"

    @classmethod
    def _missing_(cls, value: object) -> HardwareModel | None:
        if isinstance(value, str):
            normalized = value.strip()
            if normalized == value:
                return None
        else:
            normalized = str(value).strip()
        try:
            return cls(normalized)
        except ValueError:
            return None

    @property
    def state_dim(self) -> int:
        return _HARDWARE_MODEL_SPECS[self].state_dim

    @property
    def action_dim(self) -> int:
        return _HARDWARE_MODEL_SPECS[self].action_dim

    @property
    def image_resolution(self) -> tuple[int, int]:
        return _HARDWARE_MODEL_SPECS[self].image_resolution

    @property
    def gateway_cameras(self) -> tuple[str, ...]:
        return _HARDWARE_MODEL_SPECS[self].gateway_cameras

    @property
    def ultra_to_gateway_image(self) -> dict[str, str]:
        return _HARDWARE_MODEL_SPECS[self].ultra_to_gateway_image.copy()

    @property
    def gateway_to_ultra_image(self) -> dict[str, str]:
        return _HARDWARE_MODEL_SPECS[self].gateway_to_ultra_image.copy()


_GEN2_ULTRA_TO_GATEWAY_IMAGE = {
    "observation.images.head": "images/main_image_left",
    "observation.images.left_wrist": "images/left_wrist_image_left",
    "observation.images.right_wrist": "images/right_wrist_image_left",
}
_GEN2_GATEWAY_TO_ULTRA_IMAGE = {v: k for k, v in _GEN2_ULTRA_TO_GATEWAY_IMAGE.items()}

_HARDWARE_MODEL_SPECS = {
    HardwareModel.GEN2: _HardwareModelSpec(
        state_dim=97,
        action_dim=25,
        image_resolution=(360, 640),
        gateway_cameras=(
            "images/main_image_left",
            "images/left_wrist_image_left",
            "images/right_wrist_image_left",
        ),
        ultra_to_gateway_image=_GEN2_ULTRA_TO_GATEWAY_IMAGE,
        gateway_to_ultra_image=_GEN2_GATEWAY_TO_ULTRA_IMAGE,
    ),
}

DEFAULT_HARDWARE_MODEL = HardwareModel.GEN2


def _observation_keys(hardware_model: HardwareModel) -> list[str]:
    return [f"observation/{cam}" for cam in hardware_model.gateway_cameras]


def _summarize_response_value(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return f"ndarray(shape={value.shape}, dtype={value.dtype})"
    if isinstance(value, dict):
        return f"dict(keys={sorted(str(key) for key in value)})"
    if isinstance(value, list):
        return f"list(len={len(value)})"
    return type(value).__name__


def _summarize_response_payload(result: dict[str, Any]) -> str:
    parts = [
        f"{key}={_summarize_response_value(value)}"
        for key, value in sorted(result.items(), key=lambda item: str(item[0]))
    ]
    return ", ".join(parts)


def wire_joint_position_array(
    value: Any,
    hardware_model: str | HardwareModel = DEFAULT_HARDWARE_MODEL,
) -> npt.NDArray[np.float32]:
    hm = HardwareModel(hardware_model)
    assert isinstance(value, np.ndarray), f"{KEY_OBS_JOINT_POSITION} must be ndarray"
    joint = np.asarray(value, dtype=np.float32)
    assert joint.ndim == 1, f"{KEY_OBS_JOINT_POSITION} must be 1-D, got shape {joint.shape}"
    assert joint.shape == (hm.state_dim,), (
        f"{hm.value} {KEY_OBS_JOINT_POSITION} must be ({hm.state_dim},), got {joint.shape}"
    )
    return joint


def wire_inference_request_keys(*, hardware_model: HardwareModel = DEFAULT_HARDWARE_MODEL) -> frozenset[str]:
    assert isinstance(hardware_model, HardwareModel), (
        f"hardware_model must be HardwareModel, got {type(hardware_model)}"
    )
    return frozenset(
        {
            KEY_OBS_JOINT_POSITION,
            *_observation_keys(hardware_model),
            KEY_PROMPT,
            KEY_MODEL_ID,
        }
    )


def validate_ultra_arrays_for_hardware_model(
    arrays: dict[str, npt.NDArray[Any]],
    hardware_model: str | HardwareModel = DEFAULT_HARDWARE_MODEL,
) -> None:
    hm = HardwareModel(hardware_model)
    image_keys = set(hm.ultra_to_gateway_image)
    keys = set(arrays.keys())
    expected = {"observation.state", *image_keys}
    assert keys == expected, f"{hm.value} request keys {keys} != expected {expected}"
    state = arrays["observation.state"]
    expected_state_shape = (1, hm.state_dim)
    assert state.shape == expected_state_shape, (
        f"{hm.value} observation.state shape {state.shape} != {expected_state_shape}"
    )
    for key in image_keys:
        image = arrays[key]
        assert image.ndim in (3, 4), f"{hm.value} image {key} must be CHW or BCHW, got {image.shape}"
        if image.ndim == 4:
            assert image.shape[0] == 1, f"{hm.value} image {key} batch size must be 1, got {image.shape}"
            assert image.shape[1] == 3, f"{hm.value} image {key} channel dim must be 3, got {image.shape}"
            continue
        assert image.shape[0] == 3, f"{hm.value} image {key} channel dim must be 3, got {image.shape}"


def validate_wire_inference_request_frame(frame: dict[str, Any]) -> HardwareModel:
    assert isinstance(frame, dict), f"wire frame must be dict, got {type(frame)}"
    assert KEY_ENDPOINT not in frame, "inference frame must not contain endpoint"
    hardware_model = DEFAULT_HARDWARE_MODEL
    allowed = wire_inference_request_keys(hardware_model=hardware_model)
    keys = set(frame.keys())
    assert keys == allowed, f"wire inference keys {keys} != expected {allowed}"
    assert isinstance(frame[KEY_PROMPT], str), f"{KEY_PROMPT} must be str"
    assert isinstance(frame[KEY_MODEL_ID], str), f"{KEY_MODEL_ID} must be str"
    _ = wire_joint_position_array(frame[KEY_OBS_JOINT_POSITION], hardware_model)
    for k in _observation_keys(hardware_model):
        v = frame[k]
        assert isinstance(v, (bytes, np.ndarray)), f"{k} must be jpeg bytes or ndarray, got {type(v)}"
    return hardware_model


def validate_wire_inference_response(
    result: dict[str, Any],
    hardware_model: HardwareModel = DEFAULT_HARDWARE_MODEL,
) -> None:
    assert isinstance(hardware_model, HardwareModel), (
        f"hardware_model must be HardwareModel, got {type(hardware_model)}"
    )
    assert isinstance(result, dict), f"response must be dict, got {type(result)}"
    response_summary = _summarize_response_payload(result)
    assert "error" not in result, f"unexpected error payload: {response_summary}"
    allowed = frozenset({KEY_ACTIONS, KEY_INFERENCE_TIME, "policy_id"})
    assert set(result.keys()) <= allowed, (
        f"response keys {set(result.keys())} not subset of {allowed}; summary={response_summary}"
    )
    assert KEY_ACTIONS in result, f"response missing actions; summary={response_summary}"
    actions = result[KEY_ACTIONS]
    assert isinstance(actions, np.ndarray), f"actions must be ndarray, got {type(actions)}; summary={response_summary}"
    assert actions.ndim == 2, f"actions must be 2-D, got shape {actions.shape}"
    assert actions.shape[1] == hardware_model.action_dim, (
        f"actions second dim must be {hardware_model.action_dim}, got {actions.shape}"
    )
    assert np.issubdtype(actions.dtype, np.floating), f"actions must be floating ndarray, got {actions.dtype}"
    if KEY_INFERENCE_TIME in result:
        assert isinstance(result[KEY_INFERENCE_TIME], (int, float)), "inference_time must be numeric"
    if "policy_id" in result:
        assert isinstance(result["policy_id"], str), "policy_id must be str"


__all__ = [
    "DEFAULT_HARDWARE_MODEL",
    "HardwareModel",
    "ENDPOINT_RESET",
    "ENDPOINT_TELEMETRY",
    "KEY_ACTIONS",
    "KEY_ENDPOINT",
    "KEY_INFERENCE_TIME",
    "KEY_MODEL_ID",
    "KEY_OBS_JOINT_POSITION",
    "KEY_PROMPT",
    "validate_ultra_arrays_for_hardware_model",
    "validate_wire_inference_request_frame",
    "validate_wire_inference_response",
    "wire_joint_position_array",
    "wire_inference_request_keys",
]
