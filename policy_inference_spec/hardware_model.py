from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import numpy.typing as npt

from policy_inference_spec.constants import (
    ACTION_KEY,
    ENDPOINT_KEY,
    INFERENCE_TIME_KEY,
    JOINT_STATE_KEY,
    MODEL_ID_KEY,
    PROMPT_KEY,
)


@dataclass(frozen=True)
class _HardwareModelSpec:
    state_dim: int
    action_dim: int
    image_resolution: tuple[int, int]
    cameras: tuple[str, ...]


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
    def cameras(self) -> tuple[str, ...]:
        return _HARDWARE_MODEL_SPECS[self].cameras


_HARDWARE_MODEL_SPECS = {
    HardwareModel.GEN2: _HardwareModelSpec(
        state_dim=97,
        action_dim=25,
        image_resolution=(360, 640),
        cameras=(
            "images/main_image",
            "images/left_wrist_image",
            "images/right_wrist_image",
        ),
    ),
}

DEFAULT_HARDWARE_MODEL = HardwareModel.GEN2


def _observation_keys(hardware_model: HardwareModel) -> list[str]:
    return [f"observation/{cam}" for cam in hardware_model.cameras]


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


def _validate_joint_position_array(
    value: np.ndarray,
    hardware_model: str | HardwareModel = DEFAULT_HARDWARE_MODEL,
) -> None:
    hm = HardwareModel(hardware_model)
    assert isinstance(value, np.ndarray), f"{JOINT_STATE_KEY} must be ndarray"
    assert value.ndim == 1, f"{JOINT_STATE_KEY} must be 1-D, got shape {value.shape}"
    assert value.shape == (hm.state_dim,), (
        f"{hm.value} {JOINT_STATE_KEY} must be ({hm.state_dim},), got {value.shape}"
    )


def _wire_inference_request_keys(*, hardware_model: HardwareModel = DEFAULT_HARDWARE_MODEL) -> frozenset[str]:
    assert isinstance(hardware_model, HardwareModel), (
        f"hardware_model must be HardwareModel, got {type(hardware_model)}"
    )
    return frozenset(
        {
            JOINT_STATE_KEY,
            *_observation_keys(hardware_model),
            PROMPT_KEY,
            MODEL_ID_KEY,
        }
    )


def validate_ultra_arrays_for_hardware_model(
    arrays: dict[str, npt.NDArray[Any]],
    hardware_model: str | HardwareModel = DEFAULT_HARDWARE_MODEL,
) -> None:
    hm = HardwareModel(hardware_model)
    image_keys = set(_observation_keys(hm))
    keys = set(arrays.keys())
    expected = {JOINT_STATE_KEY, *image_keys}
    assert keys == expected, f"{hm.value} request keys {keys} != expected {expected}"
    joint_position = arrays[JOINT_STATE_KEY]
    expected_joint_position_shape = (1, hm.state_dim)
    assert joint_position.shape == expected_joint_position_shape, (
        f"{hm.value} {JOINT_STATE_KEY} shape {joint_position.shape} != {expected_joint_position_shape}"
    )
    for key in image_keys:
        image = arrays[key]
        assert image.ndim in (3, 4), f"{hm.value} image {key} must be HWC or BHWC, got {image.shape}"
        if image.ndim == 4:
            assert image.shape[0] == 1, f"{hm.value} image {key} batch size must be 1, got {image.shape}"
            assert image.shape[-1] == 3, f"{hm.value} image {key} channel dim must be 3, got {image.shape}"
            continue
        assert image.shape[-1] == 3, f"{hm.value} image {key} channel dim must be 3, got {image.shape}"


def validate_wire_inference_request_frame(
    frame: dict[str, Any], hardware_model: HardwareModel = DEFAULT_HARDWARE_MODEL
) -> HardwareModel:
    assert ENDPOINT_KEY not in frame, "inference frame must not contain endpoint"
    allowed = _wire_inference_request_keys(hardware_model=hardware_model)
    keys = set(frame.keys())
    assert keys == allowed, f"wire inference keys {keys} != expected {allowed}"
    assert isinstance(frame[PROMPT_KEY], str), f"{PROMPT_KEY} must be str"
    assert isinstance(frame[MODEL_ID_KEY], str), f"{MODEL_ID_KEY} must be str"
    _validate_joint_position_array(frame[JOINT_STATE_KEY], hardware_model)
    for k in _observation_keys(hardware_model):
        v = frame[k]
        assert isinstance(v, (bytes, np.ndarray)), f"{k} must be jpeg bytes or ndarray, got {type(v)}"
    return hardware_model


def validate_wire_inference_response(
    result: dict[str, Any],
    hardware_model: HardwareModel = DEFAULT_HARDWARE_MODEL,
) -> None:
    response_summary = _summarize_response_payload(result)
    assert "error" not in result, f"unexpected error payload: {response_summary}"
    allowed = frozenset({ACTION_KEY, INFERENCE_TIME_KEY, "policy_id"})
    assert set(result.keys()) <= allowed, (
        f"response keys {set(result.keys())} not subset of {allowed}; summary={response_summary}"
    )
    assert ACTION_KEY in result, f"response missing action; summary={response_summary}"
    actions = result[ACTION_KEY]
    assert isinstance(actions, np.ndarray), f"action must be ndarray, got {type(actions)}; summary={response_summary}"
    assert actions.ndim == 2, f"action must be 2-D, got shape {actions.shape}"
    assert actions.shape[1] == hardware_model.action_dim, (
        f"action second dim must be {hardware_model.action_dim}, got {actions.shape}"
    )
    assert np.issubdtype(actions.dtype, np.floating), f"action must be floating ndarray, got {actions.dtype}"
    if INFERENCE_TIME_KEY in result:
        assert isinstance(result[INFERENCE_TIME_KEY], (int, float)), "inference_time must be numeric"
    if "policy_id" in result:
        assert isinstance(result["policy_id"], str), "policy_id must be str"


__all__ = [
    "DEFAULT_HARDWARE_MODEL",
    "HardwareModel",
    "validate_ultra_arrays_for_hardware_model",
    "validate_wire_inference_request_frame",
    "validate_wire_inference_response",
]
