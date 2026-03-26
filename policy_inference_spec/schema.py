from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from policy_inference_spec.hardware_model import HardwareModel, as_hardware_model

KEY_OBS_JOINT_POSITION = "observation/joint_position"
KEY_PROMPT = "prompt"
KEY_HARDWARE_MODEL = "hardware_model"
KEY_ACTIONS = "actions"
KEY_INFERENCE_TIME = "inference_time"
KEY_ENDPOINT = "endpoint"
ENDPOINT_RESET = "reset"
ENDPOINT_TELEMETRY = "telemetry"
KEY_MODEL_ID = "model_id"

GEN2_GATEWAY_CAMERAS = ["images/main_image_left", "images/left_wrist_image_left", "images/right_wrist_image_left"]
GEN2_ULTRA_TO_GATEWAY_IMAGE = {
    "observation.images.head": "images/main_image_left",
    "observation.images.left_wrist": "images/left_wrist_image_left",
    "observation.images.right_wrist": "images/right_wrist_image_left",
}
GEN2_GATEWAY_TO_ULTRA_IMAGE = {v: k for k, v in GEN2_ULTRA_TO_GATEWAY_IMAGE.items()}

GEN1_GATEWAY_CAMERAS = ["images/main_image_left", "images/left_wrist_image_left", "images/right_wrist_image_left"]
GEN1_ULTRA_TO_GATEWAY_IMAGE = {
    "main_image": "images/main_image_left",
    "left_wrist_image": "images/left_wrist_image_left",
    "right_wrist_image": "images/right_wrist_image_left",
}
GEN1_GATEWAY_TO_ULTRA_IMAGE = {v: k for k, v in GEN1_ULTRA_TO_GATEWAY_IMAGE.items()}

GEN1_STATE_DIM = 60
GEN2_STATE_DIM = 89
WIRE_ACTION_DIMS_ALLOWED = frozenset({22, 25})


def _observation_keys_for_gen() -> list[str]:
    return [f"observation/{cam}" for cam in GEN2_GATEWAY_CAMERAS]


def _summarize_response_value(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return f"ndarray(shape={value.shape}, dtype={value.dtype})"
    if isinstance(value, dict):
        return f"dict(keys={sorted(str(key) for key in value)})"
    if isinstance(value, list):
        return f"list(len={len(value)})"
    return type(value).__name__


def _summarize_response_payload(result: dict[str, Any]) -> str:
    parts = [f"{key}={_summarize_response_value(value)}" for key, value in sorted(result.items(), key=lambda item: str(item[0]))]
    return ", ".join(parts)


def wire_joint_position_array(value: Any, hardware_model: str | HardwareModel) -> npt.NDArray[np.float32]:
    hm = assert_supported_hardware_model(hardware_model)
    assert isinstance(value, list), f"{KEY_OBS_JOINT_POSITION} must be list"
    joint = np.asarray(value, dtype=np.float32)
    assert joint.ndim == 1, f"{KEY_OBS_JOINT_POSITION} must be 1-D, got shape {joint.shape}"
    expected_dim = GEN1_STATE_DIM if hm == HardwareModel.GEN1 else GEN2_STATE_DIM
    assert joint.shape == (expected_dim,), f"{KEY_OBS_JOINT_POSITION} must be ({expected_dim},), got {joint.shape}"
    return joint


def wire_inference_request_keys(*, hardware_model: HardwareModel) -> frozenset[str]:
    base = frozenset(
        {
            KEY_OBS_JOINT_POSITION,
            *_observation_keys_for_gen(),
            KEY_PROMPT,
            KEY_MODEL_ID,
        }
    )
    if hardware_model == HardwareModel.GEN1:
        return base | {KEY_HARDWARE_MODEL}
    assert hardware_model == HardwareModel.GEN2, f"Unknown hardware_model: {hardware_model!r}"
    return base


def assert_supported_hardware_model(hardware_model: str | HardwareModel) -> HardwareModel:
    return as_hardware_model(hardware_model)


def validate_ultra_arrays_for_hardware_model(
    arrays: dict[str, npt.NDArray[Any]],
    hardware_model: str | HardwareModel,
) -> None:
    hm = assert_supported_hardware_model(hardware_model)
    image_keys = set(GEN1_ULTRA_TO_GATEWAY_IMAGE) if hm == HardwareModel.GEN1 else set(GEN2_ULTRA_TO_GATEWAY_IMAGE)
    keys = set(arrays.keys())
    expected = {"observation.state", *image_keys}
    assert keys == expected, f"{hm.value} request keys {keys} != expected {expected}"
    state = arrays["observation.state"]
    expected_state_shape = (1, GEN1_STATE_DIM) if hm == HardwareModel.GEN1 else (1, GEN2_STATE_DIM)
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
    hm_raw = frame.get(KEY_HARDWARE_MODEL)
    if hm_raw is None or (isinstance(hm_raw, str) and not hm_raw.strip()):
        hardware_model = HardwareModel.GEN2
    else:
        hardware_model = as_hardware_model(str(hm_raw).strip())
    allowed = wire_inference_request_keys(hardware_model=hardware_model)
    keys = set(frame.keys())
    assert keys == allowed, f"wire inference keys {keys} != expected {allowed}"
    assert isinstance(frame[KEY_PROMPT], str), f"{KEY_PROMPT} must be str"
    assert isinstance(frame[KEY_MODEL_ID], str), f"{KEY_MODEL_ID} must be str"
    if hardware_model == HardwareModel.GEN1:
        assert frame[KEY_HARDWARE_MODEL] == HardwareModel.GEN1.value, "gen1 wire frame must set hardware_model to gen1"
    _ = wire_joint_position_array(frame[KEY_OBS_JOINT_POSITION], hardware_model)
    for k in _observation_keys_for_gen():
        v = frame[k]
        assert isinstance(v, (bytes, np.ndarray)), f"{k} must be jpeg bytes or ndarray, got {type(v)}"
    return hardware_model


def validate_wire_inference_response(result: dict[str, Any]) -> None:
    assert isinstance(result, dict), f"response must be dict, got {type(result)}"
    response_summary = _summarize_response_payload(result)
    assert "error" not in result, f"unexpected error payload: {response_summary}"
    allowed = frozenset({KEY_ACTIONS, KEY_INFERENCE_TIME, "policy_id"})
    assert set(result.keys()) <= allowed, f"response keys {set(result.keys())} not subset of {allowed}; summary={response_summary}"
    assert KEY_ACTIONS in result, f"response missing actions; summary={response_summary}"
    actions = result[KEY_ACTIONS]
    assert isinstance(actions, np.ndarray), (
        f"actions must be ndarray, got {type(actions)}; summary={response_summary}"
    )
    assert actions.ndim == 2, f"actions must be 2-D, got shape {actions.shape}"
    assert actions.shape[1] in WIRE_ACTION_DIMS_ALLOWED, (
        f"actions second dim must be one of {sorted(WIRE_ACTION_DIMS_ALLOWED)}, got {actions.shape}"
    )
    assert np.issubdtype(actions.dtype, np.floating), f"actions must be floating ndarray, got {actions.dtype}"
    if KEY_INFERENCE_TIME in result:
        assert isinstance(result[KEY_INFERENCE_TIME], (int, float)), "inference_time must be numeric"
    if "policy_id" in result:
        assert isinstance(result["policy_id"], str), "policy_id must be str"


__all__ = [
    "HardwareModel",
    "ENDPOINT_RESET",
    "ENDPOINT_TELEMETRY",
    "GEN1_GATEWAY_CAMERAS",
    "GEN1_GATEWAY_TO_ULTRA_IMAGE",
    "GEN1_ULTRA_TO_GATEWAY_IMAGE",
    "GEN1_STATE_DIM",
    "GEN2_GATEWAY_CAMERAS",
    "GEN2_GATEWAY_TO_ULTRA_IMAGE",
    "GEN2_STATE_DIM",
    "as_hardware_model",
    "KEY_ACTIONS",
    "KEY_ENDPOINT",
    "KEY_HARDWARE_MODEL",
    "KEY_INFERENCE_TIME",
    "KEY_MODEL_ID",
    "KEY_OBS_JOINT_POSITION",
    "KEY_PROMPT",
    "WIRE_ACTION_DIMS_ALLOWED",
    "assert_supported_hardware_model",
    "validate_ultra_arrays_for_hardware_model",
    "validate_wire_inference_request_frame",
    "validate_wire_inference_response",
    "wire_joint_position_array",
    "wire_inference_request_keys",
]
