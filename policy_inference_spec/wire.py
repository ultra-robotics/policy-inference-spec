from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from policy_inference_spec.protocol import chw_from_wire_image, encode_ndarray

KEY_OBS_JOINT_POSITION = "observation/joint_position"
KEY_PROMPT = "prompt"
KEY_HARDWARE_MODEL = "hardware_model"
KEY_ACTIONS = "actions"
KEY_INFERENCE_TIME = "inference_time"
KEY_ENDPOINT = "endpoint"
ENDPOINT_RESET = "reset"
ENDPOINT_TELEMETRY = "telemetry"
KEY_MODEL_ID = "model_id"

GEN2_GATEWAY_CAMERAS = ["main_image_left", "left_wrist_image_left", "right_wrist_image_left"]
GEN2_ULTRA_TO_GATEWAY_IMAGE = {
    "observation.images.head": "main_image_left",
    "observation.images.left_wrist": "left_wrist_image_left",
    "observation.images.right_wrist": "right_wrist_image_left",
}
GEN2_GATEWAY_TO_ULTRA_IMAGE = {v: k for k, v in GEN2_ULTRA_TO_GATEWAY_IMAGE.items()}

GEN1_GATEWAY_CAMERAS = ["main_image_left", "left_wrist_image_left", "right_wrist_image_left"]
GEN1_ULTRA_TO_GATEWAY_IMAGE = {
    "main_image": "main_image_left",
    "left_wrist_image": "left_wrist_image_left",
    "right_wrist_image": "right_wrist_image_left",
}
GEN1_GATEWAY_TO_ULTRA_IMAGE = {v: k for k, v in GEN1_ULTRA_TO_GATEWAY_IMAGE.items()}

GEN1_SHAPES: dict[str, tuple[int, ...]] = {
    "observation.state": (1, 60),
    "main_image": (1, 3, 360, 1280),
    "left_wrist_image": (1, 3, 360, 1280),
    "right_wrist_image": (1, 3, 360, 1280),
}

GEN2_SHAPES: dict[str, tuple[int, ...]] = {
    "observation.state": (1, 89),
    "observation.images.head": (1, 3, 360, 640),
    "observation.images.left_wrist": (1, 3, 300, 480),
    "observation.images.right_wrist": (1, 3, 300, 480),
}


def camera_names_for_hardware_model(hardware_model: str) -> list[str]:
    if hardware_model == "gen1":
        return list(GEN1_GATEWAY_CAMERAS)
    assert hardware_model == "gen2", f"Unknown hardware_model: {hardware_model!r}"
    return list(GEN2_GATEWAY_CAMERAS)


def _validate_ultra_arrays(arrays: dict[str, npt.NDArray[Any]], hardware_model: str) -> None:
    shapes = GEN1_SHAPES if hardware_model == "gen1" else GEN2_SHAPES
    keys = set(arrays.keys())
    expected = set(shapes.keys())
    assert keys == expected, f"{hardware_model} request keys {keys} != expected {expected}"
    for k, shape in shapes.items():
        assert arrays[k].shape == shape, f"{hardware_model} array {k} shape {arrays[k].shape} != {shape}"


def ultra_arrays_to_wire_observation(
    arrays: dict[str, npt.NDArray[Any]],
    hardware_model: str,
) -> dict[str, Any]:
    _validate_ultra_arrays(arrays, hardware_model)
    out: dict[str, Any] = {
        KEY_OBS_JOINT_POSITION: np.squeeze(arrays["observation.state"], axis=0).astype(np.float32, copy=False),
    }
    if hardware_model == "gen1":
        for ultra_key, cam in GEN1_ULTRA_TO_GATEWAY_IMAGE.items():
            field = encode_ndarray(arrays[ultra_key], jpeg_quality=75, image_scale=0.5)
            assert field.codec == "jpeg", "gen1 images must use jpeg transport"
            out[f"observation/{cam}"] = field.data
        return out
    for ultra_key, cam in GEN2_ULTRA_TO_GATEWAY_IMAGE.items():
        field = encode_ndarray(arrays[ultra_key], jpeg_quality=75, image_scale=0.5)
        assert field.codec == "jpeg", "gen2 images must use jpeg transport"
        out[f"observation/{cam}"] = field.data
    return out


def wire_observation_to_ultra_arrays(
    obs: dict[str, Any],
    hardware_model: str,
) -> dict[str, npt.NDArray[Any]]:
    joint = obs[KEY_OBS_JOINT_POSITION]
    assert isinstance(joint, np.ndarray), f"{KEY_OBS_JOINT_POSITION} must be ndarray, got {type(joint)}"
    state = joint.astype(np.float32, copy=False)
    if state.ndim == 1:
        state = state[None, :]
    shapes = GEN1_SHAPES if hardware_model == "gen1" else GEN2_SHAPES
    if hardware_model == "gen1":
        out: dict[str, npt.NDArray[Any]] = {"observation.state": state}
        for cam in GEN1_GATEWAY_CAMERAS:
            key = f"observation/{cam}"
            ultra_key = GEN1_GATEWAY_TO_ULTRA_IMAGE[cam]
            assert key in obs, f"missing {key}"
            out[ultra_key] = chw_from_wire_image(obs[key], shapes[ultra_key])
        return out
    out = {"observation.state": state}
    for cam in GEN2_GATEWAY_CAMERAS:
        key = f"observation/{cam}"
        ultra_key = GEN2_GATEWAY_TO_ULTRA_IMAGE[cam]
        assert key in obs, f"missing {key}"
        out[ultra_key] = chw_from_wire_image(obs[key], shapes[ultra_key])
    return out


def parse_inference_frame(frame: dict[str, Any]) -> tuple[dict[str, Any], str, str, str]:
    d = dict(frame)
    d.pop("metadata", None)
    d.pop("predict", None)
    hm_raw = d.pop(KEY_HARDWARE_MODEL, None)
    if hm_raw is None or (isinstance(hm_raw, str) and not hm_raw.strip()):
        hardware_model = "gen2"
    else:
        hardware_model = str(hm_raw).strip()
    policy_id = str(d.pop(KEY_MODEL_ID, ""))
    prompt = str(d.pop(KEY_PROMPT, ""))
    return d, hardware_model, policy_id, prompt


def server_config_for_hardware_model(hardware_model: str) -> dict[str, Any]:
    cameras = camera_names_for_hardware_model(hardware_model)
    if hardware_model == "gen1":
        h, w = GEN1_SHAPES["main_image"][2], GEN1_SHAPES["main_image"][3]
        return {
            "camera_names": cameras,
            "image_resolution": (int(h), int(w)),
            "action_space": "joint_position",
            "needs_wrist_camera": True,
            "n_external_cameras": 1,
        }
    h, w = GEN2_SHAPES["observation.images.head"][2], GEN2_SHAPES["observation.images.head"][3]
    return {
        "camera_names": cameras,
        "image_resolution": (int(h), int(w)),
        "action_space": "joint_position",
        "needs_wrist_camera": True,
        "n_external_cameras": 1,
    }


__all__ = [
    "ENDPOINT_RESET",
    "ENDPOINT_TELEMETRY",
    "GEN1_GATEWAY_CAMERAS",
    "GEN1_GATEWAY_TO_ULTRA_IMAGE",
    "GEN1_SHAPES",
    "GEN1_ULTRA_TO_GATEWAY_IMAGE",
    "GEN2_GATEWAY_CAMERAS",
    "GEN2_GATEWAY_TO_ULTRA_IMAGE",
    "GEN2_SHAPES",
    "GEN2_ULTRA_TO_GATEWAY_IMAGE",
    "KEY_ACTIONS",
    "KEY_ENDPOINT",
    "KEY_HARDWARE_MODEL",
    "KEY_INFERENCE_TIME",
    "KEY_MODEL_ID",
    "KEY_OBS_JOINT_POSITION",
    "KEY_PROMPT",
    "camera_names_for_hardware_model",
    "parse_inference_frame",
    "server_config_for_hardware_model",
    "ultra_arrays_to_wire_observation",
    "wire_observation_to_ultra_arrays",
]
