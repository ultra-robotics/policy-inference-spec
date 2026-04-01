from __future__ import annotations

import logging
import sys
from typing import Any
from urllib.parse import urlparse

import numpy as np
import simplejpeg  # type: ignore[import-untyped]

from policy_inference_spec.protocol import DEFAULT_INFERENCE_SERVER_PORT, JOINT_STATE_KEY, MODEL_ID_KEY, PROMPT_KEY, ServerHandshake
from policy_inference_spec.hardware_model import (
    DEFAULT_HARDWARE_MODEL,
    HardwareModel,
    validate_wire_inference_request_frame,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_PREDICT_URL = f"ws://inf.ultra.tech:{DEFAULT_INFERENCE_SERVER_PORT}/ws"


def policy_ws_url(url: str) -> str:
    u = url.strip()
    parsed = urlparse(u)
    assert parsed.scheme in ("ws", "wss"), f"POLICY_SERVER_URL must be ws:// or wss://, got {url!r}"
    if parsed.path in ("", "/"):
        return parsed._replace(path="/ws").geturl()
    return u


def _log_server_config(server_config: ServerHandshake) -> None:
    LOGGER.info("Received inference server config: %s", server_config.to_payload())


def _wire_camera_names(wire_frame: dict[str, Any]) -> list[str]:
    camera_names: list[str] = []
    for key in wire_frame:
        if not key.startswith("observation/") or key == JOINT_STATE_KEY:
            continue
        camera_names.append(key.removeprefix("observation/"))
    return sorted(camera_names)


def _truncate_log_value(value: Any, *, max_chars: int = 120) -> str:
    if isinstance(value, bytes):
        preview = repr(value[:24])
        suffix = "..." if len(value) > 24 else ""
        return f"bytes(len={len(value)}, preview={preview}{suffix})"
    if isinstance(value, np.ndarray):
        preview = np.array2string(value.reshape(-1)[:6], threshold=6)
        suffix = "..." if value.size > 6 else ""
        return f"ndarray(shape={value.shape}, dtype={value.dtype}, preview={preview}{suffix})"
    rendered = repr(value)
    if len(rendered) <= max_chars:
        return rendered
    return f"{rendered[:max_chars]}..."


def _summarize_wire_frame(wire_frame: dict[str, Any]) -> dict[str, str]:
    return {key: _truncate_log_value(wire_frame[key]) for key in sorted(wire_frame.keys())}


def _summarize_server_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {
            str(key): _summarize_server_payload(value)
            for key, value in sorted(payload.items(), key=lambda item: str(item[0]))
        }
    if isinstance(payload, list):
        return f"list(len={len(payload)})"
    return _truncate_log_value(payload)


def _emit_server_error_verbatim(payload: Any) -> None:
    if isinstance(payload, str):
        print(_truncate_log_value(payload, max_chars=400), file=sys.stderr, flush=True)
        return
    if isinstance(payload, dict) and "error" in payload:
        print(_truncate_log_value(payload["error"], max_chars=400), file=sys.stderr, flush=True)


def _server_image_resolution(server_config: ServerHandshake | None) -> tuple[int, int] | None:
    return None if server_config is None else server_config.image_resolution


def _random_jpeg_bytes(rng: np.random.Generator, h: int, w: int) -> bytes:
    rgb = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    return simplejpeg.encode_jpeg(rgb, quality=75)


def _random_warmup_wire_frame(
    hardware_model: str | HardwareModel = DEFAULT_HARDWARE_MODEL,
    *,
    image_resolution: tuple[int, int] | None = None,
) -> dict[str, Any]:
    hm = HardwareModel(hardware_model)
    rng = np.random.default_rng()
    height, width = image_resolution or hm.image_resolution
    joint = rng.standard_normal(hm.state_dim, dtype=np.float32)
    frame: dict[str, Any] = {
        JOINT_STATE_KEY: joint,
        PROMPT_KEY: "",
        MODEL_ID_KEY: "",
    }
    for cam in hm.cameras:
        frame[f"observation/{cam}"] = _random_jpeg_bytes(rng, height, width)
    validate_wire_inference_request_frame(frame)
    return frame
