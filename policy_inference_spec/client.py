from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import numpy as np
import numpy.typing as npt
import simplejpeg
import websockets
from websockets.sync.client import connect as ws_connect_sync

from policy_inference_spec.constants import DEFAULT_INFERENCE_SERVER_PORT
from policy_inference_spec.hardware_model import HardwareModel, as_hardware_model
from policy_inference_spec.protocol import msgpack_decode, msgpack_encode
from policy_inference_spec.schema import (
    GEN1_GATEWAY_CAMERAS,
    GEN1_SHAPES,
    GEN1_STATE_DIM,
    GEN2_SHAPES,
    GEN2_STATE_DIM,
    GEN2_ULTRA_TO_GATEWAY_IMAGE,
    KEY_ACTIONS,
    KEY_HARDWARE_MODEL,
    KEY_INFERENCE_TIME,
    KEY_MODEL_ID,
    KEY_OBS_JOINT_POSITION,
    KEY_PROMPT,
    validate_wire_inference_request_frame,
    validate_wire_inference_response,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_PREDICT_URL = f"ws://inf.ultra.tech:{DEFAULT_INFERENCE_SERVER_PORT}/ws"


def policy_ws_url(url: str) -> str:
    u = url.strip()
    parsed = urlparse(u)
    assert parsed.scheme in ("ws", "wss"), f"POLICY_SERVER_URL must be ws:// or wss://, got {url!r}"
    return u


def _random_jpeg_bytes(rng: np.random.Generator, h: int, w: int) -> bytes:
    rgb = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    return simplejpeg.encode_jpeg(rgb, quality=75)


def _random_warmup_wire_frame(hardware_model: HardwareModel) -> dict[str, Any]:
    rng = np.random.default_rng()
    if hardware_model == HardwareModel.GEN1:
        joint = rng.standard_normal(GEN1_STATE_DIM, dtype=np.float32)
        h = int(GEN1_SHAPES["main_image"][2])
        w = int(GEN1_SHAPES["main_image"][3])
        frame: dict[str, Any] = {
            KEY_OBS_JOINT_POSITION: joint,
            KEY_PROMPT: "",
            KEY_MODEL_ID: "",
            KEY_HARDWARE_MODEL: HardwareModel.GEN1.value,
        }
        for cam in GEN1_GATEWAY_CAMERAS:
            frame[f"observation/{cam}"] = _random_jpeg_bytes(rng, h, w)
    else:
        joint = rng.standard_normal(GEN2_STATE_DIM, dtype=np.float32)
        frame: dict[str, Any] = {
            KEY_OBS_JOINT_POSITION: joint,
            KEY_PROMPT: "",
            KEY_MODEL_ID: "",
        }
        for ultra_key, cam in GEN2_ULTRA_TO_GATEWAY_IMAGE.items():
            sh = GEN2_SHAPES[ultra_key]
            hh, ww = int(sh[2]), int(sh[3])
            frame[f"observation/{cam}"] = _random_jpeg_bytes(rng, hh, ww)
    validate_wire_inference_request_frame(frame)
    return frame


@dataclass(frozen=True)
class RemotePolicyPrediction:
    actions_d: npt.NDArray[np.float32]
    total_latency_ms: float
    policy_id: str


class RemotePolicyClient:
    def __init__(
        self,
        predict_url: str,
        *,
        policy_auth_headers: dict[str, str] | None = None,
    ) -> None:
        self.predict_url = predict_url
        self._policy_auth_headers = policy_auth_headers or {}
        self._ws: Any = None
        self._server_config: dict[str, Any] | None = None
        self._connected_url: str | None = None

    async def __aenter__(self) -> RemotePolicyClient:
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
        self._server_config = None
        self._connected_url = None

    def update_connection(self, *, predict_url: str, policy_auth_headers: dict[str, str] | None = None) -> None:
        self.predict_url = predict_url
        self._policy_auth_headers = policy_auth_headers or {}
        self._connected_url = None

    def _headers(self) -> list[tuple[str, str]]:
        return [(k, v) for k, v in self._policy_auth_headers.items()]

    def warmup(self, *, hardware_model: str | HardwareModel = HardwareModel.GEN2) -> bool:
        try:
            hm = as_hardware_model(hardware_model)
            wire_frame = _random_warmup_wire_frame(hm)
            uri = policy_ws_url(self.predict_url)
            with ws_connect_sync(uri, additional_headers=self._headers()) as ws:
                first = ws.recv()
                assert isinstance(first, bytes), type(first)
                _ = msgpack_decode(first)
                ws.send(msgpack_encode(wire_frame))
                ws.recv()
            LOGGER.info("Inference server warmup complete")
            return True
        except Exception as exc:
            LOGGER.error("Inference server warmup failed for %s: %s", self.predict_url, exc, exc_info=True)
            return False

    async def _ensure_ws(self) -> None:
        uri = policy_ws_url(self.predict_url)
        if self._ws is not None and self._connected_url == uri:
            return
        if self._ws is not None:
            await self._close_ws()
        self._ws = await websockets.connect(uri, additional_headers=self._headers())
        self._connected_url = uri
        first = await self._ws.recv()
        assert isinstance(first, bytes), type(first)
        self._server_config = msgpack_decode(first)
        assert isinstance(self._server_config, dict), "ServerConfig must be a dict"

    async def _close_ws(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
        self._server_config = None
        self._connected_url = None

    async def predict(self, wire_frame: dict[str, Any]) -> RemotePolicyPrediction:
        await self._ensure_ws()
        assert self._ws is not None
        validate_wire_inference_request_frame(wire_frame)
        payload = msgpack_encode(wire_frame)
        start_time_ns = time.time_ns()
        await self._ws.send(payload)
        response_raw = await self._ws.recv()
        end_time_ns = time.time_ns()

        total_latency_ms = (end_time_ns - start_time_ns) / 1e6
        assert isinstance(response_raw, bytes), type(response_raw)
        result = msgpack_decode(response_raw)
        assert isinstance(result, dict), f"unexpected response type {type(result)}"
        validate_wire_inference_response(result)
        actions = result[KEY_ACTIONS]
        infer_raw = result.get(KEY_INFERENCE_TIME)
        server_latency_ms = float(infer_raw) if infer_raw is not None else 0.0
        policy_id_used = str(result.get("policy_id", ""))

        network_latency_ms = total_latency_ms - server_latency_ms
        LOGGER.info(
            "Inference latency: %.1fms (server) + %.1fms (network) = %.1fms (total)",
            server_latency_ms,
            network_latency_ms,
            total_latency_ms,
        )

        actions_d = np.array(actions, dtype=np.float32)
        return RemotePolicyPrediction(
            actions_d=actions_d,
            total_latency_ms=total_latency_ms,
            policy_id=policy_id_used,
        )


__all__ = [
    "DEFAULT_PREDICT_URL",
    "RemotePolicyClient",
    "RemotePolicyPrediction",
    "policy_ws_url",
]
