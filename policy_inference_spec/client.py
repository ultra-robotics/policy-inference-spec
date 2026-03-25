from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse, urlunparse

import numpy as np
import numpy.typing as npt
import torch
import websockets
from websockets.sync.client import connect as ws_connect_sync

from policy_inference_spec.constants import DEFAULT_INFERENCE_SERVER_PORT
from policy_inference_spec.protocol import msgpack_decode, msgpack_encode
from policy_inference_spec.wire import (
    KEY_ACTIONS,
    KEY_HARDWARE_MODEL,
    KEY_INFERENCE_TIME,
    KEY_MODEL_ID,
    KEY_PROMPT,
    ultra_arrays_to_wire_observation,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_PREDICT_URL = f"ws://inf.ultra.tech:{DEFAULT_INFERENCE_SERVER_PORT}/ws"


def policy_ws_url(url: str) -> str:
    u = url.strip()
    if u.startswith("http://"):
        u = "ws://" + u[len("http://") :]
    elif u.startswith("https://"):
        u = "wss://" + u[len("https://") :]
    parsed = urlparse(u)
    assert parsed.scheme in ("ws", "wss"), f"POLICY_SERVER_URL must be ws(s) or http(s), got {url!r}"
    path = (parsed.path or "/").rstrip("/") or "/"
    if path.endswith("/predict"):
        path = "/ws"
    elif path == "/":
        path = "/ws"
    elif not path.endswith("/ws"):
        path = path + "/ws"
    return urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))


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
        hardware_model: str = "gen2",
        policy_id: str | None,
        policy_auth_headers: dict[str, str] | None = None,
    ) -> None:
        self.predict_url = predict_url
        self._hardware_model = hardware_model
        self._policy_id = policy_id or ""
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

    def _warmup_arrays(self) -> dict[str, npt.NDArray[np.float32]]:
        if self._hardware_model == "gen1":
            return {
                "observation.state": np.zeros((1, 60), dtype=np.float32),
                "main_image": np.zeros((1, 3, 360, 1280), dtype=np.float32),
                "left_wrist_image": np.zeros((1, 3, 360, 1280), dtype=np.float32),
                "right_wrist_image": np.zeros((1, 3, 360, 1280), dtype=np.float32),
            }
        assert self._hardware_model == "gen2", f"Unknown warmup hardware_model: {self._hardware_model!r}"
        return {
            "observation.state": np.zeros((1, 89), dtype=np.float32),
            "observation.images.head": np.zeros((1, 3, 360, 640), dtype=np.float32),
            "observation.images.left_wrist": np.zeros((1, 3, 300, 480), dtype=np.float32),
            "observation.images.right_wrist": np.zeros((1, 3, 300, 480), dtype=np.float32),
        }

    def _wire_extra_fields(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self._hardware_model != "gen2":
            out[KEY_HARDWARE_MODEL] = self._hardware_model
        if self._policy_id:
            out[KEY_MODEL_ID] = self._policy_id
        return out

    def warmup(self) -> None:
        try:
            uri = policy_ws_url(self.predict_url)
            with ws_connect_sync(uri, additional_headers=self._headers()) as ws:
                first = ws.recv()
                assert isinstance(first, bytes), type(first)
                _ = msgpack_decode(first)
                obs = ultra_arrays_to_wire_observation(self._warmup_arrays(), self._hardware_model)
                obs[KEY_PROMPT] = ""
                obs.update(self._wire_extra_fields())
                ws.send(msgpack_encode(obs))
                ws.recv()
            LOGGER.info("Inference server warmup complete")
        except Exception as exc:
            LOGGER.error("Inference server warmup failed for %s: %s", self.predict_url, exc, exc_info=True)

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

    async def predict(
        self,
        raw_sample: dict[str, torch.Tensor],
        prompt: str,
        *,
        predict_call_multiplier: int = 1,
    ) -> RemotePolicyPrediction:
        assert predict_call_multiplier >= 1, "predict_call_multiplier must be at least 1"
        if predict_call_multiplier > 1:
            LOGGER.warning(
                "predict_call_multiplier=%s is ignored for websocket inference",
                predict_call_multiplier,
            )

        await self._ensure_ws()
        assert self._ws is not None

        arrays = self._encode_arrays(raw_sample)
        obs = ultra_arrays_to_wire_observation(arrays, self._hardware_model)
        obs[KEY_PROMPT] = prompt
        obs.update(self._wire_extra_fields())

        payload = msgpack_encode(obs)
        start_time_ns = time.time_ns()
        await self._ws.send(payload)
        response_raw = await self._ws.recv()
        end_time_ns = time.time_ns()

        total_latency_ms = (end_time_ns - start_time_ns) / 1e6
        assert isinstance(response_raw, bytes), type(response_raw)
        result = msgpack_decode(response_raw)
        assert isinstance(result, dict), f"unexpected response type {type(result)}"
        actions = result[KEY_ACTIONS]
        assert isinstance(actions, np.ndarray), f"actions must be ndarray, got {type(actions)}"
        infer_raw = result.get(KEY_INFERENCE_TIME)
        server_latency_ms = float(infer_raw) if infer_raw is not None else 0.0
        policy_id_used = str(result.get("policy_id", self._policy_id or ""))

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

    def _encode_arrays(self, raw_sample: dict[str, torch.Tensor]) -> dict[str, npt.NDArray[Any]]:
        arrays: dict[str, npt.NDArray[Any]] = {}
        for key, value in raw_sample.items():
            if "action" in key or "is_pad" in key:
                continue
            if isinstance(value, torch.Tensor):
                arr = value.cpu().numpy()
            else:
                arr = np.array(value)
            arrays[key] = arr
        return arrays


__all__ = [
    "DEFAULT_PREDICT_URL",
    "RemotePolicyClient",
    "RemotePolicyPrediction",
    "policy_ws_url",
]
