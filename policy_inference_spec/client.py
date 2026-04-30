from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import simplejpeg
import websockets

from policy_inference_spec.client_helpers import (
    DEFAULT_PREDICT_URL,
    _emit_server_error_verbatim,
    _log_server_config,
    _summarize_server_payload,
    _wire_camera_names,
    policy_ws_url,
)
from policy_inference_spec.codec import deserialize_from_msgpack, encode_image, serialize_to_msgpack
from policy_inference_spec.hardware_model import validate_wire_inference_request_frame, validate_wire_inference_response
from policy_inference_spec.protocol import (
    ACTION_KEY,
    CHUNK_ID_KEY,
    ACTION_PREFIX_KEY,
    ENDPOINT_KEY,
    ENDPOINT_REWARD,
    INFERENCE_TIME_KEY,
    JOINT_STATE_KEY,
    PREFIX_CHANGE_START_KEY,
    REWARDS_H_KEY,
    RewardSignal,
    STATUS_KEY,
    ServerFeature,
    ServerHandshake,
)
import asyncio

LOGGER = logging.getLogger(__name__)


def _wire_image_to_hwc_uint8(value: Any) -> npt.NDArray[np.uint8]:
    if isinstance(value, bytes):
        return np.ascontiguousarray(simplejpeg.decode_jpeg(value)).astype(np.uint8, copy=False)
    assert isinstance(value, np.ndarray), f"image value must be bytes or ndarray, got {type(value)}"
    if value.ndim == 4:
        assert value.shape[0] == 1, f"JPEG transport only supports batch size 1, got shape {value.shape}"
        value = value[0]
    assert value.ndim == 3, f"JPEG transport expects HWC or BHWC arrays, got shape {value.shape}"
    assert value.shape[-1] == 3, f"JPEG transport expects channel-last RGB arrays, got shape {value.shape}"
    assert value.dtype == np.uint8, f"JPEG transport expects uint8 arrays, got {value.dtype}"
    return np.ascontiguousarray(value)


@dataclass(frozen=True)
class RemotePolicyPrediction:
    actions_d: npt.NDArray[np.float32]
    total_latency_ms: float
    policy_id: str
    chunk_id: str | None


class InferenceServiceRestartedError(RuntimeError):
    pass


class RemotePolicyClient:
    _LATENCY_LOG_INTERVAL_S = 60.0

    def __init__(
        self,
        predict_url: str,
        *,
        policy_auth_headers: dict[str, str] | None = None,
    ) -> None:
        self.predict_url = policy_ws_url(predict_url)
        self._policy_auth_headers = policy_auth_headers or {}
        self._lock = asyncio.Lock()
        self._ws: Any = None
        self._server_config: ServerHandshake | None = None
        self._connected_url: str | None = None
        self._latency_window_started_at_s: float | None = None
        self._latency_total_ms: list[float] = []
        self._latency_server_ms: list[float] = []

    async def __aenter__(self) -> RemotePolicyClient:
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._close_ws()

    async def update_connection(
        self,
        *,
        predict_url: str,
        policy_auth_headers: dict[str, str] | None = None,
    ) -> None:
        self.predict_url = policy_ws_url(predict_url)
        self._policy_auth_headers = policy_auth_headers or {}
        await self._close_ws()

    def _headers(self) -> list[tuple[str, str]]:
        return [(k, v) for k, v in self._policy_auth_headers.items()]

    def _warn_on_camera_name_mismatch(self, wire_frame: dict[str, Any]) -> None:
        if self._server_config is None:
            return
        server_camera_names = self._server_config.camera_names
        sent_camera_names = _wire_camera_names(wire_frame)
        missing_camera_names = sorted(name for name in sent_camera_names if name not in set(server_camera_names))
        if missing_camera_names:
            LOGGER.warning(
                "Sending camera names not present in server config. sent=%s missing=%s server_camera_names=%s",
                sent_camera_names,
                missing_camera_names,
                sorted(server_camera_names),
            )

    def _encode_wire_frame_images(self, wire_frame: dict[str, Any]) -> dict[str, Any]:
        adapted = dict(wire_frame)
        for key, value in wire_frame.items():
            if not key.startswith("observation/") or key == JOINT_STATE_KEY:
                continue
            if isinstance(value, bytes):
                continue
            field = encode_image(_wire_image_to_hwc_uint8(value), jpeg_quality=75)
            assert field.codec == "jpeg", f"{key} must use jpeg transport"
            adapted[key] = field.data
        return adapted

    async def _ensure_ws(self) -> None:
        uri = self.predict_url
        if self._ws is not None and self._connected_url == uri:
            return
        if self._ws is not None:
            await self._close_ws()
        async with self._lock:
            self._ws = await websockets.connect(uri, additional_headers=self._headers())
            self._connected_url = uri
            first = await self._ws.recv()
        assert isinstance(first, bytes), type(first)
        server_config_payload = deserialize_from_msgpack(first)
        assert isinstance(server_config_payload, dict), "ServerConfig must be a dict"
        self._server_config = ServerHandshake.from_payload(server_config_payload)
        _log_server_config(self._server_config)

    async def _close_ws(self) -> None:
        self._log_latency_summary(force=True)
        async with self._lock:
            if self._ws is not None:
                await self._ws.close()
                self._ws = None
        self._server_config = None
        self._connected_url = None

    def _log_latency_summary(self, *, force: bool = False, now_s: float | None = None) -> None:
        if not self._latency_total_ms:
            return
        if self._latency_window_started_at_s is None:
            self._latency_window_started_at_s = now_s if now_s is not None else time.monotonic()
        elapsed_s = (now_s if now_s is not None else time.monotonic()) - self._latency_window_started_at_s
        if not force and elapsed_s < self._LATENCY_LOG_INTERVAL_S:
            return
        total = np.array(self._latency_total_ms, dtype=np.float64)
        server = np.array(self._latency_server_ms, dtype=np.float64)
        network = total - server
        LOGGER.info(
            "Inference latency stats: n=%d window=%.0fs mean=%.1fms min=%.1fms p50=%.1fms p95=%.1fms max=%.1fms server_mean=%.1fms network_mean=%.1fms",
            total.size,
            elapsed_s,
            float(total.mean()),
            float(total.min()),
            float(np.percentile(total, 50)),
            float(np.percentile(total, 95)),
            float(total.max()),
            float(server.mean()),
            float(network.mean()),
        )
        self._latency_window_started_at_s = now_s if now_s is not None else time.monotonic()
        self._latency_total_ms.clear()
        self._latency_server_ms.clear()

    def _record_latency(self, *, total_latency_ms: float, server_latency_ms: float, now_s: float | None = None) -> None:
        if self._latency_window_started_at_s is None:
            self._latency_window_started_at_s = now_s if now_s is not None else time.monotonic()
        self._latency_total_ms.append(total_latency_ms)
        self._latency_server_ms.append(server_latency_ms)
        self._log_latency_summary(now_s=now_s)

    @staticmethod
    def _close_code(exc: websockets.ConnectionClosedError) -> int | None:
        if exc.rcvd is not None:
            return exc.rcvd.code
        if exc.sent is not None:
            return exc.sent.code
        return None

    async def _raise_if_service_restarted(self, exc: websockets.ConnectionClosedError) -> None:
        if self._close_code(exc) != 1012:
            raise exc
        await self._close_ws()
        raise InferenceServiceRestartedError("Inference service restarted during prediction") from exc

    async def reward(
        self,
        rewards_h: list[float] | tuple[float, ...] = (1.0,),
        description: str | None = None,
        *,
        chunk_id: str,
    ) -> None:
        await self._ensure_ws()
        assert self._server_config is not None
        if not self._server_config.supports(ServerFeature.REWARDS):
            LOGGER.warning("Dropping reward because server does not advertise %s support", ServerFeature.REWARDS)
            return
        reward_signal = RewardSignal(
            chunk_id=chunk_id,
            rewards_h=tuple(float(reward) for reward in rewards_h),
            description=description,
        )
        async with self._lock:
            assert self._ws is not None
            await self._ws.send(serialize_to_msgpack(reward_signal.to_payload()))
            response_raw = await self._ws.recv()
        if isinstance(response_raw, str):
            _emit_server_error_verbatim(response_raw)
            raise AssertionError("unexpected text response from inference server")
        response = deserialize_from_msgpack(response_raw)
        _emit_server_error_verbatim(response)
        assert isinstance(response, dict), f"unexpected reward response type {type(response)}"
        assert response.get(STATUS_KEY) == "ok", f"unexpected reward response payload: {response}"
        if response.get(ENDPOINT_KEY) == ENDPOINT_REWARD:
            rewards_ack = response.get(REWARDS_H_KEY)
            assert rewards_ack is None or isinstance(rewards_ack, list), f"{REWARDS_H_KEY} must be list[float]"

    async def predict(
        self,
        wire_frame: dict[str, Any],
    ) -> RemotePolicyPrediction:
        try:
            await self._ensure_ws()
            wire_frame = self._encode_wire_frame_images(wire_frame)
            validate_wire_inference_request_frame(wire_frame)
            self._warn_on_camera_name_mismatch(wire_frame)
            payload = serialize_to_msgpack(wire_frame)
            start_time_ns = time.time_ns()
            async with self._lock:
                assert self._ws is not None
                await self._ws.send(payload)
                response_raw = await self._ws.recv()
            end_time_ns = time.time_ns()
        except websockets.ConnectionClosedError as exc:
            await self._raise_if_service_restarted(exc)

        total_latency_ms = (end_time_ns - start_time_ns) / 1e6
        if isinstance(response_raw, str):
            _emit_server_error_verbatim(response_raw)
            raise AssertionError("unexpected text response from inference server")
        result = deserialize_from_msgpack(response_raw)
        _emit_server_error_verbatim(result)
        assert isinstance(result, dict), f"unexpected response type {type(result)}"
        try:
            validate_wire_inference_response(result)
        except AssertionError as exc:
            LOGGER.error("Malformed inference response: %s", _summarize_server_payload(result))
            raise AssertionError(f"{exc}") from exc
        actions = result[ACTION_KEY]
        infer_raw = result.get(INFERENCE_TIME_KEY)
        server_latency_ms = 0.0
        if infer_raw is not None:
            assert isinstance(infer_raw, (int, float)), f"{INFERENCE_TIME_KEY} must be numeric"
            server_latency_ms = float(infer_raw)
        policy_id_used = str(result.get("policy_id", ""))
        chunk_id_raw = result.get(CHUNK_ID_KEY)
        chunk_id_used: str | None = None
        if chunk_id_raw is not None:
            assert isinstance(chunk_id_raw, str) and chunk_id_raw, f"{CHUNK_ID_KEY} must be a non-empty str"
            chunk_id_used = chunk_id_raw
        self._record_latency(total_latency_ms=total_latency_ms, server_latency_ms=server_latency_ms)

        actions_d = np.array(actions, dtype=np.float32)
        )
        return RemotePolicyPrediction(
            actions_d=actions_d,
            total_latency_ms=total_latency_ms,
            policy_id=policy_id_used,
            chunk_id=chunk_id_used,
        )


__all__ = [
    "DEFAULT_PREDICT_URL",
    "InferenceServiceRestartedError",
    "RemotePolicyClient",
    "RemotePolicyPrediction",
    "policy_ws_url",
]
