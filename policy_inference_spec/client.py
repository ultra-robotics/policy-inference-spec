from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import websockets
from websockets.sync.client import connect as ws_connect_sync

from policy_inference_spec.client_helpers import (
    DEFAULT_PREDICT_URL,
    _emit_server_error_verbatim,
    _log_server_config,
    _random_warmup_wire_frame,
    _server_image_resolution,
    _summarize_server_payload,
    _wire_camera_names,
    policy_ws_url,
)
from policy_inference_spec.protocol import chw_from_wire_image, encode_ndarray, msgpack_decode, msgpack_encode
from policy_inference_spec.schema import (
    DEFAULT_HARDWARE_MODEL,
    HardwareModel,
    KEY_ACTIONS,
    KEY_INFERENCE_TIME,
    KEY_OBS_JOINT_POSITION,
    validate_wire_inference_request_frame,
    validate_wire_inference_response,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RemotePolicyPrediction:
    actions_d: npt.NDArray[np.float32]
    total_latency_ms: float
    policy_id: str


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
        self._ws: Any = None
        self._server_config: dict[str, Any] | None = None
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

    def update_connection(self, *, predict_url: str, policy_auth_headers: dict[str, str] | None = None) -> None:
        self.predict_url = policy_ws_url(predict_url)
        self._policy_auth_headers = policy_auth_headers or {}
        self._connected_url = None

    def _headers(self) -> list[tuple[str, str]]:
        return [(k, v) for k, v in self._policy_auth_headers.items()]

    def _warn_on_camera_name_mismatch(self, wire_frame: dict[str, Any]) -> None:
        if self._server_config is None:
            return
        server_camera_names = self._server_config.get("camera_names")
        if not isinstance(server_camera_names, list) or not all(isinstance(name, str) for name in server_camera_names):
            return
        sent_camera_names = _wire_camera_names(wire_frame)
        missing_camera_names = sorted(name for name in sent_camera_names if name not in set(server_camera_names))
        if missing_camera_names:
            LOGGER.warning(
                "Sending camera names not present in server config. sent=%s missing=%s server_camera_names=%s",
                sent_camera_names,
                missing_camera_names,
                sorted(server_camera_names),
            )

    def _adapt_wire_frame_to_server_config(self, wire_frame: dict[str, Any]) -> dict[str, Any]:
        image_resolution = _server_image_resolution(self._server_config)
        if image_resolution is None:
            return wire_frame

        target_h, target_w = image_resolution
        adapted = dict(wire_frame)
        for key, value in wire_frame.items():
            if not key.startswith("observation/") or key == KEY_OBS_JOINT_POSITION:
                continue
            chw = chw_from_wire_image(value, (3, target_h, target_w))
            field = encode_ndarray(chw, jpeg_quality=75)
            assert field.codec == "jpeg", f"{key} must use jpeg transport"
            adapted[key] = field.data
        return adapted

    def warmup(self, *, hardware_model: str | HardwareModel = DEFAULT_HARDWARE_MODEL) -> bool:
        try:
            hm = HardwareModel(hardware_model)
            uri = self.predict_url
            with ws_connect_sync(uri, additional_headers=self._headers()) as ws:
                first = ws.recv()
                assert isinstance(first, bytes), type(first)
                server_config = msgpack_decode(first)
                assert isinstance(server_config, dict), "ServerConfig must be a dict"
                _log_server_config(server_config)
                self._server_config = server_config
                wire_frame = _random_warmup_wire_frame(
                    hm,
                    image_resolution=_server_image_resolution(server_config),
                )
                ws.send(msgpack_encode(wire_frame))
                response_raw = ws.recv()
                if isinstance(response_raw, bytes):
                    response = msgpack_decode(response_raw)
                    _emit_server_error_verbatim(response)
                else:
                    _emit_server_error_verbatim(response_raw)
            LOGGER.info("Inference server warmup complete")
            return True
        except Exception as exc:
            LOGGER.error("Inference server warmup failed for %s: %s", self.predict_url, exc, exc_info=True)
            return False

    async def _ensure_ws(self) -> None:
        uri = self.predict_url
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
        _log_server_config(self._server_config)

    async def _close_ws(self) -> None:
        self._log_latency_summary(force=True)
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

    async def predict(self, wire_frame: dict[str, Any]) -> RemotePolicyPrediction:
        try:
            await self._ensure_ws()
            assert self._ws is not None
            wire_frame = self._adapt_wire_frame_to_server_config(wire_frame)
            validate_wire_inference_request_frame(wire_frame)
            self._warn_on_camera_name_mismatch(wire_frame)
            payload = msgpack_encode(wire_frame)
            start_time_ns = time.time_ns()
            await self._ws.send(payload)
            response_raw = await self._ws.recv()
            end_time_ns = time.time_ns()
        except websockets.ConnectionClosedError as exc:
            await self._raise_if_service_restarted(exc)

        total_latency_ms = (end_time_ns - start_time_ns) / 1e6
        if isinstance(response_raw, str):
            _emit_server_error_verbatim(response_raw)
            raise AssertionError("unexpected text response from inference server")
        result = msgpack_decode(response_raw)
        _emit_server_error_verbatim(result)
        assert isinstance(result, dict), f"unexpected response type {type(result)}"
        try:
            validate_wire_inference_response(result)
        except AssertionError as exc:
            LOGGER.error("Malformed inference response: %s", _summarize_server_payload(result))
            raise AssertionError(f"{exc}") from exc
        actions = result[KEY_ACTIONS]
        infer_raw = result.get(KEY_INFERENCE_TIME)
        server_latency_ms = float(infer_raw) if infer_raw is not None else 0.0
        policy_id_used = str(result.get("policy_id", ""))
        self._record_latency(total_latency_ms=total_latency_ms, server_latency_ms=server_latency_ms)

        actions_d = np.array(actions, dtype=np.float32)
        return RemotePolicyPrediction(
            actions_d=actions_d,
            total_latency_ms=total_latency_ms,
            policy_id=policy_id_used,
        )


__all__ = [
    "DEFAULT_PREDICT_URL",
    "InferenceServiceRestartedError",
    "RemotePolicyClient",
    "RemotePolicyPrediction",
    "policy_ws_url",
]
