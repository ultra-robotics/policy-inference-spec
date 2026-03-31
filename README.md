# policy-inference-spec

Shared Python package for policy inference over WebSocket: msgpack frames, numpy tagging, and strict validation of wire keys/shapes.

## Install

From a checkout of this repo:

```bash
pip install -e .
```

## Wire protocol


- **Transport:** WebSocket, binary frames, **msgpack** payloads.
- **Handshake:** client connects to `wss://<host>/ws` (or `ws://`). The server sends the **first** message: a **ServerConfig** dict (camera names, image resolution, action space, etc.).
- **Auth:** clients that need an API key send header **`x-api-key`**.
- **NumPy:** arrays are encoded with a **`__ndarray__`** tag: `data` (bytes), `dtype`, `shape` (see `serialize_to_msgpack` / `deserialize_from_msgpack` in `policy_inference_spec.protocol`).
- **Inference request** (msgpack dict): at minimum
  - `observation/joint_position` — float32 **ndarray** joint vector, **1-D** length 97, encoded with `__ndarray__`
  - `observation/<camera_name>` — JPEG **bytes** (produced with `encode_image`)
  - `prompt` — single language string for the policy
  - `model_id` — policy id string (may be empty)
- **Inference response:** `actions` (2-D ndarray; second dim **25**), `inference_time` (server-side ms), and **`policy_id`** (string).
- **Control:** `{"endpoint": "reset"}` → `{"status": "ok"}`; `{"endpoint": "telemetry", ...}` → `{"status": "ok"}`.

Strict validation helpers live in `policy_inference_spec.hardware_model` (`validate_wire_inference_request_frame`, `validate_wire_inference_response`).
Wire and endpoint constants live in `policy_inference_spec.constants`.

## Features to be added in future

- **H.264 / `__video_frame__`** streaming frames. This package handles **JPEG bytes** and **raw uint8 ndarray** images for `observation/*` keys.
- **Real Time Chunking** support whereby action prefixes can be sent in the inference request

## Package layout

| Module | Role |
|--------|------|
| `constants.py` | Shared wire keys, endpoint names, and default server port |
| `protocol.py` | `encode_image`, `serialize_to_msgpack`, `deserialize_from_msgpack`, and ndarray msgpack tagging |
| `hardware_model.py` | Hardware-model-aware shapes and strict request/response validation |
| `client.py` | `RemotePolicyClient` (async transport + validation), `policy_ws_url`, warmup |

## License

See `LICENSE`.
