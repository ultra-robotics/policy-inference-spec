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
- **NumPy:** arrays are encoded with a **`__ndarray__`** tag: `data` (bytes), `dtype`, `shape` (see `policy_inference_spec.protocol`).
- **Inference request** (msgpack dict): at minimum
  - `observation/joint_position` — float32 joint vector **1-D** length 60 (gen1) or 89 (gen2)
  - `observation/<camera_name>` — JPEG **bytes** (encoded with `encode_ndarray` / decoded with `chw_from_wire_image`)
  - `prompt` — single language string for the policy
  - `model_id` — policy id string (may be empty)
  - **`hardware_model`** — required for **gen1** (`"gen1"`); **omit** for default **gen2** layout
- **Inference response:** `actions` (2-D ndarray; second dim **22** for gen1-style policies or **25** for gen2), `inference_time` (server-side ms), and **`policy_id`** (string).
- **Control:** `{"endpoint": "reset"}` → `{"status": "ok"}`; `{"endpoint": "telemetry", ...}` → `{"status": "ok"}`.

Strict validation helpers live in `policy_inference_spec.schema` (`validate_wire_inference_request_frame`, `validate_wire_inference_response`).

## Features to be added in future

- **H.264 / `__video_frame__`** streaming frames. This package handles **JPEG bytes** and **raw uint8 ndarray** images for `observation/*` keys.
- **Real Time Chunking** support whereby action prefixes can be sent in the inference request

## Package layout

| Module | Role |
|--------|------|
| `protocol.py` | msgpack encode/decode, `__ndarray__`, `NdarrayField`, JPEG/raw image helpers |
| `schema.py` | Wire key constants, per-generation shapes, strict request/response validation |
| `client.py` | `RemotePolicyClient` (async transport + validation), `policy_ws_url`, warmup |
| `constants.py` | Default inference port |

## License

See `LICENSE`.
