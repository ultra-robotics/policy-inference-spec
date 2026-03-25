# policy-inference-spec

Shared Python package for **Pi-compatible** policy inference over WebSocket: msgpack frames, numpy tagging, and Ultra’s field mapping to gateway camera names.

## Install

From a checkout of this repo:

```bash
pip install -e .
```

In the Ultra monorepo this is installed as an editable path dependency (`libs/submodules/policy-inference-spec`).

## Wire protocol (implemented)

This matches the behavior described in the Physical Intelligence gateway docs for the core path:

- **Transport:** WebSocket, binary frames, **msgpack** payloads.
- **Handshake:** client connects to `wss://<host>/ws` (or `ws://`). The server sends the **first** message: a **ServerConfig** dict (camera names, image resolution, action space, etc.).
- **Auth:** clients that need an API key send header **`x-api-key`** (Ultra maps station config `POLICY_AUTH_TOKEN` to this header).
- **NumPy:** arrays are encoded with a **`__ndarray__`** tag: `data` (bytes), `dtype`, `shape` (see `policy_inference_spec.protocol`).
- **Inference request** (msgpack dict): at minimum
  - `observation/joint_position` — float32 joint vector (1-D or squeezed batch)
  - `observation/<camera_name>` — JPEG **bytes** or raw image payload (see `encode_ndarray` / `chw_from_wire_image`)
  - `prompt` — single language string for the policy (no separate metadata dict on the wire)
  - **`hardware_model`** (optional) — `"gen1"` or `"gen2"`; **omit** for default **`gen2`** tensor layout
  - **`model_id`** (optional) — which policy/checkpoint the server should run  
  **`RemotePolicyClient.predict(raw_sample, prompt, ...)`** takes that string as-is; call sites (e.g. Dora `run_policy_remote`) build it from domain fields before calling.
- **Inference response:** `actions` (ndarray), `inference_time` (server-side ms, optional), and Ultra adds **`policy_id`**.
- **Control:** `{"endpoint": "reset"}` → `{"status": "ok"}`; `{"endpoint": "telemetry", ...}` → `{"status": "ok"}`.

## Optional Pi features not implemented here

- **H.264 / `__video_frame__`** streaming frames. This package handles **JPEG bytes** and **raw uint8 ndarray** images for `observation/*` keys.

## Package layout

| Module | Role |
|--------|------|
| `protocol.py` | msgpack encode/decode, `__ndarray__`, `NdarrayField`, JPEG/raw image helpers |
| `wire.py` | Gateway key names, Ultra tensor shapes ↔ `observation/<camera>_left` mapping, `server_config_for_hardware_model` |
| `client.py` | `RemotePolicyClient` (async), `policy_ws_url`, warmup |
| `constants.py` | Default inference port |

## License

See `LICENSE`.
