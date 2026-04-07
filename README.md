# policy-inference-spec

Shared Python package for the current policy-inference wire format and helper code: WebSocket transport, msgpack + NumPy encoding, strict request/response validation, an async client, feature-engineering utilities, and a minimal example server.

## Install

Requires Python 3.12+.

```bash
pip install -e .
```

Optional test dependencies:

```bash
pip install -e '.[dev]'
```

## Current Wire Protocol

- **Transport:** WebSocket binary frames carrying msgpack payloads.
- **URL handling:** `policy_ws_url()` accepts `ws://` and `wss://` URLs. If the path is empty or `/`, it rewrites the URL to end in `/ws`.
- **Auth:** `RemotePolicyClient` forwards any `policy_auth_headers` as WebSocket headers. The included smoke test uses `x-api-key`.
- **Handshake:** the server sends the first frame. `ServerHandshake` currently contains:
  - `camera_names` (required `list[str]`)
  - `image_resolution` (optional `[height, width]`)
  - `action_space` (defaults to `"joint_position"`)
  - `needs_wrist_camera` (defaults to `true`)
  - `n_external_cameras` (defaults to `1`)
  - `server_features` (optional `list[str]`)
- **Current hardware model:** the package currently models only `HardwareModel.GEN2`:
  - `state_dim = 97`
  - `action_dim = 25`
  - `image_resolution = (360, 640)`
  - camera names:
    - `images/main_image`
    - `images/left_wrist_image`
    - `images/right_wrist_image`

### Inference Request

`validate_wire_inference_request_frame()` requires an inference frame to contain exactly these keys:

- `observation/state`
- `observation/images/main_image`
- `observation/images/left_wrist_image`
- `observation/images/right_wrist_image`
- `prompt`
- `model_id`

Current validation rules:

- `observation/state` must be a 1-D `numpy.ndarray` with shape `(97,)`.
- `prompt` must be `str`.
- `model_id` must be `str` and may be empty.
- Each image field must be either JPEG `bytes` or a `numpy.ndarray`.
- Extra keys are rejected, including `hardware_model`.
- `endpoint` must not be present on inference frames.

`RemotePolicyClient.predict()` accepts image arrays in HWC or single-item BHWC `uint8` format and converts them to JPEG bytes at quality 75 before sending. It does not resize images.

### Msgpack / NumPy Encoding

- `serialize_to_msgpack()` encodes `numpy.ndarray` values with a flat `__ndarray__` tag containing `data`, `dtype`, and `shape`.
- `deserialize_from_msgpack()` reconstructs arrays from that tag.
- The codec currently supports only `uint8` and `float32` ndarrays.
- `encode_image()` produces JPEG bytes plus shape/dtype metadata in an `NdarrayField`, but the current `RemotePolicyClient` sends raw JPEG bytes on the wire for image observations.

### Inference Response

`validate_wire_inference_response()` currently requires:

- `action`: 2-D floating `numpy.ndarray` whose second dimension is `25`
- `context_embeddings`: floating `numpy.ndarray` with shape `(2, 128)`

Optional response fields:

- `inference_time`: numeric
- `policy_id`: `str`

`RemotePolicyClient.predict()` returns a `RemotePolicyPrediction` with:

- `actions_d`
- `context_embeddings`
- `total_latency_ms`
- `policy_id`

### Reward And Control Messages

- `RemotePolicyClient.reward(value=1.0, description=None)` sends `{"endpoint": "reward", ...}` only if the handshake advertises `"rewards"` in `server_features`.
- If the server does not advertise reward support, the client drops the message and logs a warning.
- `RewardSignal` serializes to:
  - `{"endpoint": "reward", "reward": <float>}`
  - optionally with `"description": <str>`
- The example server acknowledges rewards with `{"endpoint": "reward", "status": "ok", "reward": ...}` and echoes `description` when present.
- `ENDPOINT_RESET` and `ENDPOINT_TELEMETRY` constants exist in the protocol, and the example server replies `{"status": "ok"}` to those messages. There are no dedicated reset/telemetry client helpers in the current package.

## Included Modules

| Module | Role |
|--------|------|
| `policy_inference_spec.protocol` | Wire keys, handshake and reward dataclasses, server-feature enum, and protocol type aliases |
| `policy_inference_spec.codec` | Msgpack codec, ndarray tagging, and JPEG image encoding helper |
| `policy_inference_spec.hardware_model` | `gen2` hardware-model metadata and strict request/response validators |
| `policy_inference_spec.client` | Async `RemotePolicyClient`, prediction result type, restart error, and URL normalization helper |
| `policy_inference_spec.client_helpers` | Default predict URL and logging / payload helper functions used by the client |
| `policy_inference_spec.feature_engineering` | Feature-bundle schemas, action parsing, and image preprocessing utilities for training/data prep |
| `policy_inference_spec.smoke` | Smoke-test CLI for sending example predict requests |
| `server.minimal` | Minimal example inference server and example linear policy |

## Quick Checks

Run tests:

```bash
uv run --extra dev pytest tests/ -q
```

Run the smoke test against the local example server:

```bash
uv run python -m policy_inference_spec.smoke
```

## License

See `LICENSE`.
