DEFAULT_INFERENCE_SERVER_PORT = 18090
OBS_JOINT_POSITION_KEY = "observation/joint_position"  # State vector (Actually contains more than just qpos)
PROMPT_KEY = "prompt"
ACTIONS_KEY = "actions"
INFERENCE_TIME_KEY = "inference_time"
ENDPOINT_KEY = "endpoint"
ENDPOINT_RESET = "reset"
ENDPOINT_TELEMETRY = "telemetry"
MODEL_ID_KEY = "model_id"

__all__ = [
    "ACTIONS_KEY",
    "DEFAULT_INFERENCE_SERVER_PORT",
    "ENDPOINT_KEY",
    "ENDPOINT_RESET",
    "ENDPOINT_TELEMETRY",
    "INFERENCE_TIME_KEY",
    "MODEL_ID_KEY",
    "OBS_JOINT_POSITION_KEY",
    "PROMPT_KEY",
]
