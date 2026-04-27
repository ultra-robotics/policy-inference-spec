from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import policy_inference_spec.replay_rrd as replay_rrd
from policy_inference_spec.client import RemotePolicyPrediction


@pytest.mark.asyncio
async def test_replay_recording_orchestrates_predictions_and_logging(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    recording_path = tmp_path / "input.rrd"
    output_path = tmp_path / "output.rrd"
    recording_path.write_bytes(b"rrd")
    samples = [
        {
            "ts": pd.Timestamp("2026-04-01T00:00:01Z"),
            "/commanded_qpos/left_arm:Scalars:scalars": np.zeros((4, 7), dtype=np.float32),
        },
        {
            "ts": pd.Timestamp("2026-04-01T00:00:02Z"),
            "/commanded_qpos/left_arm:Scalars:scalars": np.ones((4, 7), dtype=np.float32),
        },
    ]
    captured: dict[str, object] = {}

    class FakeReplayer:
        def __init__(self, path: Path, feature_bundle: object, hz: int, publish_hz: float) -> None:
            assert path == recording_path
            assert hz == 50
            assert publish_hz == 1.0
            captured["feature_bundle_name"] = getattr(feature_bundle, "name")

        def __iter__(self):
            return iter(samples)

    async def fake_predict_sample(
        feature_bundle: object,
        sample: object,
        predict_url: str,
        policy_id: str,
        prompt: str,
        prefix_change_start: int,
    ) -> RemotePolicyPrediction:
        assert predict_url == "ws://127.0.0.1:18090/ws"
        assert policy_id == "policy-id"
        assert prompt == replay_rrd.DEFAULT_PROMPT
        assert prefix_change_start == 6
        assert sample in samples
        action_dim = getattr(feature_bundle, "action_dim")
        return RemotePolicyPrediction(
            actions_d=np.zeros((4, action_dim), dtype=np.float32),
            context_embeddings=np.zeros((2, 128), dtype=np.float32),
            total_latency_ms=1.5,
            policy_id=policy_id,
        )

    def fake_log_to_rerun(
        output: Path,
        feature_bundle: object,
        replay_samples: list[dict[str, object]],
        predictions: list[RemotePolicyPrediction],
        hz: int,
    ) -> None:
        captured["output_path"] = output
        captured["logged_sample_count"] = len(replay_samples)
        captured["logged_prediction_count"] = len(predictions)
        captured["logged_hz"] = hz
        captured["logged_bundle_name"] = getattr(feature_bundle, "name")

    monkeypatch.setattr(replay_rrd, "RerunReplayer", FakeReplayer)
    monkeypatch.setattr(replay_rrd, "predict_sample", fake_predict_sample)
    monkeypatch.setattr(replay_rrd, "log_to_rerun", fake_log_to_rerun)

    summary = await replay_rrd.replay_recording(
        recording_path=recording_path,
        output_path=output_path,
        predict_url="ws://127.0.0.1:18090/ws",
        policy_id="policy-id",
        prefix_change_start=6,
        max_samples=10,
    )

    assert summary.sample_count == 2
    assert summary.first_timestamp == pd.Timestamp("2026-04-01T00:00:01Z")
    assert summary.last_timestamp == pd.Timestamp("2026-04-01T00:00:02Z")
    assert summary.recording_duration == pd.Timedelta(seconds=1)
    assert summary.wall_time_s >= 0.0
    assert summary.speed_ratio >= 0.0
    assert summary.recording_path == recording_path
    assert summary.output_path == output_path
    assert captured == {
        "feature_bundle_name": "gen2-training-32d-state",
        "output_path": output_path,
        "logged_sample_count": 2,
        "logged_prediction_count": 2,
        "logged_hz": 50,
        "logged_bundle_name": "gen2-training-32d-state",
    }


@pytest.mark.asyncio
async def test_replay_recording_requires_samples(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    recording_path = tmp_path / "input.rrd"
    recording_path.write_bytes(b"rrd")

    class EmptyReplayer:
        def __init__(self, path: Path, feature_bundle: object, hz: int, publish_hz: float) -> None:
            _ = path, feature_bundle, hz, publish_hz

        def __iter__(self):
            return iter(())

    monkeypatch.setattr(replay_rrd, "RerunReplayer", EmptyReplayer)

    with pytest.raises(AssertionError, match="No replay samples were produced"):
        await replay_rrd.replay_recording(recording_path=recording_path, output_path=tmp_path / "output.rrd")


def test_build_action_prefix_returns_none_when_disabled() -> None:
    action_hd = np.zeros((50, 25), dtype=np.float32)
    assert replay_rrd.build_action_prefix(action_hd, 0) is None


def test_build_action_prefix_matches_pi_prefix_layout() -> None:
    action_hd = np.arange(50 * 25, dtype=np.float32).reshape(50, 25)
    action_prefix = replay_rrd.build_action_prefix(action_hd, 7)
    assert action_prefix is not None
    np.testing.assert_allclose(action_prefix[:43], action_hd[:43])
    np.testing.assert_allclose(action_prefix[43:], np.ones((7, 25), dtype=np.float32))
