# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit-tests for the `PolicyServer` core logic.
Monkey-patch the `policy` attribute with a stub so that no real model inference is performed.
"""

from __future__ import annotations

import time

import pytest
import torch

from lerobot.configs.types import PolicyFeature
from lerobot.utils.constants import OBS_STATE
from tests.utils import require_package

# -----------------------------------------------------------------------------
# Test fixtures
# -----------------------------------------------------------------------------


class MockPolicy:
    """A minimal mock for an actual policy, returning zeros.
    Refer to tests/policies for tests of the individual policies supported."""

    class _Config:
        robot_type = "dummy_robot"

        @property
        def image_features(self) -> dict[str, PolicyFeature]:
            """Empty image features since this test doesn't use images."""
            return {}

    def predict_action_chunk(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return a chunk of 20 dummy actions."""
        batch_size = len(observation[OBS_STATE])
        return torch.zeros(batch_size, 20, 6)

    def __init__(self):
        self.config = self._Config()

    def to(self, *args, **kwargs):
        # The server calls `policy.to(device)`. This stub ignores it.
        return self

    def model(self, batch: dict) -> torch.Tensor:
        # Return a chunk of 20 dummy actions.
        batch_size = len(batch["robot_type"])
        return torch.zeros(batch_size, 20, 6)


@pytest.fixture
@require_package("grpcio", "grpc")
def policy_server():
    """Fresh `PolicyServer` instance with a stubbed-out policy model."""
    # Import only when the test actually runs (after decorator check)
    from lerobot.async_inference.configs import PolicyServerConfig
    from lerobot.async_inference.policy_server import PolicyServer

    test_config = PolicyServerConfig(host="localhost", port=9999)
    server = PolicyServer(test_config)
    # Replace the real policy with our fast, deterministic stub.
    server.policy = MockPolicy()
    server.actions_per_chunk = 20
    server.device = "cpu"

    # Add mock lerobot_features that the observation similarity functions need
    server.lerobot_features = {
        OBS_STATE: {
            "dtype": "float32",
            "shape": [6],
            "names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        }
    }

    return server


# -----------------------------------------------------------------------------
# Helper utilities for tests
# -----------------------------------------------------------------------------


def _make_obs(state: torch.Tensor, timestep: int = 0, must_go: bool = False):
    """Create a TimedObservation with a given state vector."""
    # Import only when needed
    from lerobot.async_inference.helpers import TimedObservation

    return TimedObservation(
        observation={
            "joint1": state[0].item() if len(state) > 0 else 0.0,
            "joint2": state[1].item() if len(state) > 1 else 0.0,
            "joint3": state[2].item() if len(state) > 2 else 0.0,
            "joint4": state[3].item() if len(state) > 3 else 0.0,
            "joint5": state[4].item() if len(state) > 4 else 0.0,
            "joint6": state[5].item() if len(state) > 5 else 0.0,
        },
        timestamp=time.time(),
        timestep=timestep,
        must_go=must_go,
    )


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_policy_server_acp_config_validation_and_round_trip():
    from lerobot.async_inference.configs import ACPInferenceConfig, PolicyServerConfig

    with pytest.raises(ValueError, match="requires.*use_cfg=true.*batched_cfg=true"):
        PolicyServerConfig(acp_inference=ACPInferenceConfig(enable=True))

    config = PolicyServerConfig(
        acp_inference=ACPInferenceConfig(
            enable=True,
            use_cfg=True,
            batched_cfg=True,
            cfg_beta=1.5,
            profile=True,
            profile_save_chunks=False,
        )
    )
    restored = PolicyServerConfig.from_dict(config.to_dict())

    assert restored.acp_inference.cfg_beta == 1.5
    assert restored.acp_inference.profile is True
    assert restored.acp_inference.profile_save_chunks is False


def test_time_action_chunk(policy_server):
    """Verify that `_time_action_chunk` assigns correct timestamps and timesteps."""
    start_ts = time.time()
    start_t = 10
    # A chunk of 3 action tensors.
    action_tensors = [torch.randn(6) for _ in range(3)]

    timed_actions = policy_server._time_action_chunk(start_ts, action_tensors, start_t)

    assert len(timed_actions) == 3
    # Check timesteps
    assert [ta.get_timestep() for ta in timed_actions] == [10, 11, 12]
    # Check timestamps
    expected_timestamps = [
        start_ts,
        start_ts + policy_server.config.environment_dt,
        start_ts + 2 * policy_server.config.environment_dt,
    ]
    for ta, expected_ts in zip(timed_actions, expected_timestamps, strict=True):
        assert abs(ta.get_timestamp() - expected_ts) < 1e-6


def test_maybe_enqueue_observation_must_go(policy_server):
    """An observation with `must_go=True` is always enqueued."""
    obs = _make_obs(torch.zeros(6), must_go=True)
    assert policy_server._enqueue_observation(obs) is True
    assert policy_server.observation_queue.qsize() == 1
    assert policy_server.observation_queue.get_nowait() is obs


def test_must_go_clears_predicted_timesteps(policy_server):
    """must_go should allow timestep reuse across episodes."""
    policy_server._predicted_timesteps.add(35)
    policy_server.last_processed_obs = _make_obs(torch.zeros(6), timestep=35)

    obs = _make_obs(torch.ones(6) * 5, timestep=35, must_go=True)
    assert policy_server._enqueue_observation(obs) is True
    assert 35 not in policy_server._predicted_timesteps
    assert policy_server.last_processed_obs is None


def test_maybe_enqueue_observation_dissimilar(policy_server):
    """A dissimilar observation (not `must_go`) is enqueued."""
    # Set a last predicted observation.
    policy_server.last_processed_obs = _make_obs(torch.zeros(6))
    # Create a new, dissimilar observation.
    new_obs = _make_obs(torch.ones(6) * 5)  # High norm difference

    assert policy_server._enqueue_observation(new_obs) is True
    assert policy_server.observation_queue.qsize() == 1


def test_maybe_enqueue_observation_is_skipped(policy_server):
    """A similar observation (not `must_go`) is skipped."""
    # Set a last predicted observation.
    policy_server.last_processed_obs = _make_obs(torch.zeros(6))
    # Create a new, very similar observation.
    new_obs = _make_obs(torch.zeros(6) + 1e-4)

    assert policy_server._enqueue_observation(new_obs) is False
    assert policy_server.observation_queue.empty() is True


def test_obs_sanity_checks(policy_server):
    """Unit-test the private `_obs_sanity_checks` helper."""
    prev = _make_obs(torch.zeros(6), timestep=0)

    # Case 1 – timestep already predicted
    policy_server._predicted_timesteps.add(1)
    obs_same_ts = _make_obs(torch.ones(6), timestep=1)
    assert policy_server._obs_sanity_checks(obs_same_ts, prev) is False

    # Case 2 – observation too similar
    policy_server._predicted_timesteps.clear()
    obs_similar = _make_obs(torch.zeros(6) + 1e-4, timestep=2)
    assert policy_server._obs_sanity_checks(obs_similar, prev) is False

    # Case 3 – genuinely new & dissimilar observation passes
    obs_ok = _make_obs(torch.ones(6) * 5, timestep=3)
    assert policy_server._obs_sanity_checks(obs_ok, prev) is True


def test_predict_action_chunk(monkeypatch, policy_server):
    """End-to-end test of `_predict_action_chunk` with a stubbed _get_action_chunk."""
    # Import only when needed
    from lerobot.async_inference.policy_server import PolicyServer

    # Force server to act-style policy; patch method to return deterministic tensor
    policy_server.policy_type = "act"
    # NOTE(Steven): Smelly tests as the Server is a state machine being partially mocked. Adding these processors as a quick fix.
    policy_server.preprocessor = lambda obs: obs
    policy_server.postprocessor = lambda tensor: tensor
    action_dim = 6
    batch_size = 1
    actions_per_chunk = policy_server.actions_per_chunk

    def _fake_get_action_chunk(_self, _obs, _type="act"):
        return torch.zeros(batch_size, actions_per_chunk, action_dim)

    monkeypatch.setattr(PolicyServer, "_get_action_chunk", _fake_get_action_chunk, raising=True)

    obs = _make_obs(torch.zeros(6), timestep=5)
    timed_actions = policy_server._predict_action_chunk(obs)

    assert len(timed_actions) == actions_per_chunk
    assert [ta.get_timestep() for ta in timed_actions] == list(range(5, 5 + actions_per_chunk))

    for i, ta in enumerate(timed_actions):
        expected_ts = obs.get_timestamp() + i * policy_server.config.environment_dt
        assert abs(ta.get_timestamp() - expected_ts) < 1e-6


def test_get_action_chunk_preserves_legacy_policy_signature(policy_server):
    """The default server path must not pass the new explicit-noise argument."""
    policy_server.policy_type = "pi05"
    observation = {OBS_STATE: torch.zeros(1, 6)}

    chunk = policy_server._get_action_chunk(observation)

    assert chunk.shape == (1, 20, 6)


def test_server_batched_cfg_shared_noise_raw_blend_and_profile(policy_server):
    """Run the complete server-side Pi0.5 CFG path without a real model or GPU."""
    from lerobot.async_inference.configs import ACPInferenceConfig

    class _Config:
        robot_type = "dummy_robot"
        image_features = {}
        chunk_size = 2
        max_action_dim = 2
        rtc_config = None

    class _BatchedPolicy:
        def __init__(self):
            self.config = _Config()
            self.calls = []

        def predict_action_chunk(self, observation, noise=None):
            self.calls.append((observation, noise.detach().clone()))
            uncond = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], device=noise.device)
            cond = uncond + 4.0
            return torch.cat((uncond, cond), dim=0)

    class _Preprocessor:
        def __init__(self):
            self.inputs = []

        def __call__(self, observation):
            self.inputs.append(observation)
            return observation

    class _Postprocessor:
        def __init__(self):
            self.inputs = []

        def __call__(self, action):
            self.inputs.append(action.detach().clone())
            return action.square()

    class _Profiler:
        def __init__(self):
            self.records = []

        def record(self, metrics, chunks=None):
            self.records.append((metrics, chunks))

    policy_server.config.acp_inference = ACPInferenceConfig(
        enable=True,
        use_cfg=True,
        batched_cfg=True,
        cfg_beta=1.5,
        profile=True,
    )
    policy_server.policy_type = "pi05"
    policy_server.pretrained_name_or_path = "dummy/pi05"
    policy_server.actions_per_chunk = 2
    policy_server.policy = _BatchedPolicy()
    policy_server.preprocessor = _Preprocessor()
    policy_server.postprocessor = _Postprocessor()
    policy_server.acp_profiler = _Profiler()

    observation = _make_obs(torch.zeros(6), timestep=7)
    observation.observation["task"] = "Pick and place"
    timed_actions = policy_server._predict_action_chunk(observation)

    assert len(policy_server.preprocessor.inputs) == 1
    preprocessor_input = policy_server.preprocessor.inputs[0]
    assert preprocessor_input[OBS_STATE].shape == (2, 6)
    assert preprocessor_input["task"] == [
        "Pick and place",
        "Pick and place\nAdvantage: positive",
    ]

    assert len(policy_server.policy.calls) == 1
    model_input, shared_noise = policy_server.policy.calls[0]
    assert model_input[OBS_STATE].shape == (2, 6)
    assert shared_noise.shape == (2, 2, 2)
    assert torch.equal(shared_noise[0], shared_noise[1])

    expected_raw_cfg = torch.tensor([[[7.0, 8.0], [9.0, 10.0]]])
    assert len(policy_server.postprocessor.inputs) == 1
    assert torch.equal(policy_server.postprocessor.inputs[0], expected_raw_cfg)
    assert torch.equal(timed_actions[0].get_action(), torch.tensor([49.0, 64.0]))
    assert torch.equal(timed_actions[1].get_action(), torch.tensor([81.0, 100.0]))

    assert len(policy_server.acp_profiler.records) == 1
    metrics, chunks = policy_server.acp_profiler.records[0]
    assert metrics["batch_size"] == 2
    assert metrics["cfg_beta"] == 1.5
    assert metrics["shared_noise_max_abs_diff"] == 0.0
    assert metrics["wall_latency_s"] >= 0.0
    assert metrics["cuda_latency_ms"] is None
    assert torch.equal(chunks["cfg_raw"], expected_raw_cfg)
    assert torch.equal(chunks["cfg_processed"], expected_raw_cfg.square())
    assert policy_server._acp_chunk_index == 1
