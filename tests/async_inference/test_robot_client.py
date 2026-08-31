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
"""Unit-tests for the `RobotClient` action-queue logic (pure Python, no gRPC).

We monkey-patch `lerobot.robots.utils.make_robot_from_config` so that
no real hardware is accessed. Only the queue-update mechanism is verified.
"""

from __future__ import annotations

import time
from queue import Queue
from types import SimpleNamespace

import pytest
import torch

# Skip entire module if grpc is not available
pytest.importorskip("grpc")

# -----------------------------------------------------------------------------
# Test fixtures
# -----------------------------------------------------------------------------


@pytest.fixture()
def robot_client():
    """Fresh `RobotClient` instance for each test case (no threads started).
    Uses DummyRobot."""
    # Import only when the test actually runs (after decorator check)
    from lerobot.async_inference.configs import RobotClientConfig
    from lerobot.async_inference.robot_client import RobotClient
    from tests.mocks.mock_robot import MockRobotConfig

    test_config = MockRobotConfig()

    # gRPC channel is not actually used in tests, so using a dummy address
    test_config = RobotClientConfig(
        robot=test_config,
        server_address="localhost:9999",
        policy_type="test",
        pretrained_name_or_path="test",
        actions_per_chunk=20,
        rtc_enable=False,
    )

    client = RobotClient(test_config)

    # Initialize attributes that are normally set in start() method
    client.chunks_received = 0
    client.available_actions_size = []

    yield client

    if client.robot.is_connected:
        client.stop()


# -----------------------------------------------------------------------------
# Helper utilities for tests
# -----------------------------------------------------------------------------


def _make_actions(start_ts: float, start_t: int, count: int):
    """Generate `count` consecutive TimedAction objects starting at timestep `start_t`."""
    from lerobot.async_inference.helpers import TimedAction

    fps = 30  # emulates most common frame-rate
    actions = []
    for i in range(count):
        timestep = start_t + i
        timestamp = start_ts + i * (1 / fps)
        action_tensor = torch.full((6,), timestep, dtype=torch.float32)
        actions.append(TimedAction(action=action_tensor, timestep=timestep, timestamp=timestamp))
    return actions


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_robot_client_defaults_preserve_main_compatibility():
    from lerobot.async_inference.configs import RobotClientConfig
    from tests.mocks.mock_robot import MockRobotConfig

    config = RobotClientConfig(
        robot=MockRobotConfig(),
        policy_type="pi05",
        pretrained_name_or_path="test",
        actions_per_chunk=50,
    )

    assert config.rtc_enable is False
    assert config.use_cfg is False


def test_robot_client_rtc_cfg_off_maps_to_recording_client_at_30hz():
    from lerobot.async_inference.configs import RobotClientConfig
    from lerobot.async_inference.robot_client import _make_rtc_action_client
    from lerobot.policies.rtc.action_queue import ActionQueue
    from tests.mocks.mock_robot import MockRobotConfig

    config = RobotClientConfig(
        robot=MockRobotConfig(),
        server_address="localhost:9999",
        policy_type="pi05",
        pretrained_name_or_path="test",
        actions_per_chunk=50,
        fps=30,
        rtc_enable=True,
        use_cfg=False,
        rtc_inference_delay=21,
        rtc_execution_horizon=35,
        rtc_cfg_beta=1.5,
    )
    robot = SimpleNamespace(action_features={"joint_0.pos": object()})
    client = _make_rtc_action_client(config, robot)

    try:
        assert client.cfg.rtc_enable is True
        assert client.cfg.use_cfg is False
        assert client.cfg.aggregate_fn_name == "latest_only"
        assert client.environment_dt == pytest.approx(1 / 30)
        assert isinstance(client.action_queue, ActionQueue)
    finally:
        client.stop()


def test_robot_client_cfg_can_run_without_rtc():
    from lerobot.async_inference.configs import RobotClientConfig
    from lerobot.async_inference.robot_client import _make_rtc_action_client
    from tests.mocks.mock_robot import MockRobotConfig

    config = RobotClientConfig(
        robot=MockRobotConfig(),
        server_address="localhost:9999",
        policy_type="pi05",
        pretrained_name_or_path="test",
        actions_per_chunk=50,
        rtc_enable=False,
        use_cfg=True,
        aggregate_fn_name="weighted_average",
        rtc_cfg_beta=1.5,
    )
    robot = SimpleNamespace(action_features={"joint_0.pos": object()})
    client = _make_rtc_action_client(config, robot)

    try:
        assert client.cfg.rtc_enable is False
        assert client.cfg.use_cfg is True
        assert client.cfg.aggregate_fn_name == "weighted_average"
        assert isinstance(client.action_queue, list)
    finally:
        client.stop()


def test_rtc_control_loop_refills_after_send_and_retries_same_failed_action(monkeypatch):
    """Refill observes post-send state; a failed write is retried and confirmed once."""
    from lerobot.async_inference import robot_client as robot_client_module

    events = []

    class FakeRobot:
        def __init__(self):
            self.attempts = 0
            self.sent = []
            self.observation_calls = 0
            self.send_times = []

        def get_observation(self):
            self.observation_calls += 1
            events.append(("observation", len(self.sent)))
            return {"state": self.attempts}

        def send_action(self, action):
            self.attempts += 1
            self.send_times.append(clock[0])
            if self.attempts == 2:
                raise ConnectionError("robot write failed")
            self.sent.append(action)
            events.append(("send", action["joint_0.pos"]))

    class FakeRTCClient:
        def __init__(self):
            self.request_timesteps = []
            self.confirmed = 0
            self.observations = []
            self.submit_timesteps = []
            self.submit_confirmation_counts = []

        def get_action(self, *, observation, task, timestep, defer_refill_until_after_execution):
            self.request_timesteps.append(timestep)
            assert callable(observation)
            assert defer_refill_until_after_execution is True
            if timestep == 0:
                self.observations.append(observation())
            return {"joint_0.pos": float(timestep)}

        def mark_action_executed(self):
            self.confirmed += 1

        def submit_observation_if_needed(self, observation, task, timestep):
            assert callable(observation)
            assert task == "task"
            self.submit_timesteps.append(timestep)
            self.submit_confirmation_counts.append(self.confirmed)
            if timestep == 1:
                self.observations.append(observation())

    arm_config = SimpleNamespace(max_relative_target=None)
    config = SimpleNamespace(
        task="task",
        environment_dt=1 / 30,
        robot=SimpleNamespace(
            type="bi_so_follower",
            left_arm_config=arm_config,
            right_arm_config=arm_config,
        ),
    )
    robot = FakeRobot()
    client = FakeRTCClient()
    sleeps = []
    clock = [0.0]

    def fake_sleep(delay_s):
        sleeps.append(delay_s)
        clock[0] += delay_s

    monkeypatch.setattr(robot_client_module.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(robot_client_module, "precise_sleep", fake_sleep)

    assert robot_client_module._rtc_control_loop(config, robot, client, max_steps=3) == 3

    assert robot.attempts == 4
    assert robot.sent == [
        {"joint_0.pos": 0.0},
        {"joint_0.pos": 1.0},
        {"joint_0.pos": 2.0},
    ]
    assert client.request_timesteps == [0, 1, 2]
    assert client.confirmed == 3
    assert client.submit_timesteps == [1, 2, 3]
    assert client.submit_confirmation_counts == [1, 2, 3]
    assert robot.observation_calls == 2
    assert client.observations == [{"state": 0}, {"state": 1}]
    assert events[:3] == [
        ("observation", 0),
        ("send", 0.0),
        ("observation", 1),
    ]
    assert all(
        later - earlier >= config.environment_dt - 1e-9
        for earlier, later in zip(robot.send_times, robot.send_times[1:], strict=False)
    )
    assert any(delay >= 0.1 for delay in sleeps)


def test_rtc_control_loop_does_not_retry_unsafe_bimanual_relative_action():
    from lerobot.async_inference import robot_client as robot_client_module

    class FakeRobot:
        def __init__(self):
            self.attempts = 0

        def get_observation(self):
            return {"state": 0}

        def send_action(self, action):
            self.attempts += 1
            raise ConnectionError("right arm write failed after left arm may have moved")

    class FakeRTCClient:
        def __init__(self):
            self.confirmed = 0
            self.submissions = 0

        def get_action(self, *, observation, task, timestep, defer_refill_until_after_execution):
            assert defer_refill_until_after_execution is True
            return {"joint_0.pos": 1.0}

        def mark_action_executed(self):
            self.confirmed += 1

        def submit_observation_if_needed(self, observation, task, timestep):
            self.submissions += 1

    config = SimpleNamespace(
        task="task",
        environment_dt=1 / 30,
        robot=SimpleNamespace(
            type="bi_so_follower",
            left_arm_config=SimpleNamespace(max_relative_target=5.0),
            right_arm_config=SimpleNamespace(max_relative_target=None),
        ),
    )
    robot = FakeRobot()
    client = FakeRTCClient()

    with pytest.raises(ConnectionError, match="right arm write failed"):
        robot_client_module._rtc_control_loop(config, robot, client, max_steps=1)

    assert robot.attempts == 1
    assert client.confirmed == 0
    assert client.submissions == 0


def test_update_action_queue_discards_stale(robot_client):
    """`_update_action_queue` must drop actions with `timestep` <= `latest_action`."""

    # Pretend we already executed up to action #4
    robot_client.latest_action = 4

    # Incoming chunk contains timesteps 3..7 -> expect 5,6,7 kept.
    incoming = _make_actions(start_ts=time.time(), start_t=3, count=5)  # 3,4,5,6,7

    robot_client._aggregate_action_queues(incoming)

    # Extract timesteps from queue
    resulting_timesteps = [a.get_timestep() for a in robot_client.action_queue.queue]

    assert resulting_timesteps == [5, 6, 7]


@pytest.mark.parametrize(
    "weight_old, weight_new",
    [
        (1.0, 0.0),
        (0.0, 1.0),
        (0.5, 0.5),
        (0.2, 0.8),
        (0.8, 0.2),
        (0.1, 0.9),
        (0.9, 0.1),
    ],
)
def test_aggregate_action_queues_combines_actions_in_overlap(
    robot_client, weight_old: float, weight_new: float
):
    """`_aggregate_action_queues` must combine actions on overlapping timesteps according
    to the provided aggregate_fn, here tested with multiple coefficients."""
    from lerobot.async_inference.helpers import TimedAction

    robot_client.chunks_received = 0

    # Pretend we already executed up to action #4, and queue contains actions for timesteps 5..6
    robot_client.latest_action = 4
    current_actions = _make_actions(
        start_ts=time.time(), start_t=5, count=2
    )  # actions are [torch.ones(6), torch.ones(6), ...]
    current_actions = [
        TimedAction(action=10 * a.get_action(), timestep=a.get_timestep(), timestamp=a.get_timestamp())
        for a in current_actions
    ]

    for a in current_actions:
        robot_client.action_queue.put(a)

    # Incoming chunk contains timesteps 3..7 -> expect 5,6,7 kept.
    incoming = _make_actions(start_ts=time.time(), start_t=3, count=5)  # 3,4,5,6,7

    overlap_timesteps = [5, 6]  # properly tested in test_aggregate_action_queues_discards_stale
    nonoverlap_timesteps = [7]

    robot_client._aggregate_action_queues(
        incoming, aggregate_fn=lambda x1, x2: weight_old * x1 + weight_new * x2
    )

    queue_overlap_actions = []
    queue_non_overlap_actions = []
    for a in robot_client.action_queue.queue:
        if a.get_timestep() in overlap_timesteps:
            queue_overlap_actions.append(a)
        elif a.get_timestep() in nonoverlap_timesteps:
            queue_non_overlap_actions.append(a)

    queue_overlap_actions = sorted(queue_overlap_actions, key=lambda x: x.get_timestep())
    queue_non_overlap_actions = sorted(queue_non_overlap_actions, key=lambda x: x.get_timestep())

    assert torch.allclose(
        queue_overlap_actions[0].get_action(),
        weight_old * current_actions[0].get_action() + weight_new * incoming[-3].get_action(),
    )
    assert torch.allclose(
        queue_overlap_actions[1].get_action(),
        weight_old * current_actions[1].get_action() + weight_new * incoming[-2].get_action(),
    )
    assert torch.allclose(queue_non_overlap_actions[0].get_action(), incoming[-1].get_action())


@pytest.mark.parametrize(
    "chunk_size, queue_len, expected",
    [
        (20, 12, False),  # 12 / 20 = 0.6  > g=0.5 threshold, not ready to send
        (20, 8, True),  # 8  / 20 = 0.4 <= g=0.5, ready to send
        (10, 5, True),
        (10, 6, False),
    ],
)
def test_ready_to_send_observation(robot_client, chunk_size: int, queue_len: int, expected: bool):
    """Validate `_ready_to_send_observation` ratio logic for various sizes."""

    robot_client.action_chunk_size = chunk_size

    # Clear any existing actions then fill with `queue_len` dummy entries ----
    robot_client.action_queue = Queue()

    dummy_actions = _make_actions(start_ts=time.time(), start_t=0, count=queue_len)
    for act in dummy_actions:
        robot_client.action_queue.put(act)

    assert robot_client._ready_to_send_observation() is expected


@pytest.mark.parametrize(
    "g_threshold, expected",
    [
        # The condition is `queue_size / chunk_size <= g`.
        # Here, ratio = 6 / 10 = 0.6.
        (0.0, False),  # 0.6 <= 0.0 is False
        (0.1, False),
        (0.2, False),
        (0.3, False),
        (0.4, False),
        (0.5, False),
        (0.6, True),  # 0.6 <= 0.6 is True
        (0.7, True),
        (0.8, True),
        (0.9, True),
        (1.0, True),
    ],
)
def test_ready_to_send_observation_with_varying_threshold(robot_client, g_threshold: float, expected: bool):
    """Validate `_ready_to_send_observation` with fixed sizes and varying `g`."""
    # Fixed sizes for this test: ratio = 6 / 10 = 0.6
    chunk_size = 10
    queue_len = 6

    robot_client.action_chunk_size = chunk_size
    # This is the parameter we are testing
    robot_client._chunk_size_threshold = g_threshold

    # Fill queue with dummy actions
    robot_client.action_queue = Queue()
    dummy_actions = _make_actions(start_ts=time.time(), start_t=0, count=queue_len)
    for act in dummy_actions:
        robot_client.action_queue.put(act)

    assert robot_client._ready_to_send_observation() is expected


# -----------------------------------------------------------------------------
# Client-side metrics persistence
# -----------------------------------------------------------------------------


def test_save_client_metrics_writes_records_and_summary(tmp_path):
    """`save_client_metrics` must persist per-chunk records and aggregate stats."""
    import json
    import os

    from lerobot.async_inference.robot_client import save_client_metrics

    records = [
        {
            "index": 0,
            "e2e_latency_ms": 100.0,
            "steps_consumed_during_inference": 0,
            "latest_action_timestep": -1,
            "chunk_size": 20,
            "queue_size_on_arrival": 0,
            "received_at_utc": "2026-01-01T00:00:00.000Z",
        },
        {
            "index": 1,
            "e2e_latency_ms": 300.0,
            "steps_consumed_during_inference": 9,
            "latest_action_timestep": 8,
            "chunk_size": 20,
            "queue_size_on_arrival": 11,
            "received_at_utc": "2026-01-01T00:00:01.000Z",
        },
    ]
    queue_sizes = [20, 19, 18, 17]

    output_dir = save_client_metrics(
        output_root=str(tmp_path),
        run_name="unit_run",
        records=records,
        action_queue_size=queue_sizes,
        fps=30,
    )

    assert output_dir == str(tmp_path / "unit_run")
    assert os.path.isfile(os.path.join(output_dir, "records.jsonl"))
    assert os.path.isfile(os.path.join(output_dir, "summary.json"))

    with open(os.path.join(output_dir, "records.jsonl"), encoding="utf-8") as stream:
        persisted = [json.loads(line) for line in stream if line.strip()]
    assert len(persisted) == 2
    assert persisted[0]["e2e_latency_ms"] == 100.0
    assert persisted[1]["steps_consumed_during_inference"] == 9

    with open(os.path.join(output_dir, "summary.json"), encoding="utf-8") as stream:
        summary = json.load(stream)
    assert summary["num_chunks"] == 2
    assert summary["e2e_latency_ms"]["p50"] == 200.0
    assert summary["e2e_latency_ms"]["max"] == 300.0
    assert summary["steps_consumed_during_inference"]["total"] == 9
    assert summary["action_queue_size"]["max"] == 20
    assert summary["action_queue_size_series"] == queue_sizes
    assert summary["fps"] == 30


def test_save_client_metrics_rejects_path_like_run_name(tmp_path):
    from lerobot.async_inference.robot_client import save_client_metrics

    with pytest.raises(ValueError):
        save_client_metrics(
            output_root=str(tmp_path), run_name="../escape", records=[], action_queue_size=[], fps=30
        )


def test_robot_client_save_metrics(robot_client, tmp_path, monkeypatch):
    """`_save_metrics` must dump collected records/queue sizes via config paths."""
    import json
    import os

    monkeypatch.setattr(robot_client.config, "metrics_output_dir", str(tmp_path))
    monkeypatch.setattr(robot_client.config, "metrics_run_name", "fixture_run")

    robot_client.action_chunk_size = 20
    robot_client.action_queue_size = [5, 4, 3]
    with robot_client._metrics_lock:
        robot_client.client_metrics_records.append(
            {
                "index": 0,
                "e2e_latency_ms": 123.0,
                "steps_consumed_during_inference": 3,
                "latest_action_timestep": 2,
                "chunk_size": 20,
                "queue_size_on_arrival": 5,
                "received_at_utc": "2026-01-01T00:00:00.000Z",
            }
        )

    output_dir = robot_client._save_metrics()
    assert output_dir == str(tmp_path / "fixture_run")

    with open(os.path.join(output_dir, "summary.json"), encoding="utf-8") as stream:
        summary = json.load(stream)
    assert summary["num_chunks"] == 1
    assert summary["e2e_latency_ms"]["max"] == 123.0
    assert summary["action_queue_size_series"] == [5, 4, 3]
    assert summary["server_address"] == "localhost:9999"


def test_robot_client_save_metrics_skips_empty_state(robot_client, tmp_path, monkeypatch):
    """Nothing recorded -> no files written, returns None."""
    monkeypatch.setattr(robot_client.config, "metrics_output_dir", str(tmp_path))
    assert robot_client._save_metrics() is None
    assert not (tmp_path / "records.jsonl").exists()
