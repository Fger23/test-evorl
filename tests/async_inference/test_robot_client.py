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
        chunk_size_threshold=0.5,
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


def test_module_entrypoint_defaults_to_rtc_cfg():
    from lerobot.async_inference.configs import RobotClientConfig
    from tests.mocks.mock_robot import MockRobotConfig

    config = RobotClientConfig(
        robot=MockRobotConfig(),
        policy_type="pi05",
        pretrained_name_or_path="test",
        actions_per_chunk=50,
    )

    assert config.rtc_enable is True
    assert config.use_cfg is True
    assert config.chunk_size_threshold == 0.7
    assert config.fps == 30
    assert config.action_dequeue_fps == 15.0
    assert config.rtc_inference_delay == 13
    assert config.rtc_execution_horizon == 35
    assert config.rtc_cfg_beta == 1.5
    assert config.obs_queue_timeout_s == 30.0


def test_make_rtc_action_client_maps_protocol_parameters():
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
        chunk_size_threshold=0.7,
        rtc_trace_enabled=False,
    )
    robot = SimpleNamespace(action_features={"joint_0.pos": object()})
    client = _make_rtc_action_client(config, robot)

    try:
        assert client.cfg.rtc_enable is True
        assert client.cfg.use_cfg is True
        assert client.cfg.aggregate_fn_name == "latest_only"
        assert client.cfg.chunk_size_threshold == 0.7
        assert client.cfg.rtc_inference_delay == 13
        assert client.cfg.rtc_execution_horizon == 35
        assert client.cfg.rtc_cfg_beta == 1.5
        assert client.environment_dt == pytest.approx(1 / 15)
        assert isinstance(client.action_queue, ActionQueue)
    finally:
        client.stop()


def test_rtc_control_loop_confirms_only_completed_model_actions(monkeypatch):
    from lerobot.async_inference import robot_client as robot_client_module

    class FakeRobot:
        def __init__(self):
            self.sent = []

        def get_observation(self):
            return {"joint_0.pos": 0.0}

        def send_action(self, action):
            if len(self.sent) == 3:
                raise ConnectionError("robot write failed")
            self.sent.append(action)
            return action

    class FakeRTCClient:
        def __init__(self):
            self.request_timesteps = []
            self.confirmed = 0

        def get_action(self, *, observation, task, timestep):
            self.request_timesteps.append(timestep)
            return {"joint_0.pos": float(timestep)}

        def mark_action_executed(self):
            self.confirmed += 1

    config = SimpleNamespace(task="task", fps=30, action_dequeue_fps=15)
    robot = FakeRobot()
    client = FakeRTCClient()
    monkeypatch.setattr(robot_client_module, "precise_sleep", lambda _: None)

    with pytest.raises(ConnectionError, match="robot write failed"):
        robot_client_module._rtc_control_loop(config, robot, client, max_steps=3)

    assert client.request_timesteps == [0, 1]
    assert client.confirmed == 1


def test_rtc_control_loop_interpolates_15hz_actions_at_30hz(monkeypatch):
    from lerobot.async_inference import robot_client as robot_client_module

    class FakeRobot:
        def __init__(self):
            self.sent = []

        def get_observation(self):
            return {"joint_0.pos": 0.0}

        def send_action(self, action):
            self.sent.append(dict(action))
            return action

    class FakeRTCClient:
        def __init__(self):
            self.targets = iter((10.0, 20.0))
            self.timesteps = []
            self.confirmed = 0

        def get_action(self, *, observation, task, timestep):
            self.timesteps.append(timestep)
            return {"joint_0.pos": next(self.targets)}

        def mark_action_executed(self):
            self.confirmed += 1

    config = SimpleNamespace(task="task", fps=30, action_dequeue_fps=15)
    robot = FakeRobot()
    client = FakeRTCClient()
    monkeypatch.setattr(robot_client_module, "precise_sleep", lambda _: None)

    assert robot_client_module._rtc_control_loop(config, robot, client, max_steps=2) == 2

    assert [action["joint_0.pos"] for action in robot.sent] == [5.0, 10.0, 15.0, 20.0]
    assert client.timesteps == [0, 1]
    assert client.confirmed == 2


def test_rtc_control_loop_never_replays_missed_periods_as_a_burst(monkeypatch):
    from lerobot.async_inference import robot_client as robot_client_module

    clock = [0.0]
    send_times = []

    def perf_counter():
        return clock[0]

    def precise_sleep(seconds):
        clock[0] += seconds

    class FakeRobot:
        def get_observation(self):
            return {"joint_0.pos": 0.0}

        def send_action(self, action):
            send_times.append(clock[0])
            # A real bus write is not free. Pacing is start-to-start so this
            # time does not unnecessarily lower the requested command rate.
            clock[0] += 0.005
            return action

    class FakeRTCClient:
        def __init__(self):
            self.calls = 0

        def get_action(self, *, observation, task, timestep):
            self.calls += 1
            if self.calls == 2:
                # Simulate a recurrent queue wait after commands have already
                # been sent. Recovery may be immediate, but never catch-up.
                clock[0] += 1.0
            return {"joint_0.pos": float(self.calls)}

        def mark_action_executed(self):
            return None

    monkeypatch.setattr(robot_client_module.time, "perf_counter", perf_counter)
    monkeypatch.setattr(robot_client_module, "precise_sleep", precise_sleep)
    config = SimpleNamespace(task="task", fps=30, action_dequeue_fps=15)

    robot_client_module._rtc_control_loop(config, FakeRobot(), FakeRTCClient(), max_steps=2)

    assert len(send_times) == 4
    intervals = [
        later - earlier for earlier, later in zip(send_times[:-1], send_times[1:], strict=True)
    ]
    assert all(interval >= 1 / 30 - 1e-12 for interval in intervals)


def test_legacy_client_does_not_require_an_interpolation_ratio():
    from lerobot.async_inference.configs import RobotClientConfig
    from tests.mocks.mock_robot import MockRobotConfig

    config = RobotClientConfig(
        robot=MockRobotConfig(),
        policy_type="test",
        pretrained_name_or_path="test",
        actions_per_chunk=20,
        fps=20,
        rtc_enable=False,
    )

    assert config.fps == 20
    assert config.action_dequeue_fps == 15.0


@pytest.mark.parametrize("action_dequeue_fps", [0, -1, 20, 31])
def test_rtc_interpolation_rate_validation(action_dequeue_fps):
    from lerobot.async_inference.configs import RobotClientConfig
    from tests.mocks.mock_robot import MockRobotConfig

    with pytest.raises(ValueError, match="action_dequeue_fps|integer multiple"):
        RobotClientConfig(
            robot=MockRobotConfig(),
            policy_type="pi05",
            pretrained_name_or_path="test",
            actions_per_chunk=50,
            fps=30,
            action_dequeue_fps=action_dequeue_fps,
        )


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


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
