# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import threading
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from lerobot.scripts.recording_remote_policy import (
    AGGREGATE_FUNCTIONS,
    RemotePolicyActionClient,
    RemotePolicyRecordConfig,
)


@dataclass
class FakeTimedAction:
    timestamp: float
    timestep: int
    action: torch.Tensor

    def get_timestamp(self) -> float:
        return self.timestamp

    def get_timestep(self) -> int:
        return self.timestep

    def get_action(self) -> torch.Tensor:
        return self.action


class FakeStub:
    def __init__(self):
        self.ready_timeouts = []

    def Ready(self, request, timeout=None):  # noqa: N802
        self.ready_timeouts.append(timeout)
        return request


def make_client() -> RemotePolicyActionClient:
    client = object.__new__(RemotePolicyActionClient)
    client.cfg = RemotePolicyRecordConfig(
        enable=True,
        policy_type="pi05",
        pretrained_name_or_path="unused",
        actions_per_chunk=2,
        chunk_size_threshold=1.0,
        obs_queue_timeout_s=0.2,
    )
    client.robot = SimpleNamespace(action_features=["joint.pos"])
    client.environment_dt = 0.01
    client.services_pb2 = SimpleNamespace(Empty=lambda: object())
    client.stub = FakeStub()
    client.action_queue = []
    client.latest_action_timestep = -1
    client.latest_action = None
    client.aggregate_fn = AGGREGATE_FUNCTIONS[client.cfg.aggregate_fn_name]
    client._state_lock = threading.Lock()
    client._request_thread = None
    client._generation = 0
    return client


def make_action(timestep: int) -> FakeTimedAction:
    return FakeTimedAction(timestamp=float(timestep), timestep=timestep, action=torch.tensor([timestep]))


def test_reset_waits_for_inflight_request_and_rejects_stale_actions():
    client = make_client()
    request_started = threading.Event()
    release_request = threading.Event()
    reset_finished = threading.Event()

    def request_actions(observation, task, timestep, generation):
        request_started.set()
        assert release_request.wait(timeout=1)
        client._merge_actions([make_action(timestep)], generation=generation)

    client._request_actions = request_actions
    client._start_background_request({}, None, 4)
    assert request_started.wait(timeout=1)

    reset_thread = threading.Thread(target=lambda: (client.reset(), reset_finished.set()))
    reset_thread.start()
    assert not reset_finished.wait(timeout=0.05)

    release_request.set()
    reset_thread.join(timeout=1)

    assert reset_finished.is_set()
    assert client._generation == 1
    assert client.action_queue == []
    assert client.latest_action is None
    assert client.latest_action_timestep == -1
    assert client.stub.ready_timeouts == [client._rpc_timeout_s]

    client._merge_actions([make_action(5)], generation=0)
    assert client.action_queue == []


def test_background_failure_releases_request_slot_and_allows_retry(caplog):
    client = make_client()
    attempts = 0

    def failing_request(observation, task, timestep, generation):
        nonlocal attempts
        attempts += 1
        raise TimeoutError("inference timed out")

    client._request_actions = failing_request

    client.action_queue = [make_action(0)]
    assert client.get_action({}, None, 0) == {"joint.pos": 0}
    client._wait_for_background_request()
    assert client._request_thread is None

    client.action_queue = [make_action(1)]
    assert client.get_action({}, None, 1) == {"joint.pos": 1}
    client._wait_for_background_request()

    assert attempts == 2
    assert "Remote policy background request failed." in caplog.text


def test_receive_actions_uses_a_bounded_grpc_deadline():
    client = make_client()
    client.cfg.obs_queue_timeout_s = 0
    get_actions_timeouts = []

    class EmptyActionsStub(FakeStub):
        def GetActions(self, request, timeout=None):  # noqa: N802
            get_actions_timeouts.append(timeout)
            return SimpleNamespace(data=b"")

    client.stub = EmptyActionsStub()

    with pytest.raises(TimeoutError, match="Timed out waiting"):
        client._receive_actions(generation=client._generation)

    assert get_actions_timeouts == [client.environment_dt]


def test_empty_action_queue_waits_and_retries_inference(monkeypatch, caplog):
    client = make_client()
    attempts = 0

    def flaky_request(observation, task, timestep, generation):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("temporary timeout")
        client._merge_actions([make_action(timestep)], generation=generation)

    client._request_actions = flaky_request
    monkeypatch.setattr("lerobot.scripts.recording_remote_policy.time.sleep", lambda _: None)

    action = client.get_action({}, None, 7)

    assert action == {"joint.pos": 7}
    assert attempts == 2
    assert "action queue is empty and inference failed; retrying" in caplog.text
