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

"""Deterministic tests for the recording client's RTC queue coordination."""

from __future__ import annotations

import pickle  # nosec
import threading
import time
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from lerobot.async_inference.helpers import RemoteActionChunk, TimedAction
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.scripts.recording_remote_policy import RemotePolicyActionClient, RemotePolicyRecordConfig


@pytest.fixture
def rtc_client() -> RemotePolicyActionClient:
    """Build the state machine without opening a gRPC channel or worker thread."""
    cfg = RemotePolicyRecordConfig(
        enable=True,
        server_address="unused:9999",
        policy_type="pi05",
        pretrained_name_or_path="dummy/pi05",
        actions_per_chunk=6,
        chunk_size_threshold=1.0,
        obs_queue_timeout_s=0.1,
        use_cfg=False,
        rtc_enable=True,
        rtc_inference_delay=1,
        rtc_execution_horizon=3,
    )

    client = RemotePolicyActionClient.__new__(RemotePolicyActionClient)
    client.cfg = cfg
    client.robot = SimpleNamespace(action_features={"joint_0": object(), "joint_1": object()})
    client.environment_dt = 1 / 30
    client.action_queue = ActionQueue(
        RTCConfig(enabled=True, execution_horizon=cfg.rtc_execution_horizon)
    )
    client.latest_action_timestep = -1
    client.latest_action = None
    client.aggregate_fn = lambda _old, new: new
    client._state_lock = threading.RLock()
    client._request_queue = Queue(maxsize=1)
    client._pending_event = threading.Event()
    client._stop_event = threading.Event()
    client._worker_thread = None
    client._pending_result = None
    client._worker_error = None
    client._in_flight_request_id = None
    client._active_rpc_future = None
    client._active_rpc_request_id = None
    client._episode_epoch = 0
    client._request_sequence = 0
    client._executed_steps = 0
    client._awaiting_execution_confirmation = False
    client._started = True
    return client


def _seed_queue(client: RemotePolicyActionClient) -> tuple[torch.Tensor, torch.Tensor]:
    raw = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    processed = raw + 100
    assert isinstance(client.action_queue, ActionQueue)
    client.action_queue.merge(raw, processed, real_delay=0)
    return raw, processed


def _make_response(
    request_id: str,
    raw: torch.Tensor,
    processed: torch.Tensor,
) -> RemoteActionChunk:
    actions = [
        TimedAction(timestamp=float(i), timestep=i, action=action.clone())
        for i, action in enumerate(processed)
    ]
    return RemoteActionChunk(
        request_id=request_id,
        actions=actions,
        raw_actions=raw.clone(),
        observation_timestep=0,
        use_cfg=False,
        cfg_beta=None,
        rtc_enabled=True,
        inference_delay=1,
        execution_horizon=3,
        max_guidance_weight=10.0,
        prefix_attention_schedule=RTCAttentionSchedule.LINEAR,
    )


def _take_submitted_request(client: RemotePolicyActionClient):
    request = client._request_queue.get_nowait()
    client._request_queue.task_done()
    assert request is not None
    return request


def test_recording_client_cfg_off_negotiates_v2_con_only_rtc(monkeypatch):
    """The lerobot_record remote client must explicitly request CFG-off RTC."""
    from lerobot.async_inference import helpers
    from lerobot.scripts import recording_remote_policy as remote_policy_module

    cfg = RemotePolicyRecordConfig(
        enable=True,
        server_address="unused:9999",
        policy_type="pi05",
        pretrained_name_or_path="dummy/pi05",
        actions_per_chunk=50,
        use_cfg=False,
        rtc_enable=True,
        rtc_inference_delay=21,
        rtc_execution_horizon=35,
        rtc_cfg_beta=1.5,
    )
    client = RemotePolicyActionClient.__new__(RemotePolicyActionClient)
    client.cfg = cfg
    client.robot = SimpleNamespace()
    client.environment_dt = 1 / 30
    client.stub = MagicMock()
    client.services_pb2 = SimpleNamespace(
        Empty=lambda: SimpleNamespace(),
        PolicySetup=lambda *, data: SimpleNamespace(data=data),
    )
    client._started = False
    fake_thread = MagicMock()

    monkeypatch.setattr(helpers, "map_robot_keys_to_lerobot_features", lambda _robot: {})
    monkeypatch.setattr(
        remote_policy_module,
        "threading",
        SimpleNamespace(Thread=lambda **_kwargs: fake_thread),
    )

    client.start()

    setup = client.stub.SendPolicyInstructions.call_args.args[0]
    policy_config = pickle.loads(setup.data)  # nosec
    assert policy_config.protocol_version == 2
    assert policy_config.return_raw_actions is True
    assert policy_config.use_cfg is False
    assert policy_config.rtc_enabled is True
    assert policy_config.rtc_inference_delay == 21
    assert policy_config.rtc_execution_horizon == 35
    fake_thread.start.assert_called_once_with()


def test_submit_uses_one_in_flight_request_and_raw_snapshot(rtc_client):
    raw, processed = _seed_queue(rtc_client)
    barrier = threading.Barrier(9)
    results: list[bool] = []
    errors: list[BaseException] = []
    results_lock = threading.Lock()

    def submit() -> None:
        barrier.wait()
        try:
            result = rtc_client._submit_if_needed(
                observation={"state": [0.0, 1.0]}, task="test", timestep=5
            )
            with results_lock:
                results.append(result)
        except BaseException as exc:
            with results_lock:
                errors.append(exc)

    threads = [threading.Thread(target=submit) for _ in range(8)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=2)

    assert not errors
    assert sum(results) == 1
    assert rtc_client._request_queue.qsize() == 1
    request = _take_submitted_request(rtc_client)
    assert request.request_id == rtc_client._in_flight_request_id
    assert torch.equal(request.left_over, raw)
    assert not torch.equal(request.left_over, processed)

    # Removing the queued item does not permit another request while its RPC is active.
    assert not rtc_client._submit_if_needed(observation={}, task=None, timestep=6)


def test_first_request_has_no_leftover(rtc_client):
    assert rtc_client._submit_if_needed(observation={}, task=None, timestep=0)
    request = _take_submitted_request(rtc_client)

    assert request.left_over is None
    assert request.queue_size_at_submit == 0
    assert request.executed_steps_at_submit == 0


def test_lazy_observation_is_captured_only_for_a_reserved_request(rtc_client):
    """Motor/camera I/O must follow chunk requests, not the 30 Hz action loop."""

    _seed_queue(rtc_client)
    rtc_client.cfg.chunk_size_threshold = 0.5
    observation_provider = MagicMock(return_value={"state": [1.0, 2.0]})

    # A full queue needs no request, so the callable must remain untouched.
    assert not rtc_client._submit_if_needed(
        observation=observation_provider,
        task="test",
        timestep=0,
    )
    observation_provider.assert_not_called()

    # At the low-water mark exactly one caller reserves and captures a request.
    for _ in range(3):
        assert rtc_client._pop_action() is not None
        rtc_client.mark_action_executed()
    assert rtc_client._submit_if_needed(
        observation=observation_provider,
        task="test",
        timestep=3,
    )
    observation_provider.assert_called_once_with()
    request = _take_submitted_request(rtc_client)
    assert request.observation == {"state": [1.0, 2.0]}

    # The in-flight reservation prevents duplicate hardware capture.
    assert not rtc_client._submit_if_needed(
        observation=observation_provider,
        task="test",
        timestep=4,
    )
    observation_provider.assert_called_once_with()


def test_lazy_observation_failure_keeps_queue_and_backs_off(rtc_client):
    """One bad camera/serial read must not tear down otherwise safe queued motion."""

    _, processed = _seed_queue(rtc_client)
    observation_provider = MagicMock(side_effect=ConnectionError("bad status packet"))

    assert not rtc_client._submit_if_needed(
        observation=observation_provider,
        task="test",
        timestep=0,
    )
    assert rtc_client._in_flight_request_id is None
    assert rtc_client._observation_failure_count == 1
    assert torch.equal(rtc_client._pop_action(), processed[0])
    rtc_client.mark_action_executed()

    # Immediate retry is suppressed so a failing USB device is not hammered.
    assert not rtc_client._submit_if_needed(
        observation=observation_provider,
        task="test",
        timestep=1,
    )
    observation_provider.assert_called_once_with()

    rtc_client._observation_retry_not_before = 0.0
    observation_provider.side_effect = None
    observation_provider.return_value = {"state": [3.0, 4.0]}
    assert rtc_client._submit_if_needed(
        observation=observation_provider,
        task="test",
        timestep=1,
    )
    assert rtc_client._observation_failure_count == 0


def test_reset_during_lazy_observation_drops_reserved_request(rtc_client):
    """A slow camera read cannot resurrect a request from an older episode."""

    capture_started = threading.Event()
    release_capture = threading.Event()
    results = []

    def observation_provider():
        capture_started.set()
        assert release_capture.wait(timeout=2)
        return {"state": [1.0, 2.0]}

    submit_thread = threading.Thread(
        target=lambda: results.append(
            rtc_client._submit_if_needed(
                observation=observation_provider,
                task="test",
                timestep=0,
            )
        )
    )
    submit_thread.start()
    assert capture_started.wait(timeout=2)

    rtc_client._started = False  # Keep this state-machine test off the network.
    rtc_client.reset()
    release_capture.set()
    submit_thread.join(timeout=2)

    assert not submit_thread.is_alive()
    assert results == [False]
    assert rtc_client._in_flight_request_id is None
    assert rtc_client._request_queue.empty()


def test_pending_chunk_is_cropped_by_confirmed_executed_steps(rtc_client):
    old_raw, old_processed = _seed_queue(rtc_client)
    assert rtc_client._submit_if_needed(observation={}, task="test", timestep=0)
    request = _take_submitted_request(rtc_client)
    assert torch.equal(request.left_over, old_raw)

    # The main 30 Hz thread executes three old processed actions while inference runs.
    for index in range(3):
        assert torch.equal(rtc_client._pop_action(), old_processed[index])
        rtc_client.mark_action_executed()

    new_raw = torch.arange(20, 32, dtype=torch.float32).reshape(6, 2)
    new_processed = new_raw + 1_000
    response = _make_response(request.request_id, new_raw, new_processed)
    with rtc_client._state_lock:
        rtc_client._in_flight_request_id = None
        rtc_client._pending_result = (request, response)

    assert rtc_client._apply_pending_result()
    assert rtc_client._executed_steps == 3

    assert isinstance(rtc_client.action_queue, ActionQueue)
    snapshot = rtc_client.action_queue.snapshot()
    assert snapshot.queue_size == 3
    assert torch.equal(snapshot.left_over, new_raw[3:])
    assert torch.equal(rtc_client._pop_action(), new_processed[3])


def test_stale_epoch_response_does_not_replace_current_queue(rtc_client):
    old_raw, old_processed = _seed_queue(rtc_client)
    assert rtc_client._submit_if_needed(observation={}, task=None, timestep=0)
    request = _take_submitted_request(rtc_client)
    stale_response = _make_response(request.request_id, old_raw + 500, old_processed + 500)

    # Simulate an episode reset racing with a response that was already decoded.
    with rtc_client._state_lock:
        rtc_client._episode_epoch += 1
        rtc_client._in_flight_request_id = None
        rtc_client._pending_result = (request, stale_response)

    assert not rtc_client._apply_pending_result()
    assert isinstance(rtc_client.action_queue, ActionQueue)
    assert torch.equal(rtc_client.action_queue.snapshot().left_over, old_raw)
    assert torch.equal(rtc_client.action_queue.get(), old_processed[0])


def test_reset_cancels_active_rpc_and_clears_queue(rtc_client):
    class _BlockedFuture:
        def __init__(self):
            self.cancelled = False
            self.released = threading.Event()

        def cancel(self):
            self.cancelled = True
            self.released.set()
            return True

        def result(self):
            self.released.wait(timeout=2)
            if self.cancelled:
                raise RuntimeError("cancelled")
            return object()

    class _RPC:
        def __init__(self, future):
            self.blocked_future = future

        def future(self, _argument, timeout=None):
            assert timeout == 5.0
            return self.blocked_future

    _seed_queue(rtc_client)
    assert rtc_client._submit_if_needed(observation={}, task=None, timestep=0)
    request = _take_submitted_request(rtc_client)
    blocked_future = _BlockedFuture()
    errors = []

    def run_rpc():
        try:
            rtc_client._run_cancellable_rpc(
                request,
                _RPC(blocked_future),
                object(),
                timeout=5.0,
            )
        except RuntimeError as exc:
            errors.append(exc)

    rpc_thread = threading.Thread(target=run_rpc)
    rpc_thread.start()
    deadline = time.perf_counter() + 1
    while rtc_client._active_rpc_future is None and time.perf_counter() < deadline:
        time.sleep(0.001)
    assert rtc_client._active_rpc_future is blocked_future
    rtc_client._started = False  # Avoid an actual Ready RPC in this state-machine test.

    rtc_client.reset()
    rpc_thread.join(timeout=2)

    assert blocked_future.cancelled
    assert errors and str(errors[0]) == "cancelled"
    assert not rpc_thread.is_alive()
    assert rtc_client._active_rpc_future is None
    assert rtc_client._in_flight_request_id is None
    assert isinstance(rtc_client.action_queue, ActionQueue)
    assert rtc_client.action_queue.empty()


def test_previous_action_requires_execution_confirmation(rtc_client):
    _, processed = _seed_queue(rtc_client)
    # Keep this test focused on the confirmation handshake, not refill submission.
    rtc_client.cfg.chunk_size_threshold = 0.0

    first = rtc_client.get_action(observation={}, task=None, timestep=0)
    assert first == {"joint_0": processed[0, 0].item(), "joint_1": processed[0, 1].item()}

    with pytest.raises(RuntimeError, match="previous remote action was not confirmed"):
        rtc_client.get_action(observation={}, task=None, timestep=1)

    rtc_client.mark_action_executed()
    second = rtc_client.get_action(observation={}, task=None, timestep=1)
    assert second == {"joint_0": processed[1, 0].item(), "joint_1": processed[1, 1].item()}
    rtc_client.mark_action_executed()

    with pytest.raises(RuntimeError, match="No remote RTC action is awaiting"):
        rtc_client.mark_action_executed()


def test_pending_result_rejects_misaligned_processed_action_dimension(rtc_client):
    _seed_queue(rtc_client)
    assert rtc_client._submit_if_needed(observation={}, task=None, timestep=0)
    request = _take_submitted_request(rtc_client)
    raw = torch.zeros(6, 2)
    processed = torch.zeros(6, 3)
    response = _make_response(request.request_id, raw, processed)
    with rtc_client._state_lock:
        rtc_client._in_flight_request_id = None
        rtc_client._pending_result = (request, response)

    with pytest.raises(ValueError, match="not aligned"):
        rtc_client._apply_pending_result()


def test_pending_result_rejects_server_that_ignores_cfg_off_contract(rtc_client):
    """A client asking for C-only RTC must never silently accept CFG output."""
    _seed_queue(rtc_client)
    assert rtc_client._submit_if_needed(observation={}, task=None, timestep=0)
    request = _take_submitted_request(rtc_client)
    raw = torch.zeros(6, 2)
    processed = torch.ones(6, 2)
    response = _make_response(request.request_id, raw, processed)
    response.use_cfg = True
    response.cfg_beta = 1.5

    with rtc_client._state_lock:
        rtc_client._in_flight_request_id = None
        rtc_client._pending_result = (request, response)

    with pytest.raises(RuntimeError, match="did not honor.*use_cfg"):
        rtc_client._apply_pending_result()
