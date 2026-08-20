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

import pytest
import torch

from lerobot.async_inference.helpers import RemoteActionChunk, TimedAction
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.scripts.recording_remote_policy import RemotePolicyActionClient, RemotePolicyRecordConfig


class _TraceSpy:
    def __init__(self):
        self.events = []
        self.lock = threading.Lock()

    def record(self, event, **fields):
        with self.lock:
            self.events.append({"event": event, **fields})


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
        rtc_enable=True,
        rtc_inference_delay=1,
        rtc_execution_horizon=3,
    )

    client = RemotePolicyActionClient.__new__(RemotePolicyActionClient)
    client.cfg = cfg
    client.robot = SimpleNamespace(action_features={"joint_0": object(), "joint_1": object()})
    client.environment_dt = 1 / 30
    client.action_queue = ActionQueue(RTCConfig(enabled=True, execution_horizon=cfg.rtc_execution_horizon))
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
    client._active_rpc_stage = None
    client._episode_epoch = 0
    client._request_sequence = 0
    client._session_id = "test-session"
    client._executed_steps = 0
    client._awaiting_execution_confirmation = False
    client._last_get_action_metrics = None
    client._started = True
    client._rtc_trace = _TraceSpy()
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
        rtc_enabled=True,
        inference_delay=1,
        execution_horizon=3,
    )


def _take_submitted_request(client: RemotePolicyActionClient):
    request = client._request_queue.get_nowait()
    client._request_queue.task_done()
    assert request is not None
    return request


def _trace_events(client: RemotePolicyActionClient, event: str) -> list[dict]:
    return [record for record in client._rtc_trace.events if record["event"] == event]


def test_submit_uses_one_in_flight_request_and_raw_snapshot(rtc_client):
    raw, processed = _seed_queue(rtc_client)
    barrier = threading.Barrier(9)
    results: list[bool] = []
    errors: list[BaseException] = []
    results_lock = threading.Lock()

    def submit() -> None:
        barrier.wait()
        try:
            result = rtc_client._submit_if_needed(observation={"state": [0.0, 1.0]}, task="test", timestep=5)
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
    assert request.request_id == "test-session:0:1"
    assert torch.equal(request.left_over, raw)
    assert not torch.equal(request.left_over, processed)
    submitted = _trace_events(rtc_client, "request_submitted")
    assert len(submitted) == 1
    assert submitted[0]["request_id"] == request.request_id
    assert submitted[0]["queue_size_at_submit"] == 6
    assert submitted[0]["leftover_steps"] == 6
    assert submitted[0]["observation_timestep"] == 5
    assert submitted[0]["observation_snapshot_ms"] >= 0

    # Removing the queued item does not permit another request while its RPC is active.
    assert not rtc_client._submit_if_needed(observation={}, task=None, timestep=6)


def test_first_request_has_no_leftover(rtc_client):
    assert rtc_client._submit_if_needed(observation={}, task=None, timestep=0)
    request = _take_submitted_request(rtc_client)

    assert request.left_over is None
    assert request.queue_size_at_submit == 0
    assert request.executed_steps_at_submit == 0


def test_request_and_response_transport_timings_are_split(rtc_client):
    from lerobot.transport.utils import CHUNK_SIZE

    class _Observation:
        def __init__(self, transfer_state, data):
            self.transfer_state = transfer_state
            self.data = data

    class _Empty:
        pass

    class _ImmediateFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

        def cancel(self):
            return True

    class _SendRPC:
        def __init__(self):
            self.chunks = []

        def future(self, argument, timeout=None):
            assert timeout == rtc_client.cfg.obs_queue_timeout_s
            self.chunks = list(argument)
            return _ImmediateFuture(SimpleNamespace())

    response = {"status": "ok", "steps": 6}
    serialized_response = pickle.dumps(response)

    class _GetRPC:
        def future(self, _argument, timeout=None):
            assert timeout is not None and timeout > 0
            return _ImmediateFuture(SimpleNamespace(data=serialized_response))

    send_rpc = _SendRPC()
    rtc_client.services_pb2 = SimpleNamespace(Observation=_Observation, Empty=_Empty)
    rtc_client.stub = SimpleNamespace(SendObservations=send_rpc, GetActions=_GetRPC())
    assert rtc_client._submit_if_needed(
        observation={"state": [0.0, 1.0], "camera": bytearray(64)},
        task="test",
        timestep=7,
    )
    request = _take_submitted_request(rtc_client)

    send_metrics = rtc_client._send_observation(request)
    decoded_response, receive_metrics = rtc_client._receive_actions(request)

    assert decoded_response == response
    assert send_metrics.serialize_ms >= 0
    assert send_metrics.rpc_ms >= 0
    assert send_metrics.total_ms >= send_metrics.serialize_ms
    assert send_metrics.payload_bytes == sum(len(chunk.data) for chunk in send_rpc.chunks)
    assert send_metrics.chunk_count == (send_metrics.payload_bytes + CHUNK_SIZE - 1) // CHUNK_SIZE
    assert len(send_rpc.chunks) == send_metrics.chunk_count
    assert receive_metrics.rpc_ms >= 0
    assert receive_metrics.deserialize_ms >= 0
    assert receive_metrics.total_ms >= receive_metrics.deserialize_ms
    assert receive_metrics.payload_bytes == len(serialized_response)
    assert receive_metrics.poll_count == 1

    serialized_events = _trace_events(rtc_client, "observation_serialized")
    sent_events = _trace_events(rtc_client, "observation_sent")
    response_events = _trace_events(rtc_client, "action_response_deserialized")
    assert serialized_events[0]["request_payload_bytes"] == send_metrics.payload_bytes
    assert serialized_events[0]["request_chunk_count"] == send_metrics.chunk_count
    assert sent_events[0]["observation_upload_rpc_ms"] == send_metrics.rpc_ms
    assert response_events[0]["get_actions_rpc_ms"] == receive_metrics.rpc_ms
    assert response_events[0]["response_deserialize_ms"] == receive_metrics.deserialize_ms
    assert response_events[0]["response_payload_bytes"] == len(serialized_response)


def test_control_loop_trace_includes_remote_queue_wait_metrics(rtc_client):
    _, processed = _seed_queue(rtc_client)
    rtc_client.cfg.chunk_size_threshold = 0.0

    action = rtc_client.get_action(observation={}, task=None, timestep=4)
    assert action["joint_0"] == processed[0, 0].item()
    rtc_client.mark_action_executed()
    rtc_client.record_control_loop_metrics(
        timestep=4,
        get_observation_ms=1.0,
        get_action_ms=2.0,
        loop_total_ms=33.0,
        deadline_missed=False,
    )

    events = _trace_events(rtc_client, "control_loop_step")
    assert len(events) == 1
    assert events[0]["observation_timestep"] == 4
    assert events[0]["remote_get_action_ms"] >= 0
    assert events[0]["remote_queue_wait_ms"] >= 0
    assert events[0]["queue_size_after_pop"] == 5
    assert events[0]["get_observation_ms"] == 1.0
    assert events[0]["get_action_ms"] == 2.0


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

    installed = _trace_events(rtc_client, "chunk_installed")
    assert len(installed) == 1
    assert installed[0]["request_id"] == request.request_id
    assert installed[0]["queue_size_at_submit"] == 6
    assert installed[0]["leftover_steps_at_submit"] == 6
    assert installed[0]["response_steps"] == 6
    assert installed[0]["actual_delay"] == 3
    assert installed[0]["discarded_prefix_steps"] == 3
    assert installed[0]["queue_size_after_install"] == 3


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
    reset_events = _trace_events(rtc_client, "episode_reset")
    assert len(reset_events) == 1
    assert reset_events[0]["old_epoch"] == 0
    assert reset_events[0]["new_epoch"] == 1
    assert reset_events[0]["queue_size_before"] == 6
    assert reset_events[0]["cancelled_request_id"] == request.request_id
    assert reset_events[0]["cancellation_requested"] is True


def test_worker_error_trace_distinguishes_retry_and_fatal(rtc_client):
    _seed_queue(rtc_client)
    assert rtc_client._submit_if_needed(observation={}, task=None, timestep=0)
    request = _take_submitted_request(rtc_client)
    with rtc_client._state_lock:
        rtc_client._in_flight_request_id = None
        rtc_client._worker_error = (request, TimeoutError("network stalled"))

    rtc_client._raise_or_log_worker_error()

    retry = _trace_events(rtc_client, "request_retry_scheduled")
    assert len(retry) == 1
    assert retry[0]["request_id"] == request.request_id
    assert retry[0]["queue_size"] == 6
    assert retry[0]["error_type"] == "TimeoutError"
    assert retry[0]["error_message"] == "network stalled"

    assert isinstance(rtc_client.action_queue, ActionQueue)
    rtc_client.action_queue.clear()
    with rtc_client._state_lock:
        rtc_client._worker_error = (request, RuntimeError("server failed"))
    with pytest.raises(RuntimeError, match="Remote policy request"):
        rtc_client._raise_or_log_worker_error()

    fatal = _trace_events(rtc_client, "request_failure_escalated")
    assert len(fatal) == 1
    assert fatal[0]["request_id"] == request.request_id
    assert fatal[0]["error_type"] == "RuntimeError"


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
