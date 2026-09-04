#!/usr/bin/env python

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("grpc")

from lerobot.async_inference.helpers import RemoteActionChunk, TimedAction, TimedObservation
from lerobot.scripts.lerobot_infer_trc import (
    LateRTCResponseError,
    TrainedRTCInferenceClient,
    _AlignedActionQueue,
    _InferenceRequest,
    _validate_and_crop_response,
    _validate_sent_action,
)


def _request(
    *,
    prefix: torch.Tensor | None,
    prefix_steps: int,
    executed_steps: int = 0,
    queue_size: int = 5,
    expected_chunk_steps: int = 5,
) -> _InferenceRequest:
    return _InferenceRequest(
        request_id="trc-1",
        observation={"state": 0},
        timestep=executed_steps,
        action_prefix=prefix,
        conditioned_prefix_steps=prefix_steps,
        executed_steps_at_submit=executed_steps,
        queue_size_at_submit=queue_size,
        expected_chunk_steps=expected_chunk_steps,
        submitted_at=10.0,
    )


def _response(raw: torch.Tensor, request: _InferenceRequest) -> RemoteActionChunk:
    processed = raw + 100
    actions = [
        TimedAction(timestamp=float(i), timestep=request.timestep + i, action=row)
        for i, row in enumerate(processed)
    ]
    return RemoteActionChunk(
        request_id=request.request_id,
        actions=actions,
        raw_actions=raw,
        observation_timestep=request.timestep,
        training_time_rtc=True,
        prefix_steps=request.conditioned_prefix_steps,
        acp_positive_prompt=True,
        use_cfg=False,
        cfg_beta=1.0,
    )


def _bare_client(prefix_steps: int = 2) -> TrainedRTCInferenceClient:
    client = object.__new__(TrainedRTCInferenceClient)
    client.cfg = SimpleNamespace(
        rtc_prefix_steps=prefix_steps,
        actions_per_chunk=5,
        max_consecutive_late_chunks=3,
    )
    import threading

    client._state_lock = threading.RLock()
    client._pending_event = threading.Event()
    client._stop_event = threading.Event()
    client._request_thread = None
    client._pending_result = None
    client._pending_error = None
    client._request_sequence = 0
    client._executed_steps = 0
    client._queue = _AlignedActionQueue()
    client.accepted_chunks = 0
    client.late_chunks = 0
    client.consecutive_late_chunks = 0
    client.max_observed_delay = 0
    client.queue_underruns = 0
    return client


def test_response_crops_raw_and_processed_by_confirmed_actual_delay():
    prefix = torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    request = _request(prefix=prefix, prefix_steps=3)
    raw = torch.cat([prefix, torch.tensor([[6.0, 7.0], [8.0, 9.0]])])

    result = _validate_and_crop_response(
        request,
        _response(raw, request),
        executed_steps=2,
        received_at=10.5,
    )

    torch.testing.assert_close(result.raw_actions, raw[2:])
    torch.testing.assert_close(result.processed_actions, raw[2:] + 100)
    assert result.actual_delay == 2
    assert result.latency_s == pytest.approx(0.5)


def test_response_is_rejected_when_actual_delay_exceeds_conditioned_prefix():
    prefix = torch.tensor([[0.0], [1.0]])
    request = _request(prefix=prefix, prefix_steps=2, queue_size=5)
    raw = torch.cat([prefix, torch.tensor([[2.0], [3.0], [4.0]])])

    with pytest.raises(LateRTCResponseError, match="actual d=3"):
        _validate_and_crop_response(
            request,
            _response(raw, request),
            executed_steps=3,
        )


def test_response_rejects_misaligned_action_dimensions():
    prefix = torch.tensor([[0.0, 1.0]])
    request = _request(prefix=prefix, prefix_steps=1, expected_chunk_steps=4)
    raw = torch.cat([prefix, torch.ones(3, 2)])
    payload = _response(raw, request)
    payload.actions = [
        TimedAction(timestamp=float(i), timestep=i, action=torch.ones(3)) for i in range(raw.shape[0])
    ]

    with pytest.raises(ValueError, match="not aligned"):
        _validate_and_crop_response(request, payload, executed_steps=0)


def test_response_rejects_incomplete_action_chunk():
    prefix = torch.tensor([[0.0]])
    request = _request(prefix=prefix, prefix_steps=1, expected_chunk_steps=5)
    payload = _response(torch.arange(4, dtype=torch.float32).unsqueeze(1), request)

    with pytest.raises(ValueError, match="incomplete RTC chunk"):
        _validate_and_crop_response(request, payload, executed_steps=0)


def test_aligned_queue_uses_raw_prefix_and_confirms_only_after_send():
    client = _bare_client(prefix_steps=2)
    raw = torch.tensor([[1.0], [2.0], [3.0]])
    processed = raw + 100
    client._queue.replace(raw, processed)

    torch.testing.assert_close(client._queue.peek_processed(), processed[0])
    assert client._executed_steps == 0
    prefix, steps = client._queue.prefix(2)
    torch.testing.assert_close(prefix, raw[:2])
    assert steps == 2

    client._confirm_action_executed()
    assert client._executed_steps == 1
    torch.testing.assert_close(client._queue.peek_processed(), processed[1])


def test_modified_hardware_action_is_rejected_before_prefix_reuse():
    requested = {"shoulder.pos": 1.0, "elbow.pos": 2.0}

    _validate_sent_action(requested, dict(requested))
    with pytest.raises(RuntimeError, match="Robot modified action"):
        _validate_sent_action(
            requested,
            {"shoulder.pos": 1.0, "elbow.pos": 1.5},
        )


def test_submit_allows_only_one_in_flight_and_snapshots_raw_prefix(monkeypatch):
    import threading

    client = _bare_client(prefix_steps=2)
    raw = torch.tensor([[1.0], [2.0], [3.0]])
    client._queue.replace(raw, raw + 100)
    started = threading.Event()
    release = threading.Event()
    requests = []

    def blocked_worker(request):
        requests.append(request)
        started.set()
        release.wait(timeout=2)

    monkeypatch.setattr(client, "_request_worker", blocked_worker)
    assert client._start_request({"camera": torch.zeros(1)})
    assert started.wait(timeout=1)
    assert not client._start_request({"camera": torch.ones(1)})
    assert len(requests) == 1
    torch.testing.assert_close(requests[0].action_prefix, raw[:2])

    # The request owns a clone, not a view into the mutable queue.
    client._queue.raw[0].add_(50)
    torch.testing.assert_close(requests[0].action_prefix, raw[:2])
    release.set()
    client._request_thread.join(timeout=1)


def test_late_response_does_not_mutate_remaining_safe_queue():
    client = _bare_client(prefix_steps=2)
    old_raw = torch.arange(5, dtype=torch.float32).unsqueeze(1)
    client._queue.replace(old_raw, old_raw + 100)
    prefix = old_raw[:2].clone()
    request = _request(prefix=prefix, prefix_steps=2, queue_size=5)
    new_raw = torch.cat([prefix, torch.tensor([[20.0], [30.0], [40.0]])])

    for _ in range(3):
        client._confirm_action_executed()
    remaining_raw = torch.stack(list(client._queue.raw)).clone()
    remaining_processed = torch.stack(list(client._queue.processed)).clone()
    client._pending_result = (request, _response(new_raw, request), 10.5)

    assert not client._apply_pending_result()
    torch.testing.assert_close(torch.stack(list(client._queue.raw)), remaining_raw)
    torch.testing.assert_close(torch.stack(list(client._queue.processed)), remaining_processed)
    assert client.late_chunks == 1
    assert not client._has_active_request()


def test_control_thread_installs_valid_response_atomically():
    client = _bare_client(prefix_steps=3)
    old_raw = torch.arange(5, dtype=torch.float32).unsqueeze(1)
    client._queue.replace(old_raw, old_raw + 100)
    prefix = old_raw[:3].clone()
    request = _request(prefix=prefix, prefix_steps=3, queue_size=5)
    new_raw = torch.cat([prefix, torch.tensor([[20.0], [30.0]])])

    client._confirm_action_executed()
    client._confirm_action_executed()
    client._pending_result = (request, _response(new_raw, request), 10.5)

    assert client._apply_pending_result()
    torch.testing.assert_close(torch.stack(list(client._queue.raw)), new_raw[2:])
    torch.testing.assert_close(torch.stack(list(client._queue.processed)), new_raw[2:] + 100)
    assert client.accepted_chunks == 1


def test_server_runs_one_positive_conditioned_training_rtc_forward(monkeypatch):
    from lerobot.async_inference import policy_server as policy_server_module
    from lerobot.async_inference.helpers import TrainingTimeRTCMetadata
    from lerobot.utils.constants import ACTION

    seen = {}
    prefix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    class FakePolicy:
        config = SimpleNamespace(
            image_features={},
            output_features={ACTION: SimpleNamespace(shape=(2,))},
        )

        def predict_action_chunk(self, observation, **kwargs):
            seen["calls"] = seen.get("calls", 0) + 1
            seen["kwargs"] = kwargs
            return torch.cat([prefix, torch.tensor([[5.0, 6.0], [7.0, 8.0]])]).unsqueeze(0)

    class FakePreprocessor:
        def __call__(self, observation):
            seen["task"] = observation["task"]
            return observation

    server = object.__new__(policy_server_module.PolicyServer)
    server.policy = FakePolicy()
    server.preprocessor = FakePreprocessor()
    server.postprocessor = lambda action: action + 100
    server.lerobot_features = {}
    server.actions_per_chunk = 4
    server._training_time_rtc = True
    server._rtc_prefix_steps = 2
    server.last_processed_obs = None
    server.config = SimpleNamespace(environment_dt=1 / 30)
    monkeypatch.setattr(
        policy_server_module,
        "raw_observation_to_observation",
        lambda raw, _features, _images: raw,
    )
    observation = TimedObservation(
        timestamp=1.0,
        timestep=7,
        observation={"task": "Pick and place"},
        must_go=True,
        rtc_metadata=TrainingTimeRTCMetadata(
            request_id="trc-1",
            action_prefix=prefix,
            prefix_steps=2,
        ),
    )

    result = server._predict_action_chunk(observation)

    assert isinstance(result, RemoteActionChunk)
    assert seen["calls"] == 1
    assert seen["task"] == "Pick and place\nAdvantage: positive"
    assert seen["kwargs"]["training_time_rtc"] is True
    assert seen["kwargs"]["inference_delay"] == 2
    torch.testing.assert_close(seen["kwargs"]["rtc_action_prefix"], prefix)
    torch.testing.assert_close(result.raw_actions[:2], prefix)
    assert result.use_cfg is False
    assert result.cfg_beta == 1.0
