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

"""Remote policy client used by the 30 Hz recording loop.

For RTC-CFG, the main recording thread is the only queue consumer and the only
thread allowed to install a completed chunk. A single persistent worker performs
the RPC. This keeps the final raw CFG chunk and its postprocessed counterpart
strictly aligned while robot control and dataset writing remain in
``recording_loop.py``.
"""

from __future__ import annotations

import copy
import logging
import math
import pickle  # nosec
import threading
import time
from contextlib import suppress
from dataclasses import dataclass, field
from queue import Empty, Full, Queue
from typing import TYPE_CHECKING, Any

import torch

from lerobot.configs.types import RTCAttentionSchedule
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig

if TYPE_CHECKING:
    from lerobot.async_inference.helpers import TimedAction
    from lerobot.processor import RobotAction
    from lerobot.robots import Robot
else:
    RobotAction = dict[str, Any]


AGGREGATE_FUNCTIONS = {
    "weighted_average": lambda old, new: 0.3 * old + 0.7 * new,
    "latest_only": lambda old, new: new,
    "average": lambda old, new: 0.5 * old + 0.5 * new,
    "conservative": lambda old, new: 0.7 * old + 0.3 * new,
}


@dataclass
class RemotePolicyRecordConfig:
    """Configuration for using a remote async policy while recording."""

    enable: bool = False
    server_address: str = "localhost:8080"
    policy_type: str = ""
    pretrained_name_or_path: str = ""
    policy_device: str = "cpu"
    client_device: str = "cpu"
    actions_per_chunk: int = 1
    chunk_size_threshold: float = 0.0
    aggregate_fn_name: str = "weighted_average"
    obs_queue_timeout_s: float = 2.0
    rename_map: dict[str, str] = field(default_factory=dict)

    # Estimated RTC values are used inside denoising. The response is still
    # cropped with the number of actions actually sent while the RPC ran.
    # ``None`` preserves the server-authoritative behavior used by existing
    # remote-recording commands. An explicit boolean negotiates CFG on/off.
    use_cfg: bool | None = None
    rtc_enable: bool = False
    rtc_inference_delay: int = 20
    rtc_execution_horizon: int = 25
    rtc_max_guidance_weight: float = 10.0
    rtc_prefix_attention_schedule: RTCAttentionSchedule = RTCAttentionSchedule.LINEAR
    rtc_cfg_beta: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.rtc_prefix_attention_schedule, str):
            try:
                self.rtc_prefix_attention_schedule = RTCAttentionSchedule(
                    self.rtc_prefix_attention_schedule.upper()
                )
            except ValueError as exc:
                raise ValueError(
                    "`remote_policy.rtc_prefix_attention_schedule` must be one of "
                    f"{[schedule.value for schedule in RTCAttentionSchedule]}."
                ) from exc

        if self.use_cfg is not None and not isinstance(self.use_cfg, bool):
            raise ValueError("`remote_policy.use_cfg` must be true, false, or null.")

        if not self.enable:
            return
        if not self.server_address:
            raise ValueError("`remote_policy.server_address` cannot be empty.")
        if not self.policy_type:
            raise ValueError("`remote_policy.policy_type` is required when remote policy is enabled.")
        if not self.pretrained_name_or_path:
            raise ValueError(
                "`remote_policy.pretrained_name_or_path` is required when remote policy is enabled."
            )
        if self.actions_per_chunk <= 0:
            raise ValueError("`remote_policy.actions_per_chunk` must be positive.")
        if not 0 <= self.chunk_size_threshold <= 1:
            raise ValueError("`remote_policy.chunk_size_threshold` must be between 0 and 1.")
        if self.aggregate_fn_name not in AGGREGATE_FUNCTIONS:
            raise ValueError(
                f"Unknown `remote_policy.aggregate_fn_name={self.aggregate_fn_name}`. "
                f"Available: {sorted(AGGREGATE_FUNCTIONS)}"
            )
        if self.obs_queue_timeout_s < 0:
            raise ValueError("`remote_policy.obs_queue_timeout_s` must be non-negative.")

        if (
            not isinstance(self.rtc_inference_delay, int)
            or isinstance(self.rtc_inference_delay, bool)
            or self.rtc_inference_delay < 0
        ):
            raise ValueError("`remote_policy.rtc_inference_delay` must be a non-negative integer.")
        if (
            not isinstance(self.rtc_execution_horizon, int)
            or isinstance(self.rtc_execution_horizon, bool)
            or self.rtc_execution_horizon <= 0
        ):
            raise ValueError("`remote_policy.rtc_execution_horizon` must be a positive integer.")
        if not math.isfinite(self.rtc_max_guidance_weight) or self.rtc_max_guidance_weight <= 0:
            raise ValueError("`remote_policy.rtc_max_guidance_weight` must be finite and positive.")
        if not math.isfinite(self.rtc_cfg_beta) or self.rtc_cfg_beta < 0:
            raise ValueError("`remote_policy.rtc_cfg_beta` must be finite and non-negative.")
        if self.rtc_enable:
            if self.obs_queue_timeout_s == 0:
                raise ValueError("Remote RTC requires `remote_policy.obs_queue_timeout_s` to be positive.")
            if self.policy_type != "pi05":
                raise ValueError("Remote RTC currently supports only `policy_type=pi05`.")
            if self.rtc_inference_delay >= self.rtc_execution_horizon:
                raise ValueError(
                    "`remote_policy.rtc_inference_delay` must be smaller than "
                    "`remote_policy.rtc_execution_horizon`."
                )
            if self.rtc_execution_horizon > self.actions_per_chunk:
                raise ValueError(
                    "`remote_policy.rtc_execution_horizon` cannot exceed `actions_per_chunk`."
                )


@dataclass(frozen=True)
class _InferenceRequest:
    request_id: str
    episode_epoch: int
    observation: dict[str, Any]
    task: str | None
    timestep: int
    left_over: torch.Tensor | None
    executed_steps_at_submit: int
    queue_size_at_submit: int


class RemotePolicyActionClient:
    """Remote action source with one background chunk worker.

    RTC mode owns one :class:`ActionQueue` containing aligned ``raw CFG`` and
    ``processed CFG`` tensors. The worker never mutates it: responses enter a
    single pending slot and the 30 Hz main thread applies them before its next
    queue pop.
    """

    def __init__(self, cfg: RemotePolicyRecordConfig, robot: Robot, fps: int):
        if not cfg.enable:
            raise ValueError("RemotePolicyActionClient requires `cfg.enable=true`.")
        if fps <= 0:
            raise ValueError("RemotePolicyActionClient requires a positive fps.")

        self.cfg = cfg
        self.robot = robot
        self.environment_dt = 1 / fps

        import grpc

        from lerobot.transport import services_pb2, services_pb2_grpc
        from lerobot.transport.utils import grpc_channel_options

        self.services_pb2 = services_pb2
        self.channel = grpc.insecure_channel(
            cfg.server_address, grpc_channel_options(initial_backoff=f"{self.environment_dt:.4f}s")
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)

        if cfg.rtc_enable:
            rtc_config = RTCConfig(
                enabled=True,
                prefix_attention_schedule=cfg.rtc_prefix_attention_schedule,
                max_guidance_weight=cfg.rtc_max_guidance_weight,
                execution_horizon=cfg.rtc_execution_horizon,
            )
            self.action_queue: ActionQueue | list[TimedAction] = ActionQueue(rtc_config)
        else:
            # Preserve the legacy overlap-aggregation path when RTC is disabled.
            self.action_queue = []

        self.latest_action_timestep = -1
        self.latest_action: TimedAction | None = None
        self.aggregate_fn = AGGREGATE_FUNCTIONS[cfg.aggregate_fn_name]

        self._state_lock = threading.RLock()
        self._request_queue: Queue[_InferenceRequest | None] = Queue(maxsize=1)
        self._pending_event = threading.Event()
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._pending_result: tuple[_InferenceRequest, Any] | None = None
        self._worker_error: tuple[_InferenceRequest, BaseException] | None = None
        self._in_flight_request_id: str | None = None
        self._active_rpc_future: Any | None = None
        self._active_rpc_request_id: str | None = None
        self._episode_epoch = 0
        self._request_sequence = 0
        self._executed_steps = 0
        self._awaiting_execution_confirmation = False
        self._started = False

    @property
    def policy_id(self) -> str:
        return self.cfg.pretrained_name_or_path or self.cfg.policy_type or "remote_policy"

    def start(self) -> None:
        """Negotiate the payload format and start the persistent RPC worker."""
        if self._started:
            return

        from lerobot.async_inference.helpers import RemotePolicyConfig, map_robot_keys_to_lerobot_features

        self.stub.Ready(self.services_pb2.Empty())
        policy_config = RemotePolicyConfig(
            policy_type=self.cfg.policy_type,
            pretrained_name_or_path=self.cfg.pretrained_name_or_path,
            lerobot_features=map_robot_keys_to_lerobot_features(self.robot),
            actions_per_chunk=self.cfg.actions_per_chunk,
            device=self.cfg.policy_device,
            rename_map=self.cfg.rename_map,
            protocol_version=2 if self.cfg.rtc_enable else 1,
            return_raw_actions=self.cfg.rtc_enable,
            use_cfg=self.cfg.use_cfg,
            action_fps=1 / self.environment_dt,
            rtc_enabled=self.cfg.rtc_enable,
            rtc_inference_delay=self.cfg.rtc_inference_delay if self.cfg.rtc_enable else None,
            rtc_execution_horizon=self.cfg.rtc_execution_horizon if self.cfg.rtc_enable else None,
            rtc_max_guidance_weight=(
                self.cfg.rtc_max_guidance_weight if self.cfg.rtc_enable else None
            ),
            rtc_prefix_attention_schedule=(
                self.cfg.rtc_prefix_attention_schedule if self.cfg.rtc_enable else None
            ),
            rtc_cfg_beta=(
                self.cfg.rtc_cfg_beta if self.cfg.rtc_enable or self.cfg.use_cfg is True else None
            ),
        )
        self.stub.SendPolicyInstructions(self.services_pb2.PolicySetup(data=pickle.dumps(policy_config)))

        self._worker_thread = threading.Thread(
            target=self._chunk_worker,
            name="remote-policy-chunk-worker",
            daemon=True,
        )
        self._worker_thread.start()
        self._started = True
        logging.info(
            "Remote policy recording client connected to %s (CFG=%s, RTC=%s, d=%d, H=%d).",
            self.cfg.server_address,
            self.cfg.use_cfg,
            self.cfg.rtc_enable,
            self.cfg.rtc_inference_delay,
            self.cfg.rtc_execution_horizon,
        )

    def stop(self) -> None:
        """Stop the worker and invalidate any response that is still in flight."""
        if not self._started:
            self.channel.close()
            return

        with self._state_lock:
            self._episode_epoch += 1
            self._in_flight_request_id = None
            self._pending_result = None
            self._worker_error = None
            active_rpc = self._active_rpc_future
            self._active_rpc_future = None
            self._active_rpc_request_id = None
        self._stop_event.set()
        self._pending_event.set()
        if active_rpc is not None:
            active_rpc.cancel()
        self.channel.close()  # Cancels a blocking gRPC call.
        self._enqueue_worker_sentinel()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=max(self.cfg.obs_queue_timeout_s + 1.0, 1.0))
            if self._worker_thread.is_alive():
                logging.warning("Remote policy worker did not stop before the join timeout.")
        self._started = False

    def reset(self) -> None:
        """Start a new episode and make all older worker responses stale."""
        with self._state_lock:
            self._episode_epoch += 1
            self._in_flight_request_id = None
            self._pending_result = None
            self._worker_error = None
            active_rpc = self._active_rpc_future
            self._active_rpc_future = None
            self._active_rpc_request_id = None
            self._executed_steps = 0
            self._awaiting_execution_confirmation = False
            self.latest_action_timestep = -1
            self.latest_action = None
            if self.cfg.rtc_enable:
                assert isinstance(self.action_queue, ActionQueue)
                self.action_queue.clear()
            else:
                assert isinstance(self.action_queue, list)
                self.action_queue.clear()
            self._drain_queued_requests()
            self._pending_event.clear()

        if active_rpc is not None:
            active_rpc.cancel()

        # Reset server deduplication and policy episode state. An older active
        # RPC may still finish, but its epoch/request id prevents installation.
        if self._started:
            self.stub.Ready(self.services_pb2.Empty())

    def mark_action_executed(self) -> None:
        """Confirm that the action returned by ``get_action`` reached the robot.

        ``recording_loop`` calls this immediately after ``robot.send_action``
        succeeds. RTC's real delay is based on this counter, not queue pops or
        wall-clock rounding.
        """
        if not self.cfg.rtc_enable:
            return
        with self._state_lock:
            if not self._awaiting_execution_confirmation:
                raise RuntimeError("No remote RTC action is awaiting execution confirmation.")
            self._executed_steps += 1
            self._awaiting_execution_confirmation = False

    def _enqueue_worker_sentinel(self) -> None:
        try:
            self._request_queue.put_nowait(None)
        except Full:
            try:
                self._request_queue.get_nowait()
                self._request_queue.task_done()
            except Empty:
                pass
            with suppress(Full):
                self._request_queue.put_nowait(None)

    def _drain_queued_requests(self) -> None:
        while True:
            try:
                item = self._request_queue.get_nowait()
            except Empty:
                return
            self._request_queue.task_done()
            if item is None:
                self._stop_event.set()

    def _queue_size(self) -> int:
        if self.cfg.rtc_enable:
            assert isinstance(self.action_queue, ActionQueue)
            return self.action_queue.qsize()
        assert isinstance(self.action_queue, list)
        return len(self.action_queue)

    def _ready_to_send_observation(self) -> bool:
        queue_size = self._queue_size()
        if queue_size == 0:
            return True
        return queue_size / self.cfg.actions_per_chunk <= self.cfg.chunk_size_threshold

    def _submit_if_needed(self, observation: dict, task: str | None, timestep: int) -> bool:
        with self._state_lock:
            if self._stop_event.is_set() or not self._ready_to_send_observation():
                return False
            if self._in_flight_request_id is not None or self._pending_result is not None:
                return False

            left_over = None
            queue_size = self._queue_size()
            if self.cfg.rtc_enable:
                assert isinstance(self.action_queue, ActionQueue)
                snapshot = self.action_queue.snapshot()
                left_over = snapshot.left_over
                queue_size = snapshot.queue_size

            self._request_sequence += 1
            request_id = f"{self._episode_epoch}:{self._request_sequence}"
            request = _InferenceRequest(
                request_id=request_id,
                episode_epoch=self._episode_epoch,
                observation=copy.deepcopy(observation),
                task=task,
                timestep=max(timestep, 0),
                left_over=left_over,
                executed_steps_at_submit=self._executed_steps,
                queue_size_at_submit=queue_size,
            )
            self._in_flight_request_id = request_id
            self._worker_error = None
            self._pending_event.clear()
            try:
                self._request_queue.put_nowait(request)
            except Full as exc:
                self._in_flight_request_id = None
                raise RuntimeError("Remote policy worker queue unexpectedly contains another request.") from exc
            return True

    def _run_cancellable_rpc(
        self, request: _InferenceRequest, rpc, argument, *, timeout: float | None
    ):
        """Run one gRPC future that reset/stop can cancel without waiting for its deadline."""
        rpc_future = rpc.future(argument, timeout=timeout)
        with self._state_lock:
            if (
                self._stop_event.is_set()
                or request.episode_epoch != self._episode_epoch
                or request.request_id != self._in_flight_request_id
            ):
                rpc_future.cancel()
                raise RuntimeError(f"Remote policy request {request.request_id} was invalidated.")
            self._active_rpc_future = rpc_future
            self._active_rpc_request_id = request.request_id
        try:
            return rpc_future.result()
        finally:
            with self._state_lock:
                if self._active_rpc_future is rpc_future:
                    self._active_rpc_future = None
                    self._active_rpc_request_id = None

    def _send_observation(self, request: _InferenceRequest) -> None:
        from lerobot.async_inference.helpers import RTCInferenceMetadata, TimedObservation
        from lerobot.transport.utils import send_bytes_in_chunks

        raw_observation = dict(request.observation)
        if request.task is not None:
            raw_observation["task"] = request.task

        timed_observation = TimedObservation(
            timestamp=time.time(),
            timestep=request.timestep,
            observation=raw_observation,
            must_go=True,
            rtc_metadata=(
                RTCInferenceMetadata(
                    request_id=request.request_id,
                    prev_chunk_left_over=request.left_over,
                    inference_delay=self.cfg.rtc_inference_delay,
                    execution_horizon=self.cfg.rtc_execution_horizon,
                )
                if self.cfg.rtc_enable
                else None
            ),
        )
        observation_iterator = send_bytes_in_chunks(
            pickle.dumps(timed_observation),
            self.services_pb2.Observation,
            log_prefix="[RECORD_REMOTE_POLICY] Observation",
            silent=True,
        )
        self._run_cancellable_rpc(
            request,
            self.stub.SendObservations,
            observation_iterator,
            timeout=self.cfg.obs_queue_timeout_s if self.cfg.obs_queue_timeout_s > 0 else None,
        )

    def _receive_actions(self, request: _InferenceRequest) -> Any:
        deadline_t = time.perf_counter() + self.cfg.obs_queue_timeout_s
        while not self._stop_event.is_set():
            remaining = deadline_t - time.perf_counter()
            if self.cfg.obs_queue_timeout_s > 0 and remaining <= 0:
                raise TimeoutError("Timed out waiting for remote policy actions.")
            actions_chunk = self._run_cancellable_rpc(
                request,
                self.stub.GetActions,
                self.services_pb2.Empty(),
                timeout=remaining if self.cfg.obs_queue_timeout_s > 0 else None,
            )
            if actions_chunk.data:
                return pickle.loads(actions_chunk.data)  # nosec

            if self.cfg.obs_queue_timeout_s == 0 or time.perf_counter() >= deadline_t:
                raise TimeoutError("Timed out waiting for remote policy actions.")
        raise RuntimeError("Remote policy client stopped while waiting for actions.")

    def _chunk_worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                request = self._request_queue.get(timeout=0.1)
            except Empty:
                continue
            try:
                if request is None:
                    return
                self._send_observation(request)
                response = self._receive_actions(request)
                with self._state_lock:
                    if (
                        not self._stop_event.is_set()
                        and request.episode_epoch == self._episode_epoch
                        and request.request_id == self._in_flight_request_id
                    ):
                        self._pending_result = (request, response)
                        self._in_flight_request_id = None
                        self._pending_event.set()
            except Exception as exc:
                if request is not None:
                    with self._state_lock:
                        if (
                            request.episode_epoch == self._episode_epoch
                            and request.request_id == self._in_flight_request_id
                        ):
                            self._worker_error = (request, exc)
                            self._in_flight_request_id = None
                            self._pending_event.set()
            finally:
                self._request_queue.task_done()

    def _apply_pending_result(self) -> bool:
        with self._state_lock:
            pending = self._pending_result
            self._pending_result = None
            if self._worker_error is None:
                self._pending_event.clear()
        if pending is None:
            return False

        request, payload = pending
        if request.episode_epoch != self._episode_epoch:
            return False

        if not self.cfg.rtc_enable:
            if not isinstance(payload, list):
                # A v2 server response is also accepted by the new client, but
                # legacy mode consumes only its processed TimedActions.
                payload = getattr(payload, "actions", None)
            if not isinstance(payload, list):
                raise TypeError(f"Expected a list of TimedAction, got {type(payload).__name__}.")
            with self._state_lock:
                self._merge_legacy_actions(payload)
            return True

        from lerobot.async_inference.helpers import RemoteActionChunk

        if not isinstance(payload, RemoteActionChunk):
            raise RuntimeError(
                "RTC recording requires a v2 RemoteActionChunk response. "
                "Start the policy server with `acp_inference.enable=true` and matching client RTC settings."
            )
        if payload.request_id != request.request_id:
            raise RuntimeError(
                f"RTC response request_id mismatch: expected {request.request_id}, got {payload.request_id}."
            )
        if not payload.rtc_enabled:
            raise RuntimeError("The policy server returned a chunk without RTC enabled.")
        if self.cfg.use_cfg is not None:
            response_schedule = getattr(payload, "prefix_attention_schedule", None)
            if isinstance(response_schedule, str):
                with suppress(ValueError):
                    response_schedule = RTCAttentionSchedule(response_schedule.upper())

            effective_horizon = self.cfg.rtc_execution_horizon
            if request.left_over is not None and request.left_over.shape[0] > 0:
                effective_horizon = min(effective_horizon, int(request.left_over.shape[0]))
            effective_horizon = min(effective_horizon, self.cfg.actions_per_chunk)
            effective_delay = min(self.cfg.rtc_inference_delay, effective_horizon)

            expected = {
                "use_cfg": self.cfg.use_cfg,
                "inference_delay": effective_delay,
                "execution_horizon": effective_horizon,
                "max_guidance_weight": self.cfg.rtc_max_guidance_weight,
                "prefix_attention_schedule": self.cfg.rtc_prefix_attention_schedule,
            }
            actual = {
                "use_cfg": getattr(payload, "use_cfg", None),
                "inference_delay": getattr(payload, "inference_delay", None),
                "execution_horizon": getattr(payload, "execution_horizon", None),
                "max_guidance_weight": getattr(payload, "max_guidance_weight", None),
                "prefix_attention_schedule": response_schedule,
            }
            # Beta has no meaning in the CFG-off single conditional branch.
            if self.cfg.use_cfg:
                expected["cfg_beta"] = self.cfg.rtc_cfg_beta
                actual["cfg_beta"] = getattr(payload, "cfg_beta", None)

            mismatches = [
                f"{field_name}: client={expected_value!r}, server={actual[field_name]!r}"
                for field_name, expected_value in expected.items()
                if actual[field_name] != expected_value
            ]
            if mismatches:
                raise RuntimeError(
                    "The policy server did not honor the client inference contract: "
                    + "; ".join(mismatches)
                )

        raw_actions = payload.raw_actions
        if not isinstance(raw_actions, torch.Tensor) or raw_actions.ndim != 2:
            raise ValueError("RTC response raw_actions must be a [T, A] tensor.")
        if not isinstance(payload.actions, list) or not payload.actions:
            raise ValueError("RTC response must contain at least one processed TimedAction.")
        processed_actions = torch.stack([action.get_action() for action in payload.actions]).detach().cpu()
        raw_actions = raw_actions.detach().cpu()
        if raw_actions.shape != processed_actions.shape:
            raise ValueError(
                "RTC raw and processed response chunks are not aligned: "
                f"{tuple(raw_actions.shape)} != {tuple(processed_actions.shape)}."
            )
        if not bool(torch.isfinite(raw_actions).all()) or not bool(torch.isfinite(processed_actions).all()):
            raise FloatingPointError("RTC response contains non-finite raw or processed actions.")

        with self._state_lock:
            if request.episode_epoch != self._episode_epoch:
                return False
            actual_delay = self._executed_steps - request.executed_steps_at_submit
            if actual_delay < 0:
                raise RuntimeError("RTC executed-step counter moved backwards.")
            if actual_delay >= raw_actions.shape[0]:
                logging.error(
                    "Discarding fully stale RTC chunk %s: %d actions executed during a %d-step response.",
                    request.request_id,
                    actual_delay,
                    raw_actions.shape[0],
                )
                return False

            assert isinstance(self.action_queue, ActionQueue)
            self.action_queue.merge(
                original_actions=raw_actions,
                processed_actions=processed_actions,
                real_delay=actual_delay,
                action_index_before_inference=None,
            )
            logging.debug(
                "Installed RTC chunk %s: submit_queue=%d, actual_delay=%d, remaining=%d.",
                request.request_id,
                request.queue_size_at_submit,
                actual_delay,
                self.action_queue.qsize(),
            )
        return True

    def _raise_or_log_worker_error(self) -> None:
        with self._state_lock:
            worker_error = self._worker_error
            self._worker_error = None
            if self._pending_result is None:
                self._pending_event.clear()
            queue_empty = self._queue_size() == 0
        if worker_error is None:
            return
        request, error = worker_error
        if request.episode_epoch != self._episode_epoch:
            return
        if queue_empty:
            raise RuntimeError(f"Remote policy request {request.request_id} failed.") from error
        logging.warning(
            "Remote policy request %s failed while %d queued actions remain; retrying at low water mark: %s",
            request.request_id,
            self._queue_size(),
            error,
        )

    def _merge_legacy_actions(self, incoming_actions: list[TimedAction]) -> None:
        assert isinstance(self.action_queue, list)
        current_actions = {action.get_timestep(): action for action in self.action_queue}
        merged: dict[int, TimedAction] = {}

        for action in self.action_queue:
            if action.get_timestep() > self.latest_action_timestep:
                merged[action.get_timestep()] = action

        for action in incoming_actions:
            timestep = action.get_timestep()
            if timestep <= self.latest_action_timestep:
                continue
            if timestep in current_actions:
                from lerobot.async_inference.helpers import TimedAction

                action = TimedAction(
                    timestamp=action.get_timestamp(),
                    timestep=timestep,
                    action=self.aggregate_fn(current_actions[timestep].get_action(), action.get_action()),
                )
            merged[timestep] = action
        self.action_queue = [merged[timestep] for timestep in sorted(merged)]

    def _pop_action(self) -> torch.Tensor | None:
        with self._state_lock:
            if self.cfg.rtc_enable:
                assert isinstance(self.action_queue, ActionQueue)
                action = self.action_queue.get()
                if action is not None:
                    self._awaiting_execution_confirmation = True
                return action

            assert isinstance(self.action_queue, list)
            if not self.action_queue:
                return None
            timed_action = self.action_queue.pop(0)
            self.latest_action = timed_action
            self.latest_action_timestep = timed_action.get_timestep()
            return timed_action.get_action()

    def _tensor_to_action(self, action_tensor: torch.Tensor) -> RobotAction:
        if self.cfg.client_device != "cpu" and action_tensor.device.type != self.cfg.client_device:
            action_tensor = action_tensor.to(self.cfg.client_device)
        else:
            action_tensor = action_tensor.cpu()
        if action_tensor.numel() != len(self.robot.action_features):
            raise ValueError(
                "Remote action dimension does not match robot action features: "
                f"{action_tensor.numel()} != {len(self.robot.action_features)}."
            )
        return {key: action_tensor[i].item() for i, key in enumerate(self.robot.action_features)}

    def get_action(self, observation: dict, task: str | None, timestep: int) -> RobotAction:
        """Return one processed action and refill asynchronously at low water mark.

        The first chunk is a synchronous bootstrap because there is no safe
        action to execute yet. Once primed, this method only blocks if the queue
        genuinely underruns.
        """
        if not self._started:
            raise RuntimeError("RemotePolicyActionClient.start() must be called before get_action().")
        if self.cfg.rtc_enable and self._awaiting_execution_confirmation:
            raise RuntimeError(
                "The previous remote action was not confirmed. Call mark_action_executed() "
                "after robot.send_action succeeds."
            )

        wait_timeout = max(self.cfg.obs_queue_timeout_s + 1.0, 1.0)
        deadline = time.perf_counter() + wait_timeout
        while True:
            self._apply_pending_result()
            self._raise_or_log_worker_error()
            self._submit_if_needed(observation=observation, task=task, timestep=timestep)

            action_tensor = self._pop_action()
            if action_tensor is not None:
                return self._tensor_to_action(action_tensor)

            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError(
                    "Remote RTC action queue underrun: no fresh action arrived before the safety timeout."
                )
            self._pending_event.wait(timeout=min(remaining, 0.05))
