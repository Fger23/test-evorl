#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Remote PI0.5 inference for ACP-positive Training-Time RTC checkpoints.

The policy and checkpoint live on `policy_server`. The robot-side process keeps
two aligned queues: normalized actions for the next learned clean prefix and
postprocessed actions for hardware execution. There is exactly one model call
per refill: no unconditional branch and no CFG blend (beta=1 identity).
"""

import copy
import logging
import math
import pickle  # nosec
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pprint import pformat
from typing import Any

import grpc
import torch

from lerobot.async_inference.helpers import (
    RemoteActionChunk,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    TrainingTimeRTCMetadata,
    map_robot_keys_to_lerobot_features,
)
from lerobot.cameras import CameraConfig  # noqa: F401
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.reachy2_camera.configuration_reachy2_camera import (  # noqa: F401
    Reachy2CameraConfig,
)
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.rl.acp_tags import ACP_TAG_KEY
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_piper_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    piper_follower,
    reachy2,
    so_follower,
    unitree_g1 as unitree_g1_robot,
)
from lerobot.transport import services_pb2, services_pb2_grpc  # type: ignore
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging


@dataclass
class TrainedRTCInferenceConfig:
    """Robot-side options for single-branch ACP plus Training-Time RTC."""

    robot: RobotConfig
    pretrained_name_or_path: str
    task: str
    server_address: str = "localhost:8090"
    policy_device: str = "cuda"
    actions_per_chunk: int = 50
    rtc_prefix_steps: int = 21
    refill_threshold: float = 0.7
    fps: int = 30
    duration_s: float | None = None
    rpc_timeout_s: float = 30.0
    setup_timeout_s: float = 600.0
    queue_wait_timeout_s: float = 60.0
    max_consecutive_late_chunks: int = 3
    status_interval_s: float = 1.0
    rename_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.pretrained_name_or_path:
            raise ValueError("`pretrained_name_or_path` cannot be empty.")
        if not self.task:
            raise ValueError("`task` cannot be empty.")
        if any(line.strip().startswith(f"{ACP_TAG_KEY}:") for line in self.task.splitlines()):
            raise ValueError("Pass the base task only; `Advantage: positive` is appended by the server.")
        if not self.server_address:
            raise ValueError("`server_address` cannot be empty.")
        if not self.policy_device:
            raise ValueError("`policy_device` cannot be empty.")
        if (
            not isinstance(self.actions_per_chunk, int)
            or isinstance(self.actions_per_chunk, bool)
            or self.actions_per_chunk <= 1
        ):
            raise ValueError("`actions_per_chunk` must be an integer greater than one.")
        if (
            not isinstance(self.rtc_prefix_steps, int)
            or isinstance(self.rtc_prefix_steps, bool)
            or not 0 < self.rtc_prefix_steps < self.actions_per_chunk
        ):
            raise ValueError("`rtc_prefix_steps` must satisfy 0 < d < actions_per_chunk.")
        if (
            not isinstance(self.refill_threshold, int | float)
            or isinstance(self.refill_threshold, bool)
            or not math.isfinite(float(self.refill_threshold))
            or not 0 < self.refill_threshold < 1
        ):
            raise ValueError("`refill_threshold` must be a finite number in (0, 1).")
        first_refill_size = math.floor(self.actions_per_chunk * self.refill_threshold)
        if first_refill_size < self.rtc_prefix_steps:
            raise ValueError(
                "The refill low-water mark must retain at least rtc_prefix_steps actions: "
                f"floor({self.actions_per_chunk} * {self.refill_threshold})="
                f"{first_refill_size} < {self.rtc_prefix_steps}."
            )
        if not isinstance(self.fps, int) or isinstance(self.fps, bool) or self.fps <= 0:
            raise ValueError("`fps` must be a positive integer.")
        for field_name in ("rpc_timeout_s", "setup_timeout_s", "queue_wait_timeout_s"):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(float(value))
                or value <= 0
            ):
                raise ValueError(f"`{field_name}` must be finite and positive.")
        if self.duration_s is not None and (
            isinstance(self.duration_s, bool)
            or not isinstance(self.duration_s, int | float)
            or not math.isfinite(float(self.duration_s))
            or self.duration_s <= 0
        ):
            raise ValueError("`duration_s` must be finite and positive when set.")
        if (
            not isinstance(self.max_consecutive_late_chunks, int)
            or isinstance(self.max_consecutive_late_chunks, bool)
            or self.max_consecutive_late_chunks <= 0
        ):
            raise ValueError("`max_consecutive_late_chunks` must be a positive integer.")
        if (
            isinstance(self.status_interval_s, bool)
            or not isinstance(self.status_interval_s, int | float)
            or not math.isfinite(float(self.status_interval_s))
            or self.status_interval_s <= 0
        ):
            raise ValueError("`status_interval_s` must be finite and positive.")

    @property
    def environment_dt(self) -> float:
        return 1.0 / self.fps


@dataclass(frozen=True)
class _InferenceRequest:
    request_id: str
    observation: dict[str, Any]
    timestep: int
    action_prefix: torch.Tensor | None
    conditioned_prefix_steps: int
    executed_steps_at_submit: int
    queue_size_at_submit: int
    expected_chunk_steps: int
    submitted_at: float


@dataclass(frozen=True)
class _CroppedActionChunk:
    raw_actions: torch.Tensor
    processed_actions: torch.Tensor
    actual_delay: int
    latency_s: float


class LateRTCResponseError(RuntimeError):
    """The robot consumed more actions than the model's clean prefix covered."""

    def __init__(self, *, request_id: str, actual_delay: int, conditioned_prefix_steps: int):
        self.request_id = request_id
        self.actual_delay = actual_delay
        self.conditioned_prefix_steps = conditioned_prefix_steps
        super().__init__(
            f"RTC response {request_id} is unsafe: actual d={actual_delay} exceeds "
            f"conditioned d={conditioned_prefix_steps}."
        )


def _validate_sent_action(
    requested_action: dict[str, float],
    sent_action: Any,
) -> None:
    """Reject hardware-side action changes that invalidate the normalized RTC prefix."""
    if not isinstance(sent_action, dict):
        raise TypeError(
            "robot.send_action must return the action actually sent as a dictionary; "
            f"got {type(sent_action).__name__}."
        )
    if sent_action.keys() != requested_action.keys():
        raise RuntimeError(
            "robot.send_action returned different action keys. RTC cannot safely reuse "
            "the policy's normalized action as a clean prefix."
        )
    for key, requested_value in requested_action.items():
        try:
            requested_float = float(requested_value)
            sent_float = float(sent_action[key])
        except (TypeError, ValueError) as error:
            raise TypeError(f"Robot action {key!r} is not numeric.") from error
        if (
            not math.isfinite(requested_float)
            or not math.isfinite(sent_float)
            or not math.isclose(requested_float, sent_float, rel_tol=1e-6, abs_tol=1e-6)
        ):
            raise RuntimeError(
                f"Robot modified action {key!r}: requested={requested_float}, "
                f"sent={sent_float}. Stopping because the learned RTC clean prefix "
                "would no longer match the executed trajectory."
            )


def _validate_and_crop_response(
    request: _InferenceRequest,
    payload: Any,
    *,
    executed_steps: int,
    received_at: float | None = None,
) -> _CroppedActionChunk:
    """Validate the v2 contract and crop actions already executed during inference."""
    if not isinstance(payload, RemoteActionChunk):
        raise TypeError(
            "Training-Time RTC requires a RemoteActionChunk response; "
            f"got {type(payload).__name__}. Reinstall the same checkout on both endpoints."
        )
    if payload.request_id != request.request_id:
        raise RuntimeError(
            f"RTC request_id mismatch: expected {request.request_id}, got {payload.request_id}."
        )
    if payload.observation_timestep != request.timestep:
        raise RuntimeError(
            "RTC observation timestep mismatch: "
            f"client={request.timestep}, server={payload.observation_timestep}."
        )
    if payload.training_time_rtc is not True:
        raise RuntimeError("Server response did not enable Training-Time RTC.")
    if (
        payload.acp_positive_prompt is not True
        or payload.use_cfg is not False
        or isinstance(payload.cfg_beta, bool)
        or not isinstance(payload.cfg_beta, int | float)
        or float(payload.cfg_beta) != 1.0
    ):
        raise RuntimeError("Server violated the ACP contract: expected positive branch, CFG off, beta=1.")
    if payload.prefix_steps != request.conditioned_prefix_steps:
        raise RuntimeError(
            "Server used a different RTC prefix length: "
            f"client={request.conditioned_prefix_steps}, server={payload.prefix_steps}."
        )

    actual_delay = executed_steps - request.executed_steps_at_submit
    if actual_delay < 0:
        raise RuntimeError("Executed-step counter moved backwards.")
    if actual_delay > request.queue_size_at_submit:
        raise RuntimeError(
            "Executed more actions than existed when inference started: "
            f"{actual_delay} > {request.queue_size_at_submit}."
        )
    if actual_delay > request.conditioned_prefix_steps:
        raise LateRTCResponseError(
            request_id=request.request_id,
            actual_delay=actual_delay,
            conditioned_prefix_steps=request.conditioned_prefix_steps,
        )

    if not isinstance(payload.raw_actions, torch.Tensor):
        raise TypeError("RTC normalized actions must be a torch.Tensor.")
    raw_actions = payload.raw_actions.detach().cpu()
    if not isinstance(payload.actions, list) or not payload.actions:
        raise ValueError("Server returned an empty or invalid processed action chunk.")
    if not all(
        isinstance(action, TimedAction) and isinstance(action.get_action(), torch.Tensor)
        for action in payload.actions
    ):
        raise TypeError("Every processed action must be a TimedAction containing a tensor.")
    processed_actions = torch.stack([action.get_action().detach().cpu() for action in payload.actions])
    if raw_actions.ndim != 2 or processed_actions.ndim != 2:
        raise ValueError("RTC raw and processed actions must both have shape [T, A].")
    if raw_actions.shape != processed_actions.shape:
        raise ValueError(
            "RTC raw and processed chunks are not aligned: "
            f"{tuple(raw_actions.shape)} != {tuple(processed_actions.shape)}."
        )
    if raw_actions.shape[0] != request.expected_chunk_steps:
        raise ValueError(
            "Server returned an incomplete RTC chunk: "
            f"{raw_actions.shape[0]} != {request.expected_chunk_steps}."
        )
    expected_timesteps = list(range(request.timestep, request.timestep + len(payload.actions)))
    actual_timesteps = [action.get_timestep() for action in payload.actions]
    if actual_timesteps != expected_timesteps:
        raise ValueError(
            "RTC processed action timesteps are not consecutive: "
            f"expected {expected_timesteps[0]}..{expected_timesteps[-1]}."
        )
    if actual_delay >= raw_actions.shape[0]:
        raise LateRTCResponseError(
            request_id=request.request_id,
            actual_delay=actual_delay,
            conditioned_prefix_steps=request.conditioned_prefix_steps,
        )
    if not bool(torch.isfinite(raw_actions).all()) or not bool(torch.isfinite(processed_actions).all()):
        raise FloatingPointError("RTC response contains NaN or Inf.")
    if request.conditioned_prefix_steps:
        if request.action_prefix is None:
            raise RuntimeError("Nonzero RTC prefix has no request action_prefix.")
        returned_prefix = raw_actions[: request.conditioned_prefix_steps]
        expected_prefix = request.action_prefix.detach().cpu()
        if returned_prefix.shape != expected_prefix.shape or not torch.allclose(
            returned_prefix, expected_prefix, rtol=1e-5, atol=1e-6
        ):
            raise RuntimeError("Server did not hard-clamp the normalized RTC clean prefix.")

    received_at = time.perf_counter() if received_at is None else received_at
    return _CroppedActionChunk(
        raw_actions=raw_actions[actual_delay:].clone().contiguous(),
        processed_actions=processed_actions[actual_delay:].clone().contiguous(),
        actual_delay=actual_delay,
        latency_s=received_at - request.submitted_at,
    )


class _AlignedActionQueue:
    """Normalized/processed queue pair mutated only by the control thread."""

    def __init__(self) -> None:
        self.raw: deque[torch.Tensor] = deque()
        self.processed: deque[torch.Tensor] = deque()

    def __len__(self) -> int:
        if len(self.raw) != len(self.processed):
            raise RuntimeError("RTC raw and processed queues lost alignment.")
        return len(self.processed)

    def prefix(self, max_steps: int) -> tuple[torch.Tensor | None, int]:
        steps = min(max_steps, len(self))
        if steps == 0:
            return None, 0
        prefix = torch.stack(list(self.raw)[:steps]).detach().cpu().clone().contiguous()
        return prefix, steps

    def replace(self, raw_actions: torch.Tensor, processed_actions: torch.Tensor) -> None:
        if raw_actions.ndim != 2 or raw_actions.shape != processed_actions.shape:
            raise ValueError("Replacement RTC queues must be aligned [T, A] tensors.")
        self.raw = deque(row.clone() for row in raw_actions)
        self.processed = deque(row.clone() for row in processed_actions)

    def peek_processed(self) -> torch.Tensor | None:
        if not self:
            return None
        return self.processed[0].clone()

    def confirm_executed(self) -> None:
        if not self:
            raise RuntimeError("Cannot confirm an action from an empty RTC queue.")
        self.raw.popleft()
        self.processed.popleft()


class TrainedRTCInferenceClient:
    """One-in-flight remote client for ACP-positive learned RTC inference."""

    def __init__(self, cfg: TrainedRTCInferenceConfig, robot: Robot | None = None):
        self.cfg = cfg
        self.robot = make_robot_from_config(cfg.robot) if robot is None else robot
        self.channel = grpc.insecure_channel(
            cfg.server_address,
            grpc_channel_options(initial_backoff=f"{cfg.environment_dt:.4f}s"),
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)

        self._state_lock = threading.RLock()
        self._pending_event = threading.Event()
        self._stop_event = threading.Event()
        self._request_thread: threading.Thread | None = None
        self._pending_result: tuple[_InferenceRequest, Any, float] | None = None
        self._pending_error: tuple[_InferenceRequest, BaseException] | None = None
        self._request_sequence = 0
        self._executed_steps = 0
        self._queue = _AlignedActionQueue()
        self._started = False

        self.accepted_chunks = 0
        self.late_chunks = 0
        self.consecutive_late_chunks = 0
        self.max_observed_delay = 0
        self.queue_underruns = 0

    def start(self) -> None:
        if self._started:
            return
        if not self.robot.is_connected:
            self.robot.connect()

        self.stub.Ready(services_pb2.Empty(), timeout=self.cfg.rpc_timeout_s)
        policy_config = RemotePolicyConfig(
            policy_type="pi05",
            pretrained_name_or_path=self.cfg.pretrained_name_or_path,
            lerobot_features=map_robot_keys_to_lerobot_features(self.robot),
            actions_per_chunk=self.cfg.actions_per_chunk,
            device=self.cfg.policy_device,
            rename_map=self.cfg.rename_map,
            protocol_version=2,
            return_raw_actions=True,
            training_time_rtc=True,
            rtc_prefix_steps=self.cfg.rtc_prefix_steps,
            acp_positive_prompt=True,
            use_cfg=False,
            cfg_beta=1.0,
        )
        self.stub.SendPolicyInstructions(
            services_pb2.PolicySetup(data=pickle.dumps(policy_config)),
            timeout=self.cfg.setup_timeout_s,
        )
        self._started = True
        logging.info(
            "ACP+Training-Time RTC ready | server=%s | device=%s | d=%d | H=%d | CFG=off | beta=1.",
            self.cfg.server_address,
            self.cfg.policy_device,
            self.cfg.rtc_prefix_steps,
            self.cfg.actions_per_chunk,
        )

    def stop(self) -> None:
        self._stop_event.set()
        self.channel.close()
        with self._state_lock:
            request_thread = self._request_thread
        if request_thread is not None and request_thread is not threading.current_thread():
            request_thread.join(timeout=self.cfg.rpc_timeout_s + 1.0)
        if self.robot.is_connected:
            self.robot.disconnect()
        self._started = False

    def _has_active_request(self) -> bool:
        with self._state_lock:
            return (
                self._request_thread is not None
                or self._pending_result is not None
                or self._pending_error is not None
            )

    def _request_actions(self, request: _InferenceRequest) -> Any:
        raw_observation = dict(request.observation)
        raw_observation["task"] = self.cfg.task
        timed_observation = TimedObservation(
            timestamp=time.time(),
            timestep=request.timestep,
            observation=raw_observation,
            must_go=True,
            rtc_metadata=TrainingTimeRTCMetadata(
                request_id=request.request_id,
                action_prefix=request.action_prefix,
                prefix_steps=request.conditioned_prefix_steps,
            ),
        )
        observation_iterator = send_bytes_in_chunks(
            pickle.dumps(timed_observation),
            services_pb2.Observation,
            log_prefix="[TRC CLIENT] Observation",
            silent=True,
        )
        self.stub.SendObservations(observation_iterator, timeout=self.cfg.rpc_timeout_s)
        response = self.stub.GetActions(services_pb2.Empty(), timeout=self.cfg.rpc_timeout_s)
        if not response.data:
            raise TimeoutError("Policy server returned no action data.")
        return pickle.loads(response.data)  # nosec

    def _request_worker(self, request: _InferenceRequest) -> None:
        try:
            payload = self._request_actions(request)
            received_at = time.perf_counter()
            with self._state_lock:
                if not self._stop_event.is_set():
                    self._pending_result = (request, payload, received_at)
        except Exception as exc:
            with self._state_lock:
                if not self._stop_event.is_set():
                    self._pending_error = (request, exc)
        finally:
            with self._state_lock:
                if self._request_thread is threading.current_thread():
                    self._request_thread = None
                self._pending_event.set()

    def _start_request(self, observation: dict[str, Any]) -> bool:
        copied_observation = copy.deepcopy(observation)
        with self._state_lock:
            if self._stop_event.is_set() or self._has_active_request():
                return False
            action_prefix, prefix_steps = self._queue.prefix(self.cfg.rtc_prefix_steps)
            self._request_sequence += 1
            request = _InferenceRequest(
                request_id=f"trc-{self._request_sequence}",
                observation=copied_observation,
                timestep=self._executed_steps,
                action_prefix=action_prefix,
                conditioned_prefix_steps=prefix_steps,
                executed_steps_at_submit=self._executed_steps,
                queue_size_at_submit=len(self._queue),
                expected_chunk_steps=self.cfg.actions_per_chunk,
                submitted_at=time.perf_counter(),
            )
            request_thread = threading.Thread(
                target=self._request_worker,
                args=(request,),
                name=f"trc-request-{self._request_sequence}",
                daemon=True,
            )
            self._request_thread = request_thread
            self._pending_event.clear()
        try:
            request_thread.start()
        except Exception:
            with self._state_lock:
                if self._request_thread is request_thread:
                    self._request_thread = None
                self._pending_event.set()
            raise
        logging.debug(
            "Submitted RTC request %s at step=%d with d=%d and queue=%d.",
            request.request_id,
            request.timestep,
            request.conditioned_prefix_steps,
            request.queue_size_at_submit,
        )
        return True

    def _capture_and_start_request(self) -> bool:
        if self._has_active_request():
            return False
        return self._start_request(self.robot.get_observation())

    def _apply_pending_result(self) -> bool:
        with self._state_lock:
            pending_error = self._pending_error
            pending_result = self._pending_result
            self._pending_error = None
            self._pending_result = None
            self._pending_event.clear()
            executed_steps = self._executed_steps

        if pending_error is not None:
            request, error = pending_error
            raise RuntimeError(f"Remote RTC request {request.request_id} failed.") from error
        if pending_result is None:
            return False

        request, payload, received_at = pending_result
        try:
            cropped = _validate_and_crop_response(
                request,
                payload,
                executed_steps=executed_steps,
                received_at=received_at,
            )
        except LateRTCResponseError as error:
            self.late_chunks += 1
            self.consecutive_late_chunks += 1
            self.max_observed_delay = max(self.max_observed_delay, error.actual_delay)
            logging.warning(
                "Discarded unsafe RTC response %s: actual d=%d > conditioned d=%d. "
                "Continuing only with the already-safe local queue.",
                error.request_id,
                error.actual_delay,
                error.conditioned_prefix_steps,
            )
            if self.consecutive_late_chunks >= self.cfg.max_consecutive_late_chunks:
                raise RuntimeError(
                    "Too many consecutive RTC responses exceeded the conditioned prefix. "
                    "Increase --rtc-prefix-steps (within trained D), reduce latency, or reduce fps."
                ) from error
            return False

        with self._state_lock:
            if self._executed_steps != executed_steps:
                raise RuntimeError("Control step changed while installing an RTC chunk.")
            self._queue.replace(cropped.raw_actions, cropped.processed_actions)
            queue_size = len(self._queue)
        self.accepted_chunks += 1
        self.consecutive_late_chunks = 0
        self.max_observed_delay = max(self.max_observed_delay, cropped.actual_delay)
        logging.info(
            "Installed RTC chunk %s | actual d=%d | conditioned d=%d | latency=%.1f ms | remaining=%d.",
            request.request_id,
            cropped.actual_delay,
            request.conditioned_prefix_steps,
            cropped.latency_s * 1000,
            queue_size,
        )
        return True

    def _wait_until_actions(self, *, count_underrun: bool = True) -> None:
        deadline = time.perf_counter() + self.cfg.queue_wait_timeout_s
        logged_underrun = False
        while not self._stop_event.is_set():
            self._apply_pending_result()
            with self._state_lock:
                if len(self._queue):
                    return
            if count_underrun and not logged_underrun:
                self.queue_underruns += 1
                logging.warning(
                    "RTC action queue is empty; holding the robot while waiting for a fresh chunk."
                )
                logged_underrun = True
            self._capture_and_start_request()
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError("RTC action queue did not recover before queue_wait_timeout_s.")
            self._pending_event.wait(timeout=min(remaining, 0.05))
        raise RuntimeError("RTC inference client stopped while waiting for actions.")

    def _should_refill(self) -> bool:
        with self._state_lock:
            queue_size = len(self._queue)
        return (
            queue_size / self.cfg.actions_per_chunk <= self.cfg.refill_threshold
            and not self._has_active_request()
        )

    def _peek_action_dict(self) -> dict[str, float]:
        with self._state_lock:
            action_tensor = self._queue.peek_processed()
        if action_tensor is None:
            raise RuntimeError("RTC action queue is empty.")
        if action_tensor.numel() != len(self.robot.action_features):
            raise ValueError(
                "Policy action dimension does not match robot action features: "
                f"{action_tensor.numel()} != {len(self.robot.action_features)}."
            )
        return {key: action_tensor[index].item() for index, key in enumerate(self.robot.action_features)}

    def _confirm_action_executed(self) -> None:
        with self._state_lock:
            self._queue.confirm_executed()
            self._executed_steps += 1

    def run(self) -> None:
        if not self._started:
            raise RuntimeError("Call start() before run().")

        self._capture_and_start_request()
        # The bootstrap request starts from an intentionally empty queue; it is
        # startup latency, not a runtime underrun.
        self._wait_until_actions(count_underrun=False)
        control_started_at = time.perf_counter()
        last_status_at = control_started_at

        while not self._stop_event.is_set():
            loop_started_at = time.perf_counter()
            if (
                self.cfg.duration_s is not None
                and loop_started_at - control_started_at >= self.cfg.duration_s
            ):
                break

            self._apply_pending_result()
            with self._state_lock:
                queue_empty = len(self._queue) == 0
            if queue_empty:
                self._wait_until_actions()
                loop_started_at = time.perf_counter()

            action = self._peek_action_dict()
            sent_action = self.robot.send_action(action)
            _validate_sent_action(action, sent_action)
            # d counts only commands that the robot API accepted.
            self._confirm_action_executed()

            # Install a response that arrived during send_action before taking a
            # new observation/prefix snapshot, then refill from post-send state.
            self._apply_pending_result()
            if self._should_refill():
                self._capture_and_start_request()

            now = time.perf_counter()
            if now - last_status_at >= self.cfg.status_interval_s:
                with self._state_lock:
                    queue_size = len(self._queue)
                    executed_steps = self._executed_steps
                logging.info(
                    "TRC status | elapsed=%.1fs | steps=%d | queue=%d | chunks=%d | late=%d | max_d=%d.",
                    now - control_started_at,
                    executed_steps,
                    queue_size,
                    self.accepted_chunks,
                    self.late_chunks,
                    self.max_observed_delay,
                )
                last_status_at = now

            precise_sleep(max(0.0, self.cfg.environment_dt - (time.perf_counter() - loop_started_at)))

        logging.info(
            "TRC inference finished | steps=%d | accepted_chunks=%d | "
            "late_chunks=%d | underruns=%d | max_d=%d.",
            self._executed_steps,
            self.accepted_chunks,
            self.late_chunks,
            self.queue_underruns,
            self.max_observed_delay,
        )


@parser.wrap()
def infer_trc(cfg: TrainedRTCInferenceConfig) -> None:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    client = TrainedRTCInferenceClient(cfg)
    try:
        client.start()
        client.run()
    except KeyboardInterrupt:
        logging.info("Stopping ACP+Training-Time RTC inference.")
    finally:
        client.stop()


def main() -> None:
    register_third_party_plugins()
    infer_trc()


if __name__ == "__main__":
    main()
