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
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from queue import Empty, Full, Queue
from typing import TYPE_CHECKING, Any

import torch

from lerobot.async_inference.rtc_trace import create_rtc_trace, tensor_metadata
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

    # Lossless observation transport. ``none`` preserves the historical raw
    # pickle bytes; ``auto`` negotiates zlib and falls back safely; ``zlib``
    # requires a server that advertises support.
    observation_compression: str = "none"
    observation_zlib_level: int = 1
    observation_compression_min_bytes: int = 256 * 1024
    observation_compression_min_savings_ratio: float = 0.05

    # Estimated RTC values are used inside denoising. The response is still
    # cropped with the number of actions actually sent while the RPC ran.
    rtc_enable: bool = False
    rtc_inference_delay: int = 20
    rtc_execution_horizon: int = 25
    rtc_max_guidance_weight: float = 10.0
    rtc_prefix_attention_schedule: RTCAttentionSchedule = RTCAttentionSchedule.LINEAR
    rtc_cfg_beta: float = 1.0
    rtc_trace_enabled: bool = True
    rtc_trace_output_dir: str = "logs/rtc_trace"

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
        if self.observation_compression not in {"none", "auto", "zlib"}:
            raise ValueError("`remote_policy.observation_compression` must be one of: none, auto, zlib.")
        if (
            not isinstance(self.observation_zlib_level, int)
            or isinstance(self.observation_zlib_level, bool)
            or not 0 <= self.observation_zlib_level <= 9
        ):
            raise ValueError("`remote_policy.observation_zlib_level` must be an integer from 0 to 9.")
        if (
            not isinstance(self.observation_compression_min_bytes, int)
            or isinstance(self.observation_compression_min_bytes, bool)
            or self.observation_compression_min_bytes < 0
        ):
            raise ValueError(
                "`remote_policy.observation_compression_min_bytes` must be a non-negative integer."
            )
        if not 0 <= self.observation_compression_min_savings_ratio < 1:
            raise ValueError("`remote_policy.observation_compression_min_savings_ratio` must be in [0, 1).")

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
            if self.rtc_trace_enabled and not self.rtc_trace_output_dir.strip():
                raise ValueError("`remote_policy.rtc_trace_output_dir` must be non-empty.")
            if self.obs_queue_timeout_s == 0:
                raise ValueError("Remote RTC requires `remote_policy.obs_queue_timeout_s` to be positive.")
            if self.policy_type != "pi05":
                raise ValueError("Remote RTC-CFG currently supports only `policy_type=pi05`.")
            if self.rtc_inference_delay >= self.rtc_execution_horizon:
                raise ValueError(
                    "`remote_policy.rtc_inference_delay` must be smaller than "
                    "`remote_policy.rtc_execution_horizon`."
                )
            if self.rtc_execution_horizon > self.actions_per_chunk:
                raise ValueError("`remote_policy.rtc_execution_horizon` cannot exceed `actions_per_chunk`.")


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
    submitted_at_wall_s: float
    submitted_at_monotonic_s: float
    observation_snapshot_ms: float


@dataclass(frozen=True)
class _SendObservationMetrics:
    """Client-side request serialization and upload timings."""

    serialize_ms: float
    compression_ms: float
    uncompressed_payload_bytes: int
    payload_bytes: int
    chunk_count: int
    compression_ratio: float
    codec_requested: str
    codec_selected: str
    codec_used: str
    compression_skipped_reason: str | None
    rpc_ms: float
    total_ms: float


@dataclass(frozen=True)
class _ReceiveActionsMetrics:
    """Client-side response download and deserialization timings."""

    rpc_ms: float
    deserialize_ms: float
    payload_bytes: int
    poll_count: int
    total_ms: float


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
        self._active_rpc_stage: str | None = None
        self._episode_epoch = 0
        self._request_sequence = 0
        self._session_id = uuid.uuid4().hex
        self._executed_steps = 0
        self._awaiting_execution_confirmation = False
        self._last_get_action_metrics: dict[str, Any] | None = None
        self._started = False
        self._server_observation_codecs: set[str] = {"none"}
        self._server_observation_wire_version = 0
        self._observation_wire_version = 0
        self._observation_codec = "none"
        self._observation_compression_fallback_reason: str | None = None
        self._rtc_trace = (
            create_rtc_trace(role="client", output_dir=cfg.rtc_trace_output_dir)
            if cfg.rtc_enable and cfg.rtc_trace_enabled
            else None
        )
        self._trace_event(
            "client_created",
            session_id=self._session_id,
            server_address=cfg.server_address,
            fps=round(1 / self.environment_dt, 6),
            trace_path=str(self._rtc_trace.path) if self._rtc_trace is not None else None,
        )

    @property
    def policy_id(self) -> str:
        return self.cfg.pretrained_name_or_path or self.cfg.policy_type or "remote_policy"

    def _trace_event(self, event: str, **fields: Any) -> None:
        trace = getattr(self, "_rtc_trace", None)
        if trace is not None:
            trace.record(event, **fields)

    def record_control_loop_metrics(self, *, timestep: int, **metrics: Any) -> None:
        """Append one low-overhead control-loop sample to the async RTC trace.

        This method deliberately performs no filesystem I/O. The trace's
        bounded writer queue keeps a slow or unavailable disk out of the 30 Hz
        robot-control path.
        """
        with self._state_lock:
            queue_size = self._safe_queue_size()
            executed_steps = self._executed_steps
            episode_epoch = self._episode_epoch
            active_request_id = self._active_rpc_request_id or self._in_flight_request_id
            active_rpc_stage = self._active_rpc_stage
            has_pending_result = self._pending_result is not None
            last_get_action_metrics = self._last_get_action_metrics

        remote_get_action_fields: dict[str, Any] = {}
        if last_get_action_metrics is not None and last_get_action_metrics.get("timestep") == timestep:
            remote_get_action_fields = {
                "remote_get_action_ms": last_get_action_metrics.get("total_ms"),
                "remote_queue_wait_ms": last_get_action_metrics.get("queue_wait_ms"),
                "queue_size_after_pop": last_get_action_metrics.get("queue_size_after_pop"),
            }
        self._trace_event(
            "control_loop_step",
            session_id=self._session_id,
            episode_epoch=episode_epoch,
            observation_timestep=timestep,
            queue_size=queue_size,
            executed_steps=executed_steps,
            active_request_id=active_request_id,
            active_rpc_stage=active_rpc_stage,
            has_pending_result=has_pending_result,
            **remote_get_action_fields,
            **metrics,
        )

    def _safe_queue_size(self) -> int | None:
        """Best-effort queue snapshot used only for diagnostics."""
        try:
            return self._queue_size()
        except Exception:
            return None

    @staticmethod
    def _exception_trace_fields(error: BaseException) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        code_fn = getattr(error, "code", None)
        if callable(code_fn):
            with suppress(Exception):
                fields["grpc_code"] = str(code_fn())
        details_fn = getattr(error, "details", None)
        if callable(details_fn):
            with suppress(Exception):
                fields["grpc_details"] = details_fn()
        return fields

    @staticmethod
    def _metadata_items(metadata: Any) -> dict[str, str]:
        """Normalize gRPC metadata while remaining friendly to test doubles."""

        result: dict[str, str] = {}
        for item in metadata or ():
            try:
                key, value = item
            except (TypeError, ValueError):
                key = getattr(item, "key", None)
                value = getattr(item, "value", None)
            if key is not None and value is not None:
                result[str(key).lower()] = str(value)
        return result

    def _ready_and_negotiate_observation_codec(self) -> None:
        """Run Ready and select a codec without breaking old policy servers."""

        from lerobot.transport.utils import (
            OBSERVATION_CODEC_NONE,
            OBSERVATION_CODEC_ZLIB,
            OBSERVATION_CODECS_METADATA_KEY,
            OBSERVATION_PAYLOAD_VERSION,
            OBSERVATION_WIRE_VERSION_METADATA_KEY,
        )

        ready_rpc = self.stub.Ready
        with_call = getattr(ready_rpc, "with_call", None)
        if callable(with_call):
            _, call = with_call(self.services_pb2.Empty())
            trailing_metadata = call.trailing_metadata()
        else:
            # Simple stubs and old test doubles expose only __call__.  An old
            # real server also advertises no metadata and therefore behaves the
            # same as this conservative fallback.
            ready_rpc(self.services_pb2.Empty())
            trailing_metadata = ()

        metadata = self._metadata_items(trailing_metadata)
        advertised = {
            codec.strip().lower()
            for codec in metadata.get(OBSERVATION_CODECS_METADATA_KEY, OBSERVATION_CODEC_NONE).split(",")
            if codec.strip()
        }
        advertised.add(OBSERVATION_CODEC_NONE)
        try:
            wire_version = int(metadata.get(OBSERVATION_WIRE_VERSION_METADATA_KEY, "0"))
        except ValueError:
            wire_version = 0

        requested = self.cfg.observation_compression
        zlib_advertised = OBSERVATION_CODEC_ZLIB in advertised
        zlib_available = zlib_advertised and wire_version == OBSERVATION_PAYLOAD_VERSION
        fallback_reason = None
        if requested == OBSERVATION_CODEC_NONE:
            selected = OBSERVATION_CODEC_NONE
        elif requested == "auto":
            selected = OBSERVATION_CODEC_ZLIB if zlib_available else OBSERVATION_CODEC_NONE
            if selected == OBSERVATION_CODEC_NONE:
                fallback_reason = (
                    "incompatible_observation_wire_version"
                    if zlib_advertised
                    else "server_did_not_advertise_zlib"
                )
        elif not zlib_available:
            raise RuntimeError(
                "`remote_policy.observation_compression=zlib` requires a policy server that "
                f"advertises observation wire version {OBSERVATION_PAYLOAD_VERSION} and the "
                "zlib codec. Use `auto` to "
                "fall back when connecting to an older server."
            )
        else:
            selected = OBSERVATION_CODEC_ZLIB

        self._server_observation_codecs = advertised
        self._server_observation_wire_version = wire_version
        self._observation_wire_version = (
            OBSERVATION_PAYLOAD_VERSION if selected == OBSERVATION_CODEC_ZLIB else 0
        )
        self._observation_codec = selected
        self._observation_compression_fallback_reason = fallback_reason
        self._trace_event(
            "observation_transport_negotiated",
            session_id=self._session_id,
            observation_codec_requested=requested,
            observation_codec_advertised=sorted(advertised),
            observation_codec_selected=selected,
            observation_wire_version=self._observation_wire_version,
            server_observation_wire_version=wire_version,
            observation_compression_fallback_reason=fallback_reason,
        )

    def start(self) -> None:
        """Negotiate the payload format and start the persistent RPC worker."""
        if self._started:
            return

        from lerobot.async_inference.helpers import RemotePolicyConfig, map_robot_keys_to_lerobot_features

        setup_start = time.perf_counter()
        self._trace_event(
            "client_starting",
            session_id=self._session_id,
            server_address=self.cfg.server_address,
            protocol_version=2 if self.cfg.rtc_enable else 1,
            policy_type=self.cfg.policy_type,
            policy_checkpoint=self.cfg.pretrained_name_or_path,
            policy_device=self.cfg.policy_device,
            client_device=self.cfg.client_device,
            actions_per_chunk=self.cfg.actions_per_chunk,
            chunk_size_threshold=self.cfg.chunk_size_threshold,
            rpc_timeout_s=self.cfg.obs_queue_timeout_s,
            observation_codec_requested=self.cfg.observation_compression,
            observation_zlib_level=self.cfg.observation_zlib_level,
            observation_compression_min_bytes=self.cfg.observation_compression_min_bytes,
            observation_compression_min_savings_ratio=(self.cfg.observation_compression_min_savings_ratio),
            rtc_enabled=self.cfg.rtc_enable,
            cfg_beta=self.cfg.rtc_cfg_beta,
            inference_delay=self.cfg.rtc_inference_delay,
            execution_horizon=self.cfg.rtc_execution_horizon,
            max_guidance_weight=self.cfg.rtc_max_guidance_weight,
            prefix_attention_schedule=self.cfg.rtc_prefix_attention_schedule,
        )
        try:
            self._ready_and_negotiate_observation_codec()
            policy_config = RemotePolicyConfig(
                policy_type=self.cfg.policy_type,
                pretrained_name_or_path=self.cfg.pretrained_name_or_path,
                lerobot_features=map_robot_keys_to_lerobot_features(self.robot),
                actions_per_chunk=self.cfg.actions_per_chunk,
                device=self.cfg.policy_device,
                rename_map=self.cfg.rename_map,
                protocol_version=2 if self.cfg.rtc_enable else 1,
                return_raw_actions=self.cfg.rtc_enable,
                rtc_enabled=self.cfg.rtc_enable,
                rtc_inference_delay=self.cfg.rtc_inference_delay if self.cfg.rtc_enable else None,
                rtc_execution_horizon=(self.cfg.rtc_execution_horizon if self.cfg.rtc_enable else None),
                rtc_max_guidance_weight=(self.cfg.rtc_max_guidance_weight if self.cfg.rtc_enable else None),
                rtc_prefix_attention_schedule=(
                    self.cfg.rtc_prefix_attention_schedule if self.cfg.rtc_enable else None
                ),
                rtc_cfg_beta=self.cfg.rtc_cfg_beta if self.cfg.rtc_enable else None,
                observation_wire_version=self._observation_wire_version,
                observation_codec=self._observation_codec,
            )
            self.stub.SendPolicyInstructions(self.services_pb2.PolicySetup(data=pickle.dumps(policy_config)))
        except Exception as exc:
            self._trace_event(
                "client_start_failed",
                session_id=self._session_id,
                setup_ms=(time.perf_counter() - setup_start) * 1000,
                **self._exception_trace_fields(exc),
            )
            raise

        self._worker_thread = threading.Thread(
            target=self._chunk_worker,
            name="remote-policy-chunk-worker",
            daemon=True,
        )
        self._worker_thread.start()
        self._started = True
        self._trace_event(
            "client_started",
            session_id=self._session_id,
            setup_ms=(time.perf_counter() - setup_start) * 1000,
            observation_codec_requested=self.cfg.observation_compression,
            observation_codec_selected=self._observation_codec,
            observation_wire_version=self._observation_wire_version,
            server_observation_wire_version=self._server_observation_wire_version,
        )
        logging.info(
            "Remote policy recording client connected to %s (RTC=%s, d=%d, H=%d, trace=%s).",
            self.cfg.server_address,
            self.cfg.rtc_enable,
            self.cfg.rtc_inference_delay,
            self.cfg.rtc_execution_horizon,
            self._rtc_trace.path if self._rtc_trace is not None else "disabled",
        )

    def stop(self) -> None:
        """Stop the worker and invalidate any response that is still in flight."""
        with self._state_lock:
            queue_size_before = self._safe_queue_size()
            active_request_id = self._active_rpc_request_id or self._in_flight_request_id
            active_rpc_stage = self._active_rpc_stage
            executed_steps = self._executed_steps
            had_pending_result = self._pending_result is not None
            awaiting_execution_confirmation = self._awaiting_execution_confirmation
        self._trace_event(
            "client_stopping",
            session_id=self._session_id,
            episode_epoch=self._episode_epoch,
            started=self._started,
            queue_size_before=queue_size_before,
            executed_steps=executed_steps,
            cancelled_request_id=active_request_id,
            active_rpc_stage=active_rpc_stage,
            had_pending_result=had_pending_result,
            awaiting_execution_confirmation=awaiting_execution_confirmation,
        )
        if not self._started:
            self.channel.close()
            trace = getattr(self, "_rtc_trace", None)
            if trace is not None:
                trace.close()
            return

        with self._state_lock:
            self._episode_epoch += 1
            self._in_flight_request_id = None
            self._pending_result = None
            self._worker_error = None
            active_rpc = self._active_rpc_future
            self._active_rpc_future = None
            self._active_rpc_request_id = None
            self._active_rpc_stage = None
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
                self._trace_event("worker_join_timeout", session_id=self._session_id)
        self._started = False
        self._trace_event(
            "client_stopped",
            session_id=self._session_id,
            worker_alive=self._worker_thread is not None and self._worker_thread.is_alive(),
        )
        trace = getattr(self, "_rtc_trace", None)
        if trace is not None:
            trace.close()

    def reset(self) -> None:
        """Start a new episode and make all older worker responses stale."""
        with self._state_lock:
            old_epoch = self._episode_epoch
            queue_size_before = self._safe_queue_size()
            executed_steps = self._executed_steps
            had_pending_result = self._pending_result is not None
            awaiting_execution_confirmation = self._awaiting_execution_confirmation
            cancelled_request_id = self._active_rpc_request_id or self._in_flight_request_id
            active_rpc_stage = self._active_rpc_stage
            self._episode_epoch += 1
            self._in_flight_request_id = None
            self._pending_result = None
            self._worker_error = None
            active_rpc = self._active_rpc_future
            self._active_rpc_future = None
            self._active_rpc_request_id = None
            self._active_rpc_stage = None
            self._executed_steps = 0
            self._awaiting_execution_confirmation = False
            self._last_get_action_metrics = None
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

        self._trace_event(
            "episode_reset",
            session_id=self._session_id,
            old_epoch=old_epoch,
            new_epoch=self._episode_epoch,
            queue_size_before=queue_size_before,
            executed_steps_before=executed_steps,
            cancelled_request_id=cancelled_request_id,
            active_rpc_stage=active_rpc_stage,
            cancellation_requested=active_rpc is not None,
            had_pending_result=had_pending_result,
            awaiting_execution_confirmation=awaiting_execution_confirmation,
        )

        # Reset server deduplication and policy episode state. An older active
        # RPC may still finish, but its epoch/request id prevents installation.
        if self._started:
            try:
                self.stub.Ready(self.services_pb2.Empty())
            except Exception as exc:
                self._trace_event(
                    "server_reset_failed",
                    session_id=self._session_id,
                    episode_epoch=self._episode_epoch,
                    **self._exception_trace_fields(exc),
                )
                raise

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
            request_id = f"{self._session_id}:{self._episode_epoch}:{self._request_sequence}"
            submitted_at_wall_s = time.time()
            submitted_at_monotonic_s = time.perf_counter()
            snapshot_started = time.perf_counter()
            observation_snapshot = copy.deepcopy(observation)
            observation_snapshot_ms = (time.perf_counter() - snapshot_started) * 1000
            request = _InferenceRequest(
                request_id=request_id,
                episode_epoch=self._episode_epoch,
                observation=observation_snapshot,
                task=task,
                timestep=max(timestep, 0),
                left_over=left_over,
                executed_steps_at_submit=self._executed_steps,
                queue_size_at_submit=queue_size,
                submitted_at_wall_s=submitted_at_wall_s,
                submitted_at_monotonic_s=submitted_at_monotonic_s,
                observation_snapshot_ms=observation_snapshot_ms,
            )
            self._in_flight_request_id = request_id
            self._worker_error = None
            self._pending_event.clear()
            self._trace_event(
                "request_submitted",
                session_id=self._session_id,
                request_id=request.request_id,
                episode_epoch=request.episode_epoch,
                observation_timestep=request.timestep,
                submitted_at_wall_s=request.submitted_at_wall_s,
                queue_size_at_submit=request.queue_size_at_submit,
                leftover_steps=0 if request.left_over is None else int(request.left_over.shape[0]),
                leftover=tensor_metadata(request.left_over),
                executed_steps_at_submit=request.executed_steps_at_submit,
                observation_snapshot_ms=request.observation_snapshot_ms,
                estimated_inference_delay=self.cfg.rtc_inference_delay,
                execution_horizon=self.cfg.rtc_execution_horizon,
            )
            try:
                self._request_queue.put_nowait(request)
            except Full as exc:
                self._in_flight_request_id = None
                self._trace_event(
                    "request_submission_failed",
                    session_id=self._session_id,
                    request_id=request.request_id,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
                raise RuntimeError(
                    "Remote policy worker queue unexpectedly contains another request."
                ) from exc
        return True

    def _run_cancellable_rpc(
        self,
        request: _InferenceRequest,
        rpc,
        argument,
        *,
        timeout: float | None,
        stage: str = "unknown",
    ):
        """Run one gRPC future that reset/stop can cancel without waiting for its deadline."""
        rpc_start = time.perf_counter()
        try:
            rpc_future = rpc.future(argument, timeout=timeout)
        except Exception as exc:
            self._trace_event(
                "rpc_failed",
                session_id=self._session_id,
                request_id=request.request_id,
                episode_epoch=request.episode_epoch,
                stage=stage,
                phase="future_creation",
                elapsed_ms=(time.perf_counter() - rpc_start) * 1000,
                **self._exception_trace_fields(exc),
            )
            raise
        with self._state_lock:
            if (
                self._stop_event.is_set()
                or request.episode_epoch != self._episode_epoch
                or request.request_id != self._in_flight_request_id
            ):
                rpc_future.cancel()
                self._trace_event(
                    "rpc_invalidated",
                    session_id=self._session_id,
                    request_id=request.request_id,
                    request_epoch=request.episode_epoch,
                    current_epoch=self._episode_epoch,
                    stage=stage,
                    stopped=self._stop_event.is_set(),
                )
                raise RuntimeError(f"Remote policy request {request.request_id} was invalidated.")
            self._active_rpc_future = rpc_future
            self._active_rpc_request_id = request.request_id
            self._active_rpc_stage = stage
        self._trace_event(
            "rpc_started",
            session_id=self._session_id,
            request_id=request.request_id,
            episode_epoch=request.episode_epoch,
            stage=stage,
            timeout_s=timeout,
        )
        try:
            result = rpc_future.result()
            self._trace_event(
                "rpc_completed",
                session_id=self._session_id,
                request_id=request.request_id,
                episode_epoch=request.episode_epoch,
                stage=stage,
                elapsed_ms=(time.perf_counter() - rpc_start) * 1000,
            )
            return result
        except Exception as exc:
            self._trace_event(
                "rpc_failed",
                session_id=self._session_id,
                request_id=request.request_id,
                episode_epoch=request.episode_epoch,
                stage=stage,
                elapsed_ms=(time.perf_counter() - rpc_start) * 1000,
                **self._exception_trace_fields(exc),
            )
            raise
        finally:
            with self._state_lock:
                if self._active_rpc_future is rpc_future:
                    self._active_rpc_future = None
                    self._active_rpc_request_id = None
                    self._active_rpc_stage = None

    def _send_observation(self, request: _InferenceRequest) -> _SendObservationMetrics:
        from lerobot.async_inference.helpers import RTCInferenceMetadata, TimedObservation
        from lerobot.transport.utils import CHUNK_SIZE, encode_observation_payload, send_bytes_in_chunks

        total_started = time.perf_counter()
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
        serialize_started = time.perf_counter()
        serialized_observation = pickle.dumps(timed_observation)
        serialize_ms = (time.perf_counter() - serialize_started) * 1000
        codec_selected = getattr(self, "_observation_codec", "none")
        encoded_observation = encode_observation_payload(
            serialized_observation,
            codec=codec_selected,
            zlib_level=self.cfg.observation_zlib_level,
            min_bytes=self.cfg.observation_compression_min_bytes,
            min_savings_ratio=self.cfg.observation_compression_min_savings_ratio,
            require_savings=self.cfg.observation_compression == "auto",
        )
        wire_observation = encoded_observation.data
        payload_bytes = encoded_observation.wire_bytes
        chunk_count = (payload_bytes + CHUNK_SIZE - 1) // CHUNK_SIZE
        self._trace_event(
            "observation_serialized",
            session_id=self._session_id,
            request_id=request.request_id,
            episode_epoch=request.episode_epoch,
            observation_timestep=request.timestep,
            observation_serialize_ms=serialize_ms,
            observation_compress_ms=encoded_observation.compression_ms,
            observation_pickle_bytes=encoded_observation.raw_bytes,
            request_payload_bytes=payload_bytes,
            request_chunk_count=chunk_count,
            observation_compression_ratio=encoded_observation.compression_ratio,
            observation_compression_savings_bytes=encoded_observation.raw_bytes - payload_bytes,
            observation_codec_requested=self.cfg.observation_compression,
            observation_codec_selected=codec_selected,
            observation_codec_used=encoded_observation.codec,
            observation_compression_skipped_reason=encoded_observation.skipped_reason,
        )
        observation_iterator = send_bytes_in_chunks(
            wire_observation,
            self.services_pb2.Observation,
            log_prefix="[RECORD_REMOTE_POLICY] Observation",
            silent=True,
        )
        rpc_started = time.perf_counter()
        self._run_cancellable_rpc(
            request,
            self.stub.SendObservations,
            observation_iterator,
            timeout=self.cfg.obs_queue_timeout_s if self.cfg.obs_queue_timeout_s > 0 else None,
            stage="send_observation",
        )
        rpc_ms = (time.perf_counter() - rpc_started) * 1000
        metrics = _SendObservationMetrics(
            serialize_ms=serialize_ms,
            compression_ms=encoded_observation.compression_ms,
            uncompressed_payload_bytes=encoded_observation.raw_bytes,
            payload_bytes=payload_bytes,
            chunk_count=chunk_count,
            compression_ratio=encoded_observation.compression_ratio,
            codec_requested=self.cfg.observation_compression,
            codec_selected=codec_selected,
            codec_used=encoded_observation.codec,
            compression_skipped_reason=encoded_observation.skipped_reason,
            rpc_ms=rpc_ms,
            total_ms=(time.perf_counter() - total_started) * 1000,
        )
        self._trace_event(
            "observation_sent",
            session_id=self._session_id,
            request_id=request.request_id,
            episode_epoch=request.episode_epoch,
            observation_timestep=request.timestep,
            observation_snapshot_ms=request.observation_snapshot_ms,
            observation_serialize_ms=metrics.serialize_ms,
            observation_compress_ms=metrics.compression_ms,
            observation_pickle_bytes=metrics.uncompressed_payload_bytes,
            request_payload_bytes=metrics.payload_bytes,
            request_chunk_count=metrics.chunk_count,
            observation_compression_ratio=metrics.compression_ratio,
            observation_compression_savings_bytes=(
                metrics.uncompressed_payload_bytes - metrics.payload_bytes
            ),
            observation_codec_requested=metrics.codec_requested,
            observation_codec_selected=metrics.codec_selected,
            observation_codec_used=metrics.codec_used,
            observation_compression_skipped_reason=metrics.compression_skipped_reason,
            observation_upload_rpc_ms=metrics.rpc_ms,
            send_total_ms=metrics.total_ms,
        )
        return metrics

    def _receive_actions(self, request: _InferenceRequest) -> tuple[Any, _ReceiveActionsMetrics]:
        total_started = time.perf_counter()
        deadline_t = time.perf_counter() + self.cfg.obs_queue_timeout_s
        rpc_ms = 0.0
        poll_count = 0
        while not self._stop_event.is_set():
            remaining = deadline_t - time.perf_counter()
            if self.cfg.obs_queue_timeout_s > 0 and remaining <= 0:
                raise TimeoutError("Timed out waiting for remote policy actions.")
            rpc_started = time.perf_counter()
            actions_chunk = self._run_cancellable_rpc(
                request,
                self.stub.GetActions,
                self.services_pb2.Empty(),
                timeout=remaining if self.cfg.obs_queue_timeout_s > 0 else None,
                stage="get_actions",
            )
            rpc_ms += (time.perf_counter() - rpc_started) * 1000
            poll_count += 1
            if actions_chunk.data:
                payload_bytes = len(actions_chunk.data)
                deserialize_started = time.perf_counter()
                response = pickle.loads(actions_chunk.data)  # nosec
                deserialize_ms = (time.perf_counter() - deserialize_started) * 1000
                metrics = _ReceiveActionsMetrics(
                    rpc_ms=rpc_ms,
                    deserialize_ms=deserialize_ms,
                    payload_bytes=payload_bytes,
                    poll_count=poll_count,
                    total_ms=(time.perf_counter() - total_started) * 1000,
                )
                self._trace_event(
                    "action_response_deserialized",
                    session_id=self._session_id,
                    request_id=request.request_id,
                    episode_epoch=request.episode_epoch,
                    observation_timestep=request.timestep,
                    get_actions_rpc_ms=metrics.rpc_ms,
                    response_deserialize_ms=metrics.deserialize_ms,
                    response_payload_bytes=metrics.payload_bytes,
                    get_actions_poll_count=metrics.poll_count,
                    receive_total_ms=metrics.total_ms,
                )
                return response, metrics

            if self.cfg.obs_queue_timeout_s == 0 or time.perf_counter() >= deadline_t:
                raise TimeoutError("Timed out waiting for remote policy actions.")
        raise RuntimeError("Remote policy client stopped while waiting for actions.")

    def _chunk_worker(self) -> None:
        self._trace_event("worker_started", session_id=self._session_id)
        try:
            while not self._stop_event.is_set():
                try:
                    request = self._request_queue.get(timeout=0.1)
                except Empty:
                    continue
                try:
                    if request is None:
                        return
                    send_metrics = self._send_observation(request)
                    response, receive_metrics = self._receive_actions(request)
                    response_request_id = getattr(response, "request_id", None)
                    raw_actions = getattr(response, "raw_actions", None)
                    response_steps = (
                        int(raw_actions.shape[0])
                        if isinstance(raw_actions, torch.Tensor) and raw_actions.ndim >= 1
                        else len(response)
                        if isinstance(response, list)
                        else None
                    )
                    with self._state_lock:
                        queue_size_at_receive = self._safe_queue_size()
                        executed_steps_at_receive = self._executed_steps
                        current_epoch = self._episode_epoch
                        response_is_current = (
                            not self._stop_event.is_set()
                            and request.episode_epoch == current_epoch
                            and request.request_id == self._in_flight_request_id
                        )
                        self._trace_event(
                            "response_received",
                            session_id=self._session_id,
                            request_id=request.request_id,
                            response_request_id=response_request_id,
                            episode_epoch=request.episode_epoch,
                            current_epoch=current_epoch,
                            observation_timestep=request.timestep,
                            observation_snapshot_ms=request.observation_snapshot_ms,
                            observation_serialize_ms=send_metrics.serialize_ms,
                            observation_compress_ms=send_metrics.compression_ms,
                            observation_pickle_bytes=send_metrics.uncompressed_payload_bytes,
                            request_payload_bytes=send_metrics.payload_bytes,
                            request_chunk_count=send_metrics.chunk_count,
                            observation_compression_ratio=send_metrics.compression_ratio,
                            observation_compression_savings_bytes=(
                                send_metrics.uncompressed_payload_bytes - send_metrics.payload_bytes
                            ),
                            observation_codec_requested=send_metrics.codec_requested,
                            observation_codec_selected=send_metrics.codec_selected,
                            observation_codec_used=send_metrics.codec_used,
                            observation_compression_skipped_reason=(send_metrics.compression_skipped_reason),
                            # Preserve the historical aggregate fields so old
                            # analysis scripts keep working. The explicitly
                            # named fields below contain pure RPC durations.
                            send_rpc_ms=send_metrics.total_ms,
                            send_total_ms=send_metrics.total_ms,
                            observation_upload_rpc_ms=send_metrics.rpc_ms,
                            receive_rpc_ms=receive_metrics.total_ms,
                            get_actions_rpc_ms=receive_metrics.rpc_ms,
                            response_deserialize_ms=receive_metrics.deserialize_ms,
                            response_payload_bytes=receive_metrics.payload_bytes,
                            get_actions_poll_count=receive_metrics.poll_count,
                            receive_total_ms=receive_metrics.total_ms,
                            latency_field_semantics={
                                "send_rpc_ms": (
                                    "legacy observation serialization + compression + upload RPC total"
                                ),
                                "receive_rpc_ms": "legacy GetActions RPC + response deserialization total",
                                "observation_upload_rpc_ms": "pure SendObservations RPC",
                                "get_actions_rpc_ms": "pure GetActions RPC, summed across polls",
                            },
                            end_to_end_ms=(time.perf_counter() - request.submitted_at_monotonic_s) * 1000,
                            response_steps=response_steps,
                            raw_actions=tensor_metadata(raw_actions),
                            queue_size_at_receive=queue_size_at_receive,
                            executed_steps_at_receive=executed_steps_at_receive,
                            accepted_for_install=response_is_current,
                        )
                        if response_is_current:
                            self._pending_result = (request, response)
                            self._in_flight_request_id = None
                            self._pending_event.set()
                    if not response_is_current:
                        self._trace_event(
                            "response_discarded",
                            session_id=self._session_id,
                            request_id=request.request_id,
                            reason="request_invalidated_before_pending_install",
                            request_epoch=request.episode_epoch,
                            current_epoch=current_epoch,
                        )
                except Exception as exc:
                    if request is not None:
                        with self._state_lock:
                            current_epoch = self._episode_epoch
                            request_is_current = (
                                request.episode_epoch == current_epoch
                                and request.request_id == self._in_flight_request_id
                            )
                            if request_is_current:
                                self._worker_error = (request, exc)
                                self._in_flight_request_id = None
                                self._pending_event.set()
                        self._trace_event(
                            "request_failed",
                            session_id=self._session_id,
                            request_id=request.request_id,
                            request_epoch=request.episode_epoch,
                            current_epoch=current_epoch,
                            request_still_current=request_is_current,
                            end_to_end_ms=(time.perf_counter() - request.submitted_at_monotonic_s) * 1000,
                            **self._exception_trace_fields(exc),
                        )
                finally:
                    self._request_queue.task_done()
        finally:
            self._trace_event("worker_stopped", session_id=self._session_id)

    def _apply_pending_result(self) -> bool:
        with self._state_lock:
            pending_request = self._pending_result[0] if self._pending_result is not None else None
        try:
            return self._apply_pending_result_impl()
        except Exception as exc:
            self._trace_event(
                "chunk_install_failed",
                session_id=self._session_id,
                request_id=pending_request.request_id if pending_request is not None else None,
                observation_timestep=(pending_request.timestep if pending_request is not None else None),
                episode_epoch=self._episode_epoch,
                queue_size=self._safe_queue_size(),
                executed_steps=self._executed_steps,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            raise

    def _apply_pending_result_impl(self) -> bool:
        with self._state_lock:
            pending = self._pending_result
            self._pending_result = None
            if self._worker_error is None:
                self._pending_event.clear()
        if pending is None:
            return False

        request, payload = pending
        if request.episode_epoch != self._episode_epoch:
            self._trace_event(
                "response_discarded",
                session_id=self._session_id,
                request_id=request.request_id,
                reason="stale_episode_before_install",
                request_epoch=request.episode_epoch,
                current_epoch=self._episode_epoch,
            )
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
                "Start the policy server with `acp_inference.enable=true` and use an RTC-enabled client."
            )
        if payload.request_id != request.request_id:
            raise RuntimeError(
                f"RTC response request_id mismatch: expected {request.request_id}, got {payload.request_id}."
            )
        if payload.observation_timestep != request.timestep:
            raise RuntimeError(
                "RTC response observation_timestep mismatch: "
                f"expected {request.timestep}, got {payload.observation_timestep}."
            )
        if not payload.rtc_enabled:
            raise RuntimeError("The policy server returned a chunk without RTC enabled.")
        if payload.inference_delay != self.cfg.rtc_inference_delay:
            raise RuntimeError(
                "RTC response inference_delay mismatch: "
                f"expected {self.cfg.rtc_inference_delay}, got {payload.inference_delay}."
            )
        if payload.execution_horizon != self.cfg.rtc_execution_horizon:
            raise RuntimeError(
                "RTC response execution_horizon mismatch: "
                f"expected {self.cfg.rtc_execution_horizon}, got {payload.execution_horizon}."
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
                self._trace_event(
                    "response_discarded",
                    session_id=self._session_id,
                    request_id=request.request_id,
                    reason="stale_episode_during_install",
                    request_epoch=request.episode_epoch,
                    current_epoch=self._episode_epoch,
                )
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
                self._trace_event(
                    "response_discarded",
                    session_id=self._session_id,
                    request_id=request.request_id,
                    reason="fully_stale_chunk",
                    observation_timestep=request.timestep,
                    queue_size_at_submit=request.queue_size_at_submit,
                    response_steps=int(raw_actions.shape[0]),
                    executed_steps_at_submit=request.executed_steps_at_submit,
                    executed_steps_now=self._executed_steps,
                    actual_delay=actual_delay,
                    end_to_end_ms=(time.perf_counter() - request.submitted_at_monotonic_s) * 1000,
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
            queue_size_after = self.action_queue.qsize()
            self._trace_event(
                "chunk_installed",
                session_id=self._session_id,
                request_id=request.request_id,
                episode_epoch=request.episode_epoch,
                observation_timestep=request.timestep,
                response_observation_timestep=payload.observation_timestep,
                queue_size_at_submit=request.queue_size_at_submit,
                leftover_steps_at_submit=(
                    0 if request.left_over is None else int(request.left_over.shape[0])
                ),
                response_steps=int(raw_actions.shape[0]),
                action_dim=int(raw_actions.shape[1]),
                estimated_inference_delay=self.cfg.rtc_inference_delay,
                server_inference_delay=payload.inference_delay,
                actual_delay=actual_delay,
                delay_delta=actual_delay - self.cfg.rtc_inference_delay,
                execution_horizon=self.cfg.rtc_execution_horizon,
                server_execution_horizon=payload.execution_horizon,
                discarded_prefix_steps=actual_delay,
                installed_steps=int(raw_actions.shape[0]) - actual_delay,
                queue_size_after_install=queue_size_after,
                raw_actions=tensor_metadata(raw_actions),
                processed_actions=tensor_metadata(processed_actions),
                end_to_end_ms=(time.perf_counter() - request.submitted_at_monotonic_s) * 1000,
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
            self._trace_event(
                "worker_error_discarded",
                session_id=self._session_id,
                request_id=request.request_id,
                reason="stale_episode",
                request_epoch=request.episode_epoch,
                current_epoch=self._episode_epoch,
                error_type=type(error).__name__,
                error_message=str(error),
            )
            return
        if queue_empty:
            self._trace_event(
                "request_failure_escalated",
                session_id=self._session_id,
                request_id=request.request_id,
                queue_size=0,
                error_type=type(error).__name__,
                error_message=str(error),
            )
            raise RuntimeError(f"Remote policy request {request.request_id} failed.") from error
        remaining_actions = self._queue_size()
        self._trace_event(
            "request_retry_scheduled",
            session_id=self._session_id,
            request_id=request.request_id,
            queue_size=remaining_actions,
            error_type=type(error).__name__,
            error_message=str(error),
        )
        logging.warning(
            "Remote policy request %s failed while %d queued actions remain; retrying at low water mark: %s",
            request.request_id,
            remaining_actions,
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

        get_action_started = time.perf_counter()
        queue_wait_ms = 0.0
        wait_timeout = max(self.cfg.obs_queue_timeout_s + 1.0, 1.0)
        deadline = time.perf_counter() + wait_timeout
        while True:
            self._apply_pending_result()
            self._raise_or_log_worker_error()
            self._submit_if_needed(observation=observation, task=task, timestep=timestep)

            action_tensor = self._pop_action()
            if action_tensor is not None:
                action = self._tensor_to_action(action_tensor)
                with self._state_lock:
                    self._last_get_action_metrics = {
                        "timestep": timestep,
                        "total_ms": (time.perf_counter() - get_action_started) * 1000,
                        "queue_wait_ms": queue_wait_ms,
                        "queue_size_after_pop": self._safe_queue_size(),
                    }
                return action

            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                with self._state_lock:
                    active_request_id = self._active_rpc_request_id or self._in_flight_request_id
                    active_rpc_stage = self._active_rpc_stage
                    queue_size = self._safe_queue_size()
                    has_pending_result = self._pending_result is not None
                    has_worker_error = self._worker_error is not None
                    executed_steps = self._executed_steps
                self._trace_event(
                    "queue_underrun",
                    session_id=self._session_id,
                    episode_epoch=self._episode_epoch,
                    observation_timestep=timestep,
                    waited_ms=wait_timeout * 1000,
                    queue_size=queue_size,
                    executed_steps=executed_steps,
                    active_request_id=active_request_id,
                    active_rpc_stage=active_rpc_stage,
                    has_pending_result=has_pending_result,
                    has_worker_error=has_worker_error,
                )
                raise TimeoutError(
                    "Remote RTC action queue underrun: no fresh action arrived before the safety timeout."
                )
            wait_started = time.perf_counter()
            self._pending_event.wait(timeout=min(remaining, 0.05))
            queue_wait_ms += (time.perf_counter() - wait_started) * 1000
