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

"""
Example:
```shell
python -m lerobot.async_inference.policy_server \
     --host=127.0.0.1 \
     --port=8080 \
     --fps=30 \
     --inference_latency=0.033 \
     --obs_queue_timeout=1
```
"""

import logging
import math
import pickle  # nosec
import threading
import time
from concurrent import futures
from contextlib import suppress
from dataclasses import asdict, replace
from pprint import pformat
from queue import Empty, Queue
from typing import Any

import draccus
import grpc
import torch

from lerobot.configs.types import RTCAttentionSchedule
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
)
from lerobot.rl.acp_tags import build_acp_tagged_task
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import (
    OBSERVATION_CODEC_NONE,
    OBSERVATION_CODEC_ZLIB,
    OBSERVATION_CODECS_METADATA_KEY,
    OBSERVATION_PAYLOAD_VERSION,
    OBSERVATION_WIRE_VERSION_METADATA_KEY,
    decode_observation_payload,
    receive_bytes_in_chunks,
)

from .configs import ACPInferenceConfig, PolicyServerConfig
from .constants import SUPPORTED_POLICIES
from .helpers import (
    FPSTracker,
    Observation,
    RemoteActionChunk,
    RemotePolicyConfig,
    RTCInferenceMetadata,
    TimedAction,
    TimedObservation,
    get_logger,
    observations_similar,
    raw_observation_to_observation,
)
from .rtc_trace import create_rtc_trace, tensor_metadata


class PolicyServer(services_pb2_grpc.AsyncInferenceServicer):
    prefix = "policy_server"
    logger = get_logger(prefix)

    def __init__(self, config: PolicyServerConfig):
        self.config = config
        self.shutdown_event = threading.Event()

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=config.fps)

        self.observation_queue = Queue(maxsize=1)

        # gRPC cancellation stops the client wait but cannot interrupt a CUDA
        # forward already in progress. Serialize forwards so a reset cannot run
        # an old and a new Pi0.5 request concurrently on the same model.
        self._inference_lock = threading.Lock()
        self._setup_lock = threading.Lock()
        self._generation_lock = threading.Lock()
        self._server_generation = 0

        self._predicted_timesteps_lock = threading.Lock()
        self._predicted_timesteps = set()

        self.last_processed_obs = None

        # Attributes will be set by SendPolicyInstructions
        self.device = None
        self.policy_type = None
        self.pretrained_name_or_path = None
        self.lerobot_features = None
        self.actions_per_chunk = None
        self.policy = None
        self.preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None
        self.postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None
        self.acp_profiler: Any | None = None
        self._pending_profiler_shutdowns: list[Any] = []
        self._acp_profile_disabled = False
        self._acp_profile_last_dropped = 0
        self._acp_profile_last_error: str | None = None
        self._acp_chunk_index = 0
        self._stop_lock = threading.Lock()
        self._stopped = False
        self._client_protocol_version = 1
        self._client_return_raw_actions = False
        self._client_rtc_enabled = False
        self._client_observation_wire_version = 0
        self._client_observation_codec = OBSERVATION_CODEC_NONE
        # A successful policy setup installs one effective session config.
        # ``None`` keeps direct/local server usage on the static CLI config.
        self._effective_acp_inference: ACPInferenceConfig | None = None
        self._acp_parameter_source = "server_static"
        rtc_trace = config.acp_inference.rtc
        self._rtc_trace = (
            create_rtc_trace(role="server", output_dir=rtc_trace.trace_output_dir)
            if rtc_trace.enabled and rtc_trace.trace_enabled
            else None
        )
        self._trace_event(
            "server_created",
            host=config.host,
            port=config.port,
            fps=config.fps,
            observation_queue_timeout_s=config.obs_queue_timeout,
            configured_inference_latency_s=config.inference_latency,
            rtc_enabled=self.rtc_enabled,
            cfg_beta=config.acp_inference.cfg_beta,
            inference_delay=rtc_trace.inference_delay,
            execution_horizon=rtc_trace.execution_horizon,
            max_guidance_weight=rtc_trace.max_guidance_weight,
            prefix_attention_schedule=rtc_trace.prefix_attention_schedule,
            trace_path=str(self._rtc_trace.path) if self._rtc_trace is not None else None,
        )

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    @property
    def policy_image_features(self):
        return self.policy.config.image_features

    @property
    def acp_inference(self) -> ACPInferenceConfig:
        """Return the immutable-for-this-session effective ACP configuration."""

        return self._effective_acp_inference or self.config.acp_inference

    @property
    def batched_cfg_enabled(self) -> bool:
        acp = self.acp_inference
        return acp.enable and acp.use_cfg and acp.batched_cfg

    @property
    def rtc_enabled(self) -> bool:
        return self.batched_cfg_enabled and self.acp_inference.rtc.enabled

    @property
    def return_remote_action_chunk(self) -> bool:
        return (
            self.batched_cfg_enabled
            and self._client_protocol_version >= 2
            and self._client_return_raw_actions
        )

    def _cuda_timing_enabled(self, device: torch.device) -> bool:
        profiling_active = self.acp_profiler is not None and not self._acp_profile_disabled
        return device.type == "cuda" and (profiling_active or self._rtc_trace is not None)

    def _trace_event(self, event: str, **fields: Any) -> None:
        trace = getattr(self, "_rtc_trace", None)
        if trace is not None:
            trace.record(event, **fields)

    @staticmethod
    def _request_id(obs: TimedObservation | None) -> str | None:
        if obs is None:
            return None
        metadata = getattr(obs, "rtc_metadata", None)
        if isinstance(metadata, RTCInferenceMetadata):
            return metadata.request_id
        return None

    @staticmethod
    def _normalize_client_schedule(value: Any) -> RTCAttentionSchedule:
        if isinstance(value, RTCAttentionSchedule):
            return value
        if isinstance(value, str):
            try:
                return RTCAttentionSchedule(value.upper())
            except ValueError as exc:
                choices = ", ".join(schedule.value for schedule in RTCAttentionSchedule)
                raise ValueError(f"Client RTC attention schedule must be one of: {choices}.") from exc
        raise TypeError("Client `rtc_prefix_attention_schedule` must be an RTCAttentionSchedule or string.")

    def _resolve_client_acp_contract(
        self,
        policy_specs: RemotePolicyConfig,
    ) -> tuple[ACPInferenceConfig, str]:
        """Build a validated per-session ACP config without mutating server state.

        An enable-only server config is client-managed.  The legacy fully
        specified server config remains static and keeps its equality checks.
        Profile/trace paths always remain server-owned.
        """

        server_acp = self.config.acp_inference
        client_rtc_enabled = getattr(policy_specs, "rtc_enabled", False)
        if not isinstance(client_rtc_enabled, bool):
            raise TypeError("Client `rtc_enabled` must be a bool.")

        if client_rtc_enabled and not server_acp.enable:
            raise ValueError(
                "The client requested RTC-CFG, but this server does not authorize ACP. "
                "Start it with `--acp_inference.enable=true`."
            )

        if server_acp.client_managed:
            if not client_rtc_enabled:
                return (
                    replace(
                        server_acp,
                        use_cfg=False,
                        batched_cfg=False,
                        rtc=replace(server_acp.rtc, enabled=False),
                    ),
                    "client_managed_disabled",
                )

            if policy_specs.policy_type != "pi05":
                raise ValueError("Client-managed RTC-CFG currently supports only Pi0.5.")
            protocol_version = getattr(policy_specs, "protocol_version", 1)
            return_raw_actions = getattr(policy_specs, "return_raw_actions", False)
            if protocol_version < 2 or return_raw_actions is not True:
                raise ValueError(
                    "Client-managed RTC-CFG requires protocol_version>=2 and return_raw_actions=true."
                )

            delay = getattr(policy_specs, "rtc_inference_delay", None)
            horizon = getattr(policy_specs, "rtc_execution_horizon", None)
            weight = getattr(policy_specs, "rtc_max_guidance_weight", None)
            beta = getattr(policy_specs, "rtc_cfg_beta", None)
            schedule = self._normalize_client_schedule(
                getattr(policy_specs, "rtc_prefix_attention_schedule", None)
            )
            if not isinstance(delay, int) or isinstance(delay, bool) or delay < 0:
                raise ValueError("Client `rtc_inference_delay` must be a non-negative integer.")
            if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= delay:
                raise ValueError(
                    "Client `rtc_execution_horizon` must be an integer greater than inference delay."
                )
            actions_per_chunk = getattr(policy_specs, "actions_per_chunk", None)
            if (
                not isinstance(actions_per_chunk, int)
                or isinstance(actions_per_chunk, bool)
                or actions_per_chunk <= 0
            ):
                raise ValueError("Client `actions_per_chunk` must be a positive integer.")
            if horizon > actions_per_chunk:
                raise ValueError("Client RTC execution horizon cannot exceed actions_per_chunk.")
            if (
                isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or not math.isfinite(float(weight))
                or float(weight) <= 0
            ):
                raise ValueError("Client `rtc_max_guidance_weight` must be finite and > 0.")
            if (
                isinstance(beta, bool)
                or not isinstance(beta, (int, float))
                or not math.isfinite(float(beta))
                or float(beta) < 0
            ):
                raise ValueError("Client `rtc_cfg_beta` must be finite and >= 0.")

            effective_rtc = replace(
                server_acp.rtc,
                enabled=True,
                inference_delay=delay,
                execution_horizon=horizon,
                max_guidance_weight=float(weight),
                prefix_attention_schedule=schedule,
            )
            return (
                replace(
                    server_acp,
                    use_cfg=True,
                    batched_cfg=True,
                    cfg_beta=float(beta),
                    rtc=effective_rtc,
                ),
                "client_policy_setup",
            )

        server_rtc_enabled = server_acp.rtc.enabled
        if client_rtc_enabled != server_rtc_enabled:
            raise ValueError(
                "RTC client/server enable mismatch: "
                f"client={client_rtc_enabled}, server={server_rtc_enabled}."
            )
        if not server_rtc_enabled:
            return server_acp, "server_static"

        expected = {
            "rtc_inference_delay": server_acp.rtc.inference_delay,
            "rtc_execution_horizon": server_acp.rtc.execution_horizon,
            "rtc_max_guidance_weight": server_acp.rtc.max_guidance_weight,
            "rtc_prefix_attention_schedule": server_acp.rtc.prefix_attention_schedule,
            "rtc_cfg_beta": server_acp.cfg_beta,
        }
        mismatches = []
        for field_name, expected_value in expected.items():
            actual_value = getattr(policy_specs, field_name, None)
            if field_name == "rtc_prefix_attention_schedule":
                with suppress(TypeError, ValueError):
                    actual_value = self._normalize_client_schedule(actual_value)
            if actual_value != expected_value:
                mismatches.append(f"{field_name}: client={actual_value!r}, server={expected_value!r}")
        if mismatches:
            self._trace_event(
                "rtc_contract_mismatch",
                mismatches=mismatches,
                protocol_version=getattr(policy_specs, "protocol_version", 1),
                client_rtc_enabled=client_rtc_enabled,
            )
            raise ValueError("RTC client/server parameter mismatch: " + "; ".join(mismatches))
        return server_acp, "server_static"

    def _ensure_runtime_rtc_trace(self) -> None:
        rtc = self.acp_inference.rtc
        if self._rtc_trace is None and self.rtc_enabled and rtc.trace_enabled:
            self._rtc_trace = create_rtc_trace(role="server", output_dir=rtc.trace_output_dir)

    def _install_policy_session(
        self,
        policy_specs: RemotePolicyConfig,
        *,
        effective_acp: ACPInferenceConfig,
        parameter_source: str,
        protocol_version: int,
        return_raw_actions: bool,
        client_rtc_enabled: bool,
        observation_wire_version: int,
        observation_codec: str,
    ) -> float:
        """Load and atomically install one policy/runtime configuration.

        Model replacement shares the inference lock with CUDA forwards.  A
        failed load restores the previous active policy/session instead of
        leaving client fields and RTC semantics half-updated.
        """

        state_names = (
            "device",
            "policy_type",
            "pretrained_name_or_path",
            "lerobot_features",
            "actions_per_chunk",
            "policy",
            "preprocessor",
            "postprocessor",
            "_client_protocol_version",
            "_client_return_raw_actions",
            "_client_rtc_enabled",
            "_client_observation_wire_version",
            "_client_observation_codec",
            "_effective_acp_inference",
            "_acp_parameter_source",
        )

        with self._setup_lock, self._inference_lock:
            previous_state = {name: getattr(self, name) for name in state_names}
            setup_started = time.perf_counter()
            try:
                self.device = policy_specs.device
                self.policy_type = policy_specs.policy_type
                self.pretrained_name_or_path = policy_specs.pretrained_name_or_path
                self.lerobot_features = policy_specs.lerobot_features
                self.actions_per_chunk = policy_specs.actions_per_chunk
                self._client_protocol_version = protocol_version
                self._client_return_raw_actions = return_raw_actions
                self._client_rtc_enabled = client_rtc_enabled
                self._client_observation_wire_version = observation_wire_version
                self._client_observation_codec = observation_codec
                self._effective_acp_inference = effective_acp
                self._acp_parameter_source = parameter_source

                policy_class = get_policy_class(self.policy_type)
                self.policy = policy_class.from_pretrained(
                    self.pretrained_name_or_path,
                    device=self.device,
                )
                self.policy.to(self.device)
                self._configure_policy_rtc()

                device_override = {"device": self.device}
                self.preprocessor, self.postprocessor = make_pre_post_processors(
                    self.policy.config,
                    pretrained_path=self.pretrained_name_or_path,
                    preprocessor_overrides={
                        "device_processor": device_override,
                        "rename_observations_processor": {"rename_map": policy_specs.rename_map},
                    },
                    postprocessor_overrides={"device_processor": device_override},
                )
            except Exception:
                for name, value in previous_state.items():
                    setattr(self, name, value)
                raise

            self._ensure_runtime_rtc_trace()
            self._configure_acp_profiler()
            return time.perf_counter() - setup_started

    def _configure_policy_rtc(self) -> None:
        """Install an RTCProcessor into an already-loaded Pi0.5 checkpoint."""
        if not self.rtc_enabled:
            if self._acp_parameter_source.startswith("client_managed"):
                # A new non-RTC client session must not inherit a checkpoint or
                # prior session's processor.  Pi0.5 setup always loads a fresh
                # policy, but clear both wrapper/core fields defensively.
                if hasattr(self.policy.config, "rtc_config"):
                    self.policy.config.rtc_config = None
                if hasattr(self.policy, "rtc_processor"):
                    self.policy.rtc_processor = None
                core_model = getattr(self.policy, "model", None)
                if core_model is not None:
                    if hasattr(core_model, "config"):
                        core_model.config.rtc_config = None
                    if hasattr(core_model, "rtc_processor"):
                        core_model.rtc_processor = None
                return
            if self.batched_cfg_enabled and getattr(self.policy.config, "rtc_config", None) is not None:
                raise ValueError(
                    "Batched ACP-CFG profiling is a no-RTC baseline unless "
                    "`acp_inference.rtc.enabled=true`; set the checkpoint `rtc_config` to null."
                )
            return

        if self.policy_type != "pi05":
            raise ValueError("Server-side RTC-CFG inference is supported only for Pi0.5.")
        if not (
            self._client_protocol_version >= 2
            and self._client_return_raw_actions
            and self._client_rtc_enabled
        ):
            raise ValueError(
                "RTC-CFG requires a protocol-v2 client with `return_raw_actions=true` and `rtc_enabled=true`."
            )
        if not hasattr(self.policy, "init_rtc_processor"):
            raise TypeError("The loaded Pi0.5 policy does not expose `init_rtc_processor()`.")

        server_rtc = self.acp_inference.rtc
        model_chunk_size = int(self.policy.config.chunk_size)
        if self.actions_per_chunk > model_chunk_size:
            raise ValueError(
                "RTC `actions_per_chunk` cannot exceed the Pi0.5 model chunk size: "
                f"{self.actions_per_chunk} > {model_chunk_size}."
            )
        if server_rtc.execution_horizon > self.actions_per_chunk:
            raise ValueError(
                "RTC execution_horizon cannot exceed actions_per_chunk: "
                f"{server_rtc.execution_horizon} > {self.actions_per_chunk}."
            )
        rtc_config = RTCConfig(
            enabled=True,
            prefix_attention_schedule=server_rtc.prefix_attention_schedule,
            max_guidance_weight=server_rtc.max_guidance_weight,
            execution_horizon=server_rtc.execution_horizon,
        )
        # RTC has no learned parameters, so installing it after checkpoint load
        # does not affect strict state-dict loading. init_rtc_processor also
        # attaches the same processor to the already-created core model.
        self.policy.config.rtc_config = rtc_config
        core_model = getattr(self.policy, "model", None)
        if core_model is not None and hasattr(core_model, "config"):
            core_model.config.rtc_config = rtc_config
        self.policy.init_rtc_processor()

        if (
            self.policy.rtc_processor is None
            or core_model is None
            or core_model.rtc_processor is not self.policy.rtc_processor
        ):
            raise RuntimeError("Failed to attach RTCProcessor to the loaded Pi0.5 core model.")

        self.logger.info(
            "Server-side RTC enabled (source=%s, delay=%d, horizon=%d, weight=%.3f, schedule=%s).",
            self._acp_parameter_source,
            server_rtc.inference_delay,
            server_rtc.execution_horizon,
            server_rtc.max_guidance_weight,
            server_rtc.prefix_attention_schedule.value,
        )

    def _configure_acp_profiler(self) -> None:
        """Create a profiler for the currently loaded policy, if requested.

        ``Ready`` calls only reset per-episode queue state, so this profiler and
        its warm-up counter intentionally survive across episodes. A new policy
        instruction reload starts a new in-process warm-up sequence.
        """
        previous_profiler = self.acp_profiler
        if previous_profiler is not None and not self._close_acp_profiler_instance(
            previous_profiler, reason="policy_reconfigured"
        ):
            self._pending_profiler_shutdowns.append(previous_profiler)
        self.acp_profiler = None
        self._acp_profile_disabled = False
        self._acp_profile_last_dropped = 0
        self._acp_profile_last_error = None
        self._acp_chunk_index = 0
        if not self.batched_cfg_enabled:
            return

        acp = self.acp_inference
        if acp.profile:
            from lerobot.scripts.acp_inference_profile import ACPInferenceProfiler

            self.acp_profiler = ACPInferenceProfiler(
                output_root=acp.profile_output_dir,
                fps=self.config.fps,
                cfg_beta=acp.cfg_beta,
                save_chunks=acp.profile_save_chunks,
                run_name=acp.profile_run_name,
            )
            # A named profiling run may be resumed after a server restart. Keep
            # both the persisted index and warm-up classification continuous.
            self._acp_chunk_index = self.acp_profiler.next_index

        self.logger.info(
            "Server-side batched Pi0.5 ACP-CFG enabled (beta=%.3f, profiling=%s, output=%s).",
            acp.cfg_beta,
            acp.profile,
            self.acp_profiler.output_dir if self.acp_profiler is not None else "disabled",
        )

    def _close_acp_profiler_instance(
        self,
        profiler: Any,
        *,
        reason: str,
        timeout_s: float = 5.0,
    ) -> bool:
        """Best-effort bounded profiler drain that never breaks policy service."""
        completed = False
        close_error: str | None = None
        try:
            close = getattr(profiler, "close", None)
            completed = bool(close(timeout_s=timeout_s)) if callable(close) else True
        except Exception as exc:
            close_error = f"{type(exc).__name__}: {exc}"
            self.logger.warning("Could not close ACP profiler cleanly: %s", close_error)

        status = getattr(profiler, "status", None)
        status = status if isinstance(status, dict) else {}
        self._trace_event(
            "acp_profiler_closed",
            reason=reason,
            close_returned=completed,
            close_error=close_error,
            **status,
        )
        if not completed:
            self.logger.warning(
                "ACP profiler did not drain within %.1fs; server shutdown/reconfiguration will continue.",
                timeout_s,
            )
        return completed

    def _close_all_acp_profilers(self, *, reason: str) -> None:
        profilers = [*self._pending_profiler_shutdowns]
        self._pending_profiler_shutdowns.clear()
        if self.acp_profiler is not None:
            profilers.append(self.acp_profiler)
            self.acp_profiler = None
        for profiler in profilers:
            self._close_acp_profiler_instance(profiler, reason=reason)

    def _reset_server(self, *, reason: str = "internal", peer: str | None = None) -> None:
        """Flushes server state when new client connects."""
        with self._generation_lock:
            old_generation = self._server_generation
        with self.observation_queue.mutex:
            queued_obs = self.observation_queue.queue[0] if self.observation_queue.queue else None
        with self._predicted_timesteps_lock:
            predicted_timestep_count = len(self._predicted_timesteps)
        queued_request_id = self._request_id(queued_obs)
        queued_observation_timestep = (
            queued_obs.get_timestep() if isinstance(queued_obs, TimedObservation) else None
        )

        # only running inference on the latest observation received by the server
        self.shutdown_event.set()
        self.observation_queue = Queue(maxsize=1)

        with self._generation_lock:
            self._server_generation += 1
            new_generation = self._server_generation

        with self._predicted_timesteps_lock:
            self._predicted_timesteps = set()
        self._trace_event(
            "server_reset",
            reason=reason,
            peer=peer,
            old_generation=old_generation,
            new_generation=new_generation,
            queued_request_id=queued_request_id,
            queued_observation_timestep=queued_observation_timestep,
            inference_lock_busy=self._inference_lock.locked(),
            predicted_timestep_count=predicted_timestep_count,
        )

    def Ready(self, request, context):  # noqa: N802
        client_id = context.peer()
        self.logger.info(f"Client {client_id} connected and ready")
        self._reset_server(reason="client_ready", peer=client_id)
        self.shutdown_event.clear()

        supported_codecs = [OBSERVATION_CODEC_NONE]
        if self.config.accept_zlib_observations:
            supported_codecs.append(OBSERVATION_CODEC_ZLIB)
        set_trailing_metadata = getattr(context, "set_trailing_metadata", None)
        if callable(set_trailing_metadata):
            set_trailing_metadata(
                (
                    (OBSERVATION_CODECS_METADATA_KEY, ",".join(supported_codecs)),
                    (OBSERVATION_WIRE_VERSION_METADATA_KEY, str(OBSERVATION_PAYLOAD_VERSION)),
                )
            )
        self._trace_event(
            "observation_transport_advertised",
            peer=client_id,
            observation_codecs=supported_codecs,
            observation_wire_version=OBSERVATION_PAYLOAD_VERSION,
            max_observation_payload_bytes=self.config.max_observation_payload_bytes,
        )

        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        """Receive policy instructions from the robot client"""

        if not self.running:
            self.logger.warning("Server is not running. Ignoring policy instructions.")
            return services_pb2.Empty()

        client_id = context.peer()

        policy_specs = pickle.loads(request.data)  # nosec

        if not isinstance(policy_specs, RemotePolicyConfig):
            raise TypeError(f"Policy specs must be a RemotePolicyConfig. Got {type(policy_specs)}")

        self._trace_event(
            "policy_setup_received",
            peer=client_id,
            payload_bytes=len(request.data),
            policy_type=policy_specs.policy_type,
            policy_checkpoint=policy_specs.pretrained_name_or_path,
            requested_device=policy_specs.device,
            actions_per_chunk=policy_specs.actions_per_chunk,
            protocol_version=getattr(policy_specs, "protocol_version", 1),
            return_raw_actions=getattr(policy_specs, "return_raw_actions", False),
            client_rtc_enabled=getattr(policy_specs, "rtc_enabled", False),
            client_inference_delay=getattr(policy_specs, "rtc_inference_delay", None),
            client_execution_horizon=getattr(policy_specs, "rtc_execution_horizon", None),
            client_max_guidance_weight=getattr(policy_specs, "rtc_max_guidance_weight", None),
            client_prefix_attention_schedule=getattr(policy_specs, "rtc_prefix_attention_schedule", None),
            client_cfg_beta=getattr(policy_specs, "rtc_cfg_beta", None),
            client_observation_wire_version=getattr(policy_specs, "observation_wire_version", 0),
            client_observation_codec=getattr(policy_specs, "observation_codec", OBSERVATION_CODEC_NONE),
        )

        if policy_specs.policy_type not in SUPPORTED_POLICIES:
            raise ValueError(
                f"Policy type {policy_specs.policy_type} not supported. "
                f"Supported policies: {SUPPORTED_POLICIES}"
            )

        # getattr preserves compatibility with RemotePolicyConfig objects
        # pickled by version-1 clients before these capability fields existed.
        client_protocol_version = getattr(policy_specs, "protocol_version", 1)
        client_return_raw_actions = getattr(policy_specs, "return_raw_actions", False)
        client_rtc_enabled = getattr(policy_specs, "rtc_enabled", False)
        client_observation_wire_version = getattr(policy_specs, "observation_wire_version", 0)
        client_observation_codec = getattr(policy_specs, "observation_codec", OBSERVATION_CODEC_NONE)
        if (
            not isinstance(client_protocol_version, int)
            or isinstance(client_protocol_version, bool)
            or client_protocol_version < 1
        ):
            raise ValueError("Remote policy `protocol_version` must be a positive integer.")
        if not isinstance(client_return_raw_actions, bool):
            raise TypeError("Remote policy `return_raw_actions` must be a bool.")
        if (
            not isinstance(client_observation_wire_version, int)
            or isinstance(client_observation_wire_version, bool)
            or client_observation_wire_version < 0
        ):
            raise ValueError("Remote policy `observation_wire_version` must be a non-negative integer.")
        if not isinstance(client_observation_codec, str) or client_observation_codec not in {
            OBSERVATION_CODEC_NONE,
            OBSERVATION_CODEC_ZLIB,
        }:
            raise ValueError(f"Unsupported client observation codec {client_observation_codec!r}.")
        if client_observation_codec == OBSERVATION_CODEC_ZLIB:
            if not self.config.accept_zlib_observations:
                raise ValueError("This policy server is configured to reject zlib observations.")
            if client_observation_wire_version != OBSERVATION_PAYLOAD_VERSION:
                raise ValueError(
                    "zlib observation transport requires observation wire version "
                    f"{OBSERVATION_PAYLOAD_VERSION}."
                )

        effective_acp, parameter_source = self._resolve_client_acp_contract(policy_specs)
        if effective_acp.enable and effective_acp.batched_cfg and policy_specs.policy_type != "pi05":
            raise ValueError("Server-side batched ACP-CFG inference is supported only for Pi0.5.")

        self.logger.info(
            f"Receiving policy instructions from {client_id} | "
            f"Policy type: {policy_specs.policy_type} | "
            f"Pretrained name or path: {policy_specs.pretrained_name_or_path} | "
            f"Actions per chunk: {policy_specs.actions_per_chunk} | "
            f"Device: {policy_specs.device}"
        )
        setup_time = self._install_policy_session(
            policy_specs,
            effective_acp=effective_acp,
            parameter_source=parameter_source,
            protocol_version=client_protocol_version,
            return_raw_actions=client_return_raw_actions,
            client_rtc_enabled=client_rtc_enabled,
            observation_wire_version=client_observation_wire_version,
            observation_codec=client_observation_codec,
        )

        self.logger.info(
            "Time taken to put policy on %s: %.4f seconds (ACP parameters=%s)",
            self.device,
            setup_time,
            self._acp_parameter_source,
        )
        self._trace_event(
            "client_inference_contract_applied",
            peer=client_id,
            parameter_source=self._acp_parameter_source,
            batched_cfg_enabled=self.batched_cfg_enabled,
            rtc_enabled=self.rtc_enabled,
            cfg_beta=self.acp_inference.cfg_beta if self.batched_cfg_enabled else None,
            inference_delay=(self.acp_inference.rtc.inference_delay if self.rtc_enabled else None),
            execution_horizon=(self.acp_inference.rtc.execution_horizon if self.rtc_enabled else None),
            max_guidance_weight=(self.acp_inference.rtc.max_guidance_weight if self.rtc_enabled else None),
            prefix_attention_schedule=(
                self.acp_inference.rtc.prefix_attention_schedule if self.rtc_enabled else None
            ),
        )
        self._trace_event(
            "policy_loaded",
            peer=client_id,
            policy_type=self.policy_type,
            policy_checkpoint=self.pretrained_name_or_path,
            device=str(self.device),
            actions_per_chunk=self.actions_per_chunk,
            model_chunk_size=getattr(self.policy.config, "chunk_size", None),
            model_action_dim=getattr(self.policy.config, "max_action_dim", None),
            setup_ms=setup_time * 1000,
            protocol_version=self._client_protocol_version,
            return_raw_actions=self._client_return_raw_actions,
            rtc_enabled=self.rtc_enabled,
            acp_parameter_source=self._acp_parameter_source,
            cfg_beta=self.acp_inference.cfg_beta if self.batched_cfg_enabled else None,
            observation_wire_version=self._client_observation_wire_version,
            observation_codec=self._client_observation_codec,
        )

        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        """Receive observations from the robot client"""
        handler_starts = time.perf_counter()
        handler_started_wall_s = time.time()
        client_id = context.peer()
        self.logger.debug(f"Receiving observations from {client_id}")
        with self._generation_lock:
            receive_generation = self._server_generation

        stream_receive_starts = time.perf_counter()
        try:
            received_bytes = receive_bytes_in_chunks(
                request_iterator,
                None,
                self.shutdown_event,
                self.logger,
                max_size_bytes=self.config.max_observation_payload_bytes,
            )  # blocking call while looping over request_iterator
        except ValueError as exc:
            self._trace_event(
                "observation_payload_rejected",
                peer=client_id,
                stage="stream_receive",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            abort = getattr(context, "abort", None)
            if callable(abort):
                return abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            raise
        stream_receive_stops = time.perf_counter()
        payload_received_wall_s = time.time()

        if received_bytes is None:
            with self._generation_lock:
                stale_generation = receive_generation != self._server_generation
                current_generation = self._server_generation
            is_active = getattr(context, "is_active", None)
            context_active = bool(is_active()) if callable(is_active) else True
            if stale_generation or not context_active or self.shutdown_event.is_set():
                self._trace_event(
                    "observation_discarded",
                    peer=client_id,
                    reason="stream_ended_during_reset_or_cancellation",
                    receive_generation=receive_generation,
                    current_generation=current_generation,
                    context_active=context_active,
                    handler_elapsed_ms=(time.perf_counter() - handler_starts) * 1000,
                )
                return services_pb2.Empty()

            exc = ValueError("Observation stream ended before a TRANSFER_END chunk was received.")
            self._trace_event(
                "observation_payload_rejected",
                peer=client_id,
                stage="stream_receive",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            abort = getattr(context, "abort", None)
            if callable(abort):
                return abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            raise exc

        payload_decode_starts = time.perf_counter()
        try:
            decoded_observation = decode_observation_payload(
                received_bytes,
                max_uncompressed_bytes=self.config.max_observation_payload_bytes,
                allow_zlib=self.config.accept_zlib_observations,
            )
        except ValueError as exc:
            self._trace_event(
                "observation_payload_rejected",
                peer=client_id,
                stage="payload_decode",
                payload_bytes=len(received_bytes),
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            abort = getattr(context, "abort", None)
            if callable(abort):
                return abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            raise
        payload_decode_stops = time.perf_counter()

        pickle_deserialize_starts = time.perf_counter()
        try:
            timed_observation = pickle.loads(decoded_observation.data)  # nosec
            if not isinstance(timed_observation, TimedObservation):
                raise TypeError(
                    f"Decoded observation must be a TimedObservation, got {type(timed_observation).__name__}."
                )
        except Exception as exc:
            self._trace_event(
                "observation_payload_rejected",
                peer=client_id,
                stage="pickle_deserialize",
                payload_bytes=len(received_bytes),
                observation_pickle_bytes=decoded_observation.raw_bytes,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            abort = getattr(context, "abort", None)
            if callable(abort):
                return abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            raise
        pickle_deserialize_stops = time.perf_counter()

        stream_receive_ms = (stream_receive_stops - stream_receive_starts) * 1000
        payload_decode_ms = (payload_decode_stops - payload_decode_starts) * 1000
        pickle_deserialize_ms = (pickle_deserialize_stops - pickle_deserialize_starts) * 1000
        receive_and_deserialize_ms = (pickle_deserialize_stops - stream_receive_starts) * 1000

        self.logger.debug(f"Received observation #{timed_observation.get_timestep()}")

        obs_timestep = timed_observation.get_timestep()
        obs_timestamp = timed_observation.get_timestamp()
        observation_to_handler_start_wall_ms = (handler_started_wall_s - obs_timestamp) * 1000
        observation_to_payload_received_wall_ms = (payload_received_wall_s - obs_timestamp) * 1000
        request_id = self._request_id(timed_observation)
        rtc_metadata = getattr(timed_observation, "rtc_metadata", None)
        leftover = (
            rtc_metadata.prev_chunk_left_over if isinstance(rtc_metadata, RTCInferenceMetadata) else None
        )
        with self._generation_lock:
            current_generation = self._server_generation
        context_active = context.is_active()
        self._trace_event(
            "observation_received",
            request_id=request_id,
            peer=client_id,
            receive_generation=receive_generation,
            current_generation=current_generation,
            context_active=context_active,
            observation_timestep=obs_timestep,
            must_go=timed_observation.must_go,
            payload_bytes=len(received_bytes),
            request_payload_bytes=len(received_bytes),
            observation_pickle_bytes=decoded_observation.raw_bytes,
            observation_codec=decoded_observation.codec,
            observation_wire_version=decoded_observation.wire_version,
            observation_payload_decode_ms=payload_decode_ms,
            observation_decompress_ms=decoded_observation.decompression_ms,
            observation_compression_ratio=decoded_observation.compression_ratio,
            observation_compression_savings_bytes=(
                decoded_observation.raw_bytes - decoded_observation.wire_bytes
            ),
            stream_receive_ms=stream_receive_ms,
            pickle_deserialize_ms=pickle_deserialize_ms,
            receive_and_deserialize_ms=receive_and_deserialize_ms,
            handler_decode_ready_ms=(pickle_deserialize_stops - handler_starts) * 1000,
            server_handler_started_wall_s=handler_started_wall_s,
            server_payload_received_wall_s=payload_received_wall_s,
            observation_timestamp_to_server_handler_start_wall_ms=observation_to_handler_start_wall_ms,
            observation_timestamp_to_server_payload_received_wall_ms=(
                observation_to_payload_received_wall_ms
            ),
            cross_host_wall_clock_metrics_require_synchronized_clocks=True,
            # Compatibility aliases for existing trace consumers. Historically
            # ``deserialize_ms`` included stream reception; it now also includes
            # the explicitly reported payload decode/decompression stage.  The
            # ``client_to_server_wall_ms`` ended at server handler entry.
            deserialize_ms=receive_and_deserialize_ms,
            deserialize_ms_semantics=("stream_receive_plus_payload_decode_plus_pickle_deserialize"),
            client_to_server_wall_ms=observation_to_handler_start_wall_ms,
            client_to_server_wall_ms_semantics="legacy_client_timestamp_to_server_handler_start",
            leftover_steps=(
                int(leftover.shape[0])
                if isinstance(leftover, torch.Tensor) and leftover.ndim >= 1
                else 0
                if leftover is None
                else None
            ),
            leftover=tensor_metadata(leftover),
            requested_inference_delay=(
                rtc_metadata.inference_delay if isinstance(rtc_metadata, RTCInferenceMetadata) else None
            ),
            requested_execution_horizon=(
                rtc_metadata.execution_horizon if isinstance(rtc_metadata, RTCInferenceMetadata) else None
            ),
        )

        # Calculate FPS metrics
        fps_metrics = self.fps_tracker.calculate_fps_metrics(obs_timestamp)

        self.logger.debug(
            f"Received observation #{obs_timestep} | "
            f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "  # fps at which observations are received from client
            f"Target: {fps_metrics['target_fps']:.2f} | "
            f"Client timestamp to handler start: {observation_to_handler_start_wall_ms:.2f}ms"
        )

        self.logger.debug(
            f"Server handler-start timestamp: {handler_started_wall_s:.6f} | "
            f"Client timestamp: {obs_timestamp:.6f} | "
            f"Stream receive time: {stream_receive_ms / 1000:.6f}s | "
            f"Pickle deserialization time: {pickle_deserialize_ms / 1000:.6f}s"
        )

        with self._generation_lock:
            stale_generation = receive_generation != self._server_generation
            current_generation = self._server_generation
        context_active = context.is_active()
        if stale_generation or not context_active:
            self.logger.info("Discarding an observation stream invalidated by reset/cancellation.")
            self._trace_event(
                "observation_discarded",
                request_id=request_id,
                reason="reset_or_cancellation",
                receive_generation=receive_generation,
                current_generation=current_generation,
                context_active=context_active,
                observation_timestep=obs_timestep,
                handler_elapsed_ms=(time.perf_counter() - handler_starts) * 1000,
                observation_queue_size=self.observation_queue.qsize(),
            )
            return services_pb2.Empty()

        observation_queue_size_before = self.observation_queue.qsize()
        enqueue_starts = time.perf_counter()
        accepted = self._enqueue_observation(
            timed_observation  # wrapping a RawObservation
        )
        enqueue_ms = (time.perf_counter() - enqueue_starts) * 1000
        observation_queue_size_after = self.observation_queue.qsize()
        self._trace_event(
            "observation_enqueued",
            request_id=request_id,
            observation_timestep=obs_timestep,
            generation=current_generation,
            accepted=accepted,
            enqueue_ms=enqueue_ms,
            handler_elapsed_ms=(time.perf_counter() - handler_starts) * 1000,
            observation_queue_size_before=observation_queue_size_before,
            observation_queue_size_after=observation_queue_size_after,
            # Compatibility alias for the queue size after the enqueue attempt.
            observation_queue_size=observation_queue_size_after,
        )
        if not accepted:
            self.logger.warning(
                f"Observation #{obs_timestep} was filtered out (must_go={timed_observation.must_go}). "
                f"GetActions will return empty until a must_go observation is enqueued; "
                f"sync clients that block on GetActions may time out."
            )

        return services_pb2.Empty()

    def GetActions(self, request, context):  # noqa: N802
        """Returns actions to the robot client. Actions are sent as a single
        chunk, containing multiple actions."""
        client_id = context.peer()
        self.logger.debug(f"Client {client_id} connected for action streaming")
        obs = None
        request_id = None
        request_generation = None
        phase = "waiting_for_observation"

        # Generate action based on the most recent observation and its timestep
        try:
            getactions_starts = time.perf_counter()
            with self._generation_lock:
                request_generation = self._server_generation
            queue_wait_starts = time.perf_counter()
            obs = self.observation_queue.get(timeout=self.config.obs_queue_timeout)
            queue_wait_ms = (time.perf_counter() - queue_wait_starts) * 1000
            request_id = self._request_id(obs)
            phase = "inference"
            self._trace_event(
                "inference_dequeued",
                request_id=request_id,
                peer=client_id,
                generation=request_generation,
                observation_timestep=obs.get_timestep(),
                must_go=obs.must_go,
                observation_queue_wait_ms=queue_wait_ms,
                observation_queue_size_after_dequeue=self.observation_queue.qsize(),
            )
            self.logger.info(
                f"Running inference for observation #{obs.get_timestep()} (must_go: {obs.must_go})"
            )

            with self._predicted_timesteps_lock:
                self._predicted_timesteps.add(obs.get_timestep())

            inference_starts = time.perf_counter()
            action_chunk = self._run_serialized_inference(obs, request_generation, context)
            serialized_inference_ms = (time.perf_counter() - inference_starts) * 1000
            if action_chunk is None:
                return services_pb2.Empty()

            phase = "serialization"
            pickle_serialize_starts = time.perf_counter()
            actions_bytes = pickle.dumps(action_chunk)  # nosec
            response_pickle_serialize_ms = (time.perf_counter() - pickle_serialize_starts) * 1000

            # Create and return the action chunk
            protobuf_build_starts = time.perf_counter()
            actions = services_pb2.Actions(data=actions_bytes)
            response_protobuf_build_ms = (time.perf_counter() - protobuf_build_starts) * 1000
            rpc_ready_ms = (time.perf_counter() - getactions_starts) * 1000
            response_steps = (
                len(action_chunk.actions)
                if isinstance(action_chunk, RemoteActionChunk)
                else len(action_chunk)
            )
            self._trace_event(
                "response_ready",
                request_id=request_id,
                peer=client_id,
                generation=request_generation,
                observation_timestep=obs.get_timestep(),
                observation_queue_wait_ms=queue_wait_ms,
                serialized_inference_ms=serialized_inference_ms,
                response_pickle_serialize_ms=response_pickle_serialize_ms,
                response_protobuf_build_ms=response_protobuf_build_ms,
                rpc_ready_ms=rpc_ready_ms,
                response_payload_bytes=len(actions_bytes),
                payload_bytes=len(actions_bytes),
                response_steps=response_steps,
                # Compatibility aliases for existing trace consumers. These
                # legacy fields exclude observation queue wait, protobuf build,
                # configured pacing and transport back to the client.
                inference_rpc_ms=serialized_inference_ms,
                inference_rpc_ms_semantics="legacy_serialized_inference_including_lock_wait",
                serialize_ms=response_pickle_serialize_ms,
                serialize_ms_semantics="legacy_response_pickle_serialize",
                total_rpc_ms=serialized_inference_ms + response_pickle_serialize_ms,
                total_rpc_ms_semantics="legacy_inference_plus_pickle_serialize",
            )

            self.logger.info(
                f"Action chunk #{obs.get_timestep()} generated | RPC ready time: {rpc_ready_ms:.2f}ms"
            )

            self.logger.debug(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Queue wait: {queue_wait_ms / 1000:.2f}s | "
                f"Inference time: {serialized_inference_ms / 1000:.2f}s | "
                f"Serialize time: {response_pickle_serialize_ms / 1000:.2f}s | "
                f"RPC ready time: {rpc_ready_ms / 1000:.2f}s"
            )

            phase = "latency_pacing"
            pacing_starts = time.perf_counter()
            requested_pacing_s = max(
                0, self.config.inference_latency - max(0, pacing_starts - getactions_starts)
            )
            time.sleep(requested_pacing_s)  # sleep controls inference latency
            pacing_sleep_ms = (time.perf_counter() - pacing_starts) * 1000
            self._trace_event(
                "response_returning",
                request_id=request_id,
                peer=client_id,
                generation=request_generation,
                observation_timestep=obs.get_timestep(),
                rpc_ready_ms=rpc_ready_ms,
                requested_pacing_ms=requested_pacing_s * 1000,
                pacing_sleep_ms=pacing_sleep_ms,
                handler_elapsed_ms=(time.perf_counter() - getactions_starts) * 1000,
                response_payload_bytes=len(actions_bytes),
            )

            return actions

        except Empty:  # no observation added to queue in obs_queue_timeout
            self._trace_event(
                "get_actions_empty",
                peer=client_id,
                generation=request_generation,
                observation_queue_timeout_s=self.config.obs_queue_timeout,
                observation_queue_wait_ms=(time.perf_counter() - queue_wait_starts) * 1000,
                handler_elapsed_ms=(time.perf_counter() - getactions_starts) * 1000,
            )
            return services_pb2.Empty()

        except Exception as e:
            self.logger.exception("Policy inference failed while serving GetActions")
            self._trace_event(
                "inference_error",
                request_id=request_id,
                peer=client_id,
                generation=request_generation,
                observation_timestep=obs.get_timestep() if obs is not None else None,
                phase=phase,
                error_type=type(e).__name__,
                error_message=str(e),
            )
            context.abort(grpc.StatusCode.INTERNAL, f"Policy inference failed: {e}")

    def _run_serialized_inference(
        self,
        obs: TimedObservation,
        request_generation: int,
        context,
    ) -> list[TimedAction] | RemoteActionChunk | None:
        """Run at most one model forward and discard work invalidated before or during it."""
        lock_wait_start = time.perf_counter()
        with self._inference_lock:
            lock_wait_ms = (time.perf_counter() - lock_wait_start) * 1000
            with self._generation_lock:
                stale_generation = request_generation != self._server_generation
                current_generation = self._server_generation
            is_active = getattr(context, "is_active", None)
            cancelled = callable(is_active) and not is_active()
            if stale_generation or cancelled:
                self.logger.info(
                    "Skipping invalidated inference request (stale_generation=%s, cancelled=%s).",
                    stale_generation,
                    cancelled,
                )
                self._trace_event(
                    "inference_skipped",
                    request_id=self._request_id(obs),
                    observation_timestep=obs.get_timestep(),
                    request_generation=request_generation,
                    current_generation=current_generation,
                    stale_generation=stale_generation,
                    context_cancelled=cancelled,
                    inference_lock_wait_ms=lock_wait_ms,
                )
                return None
            self._trace_event(
                "inference_started",
                request_id=self._request_id(obs),
                observation_timestep=obs.get_timestep(),
                generation=request_generation,
                inference_lock_wait_ms=lock_wait_ms,
            )
            result = self._predict_action_chunk(obs)
            with self._generation_lock:
                invalidated_after_forward = request_generation != self._server_generation
                generation_after_forward = self._server_generation
            cancelled_after_forward = callable(is_active) and not is_active()
            if invalidated_after_forward or cancelled_after_forward:
                self.logger.info(
                    "Discarding inference result invalidated during model forward "
                    "(stale_generation=%s, cancelled=%s).",
                    invalidated_after_forward,
                    cancelled_after_forward,
                )
                self._trace_event(
                    "inference_invalidated_after_forward",
                    request_id=self._request_id(obs),
                    observation_timestep=obs.get_timestep(),
                    request_generation=request_generation,
                    current_generation=generation_after_forward,
                    stale_generation=invalidated_after_forward,
                    context_cancelled=cancelled_after_forward,
                )
                return None
            return result

    def _obs_sanity_checks(self, obs: TimedObservation, previous_obs: TimedObservation) -> bool:
        """Check if the observation is valid to be processed by the policy"""
        with self._predicted_timesteps_lock:
            predicted_timesteps = self._predicted_timesteps

        if obs.get_timestep() in predicted_timesteps:
            self.logger.debug(f"Skipping observation #{obs.get_timestep()} - Timestep predicted already!")
            return False

        elif observations_similar(obs, previous_obs, lerobot_features=self.lerobot_features):
            self.logger.debug(
                f"Skipping observation #{obs.get_timestep()} - Observation too similar to last obs predicted!"
            )
            return False

        else:
            return True

    def _enqueue_observation(self, obs: TimedObservation) -> bool:
        """Enqueue an observation if it must go through processing, otherwise skip it.
        Observations not in queue are never run through the policy network"""

        if obs.must_go:
            # A forced observation may intentionally reuse a timestep after an
            # episode/client reset. Remove only that timestep's dedup state;
            # preserving all other entries still prevents stale duplicates.
            with self._predicted_timesteps_lock:
                self._predicted_timesteps.discard(obs.get_timestep())
            if (
                self.last_processed_obs is not None
                and self.last_processed_obs.get_timestep() == obs.get_timestep()
            ):
                self.last_processed_obs = None

        if (
            obs.must_go
            or self.last_processed_obs is None
            or self._obs_sanity_checks(obs, self.last_processed_obs)
        ):
            last_obs = self.last_processed_obs.get_timestep() if self.last_processed_obs else "None"
            self.logger.debug(
                f"Enqueuing observation. Must go: {obs.must_go} | Last processed obs: {last_obs}"
            )

            # If queue is full, get the old observation to make room
            if self.observation_queue.full():
                # pops from queue
                discarded_obs = self.observation_queue.get_nowait()
                self.logger.debug("Observation queue was full, removed oldest observation")
                self._trace_event(
                    "observation_queue_replaced",
                    discarded_request_id=self._request_id(discarded_obs),
                    discarded_observation_timestep=discarded_obs.get_timestep(),
                    replacement_request_id=self._request_id(obs),
                    replacement_observation_timestep=obs.get_timestep(),
                )

            # Now put the new observation (never blocks as queue is non-full here)
            self.observation_queue.put(obs)
            return True

        return False

    def _time_action_chunk(self, t_0: float, action_chunk: list[torch.Tensor], i_0: int) -> list[TimedAction]:
        """Turn a chunk of actions into a list of TimedAction instances,
        with the first action corresponding to t_0 and the rest corresponding to
        t_0 + i*environment_dt for i in range(len(action_chunk))
        """
        return [
            TimedAction(timestamp=t_0 + i * self.config.environment_dt, timestep=i_0 + i, action=action)
            for i, action in enumerate(action_chunk)
        ]

    def _prepare_batched_cfg_observation(self, observation: Observation) -> dict[str, Any]:
        """Build and preprocess a fixed-order ``[unconditional, conditional]`` batch."""
        if self.preprocessor is None:
            raise RuntimeError("Policy preprocessor is not initialized.")

        task_value = observation.get("task", "")
        if task_value is None:
            task = ""
        elif isinstance(task_value, str):
            task = task_value
        elif isinstance(task_value, list) and len(task_value) == 1 and isinstance(task_value[0], str):
            task = task_value[0]
        else:
            raise ValueError(
                "Batched ACP-CFG expects one task string before preprocessing, "
                f"got {type(task_value).__name__}."
            )

        batched_observation = dict(observation)
        for key, value in list(batched_observation.items()):
            if not isinstance(value, torch.Tensor):
                continue
            if value.ndim == 0 or value.shape[0] != 1:
                raise ValueError(
                    "Batched ACP-CFG expects a single observation before batching, "
                    f"got {key} shape={tuple(value.shape)}."
                )
            repeats = (2,) + (1,) * (value.ndim - 1)
            batched_observation[key] = value.repeat(repeats)

        # Row 0 is U and row 1 is C throughout model inference and profiling.
        batched_observation["task"] = [task, build_acp_tagged_task(task, is_positive=True)]
        return self.preprocessor(batched_observation)

    def _prepare_rtc_inputs(
        self,
        metadata: RTCInferenceMetadata | None,
        *,
        chunk_size: int,
        max_action_dim: int,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, int, int]:
        """Canonicalize one raw leftover and share it across the U/C rows."""
        rtc = self.acp_inference.rtc
        requested_delay = (
            metadata.inference_delay
            if metadata is not None and metadata.inference_delay is not None
            else rtc.inference_delay
        )
        requested_horizon = (
            metadata.execution_horizon
            if metadata is not None and metadata.execution_horizon is not None
            else rtc.execution_horizon
        )
        if not isinstance(requested_delay, int) or isinstance(requested_delay, bool) or requested_delay < 0:
            raise ValueError("RTC `inference_delay` must be a non-negative integer.")
        if (
            not isinstance(requested_horizon, int)
            or isinstance(requested_horizon, bool)
            or requested_horizon <= requested_delay
        ):
            raise ValueError("RTC `execution_horizon` must be an integer greater than inference delay.")
        if requested_delay != rtc.inference_delay or requested_horizon != rtc.execution_horizon:
            raise ValueError(
                "Per-request RTC parameters must match the negotiated session configuration: "
                f"client delay/horizon={requested_delay}/{requested_horizon}, "
                f"session={rtc.inference_delay}/{rtc.execution_horizon}."
            )

        leftover = metadata.prev_chunk_left_over if metadata is not None else None
        if leftover is None:
            return None, min(requested_delay, chunk_size), min(requested_horizon, chunk_size)
        if not isinstance(leftover, torch.Tensor):
            raise TypeError("RTC `prev_chunk_left_over` must be a torch.Tensor or None.")
        if leftover.ndim == 3:
            if leftover.shape[0] != 1:
                raise ValueError(
                    "RTC request leftover must be unbatched [L, A] or have one row [1, L, A], "
                    f"got {tuple(leftover.shape)}."
                )
            leftover = leftover.squeeze(0)
        if leftover.ndim != 2:
            raise ValueError(f"RTC request leftover must have shape [L, A], got {tuple(leftover.shape)}.")
        if leftover.shape[1] > max_action_dim:
            raise ValueError(
                "RTC leftover action dimension exceeds Pi0.5 max_action_dim: "
                f"{leftover.shape[1]} > {max_action_dim}."
            )
        if not bool(torch.isfinite(leftover).all()):
            raise FloatingPointError("RTC leftover contains NaN or Inf values.")

        # The earliest leftover positions are the temporal prefix corresponding
        # to the new chunk, so keep the first T entries if a malformed client
        # supplies more than one model horizon.
        leftover = leftover[:chunk_size].detach().to(device=device, dtype=torch.float32).contiguous()
        leftover_length = int(leftover.shape[0])
        if leftover_length == 0:
            return None, min(requested_delay, chunk_size), min(requested_horizon, chunk_size)

        shared_leftover = leftover.unsqueeze(0).repeat(2, 1, 1)
        effective_horizon = min(requested_horizon, leftover_length, chunk_size)
        effective_delay = min(requested_delay, effective_horizon)
        return shared_leftover, effective_delay, effective_horizon

    def _get_batched_cfg_action_chunk(
        self,
        observation: dict[str, Any],
        rtc_metadata: RTCInferenceMetadata | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any], float | None]:
        """Run one Pi0.5 batch=2 call and blend U/C in normalized action space."""
        if self.policy is None or self.actions_per_chunk is None or self.device is None:
            raise RuntimeError("PolicyServer must receive policy instructions before inference.")

        chunk_size = int(self.policy.config.chunk_size)
        max_action_dim = int(self.policy.config.max_action_dim)
        device = torch.device(self.device)

        # Match Pi0.5's native sample_noise implementation exactly, then copy
        # the same initial noise into the U and C rows.
        base_noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=(1, chunk_size, max_action_dim),
            dtype=torch.float32,
            device=device,
        )
        shared_noise = base_noise.repeat(2, 1, 1)

        cuda_start = cuda_end = None
        if self._cuda_timing_enabled(device):
            cuda_start = torch.cuda.Event(enable_timing=True)
            cuda_end = torch.cuda.Event(enable_timing=True)
            cuda_start.record(torch.cuda.current_stream(device))

        rtc_leftover = None
        rtc_delay = None
        rtc_horizon = None
        if self.rtc_enabled:
            rtc_leftover, rtc_delay, rtc_horizon = self._prepare_rtc_inputs(
                rtc_metadata,
                chunk_size=chunk_size,
                max_action_dim=max_action_dim,
                device=device,
            )
            raw_chunks = self.policy.predict_action_chunk(
                observation,
                noise=shared_noise,
                prev_chunk_left_over=rtc_leftover,
                inference_delay=rtc_delay,
                execution_horizon=rtc_horizon,
            )
        else:
            raw_chunks = self.policy.predict_action_chunk(observation, noise=shared_noise)

        if cuda_end is not None:
            cuda_end.record(torch.cuda.current_stream(device))
        cuda_latency_ms = None

        if raw_chunks.ndim != 3 or raw_chunks.shape[0] != 2:
            raise ValueError(
                f"Batched ACP-CFG requires Pi0.5 to return [2, T, A], got {tuple(raw_chunks.shape)}."
            )

        raw_uncond = raw_chunks[0:1]
        raw_cond = raw_chunks[1:2]
        raw_cfg = raw_uncond + self.acp_inference.cfg_beta * (raw_cond - raw_uncond)
        if not bool(torch.isfinite(raw_cfg).all()):
            raise FloatingPointError("Batched ACP-CFG produced a non-finite raw action chunk.")
        if cuda_end is not None:
            # The finite-value check above synchronizes this stream, so reading
            # the events adds no extra CUDA synchronization to the trace path.
            cuda_latency_ms = cuda_start.elapsed_time(cuda_end)

        execution_steps = min(int(self.actions_per_chunk), int(raw_cfg.shape[1]))
        if execution_steps <= 0:
            raise ValueError("`actions_per_chunk` must select at least one Pi0.5 action.")

        raw_uncond = raw_uncond[:, :execution_steps]
        raw_cond = raw_cond[:, :execution_steps]
        raw_cfg = raw_cfg[:, :execution_steps]
        artifacts = {
            "noise": base_noise,
            "uncond_raw": raw_uncond,
            "cond_raw": raw_cond,
            "cfg_raw": raw_cfg,
            "shared_noise": shared_noise,
            "rtc_leftover": rtc_leftover,
            "rtc_inference_delay": rtc_delay,
            "rtc_execution_horizon": rtc_horizon,
        }
        return raw_cfg, artifacts, cuda_latency_ms

    def _postprocess_action_chunk(self, action_tensor: torch.Tensor) -> torch.Tensor:
        """Postprocess a chunk while preserving the legacy path for non-CFG policies."""
        if self.postprocessor is None:
            raise RuntimeError("Policy postprocessor is not initialized.")

        if self.batched_cfg_enabled:
            processed = self.postprocessor(action_tensor)
            if processed.ndim != 3 or processed.shape[0] != 1:
                raise ValueError(
                    f"Batched ACP-CFG postprocessing must preserve [1, T, A], got {tuple(processed.shape)}."
                )
            if not bool(torch.isfinite(processed).all()):
                raise FloatingPointError("Batched ACP-CFG produced a non-finite processed action chunk.")
            return processed

        # Preserve the original server behavior for every policy when ACP-CFG
        # is disabled: postprocess one timestep at a time.
        _, chunk_size, _ = action_tensor.shape
        processed_actions = []
        for i in range(chunk_size):
            processed_actions.append(self.postprocessor(action_tensor[:, i, :]))
        return torch.stack(processed_actions, dim=1)

    def _get_action_chunk(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        """Get an action chunk from the policy. The chunk contains only"""
        chunk = self.policy.predict_action_chunk(observation)
        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)  # adding batch dimension, now shape is (B, chunk_size, action_dim)

        return chunk[:, : self.actions_per_chunk, :]

    def _record_acp_profile(
        self,
        *,
        observation_t: TimedObservation,
        artifacts: dict[str, Any],
        processed_cfg: torch.Tensor,
        timings_s: dict[str, float],
        wall_latency_s: float,
        cuda_latency_ms: float | None,
        memory_before: int | None,
        peak_memory: int | None,
    ) -> None:
        profiler = self.acp_profiler
        if profiler is None or self._acp_profile_disabled:
            return

        raw_uncond = artifacts["uncond_raw"]
        raw_cond = artifacts["cond_raw"]
        raw_cfg = artifacts["cfg_raw"]
        shared_noise = artifacts["shared_noise"]
        branch_delta = raw_cond - raw_uncond
        acp = self.acp_inference

        metrics = {
            "chunk_index": self._acp_chunk_index,
            "observation_timestep": observation_t.get_timestep(),
            "policy_type": self.policy_type,
            "policy_checkpoint": self.pretrained_name_or_path,
            "device": str(self.device),
            "dtype": str(raw_cfg.dtype),
            "batch_size": 2,
            "cfg_beta": acp.cfg_beta,
            "acp_parameter_source": self._acp_parameter_source,
            "fps": self.config.fps,
            "model_chunk_size": int(self.policy.config.chunk_size),
            "execution_steps": int(processed_cfg.shape[1]),
            "action_dim": int(processed_cfg.shape[2]),
            "wall_latency_s": wall_latency_s,
            "prepare_latency_ms": timings_s["prepare"] * 1000,
            "prepare_calling_thread_cpu_latency_ms": timings_s["prepare_cpu"] * 1000,
            "prepare_non_calling_thread_or_wait_ms": (
                max(timings_s["prepare"] - timings_s["prepare_cpu"], 0) * 1000
            ),
            "preprocess_latency_ms": timings_s["preprocess"] * 1000,
            "preprocess_calling_thread_cpu_latency_ms": timings_s["preprocess_cpu"] * 1000,
            "preprocess_non_calling_thread_or_wait_ms": (
                max(timings_s["preprocess"] - timings_s["preprocess_cpu"], 0) * 1000
            ),
            "model_and_cfg_latency_ms": timings_s["inference"] * 1000,
            "postprocess_and_d2h_latency_ms": timings_s["postprocess"] * 1000,
            "cuda_latency_ms": cuda_latency_ms,
            "cuda_memory_allocated_before_bytes": memory_before,
            "cuda_peak_memory_bytes": peak_memory,
            "cuda_peak_memory_increment_bytes": (
                max(peak_memory - memory_before, 0)
                if peak_memory is not None and memory_before is not None
                else None
            ),
            "shared_noise_max_abs_diff": (shared_noise[0] - shared_noise[1]).abs().max().item(),
            "warmup": self._acp_chunk_index < acp.profile_warmup_chunks,
            "raw_cfg_min": raw_cfg.min().item(),
            "raw_cfg_max": raw_cfg.max().item(),
            "raw_cfg_mean": raw_cfg.mean().item(),
            "raw_cfg_std": raw_cfg.std(unbiased=False).item(),
            "raw_cfg_out_of_range_ratio": (raw_cfg.abs() > 1.0).float().mean().item(),
            "processed_cfg_min": processed_cfg.min().item(),
            "processed_cfg_max": processed_cfg.max().item(),
            "cond_uncond_delta_l2_mean": torch.linalg.vector_norm(branch_delta, dim=-1).mean().item(),
            "rtc_enabled": self.rtc_enabled,
            "rtc_inference_delay": artifacts["rtc_inference_delay"],
            "rtc_execution_horizon": artifacts["rtc_execution_horizon"],
            "rtc_leftover_steps": (
                int(artifacts["rtc_leftover"].shape[1]) if artifacts["rtc_leftover"] is not None else 0
            ),
        }
        profiler.record(
            metrics,
            chunks={
                "noise": artifacts["noise"],
                "uncond_raw": raw_uncond,
                "cond_raw": raw_cond,
                "cfg_raw": raw_cfg,
                "cfg_processed": processed_cfg,
            },
        )
        status = getattr(profiler, "status", None)
        if not isinstance(status, dict):
            return
        dropped_records = int(status.get("dropped_records") or 0)
        writer_error = status.get("writer_error")
        if dropped_records > self._acp_profile_last_dropped or writer_error != self._acp_profile_last_error:
            self._trace_event(
                "acp_profiler_health",
                request_id=self._request_id(observation_t),
                observation_timestep=observation_t.get_timestep(),
                **status,
            )
            self._acp_profile_last_dropped = dropped_records
            self._acp_profile_last_error = writer_error
        if writer_error is not None:
            # Stop paying GPU/statistics overhead once persistence has failed.
            # The profiler object is retained so shutdown can still drain it.
            self._acp_profile_disabled = True
            self.logger.warning("Disabling ACP profiling after writer failure: %s", writer_error)

    def _predict_action_chunk(self, observation_t: TimedObservation) -> list[TimedAction] | RemoteActionChunk:
        """Predict an action chunk based on an observation.

        Pipeline:
        1. Convert raw observation to LeRobot format
        2. Apply preprocessor (tokenization, normalization, batching, device placement)
        3. Run policy inference to get action chunk
        4. Apply postprocessor (unnormalization, device movement)
        5. Convert to TimedAction list
        """
        memory_before = peak_memory = None
        profile_device = torch.device(self.device) if self.device is not None else None
        if (
            self.acp_profiler is not None
            and not self._acp_profile_disabled
            and profile_device is not None
            and profile_device.type == "cuda"
        ):
            torch.cuda.synchronize(profile_device)
            torch.cuda.reset_peak_memory_stats(profile_device)
            memory_before = torch.cuda.memory_allocated(profile_device)

        pipeline_start = time.perf_counter()

        """1. Prepare observation"""
        start_prepare = time.perf_counter()
        start_prepare_cpu = time.thread_time()
        observation: Observation = raw_observation_to_observation(
            observation_t.get_observation(),
            self.lerobot_features,
            self.policy_image_features,
        )
        prepare_cpu_time = time.thread_time() - start_prepare_cpu
        prepare_time = time.perf_counter() - start_prepare

        """2. Apply preprocessor"""
        start_preprocess = time.perf_counter()
        start_preprocess_cpu = time.thread_time()
        if self.batched_cfg_enabled:
            observation = self._prepare_batched_cfg_observation(observation)
        else:
            observation = self.preprocessor(observation)
        self.last_processed_obs: TimedObservation = observation_t
        preprocessing_cpu_time = time.thread_time() - start_preprocess_cpu
        preprocessing_time = time.perf_counter() - start_preprocess

        """3. Get action chunk"""
        start_inference = time.perf_counter()
        acp_artifacts: dict[str, Any] | None = None
        cuda_latency_ms = None
        if self.batched_cfg_enabled:
            rtc_metadata = getattr(observation_t, "rtc_metadata", None)
            if rtc_metadata is not None and not isinstance(rtc_metadata, RTCInferenceMetadata):
                raise TypeError("TimedObservation.rtc_metadata must be RTCInferenceMetadata or None.")
            if self.rtc_enabled and (rtc_metadata is None or not rtc_metadata.request_id):
                raise ValueError("RTC-CFG observations require non-empty request metadata and request_id.")
            action_tensor, acp_artifacts, cuda_latency_ms = self._get_batched_cfg_action_chunk(
                observation,
                rtc_metadata=rtc_metadata,
            )
        else:
            action_tensor = self._get_action_chunk(observation)
        inference_time = time.perf_counter() - start_inference
        self.logger.info(f"Policy inference took {inference_time:.4f}s, action shape: {action_tensor.shape}")

        """4. Apply postprocessor"""
        start_postprocess = time.perf_counter()
        processed_action_tensor = self._postprocess_action_chunk(action_tensor)
        if processed_action_tensor.ndim != 3 or processed_action_tensor.shape[0] != 1:
            raise ValueError(
                "PolicyServer expects one postprocessed action chunk [1, T, A], "
                f"got {tuple(processed_action_tensor.shape)}."
            )

        action_tensor = processed_action_tensor.squeeze(0).detach().cpu()
        self.logger.debug(f"Postprocessed action shape: {action_tensor.shape}")
        postprocess_stops = time.perf_counter()
        postprocessing_time = postprocess_stops - start_postprocess

        if (
            self.acp_profiler is not None
            and not self._acp_profile_disabled
            and profile_device is not None
            and profile_device.type == "cuda"
        ):
            peak_memory = torch.cuda.max_memory_allocated(profile_device)

        """5. Convert to TimedAction list"""
        action_chunk = self._time_action_chunk(
            observation_t.get_timestamp(), list(action_tensor), observation_t.get_timestep()
        )
        wall_latency_s = postprocess_stops - pipeline_start

        rtc_metadata = getattr(observation_t, "rtc_metadata", None)
        requested_leftover = (
            rtc_metadata.prev_chunk_left_over if isinstance(rtc_metadata, RTCInferenceMetadata) else None
        )
        raw_cfg = acp_artifacts["cfg_raw"] if acp_artifacts is not None else None
        effective_leftover = acp_artifacts["rtc_leftover"] if acp_artifacts is not None else None
        self._trace_event(
            "inference_completed",
            request_id=self._request_id(observation_t),
            observation_timestep=observation_t.get_timestep(),
            rtc_enabled=self.rtc_enabled,
            rtc_applied=effective_leftover is not None,
            requested_leftover_steps=(
                int(requested_leftover.shape[0])
                if isinstance(requested_leftover, torch.Tensor) and requested_leftover.ndim >= 1
                else 0
            ),
            effective_leftover_steps=(
                int(effective_leftover.shape[1])
                if isinstance(effective_leftover, torch.Tensor) and effective_leftover.ndim == 3
                else 0
            ),
            effective_inference_delay=(
                acp_artifacts["rtc_inference_delay"] if acp_artifacts is not None else None
            ),
            effective_execution_horizon=(
                acp_artifacts["rtc_execution_horizon"] if acp_artifacts is not None else None
            ),
            cfg_beta=self.acp_inference.cfg_beta if self.batched_cfg_enabled else None,
            prepare_ms=prepare_time * 1000,
            prepare_calling_thread_cpu_ms=prepare_cpu_time * 1000,
            prepare_non_calling_thread_or_wait_ms=max(prepare_time - prepare_cpu_time, 0) * 1000,
            preprocess_ms=preprocessing_time * 1000,
            preprocess_calling_thread_cpu_ms=preprocessing_cpu_time * 1000,
            preprocess_non_calling_thread_or_wait_ms=(
                max(preprocessing_time - preprocessing_cpu_time, 0) * 1000
            ),
            model_and_cfg_ms=inference_time * 1000,
            postprocess_and_d2h_ms=postprocessing_time * 1000,
            wall_ms=wall_latency_s * 1000,
            cuda_latency_ms=cuda_latency_ms,
            raw_actions=tensor_metadata(raw_cfg),
            processed_actions=tensor_metadata(action_tensor),
        )

        if self.batched_cfg_enabled:
            if acp_artifacts is None:
                raise RuntimeError("Missing ACP-CFG inference artifacts.")
            try:
                self._record_acp_profile(
                    observation_t=observation_t,
                    artifacts=acp_artifacts,
                    processed_cfg=processed_action_tensor,
                    timings_s={
                        "prepare": prepare_time,
                        "prepare_cpu": prepare_cpu_time,
                        "preprocess": preprocessing_time,
                        "preprocess_cpu": preprocessing_cpu_time,
                        "inference": inference_time,
                        "postprocess": postprocessing_time,
                    },
                    wall_latency_s=wall_latency_s,
                    cuda_latency_ms=cuda_latency_ms,
                    memory_before=memory_before,
                    peak_memory=peak_memory,
                )
            except Exception as exc:
                # Profiling is diagnostic-only. A reduction/serialization bug
                # must never turn a valid robot action into an INTERNAL RPC.
                self._acp_profile_disabled = True
                self._trace_event(
                    "acp_profiler_record_failed",
                    request_id=self._request_id(observation_t),
                    observation_timestep=observation_t.get_timestep(),
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
                self.logger.warning(
                    "Disabling ACP profiling after a record error; action inference will continue.",
                    exc_info=True,
                )
            self._acp_chunk_index += 1

        self.logger.info(
            f"Observation {observation_t.get_timestep()} | Total time: {1000 * wall_latency_s:.2f}ms"
        )

        self.logger.debug(
            f"Observation {observation_t.get_timestep()} | "
            f"Prepare time: {1000 * prepare_time:.2f}ms | "
            f"Preprocessing time: {1000 * preprocessing_time:.2f}ms | "
            f"Inference time: {1000 * inference_time:.2f}ms | "
            f"Postprocessing time: {1000 * postprocessing_time:.2f}ms | "
            f"Total time: {1000 * wall_latency_s:.2f}ms"
        )

        if self.return_remote_action_chunk:
            if acp_artifacts is None:
                raise RuntimeError("Raw ACP-CFG actions are unavailable for a protocol-v2 response.")
            rtc_metadata = getattr(observation_t, "rtc_metadata", None)
            request_id = (
                rtc_metadata.request_id
                if isinstance(rtc_metadata, RTCInferenceMetadata) and rtc_metadata.request_id
                else f"obs-{observation_t.get_timestep()}"
            )
            raw_actions = acp_artifacts["cfg_raw"].squeeze(0).detach().cpu().clone()
            if raw_actions.shape != action_tensor.shape:
                raise ValueError(
                    "Raw and processed action chunks must be aligned before transport, "
                    f"got {tuple(raw_actions.shape)} and {tuple(action_tensor.shape)}."
                )
            return RemoteActionChunk(
                request_id=request_id,
                actions=action_chunk,
                raw_actions=raw_actions,
                observation_timestep=observation_t.get_timestep(),
                rtc_enabled=self.rtc_enabled,
                inference_delay=acp_artifacts["rtc_inference_delay"],
                execution_horizon=acp_artifacts["rtc_execution_horizon"],
            )

        return action_chunk

    def stop(self):
        """Stop the server after quiescing inference and draining diagnostics."""
        with self._stop_lock:
            if self._stopped:
                return
            self._stopped = True

        self._trace_event("server_stopping")
        self._reset_server(reason="server_stop")
        self.logger.info("Server stopping...")

        quiesce_starts = time.perf_counter()
        with self._inference_lock:
            pass
        self._trace_event(
            "server_inference_quiesced",
            wait_ms=(time.perf_counter() - quiesce_starts) * 1000,
        )

        self._close_all_acp_profilers(reason="server_stop")
        self._trace_event("server_stopped")
        trace = getattr(self, "_rtc_trace", None)
        if trace is not None:
            trace_status = trace.close(timeout_s=5.0)
            if isinstance(trace_status, dict) and (
                trace_status.get("writer_alive", False) or trace_status.get("pending_events", 0) > 0
            ):
                self.logger.warning("RTC trace did not fully drain at shutdown: %s", trace_status)


@draccus.wrap()
def serve(cfg: PolicyServerConfig):
    """Start the PolicyServer with the given configuration.

    Args:
        config: PolicyServerConfig instance. If None, uses default configuration.
    """
    logging.info(pformat(asdict(cfg)))

    # Create the server instance first
    policy_server = PolicyServer(cfg)

    # Setup and start gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")

    policy_server.logger.info(f"PolicyServer started on {cfg.host}:{cfg.port}")
    server.start()
    policy_server._trace_event("server_listening", host=cfg.host, port=cfg.port)

    try:
        server.wait_for_termination()
    finally:
        # Stop accepting new RPCs before invalidating/waiting for any forward
        # already in progress. Both calls are idempotent.
        server.stop(grace=0)
        policy_server.stop()
        policy_server.logger.info("Server terminated")


if __name__ == "__main__":
    serve()
