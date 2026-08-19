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
import pickle  # nosec
import threading
import time
from concurrent import futures
from contextlib import suppress
from dataclasses import asdict
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
from lerobot.transport.utils import receive_bytes_in_chunks

from .configs import PolicyServerConfig
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
        self._acp_chunk_index = 0
        self._client_protocol_version = 1
        self._client_return_raw_actions = False
        self._client_rtc_enabled = False

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    @property
    def policy_image_features(self):
        return self.policy.config.image_features

    @property
    def batched_cfg_enabled(self) -> bool:
        acp = self.config.acp_inference
        return acp.enable and acp.use_cfg and acp.batched_cfg

    @property
    def rtc_enabled(self) -> bool:
        return self.batched_cfg_enabled and self.config.acp_inference.rtc.enabled

    @property
    def return_remote_action_chunk(self) -> bool:
        return (
            self.batched_cfg_enabled
            and self._client_protocol_version >= 2
            and self._client_return_raw_actions
        )

    def _validate_client_rtc_contract(self, policy_specs: RemotePolicyConfig) -> None:
        """Fail setup early if the robot and H200 would use different RTC/CFG parameters."""
        if self._client_rtc_enabled and not self.rtc_enabled:
            raise ValueError(
                "The client requested RTC, but the server was not started with "
                "`acp_inference.rtc.enabled=true`."
            )
        if not self.rtc_enabled:
            return

        rtc = self.config.acp_inference.rtc
        expected = {
            "rtc_inference_delay": rtc.inference_delay,
            "rtc_execution_horizon": rtc.execution_horizon,
            "rtc_max_guidance_weight": rtc.max_guidance_weight,
            "rtc_prefix_attention_schedule": rtc.prefix_attention_schedule,
            "rtc_cfg_beta": self.config.acp_inference.cfg_beta,
        }
        mismatches = []
        for field_name, expected_value in expected.items():
            actual_value = getattr(policy_specs, field_name, None)
            if field_name == "rtc_prefix_attention_schedule" and isinstance(actual_value, str):
                with suppress(ValueError):
                    actual_value = RTCAttentionSchedule(actual_value.upper())
            if actual_value != expected_value:
                mismatches.append(f"{field_name}: client={actual_value!r}, server={expected_value!r}")
        if mismatches:
            raise ValueError("RTC client/server parameter mismatch: " + "; ".join(mismatches))

    def _configure_policy_rtc(self) -> None:
        """Install an RTCProcessor into an already-loaded Pi0.5 checkpoint."""
        if not self.rtc_enabled:
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
                "RTC-CFG requires a protocol-v2 client with `return_raw_actions=true` "
                "and `rtc_enabled=true`."
            )
        if not hasattr(self.policy, "init_rtc_processor"):
            raise TypeError("The loaded Pi0.5 policy does not expose `init_rtc_processor()`.")

        server_rtc = self.config.acp_inference.rtc
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
            "Server-side RTC enabled (delay=%d, horizon=%d, weight=%.3f, schedule=%s).",
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
        self.acp_profiler = None
        self._acp_chunk_index = 0
        if not self.batched_cfg_enabled:
            return

        acp = self.config.acp_inference
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
            "Server-side batched Pi0.5 ACP-CFG enabled "
            "(beta=%.3f, profiling=%s, output=%s).",
            acp.cfg_beta,
            acp.profile,
            self.acp_profiler.output_dir if self.acp_profiler is not None else "disabled",
        )

    def _reset_server(self) -> None:
        """Flushes server state when new client connects."""
        # only running inference on the latest observation received by the server
        self.shutdown_event.set()
        self.observation_queue = Queue(maxsize=1)

        with self._generation_lock:
            self._server_generation += 1

        with self._predicted_timesteps_lock:
            self._predicted_timesteps = set()

    def Ready(self, request, context):  # noqa: N802
        client_id = context.peer()
        self.logger.info(f"Client {client_id} connected and ready")
        self._reset_server()
        self.shutdown_event.clear()

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

        if policy_specs.policy_type not in SUPPORTED_POLICIES:
            raise ValueError(
                f"Policy type {policy_specs.policy_type} not supported. "
                f"Supported policies: {SUPPORTED_POLICIES}"
            )
        if self.batched_cfg_enabled and policy_specs.policy_type != "pi05":
            raise ValueError("Server-side batched ACP-CFG inference is supported only for Pi0.5.")

        # getattr preserves compatibility with RemotePolicyConfig objects
        # pickled by version-1 clients before these capability fields existed.
        self._client_protocol_version = getattr(policy_specs, "protocol_version", 1)
        self._client_return_raw_actions = getattr(policy_specs, "return_raw_actions", False)
        self._client_rtc_enabled = getattr(policy_specs, "rtc_enabled", False)
        if (
            not isinstance(self._client_protocol_version, int)
            or isinstance(self._client_protocol_version, bool)
            or self._client_protocol_version < 1
        ):
            raise ValueError("Remote policy `protocol_version` must be a positive integer.")
        self._validate_client_rtc_contract(policy_specs)

        self.logger.info(
            f"Receiving policy instructions from {client_id} | "
            f"Policy type: {policy_specs.policy_type} | "
            f"Pretrained name or path: {policy_specs.pretrained_name_or_path} | "
            f"Actions per chunk: {policy_specs.actions_per_chunk} | "
            f"Device: {policy_specs.device}"
        )

        self.device = policy_specs.device
        self.policy_type = policy_specs.policy_type  # act, pi0, etc.
        self.pretrained_name_or_path = policy_specs.pretrained_name_or_path
        self.lerobot_features = policy_specs.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk

        policy_class = get_policy_class(self.policy_type)

        start = time.perf_counter()
        # self.policy = policy_class.from_pretrained(policy_specs.pretrained_name_or_path)
        # self.policy.to(self.device)
        # Pass device parameter to from_pretrained to ensure model is initialized with correct device
        self.policy = policy_class.from_pretrained(
            policy_specs.pretrained_name_or_path,
            device=self.device,
        )
        # Move policy to device (in case device wasn't set during initialization)
        self.policy.to(self.device)

        self._configure_policy_rtc()

        # Load preprocessor and postprocessor, overriding device to match requested device
        device_override = {"device": self.device}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=policy_specs.pretrained_name_or_path,
            preprocessor_overrides={
                "device_processor": device_override,
                "rename_observations_processor": {"rename_map": policy_specs.rename_map},
            },
            postprocessor_overrides={"device_processor": device_override},
        )

        self._configure_acp_profiler()

        end = time.perf_counter()

        self.logger.info(f"Time taken to put policy on {self.device}: {end - start:.4f} seconds")

        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        """Receive observations from the robot client"""
        client_id = context.peer()
        self.logger.debug(f"Receiving observations from {client_id}")
        with self._generation_lock:
            receive_generation = self._server_generation

        receive_time = time.time()  # comparing timestamps so need time.time()
        start_deserialize = time.perf_counter()
        received_bytes = receive_bytes_in_chunks(
            request_iterator, None, self.shutdown_event, self.logger
        )  # blocking call while looping over request_iterator
        timed_observation = pickle.loads(received_bytes)  # nosec
        deserialize_time = time.perf_counter() - start_deserialize

        self.logger.debug(f"Received observation #{timed_observation.get_timestep()}")

        obs_timestep = timed_observation.get_timestep()
        obs_timestamp = timed_observation.get_timestamp()

        # Calculate FPS metrics
        fps_metrics = self.fps_tracker.calculate_fps_metrics(obs_timestamp)

        self.logger.debug(
            f"Received observation #{obs_timestep} | "
            f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "  # fps at which observations are received from client
            f"Target: {fps_metrics['target_fps']:.2f} | "
            f"One-way latency: {(receive_time - obs_timestamp) * 1000:.2f}ms"
        )

        self.logger.debug(
            f"Server timestamp: {receive_time:.6f} | "
            f"Client timestamp: {obs_timestamp:.6f} | "
            f"Deserialization time: {deserialize_time:.6f}s"
        )

        with self._generation_lock:
            stale_generation = receive_generation != self._server_generation
        if stale_generation or not context.is_active():
            self.logger.info("Discarding an observation stream invalidated by reset/cancellation.")
            return services_pb2.Empty()

        if not self._enqueue_observation(
            timed_observation  # wrapping a RawObservation
        ):
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

        # Generate action based on the most recent observation and its timestep
        try:
            getactions_starts = time.perf_counter()
            with self._generation_lock:
                request_generation = self._server_generation
            obs = self.observation_queue.get(timeout=self.config.obs_queue_timeout)
            self.logger.info(
                f"Running inference for observation #{obs.get_timestep()} (must_go: {obs.must_go})"
            )

            with self._predicted_timesteps_lock:
                self._predicted_timesteps.add(obs.get_timestep())

            start_time = time.perf_counter()
            action_chunk = self._run_serialized_inference(obs, request_generation, context)
            inference_time = time.perf_counter() - start_time
            if action_chunk is None:
                return services_pb2.Empty()

            start_time = time.perf_counter()
            actions_bytes = pickle.dumps(action_chunk)  # nosec
            serialize_time = time.perf_counter() - start_time

            # Create and return the action chunk
            actions = services_pb2.Actions(data=actions_bytes)

            self.logger.info(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Total time: {(inference_time + serialize_time) * 1000:.2f}ms"
            )

            self.logger.debug(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Inference time: {inference_time:.2f}s |"
                f"Serialize time: {serialize_time:.2f}s |"
                f"Total time: {inference_time + serialize_time:.2f}s"
            )

            time.sleep(
                max(0, self.config.inference_latency - max(0, time.perf_counter() - getactions_starts))
            )  # sleep controls inference latency

            return actions

        except Empty:  # no observation added to queue in obs_queue_timeout
            return services_pb2.Empty()

        except Exception as e:
            self.logger.exception("Policy inference failed while serving GetActions")
            context.abort(grpc.StatusCode.INTERNAL, f"Policy inference failed: {e}")

    def _run_serialized_inference(
        self,
        obs: TimedObservation,
        request_generation: int,
        context,
    ) -> list[TimedAction] | RemoteActionChunk | None:
        """Run at most one model forward and discard work invalidated before it starts."""
        with self._inference_lock:
            with self._generation_lock:
                stale_generation = request_generation != self._server_generation
            is_active = getattr(context, "is_active", None)
            cancelled = callable(is_active) and not is_active()
            if stale_generation or cancelled:
                self.logger.info(
                    "Skipping invalidated inference request (stale_generation=%s, cancelled=%s).",
                    stale_generation,
                    cancelled,
                )
                return None
            return self._predict_action_chunk(obs)

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
                _ = self.observation_queue.get_nowait()
                self.logger.debug("Observation queue was full, removed oldest observation")

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
        elif (
            isinstance(task_value, list)
            and len(task_value) == 1
            and isinstance(task_value[0], str)
        ):
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
        rtc = self.config.acp_inference.rtc
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
        if (
            not isinstance(requested_delay, int)
            or isinstance(requested_delay, bool)
            or requested_delay < 0
        ):
            raise ValueError("RTC `inference_delay` must be a non-negative integer.")
        if (
            not isinstance(requested_horizon, int)
            or isinstance(requested_horizon, bool)
            or requested_horizon <= requested_delay
        ):
            raise ValueError("RTC `execution_horizon` must be an integer greater than inference delay.")
        if requested_delay != rtc.inference_delay or requested_horizon != rtc.execution_horizon:
            raise ValueError(
                "Per-request RTC parameters must match the negotiated server configuration: "
                f"client delay/horizon={requested_delay}/{requested_horizon}, "
                f"server={rtc.inference_delay}/{rtc.execution_horizon}."
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
            raise ValueError(
                f"RTC request leftover must have shape [L, A], got {tuple(leftover.shape)}."
            )
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
        if self.acp_profiler is not None and device.type == "cuda":
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
            torch.cuda.synchronize(device)
            cuda_latency_ms = cuda_start.elapsed_time(cuda_end)
        else:
            cuda_latency_ms = None

        if raw_chunks.ndim != 3 or raw_chunks.shape[0] != 2:
            raise ValueError(
                "Batched ACP-CFG requires Pi0.5 to return [2, T, A], "
                f"got {tuple(raw_chunks.shape)}."
            )

        raw_uncond = raw_chunks[0:1]
        raw_cond = raw_chunks[1:2]
        raw_cfg = raw_uncond + self.config.acp_inference.cfg_beta * (raw_cond - raw_uncond)
        if not bool(torch.isfinite(raw_cfg).all()):
            raise FloatingPointError("Batched ACP-CFG produced a non-finite raw action chunk.")

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
                    "Batched ACP-CFG postprocessing must preserve [1, T, A], "
                    f"got {tuple(processed.shape)}."
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
        if self.acp_profiler is None:
            return

        raw_uncond = artifacts["uncond_raw"]
        raw_cond = artifacts["cond_raw"]
        raw_cfg = artifacts["cfg_raw"]
        shared_noise = artifacts["shared_noise"]
        branch_delta = raw_cond - raw_uncond
        acp = self.config.acp_inference

        metrics = {
            "chunk_index": self._acp_chunk_index,
            "observation_timestep": observation_t.get_timestep(),
            "policy_type": self.policy_type,
            "policy_checkpoint": self.pretrained_name_or_path,
            "device": str(self.device),
            "dtype": str(raw_cfg.dtype),
            "batch_size": 2,
            "cfg_beta": acp.cfg_beta,
            "fps": self.config.fps,
            "model_chunk_size": int(self.policy.config.chunk_size),
            "execution_steps": int(processed_cfg.shape[1]),
            "action_dim": int(processed_cfg.shape[2]),
            "wall_latency_s": wall_latency_s,
            "prepare_latency_ms": timings_s["prepare"] * 1000,
            "preprocess_latency_ms": timings_s["preprocess"] * 1000,
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
            "cond_uncond_delta_l2_mean": torch.linalg.vector_norm(
                branch_delta, dim=-1
            ).mean().item(),
            "rtc_enabled": self.rtc_enabled,
            "rtc_inference_delay": artifacts["rtc_inference_delay"],
            "rtc_execution_horizon": artifacts["rtc_execution_horizon"],
            "rtc_leftover_steps": (
                int(artifacts["rtc_leftover"].shape[1])
                if artifacts["rtc_leftover"] is not None
                else 0
            ),
        }
        self.acp_profiler.record(
            metrics,
            chunks={
                "noise": artifacts["noise"],
                "uncond_raw": raw_uncond,
                "cond_raw": raw_cond,
                "cfg_raw": raw_cfg,
                "cfg_processed": processed_cfg,
            },
        )

    def _predict_action_chunk(
        self, observation_t: TimedObservation
    ) -> list[TimedAction] | RemoteActionChunk:
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
        if self.acp_profiler is not None and profile_device is not None and profile_device.type == "cuda":
            torch.cuda.synchronize(profile_device)
            torch.cuda.reset_peak_memory_stats(profile_device)
            memory_before = torch.cuda.memory_allocated(profile_device)

        pipeline_start = time.perf_counter()

        """1. Prepare observation"""
        start_prepare = time.perf_counter()
        observation: Observation = raw_observation_to_observation(
            observation_t.get_observation(),
            self.lerobot_features,
            self.policy_image_features,
        )
        prepare_time = time.perf_counter() - start_prepare

        """2. Apply preprocessor"""
        start_preprocess = time.perf_counter()
        if self.batched_cfg_enabled:
            observation = self._prepare_batched_cfg_observation(observation)
        else:
            observation = self.preprocessor(observation)
        self.last_processed_obs: TimedObservation = observation_t
        preprocessing_time = time.perf_counter() - start_preprocess

        """3. Get action chunk"""
        start_inference = time.perf_counter()
        acp_artifacts: dict[str, Any] | None = None
        cuda_latency_ms = None
        if self.batched_cfg_enabled:
            rtc_metadata = getattr(observation_t, "rtc_metadata", None)
            if rtc_metadata is not None and not isinstance(rtc_metadata, RTCInferenceMetadata):
                raise TypeError("TimedObservation.rtc_metadata must be RTCInferenceMetadata or None.")
            if self.rtc_enabled and (
                rtc_metadata is None or not rtc_metadata.request_id
            ):
                raise ValueError("RTC-CFG observations require non-empty request metadata and request_id.")
            action_tensor, acp_artifacts, cuda_latency_ms = self._get_batched_cfg_action_chunk(
                observation,
                rtc_metadata=rtc_metadata,
            )
        else:
            action_tensor = self._get_action_chunk(observation)
        inference_time = time.perf_counter() - start_inference
        self.logger.info(
            f"Policy inference took {inference_time:.4f}s, action shape: {action_tensor.shape}"
        )

        # ========== 调试信息开始 ==========
        print(f"[DEBUG] action_tensor.shape = {action_tensor.shape}")   # 预期 (B, chunk_size, action_dim)
        # print(f"[DEBUG] self.policy.config.action_dim = {self.policy.config.action_dim}")
        # 如果 postprocessor 内部有 mean, std，也可打印（需要访问内部属性）
        if hasattr(self.postprocessor, 'steps'):
            for step in self.postprocessor.steps:
                if hasattr(step, 'mean') and hasattr(step, 'std'):
                    print(f"[DEBUG] step {step.__class__.__name__} mean len={len(step.mean)} std len={len(step.std)}")
        # ========== 调试信息结束 ==========

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

        if self.acp_profiler is not None and profile_device is not None and profile_device.type == "cuda":
            peak_memory = torch.cuda.max_memory_allocated(profile_device)

        """5. Convert to TimedAction list"""
        action_chunk = self._time_action_chunk(
            observation_t.get_timestamp(), list(action_tensor), observation_t.get_timestep()
        )
        wall_latency_s = postprocess_stops - pipeline_start

        if self.batched_cfg_enabled:
            if acp_artifacts is None:
                raise RuntimeError("Missing ACP-CFG inference artifacts.")
            self._record_acp_profile(
                observation_t=observation_t,
                artifacts=acp_artifacts,
                processed_cfg=processed_action_tensor,
                timings_s={
                    "prepare": prepare_time,
                    "preprocess": preprocessing_time,
                    "inference": inference_time,
                    "postprocess": postprocessing_time,
                },
                wall_latency_s=wall_latency_s,
                cuda_latency_ms=cuda_latency_ms,
                memory_before=memory_before,
                peak_memory=peak_memory,
            )
            self._acp_chunk_index += 1

        self.logger.info(
            f"Observation {observation_t.get_timestep()} | "
            f"Total time: {1000 * wall_latency_s:.2f}ms"
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
        """Stop the server"""
        self._reset_server()
        self.logger.info("Server stopping...")


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

    server.wait_for_termination()

    policy_server.logger.info("Server terminated")


if __name__ == "__main__":
    serve()
