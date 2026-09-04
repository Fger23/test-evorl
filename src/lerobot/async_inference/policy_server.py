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
from dataclasses import asdict
from pprint import pformat
from queue import Empty, Queue
from typing import Any

import draccus
import grpc
import torch

from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
)
from lerobot.rl.acp_tags import ACP_TAG_KEY, build_acp_tagged_task
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import receive_bytes_in_chunks
from lerobot.utils.constants import ACTION

from .configs import PolicyServerConfig
from .constants import SUPPORTED_POLICIES
from .helpers import (
    FPSTracker,
    Observation,
    RemoteActionChunk,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    TrainingTimeRTCMetadata,
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

        self._predicted_timesteps_lock = threading.Lock()
        self._predicted_timesteps = set()
        self._inference_lock = threading.Lock()

        self.last_processed_obs = None

        # Attributes will be set by SendPolicyInstructions
        self.device = None
        self.policy_type = None
        self.lerobot_features = None
        self.actions_per_chunk = None
        self.policy = None
        self.preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None
        self.postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None
        self._client_protocol_version = 1
        self._client_return_raw_actions = False
        self._training_time_rtc = False
        self._rtc_prefix_steps = 0
        self._acp_positive_prompt = False
        self._use_cfg = False
        self._cfg_beta = 1.0

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    @property
    def policy_image_features(self):
        return self.policy.config.image_features

    def _reset_server(self) -> None:
        """Flushes server state when new client connects."""
        # only running inference on the latest observation received by the server
        self.shutdown_event.set()
        self.observation_queue = Queue(maxsize=1)

        with self._predicted_timesteps_lock:
            self._predicted_timesteps = set()
        self.last_processed_obs = None
        self.fps_tracker.reset()

    def _validate_training_time_rtc_setup(self, policy_specs: RemotePolicyConfig) -> None:
        """Validate the single-branch ACP plus learned-RTC transport contract."""
        training_time_rtc = getattr(policy_specs, "training_time_rtc", False)
        if not isinstance(training_time_rtc, bool):
            raise TypeError("`training_time_rtc` must be true or false.")
        if not training_time_rtc:
            return

        if policy_specs.policy_type != "pi05":
            raise ValueError("Training-Time RTC inference currently supports only `policy_type=pi05`.")
        if getattr(policy_specs, "protocol_version", 1) < 2 or not getattr(
            policy_specs, "return_raw_actions", False
        ):
            raise ValueError(
                "Training-Time RTC requires `protocol_version>=2` and `return_raw_actions=true`."
            )
        if not getattr(policy_specs, "acp_positive_prompt", False):
            raise ValueError("This Training-Time RTC entry point requires the positive ACP prompt branch.")
        if getattr(policy_specs, "use_cfg", False):
            raise ValueError(
                "Training-Time RTC uses one positive ACP branch; uncon/cond CFG must stay disabled."
            )
        cfg_beta = getattr(policy_specs, "cfg_beta", 1.0)
        if (
            isinstance(cfg_beta, bool)
            or not isinstance(cfg_beta, int | float)
            or not math.isfinite(float(cfg_beta))
            or float(cfg_beta) != 1.0
        ):
            raise ValueError("Single-branch ACP inference requires `cfg_beta=1`.")

        model_chunk_size = int(self.policy.config.chunk_size)
        trained_max_delay = int(getattr(self.policy.config, "rtc_training_max_delay", 0))
        configured_prefix = getattr(policy_specs, "rtc_prefix_steps", 0)
        if (
            not isinstance(configured_prefix, int)
            or isinstance(configured_prefix, bool)
            or configured_prefix <= 0
        ):
            raise ValueError("`rtc_prefix_steps` must be a positive integer.")
        if trained_max_delay <= 0:
            raise ValueError(
                "The loaded PI0.5 checkpoint was not trained for RTC: "
                "`policy.rtc_training_max_delay` is zero."
            )
        if configured_prefix > trained_max_delay:
            raise ValueError(
                f"Requested RTC prefix d={configured_prefix} exceeds the checkpoint training limit "
                f"D={trained_max_delay}."
            )
        if configured_prefix >= model_chunk_size:
            raise ValueError(
                f"RTC prefix d={configured_prefix} must be smaller than model chunk size "
                f"H={model_chunk_size}."
            )
        if (
            not isinstance(policy_specs.actions_per_chunk, int)
            or isinstance(policy_specs.actions_per_chunk, bool)
            or policy_specs.actions_per_chunk <= configured_prefix
            or policy_specs.actions_per_chunk > model_chunk_size
        ):
            raise ValueError(
                "`actions_per_chunk` must satisfy rtc_prefix_steps < actions_per_chunk "
                f"<= model chunk size ({model_chunk_size})."
            )
        legacy_rtc = getattr(self.policy.config, "rtc_config", None)
        if legacy_rtc is not None and getattr(legacy_rtc, "enabled", False):
            raise ValueError(
                "The checkpoint enables legacy gradient-guided RTC. Disable `rtc_config.enabled` "
                "when using learned Training-Time RTC."
            )

    def Ready(self, request, context):  # noqa: N802
        client_id = context.peer()
        self.logger.info(f"Client {client_id} connected and ready")
        # A new Ready call takes ownership of this single-client server only
        # after any older model call has returned.
        with self._inference_lock:
            self._reset_server()
            reset_policy = getattr(self.policy, "reset", None)
            if callable(reset_policy):
                reset_policy()
            self.shutdown_event.clear()

        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        """Load and install a policy session without exposing partial state."""
        state_names = (
            "device",
            "policy_type",
            "lerobot_features",
            "actions_per_chunk",
            "policy",
            "preprocessor",
            "postprocessor",
            "_client_protocol_version",
            "_client_return_raw_actions",
            "_training_time_rtc",
            "_rtc_prefix_steps",
            "_acp_positive_prompt",
            "_use_cfg",
            "_cfg_beta",
        )
        with self._inference_lock:
            previous_state = {name: getattr(self, name) for name in state_names}
            try:
                return self._send_policy_instructions(request, context)
            except Exception:
                for name, value in previous_state.items():
                    setattr(self, name, value)
                raise

    def _send_policy_instructions(self, request, context):
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
        training_time_rtc = getattr(policy_specs, "training_time_rtc", False)
        protocol_version = getattr(policy_specs, "protocol_version", 1)
        return_raw_actions = getattr(policy_specs, "return_raw_actions", False)
        acp_positive_prompt = getattr(policy_specs, "acp_positive_prompt", False)
        use_cfg = getattr(policy_specs, "use_cfg", False)
        rtc_prefix_steps = getattr(policy_specs, "rtc_prefix_steps", 0)
        cfg_beta = getattr(policy_specs, "cfg_beta", 1.0)
        if not isinstance(training_time_rtc, bool):
            raise TypeError("`training_time_rtc` must be true or false.")
        if not isinstance(protocol_version, int) or isinstance(protocol_version, bool):
            raise TypeError("`protocol_version` must be an integer.")
        for field_name, value in (
            ("return_raw_actions", return_raw_actions),
            ("acp_positive_prompt", acp_positive_prompt),
            ("use_cfg", use_cfg),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"`{field_name}` must be true or false.")
        if training_time_rtc:
            if protocol_version < 2 or not return_raw_actions:
                raise ValueError(
                    "Training-Time RTC requires `protocol_version>=2` and `return_raw_actions=true`."
                )
            if not acp_positive_prompt or use_cfg:
                raise ValueError("Training-Time RTC requires ACP positive=true and CFG=false.")
            if (
                not isinstance(rtc_prefix_steps, int)
                or isinstance(rtc_prefix_steps, bool)
                or rtc_prefix_steps <= 0
            ):
                raise ValueError("`rtc_prefix_steps` must be a positive integer.")
            if (
                isinstance(cfg_beta, bool)
                or not isinstance(cfg_beta, int | float)
                or not math.isfinite(float(cfg_beta))
                or float(cfg_beta) != 1.0
            ):
                raise ValueError("Single-branch ACP inference requires `cfg_beta=1`.")
        elif (
            protocol_version != 1
            or return_raw_actions
            or acp_positive_prompt
            or use_cfg
            or rtc_prefix_steps != 0
            or cfg_beta != 1.0
        ):
            raise ValueError(
                "Protocol-v2/raw/ACP fields are supported only with `training_time_rtc=true` on this server."
            )

        self.logger.info(
            f"Receiving policy instructions from {client_id} | "
            f"Policy type: {policy_specs.policy_type} | "
            f"Pretrained name or path: {policy_specs.pretrained_name_or_path} | "
            f"Actions per chunk: {policy_specs.actions_per_chunk} | "
            f"Device: {policy_specs.device}"
        )

        self.device = policy_specs.device
        self.policy_type = policy_specs.policy_type  # act, pi0, etc.
        self.lerobot_features = policy_specs.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk
        self._client_protocol_version = protocol_version
        self._client_return_raw_actions = return_raw_actions
        self._training_time_rtc = training_time_rtc
        self._rtc_prefix_steps = rtc_prefix_steps
        self._acp_positive_prompt = acp_positive_prompt
        self._use_cfg = use_cfg
        self._cfg_beta = float(cfg_beta) if training_time_rtc else 1.0

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
        self._validate_training_time_rtc_setup(policy_specs)

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

        end = time.perf_counter()

        self.logger.info(f"Time taken to put policy on {self.device}: {end - start:.4f} seconds")
        if self._training_time_rtc:
            self.logger.info(
                "Training-Time RTC inference ready | ACP=positive | CFG=off | beta=1 | "
                "requested d=%d | trained D=%d | H=%d",
                self._rtc_prefix_steps,
                self.policy.config.rtc_training_max_delay,
                self.policy.config.chunk_size,
            )

        return services_pb2.Empty()

    def _validate_training_time_rtc_metadata(
        self, metadata: TrainingTimeRTCMetadata | None
    ) -> TrainingTimeRTCMetadata:
        if not isinstance(metadata, TrainingTimeRTCMetadata):
            raise TypeError("Training-Time RTC observations require `TrainingTimeRTCMetadata`.")
        if not metadata.request_id:
            raise ValueError("Training-Time RTC `request_id` cannot be empty.")
        if (
            not isinstance(metadata.prefix_steps, int)
            or isinstance(metadata.prefix_steps, bool)
            or metadata.prefix_steps < 0
            or metadata.prefix_steps > self._rtc_prefix_steps
        ):
            raise ValueError(
                "Per-request RTC prefix must satisfy "
                f"0 <= d <= configured d ({self._rtc_prefix_steps}), got {metadata.prefix_steps}."
            )
        if metadata.prefix_steps == 0:
            if metadata.action_prefix is not None:
                raise ValueError("RTC bootstrap d=0 must not include an action prefix.")
            return metadata

        action_prefix = metadata.action_prefix
        if not isinstance(action_prefix, torch.Tensor) or action_prefix.ndim != 2:
            shape = getattr(action_prefix, "shape", None)
            raise ValueError(f"RTC action prefix must have shape [d, A], got {shape}.")
        expected_action_dim = int(self.policy.config.output_features[ACTION].shape[0])
        if action_prefix.shape[0] != metadata.prefix_steps:
            raise ValueError(
                "RTC action prefix length must equal prefix_steps: "
                f"{action_prefix.shape[0]} != {metadata.prefix_steps}."
            )
        if action_prefix.shape[1] != expected_action_dim:
            raise ValueError(
                "RTC normalized prefix action dimension does not match the policy: "
                f"{action_prefix.shape[1]} != {expected_action_dim}."
            )
        if not bool(torch.isfinite(action_prefix).all()):
            raise ValueError("RTC normalized action prefix contains NaN or Inf.")
        return metadata

    def SendObservations(self, request_iterator, context):  # noqa: N802
        """Receive observations from the robot client"""
        client_id = context.peer()
        self.logger.debug(f"Receiving observations from {client_id}")

        receive_time = time.time()  # comparing timestamps so need time.time()
        start_deserialize = time.perf_counter()
        received_bytes = receive_bytes_in_chunks(
            request_iterator, None, self.shutdown_event, self.logger
        )  # blocking call while looping over request_iterator
        timed_observation = pickle.loads(received_bytes)  # nosec
        if not isinstance(timed_observation, TimedObservation):
            raise TypeError(f"Expected TimedObservation, got {type(timed_observation).__name__}.")
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

        # Validate and enqueue against one stable policy-session snapshot.
        with self._inference_lock:
            if self._training_time_rtc:
                if not timed_observation.must_go:
                    raise ValueError("Training-Time RTC observations must set `must_go=true`.")
                self._validate_training_time_rtc_metadata(timed_observation.rtc_metadata)
            enqueued = self._enqueue_observation(timed_observation)

        if not enqueued:
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
            obs = self.observation_queue.get(timeout=self.config.obs_queue_timeout)
            self.logger.info(
                f"Running inference for observation #{obs.get_timestep()} (must_go: {obs.must_go})"
            )

            with self._predicted_timesteps_lock:
                self._predicted_timesteps.add(obs.get_timestep())

            start_time = time.perf_counter()
            with self._inference_lock:
                action_chunk = self._predict_action_chunk(obs)
            inference_time = time.perf_counter() - start_time

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
            self.logger.exception("Policy inference failed")
            if context is not None:
                context.abort(grpc.StatusCode.INTERNAL, str(e))
            raise

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
            # A mandatory observation may intentionally reuse a timestep (for
            # example after an episode reset). Do not let stale deduplication
            # state suppress it or contaminate the next similarity check.
            with self._predicted_timesteps_lock:
                self._predicted_timesteps.discard(obs.get_timestep())
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

    def _get_action_chunk(
        self,
        observation: dict[str, torch.Tensor],
        rtc_metadata: TrainingTimeRTCMetadata | None = None,
    ) -> torch.Tensor:
        """Run one policy forward and retain the normalized action chunk."""
        inference_kwargs: dict[str, Any] = {}
        if self._training_time_rtc:
            rtc_metadata = self._validate_training_time_rtc_metadata(rtc_metadata)
            inference_kwargs = {
                "training_time_rtc": True,
                "rtc_action_prefix": rtc_metadata.action_prefix,
                "inference_delay": rtc_metadata.prefix_steps,
            }

        chunk = self.policy.predict_action_chunk(observation, **inference_kwargs)
        if chunk.ndim == 2:
            chunk = chunk.unsqueeze(0)
        if chunk.ndim != 3 or chunk.shape[0] != 1:
            raise ValueError(
                f"PolicyServer expects one action chunk with shape [1, T, A], got {tuple(chunk.shape)}."
            )
        chunk = chunk[:, : self.actions_per_chunk, :]
        if chunk.shape[1] != self.actions_per_chunk:
            raise ValueError(
                f"Policy returned fewer actions than requested: {chunk.shape[1]} != {self.actions_per_chunk}."
            )
        if not bool(torch.isfinite(chunk).all()):
            raise FloatingPointError("Policy produced a non-finite normalized action chunk.")
        return chunk

    def _postprocess_action_chunk(self, action_tensor: torch.Tensor) -> torch.Tensor:
        """Unnormalize a full chunk while preserving legacy per-step processing."""
        if self.postprocessor is None:
            raise RuntimeError("Policy postprocessor is not initialized.")
        processed_actions = [
            self.postprocessor(action_tensor[:, i, :]) for i in range(action_tensor.shape[1])
        ]
        processed = torch.stack(processed_actions, dim=1)
        if processed.ndim != 3 or processed.shape[0] != 1:
            raise ValueError(
                f"Policy postprocessor must return one aligned [1, T, A] chunk, got {tuple(processed.shape)}."
            )
        if not bool(torch.isfinite(processed).all()):
            raise FloatingPointError("Policy postprocessor produced NaN or Inf actions.")
        return processed

    def _predict_action_chunk(self, observation_t: TimedObservation) -> list[TimedAction] | RemoteActionChunk:
        """Predict an action chunk based on an observation.

        Pipeline:
        1. Convert raw observation to LeRobot format
        2. Apply preprocessor (tokenization, normalization, batching, device placement)
        3. Run policy inference to get action chunk
        4. Apply postprocessor (unnormalization, device movement)
        5. Convert to TimedAction list
        """
        """1. Prepare observation"""
        start_prepare = time.perf_counter()
        raw_observation = dict(observation_t.get_observation())
        if self._training_time_rtc:
            task = raw_observation.get("task")
            if task is not None and not isinstance(task, str):
                raise TypeError(f"ACP task must be a string or None, got {type(task).__name__}.")
            if task is not None and any(
                line.strip().startswith(f"{ACP_TAG_KEY}:") for line in task.splitlines()
            ):
                raise ValueError(
                    "Pass the base task only; the server appends `Advantage: positive` exactly once."
                )
            raw_observation["task"] = build_acp_tagged_task(task, is_positive=True)
        observation: Observation = raw_observation_to_observation(
            raw_observation,
            self.lerobot_features,
            self.policy_image_features,
        )
        prepare_time = time.perf_counter() - start_prepare

        """2. Apply preprocessor"""
        start_preprocess = time.perf_counter()
        observation = self.preprocessor(observation)
        self.last_processed_obs: TimedObservation = observation_t
        preprocessing_time = time.perf_counter() - start_preprocess

        """3. Get action chunk"""
        start_inference = time.perf_counter()
        rtc_metadata = observation_t.rtc_metadata if self._training_time_rtc else None
        if self._training_time_rtc:
            action_tensor = self._get_action_chunk(observation, rtc_metadata=rtc_metadata)
        else:
            # Keep the legacy call signature for existing clients and test doubles.
            action_tensor = self._get_action_chunk(observation)
        inference_time = time.perf_counter() - start_inference
        self.logger.info(f"Policy inference took {inference_time:.4f}s, action shape: {action_tensor.shape}")

        """4. Apply postprocessor"""
        start_postprocess = time.perf_counter()
        raw_action_tensor = action_tensor.squeeze(0).detach().cpu().clone()
        if self._training_time_rtc and rtc_metadata is not None and rtc_metadata.prefix_steps > 0:
            expected_prefix = rtc_metadata.action_prefix
            if expected_prefix is None or not torch.allclose(
                raw_action_tensor[: rtc_metadata.prefix_steps],
                expected_prefix.detach().cpu(),
                rtol=1e-5,
                atol=1e-6,
            ):
                raise RuntimeError("PI0.5 did not hard-clamp the requested normalized RTC clean prefix.")
        processed_action_tensor = self._postprocess_action_chunk(action_tensor).squeeze(0).detach().cpu()
        if raw_action_tensor.shape != processed_action_tensor.shape:
            raise ValueError(
                "Normalized and robot-space action chunks must stay aligned: "
                f"{tuple(raw_action_tensor.shape)} != {tuple(processed_action_tensor.shape)}."
            )
        self.logger.debug(f"Postprocessed action shape: {processed_action_tensor.shape}")

        """5. Convert to TimedAction list"""
        action_chunk = self._time_action_chunk(
            observation_t.get_timestamp(),
            list(processed_action_tensor),
            observation_t.get_timestep(),
        )
        postprocess_stops = time.perf_counter()
        postprocessing_time = postprocess_stops - start_postprocess

        self.logger.info(
            f"Observation {observation_t.get_timestep()} | "
            f"Total time: {1000 * (postprocess_stops - start_prepare):.2f}ms"
        )

        self.logger.debug(
            f"Observation {observation_t.get_timestep()} | "
            f"Prepare time: {1000 * prepare_time:.2f}ms | "
            f"Preprocessing time: {1000 * preprocessing_time:.2f}ms | "
            f"Inference time: {1000 * inference_time:.2f}ms | "
            f"Postprocessing time: {1000 * postprocessing_time:.2f}ms | "
            f"Total time: {1000 * (postprocess_stops - start_prepare):.2f}ms"
        )

        if self._training_time_rtc:
            if rtc_metadata is None:
                raise RuntimeError("Training-Time RTC metadata disappeared during inference.")
            return RemoteActionChunk(
                request_id=rtc_metadata.request_id,
                actions=action_chunk,
                raw_actions=raw_action_tensor,
                observation_timestep=observation_t.get_timestep(),
                training_time_rtc=True,
                prefix_steps=rtc_metadata.prefix_steps,
                acp_positive_prompt=True,
                use_cfg=False,
                cfg_beta=1.0,
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
