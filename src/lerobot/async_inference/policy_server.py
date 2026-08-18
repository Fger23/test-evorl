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
    RemotePolicyConfig,
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

        if self.batched_cfg_enabled and getattr(self.policy.config, "rtc_config", None) is not None:
            raise ValueError(
                "Batched ACP-CFG profiling is a no-RTC baseline; "
                "set the Pi0.5 `rtc_config` to null."
            )

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
            obs = self.observation_queue.get(timeout=self.config.obs_queue_timeout)
            self.logger.info(
                f"Running inference for observation #{obs.get_timestep()} (must_go: {obs.must_go})"
            )

            with self._predicted_timesteps_lock:
                self._predicted_timesteps.add(obs.get_timestep())

            start_time = time.perf_counter()
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
            self.logger.error(f"Error in StreamActions: {e}")

            return services_pb2.Empty()

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

    def _get_batched_cfg_action_chunk(
        self, observation: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], float | None]:
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
        artifacts: dict[str, torch.Tensor],
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

    def _predict_action_chunk(self, observation_t: TimedObservation) -> list[TimedAction]:
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
        acp_artifacts = None
        cuda_latency_ms = None
        if self.batched_cfg_enabled:
            action_tensor, acp_artifacts, cuda_latency_ms = self._get_batched_cfg_action_chunk(
                observation
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
