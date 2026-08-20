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

"""Human-in-loop recording helpers used by `lerobot_record.py`."""

import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import PolicyAction, PolicyProcessorPipeline, RobotAction
from lerobot.rl.acp_tags import build_acp_tagged_task
from lerobot.robots import Robot
from lerobot.teleoperators import Teleoperator
from lerobot.utils.control_utils import predict_action


@dataclass
class ACPInferenceConfig:
    enable: bool = False
    use_cfg: bool = False
    cfg_beta: float = 1.0
    # Run unconditional/conditional prompts as one Pi0.5 batch with shared noise.
    batched_cfg: bool = False
    # Persist per-chunk timing and action diagnostics for choosing RTC parameters later.
    profile: bool = False
    profile_output_dir: str = "outputs/acp_inference_profile"
    # Full tensor dumps add synchronous disk I/O; opt in after the timing-only run is stable.
    profile_save_chunks: bool = False
    profile_run_name: str | None = None
    profile_warmup_chunks: int = 3


POLICY_RUNTIME_STATE_KEYS = ("_action_queue", "_queues", "_prev_mean")

# 状态机
INTERVENTION_STATE_POLICY = 0.0
INTERVENTION_STATE_ACTIVE = 1.0
INTERVENTION_STATE_RELEASE = 2.0


# 生成相同噪声
def _get_torch_rng_state(device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    return cpu_state, cuda_state


# 恢复相同噪声
def _set_torch_rng_state(
    device: torch.device, cpu_state: torch.Tensor, cuda_state: torch.Tensor | None
) -> None:
    torch.set_rng_state(cpu_state)
    if device.type == "cuda" and cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, device)


def _clone_runtime_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, deque):
        return deque((_clone_runtime_value(item) for item in value), maxlen=value.maxlen)
    if isinstance(value, dict):
        return {key: _clone_runtime_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_runtime_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_runtime_value(item) for item in value)
    return deepcopy(value)


def _capture_policy_runtime_state(policy: PreTrainedPolicy) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for key in POLICY_RUNTIME_STATE_KEYS:
        if hasattr(policy, key):
            state[key] = _clone_runtime_value(getattr(policy, key))
    return state


def _restore_policy_runtime_state(policy: PreTrainedPolicy, state: dict[str, Any]) -> None:
    for key, value in state.items():
        setattr(policy, key, _clone_runtime_value(value))


def _predict_policy_action_with_runtime_state(
    *,
    observation_frame: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    use_amp: bool,
    task: str | None,
    robot_type: str | None,
    runtime_state: dict[str, Any],
) -> PolicyAction:
    _restore_policy_runtime_state(policy, runtime_state)
    action = predict_action(
        observation=observation_frame,
        policy=policy,
        device=device,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        use_amp=use_amp,
        task=task,
        robot_type=robot_type,
    )
    runtime_state.clear()
    runtime_state.update(_capture_policy_runtime_state(policy))
    return action


def _prepare_batched_cfg_observation(
    *,
    observation_frame: dict[str, np.ndarray],
    device: torch.device,
    task: str | None,
    robot_type: str | None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
) -> dict[str, Any]:
    """Build a fixed-order [unconditional, conditional] Pi0.5 inference batch."""
    conditional_task = build_acp_tagged_task(task, is_positive=True)
    observation = prepare_observation_for_inference(
        dict(observation_frame),
        device=device,
        task=task,
        robot_type=robot_type,
    )

    for key, value in list(observation.items()):
        if not isinstance(value, torch.Tensor):
            continue
        if value.ndim == 0 or value.shape[0] != 1:
            raise ValueError(
                f"Expected a single observation before CFG batching, got {key} shape={tuple(value.shape)}."
            )
        repeats = (2,) + (1,) * (value.ndim - 1)
        observation[key] = value.repeat(repeats)

    # Row 0 is U and row 1 is C throughout inference and profiling.
    observation["task"] = [task or "", conditional_task]
    observation["robot_type"] = [robot_type or "", robot_type or ""]
    return preprocessor(observation)


class ACPBatchedCFGRuntime:
    """Cache and profile postprocessed actions from one batched Pi0.5 CFG chunk."""

    def __init__(
        self,
        *,
        config: ACPInferenceConfig,
        fps: int,
        profiler: Any | None = None,
    ) -> None:
        if not config.enable or not config.use_cfg or not config.batched_cfg:
            raise ValueError("ACPBatchedCFGRuntime requires enabled batched CFG inference.")
        self.config = config
        self.fps = fps
        self._action_queue: deque[PolicyAction] = deque()
        self._chunk_index = 0

        if profiler is None and config.profile:
            from lerobot.scripts.acp_inference_profile import ACPInferenceProfiler

            profiler = ACPInferenceProfiler(
                output_root=config.profile_output_dir,
                fps=fps,
                cfg_beta=config.cfg_beta,
                save_chunks=config.profile_save_chunks,
                run_name=config.profile_run_name,
            )
        self.profiler = profiler

    @property
    def queue_size(self) -> int:
        return len(self._action_queue)

    def reset(self) -> None:
        """Discard stale actions while retaining all profiler records."""
        self._action_queue.clear()

    def close(self) -> None:
        """Flush the optional asynchronous profiler during clean shutdown."""
        close = getattr(self.profiler, "close", None)
        if callable(close):
            close()

    def predict_action(
        self,
        *,
        observation_frame: dict[str, np.ndarray],
        policy: PreTrainedPolicy,
        device: torch.device,
        preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
        postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
        use_amp: bool,
        task: str | None,
        robot_type: str | None,
    ) -> PolicyAction:
        if self._action_queue:
            return self._action_queue.popleft()

        if getattr(policy.config, "rtc_config", None) is not None:
            raise ValueError(
                "Batched CFG profiling is a no-RTC baseline; set `policy.rtc_config=null` for this run."
            )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            memory_before = torch.cuda.memory_allocated(device)
        else:
            memory_before = None

        start_t = time.perf_counter()
        cuda_start = cuda_end = None
        with (
            torch.inference_mode(),
            torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
        ):
            batch = _prepare_batched_cfg_observation(
                observation_frame=observation_frame,
                device=device,
                task=task,
                robot_type=robot_type,
                preprocessor=preprocessor,
            )

            chunk_size = int(policy.config.chunk_size)
            max_action_dim = int(policy.config.max_action_dim)
            # Match Pi0.5's sample_noise implementation without depending on policy wrapper internals.
            base_noise = torch.normal(
                mean=0.0,
                std=1.0,
                size=(1, chunk_size, max_action_dim),
                dtype=torch.float32,
                device=device,
            )
            shared_noise = base_noise.repeat(2, 1, 1)

            if device.type == "cuda":
                cuda_start = torch.cuda.Event(enable_timing=True)
                cuda_end = torch.cuda.Event(enable_timing=True)
                stream = torch.cuda.current_stream(device)
                cuda_start.record(stream)

            raw_chunks = policy.predict_action_chunk(batch, noise=shared_noise)

            if cuda_end is not None:
                cuda_end.record(torch.cuda.current_stream(device))

            if raw_chunks.ndim != 3 or raw_chunks.shape[0] != 2:
                raise ValueError(
                    "Batched CFG requires policy.predict_action_chunk to return [2, T, A], "
                    f"got {tuple(raw_chunks.shape)}."
                )

            raw_uncond = raw_chunks[0:1]
            raw_cond = raw_chunks[1:2]
            raw_cfg = raw_uncond + self.config.cfg_beta * (raw_cond - raw_uncond)
            processed_cfg = postprocessor(raw_cfg)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            cuda_latency_ms = cuda_start.elapsed_time(cuda_end)
            peak_memory = torch.cuda.max_memory_allocated(device)
        else:
            cuda_latency_ms = None
            peak_memory = None
        wall_latency_s = time.perf_counter() - start_t

        if processed_cfg.ndim != 3 or processed_cfg.shape[0] != 1:
            raise ValueError(
                f"CFG postprocessor must preserve a [1, T, A] action chunk, got {tuple(processed_cfg.shape)}."
            )

        execution_steps = min(int(policy.config.n_action_steps), processed_cfg.shape[1])
        if execution_steps <= 0:
            raise ValueError("Pi0.5 `n_action_steps` must be positive for batched CFG inference.")
        self._action_queue.extend(
            action.detach() for action in processed_cfg[:, :execution_steps].transpose(0, 1)
        )

        shared_noise_error = (shared_noise[0] - shared_noise[1]).abs().max().item()
        branch_delta = raw_cond - raw_uncond
        metrics = {
            "chunk_index": self._chunk_index,
            "device": str(device),
            "dtype": str(raw_chunks.dtype),
            "batch_size": 2,
            "cfg_beta": self.config.cfg_beta,
            "fps": self.fps,
            "chunk_size": int(raw_chunks.shape[1]),
            "action_dim": int(raw_chunks.shape[2]),
            "execution_steps": execution_steps,
            "queue_depth_before": 0,
            "queue_depth_after": execution_steps,
            "wall_latency_s": wall_latency_s,
            "cuda_latency_ms": cuda_latency_ms,
            "cuda_memory_allocated_before_bytes": memory_before,
            "cuda_peak_memory_bytes": peak_memory,
            "shared_noise_max_abs_diff": shared_noise_error,
            "warmup": self._chunk_index < self.config.profile_warmup_chunks,
            "raw_cfg_min": raw_cfg.min().item(),
            "raw_cfg_max": raw_cfg.max().item(),
            "raw_cfg_mean": raw_cfg.mean().item(),
            "raw_cfg_std": raw_cfg.std(unbiased=False).item(),
            "raw_cfg_out_of_range_ratio": (raw_cfg.abs() > 1.0).float().mean().item(),
            "cond_uncond_delta_l2_mean": torch.linalg.vector_norm(branch_delta, dim=-1).mean().item(),
        }
        if self.profiler is not None:
            self.profiler.record(
                metrics,
                chunks={
                    "noise": base_noise,
                    "uncond_raw": raw_uncond,
                    "cond_raw": raw_cond,
                    "cfg_raw": raw_cfg,
                    "cfg_processed": processed_cfg,
                },
            )

        self._chunk_index += 1
        return self._action_queue.popleft()


def _predict_policy_action_with_acp_inference(
    *,
    observation_frame: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    use_amp: bool,
    task: str | None,
    robot_type: str | None,
    acp_inference: ACPInferenceConfig,
    cond_runtime_state: dict[str, Any] | None = None,
    uncond_runtime_state: dict[str, Any] | None = None,
    batched_cfg_runtime: ACPBatchedCFGRuntime | None = None,
) -> PolicyAction:
    if not acp_inference.enable:
        return predict_action(
            observation=observation_frame,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=use_amp,
            task=task,
            robot_type=robot_type,
        )

    conditional_task = build_acp_tagged_task(task, is_positive=True)
    if not acp_inference.use_cfg:
        return predict_action(
            observation=observation_frame,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=use_amp,
            task=conditional_task,
            robot_type=robot_type,
        )

    if acp_inference.batched_cfg:
        if batched_cfg_runtime is None:
            raise ValueError("Batched CFG inference requires an ACPBatchedCFGRuntime.")
        return batched_cfg_runtime.predict_action(
            observation_frame=observation_frame,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=use_amp,
            task=task,
            robot_type=robot_type,
        )

    if cond_runtime_state is None or uncond_runtime_state is None:
        raise ValueError("CFG inference requires cond/uncond runtime states.")

    cpu_state, cuda_state = _get_torch_rng_state(device)
    action_cond = _predict_policy_action_with_runtime_state(
        observation_frame=observation_frame,
        policy=policy,
        device=device,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        use_amp=use_amp,
        task=conditional_task,
        robot_type=robot_type,
        runtime_state=cond_runtime_state,
    )
    _set_torch_rng_state(device, cpu_state, cuda_state)
    action_uncond = _predict_policy_action_with_runtime_state(
        observation_frame=observation_frame,
        policy=policy,
        device=device,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        use_amp=use_amp,
        task=task,
        robot_type=robot_type,
        runtime_state=uncond_runtime_state,
    )
    return action_uncond + acp_inference.cfg_beta * (action_cond - action_uncond)


class PolicySyncDualArmExecutor:
    """Broadcast one policy-derived robot action to follower + teleop arm."""

    def __init__(self, robot: Robot, teleop: Teleoperator, parallel_dispatch: bool = True):
        self.robot = robot
        self.teleop = teleop
        self.parallel_dispatch = parallel_dispatch
        self._pool = ThreadPoolExecutor(max_workers=2) if parallel_dispatch else None

    def send_action(self, action: RobotAction) -> RobotAction:
        if self._pool is None:
            sent_action = self.robot.send_action(action)
            self.teleop.send_feedback(action)
            return sent_action

        robot_future = self._pool.submit(self.robot.send_action, action)
        teleop_future = self._pool.submit(self.teleop.send_feedback, action)
        sent_action = robot_future.result()
        teleop_future.result()
        return sent_action

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
