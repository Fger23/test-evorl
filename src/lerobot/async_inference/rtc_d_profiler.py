#!/usr/bin/env python

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

"""Headless gRPC client used to profile RTC delay without controlling a robot."""

import json
import math
import pickle  # nosec
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.utils.constants import OBS_IMAGES, OBS_STATE


@dataclass(frozen=True)
class SyntheticObservationSpec:
    lerobot_features: dict[str, dict[str, Any]]
    raw_observation: dict[str, Any]
    camera_count: int
    state_dim: int


def build_synthetic_observation_spec(
    config_path: Path,
    *,
    camera_width: int,
    camera_height: int,
    task: str,
) -> SyntheticObservationSpec:
    """Build policy-shaped state/images without connecting to robot hardware."""
    if camera_width <= 0 or camera_height <= 0:
        raise ValueError(
            f"Synthetic camera width/height must be positive, got {camera_width}x{camera_height}."
        )

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    policy = payload.get("policy", payload)
    if not isinstance(policy, dict):
        raise ValueError(f"Policy config in {config_path} must be a JSON object.")
    input_features = policy.get("input_features")
    if not isinstance(input_features, dict) or not input_features:
        raise ValueError(f"Policy config {config_path} does not contain usable input_features.")

    lerobot_features: dict[str, dict[str, Any]] = {}
    raw_observation: dict[str, Any] = {"task": task}
    camera_count = 0
    state_dim = 0

    for feature_name, feature in input_features.items():
        if not isinstance(feature, dict) or "shape" not in feature:
            continue
        feature_type = str(feature.get("type", ""))
        shape = tuple(int(dim) for dim in feature["shape"])

        if feature_name == OBS_STATE or feature_type.endswith("STATE"):
            if len(shape) != 1:
                raise ValueError(f"State feature {feature_name!r} must be one-dimensional, got {shape}.")
            names = [f"profile_state_{index}" for index in range(shape[0])]
            lerobot_features[OBS_STATE] = {
                "dtype": "float32",
                "shape": shape,
                "names": names,
            }
            raw_observation.update(dict.fromkeys(names, 0.0))
            state_dim = shape[0]
            continue

        if feature_name.startswith(f"{OBS_IMAGES}.") or feature_type.endswith("VISUAL"):
            if len(shape) != 3:
                raise ValueError(f"Visual feature {feature_name!r} must be CHW, got {shape}.")
            camera_name = feature_name.removeprefix(f"{OBS_IMAGES}.")
            if camera_name == feature_name:
                raise ValueError(
                    f"Visual feature {feature_name!r} must use the '{OBS_IMAGES}.<camera>' naming convention."
                )
            channels = shape[0]
            lerobot_features[feature_name] = {
                "dtype": "image",
                "shape": (camera_height, camera_width, channels),
                "names": ["height", "width", "channels"],
            }
            raw_observation[camera_name] = np.zeros((camera_height, camera_width, channels), dtype=np.uint8)
            camera_count += 1

    if state_dim == 0:
        raise ValueError(f"PI0.5 profiling requires an {OBS_STATE!r} feature in {config_path}.")
    if camera_count == 0:
        raise ValueError(f"PI0.5 profiling requires at least one visual input feature in {config_path}.")

    return SyntheticObservationSpec(
        lerobot_features=lerobot_features,
        raw_observation=raw_observation,
        camera_count=camera_count,
        state_dim=state_dim,
    )


def wait_for_grpc_server(server_address: str, timeout_s: float) -> None:
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError(f"Server startup timeout must be positive, got {timeout_s}.")

    import grpc

    channel = grpc.insecure_channel(server_address)
    try:
        grpc.channel_ready_future(channel).result(timeout=timeout_s)
    except grpc.FutureTimeoutError as exc:
        raise TimeoutError(f"Timed out waiting for gRPC server at {server_address}.") from exc
    finally:
        channel.close()


def run_headless_profile_client(
    *,
    server_address: str,
    policy_path: str,
    observation_spec: SyntheticObservationSpec,
    actions_per_chunk: int,
    fps: int,
    duration_s: float,
    warmup_steps: int,
    request_timeout_s: float,
    device: str = "cuda",
    on_profile_start: Callable[[float], None] | None = None,
    on_sample: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run full preprocessing/model/postprocessing RPCs while dropping all returned actions."""
    if actions_per_chunk <= 0:
        raise ValueError(f"actions_per_chunk must be positive, got {actions_per_chunk}.")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}.")
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError(f"duration_s must be finite and positive, got {duration_s}.")
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}.")

    import grpc

    from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation
    from lerobot.transport import services_pb2, services_pb2_grpc
    from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks

    channel = grpc.insecure_channel(server_address, grpc_channel_options(initial_backoff="0.1s"))
    stub = services_pb2_grpc.AsyncInferenceStub(channel)
    policy_config = RemotePolicyConfig(
        policy_type="pi05",
        pretrained_name_or_path=policy_path,
        lerobot_features=observation_spec.lerobot_features,
        actions_per_chunk=actions_per_chunk,
        device=device,
    )

    def request_action_chunk(timestep: int) -> dict[str, Any]:
        raw_observation = dict(observation_spec.raw_observation)
        observation_timestamp = time.time()
        timed_observation = TimedObservation(
            timestamp=observation_timestamp,
            timestep=timestep,
            observation=raw_observation,
            must_go=True,
        )
        observation_iterator = send_bytes_in_chunks(
            pickle.dumps(timed_observation),  # nosec
            services_pb2.Observation,
            log_prefix="[RTC_D_PROFILE] Observation",
            silent=True,
        )
        stub.SendObservations(observation_iterator, timeout=request_timeout_s)
        actions_message = stub.GetActions(services_pb2.Empty(), timeout=request_timeout_s)
        receive_timestamp = time.time()
        if not actions_message.data:
            raise RuntimeError("Policy server returned an empty action chunk during RTC d profiling.")
        timed_actions = pickle.loads(actions_message.data)  # nosec
        if not timed_actions:
            raise RuntimeError("Policy server returned zero actions during RTC d profiling.")

        latency_s = max(0.0, receive_timestamp - observation_timestamp)
        return {
            "source": "headless_grpc_profile",
            "timestamp": receive_timestamp,
            "latency_ms": latency_s * 1000.0,
            "control_hz": fps,
            "d": math.ceil(latency_s * fps),
            "first_action_timestep": timed_actions[0].get_timestep(),
            "action_count": len(timed_actions),
        }

    try:
        stub.Ready(services_pb2.Empty(), timeout=request_timeout_s)
        stub.SendPolicyInstructions(
            services_pb2.PolicySetup(data=pickle.dumps(policy_config)),  # nosec
            timeout=request_timeout_s,
        )

        timestep = 0
        for _ in range(warmup_steps):
            request_action_chunk(timestep)
            timestep += actions_per_chunk

        start = time.monotonic()
        deadline = start + duration_s
        if on_profile_start is not None:
            on_profile_start(deadline)

        events: list[dict[str, Any]] = []
        while time.monotonic() < deadline:
            event = request_action_chunk(timestep)
            timestep += actions_per_chunk
            events.append(event)
            if on_sample is not None:
                on_sample(event)
        return events
    finally:
        channel.close()
