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

import json
import pickle  # nosec
import time
from concurrent import futures

import pytest
import torch

grpc = pytest.importorskip("grpc")

from lerobot.async_inference.helpers import TimedAction  # noqa: E402
from lerobot.async_inference.rtc_d_profiler import (  # noqa: E402
    SyntheticObservationSpec,
    build_synthetic_observation_spec,
    run_headless_profile_client,
)
from lerobot.transport import services_pb2, services_pb2_grpc  # noqa: E402


def write_pi05_config(path, *, include_state=True):
    input_features = {
        "observation.images.front": {"type": "VISUAL", "shape": [3, 224, 224]},
        "observation.images.wrist": {"type": "VISUAL", "shape": [3, 224, 224]},
    }
    if include_state:
        input_features["observation.state"] = {"type": "STATE", "shape": [7]}
    path.write_text(json.dumps({"type": "pi05", "input_features": input_features}), encoding="utf-8")


def test_build_synthetic_observation_matches_policy_features(tmp_path):
    config_path = tmp_path / "config.json"
    write_pi05_config(config_path)

    spec = build_synthetic_observation_spec(
        config_path,
        camera_width=640,
        camera_height=480,
        task="profile",
    )

    assert spec.camera_count == 2
    assert spec.state_dim == 7
    assert spec.raw_observation["front"].shape == (480, 640, 3)
    assert spec.raw_observation["wrist"].shape == (480, 640, 3)
    assert spec.raw_observation["task"] == "profile"
    assert spec.lerobot_features["observation.state"]["shape"] == (7,)
    assert len([key for key in spec.raw_observation if key.startswith("profile_state_")]) == 7


def test_build_synthetic_observation_requires_state(tmp_path):
    config_path = tmp_path / "config.json"
    write_pi05_config(config_path, include_state=False)

    with pytest.raises(ValueError, match="observation.state"):
        build_synthetic_observation_spec(
            config_path,
            camera_width=640,
            camera_height=480,
            task="profile",
        )


class FakeProfileServer(services_pb2_grpc.AsyncInferenceServicer):
    def __init__(self):
        self.observation = None
        self.policy_config = None

    def Ready(self, request, context):  # noqa: N802
        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        self.policy_config = pickle.loads(request.data)  # nosec
        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        self.observation = pickle.loads(b"".join(request.data for request in request_iterator))  # nosec
        return services_pb2.Empty()

    def GetActions(self, request, context):  # noqa: N802
        action = TimedAction(
            timestamp=self.observation.get_timestamp(),
            timestep=self.observation.get_timestep(),
            action=torch.zeros(2),
        )
        return services_pb2.Actions(data=pickle.dumps([action]))


def test_headless_client_runs_full_grpc_round_trip_without_robot():
    servicer = FakeProfileServer()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    spec = SyntheticObservationSpec(
        lerobot_features={"observation.state": {"dtype": "float32", "shape": (1,), "names": ["joint"]}},
        raw_observation={"joint": 0.0, "task": "profile"},
        camera_count=0,
        state_dim=1,
    )

    try:
        events = run_headless_profile_client(
            server_address=f"127.0.0.1:{port}",
            policy_path="unused",
            observation_spec=spec,
            actions_per_chunk=1,
            fps=30,
            duration_s=0.05,
            warmup_steps=1,
            request_timeout_s=1,
            device="cuda:7",
        )
    finally:
        server.stop(grace=0).wait(timeout=1)

    assert events
    assert all(event["source"] == "headless_grpc_profile" for event in events)
    assert all(event["action_count"] == 1 for event in events)
    assert servicer.policy_config.device == "cuda:7"
    assert servicer.observation.get_observation()["task"] == "profile"
    assert time.time() >= events[-1]["timestamp"]
