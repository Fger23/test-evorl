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

"""Remote policy client used by recording loops.

This adapter talks to ``lerobot.async_inference.policy_server`` but leaves robot
control and dataset writing inside ``recording_loop.py`` so human-in-loop
takeover semantics stay in one place.
"""
import threading
import logging
import pickle  # nosec
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

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
    aggregate_fn_name: str = "latest_only"
    obs_queue_timeout_s: float = 2.0
    rename_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
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


class RemotePolicyActionClient:
    """Small synchronous client for remote policy actions during recording."""

    def __init__(self, cfg: RemotePolicyRecordConfig, robot: "Robot", fps: int):
        if not cfg.enable:
            raise ValueError("RemotePolicyActionClient requires `cfg.enable=true`.")

        self.cfg = cfg
        self.robot = robot
        self.environment_dt = 1 / fps
        from lerobot.transport import services_pb2, services_pb2_grpc
        from lerobot.transport.utils import grpc_channel_options

        import grpc

        self.services_pb2 = services_pb2
        self.channel = grpc.insecure_channel(
            cfg.server_address, grpc_channel_options(initial_backoff=f"{self.environment_dt:.4f}s")
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.action_queue: list["TimedAction"] = []
        self.latest_action_timestep = -1
        self.latest_action = None
        self.aggregate_fn = AGGREGATE_FUNCTIONS[cfg.aggregate_fn_name]

    @property
    def policy_id(self) -> str:
        return self.cfg.pretrained_name_or_path or self.cfg.policy_type or "remote_policy"

    def start(self) -> None:
        from lerobot.async_inference.helpers import RemotePolicyConfig, map_robot_keys_to_lerobot_features

        self.stub.Ready(self.services_pb2.Empty())
        policy_config = RemotePolicyConfig(
            policy_type=self.cfg.policy_type,
            pretrained_name_or_path=self.cfg.pretrained_name_or_path,
            lerobot_features=map_robot_keys_to_lerobot_features(self.robot),
            actions_per_chunk=self.cfg.actions_per_chunk,
            device=self.cfg.policy_device,
            rename_map=self.cfg.rename_map,
        )
        self.stub.SendPolicyInstructions(self.services_pb2.PolicySetup(data=pickle.dumps(policy_config)))
        logging.info("Remote policy recording client connected to %s.", self.cfg.server_address)

    def stop(self) -> None:
        self.channel.close()

    def reset(self) -> None:
        self.action_queue.clear()
        self.latest_action_timestep = -1
        self.latest_action = None
        # Sync server dedup / policy internal state when a new episode starts or
        # control returns to the policy after human takeover.
        self.stub.Ready(self.services_pb2.Empty())

    def _ready_to_send_observation(self) -> bool:
        if not self.action_queue:
            return True
        return len(self.action_queue) / self.cfg.actions_per_chunk <= self.cfg.chunk_size_threshold

    def _send_observation(self, observation: dict, timestep: int, task: str | None, must_go: bool) -> None:
        from lerobot.async_inference.helpers import TimedObservation
        from lerobot.transport.utils import send_bytes_in_chunks

        raw_observation = dict(observation)
        if task is not None:
            raw_observation["task"] = task

        timed_observation = TimedObservation(
            timestamp=time.time(),
            timestep=max(timestep, 0),
            observation=raw_observation,
            must_go=must_go,
        )
        observation_bytes = pickle.dumps(timed_observation)
        observation_iterator = send_bytes_in_chunks(
            observation_bytes,
            self.services_pb2.Observation,
            log_prefix="[RECORD_REMOTE_POLICY] Observation",
            silent=True,
        )
        self.stub.SendObservations(observation_iterator)

    def _receive_actions(self) -> None:
        deadline_t = time.perf_counter() + self.cfg.obs_queue_timeout_s
        while True:
            actions_chunk = self.stub.GetActions(self.services_pb2.Empty())
            if actions_chunk.data:
                timed_actions = pickle.loads(actions_chunk.data)  # nosec
                self._merge_actions(timed_actions)
                return

            if self.cfg.obs_queue_timeout_s == 0 or time.perf_counter() >= deadline_t:
                raise TimeoutError("Timed out waiting for remote policy actions.")

    def _merge_actions(self, incoming_actions: list["TimedAction"]) -> None:
        current_actions = {action.get_timestep(): action for action in self.action_queue}
        merged = {}

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

        if incoming_actions and not merged:
            logging.warning(
                "Remote policy returned %d actions but all were discarded because their timesteps "
                "<= latest_action_timestep=%d. Incoming range: %d..%d.",
                len(incoming_actions),
                self.latest_action_timestep,
                incoming_actions[0].get_timestep(),
                incoming_actions[-1].get_timestep(),
            )

        self.action_queue = [merged[timestep] for timestep in sorted(merged)]

    def _tensor_to_action(self, action_tensor: torch.Tensor) -> RobotAction:
        if self.cfg.client_device != "cpu" and action_tensor.device.type != self.cfg.client_device:
            action_tensor = action_tensor.to(self.cfg.client_device)
        else:
            action_tensor = action_tensor.cpu()
        return {key: action_tensor[i].item() for i, key in enumerate(self.robot.action_features)}

    # def get_action(self, observation: dict, task: str | None, timestep: int) -> RobotAction:
    #     if self._ready_to_send_observation():
    #         # This client sends an observation and then blocks on GetActions. Unlike the
    #         # async RobotClient, there is no background thread to retry later, so we must
    #         # always force inference here. Otherwise the server may filter a similar obs
    #         # (must_go=False) and GetActions times out while the action queue still has
    #         # steps left to execute.
    #         self._send_observation(
    #             observation=observation,
    #             timestep=timestep,
    #             task=task,
    #             must_go=True,
    #         )
    #         self._receive_actions()

    #     if not self.action_queue:
    #         raise TimeoutError("Remote policy returned no actions.")

    #     timed_action = self.action_queue.pop(0)
    #     self.latest_action_timestep = timed_action.get_timestep()
    #     return self._tensor_to_action(timed_action.get_action())

    # 异步推理，无需等待
    def get_action(self, observation: dict, task: str | None, timestep: int) -> RobotAction:
        if self._ready_to_send_observation():
            # This client sends an observation and then blocks on GetActions. Unlike the
            # async RobotClient, there is no background thread to retry later, so we must
            # always force inference here. Otherwise the server may filter a similar obs
            # (must_go=False) and GetActions times out while the action queue still has
            # steps left to execute.
            self._send_observation(
                observation=observation,
                timestep=timestep,
                task=task,
                must_go=True,
            )
            if not self.action_queue or self.latest_action is None:
                logging.warning("action queue is null or latest_action is None, wait for receive actions.")
                self._receive_actions() # 如果动作队列为空，则需要等待推理完成
            else:
                threading.Thread(target=self._receive_actions(), daemon=True).start()

        if not self.action_queue:
            # raise TimeoutError("Remote policy returned no actions.")
            logging.warning("Remote policy action queue is null, use latest_action.")
            timed_action = self.latest_action

        timed_action = self.action_queue.pop(0)
        self.latest_action = timed_action
        self.latest_action_timestep = timed_action.get_timestep()
        return self._tensor_to_action(timed_action.get_action())
