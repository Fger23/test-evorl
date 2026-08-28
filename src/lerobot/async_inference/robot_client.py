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
Example command:
```shell
python src/lerobot/async_inference/robot_client.py \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --task="dummy" \
    --server_address=127.0.0.1:8080 \
    --policy_type=act \
    --pretrained_name_or_path=user/model \
    --policy_device=mps \
    --client_device=cpu \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True
```
"""

import json
import logging
import math
import os
import pickle  # nosec
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime, timezone
from pprint import pformat
from queue import Queue
from typing import Any

import draccus
import grpc
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_piper_follower,
    bi_so_follower,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    piper_follower,
    so_follower,
)
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.utils.robot_utils import precise_sleep

from .configs import RobotClientConfig
from .constants import SUPPORTED_ROBOTS
from .helpers import (
    Action,
    FPSTracker,
    Observation,
    RawObservation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    map_robot_keys_to_lerobot_features,
    visualize_action_queue_size,
)


def _utc_string() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _percentile(values: list[float], quantile: float) -> float | None:
    """Linearly interpolated percentile without requiring NumPy."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def _latency_stats(values: list[float]) -> dict[str, float | None]:
    return {
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": float(max(values)) if values else None,
        "mean": float(sum(values) / len(values)) if values else None,
    }


def save_client_metrics(
    output_root: str,
    run_name: str | None,
    records: list[dict[str, Any]],
    action_queue_size: list[int],
    fps: float,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Persist client-side metrics (e2e latency, steps consumed, queue size) to disk.

    Writes ``{output_root}/{run_name}/records.jsonl`` (one line per received
    action chunk) and ``summary.json`` (aggregated statistics), mirroring the
    layout of the server-side ACP profiler output.

    Returns the run directory path.
    """
    if run_name is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        run_name = f"{timestamp}_{os.getpid()}"
    elif not run_name or run_name in {".", ".."} or "/" in run_name or "\\" in run_name:
        raise ValueError("metrics_run_name must be a non-empty directory name, not a path")

    output_dir = os.path.join(output_root, run_name)
    os.makedirs(output_dir, exist_ok=True)

    records_path = os.path.join(output_dir, "records.jsonl")
    with open(records_path, "w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False))
            stream.write("\n")

    e2e_values = [r["e2e_latency_ms"] for r in records if r.get("e2e_latency_ms") is not None]
    steps_values = [
        r["steps_consumed_during_inference"]
        for r in records
        if r.get("steps_consumed_during_inference") is not None
    ]
    queue_stats = {
        "count": len(action_queue_size),
        "min": int(min(action_queue_size)) if action_queue_size else None,
        "max": int(max(action_queue_size)) if action_queue_size else None,
        "mean": float(sum(action_queue_size) / len(action_queue_size)) if action_queue_size else None,
    }
    summary = {
        "run_name": run_name,
        "created_at_utc": _utc_string(),
        "fps": fps,
        "num_chunks": len(records),
        "e2e_latency_ms": _latency_stats(e2e_values),
        "steps_consumed_during_inference": {
            **_latency_stats([float(v) for v in steps_values]),
            "total": int(sum(steps_values)) if steps_values else 0,
        },
        "action_queue_size": queue_stats,
        "action_queue_size_series": action_queue_size,
        **(metadata or {}),
    }
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")

    # `math` imported for symmetry with percentile helpers; keep reference alive
    _ = math

    return output_dir


class RobotClient:
    prefix = "robot_client"
    logger = get_logger(prefix)

    def __init__(self, config: RobotClientConfig):
        """Initialize RobotClient with unified configuration.

        Args:
            config: RobotClientConfig containing all configuration parameters
        """
        # Store configuration
        self.config = config
        self.robot = make_robot_from_config(config.robot)
        self.robot.connect()

        lerobot_features = map_robot_keys_to_lerobot_features(self.robot)

        # Use environment variable if server_address is not provided in config
        self.server_address = config.server_address

        self.policy_config = RemotePolicyConfig(
            config.policy_type,
            config.pretrained_name_or_path,
            lerobot_features,
            config.actions_per_chunk,
            config.policy_device,
        )
        self.channel = grpc.insecure_channel(
            self.server_address, grpc_channel_options(initial_backoff=f"{config.environment_dt:.4f}s")
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.logger.info(f"Initializing client to connect to server at {self.server_address}")

        self.shutdown_event = threading.Event()

        # Initialize client side variables
        self.latest_action_lock = threading.Lock()
        self.latest_action = -1
        self.action_chunk_size = -1

        self._chunk_size_threshold = config.chunk_size_threshold

        self.action_queue = Queue()
        self.action_queue_lock = threading.Lock()  # Protect queue operations
        self.action_queue_size = []
        self.start_barrier = threading.Barrier(2)  # 2 threads: action receiver, control loop

        # Client-side metrics (persisted on stop)
        self._metrics_lock = threading.Lock()
        self.client_metrics_records: list[dict[str, Any]] = []
        self._last_chunk_latest_action = -1

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=self.config.fps)

        self.logger.info("Robot connected and ready")

        # Use an event for thread-safe coordination
        self.must_go = threading.Event()
        self.must_go.set()  # Initially set - observations qualify for direct processing

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    def start(self):
        """Start the robot client and connect to the policy server"""
        try:
            # client-server handshake
            start_time = time.perf_counter()
            self.stub.Ready(services_pb2.Empty())
            end_time = time.perf_counter()
            self.logger.debug(f"Connected to policy server in {end_time - start_time:.4f}s")

            # send policy instructions
            policy_config_bytes = pickle.dumps(self.policy_config)
            policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)

            self.logger.info("Sending policy instructions to policy server")
            self.logger.debug(
                f"Policy type: {self.policy_config.policy_type} | "
                f"Pretrained name or path: {self.policy_config.pretrained_name_or_path} | "
                f"Device: {self.policy_config.device}"
            )

            self.stub.SendPolicyInstructions(policy_setup)

            self.shutdown_event.clear()

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Failed to connect to policy server: {e}")
            return False

    def stop(self):
        """Stop the robot client"""
        self.shutdown_event.set()

        self.robot.disconnect()
        self.logger.debug("Robot disconnected")

        self.channel.close()
        self.logger.debug("Client stopped, channel closed")

        self._save_metrics()

    def _save_metrics(self) -> str | None:
        """Persist client-side metrics (e2e latency, steps consumed, queue size).

        Returns the run directory path, or None when there is nothing to save.
        """
        with self._metrics_lock:
            records = list(self.client_metrics_records)
        with self.action_queue_lock:
            queue_sizes = list(self.action_queue_size)

        if not records and not queue_sizes:
            return None

        output_dir = save_client_metrics(
            output_root=self.config.metrics_output_dir,
            run_name=self.config.metrics_run_name,
            records=records,
            action_queue_size=queue_sizes,
            fps=self.config.fps,
            metadata={
                "server_address": self.server_address,
                "policy_type": self.config.policy_type,
                "pretrained_name_or_path": self.config.pretrained_name_or_path,
                "actions_per_chunk": self.config.actions_per_chunk,
                "chunk_size_threshold": self._chunk_size_threshold,
            },
        )
        self.logger.info(f"Client metrics saved to {output_dir}")
        return output_dir

    def send_observation(
        self,
        obs: TimedObservation,
    ) -> bool:
        """Send observation to the policy server.
        Returns True if the observation was sent successfully, False otherwise."""
        if not self.running:
            raise RuntimeError("Client not running. Run RobotClient.start() before sending observations.")

        if not isinstance(obs, TimedObservation):
            raise ValueError("Input observation needs to be a TimedObservation!")

        start_time = time.perf_counter()
        observation_bytes = pickle.dumps(obs)
        serialize_time = time.perf_counter() - start_time
        self.logger.debug(f"Observation serialization time: {serialize_time:.6f}s")

        try:
            observation_iterator = send_bytes_in_chunks(
                observation_bytes,
                services_pb2.Observation,
                log_prefix="[CLIENT] Observation",
                silent=True,
            )
            _ = self.stub.SendObservations(observation_iterator)
            obs_timestep = obs.get_timestep()
            self.logger.debug(f"Sent observation #{obs_timestep} | ")

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Error sending observation #{obs.get_timestep()}: {e}")
            return False

    def _inspect_action_queue(self):
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
            timestamps = sorted([action.get_timestep() for action in self.action_queue.queue])
        self.logger.debug(f"Queue size: {queue_size}, Queue contents: {timestamps}")
        return queue_size, timestamps

    def _aggregate_action_queues(
        self,
        incoming_actions: list[TimedAction],
        aggregate_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ):
        """Finds the same timestep actions in the queue and aggregates them using the aggregate_fn"""
        if aggregate_fn is None:
            # default aggregate function: take the latest action
            def aggregate_fn(x1, x2):
                return x2

        future_action_queue = Queue()
        with self.action_queue_lock:
            internal_queue = self.action_queue.queue

        current_action_queue = {action.get_timestep(): action.get_action() for action in internal_queue}

        for new_action in incoming_actions:
            with self.latest_action_lock:
                latest_action = self.latest_action

            # New action is older than the latest action in the queue, skip it
            if new_action.get_timestep() <= latest_action:
                continue

            # If the new action's timestep is not in the current action queue, add it directly
            elif new_action.get_timestep() not in current_action_queue:
                future_action_queue.put(new_action)
                continue

            # If the new action's timestep is in the current action queue, aggregate it
            # TODO: There is probably a way to do this with broadcasting of the two action tensors
            future_action_queue.put(
                TimedAction(
                    timestamp=new_action.get_timestamp(),
                    timestep=new_action.get_timestep(),
                    action=aggregate_fn(
                        current_action_queue[new_action.get_timestep()], new_action.get_action()
                    ),
                )
            )

        with self.action_queue_lock:
            self.action_queue = future_action_queue

    def receive_actions(self, verbose: bool = False):
        """Receive actions from the policy server"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Action receiving thread starting")

        while self.running:
            try:
                # Use StreamActions to get a stream of actions from the server
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                if len(actions_chunk.data) == 0:
                    continue  # received `Empty` from server, wait for next call

                receive_time = time.time()

                # Deserialize bytes back into list[TimedAction]
                deserialize_start = time.perf_counter()
                timed_actions = pickle.loads(actions_chunk.data)  # nosec
                deserialize_time = time.perf_counter() - deserialize_start

                # Log device type of received actions
                if len(timed_actions) > 0:
                    received_device = timed_actions[0].get_action().device.type
                    self.logger.debug(f"Received actions on device: {received_device}")

                # Move actions to client_device (e.g., for downstream planners that need GPU)
                client_device = self.config.client_device
                if client_device != "cpu":
                    for timed_action in timed_actions:
                        if timed_action.get_action().device.type != client_device:
                            timed_action.action = timed_action.get_action().to(client_device)
                    self.logger.debug(f"Converted actions to device: {client_device}")
                else:
                    self.logger.debug(f"Actions kept on device: {client_device}")

                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))

                # Record client-side metrics: end-to-end latency (observation sent ->
                # full chunk received) and steps consumed while this inference ran
                if len(timed_actions) > 0:
                    with self.latest_action_lock:
                        latest_action = self.latest_action
                    with self.action_queue_lock:
                        queue_size_on_arrival = self.action_queue.qsize()
                    steps_consumed = max(latest_action - self._last_chunk_latest_action, 0)
                    e2e_latency_ms = (receive_time - timed_actions[0].get_timestamp()) * 1000
                    with self._metrics_lock:
                        self.client_metrics_records.append(
                            {
                                "index": len(self.client_metrics_records),
                                "e2e_latency_ms": e2e_latency_ms,
                                "steps_consumed_during_inference": steps_consumed,
                                "latest_action_timestep": latest_action,
                                "chunk_size": len(timed_actions),
                                "queue_size_on_arrival": queue_size_on_arrival,
                                "received_at_utc": _utc_string(),
                            }
                        )
                        self._last_chunk_latest_action = latest_action

                # Calculate network latency if we have matching observations
                if len(timed_actions) > 0 and verbose:
                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.debug(f"Current latest action: {latest_action}")

                    # Get queue state before changes
                    old_size, old_timesteps = self._inspect_action_queue()
                    if not old_timesteps:
                        old_timesteps = [latest_action]  # queue was empty

                    # Log incoming actions
                    incoming_timesteps = [a.get_timestep() for a in timed_actions]

                    first_action_timestep = timed_actions[0].get_timestep()
                    server_to_client_latency = (receive_time - timed_actions[0].get_timestamp()) * 1000

                    self.logger.info(
                        f"Received action chunk for step #{first_action_timestep} | "
                        f"Latest action: #{latest_action} | "
                        f"Incoming actions: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Network latency (server->client): {server_to_client_latency:.2f}ms | "
                        f"Deserialization time: {deserialize_time * 1000:.2f}ms"
                    )

                # Update action queue
                start_time = time.perf_counter()
                self._aggregate_action_queues(timed_actions, self.config.aggregate_fn)
                queue_update_time = time.perf_counter() - start_time

                self.must_go.set()  # after receiving actions, next empty queue triggers must-go processing!

                if verbose:
                    # Get queue state after changes
                    new_size, new_timesteps = self._inspect_action_queue()

                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.info(
                        f"Latest action: {latest_action} | "
                        f"Old action steps: {old_timesteps[0]}:{old_timesteps[-1]} | "
                        f"Incoming action steps: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Updated action steps: {new_timesteps[0]}:{new_timesteps[-1]}"
                    )
                    self.logger.debug(
                        f"Queue update complete ({queue_update_time:.6f}s) | "
                        f"Before: {old_size} items | "
                        f"After: {new_size} items | "
                    )

            except grpc.RpcError as e:
                self.logger.error(f"Error receiving actions: {e}")

    def actions_available(self):
        """Check if there are actions available in the queue"""
        with self.action_queue_lock:
            return not self.action_queue.empty()

    def _action_tensor_to_action_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        action = {key: action_tensor[i].item() for i, key in enumerate(self.robot.action_features)}
        return action

    def control_loop_action(self, verbose: bool = False) -> dict[str, Any]:
        """Reading and performing actions in local queue"""

        # Lock only for queue operations
        get_start = time.perf_counter()
        with self.action_queue_lock:
            self.action_queue_size.append(self.action_queue.qsize())
            # Get action from queue
            timed_action = self.action_queue.get_nowait()
        get_end = time.perf_counter() - get_start

        _performed_action = self.robot.send_action(
            self._action_tensor_to_action_dict(timed_action.get_action())
        )
        with self.latest_action_lock:
            self.latest_action = timed_action.get_timestep()

        if verbose:
            with self.action_queue_lock:
                current_queue_size = self.action_queue.qsize()

            self.logger.debug(
                f"Ts={timed_action.get_timestamp()} | "
                f"Action #{timed_action.get_timestep()} performed | "
                f"Queue size: {current_queue_size}"
            )

            self.logger.debug(
                f"Popping action from queue to perform took {get_end:.6f}s | Queue size: {current_queue_size}"
            )

        return _performed_action

    def _ready_to_send_observation(self):
        """Flags when the client is ready to send an observation"""
        with self.action_queue_lock:
            return self.action_queue.qsize() / self.action_chunk_size <= self._chunk_size_threshold

    def control_loop_observation(self, task: str, verbose: bool = False) -> RawObservation:
        try:
            # Get serialized observation bytes from the function
            start_time = time.perf_counter()

            raw_observation: RawObservation = self.robot.get_observation()
            raw_observation["task"] = task

            with self.latest_action_lock:
                latest_action = self.latest_action

            observation = TimedObservation(
                timestamp=time.time(),  # need time.time() to compare timestamps across client and server
                observation=raw_observation,
                timestep=max(latest_action, 0),
            )

            obs_capture_time = time.perf_counter() - start_time

            # Force inference whenever we send an observation. When chunk_size_threshold > 0
            # we also send while the queue still has actions; those obs used to use
            # must_go=False and could be filtered as "too similar", starving GetActions.
            with self.action_queue_lock:
                observation.must_go = True
                current_queue_size = self.action_queue.qsize()

            _ = self.send_observation(observation)

            self.logger.debug(f"QUEUE SIZE: {current_queue_size} (Must go: {observation.must_go})")
            if observation.must_go:
                # must-go event will be set again after receiving actions
                self.must_go.clear()

            if verbose:
                # Calculate comprehensive FPS metrics
                fps_metrics = self.fps_tracker.calculate_fps_metrics(observation.get_timestamp())

                self.logger.info(
                    f"Obs #{observation.get_timestep()} | "
                    f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
                    f"Target: {fps_metrics['target_fps']:.2f}"
                )

                self.logger.debug(
                    f"Ts={observation.get_timestamp():.6f} | Capturing observation took {obs_capture_time:.6f}s"
                )

            return raw_observation

        except Exception as e:
            self.logger.error(f"Error in observation sender: {e}")

    def control_loop(self, task: str, verbose: bool = False) -> tuple[Observation, Action]:
        """Combined function for executing actions and streaming observations"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Control loop thread starting")

        _performed_action = None
        _captured_observation = None

        while self.running:
            control_loop_start = time.perf_counter()
            """Control loop: (1) Performing actions, when available"""
            if self.actions_available():
                _performed_action = self.control_loop_action(verbose)

            """Control loop: (2) Streaming observations to the remote policy server"""
            if self._ready_to_send_observation():
                _captured_observation = self.control_loop_observation(task, verbose)

            self.logger.debug(f"Control loop (ms): {(time.perf_counter() - control_loop_start) * 1000:.2f}")
            # Dynamically adjust sleep time to maintain the desired control frequency
            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))

        return _captured_observation, _performed_action


def _make_rtc_action_client(cfg: RobotClientConfig, robot: Robot):
    """Build the shared CFG/RTC action client for this entrypoint."""
    from lerobot.scripts.recording_remote_policy import (
        RemotePolicyActionClient,
        RemotePolicyRecordConfig,
    )

    remote_cfg = RemotePolicyRecordConfig(
        enable=True,
        server_address=cfg.server_address,
        policy_type=cfg.policy_type,
        pretrained_name_or_path=cfg.pretrained_name_or_path,
        policy_device=cfg.policy_device,
        client_device=cfg.client_device,
        actions_per_chunk=cfg.actions_per_chunk,
        chunk_size_threshold=cfg.chunk_size_threshold,
        # RTC atomically installs one aligned raw/processed chunk. When RTC is
        # disabled, this client transparently uses its protocol-v1 queue.
        aggregate_fn_name="latest_only",
        obs_queue_timeout_s=cfg.obs_queue_timeout_s,
        use_cfg=cfg.use_cfg,
        rtc_enable=cfg.rtc_enable,
        rtc_inference_delay=cfg.rtc_inference_delay,
        rtc_execution_horizon=cfg.rtc_execution_horizon,
        rtc_max_guidance_weight=cfg.rtc_max_guidance_weight,
        rtc_prefix_attention_schedule=cfg.rtc_prefix_attention_schedule,
        rtc_cfg_beta=cfg.rtc_cfg_beta,
    )
    return RemotePolicyActionClient(cfg=remote_cfg, robot=robot, fps=cfg.fps)


def _rtc_control_loop(
    cfg: RobotClientConfig,
    robot: Robot,
    action_client,
    *,
    max_steps: int | None = None,
) -> int:
    """Execute one RTC model action per robot command at ``cfg.fps``.

    ``max_steps`` exists for deterministic unit tests. Production runs until
    Ctrl+C. The actual-delay counter advances only after the corresponding
    command has reached ``robot.send_action`` successfully.
    """
    executed_steps = 0
    while max_steps is None or executed_steps < max_steps:
        loop_start = time.perf_counter()
        observation = robot.get_observation()
        action = action_client.get_action(
            observation=observation,
            task=cfg.task,
            timestep=executed_steps,
        )

        robot.send_action(action)
        action_client.mark_action_executed()
        executed_steps += 1

        precise_sleep(max(cfg.environment_dt - (time.perf_counter() - loop_start), 0.0))

    return executed_steps


def run_rtc_client(cfg: RobotClientConfig) -> None:
    """Run the 30 Hz client-authoritative CFG/RTC lifecycle."""
    logger = get_logger("rtc_robot_client")
    robot = make_robot_from_config(cfg.robot)
    action_client = None

    try:
        robot.connect()
        action_client = _make_rtc_action_client(cfg, robot)
        action_client.start()
        # Keep the freshly loaded server policy and the local raw/processed
        # ActionQueue in the same episode epoch.
        action_client.reset()
        logger.info(
            "Remote action client ready: server=%s, cfg=%s, rtc=%s, fps=%d, "
            "d=%d, H=%d, beta=%s, threshold=%.3f",
            cfg.server_address,
            cfg.use_cfg,
            cfg.rtc_enable,
            cfg.fps,
            cfg.rtc_inference_delay,
            cfg.rtc_execution_horizon,
            f"{cfg.rtc_cfg_beta:.3f}" if cfg.use_cfg else "disabled",
            cfg.chunk_size_threshold,
        )
        _rtc_control_loop(cfg, robot, action_client)
    except KeyboardInterrupt:
        logger.info("RTC client interrupted by user")
    finally:
        if action_client is not None:
            action_client.stop()
        if robot.is_connected:
            robot.disconnect()
        logger.info("RTC client stopped")


@draccus.wrap()
def async_client(cfg: RobotClientConfig):
    logging.info(pformat(asdict(cfg)))

    if cfg.robot.type not in SUPPORTED_ROBOTS:
        raise ValueError(f"Robot {cfg.robot.type} not yet supported!")

    # Use the shared client whenever RTC or client-controlled CFG is enabled.
    # This keeps the two switches independent: CFG can remain enabled while
    # RTC is disabled. With both switches false, retain the historical client.
    if cfg.rtc_enable or cfg.use_cfg:
        run_rtc_client(cfg)
        return

    client = RobotClient(cfg)

    if client.start():
        client.logger.info("Starting action receiver thread...")

        # Create and start action receiver thread
        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)

        # Start action receiver thread
        action_receiver_thread.start()

        try:
            # The main thread runs the control loop
            client.control_loop(task=cfg.task)

        finally:
            client.stop()
            action_receiver_thread.join()
            if cfg.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_size)
            client.logger.info("Client stopped")


if __name__ == "__main__":
    async_client()  # run the client
