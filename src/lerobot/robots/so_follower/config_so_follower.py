#!/usr/bin/env python

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

from dataclasses import dataclass, field
from typing import TypeAlias

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


def _validate_so_follower_config(config: "SOFollowerConfig") -> None:
    if (
        not isinstance(config.camera_latest_max_age_ms, int)
        or isinstance(config.camera_latest_max_age_ms, bool)
        or config.camera_latest_max_age_ms <= 0
    ):
        raise ValueError("`camera_latest_max_age_ms` must be a positive integer.")


@dataclass
class SOFollowerConfig:
    """Base configuration class for SO Follower robots."""

    # Port to connect to the arm
    port: str

    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a dictionary that maps motor
    # names to the max_relative_target value for that motor.
    max_relative_target: float | dict[str, float] | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # `async_read` waits for a frame that has not yet been consumed, which can
    # synchronize the control loop to the camera FPS. `read_latest` instead
    # peeks at the camera's existing background buffer and bounds staleness.
    use_latest_camera_frames: bool = False
    camera_latest_max_age_ms: int = 200

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = False

    def __post_init__(self) -> None:
        _validate_so_follower_config(self)


@RobotConfig.register_subclass("so101_follower")
@RobotConfig.register_subclass("so100_follower")
@dataclass
class SOFollowerRobotConfig(RobotConfig, SOFollowerConfig):
    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_so_follower_config(self)


SO100FollowerConfig: TypeAlias = SOFollowerRobotConfig
SO101FollowerConfig: TypeAlias = SOFollowerRobotConfig
