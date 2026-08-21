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

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from lerobot.robots.so_follower import (
    SO100Follower,
    SO100FollowerConfig,
    SOFollowerConfig,
)


def _make_bus_mock() -> MagicMock:
    """Return a bus mock with just the attributes used by the robot."""
    bus = MagicMock(name="FeetechBusMock")
    bus.is_connected = False

    def _connect():
        bus.is_connected = True

    def _disconnect(_disable=True):
        bus.is_connected = False

    bus.connect.side_effect = _connect
    bus.disconnect.side_effect = _disconnect

    @contextmanager
    def _dummy_cm():
        yield

    bus.torque_disabled.side_effect = _dummy_cm

    return bus


@pytest.fixture
def follower():
    bus_mock = _make_bus_mock()

    def _bus_side_effect(*_args, **kwargs):
        bus_mock.motors = kwargs["motors"]
        motors_order: list[str] = list(bus_mock.motors)

        bus_mock.sync_read.return_value = {motor: idx for idx, motor in enumerate(motors_order, 1)}
        bus_mock.sync_write.return_value = None
        bus_mock.write.return_value = None
        bus_mock.disable_torque.return_value = None
        bus_mock.enable_torque.return_value = None
        bus_mock.is_calibrated = True
        return bus_mock

    with (
        patch(
            "lerobot.robots.so_follower.so_follower.FeetechMotorsBus",
            side_effect=_bus_side_effect,
        ),
        patch.object(SO100Follower, "configure", lambda self: None),
    ):
        cfg = SO100FollowerConfig(port="/dev/null")
        robot = SO100Follower(cfg)
        yield robot
        if robot.is_connected:
            robot.disconnect()


def test_connect_disconnect(follower):
    assert not follower.is_connected

    follower.connect()
    assert follower.is_connected

    follower.disconnect()
    assert not follower.is_connected


def test_get_observation(follower):
    follower.connect()
    obs = follower.get_observation()

    expected_keys = {f"{m}.pos" for m in follower.bus.motors}
    assert set(obs.keys()) == expected_keys

    for idx, motor in enumerate(follower.bus.motors, 1):
        assert obs[f"{motor}.pos"] == idx


def test_get_observation_uses_async_camera_read_by_default(follower):
    follower.connect()
    camera = MagicMock()
    camera.is_connected = True
    camera.async_read.return_value = "async-frame"
    follower.cameras = {"wrist": camera}

    obs = follower.get_observation()

    assert obs["wrist"] == "async-frame"
    camera.async_read.assert_called_once_with()
    camera.read_latest.assert_not_called()


def test_get_observation_can_peek_latest_camera_frame(follower):
    follower.connect()
    follower.config.use_latest_camera_frames = True
    follower.config.camera_latest_max_age_ms = 125
    camera = MagicMock()
    camera.is_connected = True
    camera.read_latest.return_value = "latest-frame"
    follower.cameras = {"wrist": camera}

    obs = follower.get_observation()

    assert obs["wrist"] == "latest-frame"
    camera.read_latest.assert_called_once_with(max_age_ms=125)
    camera.async_read.assert_not_called()


@pytest.mark.parametrize("config_cls", [SOFollowerConfig, SO100FollowerConfig])
@pytest.mark.parametrize("invalid_max_age_ms", [True, False, 0, -1, 1.5, "200", None])
def test_camera_latest_max_age_must_be_a_positive_integer(config_cls, invalid_max_age_ms):
    with pytest.raises(ValueError, match="camera_latest_max_age_ms"):
        config_cls(port="/dev/null", camera_latest_max_age_ms=invalid_max_age_ms)


def test_send_action(follower):
    follower.connect()

    action = {f"{m}.pos": i * 10 for i, m in enumerate(follower.bus.motors, 1)}
    returned = follower.send_action(action)

    assert returned == action

    goal_pos = {m: (i + 1) * 10 for i, m in enumerate(follower.bus.motors)}
    follower.bus.sync_write.assert_called_once_with("Goal_Position", goal_pos)
