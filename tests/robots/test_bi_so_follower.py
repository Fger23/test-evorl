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

from unittest.mock import MagicMock, patch

from lerobot.robots.bi_so_follower import BiSOFollower, BiSOFollowerConfig
from lerobot.robots.so_follower import SOFollowerConfig


def test_bi_so_follower_forwards_latest_camera_options(tmp_path):
    arm_configs = []

    def make_arm(config):
        arm_configs.append(config)
        arm = MagicMock()
        arm.cameras = {}
        return arm

    config = BiSOFollowerConfig(
        id="test",
        calibration_dir=tmp_path,
        left_arm_config=SOFollowerConfig(
            port="left",
            use_latest_camera_frames=True,
            camera_latest_max_age_ms=125,
        ),
        right_arm_config=SOFollowerConfig(
            port="right",
            use_latest_camera_frames=False,
            camera_latest_max_age_ms=275,
        ),
    )

    with patch(
        "lerobot.robots.bi_so_follower.bi_so_follower.SOFollower",
        side_effect=make_arm,
    ):
        BiSOFollower(config)

    assert len(arm_configs) == 2
    assert arm_configs[0].use_latest_camera_frames is True
    assert arm_configs[0].camera_latest_max_age_ms == 125
    assert arm_configs[1].use_latest_camera_frames is False
    assert arm_configs[1].camera_latest_max_age_ms == 275
