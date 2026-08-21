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

import pickle  # nosec
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from lerobot.scripts.recording_remote_policy_main import (
    RemotePolicyActionClient,
    RemotePolicyRecordConfig,
)


def test_record_remote_policy_uses_main_compatible_defaults_and_types():
    cfg = RemotePolicyRecordConfig()

    assert cfg.aggregate_fn_name == "latest_only"
    assert cfg.obs_queue_timeout_s == 10.0
    assert not hasattr(cfg, "rtc_enable")
    assert not hasattr(RemotePolicyActionClient, "mark_action_executed")


def test_record_remote_policy_start_negotiates_protocol_v1():
    cfg = RemotePolicyRecordConfig(
        enable=True,
        policy_type="pi05",
        pretrained_name_or_path="policy/checkpoint",
        actions_per_chunk=50,
    )
    client = RemotePolicyActionClient.__new__(RemotePolicyActionClient)
    client.cfg = cfg
    client.environment_dt = 1 / 30
    client.robot = SimpleNamespace()
    client.stub = MagicMock()
    client.services_pb2 = SimpleNamespace(
        Empty=lambda: SimpleNamespace(),
        PolicySetup=lambda *, data: SimpleNamespace(data=data),
    )

    with patch(
        "lerobot.async_inference.helpers.map_robot_keys_to_lerobot_features",
        return_value={"observation.state": object()},
    ):
        client.start()

    client.stub.Ready.assert_called_once()
    assert client.stub.Ready.call_args.kwargs == {"timeout": 10.0}
    setup = client.stub.SendPolicyInstructions.call_args.args[0]
    policy_config = pickle.loads(setup.data)  # nosec
    assert policy_config.protocol_version == 1
    assert policy_config.return_raw_actions is False
    assert policy_config.rtc_enabled is False
    assert policy_config.rtc_inference_delay is None
    assert policy_config.rtc_execution_horizon is None
