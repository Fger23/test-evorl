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
import socket

import pytest

from lerobot.policies.rtc.delay_telemetry import (
    RTCDelayTelemetrySender,
    latency_seconds_to_delay_steps,
    parse_udp_address,
)


@pytest.mark.parametrize(
    "latency_s, control_hz, expected",
    [(0.0, 30.0, 0), (0.2, 30.0, 6), (0.201, 30.0, 7)],
)
def test_latency_seconds_to_delay_steps_uses_controller_ticks(latency_s, control_hz, expected):
    assert latency_seconds_to_delay_steps(latency_s, control_hz) == expected


def test_parse_udp_address_rejects_invalid_address():
    with pytest.raises(ValueError, match="HOST:PORT"):
        parse_udp_address("localhost")


def test_telemetry_sender_emits_nonblocking_json_datagram():
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(1.0)
    port = receiver.getsockname()[1]
    sender = RTCDelayTelemetrySender(f"127.0.0.1:{port}", control_hz=30, source="test_client")

    try:
        sender.emit(
            observation_timestamp=10.0,
            receive_timestamp=10.201,
            first_action_timestep=10,
            latest_action_timestep=15,
            metadata={"run": "test"},
        )
        payload = json.loads(receiver.recvfrom(65535)[0])
    finally:
        sender.close()
        receiver.close()

    assert payload["source"] == "test_client"
    assert payload["d"] == 7
    assert payload["latency_ms"] == pytest.approx(201.0)
    assert payload["stale_action_steps"] == 6
    assert payload["run"] == "test"
