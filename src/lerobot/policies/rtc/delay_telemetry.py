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

"""Lightweight, non-blocking RTC delay telemetry for real-time clients."""

import json
import math
import socket
from typing import Any


def parse_udp_address(address: str) -> tuple[str, int]:
    host, separator, port_text = address.rpartition(":")
    if not separator or not host or not port_text:
        raise ValueError(f"RTC d monitor address must use HOST:PORT, got {address!r}.")
    try:
        port = int(port_text)
    except ValueError as exc:
        raise ValueError(f"RTC d monitor port must be an integer, got {port_text!r}.") from exc
    if not 1 <= port <= 65535:
        raise ValueError(f"RTC d monitor port must be within [1, 65535], got {port}.")
    return host, port


def latency_seconds_to_delay_steps(latency_s: float, control_hz: float) -> int:
    if not math.isfinite(latency_s) or latency_s < 0:
        raise ValueError(f"latency_s must be finite and non-negative, got {latency_s}.")
    if not math.isfinite(control_hz) or control_hz <= 0:
        raise ValueError(f"control_hz must be finite and positive, got {control_hz}.")
    return math.ceil(latency_s * control_hz)


class RTCDelayTelemetrySender:
    """Send one small UDP JSON event for every returned action chunk."""

    def __init__(self, address: str, control_hz: float, source: str):
        self.target = parse_udp_address(address)
        if not math.isfinite(control_hz) or control_hz <= 0:
            raise ValueError(f"control_hz must be finite and positive, got {control_hz}.")
        self.control_hz = float(control_hz)
        self.source = source
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setblocking(False)

    def emit(
        self,
        *,
        observation_timestamp: float,
        receive_timestamp: float,
        first_action_timestep: int,
        latest_action_timestep: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        latency_s = max(0.0, receive_timestamp - observation_timestamp)
        payload: dict[str, Any] = {
            "source": self.source,
            "timestamp": receive_timestamp,
            "latency_ms": latency_s * 1000.0,
            "control_hz": self.control_hz,
            "d": latency_seconds_to_delay_steps(latency_s, self.control_hz),
            "first_action_timestep": int(first_action_timestep),
            "latest_action_timestep": int(latest_action_timestep),
            "stale_action_steps": max(0, int(latest_action_timestep) - int(first_action_timestep) + 1),
        }
        if metadata:
            payload.update(metadata)
        try:
            self.socket.sendto(json.dumps(payload, separators=(",", ":")).encode("utf-8"), self.target)
        except (BlockingIOError, OSError):
            # Telemetry must never slow down or interrupt the robot control loop.
            return

    def close(self) -> None:
        self.socket.close()
