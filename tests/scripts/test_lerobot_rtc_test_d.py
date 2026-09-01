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

import pytest

import lerobot.scripts.lerobot_rtc_test_d as rtc_cli
from lerobot.scripts.lerobot_rtc_test_d import (
    OVERFLOW_EXIT_CODE,
    RTCPolicyLimits,
    latency_ms_to_delay_steps,
    load_policy_limits,
    main,
    parse_telemetry_datagram,
    summarize_delay_distribution,
)


@pytest.mark.parametrize(
    "latency_ms, control_hz, expected",
    [(0.0, 30.0, 0), (200.0, 30.0, 6), (201.0, 30.0, 7)],
)
def test_latency_ms_to_delay_steps(latency_ms, control_hz, expected):
    assert latency_ms_to_delay_steps(latency_ms, control_hz) == expected


def test_distribution_recommends_percentile_plus_margin():
    report = summarize_delay_distribution(
        [1, 2, 2, 3, 4, 10],
        RTCPolicyLimits(chunk_size=50),
        recommend_percentile=100,
        safety_margin_steps=2,
    )

    assert report["status"] == "profiled"
    assert report["histogram"]["2"] == {"count": 2, "fraction": pytest.approx(2 / 6)}
    assert report["recommended_rtc_training_max_delay"] == 12
    assert report["quantiles"]["p100"] == 10


def test_distribution_reports_existing_training_overflow():
    report = summarize_delay_distribution(
        [1, 2, 3, 4, 5],
        RTCPolicyLimits(chunk_size=50, rtc_training_max_delay=3),
        recommend_percentile=99,
        safety_margin_steps=2,
    )

    assert report["status"] == "training_range_exceeded"
    assert report["existing_training_overflow_count"] == 2
    assert report["existing_training_overflow_fraction"] == pytest.approx(0.4)


def test_distribution_reports_exhausted_chunk():
    report = summarize_delay_distribution(
        [10, 50],
        RTCPolicyLimits(chunk_size=50),
        recommend_percentile=99,
        safety_margin_steps=2,
    )

    assert report["status"] == "chunk_exhausted"
    assert report["requires_larger_chunk"] is True


def test_load_policy_limits_from_training_output(tmp_path):
    config_dir = tmp_path / "checkpoints" / "last" / "pretrained_model"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "config.json"
    config_path.write_text(
        json.dumps({"type": "pi05", "chunk_size": 50, "rtc_training_max_delay": 10}),
        encoding="utf-8",
    )

    limits = load_policy_limits(str(tmp_path))

    assert limits.chunk_size == 50
    assert limits.rtc_training_max_delay == 10
    assert limits.config_path == str(config_path.resolve())


def test_load_policy_limits_treats_disabled_training_rtc_as_unset(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"type": "pi05", "chunk_size": 50, "rtc_training_max_delay": 0}),
        encoding="utf-8",
    )

    assert load_policy_limits(str(tmp_path)).rtc_training_max_delay is None


def test_parse_telemetry_converts_latency_when_d_is_absent():
    event = parse_telemetry_datagram(b'{"latency_ms":201,"control_hz":30}', None)
    assert event["d"] == 7


def test_cli_offline_json_report(capsys):
    exit_code = main(
        [
            "--chunk-size=50",
            "--delay-steps",
            "3",
            "4",
            "5",
            "--recommend-percentile=100",
            "--safety-margin-steps=1",
            "--json",
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert report["mode"] == "offline"
    assert report["recommended_rtc_training_max_delay"] == 6


def test_cli_returns_nonzero_when_existing_d_is_exceeded(capsys):
    exit_code = main(
        [
            "--chunk-size=50",
            "--rtc-training-max-delay=10",
            "--delay-steps",
            "8",
            "11",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == OVERFLOW_EXIT_CODE
    assert "TRAINING_RANGE_EXCEEDED" in output


def test_one_command_headless_mode_defaults_to_d_txt(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "type": "pi05",
                "chunk_size": 50,
                "rtc_training_max_delay": 10,
                "input_features": {
                    "observation.state": {"type": "STATE", "shape": [7]},
                    "observation.images.front": {"type": "VISUAL", "shape": [3, 224, 224]},
                },
            }
        ),
        encoding="utf-8",
    )
    profile_metadata = {
        "gpu_id": 7,
        "server_address": "127.0.0.1:8090",
        "server_port": 8090,
        "fps": 30,
        "duration_s": 600.0,
        "warmup_steps": 3,
        "actions_per_chunk": 50,
        "camera_width": 640,
        "camera_height": 480,
        "camera_count": 1,
        "state_dim": 7,
        "robot_actions_sent": False,
        "server_log": str(tmp_path / "d_server.log"),
    }

    def fake_integrated_profile(args, limits):
        assert args.gpu_id == 7
        assert args.server_port == 8090
        assert limits.chunk_size == 50
        return (
            [
                {"source": "headless_grpc_profile", "d": 6, "latency_ms": 200.0},
                {"source": "headless_grpc_profile", "d": 8, "latency_ms": 250.0},
            ],
            profile_metadata,
        )

    monkeypatch.setattr(rtc_cli, "run_integrated_headless_profile", fake_integrated_profile)
    monkeypatch.chdir(tmp_path)

    exit_code = rtc_cli.main([f"--policy-path={tmp_path}", "--gpu-id=7"])
    output = capsys.readouterr().out
    report_text = (tmp_path / "d.txt").read_text(encoding="utf-8")

    assert exit_code == 0
    assert "saved report" in output
    assert "physical GPU / server: 7 / 127.0.0.1:8090" in report_text
    assert "robot actions sent: False" in report_text
    assert "d=6: 1" in report_text


def test_external_server_mode_accepts_existing_client_parameter_names(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"type": "pi05", "chunk_size": 50, "input_features": {}}),
        encoding="utf-8",
    )

    def fake_integrated_profile(args, limits):
        assert args.server_address == "127.0.0.1:8090"
        assert args.gpu_id is None
        assert args.policy_device == "cuda:7"
        return (
            [{"source": "headless_grpc_profile", "d": 6, "latency_ms": 200.0}],
            {
                "gpu_id": None,
                "server_address": args.server_address,
                "server_port": 8090,
                "server_mode": "external",
                "requested_device": "cuda:7",
                "fps": 30,
                "duration_s": 600.0,
                "warmup_steps": 3,
                "actions_per_chunk": limits.chunk_size,
                "camera_width": 640,
                "camera_height": 480,
                "camera_count": 0,
                "state_dim": 0,
                "robot_actions_sent": False,
                "server_log": None,
            },
        )

    monkeypatch.setattr(rtc_cli, "run_integrated_headless_profile", fake_integrated_profile)
    monkeypatch.chdir(tmp_path)

    exit_code = rtc_cli.main(
        [
            f"--pretrained_name_or_path={tmp_path}",
            "--server_address=127.0.0.1:8090",
            "--policy_device=cuda:7",
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "d.txt").is_file()
