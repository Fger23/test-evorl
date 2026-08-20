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

"""Control-loop timing coverage for remote policy recording."""

from __future__ import annotations

from types import SimpleNamespace

from lerobot.scripts import recording_loop as recording_loop_module
from lerobot.utils.constants import ACTION


def test_remote_record_loop_emits_one_complete_async_timing_sample(monkeypatch):
    events = {
        "exit_early": False,
        "toggle_intervention": False,
    }

    class _Robot:
        name = "timing-test-robot"
        robot_type = "timing-test-robot"
        action_features = {"joint.pos": object()}

        def get_observation(self):
            return {"joint.pos": 0.0}

        def send_action(self, action):
            events["exit_early"] = True
            return action

    class _Dataset:
        fps = 30
        features = {ACTION: {"names": ["joint.pos"]}}

        def __init__(self):
            self.frames = []

        def add_frame(self, frame):
            self.frames.append(frame)

    class _RemoteClient:
        def __init__(self):
            self.reset_count = 0
            self.executed_count = 0
            self.metrics = []

        def reset(self):
            self.reset_count += 1

        def get_action(self, *, observation, task, timestep):
            assert observation == {"joint.pos": 0.0}
            assert task == "move"
            assert timestep == 0
            return {"joint.pos": 1.0}

        def mark_action_executed(self):
            self.executed_count += 1

        def record_control_loop_metrics(self, *, timestep, **metrics):
            self.metrics.append({"timestep": timestep, **metrics})

    dataset = _Dataset()
    remote_client = _RemoteClient()
    display_calls = []
    monkeypatch.setattr(
        recording_loop_module,
        "build_dataset_frame",
        lambda _features, values, prefix: {f"{prefix}.sample": values},
    )
    monkeypatch.setattr(recording_loop_module, "precise_sleep", lambda _duration_s: None)
    monkeypatch.setattr(
        recording_loop_module,
        "log_rerun_data",
        lambda **kwargs: display_calls.append(kwargs),
    )

    recording_loop_module.record_loop(
        robot=_Robot(),
        events=events,
        fps=30,
        teleop_action_processor=lambda pair: pair[0],
        robot_action_processor=lambda pair: pair[0],
        robot_observation_processor=lambda observation: observation,
        dataset=dataset,
        remote_policy_client=remote_client,
        control_time_s=1,
        single_task="move",
        display_data=True,
        acp_inference=SimpleNamespace(),
    )

    assert remote_client.reset_count == 1
    assert remote_client.executed_count == 1
    assert len(dataset.frames) == 1
    assert len(display_calls) == 1
    assert len(remote_client.metrics) == 1

    sample = remote_client.metrics[0]
    assert sample["timestep"] == 0
    assert sample["target_period_ms"] == 1000 / 30
    assert sample["loop_start_interval_ms"] is None
    for field in (
        "get_observation_ms",
        "observation_process_ms",
        "get_action_ms",
        "action_process_ms",
        "send_action_ms",
        "execution_confirmation_ms",
        "dataset_ms",
        "display_ms",
        "active_work_ms",
        "sleep_requested_ms",
        "sleep_ms",
        "loop_total_ms",
        "deadline_overrun_ms",
    ):
        assert sample[field] >= 0
    assert isinstance(sample["deadline_missed"], bool)
    assert sample["selected_from_policy"] is True
