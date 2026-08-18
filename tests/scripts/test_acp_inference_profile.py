# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from pathlib import Path

import pytest
import torch

from lerobot.scripts.acp_inference_profile import ACPInferenceProfiler


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_record_appends_jsonl_and_updates_summary(tmp_path: Path) -> None:
    profiler = ACPInferenceProfiler(tmp_path, fps=30, cfg_beta=1.5, save_chunks=False, run_name="run")

    first = profiler.record({"wall_ms": 100.0, "cuda_ms": 80.0, "checkpoint": Path("model")})
    second = profiler.record({"wall_ms": torch.tensor(300.0), "cuda_ms": torch.tensor(240.0)})

    lines = profiler.records_path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    assert records == [first, second]
    assert first["checkpoint"] == "model"
    assert [record["index"] for record in records] == [0, 1]

    summary = _read_json(profiler.summary_path)
    assert summary["count"] == 2
    assert summary["wall_ms"] == {"p50": 200.0, "p95": 290.0, "p99": 298.0, "max": 300.0}
    assert summary["cuda_ms"] == {"p50": 160.0, "p95": 232.0, "p99": 238.4, "max": 240.0}
    assert summary["D99"] == 9
    assert summary["recommended_precision_horizon"] == 15
    assert summary["recommended_smooth_horizon"] == 17
    assert not list(profiler.output_dir.glob("*.tmp"))


def test_missing_cuda_latency_is_reported_as_null(tmp_path: Path) -> None:
    profiler = ACPInferenceProfiler(tmp_path, fps=20, cfg_beta=1.5, save_chunks=False, run_name="run")
    profiler.record({"wall_ms": 125.0})

    summary = _read_json(profiler.summary_path)
    assert summary["cuda_ms"] == {"p50": None, "p95": None, "p99": None, "max": None}
    assert summary["D99"] == 3
    assert summary["recommended_precision_horizon"] == 10
    assert summary["recommended_smooth_horizon"] == 11


def test_integration_latency_names_are_normalized(tmp_path: Path) -> None:
    profiler = ACPInferenceProfiler(tmp_path, fps=30, cfg_beta=1.5, save_chunks=False, run_name="run")

    record = profiler.record({"wall_latency_s": torch.tensor(0.25), "cuda_latency_ms": 200.0})

    assert record["wall_ms"] == pytest.approx(250.0)
    assert record["cuda_ms"] == 200.0
    summary = _read_json(profiler.summary_path)
    assert summary["wall_ms"]["p99"] == pytest.approx(250.0)
    assert summary["cuda_ms"]["p99"] == 200.0
    assert summary["D99"] == 8


def test_warmup_samples_are_excluded_from_recommendations(tmp_path: Path) -> None:
    profiler = ACPInferenceProfiler(tmp_path, fps=30, cfg_beta=1.5, save_chunks=False, run_name="run")
    profiler.record({"wall_ms": 1000.0, "cuda_ms": 900.0, "warmup": True})
    profiler.record({"wall_ms": 100.0, "cuda_ms": 80.0, "warmup": False})

    summary = _read_json(profiler.summary_path)
    assert summary["count"] == 2
    assert summary["wall_ms"]["p99"] == 991.0
    assert summary["cuda_ms"]["p99"] == pytest.approx(891.8)
    assert summary["steady_state_count"] == 1
    assert summary["steady_state_wall_ms"]["p99"] == 100.0
    assert summary["steady_state_cuda_ms"]["p99"] == 80.0
    assert summary["recommendation_source"] == "steady_state"
    assert summary["D99"] == 3
    assert summary["recommended_precision_horizon"] == 10
    assert summary["recommended_smooth_horizon"] == 11


def test_recommendations_fall_back_to_all_warmup_samples(tmp_path: Path) -> None:
    profiler = ACPInferenceProfiler(tmp_path, fps=30, cfg_beta=1.5, save_chunks=False, run_name="run")
    profiler.record({"wall_ms": 400.0, "warmup": True})

    summary = _read_json(profiler.summary_path)
    assert summary["steady_state_count"] == 0
    assert summary["steady_state_wall_ms"]["p99"] is None
    assert summary["recommendation_source"] == "all_samples_fallback"
    assert summary["D99"] == 12
    assert summary["recommended_precision_horizon"] == 18
    assert summary["recommended_smooth_horizon"] == 20


def test_chunks_are_saved_with_cpu_tensors(tmp_path: Path) -> None:
    profiler = ACPInferenceProfiler(tmp_path, fps=30, cfg_beta=1.5, run_name="run")
    chunks = {
        "uncond": torch.arange(6, dtype=torch.float32).reshape(1, 2, 3).requires_grad_(),
        "cond": torch.ones(1, 2, 3),
    }

    record = profiler.record({"wall_ms": 10.0}, chunks)

    assert record["chunk_file"] == "chunks/000000.pt"
    saved = torch.load(profiler.output_dir / record["chunk_file"], weights_only=True)
    assert set(saved) == set(chunks)
    assert all(tensor.device.type == "cpu" and not tensor.requires_grad for tensor in saved.values())
    assert torch.equal(saved["uncond"], chunks["uncond"])


def test_named_run_resumes_without_overwriting_chunks(tmp_path: Path) -> None:
    first = ACPInferenceProfiler(tmp_path, fps=30, cfg_beta=1.5, run_name="run")
    first.record({"wall_ms": 10.0}, {"cfg": torch.tensor([1.0])})

    resumed = ACPInferenceProfiler(tmp_path, fps=30, cfg_beta=1.5, run_name="run")
    record = resumed.record({"wall_ms": 20.0}, {"cfg": torch.tensor([2.0])})

    assert record["index"] == 1
    assert (resumed.chunks_dir / "000000.pt").exists()
    assert (resumed.chunks_dir / "000001.pt").exists()
    assert _read_json(resumed.summary_path)["count"] == 2


@pytest.mark.parametrize("wall_ms", [None, -1.0, float("nan"), float("inf")])
def test_invalid_wall_latency_is_rejected(tmp_path: Path, wall_ms: float | None) -> None:
    profiler = ACPInferenceProfiler(tmp_path, fps=30, cfg_beta=1.5, save_chunks=False, run_name="run")

    with pytest.raises(ValueError):
        profiler.record({"wall_ms": wall_ms})
