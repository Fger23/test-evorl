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

"""Persistent profiling records for batched ACP inference.

The profiler deliberately has no dependency on the recording loop.  Callers
measure one chunk inference, then pass the resulting metrics and (optionally)
the raw chunks to :meth:`ACPInferenceProfiler.record`.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

_PERCENTILES = ("p50", "p95", "p99")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_string(value: datetime | None = None) -> str:
    value = value or _utc_now()
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _default_run_name() -> str:
    timestamp = _utc_now().strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{timestamp}_{os.getpid()}"


def _jsonable(value: Any) -> Any:
    """Convert supported metric values to values accepted by ``json``."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise TypeError("Only scalar tensors are supported in metrics; pass full tensors via chunks")
        return _jsonable(value.detach().cpu().item())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise TypeError(f"Unsupported metric value type: {type(value).__name__}")


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    """Return a linearly interpolated percentile without requiring NumPy."""
    if not values:
        return None

    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return float(ordered[lower_index])

    fraction = position - lower_index
    return float(ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * fraction)


def _latency_stats(values: Sequence[float]) -> dict[str, float | None]:
    return {
        **{name: _percentile(values, int(name[1:]) / 100) for name in _PERCENTILES},
        "max": float(max(values)) if values else None,
    }


def _validate_latency(value: Any, name: str, *, required: bool) -> float | None:
    if value is None:
        if required:
            raise ValueError(f"metrics must include a non-null {name}")
        return None
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a number, not bool")

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a number") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite, non-negative number")
    return result


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))


class ACPInferenceProfiler:
    """Append per-chunk ACP metrics and maintain an aggregate summary.

    Args:
        output_root: Parent directory for profiling runs.
        fps: Robot control frequency, used to convert p99 latency to action steps.
        cfg_beta: CFG coefficient used by the profiled inference run.
        save_chunks: Whether tensors passed to ``record`` should be persisted.
        run_name: Optional stable run directory name. If omitted, a UTC timestamp
            and process ID are used. Reusing a name resumes the existing JSONL run.
    """

    def __init__(
        self,
        output_root: str | Path,
        fps: float,
        cfg_beta: float,
        save_chunks: bool = True,
        run_name: str | None = None,
    ) -> None:
        if isinstance(fps, bool) or not math.isfinite(float(fps)) or float(fps) <= 0:
            raise ValueError("fps must be a finite number greater than zero")
        if isinstance(cfg_beta, bool) or not math.isfinite(float(cfg_beta)):
            raise ValueError("cfg_beta must be a finite number")

        if run_name is None:
            run_name = _default_run_name()
        elif not run_name or Path(run_name).name != run_name or run_name in {".", ".."}:
            raise ValueError("run_name must be a non-empty directory name, not a path")

        self.fps = float(fps)
        self.cfg_beta = float(cfg_beta)
        self.save_chunks = save_chunks
        self.run_name = run_name
        self.output_dir = Path(output_root) / run_name
        self.records_path = self.output_dir / "records.jsonl"
        self.summary_path = self.output_dir / "summary.json"
        self.chunks_dir = self.output_dir / "chunks"
        self._lock = threading.Lock()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.save_chunks:
            self.chunks_dir.mkdir(parents=True, exist_ok=True)

        self._records = self._load_existing_records()
        self._validate_existing_run()
        self._next_index = self._find_next_index()
        self._created_at_utc = self._load_created_at() or _utc_string()
        if not self.records_path.exists():
            self.records_path.touch()
        self._write_summary_atomic()

    def record(
        self,
        metrics: dict[str, Any],
        chunks: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        """Persist one chunk inference and return its JSON-compatible record."""
        if not isinstance(metrics, dict):
            raise TypeError("metrics must be a dict")

        json_metrics = _jsonable(metrics)
        if "wall_ms" in json_metrics:
            wall_ms = _validate_latency(json_metrics["wall_ms"], "wall_ms", required=True)
        else:
            wall_latency_s = _validate_latency(
                json_metrics.get("wall_latency_s"), "wall_latency_s", required=True
            )
            wall_ms = wall_latency_s * 1000

        cuda_metric = json_metrics.get("cuda_ms", json_metrics.get("cuda_latency_ms"))
        cuda_ms = _validate_latency(cuda_metric, "cuda_ms", required=False)
        warmup = json_metrics.get("warmup", False)
        if not isinstance(warmup, bool):
            raise TypeError("warmup must be a bool")
        json_metrics["wall_ms"] = wall_ms
        json_metrics["warmup"] = warmup
        if cuda_metric is not None:
            json_metrics["cuda_ms"] = cuda_ms

        with self._lock:
            index = self._next_index
            chunk_file = self._save_chunks(index, chunks)
            record = {
                **json_metrics,
                "index": index,
                "recorded_at_utc": _utc_string(),
                "fps": self.fps,
                "cfg_beta": self.cfg_beta,
                "chunk_file": chunk_file,
            }

            serialized = json.dumps(record, ensure_ascii=False, allow_nan=False)
            with self.records_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(serialized)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())

            self._records.append(record)
            self._next_index += 1
            self._write_summary_atomic()
            return record

    def _load_existing_records(self) -> list[dict[str, Any]]:
        if not self.records_path.exists():
            return []

        records: list[dict[str, Any]] = []
        with self.records_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {self.records_path} at line {line_number}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Expected an object in {self.records_path} at line {line_number}")
                records.append(record)
        return records

    def _validate_existing_run(self) -> None:
        for record in self._records:
            record_fps = record.get("fps", self.fps)
            record_beta = record.get("cfg_beta", self.cfg_beta)
            if float(record_fps) != self.fps or float(record_beta) != self.cfg_beta:
                raise ValueError(
                    f"Existing run {self.output_dir} uses fps={record_fps}, cfg_beta={record_beta}; "
                    f"requested fps={self.fps}, cfg_beta={self.cfg_beta}"
                )
            _validate_latency(record.get("wall_ms"), "wall_ms", required=True)
            _validate_latency(record.get("cuda_ms"), "cuda_ms", required=False)
            if not isinstance(record.get("warmup", False), bool):
                raise ValueError("Existing records must use a boolean warmup field")

    def _find_next_index(self) -> int:
        indices = [record.get("index") for record in self._records]
        integer_indices = [
            index for index in indices if isinstance(index, int) and not isinstance(index, bool)
        ]
        return max(integer_indices, default=-1) + 1

    def _load_created_at(self) -> str | None:
        if not self.summary_path.exists():
            return None
        try:
            with self.summary_path.open(encoding="utf-8") as stream:
                summary = json.load(stream)
        except (OSError, json.JSONDecodeError):
            return None
        created_at = summary.get("created_at_utc") if isinstance(summary, dict) else None
        return created_at if isinstance(created_at, str) else None

    def _save_chunks(self, index: int, chunks: dict[str, torch.Tensor] | None) -> str | None:
        if not self.save_chunks or chunks is None:
            return None
        if not isinstance(chunks, dict):
            raise TypeError("chunks must be a dict[str, torch.Tensor]")

        cpu_chunks: dict[str, torch.Tensor] = {}
        for name, tensor in chunks.items():
            if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
                raise TypeError("chunks must be a dict[str, torch.Tensor]")
            cpu_chunks[name] = tensor.detach().cpu()

        destination = self.chunks_dir / f"{index:06d}.pt"
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=self.chunks_dir
        )
        os.close(fd)
        try:
            torch.save(cpu_chunks, temporary_name)
            os.replace(temporary_name, destination)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
        return destination.relative_to(self.output_dir).as_posix()

    def _build_summary(self) -> dict[str, Any]:
        wall_values = [
            latency
            for record in self._records
            if (latency := _validate_latency(record.get("wall_ms"), "wall_ms", required=False)) is not None
        ]
        cuda_values = [
            latency
            for record in self._records
            if (latency := _validate_latency(record.get("cuda_ms"), "cuda_ms", required=False)) is not None
        ]
        steady_state_records = [record for record in self._records if not record.get("warmup", False)]
        steady_state_wall_values = [
            latency
            for record in steady_state_records
            if (latency := _validate_latency(record.get("wall_ms"), "wall_ms", required=False)) is not None
        ]
        steady_state_cuda_values = [
            latency
            for record in steady_state_records
            if (latency := _validate_latency(record.get("cuda_ms"), "cuda_ms", required=False)) is not None
        ]

        wall_stats = _latency_stats(wall_values)
        cuda_stats = _latency_stats(cuda_values)
        steady_state_wall_stats = _latency_stats(steady_state_wall_values)
        steady_state_cuda_stats = _latency_stats(steady_state_cuda_values)
        if steady_state_wall_stats["p99"] is not None:
            wall_p99 = steady_state_wall_stats["p99"]
            recommendation_source = "steady_state"
        elif wall_stats["p99"] is not None:
            wall_p99 = wall_stats["p99"]
            recommendation_source = "all_samples_fallback"
        else:
            wall_p99 = None
            recommendation_source = "unavailable"
        d99 = math.ceil(wall_p99 / 1000 * self.fps) if wall_p99 is not None else None
        precision_horizon = _clamp(d99 + 6, 10, 20) if d99 is not None else None
        smooth_horizon = _clamp(d99 + 8, 10, 20) if d99 is not None else None

        return {
            "run_name": self.run_name,
            "created_at_utc": self._created_at_utc,
            "updated_at_utc": _utc_string(),
            "count": len(self._records),
            "fps": self.fps,
            "cfg_beta": self.cfg_beta,
            "save_chunks": self.save_chunks,
            "wall_ms": wall_stats,
            "cuda_ms": cuda_stats,
            "steady_state_count": len(steady_state_records),
            "steady_state_wall_ms": steady_state_wall_stats,
            "steady_state_cuda_ms": steady_state_cuda_stats,
            "recommendation_source": recommendation_source,
            "D99": d99,
            "recommended_precision_horizon": precision_horizon,
            "recommended_smooth_horizon": smooth_horizon,
        }

    def _write_summary_atomic(self) -> None:
        summary = self._build_summary()
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{self.summary_path.name}.", suffix=".tmp", dir=self.output_dir
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(summary, stream, indent=2, ensure_ascii=False, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, self.summary_path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
