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
import queue
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

_PERCENTILES = ("p50", "p95", "p99")
_STOP = object()


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
    """Append per-chunk ACP metrics without blocking inference on disk I/O.

    ``record`` validates and enqueues a record only.  A daemon writer owns the
    records file for the lifetime of the profiler, persists records in batches,
    and updates the aggregate summary periodically.  Disk failures are exposed
    through :attr:`writer_error`; they never propagate through ``record``.

    Args:
        output_root: Parent directory for profiling runs.
        fps: Robot control frequency, used to convert p99 latency to action steps.
        cfg_beta: CFG coefficient used by the profiled inference run.
        save_chunks: Whether tensors passed to ``record`` should be persisted.
        run_name: Optional stable run directory name. If omitted, a UTC timestamp
            and process ID are used. Reusing a name resumes the existing JSONL run.
        queue_capacity: Maximum number of records waiting for the writer. New
            records are dropped instead of blocking inference when it is full.
        write_batch_size: Flush the records stream after this many records.
        flush_interval_s: Flush a partial batch after this many seconds.
        summary_interval_records: Rewrite ``summary.json`` after this many newly
            persisted records. The final summary is always written by ``close``.
    """

    def __init__(
        self,
        output_root: str | Path,
        fps: float,
        cfg_beta: float,
        save_chunks: bool = True,
        run_name: str | None = None,
        queue_capacity: int = 256,
        write_batch_size: int = 20,
        flush_interval_s: float = 1.0,
        summary_interval_records: int = 100,
    ) -> None:
        if isinstance(fps, bool) or not math.isfinite(float(fps)) or float(fps) <= 0:
            raise ValueError("fps must be a finite number greater than zero")
        if isinstance(cfg_beta, bool) or not math.isfinite(float(cfg_beta)):
            raise ValueError("cfg_beta must be a finite number")
        for value, name in (
            (queue_capacity, "queue_capacity"),
            (write_batch_size, "write_batch_size"),
            (summary_interval_records, "summary_interval_records"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be an integer greater than zero")
        if (
            isinstance(flush_interval_s, bool)
            or not math.isfinite(float(flush_interval_s))
            or float(flush_interval_s) <= 0
        ):
            raise ValueError("flush_interval_s must be a finite number greater than zero")

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
        self.queue_capacity = queue_capacity
        self.write_batch_size = write_batch_size
        self.flush_interval_s = float(flush_interval_s)
        self.summary_interval_records = summary_interval_records
        self._state_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=queue_capacity)
        self._pending_records = 0
        self._dropped_records = 0
        self._writer_error: str | None = None
        self._closed = False
        self._close_completed = False
        self._stop_enqueued = False

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
        self._last_summary_count = len(self._records)
        self._writer = threading.Thread(
            target=self._writer_loop,
            name=f"acp-profiler-{self.run_name}",
            daemon=True,
        )
        self._writer.start()

    @property
    def next_index(self) -> int:
        """Return the index that will be assigned to the next recorded chunk."""
        with self._state_lock:
            return self._next_index

    @property
    def pending_records(self) -> int:
        """Number of accepted records not yet flushed by the writer."""
        with self._state_lock:
            return self._pending_records

    @property
    def dropped_records(self) -> int:
        """Number of records dropped due to backpressure or writer failure."""
        with self._state_lock:
            return self._dropped_records

    @property
    def writer_error(self) -> str | None:
        """First background writer failure, or ``None`` while healthy."""
        with self._state_lock:
            return self._writer_error

    @property
    def closed(self) -> bool:
        with self._state_lock:
            return self._closed

    @property
    def status(self) -> dict[str, Any]:
        """Return a snapshot suitable for health logs and shutdown reports."""
        with self._state_lock:
            return {
                "pending_records": self._pending_records,
                "dropped_records": self._dropped_records,
                "writer_error": self._writer_error,
                "closed": self._closed,
                "close_completed": self._close_completed,
                "stop_enqueued": self._stop_enqueued,
                "writer_alive": self._writer.is_alive(),
            }

    def record(
        self,
        metrics: dict[str, Any],
        chunks: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        """Enqueue one chunk inference and return its JSON-compatible record.

        Validation may still reject invalid caller data, but this method never
        performs file I/O and never raises because the writer or queue failed.
        """
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

        chunk_snapshot = self._prepare_chunks(chunks)
        with self._close_lock, self._state_lock:
            index = self._next_index
            self._next_index += 1
            chunk_file = (
                (self.chunks_dir / f"{index:06d}.pt").relative_to(self.output_dir).as_posix()
                if self.save_chunks and chunk_snapshot is not None
                else None
            )
            record = {
                **json_metrics,
                "index": index,
                "recorded_at_utc": _utc_string(),
                "fps": self.fps,
                "cfg_beta": self.cfg_beta,
                "chunk_file": chunk_file,
            }
            serialized = json.dumps(record, ensure_ascii=False, allow_nan=False)
            if self._closed or self._writer_error is not None:
                self._dropped_records += 1
                return record
            self._pending_records += 1
            try:
                self._queue.put_nowait((record, serialized, chunk_snapshot))
            except queue.Full:
                self._pending_records -= 1
                self._dropped_records += 1
        return record

    def close(self, timeout_s: float = 5.0) -> bool:
        """Drain accepted records, write a final summary, and stop the writer.

        The wait is bounded so a stuck filesystem cannot hang server shutdown.
        ``False`` means the writer is still alive or records remain pending;
        callers may inspect :attr:`status` and retry. This method is idempotent.
        """
        if isinstance(timeout_s, bool) or not math.isfinite(float(timeout_s)) or float(timeout_s) < 0:
            raise ValueError("timeout_s must be a finite, non-negative number")
        deadline = time.monotonic() + float(timeout_s)
        if not self._close_lock.acquire(timeout=max(0.0, deadline - time.monotonic())):
            return False
        try:
            with self._state_lock:
                if self._close_completed and not self._writer.is_alive():
                    return True
                self._closed = True

            while self._writer.is_alive() and not self._stop_enqueued:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                try:
                    self._queue.put(_STOP, timeout=min(0.1, remaining))
                    with self._state_lock:
                        self._stop_enqueued = True
                except queue.Full:
                    continue

            remaining = max(0.0, deadline - time.monotonic())
            self._writer.join(timeout=remaining)
            if self._writer.is_alive():
                return False

            # A writer can fail between the is_alive() check and consuming the
            # stop token. Reclaim anything it left behind before waiting on the
            # queue's unfinished-task counter.
            abandoned_records = 0
            while True:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    if item is not _STOP:
                        abandoned_records += 1
                finally:
                    self._queue.task_done()
            if abandoned_records:
                with self._state_lock:
                    self._pending_records = max(0, self._pending_records - abandoned_records)
                    self._dropped_records += abandoned_records
            with self._state_lock:
                self._close_completed = self._pending_records == 0
                return self._close_completed
        finally:
            self._close_lock.release()

    def __enter__(self) -> ACPInferenceProfiler:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def _prepare_chunks(self, chunks: dict[str, torch.Tensor] | None) -> dict[str, torch.Tensor] | None:
        if not self.save_chunks or chunks is None:
            return None
        if not isinstance(chunks, dict):
            raise TypeError("chunks must be a dict[str, torch.Tensor]")
        snapshot: dict[str, torch.Tensor] = {}
        for name, tensor in chunks.items():
            if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
                raise TypeError("chunks must be a dict[str, torch.Tensor]")
            # Keep the storage alive until the background writer performs D2H.
            # Inference outputs are immutable after record() in both callers.
            snapshot[name] = tensor.detach()
        return snapshot

    def _writer_loop(self) -> None:
        """Own the append stream and service queued records until ``close``."""
        batch: list[tuple[dict[str, Any], str, dict[str, torch.Tensor] | None]] = []
        try:
            with self.records_path.open("a", encoding="utf-8", newline="\n") as stream:
                last_flush = time.monotonic()
                while True:
                    if batch:
                        timeout = max(0.0, self.flush_interval_s - (time.monotonic() - last_flush))
                    else:
                        timeout = self.flush_interval_s
                    try:
                        item = self._queue.get(timeout=timeout)
                    except queue.Empty:
                        if batch:
                            self._persist_batch(stream, batch)
                            batch = []
                            last_flush = time.monotonic()
                        continue

                    if item is _STOP:
                        try:
                            if batch:
                                self._persist_batch(stream, batch)
                                batch = []
                            stream.flush()
                            os.fsync(stream.fileno())
                            self._write_summary_atomic()
                            self._last_summary_count = self._persisted_count()
                        finally:
                            self._queue.task_done()
                        with self._state_lock:
                            self._close_completed = self._pending_records == 0
                        return

                    batch.append(item)
                    if len(batch) >= self.write_batch_size:
                        self._persist_batch(stream, batch)
                        batch = []
                        last_flush = time.monotonic()
        except Exception as exc:
            # Profiling is observational: a broken output filesystem must not
            # turn a successful action inference into an RPC failure.
            self._set_writer_failure(exc, batch)

    def _persist_batch(
        self,
        stream: Any,
        batch: list[tuple[dict[str, Any], str, dict[str, torch.Tensor] | None]],
    ) -> None:
        for record, _, chunks in batch:
            if chunks is not None:
                self._save_chunks(record["index"], chunks)

        stream.write("".join(f"{serialized}\n" for _, serialized, _ in batch))
        stream.flush()

        with self._state_lock:
            self._records.extend(record for record, _, _ in batch)
            self._pending_records -= len(batch)
            persisted_count = len(self._records)

        for _ in batch:
            self._queue.task_done()

        if persisted_count - self._last_summary_count >= self.summary_interval_records:
            try:
                self._write_summary_atomic()
                self._last_summary_count = persisted_count
            except Exception as exc:
                # Records remain usable even when the convenience summary
                # cannot be refreshed. Stop accepting new work, but do not let
                # the error escape the writer thread or inference caller.
                with self._state_lock:
                    if self._writer_error is None:
                        self._writer_error = f"{type(exc).__name__}: {exc}"

    def _set_writer_failure(
        self,
        exc: Exception,
        active_batch: list[tuple[dict[str, Any], str, dict[str, torch.Tensor] | None]],
    ) -> None:
        error = f"{type(exc).__name__}: {exc}"
        # Publish the failure before draining so concurrent record() calls stop
        # enqueueing work that this writer can no longer consume.
        with self._state_lock:
            if self._writer_error is None:
                self._writer_error = error

        failed_items = len(active_batch)
        for _ in active_batch:
            self._queue.task_done()

        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            try:
                if item is not _STOP:
                    failed_items += 1
            finally:
                self._queue.task_done()

        with self._state_lock:
            self._pending_records = max(0, self._pending_records - failed_items)
            self._dropped_records += failed_items
            if self._closed and self._pending_records == 0:
                self._close_completed = True

    def _persisted_count(self) -> int:
        with self._state_lock:
            return len(self._records)

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
        with self._state_lock:
            records = list(self._records)
            pending_records = self._pending_records
            dropped_records = self._dropped_records
            writer_error = self._writer_error
        wall_values = [
            latency
            for record in records
            if (latency := _validate_latency(record.get("wall_ms"), "wall_ms", required=False)) is not None
        ]
        cuda_values = [
            latency
            for record in records
            if (latency := _validate_latency(record.get("cuda_ms"), "cuda_ms", required=False)) is not None
        ]
        steady_state_records = [record for record in records if not record.get("warmup", False)]
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
            "count": len(records),
            "fps": self.fps,
            "cfg_beta": self.cfg_beta,
            "save_chunks": self.save_chunks,
            "pending_records": pending_records,
            "dropped_records": dropped_records,
            "writer_error": writer_error,
            "wall_ms": wall_stats,
            "cuda_ms": cuda_stats,
            "steady_state_count": len(steady_state_records),
            "steady_state_wall_ms": steady_state_wall_stats,
            "steady_state_cuda_ms": steady_state_cuda_stats,
            "recommendation_source": recommendation_source,
            "latency_scope": "server_pipeline_only",
            "server_compute_D99": d99,
            "D99": d99,
            "D99_semantics": (
                "legacy ceil(server pipeline wall p99 * configured fps); not the remote RTC inference_delay"
            ),
            "recommended_precision_horizon": precision_horizon,
            "recommended_smooth_horizon": smooth_horizon,
            "rtc_parameter_recommendation": {
                "available": False,
                "reason": (
                    "Join client actual_delay, executed control rate, queue coverage, and end-to-end "
                    "latency by request_id before selecting RTC inference_delay/execution_horizon."
                ),
            },
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
