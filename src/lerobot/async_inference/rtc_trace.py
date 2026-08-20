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

"""Low-overhead, crash-oriented traces for remote RTC inference.

The trace intentionally contains metadata only. Images, prompts, actions, and
other tensor contents are never serialized. Client and server files can be
joined by ``request_id`` when debugging a failed robot run.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import socket
import threading
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any

import torch


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _safe_filename_part(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return sanitized or "unknown"


def _jsonable(value: Any) -> Any:
    """Return bounded metadata that can be encoded with ``allow_nan=False``."""
    if value is None or isinstance(value, str | int | bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, torch.Tensor):
        return tensor_metadata(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list | set):
        return [_jsonable(item) for item in value]
    return repr(value)


def tensor_metadata(tensor: Any | None) -> dict[str, Any] | None:
    """Describe a tensor without reading its values or synchronizing CUDA."""
    if tensor is None:
        return None
    if not isinstance(tensor, torch.Tensor):
        return {"type": type(tensor).__name__, "is_tensor": False}
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "device": str(tensor.device),
    }


class RTCTraceLogger:
    """Non-blocking JSONL trace with a bounded queue and file rotation."""

    schema_version = 1
    default_max_file_bytes = 64 * 1024 * 1024
    default_max_files = 4
    default_queue_capacity = 4096

    def __init__(
        self,
        role: str,
        output_dir: str | Path,
        *,
        max_file_bytes: int = default_max_file_bytes,
        max_files: int = default_max_files,
        queue_capacity: int = default_queue_capacity,
    ) -> None:
        if max_file_bytes <= 0 or max_files <= 0 or queue_capacity <= 0:
            raise ValueError("RTC trace rotation and queue limits must be positive integers.")
        self.role = role
        self.output_dir = Path(output_dir)
        self.max_file_bytes = max_file_bytes
        self.max_files = max_files
        self._state_lock = threading.Lock()
        self._closed = False
        self._disabled = False
        self._failure_reported = False
        self._serialization_failure_reported = False
        self._dropped_events = 0
        self._total_dropped_events = 0
        self._serialization_errors = 0
        self._lost_events_after_writer_failure = 0
        self._queue: Queue[str] = Queue(maxsize=queue_capacity)
        self._close_complete = threading.Event()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        hostname = _safe_filename_part(socket.gethostname())
        role_part = _safe_filename_part(role)
        self._file_stem = f"rtc_{role_part}_{hostname}_{os.getpid()}_{timestamp}"
        self._segment_index = 0
        self._segment_paths: list[Path] = []
        self.path = self._segment_path(self._segment_index)
        self._stream = self.path.open("a", encoding="utf-8", buffering=1, newline="\n")
        self._segment_paths.append(self.path)
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name=f"rtc-{role_part}-trace-writer",
            daemon=True,
        )
        self._writer_thread.start()

    def _segment_path(self, index: int) -> Path:
        suffix = "" if index == 0 else f"_part{index:04d}"
        return self.output_dir / f"{self._file_stem}{suffix}.jsonl"

    def _serialize_record(self, event: str, fields: Mapping[str, Any]) -> str:
        record = {
            **{key: _jsonable(value) for key, value in fields.items()},
            "schema_version": self.schema_version,
            "timestamp_utc": _utc_timestamp(),
            "monotonic_s": time.perf_counter(),
            "role": self.role,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "thread": threading.current_thread().name,
            "event": event,
        }
        return json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":"))

    def record(self, event: str, **fields: Any) -> None:
        """Enqueue one event without waiting for filesystem I/O."""
        with self._state_lock:
            if self._closed or self._disabled:
                return
        try:
            line = self._serialize_record(event, fields)
        except Exception:
            # A malformed diagnostic field must not permanently disable all
            # later traces. Only writer/filesystem failures disable persistence.
            with self._state_lock:
                self._serialization_errors += 1
                report_failure = not self._serialization_failure_reported
                self._serialization_failure_reported = True
            if report_failure:
                logging.getLogger(__name__).warning(
                    "Could not serialize one RTC trace event; later events will still be recorded.",
                    exc_info=True,
                )
            return

        with self._state_lock:
            if self._closed or self._disabled:
                return
            try:
                self._queue.put_nowait(line)
            except Full:
                # Never block robot control on diagnostics. The writer emits
                # a marker with this count after it catches up.
                self._dropped_events += 1
                self._total_dropped_events += 1

    def _take_dropped_events(self) -> int:
        with self._state_lock:
            dropped = self._dropped_events
            self._dropped_events = 0
            return dropped

    def _write_line(self, line: str) -> None:
        line_bytes = len(line.encode("utf-8")) + 1
        if self._stream.tell() > 0 and self._stream.tell() + line_bytes > self.max_file_bytes:
            self._rotate()
        self._stream.write(line + "\n")
        self._stream.flush()

    def _rotate(self) -> None:
        self._stream.flush()
        self._stream.close()
        self._segment_index += 1
        self.path = self._segment_path(self._segment_index)
        self._stream = self.path.open("a", encoding="utf-8", buffering=1, newline="\n")
        self._segment_paths.append(self.path)
        while len(self._segment_paths) > self.max_files:
            expired_path = self._segment_paths.pop(0)
            expired_path.unlink(missing_ok=True)

    def _write_dropped_marker(self, dropped: int) -> None:
        if dropped <= 0:
            return
        marker = self._serialize_record("trace_events_dropped", {"dropped_events": dropped})
        self._write_line(marker)

    def _writer_loop(self) -> None:
        try:
            while True:
                try:
                    item = self._queue.get(timeout=0.1)
                except Empty:
                    with self._state_lock:
                        should_close = self._closed and self._queue.empty()
                    if should_close:
                        self._write_dropped_marker(self._take_dropped_events())
                        return
                    continue
                try:
                    dropped = self._take_dropped_events()
                    self._write_dropped_marker(dropped)
                    self._write_line(item)
                finally:
                    self._queue.task_done()
        except Exception:
            with self._state_lock:
                self._lost_events_after_writer_failure += self._queue.qsize()
            self._report_failure()
        finally:
            try:
                self._stream.flush()
                self._stream.close()
            except Exception:
                self._report_failure()
            finally:
                self._close_complete.set()

    def _report_failure(self) -> None:
        with self._state_lock:
            if self._failure_reported:
                return
            self._failure_reported = True
            self._disabled = True
        logging.getLogger(__name__).warning(
            "RTC trace persistence failed; robot control will continue without structured traces.",
            exc_info=True,
        )

    @property
    def status(self) -> dict[str, Any]:
        """Return a best-effort diagnostics snapshot without touching disk."""
        with self._state_lock:
            return {
                "closed": self._closed,
                "disabled": self._disabled,
                "pending_events": self._queue.qsize(),
                "unreported_dropped_events": self._dropped_events,
                "dropped_events": self._total_dropped_events,
                "serialization_errors": self._serialization_errors,
                "lost_events_after_writer_failure": self._lost_events_after_writer_failure,
                "writer_alive": self._writer_thread.is_alive(),
            }

    def close(self, timeout_s: float = 2.0) -> dict[str, Any]:
        """Request a drain and wait briefly without risking an unbounded stop."""
        with self._state_lock:
            self._closed = True
        completed = self._close_complete.wait(timeout=max(timeout_s, 0.0))
        if not completed:
            logging.getLogger(__name__).warning(
                "RTC trace writer did not drain within %.2fs; shutdown will continue.", timeout_s
            )
        else:
            self._writer_thread.join(timeout=0)
        return self.status


def create_rtc_trace(role: str, output_dir: str | Path) -> RTCTraceLogger | None:
    """Create a best-effort trace writer without making logging a startup dependency."""
    try:
        return RTCTraceLogger(role=role, output_dir=output_dir)
    except Exception:
        logging.getLogger(__name__).warning(
            "Could not create RTC %s trace under %s; continuing without it.",
            role,
            output_dir,
            exc_info=True,
        )
        return None
