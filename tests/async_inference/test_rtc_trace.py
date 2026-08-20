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

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import torch

from lerobot.async_inference.rtc_trace import RTCTraceLogger
from lerobot.configs.types import RTCAttentionSchedule


def _read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_trace_appends_valid_metadata_only_jsonl(tmp_path):
    trace = RTCTraceLogger(role="client", output_dir=tmp_path)
    tensor = torch.zeros(3, 2, dtype=torch.float32)

    trace.record(
        "request_submitted",
        request_id="session:0:1",
        output=tmp_path,
        schedule=RTCAttentionSchedule.LINEAR,
        dimensions=(3, 2),
        tensor=tensor,
    )
    trace.record("chunk_installed", request_id="session:0:1", queue_size=47)
    trace.close()

    records = _read_records(trace.path)
    assert [record["event"] for record in records] == ["request_submitted", "chunk_installed"]
    assert all(record["schema_version"] == 1 for record in records)
    assert all(record["role"] == "client" for record in records)
    assert all(record["request_id"] == "session:0:1" for record in records)
    assert all(record["timestamp_utc"].endswith("Z") for record in records)
    assert all(record["monotonic_s"] > 0 for record in records)
    assert records[0]["schedule"] == "LINEAR"
    assert records[0]["dimensions"] == [3, 2]
    assert records[0]["tensor"] == {"shape": [3, 2], "dtype": "float32", "device": "cpu"}
    assert "tensor([[" not in trace.path.read_text(encoding="utf-8")


def test_trace_writes_complete_lines_from_multiple_threads(tmp_path):
    trace = RTCTraceLogger(role="server", output_dir=tmp_path)
    thread_count = 8
    events_per_thread = 25

    def write_events(worker: int) -> None:
        for sequence in range(events_per_thread):
            trace.record("event", worker=worker, sequence=sequence)

    threads = [threading.Thread(target=write_events, args=(worker,)) for worker in range(thread_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)
    trace.close()

    records = _read_records(trace.path)
    assert len(records) == thread_count * events_per_thread
    assert {(record["worker"], record["sequence"]) for record in records} == {
        (worker, sequence) for worker in range(thread_count) for sequence in range(events_per_thread)
    }


def test_trace_slow_disk_never_blocks_producer_and_reports_drops(tmp_path):
    trace = RTCTraceLogger(role="client", output_dir=tmp_path, queue_capacity=2)
    wrapped_stream = trace._stream
    writer_started = threading.Event()
    release_writer = threading.Event()

    class _SlowStream:
        def tell(self):
            return wrapped_stream.tell()

        def write(self, line):
            writer_started.set()
            release_writer.wait(timeout=2)
            return wrapped_stream.write(line)

        def flush(self):
            return wrapped_stream.flush()

        def close(self):
            return wrapped_stream.close()

    trace._stream = _SlowStream()
    trace.record("first")
    assert writer_started.wait(timeout=1)

    start = time.perf_counter()
    for sequence in range(100):
        trace.record("while_disk_is_slow", sequence=sequence)
    producer_elapsed = time.perf_counter() - start

    assert producer_elapsed < 0.1
    release_writer.set()
    trace.close()
    records = _read_records(trace.path)
    dropped = [record for record in records if record["event"] == "trace_events_dropped"]
    assert dropped
    assert sum(record["dropped_events"] for record in dropped) > 0


def test_trace_close_drains_a_full_queue_without_discarding_accepted_events(tmp_path):
    trace = RTCTraceLogger(role="client", output_dir=tmp_path, queue_capacity=3)
    wrapped_stream = trace._stream
    writer_started = threading.Event()
    release_writer = threading.Event()

    class _SlowStream:
        def tell(self):
            return wrapped_stream.tell()

        def write(self, line):
            writer_started.set()
            release_writer.wait(timeout=2)
            return wrapped_stream.write(line)

        def flush(self):
            return wrapped_stream.flush()

        def close(self):
            return wrapped_stream.close()

    trace._stream = _SlowStream()
    trace.record("first")
    assert writer_started.wait(timeout=1)
    for sequence in range(3):
        trace.record("queued", sequence=sequence)

    close_threads = [threading.Thread(target=trace.close) for _ in range(2)]
    for thread in close_threads:
        thread.start()
    time.sleep(0.05)
    assert all(thread.is_alive() for thread in close_threads)

    release_writer.set()
    for thread in close_threads:
        thread.join(timeout=2)
    assert all(not thread.is_alive() for thread in close_threads)

    records = _read_records(trace.path)
    assert [record["event"] for record in records] == ["first", "queued", "queued", "queued"]
    assert [record["sequence"] for record in records[1:]] == [0, 1, 2]


def test_trace_rotates_and_bounds_retained_files(tmp_path):
    trace = RTCTraceLogger(
        role="server",
        output_dir=tmp_path,
        max_file_bytes=512,
        max_files=2,
    )
    for sequence in range(40):
        trace.record("large_event", sequence=sequence, detail="x" * 100)
    trace.close()

    trace_files = list(tmp_path.glob("rtc_server_*.jsonl"))
    assert 1 <= len(trace_files) <= 2
    assert all(_read_records(path) for path in trace_files)


def test_trace_write_failure_does_not_escape_into_control_path(tmp_path, caplog):
    trace = RTCTraceLogger(role="client", output_dir=tmp_path)
    trace._stream.close()

    class _BrokenStream:
        def write(self, _line):
            raise OSError("disk full")

        def flush(self):
            raise OSError("disk full")

        def close(self):
            pass

    trace._stream = _BrokenStream()

    trace.record("request_submitted", request_id="safe")
    trace.record("request_submitted", request_id="still-safe")
    trace.close()

    assert "robot control will continue" in caplog.text


def test_trace_serialization_failure_drops_only_bad_event(tmp_path, monkeypatch, caplog):
    trace = RTCTraceLogger(role="client", output_dir=tmp_path)
    original_serialize = trace._serialize_record
    calls = 0

    def fail_once(event, fields):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TypeError("bad diagnostics field")
        return original_serialize(event, fields)

    monkeypatch.setattr(trace, "_serialize_record", fail_once)
    trace.record("bad")
    trace.record("good", request_id="still-recorded")
    status = trace.close()

    records = _read_records(trace.path)
    assert [record["event"] for record in records] == ["good"]
    assert status["serialization_errors"] == 1
    assert status["disabled"] is False
    assert "later events will still be recorded" in caplog.text
