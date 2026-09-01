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

"""Profile real frontend/backend RTC delay d and recommend a training limit."""

import argparse
import json
import math
import os
import re
import socket
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from lerobot.policies.rtc.delay_telemetry import parse_udp_address

CONFIG_NAME = "config.json"
DEFAULT_LISTEN_ADDRESS = "127.0.0.1:9080"
DEFAULT_DURATION_S = 600.0
OVERFLOW_EXIT_CODE = 2
NO_SAMPLES_EXIT_CODE = 3


@dataclass(frozen=True)
class RTCPolicyLimits:
    chunk_size: int
    rtc_training_max_delay: int | None = None
    policy_type: str | None = None
    config_path: str | None = None


def latency_ms_to_delay_steps(latency_ms: float, control_hz: float) -> int:
    """Convert elapsed wall time to the number of controller ticks d."""
    if not math.isfinite(latency_ms) or latency_ms < 0:
        raise ValueError(f"latency_ms must be finite and non-negative, got {latency_ms}.")
    if not math.isfinite(control_hz) or control_hz <= 0:
        raise ValueError(f"control_hz must be finite and positive, got {control_hz}.")
    return math.ceil(latency_ms * control_hz / 1000.0)


def _resolve_local_config(path: Path) -> Path:
    if path.is_file():
        return path

    candidates = (
        path / CONFIG_NAME,
        path / "pretrained_model" / CONFIG_NAME,
        path / "checkpoints" / "last" / "pretrained_model" / CONFIG_NAME,
        path / "checkpoints" / "last" / CONFIG_NAME,
        path / "last" / "pretrained_model" / CONFIG_NAME,
        path / "last" / CONFIG_NAME,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    checked = "\n  - ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find {CONFIG_NAME} under {path}. Checked:\n  - {checked}")


def resolve_policy_config(policy_path: str, revision: str | None = None) -> Path:
    """Resolve a local checkpoint/output directory or a Hugging Face policy id."""
    path = Path(policy_path).expanduser()
    if path.exists():
        return _resolve_local_config(path)

    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(repo_id=policy_path, filename=CONFIG_NAME, revision=revision)
    return Path(downloaded)


def load_policy_limits(policy_path: str, revision: str | None = None) -> RTCPolicyLimits:
    config_path = resolve_policy_config(policy_path, revision)
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in policy config {config_path}: {exc}") from exc

    policy = payload.get("policy", payload)
    if not isinstance(policy, dict):
        raise ValueError(f"Policy config in {config_path} must be a JSON object.")
    if "chunk_size" not in policy:
        raise ValueError(f"Policy config {config_path} is missing 'chunk_size'.")

    policy_type = policy.get("type")
    if policy_type is not None and policy_type != "pi05":
        raise ValueError(f"RTC trained-delay profiling requires a PI0.5 config, got type={policy_type!r}.")

    training_max_delay = policy.get("rtc_training_max_delay")
    if training_max_delay is not None:
        training_max_delay = int(training_max_delay)
        # Zero means Training-Time RTC is disabled, not that d=0 is a trained deployment limit.
        if training_max_delay == 0:
            training_max_delay = None
    return RTCPolicyLimits(
        chunk_size=int(policy["chunk_size"]),
        rtc_training_max_delay=training_max_delay,
        policy_type=policy_type,
        config_path=str(config_path.resolve()),
    )


def validate_policy_limits(limits: RTCPolicyLimits) -> None:
    if limits.chunk_size <= 1:
        raise ValueError(f"chunk_size must be greater than 1, got {limits.chunk_size}.")
    if limits.rtc_training_max_delay is not None and not (
        0 <= limits.rtc_training_max_delay < limits.chunk_size
    ):
        raise ValueError(
            "rtc_training_max_delay must satisfy "
            f"0 <= delay < chunk_size ({limits.chunk_size}), got {limits.rtc_training_max_delay}."
        )


def percentile(values: list[int], q: float) -> float:
    """Compute a linearly interpolated percentile without a NumPy dependency."""
    if not values:
        raise ValueError("Cannot compute a percentile without samples.")
    if not 0 <= q <= 100:
        raise ValueError(f"percentile must be within [0, 100], got {q}.")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(ordered[lower])
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_delay_distribution(
    delay_steps: list[int],
    limits: RTCPolicyLimits,
    recommend_percentile: float,
    safety_margin_steps: int,
) -> dict[str, Any]:
    """Build distribution, boundary checks, and a suggested training D."""
    validate_policy_limits(limits)
    if not delay_steps:
        raise ValueError("At least one d sample is required.")
    if any(not isinstance(delay, int) or isinstance(delay, bool) or delay < 0 for delay in delay_steps):
        raise ValueError(f"d samples must be non-negative integers, got {delay_steps}.")
    if not 0 < recommend_percentile <= 100:
        raise ValueError(f"recommend_percentile must be within (0, 100], got {recommend_percentile}.")
    if safety_margin_steps < 0:
        raise ValueError(f"safety_margin_steps must be non-negative, got {safety_margin_steps}.")

    counts = Counter(delay_steps)
    sample_count = len(delay_steps)
    maximum = max(delay_steps)
    p_value = percentile(delay_steps, recommend_percentile)
    recommended_uncapped = math.ceil(p_value) + safety_margin_steps
    recommended = min(recommended_uncapped, limits.chunk_size - 1)

    training_limit = limits.rtc_training_max_delay
    if maximum >= limits.chunk_size:
        status = "chunk_exhausted"
        message = (
            f"Observed max d={maximum} reaches/exceeds chunk_size={limits.chunk_size}; "
            "increase the chunk or reduce end-to-end latency."
        )
    elif training_limit is not None and maximum > training_limit:
        status = "training_range_exceeded"
        message = (
            f"Observed max d={maximum} exceeds the existing checkpoint training limit D={training_limit}."
        )
    else:
        status = "profiled" if training_limit is None else "pass"
        message = (
            f"Observed max d={maximum}; suggested training D={recommended} "
            f"from p{recommend_percentile:g} plus {safety_margin_steps} safety steps."
        )

    histogram = {
        str(delay): {"count": count, "fraction": count / sample_count}
        for delay, count in sorted(counts.items())
    }
    quantiles = {
        "p50": percentile(delay_steps, 50),
        "p90": percentile(delay_steps, 90),
        "p95": percentile(delay_steps, 95),
        "p99": percentile(delay_steps, 99),
        "p99_9": percentile(delay_steps, 99.9),
        "p100": float(maximum),
    }
    overflow_count = (
        sum(delay > training_limit for delay in delay_steps) if training_limit is not None else None
    )

    return {
        "status": status,
        "safe": status in {"pass", "profiled"},
        "message": message,
        "sample_count": sample_count,
        "min_d": min(delay_steps),
        "mean_d": sum(delay_steps) / sample_count,
        "max_observed_d": maximum,
        "quantiles": quantiles,
        "histogram": histogram,
        "recommend_percentile": recommend_percentile,
        "safety_margin_steps": safety_margin_steps,
        "recommendation_basis_d": p_value,
        "recommended_rtc_training_max_delay": recommended,
        "recommended_uncapped_d": recommended_uncapped,
        "recommendation_was_capped": recommended_uncapped >= limits.chunk_size,
        "requires_larger_chunk": maximum >= limits.chunk_size,
        "max_structural_d": limits.chunk_size - 1,
        "existing_training_overflow_count": overflow_count,
        "existing_training_overflow_fraction": (
            None if overflow_count is None else overflow_count / sample_count
        ),
        "training_headroom_steps": None if training_limit is None else training_limit - maximum,
    }


def load_latency_file(path_value: str) -> list[float]:
    """Read latency milliseconds from JSON or comma/whitespace-delimited text."""
    if path_value == "-":
        raw = sys.stdin.read()
        source = "stdin"
    else:
        path = Path(path_value).expanduser()
        raw = path.read_text(encoding="utf-8")
        source = str(path)

    raw = raw.strip()
    if not raw:
        raise ValueError(f"Latency input {source} is empty.")

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        values: Any = [part for part in re.split(r"[,\s]+", raw) if part]
    else:
        if isinstance(payload, dict):
            if "latencies_ms" in payload:
                values = payload["latencies_ms"]
            elif "latency_ms" in payload:
                values = [payload["latency_ms"]]
            else:
                raise ValueError(
                    f"Latency JSON object in {source} must contain 'latencies_ms' or 'latency_ms'."
                )
        else:
            values = payload

    if not isinstance(values, list):
        raise ValueError(f"Latency input {source} must contain a list of millisecond values.")
    try:
        latencies = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Latency input {source} contains a non-numeric value.") from exc

    for latency in latencies:
        if not math.isfinite(latency) or latency < 0:
            raise ValueError(f"Latency values must be finite and non-negative, got {latency} in {source}.")
    return latencies


def parse_telemetry_datagram(data: bytes, fallback_control_hz: float | None) -> dict[str, Any]:
    try:
        event = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("RTC telemetry datagram must be a UTF-8 JSON object.") from exc
    if not isinstance(event, dict):
        raise ValueError("RTC telemetry datagram must be a JSON object.")

    if "d" in event:
        delay = event["d"]
        if not isinstance(delay, int) or isinstance(delay, bool) or delay < 0:
            raise ValueError(f"Telemetry d must be a non-negative integer, got {delay!r}.")
    elif "latency_ms" in event:
        control_hz = event.get("control_hz", event.get("fps", fallback_control_hz))
        if control_hz is None:
            raise ValueError("Telemetry with latency_ms requires control_hz/fps in the event or CLI.")
        delay = latency_ms_to_delay_steps(float(event["latency_ms"]), float(control_hz))
        event["d"] = delay
        event["control_hz"] = float(control_hz)
    else:
        raise ValueError("RTC telemetry requires either 'd' or 'latency_ms'.")
    return event


def collect_udp_samples(
    address: str,
    duration_s: float,
    fallback_control_hz: float | None,
    progress_interval_s: float,
) -> list[dict[str, Any]]:
    """Collect RTC telemetry events for a fixed duration, stopping cleanly on Ctrl-C."""
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError(f"duration_s must be finite and positive, got {duration_s}.")
    if not math.isfinite(progress_interval_s) or progress_interval_s <= 0:
        raise ValueError(f"progress_interval_s must be finite and positive, got {progress_interval_s}.")

    host, port = parse_udp_address(address)
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind((host, port))
    receiver.settimeout(1.0)
    events: list[dict[str, Any]] = []
    start = time.monotonic()
    deadline = start + duration_s
    next_progress = start + progress_interval_s

    print(
        f"Listening for RTC d telemetry on udp://{address} for {duration_s:g}s. Press Ctrl-C to stop early.",
        file=sys.stderr,
    )
    try:
        while True:
            now = time.monotonic()
            if now >= deadline:
                break
            receiver.settimeout(min(1.0, deadline - now))
            try:
                data, peer = receiver.recvfrom(65535)
            except TimeoutError:
                data = None
            if data is not None:
                try:
                    event = parse_telemetry_datagram(data, fallback_control_hz)
                except ValueError as exc:
                    print(f"Ignoring invalid telemetry from {peer[0]}:{peer[1]}: {exc}", file=sys.stderr)
                else:
                    event.setdefault("collector_received_at", time.time())
                    events.append(event)

            now = time.monotonic()
            if now >= next_progress:
                elapsed = now - start
                maximum = max((event["d"] for event in events), default="n/a")
                print(
                    f"RTC d monitor: elapsed={elapsed:.0f}s samples={len(events)} max_d={maximum}",
                    file=sys.stderr,
                )
                next_progress = now + progress_interval_s
    except KeyboardInterrupt:
        print("RTC d monitor stopped by user; summarizing collected samples.", file=sys.stderr)
    finally:
        receiver.close()
    return events


def _has_offline_samples(args: argparse.Namespace) -> bool:
    return bool(args.latency_ms or args.latency_file or args.delay_steps)


def _is_headless_mode(args: argparse.Namespace) -> bool:
    return bool(args.policy_path and args.listen is None and not _has_offline_samples(args))


def _ensure_tcp_port_available(host: str, port: int) -> None:
    if not 1 <= port <= 65535:
        raise ValueError(f"Server port must be within [1, 65535], got {port}.")
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind((host, port))
    except OSError as exc:
        raise OSError(f"TCP port {host}:{port} is already in use.") from exc
    finally:
        probe.close()


def _server_port_from_address(server_address: str) -> int:
    host, separator, port_text = server_address.rpartition(":")
    if not separator or not host or not port_text.isdecimal():
        raise ValueError(
            f"Server address must use host:port form, for example 127.0.0.1:8090; got {server_address!r}."
        )
    port = int(port_text)
    if not 1 <= port <= 65535:
        raise ValueError(f"Server port must be within [1, 65535], got {port}.")
    return port


def _server_policy_reference(policy_path: str, config_path: str | None) -> str:
    local_path = Path(policy_path).expanduser()
    if local_path.exists():
        if config_path is None:
            raise ValueError(f"Could not resolve a config for local policy path {policy_path}.")
        return str(Path(config_path).parent)
    return policy_path


def _run_countdown(
    deadline: float,
    samples: list[dict[str, Any]],
    stop_event: threading.Event,
) -> None:
    while not stop_event.is_set():
        remaining = max(0, math.ceil(deadline - time.monotonic()))
        minutes, seconds = divmod(remaining, 60)
        delays = [event["d"] for event in samples]
        last_delay: int | str = delays[-1] if delays else "-"
        max_delay: int | str = max(delays) if delays else "-"
        p99: str = f"{percentile(delays, 99):.1f}" if delays else "-"
        line = (
            f"\rRemaining {minutes:02d}:{seconds:02d} | samples={len(delays)} | "
            f"last_d={last_delay} | p99={p99} | max_d={max_delay}"
        )
        print(line.ljust(110), end="", flush=True)
        if remaining == 0:
            break
        stop_event.wait(1.0)


def run_integrated_headless_profile(
    args: argparse.Namespace,
    limits: RTCPolicyLimits,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Profile a private or external server without constructing a Robot."""
    if args.gpu_id is None or args.gpu_id < 0:
        raise ValueError("Headless profiling requires --gpu-id=<non-negative physical GPU index>.")
    if limits.config_path is None:
        raise ValueError("Headless profiling could not resolve the policy config path.")
    if args.fps <= 0:
        raise ValueError(f"fps must be positive, got {args.fps}.")

    from lerobot.async_inference.rtc_d_profiler import (
        build_synthetic_observation_spec,
        run_headless_profile_client,
        wait_for_grpc_server,
    )

    external_server = args.server_address is not None
    host = "127.0.0.1"
    server_address = args.server_address.strip() if external_server else f"{host}:{args.server_port}"
    server_port = _server_port_from_address(server_address)
    actions_per_chunk = args.actions_per_chunk or limits.chunk_size
    policy_reference = _server_policy_reference(args.policy_path, limits.config_path)
    observation_spec = build_synthetic_observation_spec(
        Path(limits.config_path),
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        task=args.task,
    )

    server_log_path: Path | None = None
    server_log = None
    server_process: subprocess.Popen | None = None
    if external_server:
        device = f"cuda:{args.gpu_id}"
        print(
            f"Connecting to PI0.5 server at {server_address}; requesting {device}. "
            "Robot actions are disabled."
        )
    else:
        _ensure_tcp_port_available(host, args.server_port)
        device = "cuda"
        server_log_path = args.server_log.expanduser()
        server_log_path.parent.mkdir(parents=True, exist_ok=True)
        server_command = [
            sys.executable,
            "-m",
            "lerobot.async_inference.policy_server",
            f"--host={host}",
            f"--port={args.server_port}",
            f"--fps={args.fps}",
            "--inference_latency=0",
            "--obs_queue_timeout=5",
        ]
        server_environment = os.environ.copy()
        server_environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        server_environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        server_environment.setdefault("PYTHONIOENCODING", "utf-8")

        print(
            f"Starting local PI0.5 server at {server_address} on physical GPU {args.gpu_id}; "
            "robot actions are disabled."
        )
        print(f"Server log: {server_log_path.resolve()}")
        server_log = server_log_path.open("w", encoding="utf-8")
        server_process = subprocess.Popen(  # noqa: S603
            server_command,
            env=server_environment,
            stdout=server_log,
            stderr=subprocess.STDOUT,
        )

    countdown_samples: list[dict[str, Any]] = []
    countdown_stop = threading.Event()
    countdown_thread: threading.Thread | None = None

    def start_countdown(deadline: float) -> None:
        nonlocal countdown_thread
        print(
            f"Warm-up complete. Profiling for {args.duration_s:g}s with {args.fps} Hz control timing; "
            "returned actions will be discarded."
        )
        countdown_thread = threading.Thread(
            target=_run_countdown,
            args=(deadline, countdown_samples, countdown_stop),
            daemon=True,
            name="rtc-d-countdown",
        )
        countdown_thread.start()

    try:
        wait_for_grpc_server(server_address, args.server_startup_timeout_s)
        print(
            f"Loading policy and running {args.warmup_steps} warm-up requests "
            f"({observation_spec.camera_count} cameras, state_dim={observation_spec.state_dim})..."
        )
        events = run_headless_profile_client(
            server_address=server_address,
            policy_path=policy_reference,
            observation_spec=observation_spec,
            actions_per_chunk=actions_per_chunk,
            fps=args.fps,
            duration_s=args.duration_s,
            warmup_steps=args.warmup_steps,
            request_timeout_s=args.request_timeout_s,
            device=device,
            on_profile_start=start_countdown,
            on_sample=countdown_samples.append,
        )
    finally:
        countdown_stop.set()
        if countdown_thread is not None:
            countdown_thread.join(timeout=2)
            print()
        if server_process is not None and server_process.poll() is None:
            server_process.terminate()
            try:
                server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_process.kill()
                server_process.wait(timeout=5)
        if server_log is not None:
            server_log.close()

    metadata = {
        "gpu_id": args.gpu_id,
        "server_address": server_address,
        "server_port": server_port,
        "server_mode": "external" if external_server else "private",
        "requested_device": device,
        "fps": args.fps,
        "duration_s": args.duration_s,
        "warmup_steps": args.warmup_steps,
        "actions_per_chunk": actions_per_chunk,
        "camera_width": args.camera_width,
        "camera_height": args.camera_height,
        "camera_count": observation_spec.camera_count,
        "state_dim": observation_spec.state_dim,
        "robot_actions_sent": False,
        "server_log": None if server_log_path is None else str(server_log_path.resolve()),
    }
    return events, metadata


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Record real frontend/backend RTC delay d (default 10 minutes), show its distribution, and "
            "recommend policy.rtc_training_max_delay."
        )
    )
    parser.add_argument(
        "--listen",
        default=None,
        help=f"UDP telemetry listen address. Defaults to {DEFAULT_LISTEN_ADDRESS} in live mode.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=DEFAULT_DURATION_S,
        help=f"Live profiling duration in seconds (default: {DEFAULT_DURATION_S:g}).",
    )
    parser.add_argument(
        "--progress-interval-s",
        type=float,
        default=30.0,
        help="Live progress reporting interval in seconds.",
    )
    parser.add_argument(
        "--policy-path",
        help="Local config/checkpoint/output directory or Hugging Face policy id.",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="GPU index: physical GPU for a private server, or requested cuda:<id> for an external server.",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=8090,
        help="Private local gRPC server port in one-command headless mode (default: 8090).",
    )
    parser.add_argument(
        "--server-address",
        default=None,
        help="Connect to an already running server, e.g. 127.0.0.1:8090; GPU is then requested as cuda:<gpu-id>.",
    )
    parser.add_argument("--fps", type=int, default=30, help="Controller frequency (default: 30).")
    parser.add_argument(
        "--actions-per-chunk",
        type=int,
        default=None,
        help="Returned actions per request; defaults to checkpoint chunk_size.",
    )
    parser.add_argument("--warmup-steps", type=int, default=3, help="Unrecorded warm-up requests.")
    parser.add_argument(
        "--camera-width",
        type=int,
        default=640,
        help="Synthetic raw camera width used for full serialization/preprocessing cost.",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=480,
        help="Synthetic raw camera height used for full serialization/preprocessing cost.",
    )
    parser.add_argument(
        "--task",
        default="Profile RTC delay.\nAdvantage: positive",
        help="Synthetic task prompt passed through the real policy tokenizer.",
    )
    parser.add_argument(
        "--server-startup-timeout-s",
        type=float,
        default=60.0,
        help="Seconds to wait for the private or external server socket.",
    )
    parser.add_argument(
        "--request-timeout-s",
        type=float,
        default=900.0,
        help="Timeout for model loading and each profiling RPC.",
    )
    parser.add_argument(
        "--server-log",
        type=Path,
        default=Path("d_server.log"),
        help="Private server log path (default: d_server.log).",
    )
    parser.add_argument("--revision", default=None, help="Optional Hugging Face policy revision.")
    parser.add_argument("--chunk-size", type=int, default=None, help="Override/read-free action chunk size.")
    parser.add_argument(
        "--rtc-training-max-delay",
        type=int,
        default=None,
        help="Optional existing D to check; omit it when profiling before training.",
    )
    parser.add_argument(
        "--control-hz",
        type=float,
        default=None,
        help="Fallback controller frequency for latency-only telemetry or offline latency samples.",
    )
    parser.add_argument(
        "--latency-ms",
        type=float,
        nargs="+",
        action="extend",
        default=[],
        help="Offline end-to-end latency values in milliseconds; may be repeated.",
    )
    parser.add_argument(
        "--latency-file",
        action="append",
        default=[],
        help="Offline JSON/text latency file, or '-' for stdin; may be repeated.",
    )
    parser.add_argument(
        "--delay-steps",
        type=int,
        nargs="+",
        action="extend",
        default=[],
        help="Offline d values in controller steps; may be repeated.",
    )
    parser.add_argument(
        "--recommend-percentile",
        type=float,
        default=99.0,
        help="Percentile used for the D recommendation (default: 99).",
    )
    parser.add_argument(
        "--safety-margin-steps",
        type=int,
        default=2,
        help="Extra controller steps added to the selected percentile (default: 2).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Report path. One-command headless mode defaults to d.txt.",
    )
    parser.add_argument("--json", action="store_true", help="Print the final report as JSON.")
    parser.add_argument(
        "--allow-overflow",
        action="store_true",
        help=f"Return exit code 0 on overflow (default overflow code: {OVERFLOW_EXIT_CODE}).",
    )
    return parser


def _resolve_limits(args: argparse.Namespace) -> RTCPolicyLimits:
    if args.policy_path:
        loaded = load_policy_limits(args.policy_path, args.revision)
    else:
        loaded = RTCPolicyLimits(chunk_size=0)

    chunk_size = args.chunk_size if args.chunk_size is not None else loaded.chunk_size
    training_max_delay = (
        args.rtc_training_max_delay
        if args.rtc_training_max_delay is not None
        else loaded.rtc_training_max_delay
    )
    if chunk_size == 0:
        raise ValueError("Provide --policy-path or --chunk-size so the RTC structural limit is known.")

    return RTCPolicyLimits(
        chunk_size=chunk_size,
        rtc_training_max_delay=training_max_delay,
        policy_type=loaded.policy_type,
        config_path=loaded.config_path,
    )


def _collect_inputs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], bool]:
    latencies = list(args.latency_ms)
    for latency_file in args.latency_file:
        latencies.extend(load_latency_file(latency_file))

    if latencies and args.control_hz is None:
        raise ValueError("--control-hz is required for offline latency samples.")

    events = [
        {
            "source": "offline_latency",
            "latency_ms": latency,
            "control_hz": args.control_hz,
            "d": latency_ms_to_delay_steps(latency, args.control_hz),
        }
        for latency in latencies
    ]
    events.extend({"source": "offline_d", "d": delay} for delay in args.delay_steps)

    live_mode = args.listen is not None or not events
    if live_mode:
        listen_address = args.listen or DEFAULT_LISTEN_ADDRESS
        events.extend(
            collect_udp_samples(
                listen_address,
                duration_s=args.duration_s,
                fallback_control_hz=args.control_hz,
                progress_interval_s=args.progress_interval_s,
            )
        )
    return events, live_mode


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], bool]:
    limits = _resolve_limits(args)
    validate_policy_limits(limits)
    profile_metadata = None
    if _is_headless_mode(args):
        events, profile_metadata = run_integrated_headless_profile(args, limits)
        mode = "headless"
        live_mode = True
    else:
        events, live_mode = _collect_inputs(args)
        mode = "live" if live_mode else "offline"
    if not events:
        raise RuntimeError("No RTC d samples were received during the profiling window.")

    delays = [event["d"] for event in events]
    distribution = summarize_delay_distribution(
        delays,
        limits,
        recommend_percentile=args.recommend_percentile,
        safety_margin_steps=args.safety_margin_steps,
    )
    latencies = [float(event["latency_ms"]) for event in events if "latency_ms" in event]
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "mode": mode,
        "policy": asdict(limits),
        "profile": profile_metadata,
        "latency_ms": {
            "sample_count": len(latencies),
            "min": min(latencies) if latencies else None,
            "mean": sum(latencies) / len(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "events": events,
        **distribution,
    }
    return report, live_mode


def format_text_report(report: dict[str, Any]) -> str:
    policy = report["policy"]
    quantiles = report["quantiles"]
    lines = [
        f"RTC d profile: {report['status'].upper()}",
        f"  created_at: {report['created_at']}",
        f"  mode / samples: {report['mode']} / {report['sample_count']}",
        f"  chunk_size / structural max d: {policy['chunk_size']} / {report['max_structural_d']}",
        f"  d min / mean / max: {report['min_d']} / {report['mean_d']:.2f} / {report['max_observed_d']}",
        (
            "  d p50 / p90 / p95 / p99 / p99.9: "
            f"{quantiles['p50']:.2f} / {quantiles['p90']:.2f} / {quantiles['p95']:.2f} / "
            f"{quantiles['p99']:.2f} / {quantiles['p99_9']:.2f}"
        ),
        (
            f"  suggested D: {report['recommended_rtc_training_max_delay']} "
            f"(p{report['recommend_percentile']:g} + {report['safety_margin_steps']} steps)"
        ),
        "  d histogram:",
    ]
    profile = report.get("profile")
    if profile is not None:
        lines[3:3] = [
            f"  GPU / server: {profile['gpu_id']} / {profile['server_address']}",
            f"  requested duration / fps: {profile['duration_s']:g}s / {profile['fps']} Hz",
            f"  robot actions sent: {profile['robot_actions_sent']}",
            (
                f"  synthetic input: {profile['camera_count']} cameras at "
                f"{profile['camera_width']}x{profile['camera_height']}, state_dim={profile['state_dim']}"
            ),
        ]
    for delay, item in report["histogram"].items():
        lines.append(f"    d={delay}: {item['count']} ({item['fraction']:.2%})")

    if policy["rtc_training_max_delay"] is not None:
        lines.append(
            f"  existing D / overflow: {policy['rtc_training_max_delay']} / "
            f"{report['existing_training_overflow_count']} ({report['existing_training_overflow_fraction']:.2%})"
        )
    if report["latency_ms"]["sample_count"]:
        latency = report["latency_ms"]
        lines.append(
            f"  latency ms min / mean / max: {latency['min']:.2f} / {latency['mean']:.2f} / {latency['max']:.2f}"
        )
    if policy["config_path"]:
        lines.append(f"  config: {policy['config_path']}")
    lines.append(f"  result: {report['message']}")
    return "\n".join(lines)


def _default_output_path() -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return Path("rtc_d_profiles") / f"rtc_d_profile_{timestamp}.json"


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        report, live_mode = build_report(args)
    except RuntimeError as exc:
        print(f"RTC d profiling failed: {exc}", file=sys.stderr)
        return NO_SAMPLES_EXIT_CODE
    except (FileNotFoundError, OSError, ValueError) as exc:
        parser.error(str(exc))

    if report["mode"] == "headless":
        output_path = args.output or Path("d.txt")
    else:
        output_path = args.output or (_default_output_path() if live_mode else None)
    if output_path is not None:
        output_path = output_path.expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if report["mode"] == "headless" or output_path.suffix.lower() == ".txt":
            output_contents = format_text_report(report) + "\n"
        else:
            output_contents = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        output_path.write_text(output_contents, encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(format_text_report(report))
        if output_path is not None:
            print(f"  saved report: {output_path.resolve()}")

    if report["safe"] or args.allow_overflow:
        return 0
    return OVERFLOW_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
