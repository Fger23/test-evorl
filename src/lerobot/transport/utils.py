#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
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

import io
import json
import logging
import pickle  # nosec B403: Safe usage for internal serialization only
import struct
import time
import zlib
from dataclasses import dataclass
from multiprocessing.synchronize import Event as MpEvent
from queue import Queue
from typing import Any

import torch

from lerobot.transport import services_pb2
from lerobot.utils.transition import Transition

# FIX for protobuf: Assign the enum to a variable and ignore the type error once
TransferState = services_pb2.TransferState  # type: ignore[attr-defined]

CHUNK_SIZE = 2 * 1024 * 1024  # 2 MB
MAX_MESSAGE_SIZE = 4 * 1024 * 1024  # 4 MB

# Observation payloads historically contained the raw pickle bytes.  Keep that
# representation for the ``none`` codec and use an unmistakable binary envelope
# only when compression is actually applied.  This lets a new server accept old
# clients without changing the protobuf schema.
OBSERVATION_PAYLOAD_MAGIC = b"LEROBOT\x00"
OBSERVATION_PAYLOAD_VERSION = 1
OBSERVATION_CODEC_NONE = "none"
OBSERVATION_CODEC_ZLIB = "zlib"
SUPPORTED_OBSERVATION_CODECS = (OBSERVATION_CODEC_NONE, OBSERVATION_CODEC_ZLIB)
OBSERVATION_CODECS_METADATA_KEY = "lerobot-observation-codecs"
OBSERVATION_WIRE_VERSION_METADATA_KEY = "lerobot-observation-wire-version"
DEFAULT_MAX_OBSERVATION_PAYLOAD_BYTES = 64 * 1024 * 1024
_OBSERVATION_CODEC_TO_ID = {OBSERVATION_CODEC_ZLIB: 1}
_OBSERVATION_ID_TO_CODEC = {value: key for key, value in _OBSERVATION_CODEC_TO_ID.items()}
_OBSERVATION_PAYLOAD_HEADER = struct.Struct("!8sBBQ")


@dataclass(frozen=True)
class EncodedObservationPayload:
    """An observation payload plus lossless compression diagnostics."""

    data: bytes
    codec: str
    raw_bytes: int
    wire_bytes: int
    compression_ratio: float
    compression_ms: float
    skipped_reason: str | None = None


@dataclass(frozen=True)
class DecodedObservationPayload:
    """Decoded pickle bytes plus wire-format diagnostics."""

    data: bytes
    codec: str
    wire_version: int
    wire_bytes: int
    raw_bytes: int
    compression_ratio: float
    decompression_ms: float


def encode_observation_payload(
    payload: bytes,
    *,
    codec: str = OBSERVATION_CODEC_NONE,
    zlib_level: int = 1,
    min_bytes: int = 0,
    min_savings_ratio: float = 0.0,
    require_savings: bool = True,
) -> EncodedObservationPayload:
    """Encode serialized observation bytes without changing their contents.

    ``none`` deliberately returns the original bytes without an envelope.  A
    zlib envelope is emitted only when compression is used, so new servers can
    continue to consume legacy clients and ``auto`` mode can fall back per
    request when compression is not worthwhile.
    """

    if codec not in SUPPORTED_OBSERVATION_CODECS:
        raise ValueError(
            f"Unsupported observation codec {codec!r}; expected one of {SUPPORTED_OBSERVATION_CODECS}."
        )
    if not 0 <= zlib_level <= 9:
        raise ValueError("zlib_level must be between 0 and 9.")
    if min_bytes < 0:
        raise ValueError("min_bytes must be non-negative.")
    if not 0 <= min_savings_ratio < 1:
        raise ValueError("min_savings_ratio must be in [0, 1).")

    raw_bytes = len(payload)
    if codec == OBSERVATION_CODEC_NONE:
        return EncodedObservationPayload(
            data=payload,
            codec=OBSERVATION_CODEC_NONE,
            raw_bytes=raw_bytes,
            wire_bytes=raw_bytes,
            compression_ratio=1.0,
            compression_ms=0.0,
            skipped_reason="codec_none",
        )
    if raw_bytes < min_bytes:
        return EncodedObservationPayload(
            data=payload,
            codec=OBSERVATION_CODEC_NONE,
            raw_bytes=raw_bytes,
            wire_bytes=raw_bytes,
            compression_ratio=1.0,
            compression_ms=0.0,
            skipped_reason="below_min_bytes",
        )

    compression_started = time.perf_counter()
    try:
        compressed = zlib.compress(payload, level=zlib_level)
    except zlib.error as exc:
        compression_ms = (time.perf_counter() - compression_started) * 1000
        if require_savings:
            # ``auto`` transport must remain an optimization only: a codec
            # failure must not prevent an otherwise valid raw observation from
            # reaching the policy server.
            return EncodedObservationPayload(
                data=payload,
                codec=OBSERVATION_CODEC_NONE,
                raw_bytes=raw_bytes,
                wire_bytes=raw_bytes,
                compression_ratio=1.0,
                compression_ms=compression_ms,
                skipped_reason="compression_error_fallback",
            )
        raise ValueError(f"Unable to compress observation payload: {exc}") from exc
    header = _OBSERVATION_PAYLOAD_HEADER.pack(
        OBSERVATION_PAYLOAD_MAGIC,
        OBSERVATION_PAYLOAD_VERSION,
        _OBSERVATION_CODEC_TO_ID[OBSERVATION_CODEC_ZLIB],
        raw_bytes,
    )
    encoded = header + compressed
    compression_ms = (time.perf_counter() - compression_started) * 1000
    compression_ratio = len(encoded) / raw_bytes if raw_bytes else 1.0
    savings_ratio = 1.0 - compression_ratio
    if require_savings and savings_ratio < min_savings_ratio:
        return EncodedObservationPayload(
            data=payload,
            codec=OBSERVATION_CODEC_NONE,
            raw_bytes=raw_bytes,
            wire_bytes=raw_bytes,
            compression_ratio=1.0,
            compression_ms=compression_ms,
            skipped_reason="insufficient_savings",
        )

    return EncodedObservationPayload(
        data=encoded,
        codec=OBSERVATION_CODEC_ZLIB,
        raw_bytes=raw_bytes,
        wire_bytes=len(encoded),
        compression_ratio=compression_ratio,
        compression_ms=compression_ms,
    )


def decode_observation_payload(
    payload: bytes,
    *,
    max_uncompressed_bytes: int = DEFAULT_MAX_OBSERVATION_PAYLOAD_BYTES,
    allow_zlib: bool = True,
) -> DecodedObservationPayload:
    """Decode an observation payload while accepting the legacy raw format."""

    if max_uncompressed_bytes <= 0:
        raise ValueError("max_uncompressed_bytes must be positive.")

    wire_bytes = len(payload)
    if not payload.startswith(OBSERVATION_PAYLOAD_MAGIC):
        if wire_bytes > max_uncompressed_bytes:
            raise ValueError(
                f"Observation payload is {wire_bytes} bytes, exceeding the configured "
                f"limit of {max_uncompressed_bytes} bytes."
            )
        return DecodedObservationPayload(
            data=payload,
            codec=OBSERVATION_CODEC_NONE,
            wire_version=0,
            wire_bytes=wire_bytes,
            raw_bytes=wire_bytes,
            compression_ratio=1.0,
            decompression_ms=0.0,
        )

    if len(payload) < _OBSERVATION_PAYLOAD_HEADER.size:
        raise ValueError("Truncated observation compression envelope.")
    magic, version, codec_id, raw_bytes = _OBSERVATION_PAYLOAD_HEADER.unpack_from(payload)
    if magic != OBSERVATION_PAYLOAD_MAGIC:
        raise ValueError("Invalid observation compression envelope magic.")
    if version != OBSERVATION_PAYLOAD_VERSION:
        raise ValueError(f"Unsupported observation payload version {version}.")
    codec = _OBSERVATION_ID_TO_CODEC.get(codec_id)
    if codec is None:
        raise ValueError(f"Unsupported observation payload codec id {codec_id}.")
    if codec == OBSERVATION_CODEC_ZLIB and not allow_zlib:
        raise ValueError("This policy server does not accept zlib-compressed observations.")
    if raw_bytes > max_uncompressed_bytes:
        raise ValueError(
            f"Observation declares {raw_bytes} uncompressed bytes, exceeding the configured "
            f"limit of {max_uncompressed_bytes} bytes."
        )

    compressed = payload[_OBSERVATION_PAYLOAD_HEADER.size :]
    decompression_started = time.perf_counter()
    decode_limit = min(max_uncompressed_bytes, raw_bytes)
    try:
        decompressor = zlib.decompressobj()
        decoded = decompressor.decompress(compressed, decode_limit + 1)
        if len(decoded) > decode_limit or decompressor.unconsumed_tail:
            raise ValueError(
                "Decompressed observation exceeds its declared size or the configured "
                f"limit of {max_uncompressed_bytes} bytes."
            )
        decoded += decompressor.flush(decode_limit - len(decoded) + 1)
    except zlib.error as exc:
        raise ValueError(f"Invalid zlib observation payload: {exc}") from exc
    decompression_ms = (time.perf_counter() - decompression_started) * 1000

    if len(decoded) > decode_limit:
        raise ValueError(
            "Decompressed observation exceeds its declared size or the configured "
            f"limit of {max_uncompressed_bytes} bytes."
        )
    if not decompressor.eof or decompressor.unused_data:
        raise ValueError("Truncated zlib observation payload or unexpected trailing data.")
    if len(decoded) != raw_bytes:
        raise ValueError(
            f"Observation size mismatch: envelope declares {raw_bytes} bytes, decoded {len(decoded)} bytes."
        )

    return DecodedObservationPayload(
        data=decoded,
        codec=codec,
        wire_version=version,
        wire_bytes=wire_bytes,
        raw_bytes=len(decoded),
        compression_ratio=wire_bytes / len(decoded) if decoded else 1.0,
        decompression_ms=decompression_ms,
    )


def bytes_buffer_size(buffer: io.BytesIO) -> int:
    buffer.seek(0, io.SEEK_END)
    result = buffer.tell()
    buffer.seek(0)
    return result


def send_bytes_in_chunks(buffer: bytes, message_class: Any, log_prefix: str = "", silent: bool = True):
    bytes_buffer: io.BytesIO = io.BytesIO(buffer)
    size_in_bytes = bytes_buffer_size(bytes_buffer)

    sent_bytes = 0

    logging_method = logging.info if not silent else logging.debug

    logging_method(f"{log_prefix} Buffer size {size_in_bytes / 1024 / 1024} MB with")

    while sent_bytes < size_in_bytes:
        transfer_state = TransferState.TRANSFER_MIDDLE

        if sent_bytes + CHUNK_SIZE >= size_in_bytes:
            transfer_state = TransferState.TRANSFER_END
        elif sent_bytes == 0:
            transfer_state = TransferState.TRANSFER_BEGIN

        size_to_read = min(CHUNK_SIZE, size_in_bytes - sent_bytes)
        chunk = bytes_buffer.read(size_to_read)

        yield message_class(transfer_state=transfer_state, data=chunk)
        sent_bytes += size_to_read
        logging_method(f"{log_prefix} Sent {sent_bytes}/{size_in_bytes} bytes with state {transfer_state}")

    logging_method(f"{log_prefix} Published {sent_bytes / 1024 / 1024} MB")


def receive_bytes_in_chunks(
    iterator,
    queue: Queue | None,
    shutdown_event: MpEvent,
    log_prefix: str = "",
    max_size_bytes: int | None = None,
):
    bytes_buffer = io.BytesIO()
    step = 0

    if max_size_bytes is not None and max_size_bytes <= 0:
        raise ValueError("max_size_bytes must be positive when provided.")

    def write_chunk(data: bytes) -> None:
        if max_size_bytes is not None and bytes_buffer.tell() + len(data) > max_size_bytes:
            raise ValueError(f"Received payload exceeds the configured limit of {max_size_bytes} bytes.")
        bytes_buffer.write(data)

    logging.info(f"{log_prefix} Starting receiver")
    for item in iterator:
        logging.debug(f"{log_prefix} Received item")
        if shutdown_event.is_set():
            logging.info(f"{log_prefix} Shutting down receiver")
            return

        if item.transfer_state == TransferState.TRANSFER_BEGIN:
            bytes_buffer.seek(0)
            bytes_buffer.truncate(0)
            write_chunk(item.data)
            logging.debug(f"{log_prefix} Received data at step 0")
            step = 0
        elif item.transfer_state == TransferState.TRANSFER_MIDDLE:
            write_chunk(item.data)
            step += 1
            logging.debug(f"{log_prefix} Received data at step {step}")
        elif item.transfer_state == TransferState.TRANSFER_END:
            write_chunk(item.data)
            logging.debug(f"{log_prefix} Received data at step end size {bytes_buffer_size(bytes_buffer)}")

            if queue is not None:
                queue.put(bytes_buffer.getvalue())
            else:
                return bytes_buffer.getvalue()

            bytes_buffer.seek(0)
            bytes_buffer.truncate(0)
            step = 0

            logging.debug(f"{log_prefix} Queue updated")
        else:
            logging.warning(f"{log_prefix} Received unknown transfer state {item.transfer_state}")
            raise ValueError(f"Received unknown transfer state {item.transfer_state}")


def state_to_bytes(state_dict: dict[str, torch.Tensor]) -> bytes:
    """Convert model state dict to flat array for transmission"""
    bytes_buffer = io.BytesIO()

    torch.save(state_dict, bytes_buffer)

    return bytes_buffer.getvalue()


def bytes_to_state_dict(buffer: bytes) -> dict[str, torch.Tensor]:
    bytes_buffer = io.BytesIO(buffer)
    bytes_buffer.seek(0)
    return torch.load(bytes_buffer, weights_only=True)


def python_object_to_bytes(python_object: Any) -> bytes:
    return pickle.dumps(python_object)


def bytes_to_python_object(buffer: bytes) -> Any:
    bytes_buffer = io.BytesIO(buffer)
    bytes_buffer.seek(0)
    obj = pickle.load(bytes_buffer)  # nosec B301: Safe usage of pickle.load
    # Add validation checks here
    return obj


def bytes_to_transitions(buffer: bytes) -> list[Transition]:
    bytes_buffer = io.BytesIO(buffer)
    bytes_buffer.seek(0)
    transitions = torch.load(bytes_buffer, weights_only=True)
    return transitions


def transitions_to_bytes(transitions: list[Transition]) -> bytes:
    bytes_buffer = io.BytesIO()
    torch.save(transitions, bytes_buffer)
    return bytes_buffer.getvalue()


def grpc_channel_options(
    max_receive_message_length: int = MAX_MESSAGE_SIZE,
    max_send_message_length: int = MAX_MESSAGE_SIZE,
    enable_retries: bool = True,
    initial_backoff: str = "0.1s",
    max_attempts: int = 5,
    backoff_multiplier: float = 2,
    max_backoff: str = "2s",
):
    service_config = {
        "methodConfig": [
            {
                "name": [{}],  # Applies to ALL methods in ALL services
                "retryPolicy": {
                    "maxAttempts": max_attempts,  # Max retries (total attempts = 5)
                    "initialBackoff": initial_backoff,  # First retry after 0.1s
                    "maxBackoff": max_backoff,  # Max wait time between retries
                    "backoffMultiplier": backoff_multiplier,  # Exponential backoff factor
                    "retryableStatusCodes": [
                        "UNAVAILABLE",
                        "DEADLINE_EXCEEDED",
                    ],  # Retries on network failures
                },
            }
        ]
    }

    service_config_json = json.dumps(service_config)

    retries_option = 1 if enable_retries else 0

    return [
        ("grpc.max_receive_message_length", max_receive_message_length),
        ("grpc.max_send_message_length", max_send_message_length),
        ("grpc.enable_retries", retries_option),
        ("grpc.service_config", service_config_json),
    ]
