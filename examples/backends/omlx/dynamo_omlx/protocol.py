# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire protocol between the oMLX bridge (macOS) and the Dynamo proxy (Linux).

Stdlib only: the bridge runs inside oMLX's Python environment and must not
import ``dynamo``, while the proxy runs inside ai-dynamo's environment.

Every stream is NDJSON: one compact JSON object per line, discriminated by the
``"t"`` field. Consumers ignore unknown frame types and unknown fields, so
either side can add fields without a protocol version bump.

Generate request (``POST {API_PREFIX}/generate``), prompt as Dynamo tokenized it::

    {"v":1,"request_id":"...","token_ids":[...],"max_tokens":N,"stop_token_ids":[...],
     "skip_cache_store":false,"temperature":...,
     "images":[{"data":"<base64 image file>","mm_hash":"<16 hex digits>"|null}, ...],
     "expanded_prompt_len":N|null,"min_pixels":N|null,"max_pixels":N|null}

``images`` lists one entry per image placeholder token in ``token_ids``, in
order; ``mm_hash`` is the image's identity in Dynamo's KV router,
``expanded_prompt_len`` the prompt length the frontend routed with after
expanding the images, and ``min_pixels``/``max_pixels`` the pixel range it
counted their tokens with. A bridge that predates images would ignore them and read
the placeholders as text, so the proxy sends images only to a bridge whose
``/info`` reports ``model.vision``.

Generate stream::

    {"t":"start","v":1,"rid":"...","inc":"..."}
    {"t":"hb"}                                   # while waiting for tokens
    {"t":"tok","ids":[...]}
    {"t":"end","finish":"stop|length|abort|error","usage":{...},"error":{...}|null}

KV snapshot (``GET {API_PREFIX}/kv/snapshot``)::

    {"t":"snap_begin","v":1,"inc":"...","as_of":N,"block_size":B,"count":M,"state":"..."}
    {"t":"snap","blocks":[{"h":ext,"p":parent|null,"tok":"<b64 u32le>"}, ...]}
    {"t":"snap_end","as_of":N,"count":M}

KV event stream (``GET {API_PREFIX}/kv/stream?inc=...&after=N``)::

    {"t":"hello","v":1,"inc":"...","seq":tail,"floor":F,"block_size":B,"state":"...","hb_s":2.0}
    {"t":"ev","seq":N,"op":"store","blocks":[{"h":ext,"p":parent|null,"tok":"..."}, ...]}
    {"t":"ev","seq":N,"op":"remove","hashes":[ext, ...]}
    {"t":"ev","seq":N,"op":"clear"}
    {"t":"hb","seq":tail}
    {"t":"reset","reason":"incarnation|floor|ahead|disabled|shutdown"}   # then EOF

Snapshot blocks are parent-first, store events are parent-first chain
segments, and remove events list whole subtrees leaf-first. A block's tokens
are its routing tokens: the image tokens of an image request carry the
image's pad value (:func:`pad_value_for_mm_hash`), as in Dynamo's router.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import struct
import sys
from array import array
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

PROTOCOL_VERSION = 1
# Version of block_hash(). Every bridge and proxy in one Dynamo model must
# agree on it; the proxy refuses to start when its bridge reports another one.
HASH_VERSION = 1
API_PREFIX = "/dynamo/v1"
NDJSON_MEDIA_TYPE = "application/x-ndjson"
# Largest NDJSON line either side accepts.
MAX_LINE_BYTES = 16 * 1024 * 1024
# Blocks per store event or snapshot chunk. 16 blocks of 8192 tokens encode to
# about 700 KB, far below MAX_LINE_BYTES.
MAX_BLOCKS_PER_FRAME = 16

_HASH_DOMAIN = b"dynamo-omlx-kv-hash-v1"
_HASH_PERSON = b"dynomlx1"
# Dynamo's KV router sees every token of an image as one pad value derived from
# the image's mm_hash, above any vocabulary, so blocks holding different images
# hash differently. Mirrors dynamo_kv_router::protocols::pad_value_for_mm_hash.
MM_PAD_SHIFT_VALUE = 1_000_000
MM_PAD_HASH_MASK = (1 << 30) - 1
_MM_HASH_HEX_DIGITS = 16
_I64 = struct.Struct("<q")
_U64 = struct.Struct("<Q")
_I64_MIN = -(2**63)
_I64_MAX = 2**63 - 1

if array("I").itemsize != 4:  # every supported platform has a 32-bit C unsigned int
    raise ImportError("dynamo_omlx requires a 32-bit C unsigned int")


class ProtocolError(ValueError):
    """A peer sent data that violates this protocol."""


class Frame:
    """Values of the ``"t"`` discriminator."""

    START = "start"
    TOKENS = "tok"
    END = "end"
    HELLO = "hello"
    EVENT = "ev"
    RESET = "reset"
    SNAPSHOT_BEGIN = "snap_begin"
    SNAPSHOT_BLOCKS = "snap"
    SNAPSHOT_END = "snap_end"
    HEARTBEAT = "hb"


class KvOp:
    STORE = "store"
    REMOVE = "remove"
    CLEAR = "clear"


class KvState:
    """Bridge-side KV tracking state, reported by ``/info`` and ``hello``."""

    LIVE = "live"
    DEGRADED = "degraded"  # hash verification or hooks failed; nothing is announced
    DISABLED = "disabled"  # KV events turned off or the cache stack is unavailable


class ResetReason:
    INCARNATION = "incarnation"  # client state belongs to another bridge process
    FLOOR = "floor"  # requested events already left the replay ring
    AHEAD = "ahead"  # client claims events the bridge never produced
    DISABLED = "disabled"
    SHUTDOWN = "shutdown"


class FinishReason:
    STOP = "stop"
    LENGTH = "length"
    ABORT = "abort"  # cancelled on the client's request
    ERROR = "error"


class ErrorCode:
    INVALID_REQUEST = "invalid_request"
    MODEL_NOT_FOUND = "model_not_found"
    MODEL_NOT_READY = "model_not_ready"
    OVERLOADED = "overloaded"
    PREFILL_MEMORY_EXCEEDED = "prefill_memory_exceeded"
    ABORTED_BY_ENGINE = "aborted_by_engine"
    SHUTTING_DOWN = "shutting_down"
    ENGINE_ERROR = "engine_error"
    INTERNAL = "internal"


# Failures another worker may serve; the proxy maps them to Dynamo errors that
# the frontend is allowed to migrate.
RETRYABLE_ERRORS = frozenset(
    {
        ErrorCode.MODEL_NOT_FOUND,
        ErrorCode.MODEL_NOT_READY,
        ErrorCode.OVERLOADED,
        ErrorCode.ABORTED_BY_ENGINE,
        ErrorCode.SHUTTING_DOWN,
    }
)

# HTTP status for errors the bridge reports before a stream starts.
ERROR_HTTP_STATUS = {
    ErrorCode.INVALID_REQUEST: 400,
    ErrorCode.PREFILL_MEMORY_EXCEEDED: 400,
    ErrorCode.MODEL_NOT_FOUND: 404,
    ErrorCode.MODEL_NOT_READY: 503,
    ErrorCode.OVERLOADED: 503,
    ErrorCode.ABORTED_BY_ENGINE: 503,
    ErrorCode.SHUTTING_DOWN: 503,
    ErrorCode.ENGINE_ERROR: 500,
    ErrorCode.INTERNAL: 500,
}


def error_payload(code: str, message: str) -> dict[str, Any]:
    return {"code": code, "message": message, "retryable": code in RETRYABLE_ERRORS}


def encode_frame(frame: dict[str, Any]) -> bytes:
    return json.dumps(frame, separators=(",", ":")).encode() + b"\n"


def decode_frame(line: bytes | str) -> dict[str, Any]:
    try:
        frame = json.loads(line)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ProtocolError(f"invalid NDJSON line: {e}") from e
    if not isinstance(frame, dict) or not isinstance(frame.get("t"), str):
        raise ProtocolError("frame must be a JSON object with a string 't' field")
    return frame


def tokens_to_array(tokens: Iterable[int]) -> array:
    """Copy token ids into a ``uint32`` array, rejecting out-of-range values."""
    try:
        return array("I", tokens)
    except (OverflowError, TypeError) as e:
        raise ProtocolError(f"token ids must be integers in [0, 2**32): {e}") from e


def _le_bytes(tokens: array | Sequence[int]) -> bytes:
    arr = (
        tokens
        if isinstance(tokens, array) and tokens.typecode == "I"
        else tokens_to_array(tokens)
    )
    if sys.byteorder == "big":
        arr = array("I", arr)
        arr.byteswap()
    return arr.tobytes()


def encode_tokens(tokens: array | Sequence[int]) -> str:
    """Base64 of the token ids as little-endian ``uint32``."""
    return base64.b64encode(_le_bytes(tokens)).decode("ascii")


def decode_tokens(data: str) -> array:
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, TypeError, ValueError) as e:
        raise ProtocolError(f"invalid base64 token payload: {e}") from e
    if len(raw) % 4:
        raise ProtocolError("token payload length is not a multiple of 4")
    arr = array("I")
    arr.frombytes(raw)
    if sys.byteorder == "big":
        arr.byteswap()
    return arr


def block_hash(
    block_size: int, parent: int | None, tokens: array | Sequence[int]
) -> int:
    """External block hash announced to the Dynamo router, as a signed i64.

    The router requires external hashes to be a deterministic function of the
    token chain and identical on every worker. oMLX's own block hash is salted
    with the local model directory name, so the bridge derives this one from
    ``(block_size, parent hash, tokens)`` instead.
    """
    h = hashlib.blake2b(digest_size=8, person=_HASH_PERSON)
    h.update(_HASH_DOMAIN)
    h.update(_U64.pack(block_size))
    h.update(b"\x00" if parent is None else b"\x01" + _I64.pack(parent))
    h.update(_le_bytes(tokens))
    return int.from_bytes(h.digest(), "big", signed=True)


def pad_value_for_mm_hash(mm_hash: int) -> int:
    """The token Dynamo's router sees at every position of the image ``mm_hash``."""
    return MM_PAD_SHIFT_VALUE + (mm_hash & MM_PAD_HASH_MASK)


def format_mm_hash(mm_hash: int) -> str:
    """``mm_hash`` as Dynamo's frontend writes it: 16 lowercase hex digits."""
    return f"{mm_hash:016x}"


def parse_mm_hash(value: Any) -> int | None:
    """An image's mm_hash from :func:`format_mm_hash` form; None stays None."""
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != _MM_HASH_HEX_DIGITS
        or value.strip("0123456789abcdefABCDEF")
    ):
        raise ProtocolError(
            f"mm_hash must be {_MM_HASH_HEX_DIGITS} hex digits, got {value!r}"
        )
    return int(value, 16)


@dataclass(frozen=True, slots=True)
class KvBlock:
    """One announced block: external hash, parent external hash, and tokens."""

    ext: int
    parent: int | None
    tokens: array

    def to_wire(self) -> dict[str, Any]:
        return {"h": self.ext, "p": self.parent, "tok": encode_tokens(self.tokens)}


def _i64(value: Any, what: str) -> int:
    # bool is an int subclass; reject it explicitly.
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not _I64_MIN <= value <= _I64_MAX
    ):
        raise ProtocolError(f"{what} must be a signed 64-bit integer, got {value!r}")
    return value


def parse_blocks(items: Any, block_size: int) -> list[KvBlock]:
    """Validate the ``blocks`` list of a store event or snapshot chunk."""
    if not isinstance(items, list):
        raise ProtocolError("'blocks' must be a list")
    blocks: list[KvBlock] = []
    for item in items:
        if not isinstance(item, dict):
            raise ProtocolError("each block must be an object")
        parent = item.get("p")
        tokens = decode_tokens(item.get("tok", ""))
        if len(tokens) != block_size:
            raise ProtocolError(
                f"block carries {len(tokens)} tokens, expected {block_size}"
            )
        blocks.append(
            KvBlock(
                ext=_i64(item.get("h"), "block hash"),
                parent=None if parent is None else _i64(parent, "parent hash"),
                tokens=tokens,
            )
        )
    return blocks


def parse_hashes(items: Any) -> list[int]:
    if not isinstance(items, list):
        raise ProtocolError("'hashes' must be a list")
    return [_i64(h, "block hash") for h in items]
