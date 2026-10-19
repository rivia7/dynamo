# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bridge command line: ``[bridge flags] -- serve [omlx serve flags]``."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

_SEPARATOR = "--"


@dataclass(frozen=True)
class BridgeConfig:
    model_id: str
    # Preset for oMLX's paged cache block size. oMLX may still enlarge it (for
    # example to 8192 for Qwen3.8-Flash-Next); /info reports the value in effect.
    kv_block_size: int | None = None
    kv_events: bool = True
    pin_model: bool = True
    preload: bool = True
    verify_hashes: bool = True
    ring_max_events: int = 100_000
    ring_max_bytes: int = 64 * 1024 * 1024
    audit_interval_s: float = 30.0
    # oMLX's multi-token prediction for the model, written to oMLX's per-model
    # settings before the model loads; None leaves those settings alone.
    mtp: bool | None = None
    mtp_depth: int | None = None
    # Not flags: the proxy's stall and read timeouts are multiples of these.
    kv_heartbeat_s: float = 2.0
    generate_heartbeat_s: float = 5.0
    check: bool = False


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return parsed


def _mtp_depth(value: str) -> int:
    parsed = int(value)
    # oMLX clamps deeper requests to MAX_LIGHTNING_MTP_DRAFT_TOKENS (8).
    if not 1 <= parsed <= 8:
        raise argparse.ArgumentTypeError(f"expected 1 to 8, got {value}")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive number, got {value}")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dynamo_omlx.bridge",
        description=(
            "Run 'omlx serve' with the Dynamo bridge mounted under /dynamo/v1. "
            "Bridge flags come first, then '--', then the unchanged omlx command, "
            "for example: --model-id MyModel -- serve --model-dir ~/models"
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--model-id",
        default=os.environ.get("OMLX_BRIDGE_MODEL_ID"),
        help="oMLX model id (model directory name) served to Dynamo. Env: OMLX_BRIDGE_MODEL_ID.",
    )
    parser.add_argument(
        "--kv-block-size",
        type=_positive_int,
        default=None,
        help="Preset for oMLX's paged cache block size. Every Mac behind one Dynamo model must end up with the same value.",
    )
    parser.add_argument(
        "--no-kv-events",
        dest="kv_events",
        action="store_false",
        help="Serve tokens only; publish no KV cache state.",
    )
    parser.add_argument(
        "--no-pin",
        dest="pin_model",
        action="store_false",
        help="Do not pin the model in oMLX's engine pool.",
    )
    parser.add_argument(
        "--no-preload",
        dest="preload",
        action="store_false",
        help="Do not load the model when the server starts.",
    )
    parser.add_argument(
        "--no-verify-hashes",
        dest="verify_hashes",
        action="store_false",
        help="Skip recomputing oMLX block hashes before announcing blocks.",
    )
    parser.add_argument(
        "--ring-max-events", type=_positive_int, default=BridgeConfig.ring_max_events
    )
    parser.add_argument(
        "--ring-max-bytes", type=_positive_int, default=BridgeConfig.ring_max_bytes
    )
    parser.add_argument(
        "--audit-interval",
        dest="audit_interval_s",
        type=_positive_float,
        default=BridgeConfig.audit_interval_s,
        help="Seconds between checks that announced blocks still exist in the SSD cache.",
    )
    parser.add_argument(
        "--mtp",
        choices=("on", "off"),
        default=None,
        help=(
            "Turn oMLX's multi-token prediction (speculative decoding with the model's "
            "own MTP head) on or off in oMLX's per-model settings. Default: leave them alone."
        ),
    )
    parser.add_argument(
        "--mtp-depth",
        type=_mtp_depth,
        default=None,
        help="Most draft tokens per MTP verify cycle, 1 to 8. Default: oMLX's, 3 for Qwen3.8-Flash-Next.",
    )
    parser.add_argument(
        "--check", action="store_true", help="Check oMLX compatibility and exit."
    )
    return parser


def parse_bridge_args(argv: list[str]) -> tuple[BridgeConfig, list[str]]:
    """Split ``argv`` at the first ``--`` into bridge flags and omlx arguments."""
    if _SEPARATOR in argv:
        split = argv.index(_SEPARATOR)
        bridge_argv, omlx_argv = argv[:split], argv[split + 1 :]
    else:
        bridge_argv, omlx_argv = argv, []
    parser = build_parser()
    args = parser.parse_args(bridge_argv)
    if not args.check:
        if not args.model_id:
            parser.error("--model-id (or OMLX_BRIDGE_MODEL_ID) is required")
        if not omlx_argv or omlx_argv[0] != "serve":
            parser.error(
                "pass the omlx command after '--', for example: -- serve --model-dir ~/models"
            )
    values = vars(args)
    values["mtp"] = None if args.mtp is None else args.mtp == "on"
    return BridgeConfig(**values), omlx_argv
