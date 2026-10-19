# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Proxy command line: Dynamo runtime flags plus the oMLX proxy flags."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dynamo.common.configuration.groups.runtime_args import (
    DynamoRuntimeArgGroup,
    DynamoRuntimeConfig,
)
from dynamo.common.utils.runtime import parse_endpoint

DEFAULT_COMPONENT = "backend"
DEFAULT_ENDPOINT = "generate"


class RuntimeConfig(DynamoRuntimeConfig):
    component: str


@dataclass(frozen=True)
class ProxyConfig:
    bridge_url: str
    bridge_api_key: str | None
    model_path: str
    served_model_name: str
    context_length: int | None
    bos_token_id: int
    expect_block_size: int | None
    total_kv_blocks: int | None
    max_output_tokens: int
    kv_events: bool
    startup_timeout_s: float
    mac_down_grace_s: float
    stats_interval_s: float
    generate_read_timeout_s: float


def _text_config(model_config: dict[str, Any]) -> dict[str, Any]:
    text_config = model_config.get("text_config")
    return text_config if isinstance(text_config, dict) else {}


def _config_value(model_config: dict[str, Any], key: str) -> Any:
    value = model_config.get(key)
    return value if value is not None else _text_config(model_config).get(key)


def _read_model_config(model_path: Path) -> dict[str, Any]:
    config_file = model_path / "config.json"
    if not config_file.is_file():
        raise SystemExit(f"--model-path {model_path} has no config.json")
    with config_file.open() as f:
        return json.load(f)


def parse_args(argv: list[str] | None = None) -> tuple[ProxyConfig, RuntimeConfig]:
    parser = argparse.ArgumentParser(
        description="Dynamo worker that serves a remote oMLX bridge (python -m dynamo_omlx.bridge on the Mac).",
        formatter_class=argparse.RawTextHelpFormatter,
        allow_abbrev=False,
    )
    DynamoRuntimeArgGroup().add_arguments(parser)
    group = parser.add_argument_group("oMLX proxy")
    group.add_argument(
        "--bridge-url",
        default=os.environ.get("OMLX_BRIDGE_URL"),
        help="Base URL of the Mac's bridge, e.g. http://mac-studio:8000. Env: OMLX_BRIDGE_URL.",
    )
    group.add_argument(
        "--bridge-api-key",
        default=os.environ.get("OMLX_API_KEY"),
        help="oMLX API key; required when oMLX listens on a network address. Env: OMLX_API_KEY.",
    )
    group.add_argument(
        "--model-path",
        required=True,
        help=(
            "Local directory with the model's config.json, tokenizer and chat template (no weights needed).\n"
            "Every proxy of one model must use byte-identical files at the same path."
        ),
    )
    group.add_argument(
        "--served-model-name",
        default=None,
        help="Model name clients request (default: directory name).",
    )
    group.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="Override max_position_embeddings from config.json.",
    )
    group.add_argument(
        "--expect-block-size",
        type=int,
        default=None,
        help="Refuse to start unless the Mac's oMLX paged cache block size equals this value.",
    )
    group.add_argument(
        "--total-kv-blocks",
        type=int,
        default=None,
        help="KV capacity to advertise (default: max_num_seqs * ceil(context_length / block_size)).",
    )
    group.add_argument(
        "--max-output-tokens",
        type=int,
        default=32768,
        help="Cap when a request sets no max_tokens.",
    )
    group.add_argument(
        "--no-kv-events",
        dest="kv_events",
        action="store_false",
        help="Do not publish KV cache events.",
    )
    group.add_argument(
        "--startup-timeout",
        type=float,
        default=0.0,
        help="Seconds to wait for the Mac's model to be ready before failing (0: wait forever).",
    )
    group.add_argument(
        "--mac-down-grace",
        type=float,
        default=30.0,
        help="Exit (and deregister) after the bridge is unreachable for this many seconds.",
    )
    group.add_argument(
        "--stats-interval",
        type=float,
        default=1.0,
        help="Seconds between bridge stats polls.",
    )
    group.add_argument(
        "--generate-read-timeout",
        type=float,
        default=60.0,
        help="Fail a generation after this many seconds without any frame (the bridge sends heartbeats).",
    )

    args = parser.parse_args(argv)
    # Values read from files or Secrets often end in a newline, which would make
    # an invalid URL or Authorization header.
    bridge_url = (args.bridge_url or "").strip()
    bridge_api_key = (args.bridge_api_key or "").strip() or None
    if not bridge_url:
        parser.error("--bridge-url (or OMLX_BRIDGE_URL) is required")

    runtime_cfg = RuntimeConfig.from_cli_args(args)
    runtime_cfg.validate()
    if runtime_cfg.custom_jinja_template:
        template = os.path.expanduser(
            os.path.expandvars(runtime_cfg.custom_jinja_template)
        )
        if not os.path.isfile(template):
            raise FileNotFoundError(f"Custom Jinja template file not found: {template}")
        runtime_cfg.custom_jinja_template = template
    else:
        runtime_cfg.custom_jinja_template = None
    endpoint = (
        runtime_cfg.endpoint
        or f"dyn://{runtime_cfg.namespace}.{DEFAULT_COMPONENT}.{DEFAULT_ENDPOINT}"
    )
    runtime_cfg.namespace, runtime_cfg.component, runtime_cfg.endpoint = parse_endpoint(
        endpoint
    )

    model_path = Path(args.model_path).expanduser()
    model_config = _read_model_config(model_path)
    bos_token_id = _config_value(model_config, "bos_token_id")
    if isinstance(bos_token_id, list):
        bos_token_id = bos_token_id[0] if bos_token_id else None
    proxy_cfg = ProxyConfig(
        bridge_url=bridge_url,
        bridge_api_key=bridge_api_key,
        # Keep the path as given: the model card checksum covers it.
        model_path=args.model_path,
        served_model_name=args.served_model_name or model_path.resolve().name,
        context_length=args.context_length
        or _config_value(model_config, "max_position_embeddings"),
        bos_token_id=int(bos_token_id) if bos_token_id is not None else 0,
        expect_block_size=args.expect_block_size,
        total_kv_blocks=args.total_kv_blocks,
        max_output_tokens=args.max_output_tokens,
        kv_events=args.kv_events,
        startup_timeout_s=args.startup_timeout,
        mac_down_grace_s=args.mac_down_grace,
        stats_interval_s=args.stats_interval,
        generate_read_timeout_s=args.generate_read_timeout,
    )
    return proxy_cfg, runtime_cfg
