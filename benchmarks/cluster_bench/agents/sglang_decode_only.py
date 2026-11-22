# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D-Only worker: ``python3 -m dynamo.sglang`` decode without a prefill peer.

Started by scripts/launch_node.sh inside the worker container when
``CB_DECODE_ONLY=1`` (domain/stages.py renders it for D-Only families, with
``--disaggregation-mode decode --disaggregation-transfer-backend fake``).
Two patches, then the stock entry point:

1. A request without ``bootstrap_info`` gets SGLang's fake bootstrap address
   (``FAKE_BOOTSTRAP_HOST``), the same address Dynamo's own decode health check
   sends. SGLang then allocates the prompt's KV pages, skips the transfer
   (``FakeKVReceiver`` succeeds at once) and decodes, so the decode stage
   carries its real batch, KV footprint and attention cost; only the KV
   contents (and with them the tokens) are meaningless.
2. The worker registers as a decode worker that needs no prefill peer
   (``needs=[]``, the value aggregated workers register), so the frontend
   reports the model ready and, with no prefill workers discovered, routes
   requests straight to decode (prefill router not activated).

``--check`` installs both patches and exits: preflight runs it in the image.

Verified against the Dynamo 1.6.2 / SGLang 0.5.21 sources of the pinned image
(and SGLang 0.5.23 of the nightly images); any other version that lacks the
patched names fails here, before serving.
"""

from __future__ import annotations

import logging
import random
import sys
from typing import Any, AsyncIterator

logger = logging.getLogger("cluster_bench.decode_only")
EXIT_UNSUPPORTED = 3


def _bootstrap_port(handler: Any) -> int:
    """The engine's bootstrap port; metadata only (the fake receiver ignores it)."""
    for path in (
        ("config", "server_args"),
        ("engine", "tokenizer_manager", "server_args"),
    ):
        obj: Any = handler
        for attr in path:
            obj = getattr(obj, attr, None)
        port = getattr(obj, "disaggregation_bootstrap_port", None)
        if isinstance(port, int) and port > 0:
            return port
    return 0


def fake_bootstrap(handler: Any, fake_host: str) -> dict[str, Any]:
    return {
        "bootstrap_host": fake_host,
        "bootstrap_port": _bootstrap_port(handler),
        "bootstrap_room": random.getrandbits(63),
    }


def install(
    handler_cls: Any,
    init_module: Any,
    decode_mode: Any,
    decode_type: Any,
    fake_host: str,
) -> None:
    """Patch ``handler_cls.generate`` and ``init_module``'s registration."""
    missing = [
        name
        for owner, name in (
            (handler_cls, "generate"),
            (init_module, "register_model_with_readiness_gate"),
        )
        if not hasattr(owner, name)
    ]
    if missing:
        raise RuntimeError(f"dynamo.sglang lacks {missing}; D-Only cannot run here")
    original_generate = handler_cls.generate
    original_register = init_module.register_model_with_readiness_gate

    async def generate(self: Any, request: Any, context: Any) -> AsyncIterator[Any]:
        if getattr(self, "serving_mode", None) == decode_mode and not request.get(
            "bootstrap_info"
        ):
            request = {**request, "bootstrap_info": fake_bootstrap(self, fake_host)}
        async for item in original_generate(self, request, context):
            yield item

    async def register(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("worker_type") == decode_type:
            kwargs["needs"] = []
        return await original_register(*args, **kwargs)

    handler_cls.generate = generate
    init_module.register_model_with_readiness_gate = register


def main(argv: list[str] | None = None) -> int:
    """``--check``: install the patches and exit (preflight); else serve."""
    check = (sys.argv[1:] if argv is None else argv) == ["--check"]
    try:
        from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST

        from dynamo.common.constants import DisaggregationMode
        from dynamo.llm import WorkerType
        from dynamo.sglang import init_llm
        from dynamo.sglang.main import main as dynamo_main
        from dynamo.sglang.request_handlers import DecodeWorkerHandler

        install(
            DecodeWorkerHandler,
            init_llm,
            DisaggregationMode.DECODE,
            WorkerType.Decode,
            FAKE_BOOTSTRAP_HOST,
        )
    except (ImportError, AttributeError, RuntimeError) as exc:
        print(f"cluster_bench decode-only shim: {exc}", file=sys.stderr, flush=True)
        return EXIT_UNSUPPORTED
    print(
        "cluster_bench decode-only shim active: fake KV transfer "
        f"(bootstrap host {FAKE_BOOTSTRAP_HOST}), no prefill peer",
        flush=True,
    )
    if check:
        return 0
    dynamo_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
