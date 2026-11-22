# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""agents/sglang_decode_only.py against stand-ins for dynamo.sglang (no GPUs)."""

from __future__ import annotations

import asyncio
import sys
import types
from typing import Any

import pytest

from benchmarks.cluster_bench.agents import sglang_decode_only as shim

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

FAKE_HOST = "2.2.2.2"


class Handler:
    serving_mode = "decode"

    def __init__(self, port: int = 8998):
        self.config = types.SimpleNamespace(
            server_args=types.SimpleNamespace(disaggregation_bootstrap_port=port)
        )

    async def generate(self, request: dict[str, Any], context: Any):
        yield request


def init_module() -> types.SimpleNamespace:
    seen: list[dict[str, Any]] = []

    async def register_model_with_readiness_gate(*args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs)
        return kwargs

    return types.SimpleNamespace(
        register_model_with_readiness_gate=register_model_with_readiness_gate, seen=seen
    )


def collect(gen: Any) -> list[Any]:
    async def run() -> list[Any]:
        return [item async for item in gen]

    return asyncio.run(run())


def test_decode_requests_get_the_fake_bootstrap() -> None:
    handler_cls = type("DecodeHandler", (Handler,), {})
    shim.install(handler_cls, init_module(), "decode", "Decode", FAKE_HOST)
    (sent,) = collect(handler_cls().generate({"token_ids": [1, 2]}, None))
    info = sent["bootstrap_info"]
    assert (info["bootstrap_host"], info["bootstrap_port"]) == (FAKE_HOST, 8998)
    assert 0 <= info["bootstrap_room"] < 2**63
    given = {"token_ids": [1], "bootstrap_info": {"bootstrap_host": "10.0.0.5"}}
    assert collect(handler_cls().generate(given, None)) == [given]
    prefill = handler_cls()
    prefill.serving_mode = "prefill"
    assert collect(prefill.generate({"token_ids": [1]}, None)) == [{"token_ids": [1]}]


def test_decode_workers_register_without_a_prefill_peer() -> None:
    module = init_module()
    shim.install(type("H", (Handler,), {}), module, "decode", "Decode", FAKE_HOST)
    asyncio.run(
        module.register_model_with_readiness_gate(
            "ep", worker_type="Decode", needs=["p"]
        )
    )
    asyncio.run(
        module.register_model_with_readiness_gate(
            "ep", worker_type="Prefill", needs=["x"]
        )
    )
    assert module.seen == [
        {"worker_type": "Decode", "needs": []},
        {"worker_type": "Prefill", "needs": ["x"]},
    ]


def test_install_refuses_a_dynamo_without_the_patched_names() -> None:
    with pytest.raises(RuntimeError, match="register_model_with_readiness_gate"):
        shim.install(Handler, types.SimpleNamespace(), "decode", "Decode", FAKE_HOST)


def test_bootstrap_port_falls_back_to_the_engine_args() -> None:
    engine = types.SimpleNamespace(
        tokenizer_manager=types.SimpleNamespace(
            server_args=types.SimpleNamespace(disaggregation_bootstrap_port=9001)
        )
    )
    assert shim._bootstrap_port(types.SimpleNamespace(engine=engine)) == 9001
    assert shim._bootstrap_port(object()) == 0


def fake_dynamo(
    monkeypatch: pytest.MonkeyPatch, init_llm: Any
) -> tuple[type, list[bool]]:
    """sys.modules stand-ins for the names the shim imports."""
    handler = type("DecodeWorkerHandler", (Handler,), {})
    ran: list[bool] = []

    def module(name: str, **attrs: Any) -> None:
        mod = types.ModuleType(name)
        for key, val in attrs.items():
            setattr(mod, key, val)
        monkeypatch.setitem(sys.modules, name, mod)

    for name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.disaggregation",
        "dynamo",
        "dynamo.common",
    ):
        module(name)
    module("sglang.srt.disaggregation.utils", FAKE_BOOTSTRAP_HOST=FAKE_HOST)
    module(
        "dynamo.common.constants",
        DisaggregationMode=types.SimpleNamespace(DECODE="decode"),
    )
    module("dynamo.llm", WorkerType=types.SimpleNamespace(Decode="Decode"))
    module("dynamo.sglang", init_llm=init_llm)
    monkeypatch.setitem(sys.modules, "dynamo.sglang.init_llm", init_llm)
    module("dynamo.sglang.main", main=lambda: ran.append(True))
    module("dynamo.sglang.request_handlers", DecodeWorkerHandler=handler)
    return handler, ran


def test_check_installs_the_patches_and_exits(monkeypatch, capsys) -> None:
    handler, ran = fake_dynamo(monkeypatch, init_module())
    original = handler.generate
    assert shim.main(["--check"]) == 0
    assert handler.generate is not original and ran == []
    assert "decode-only shim active" in capsys.readouterr().out


def test_serving_runs_the_stock_entry_point(monkeypatch) -> None:
    _handler, ran = fake_dynamo(monkeypatch, init_module())
    assert shim.main(["--model-path", "/m"]) == 0 and ran == [True]


def test_check_fails_on_an_image_without_the_patched_names(monkeypatch, capsys) -> None:
    fake_dynamo(monkeypatch, types.SimpleNamespace())
    assert shim.main(["--check"]) == shim.EXIT_UNSUPPORTED
    assert "D-Only cannot run here" in capsys.readouterr().err


def test_check_fails_without_sglang(monkeypatch) -> None:
    for name in ("sglang", "sglang.srt", "sglang.srt.disaggregation"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "sglang.srt.disaggregation.utils", None)
    assert shim.main(["--check"]) == shim.EXIT_UNSUPPORTED
