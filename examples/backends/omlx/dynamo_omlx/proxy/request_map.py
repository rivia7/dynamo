# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Maps a Dynamo ``PreprocessedRequest`` onto a bridge generate body."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dynamo_omlx.protocol import PROTOCOL_VERSION, ProtocolError, tokens_to_array

_SAMPLING_KEYS = (
    "temperature",
    "top_p",
    "min_p",
    "repetition_penalty",
    "presence_penalty",
    "frequency_penalty",
    "seed",
)


class RequestRejected(ValueError):
    """The request asks for something the oMLX backend cannot honour."""


def _unsupported(request: Mapping[str, Any], probe: bool) -> list[str]:
    stop = request.get("stop_conditions") or {}
    sampling = request.get("sampling_options") or {}
    output = request.get("output_options") or {}
    routing = request.get("routing") or {}
    checks = [
        # The runtime canary asks for ignore_eos with max_tokens=1; dropping it is harmless.
        ("ignore_eos", bool(stop.get("ignore_eos")) and not probe),
        ("min_tokens", (stop.get("min_tokens") or 0) > 0),
        ("max_thinking_tokens", stop.get("max_thinking_tokens") is not None),
        ("n > 1", (sampling.get("n") or 1) > 1 or (sampling.get("best_of") or 1) > 1),
        ("beam search", bool(sampling.get("use_beam_search"))),
        ("guided decoding", bool(sampling.get("guided_decoding"))),
        (
            "logprobs",
            output.get("logprobs") is not None
            or output.get("prompt_logprobs") is not None,
        ),
        ("LoRA adapters", bool(routing.get("lora_name"))),
        # oMLX shares one prefix cache per model, so it cannot isolate cache salts.
        ("cache_salt", bool(routing.get("cache_salt"))),
    ]
    checks.extend(
        (key, bool(request.get(key)))
        for key in (
            "multi_modal_data",
            "prompt_embeds",
            "prefill_result",
            "encoder_result",
        )
    )
    return [name for name, present in checks if present]


def _stop_token_ids(request: Mapping[str, Any], probe: bool) -> list[int]:
    stop = request.get("stop_conditions") or {}
    groups = [
        stop.get("stop_token_ids"),
        stop.get("stop_token_ids_hidden"),
        stop.get("stop_token_ids_visible"),
    ]
    if not (probe and stop.get("ignore_eos")):
        groups.append(request.get("eos_token_ids"))
    merged: dict[int, None] = {}
    for group in groups:
        for token_id in group or ():
            merged[token_id] = None
    return list(merged)


def build_generate_body(
    request: Mapping[str, Any],
    *,
    request_id: str,
    context_length: int | None,
    max_output_tokens: int,
    probe: bool = False,
) -> dict[str, Any]:
    """Validate ``request`` and build the bridge ``/generate`` body.

    Raises :class:`RequestRejected` for anything oMLX cannot serve faithfully.
    Unset sampling fields stay unset so oMLX applies its per-model defaults.
    """
    token_ids = request.get("token_ids")
    if not isinstance(token_ids, list) or not token_ids:
        raise RequestRejected("request has no token_ids")
    try:
        tokens_to_array(token_ids)
    except ProtocolError as e:
        raise RequestRejected(str(e)) from e
    unsupported = _unsupported(request, probe)
    if unsupported:
        raise RequestRejected(
            "the oMLX backend does not support: " + ", ".join(unsupported)
        )

    budget = max_output_tokens
    if context_length is not None:
        if len(token_ids) >= context_length:
            raise RequestRejected(
                f"prompt has {len(token_ids)} tokens but the model context is {context_length}"
            )
        budget = min(budget, context_length - len(token_ids))
    requested = (request.get("stop_conditions") or {}).get("max_tokens")
    max_tokens = budget if requested is None else max(1, min(requested, budget))

    body: dict[str, Any] = {
        "v": PROTOCOL_VERSION,
        "request_id": request_id,
        "token_ids": token_ids,
        "max_tokens": max_tokens,
        "stop_token_ids": _stop_token_ids(request, probe),
        # A canary must not add blocks to the prefix cache.
        "skip_cache_store": probe,
    }
    sampling = request.get("sampling_options") or {}
    for key in _SAMPLING_KEYS:
        if sampling.get(key) is not None:
            body[key] = sampling[key]
    top_k = sampling.get("top_k")
    if top_k is not None:
        body["top_k"] = max(0, top_k)  # vLLM-style -1 means disabled; oMLX uses 0
    return body
