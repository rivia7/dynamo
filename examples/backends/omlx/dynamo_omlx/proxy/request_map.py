# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Maps a Dynamo ``PreprocessedRequest`` onto a bridge generate body."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from dynamo_omlx.protocol import (
    PROTOCOL_VERSION,
    ProtocolError,
    parse_mm_hash,
    tokens_to_array,
)

try:
    import xxhash
except ImportError:  # ai-dynamo does not require it; the frontend's hashes do then
    xxhash = None

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


@dataclass(frozen=True)
class ImageRef:
    """One image of a request, as the Dynamo frontend passed it on."""

    url: str  # a data: or http(s) URL
    # The image's identity in Dynamo's KV router; None when it is unknown.
    mm_hash: int | None


def _url_mm_hash(url: str) -> int | None:
    # What the frontend hashes an image URL to when it does not decode images.
    return xxhash.xxh3_64_intdigest(url.encode()) if xxhash is not None else None


def _frontend_mm_hashes(request: Mapping[str, Any], count: int) -> list[int | None]:
    """The frontend's mm_hash of each image, sent while it routes by images."""
    extra = request.get("extra_args")
    hashes = extra.get("mm_hashes") if isinstance(extra, Mapping) else None
    if not isinstance(hashes, list) or len(hashes) != count:
        return [None] * count
    try:
        return [parse_mm_hash(value) for value in hashes]
    except ProtocolError:
        return [None] * count


def image_refs(request: Mapping[str, Any]) -> list[ImageRef]:
    """The images of ``request`` in prompt order; raises RequestRejected for other media."""
    media = request.get("multi_modal_data")
    if not media:
        return []
    if not isinstance(media, Mapping):
        raise RequestRejected("multi_modal_data must be an object")
    other = sorted(
        kind for kind, items in media.items() if kind != "image_url" and items
    )
    if other:
        raise RequestRejected("the oMLX backend does not support: " + ", ".join(other))
    urls = []
    for item in media.get("image_url") or ():
        url = item.get("Url") if isinstance(item, Mapping) else None
        if not isinstance(url, str):
            # Decoded media needs the frontend's NIXL path; UuidOnly needs a media cache.
            kind = (
                next(iter(item), "?")
                if isinstance(item, Mapping)
                else type(item).__name__
            )
            raise RequestRejected(f"the oMLX backend takes images as URLs, not {kind}")
        urls.append(url)
    hashes = _frontend_mm_hashes(request, len(urls))
    return [
        ImageRef(url=url, mm_hash=mm_hash if mm_hash is not None else _url_mm_hash(url))
        for url, mm_hash in zip(urls, hashes)
    ]


def _expanded_prompt_len(request: Mapping[str, Any]) -> int | None:
    """The prompt length with images, when the frontend routed by images."""
    info = request.get("mm_routing_info")
    length = info.get("expanded_prompt_len") if isinstance(info, Mapping) else None
    return length if isinstance(length, int) and not isinstance(length, bool) else None


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
) -> tuple[dict[str, Any], list[ImageRef]]:
    """Validate ``request`` and build the bridge ``/generate`` body.

    Returns the body and the request's images, which the caller loads into the
    body's ``images``. Raises :class:`RequestRejected` for anything oMLX cannot
    serve faithfully. Unset sampling fields stay unset so oMLX applies its
    per-model defaults.
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
    images = image_refs(request)
    expanded_prompt_len = _expanded_prompt_len(request) if images else None
    # Without the frontend's count, the bridge checks the context once it has
    # expanded the images.
    prompt_len = expanded_prompt_len or len(token_ids)

    budget = max_output_tokens
    if context_length is not None:
        if prompt_len >= context_length:
            raise RequestRejected(
                f"prompt has {prompt_len} tokens but the model context is {context_length}"
            )
        budget = min(budget, context_length - prompt_len)
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
    if expanded_prompt_len is not None:
        body["expanded_prompt_len"] = expanded_prompt_len
    sampling = request.get("sampling_options") or {}
    for key in _SAMPLING_KEYS:
        if sampling.get(key) is not None:
            body[key] = sampling[key]
    top_k = sampling.get("top_k")
    if top_k is not None:
        body["top_k"] = max(0, top_k)  # vLLM-style -1 means disabled; oMLX uses 0
    return body, images
