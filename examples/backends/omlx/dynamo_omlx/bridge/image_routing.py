# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where the images of a prompt sit, and the tokens Dynamo's router sees there.

Dynamo's frontend tokenizes one placeholder per image. For routing it expands
each placeholder into one pad value per image token, derived from the image's
mm_hash, and hashes blocks over those values. The bridge expands the
placeholders for oMLX by the same counts, remembers where each image landed,
and announces stored blocks with the image tokens replaced by the pad value,
so that the router's hashes of a cached prompt and of a new request agree.
"""

from __future__ import annotations

import threading
from array import array
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

from dynamo_omlx.protocol import pad_value_for_mm_hash


@dataclass(frozen=True, slots=True)
class ImageSpan:
    """The tokens of one image in an expanded prompt."""

    start: int
    length: int
    # The image's identity in Dynamo's router; None when the proxy had none.
    mm_hash: int | None


def expand_placeholders(
    token_ids: Sequence[int], placeholder: int, lengths: Sequence[int]
) -> tuple[list[int], list[int]]:
    """Replace the i-th ``placeholder`` token with ``lengths[i]`` copies of it.

    Returns the expanded tokens and the index where each image starts in them.
    """
    positions = [i for i, token in enumerate(token_ids) if token == placeholder]
    if len(positions) != len(lengths):
        raise ValueError(
            f"the prompt has {len(positions)} image placeholders for {len(lengths)} images"
        )
    expanded: list[int] = []
    starts: list[int] = []
    previous = 0
    for position, length in zip(positions, lengths):
        expanded.extend(token_ids[previous:position])
        starts.append(len(expanded))
        expanded.extend([placeholder] * length)
        previous = position + 1
    expanded.extend(token_ids[previous:])
    return expanded, starts


def routing_tokens(
    tokens: Sequence[int], spans: Sequence[ImageSpan]
) -> tuple[array, int]:
    """``tokens`` as Dynamo's router sees them, and how many of them it can know.

    Each image's tokens become the image's pad value. The router cannot know an
    image without an mm_hash, so the known prefix ends where such an image starts.
    """
    routed = array("I", tokens)
    known = len(routed)
    for span in sorted(spans, key=lambda s: s.start):
        start = min(span.start, len(routed))
        end = min(span.start + span.length, len(routed))
        if span.mm_hash is None:
            known = start
            break
        pad = pad_value_for_mm_hash(span.mm_hash)
        routed[start:end] = array("I", [pad]) * (end - start)
    return routed, known


class ImageRequests:
    """Image spans of recent requests by oMLX request id, for the store hook.

    oMLX stores a request's cache on its own thread after the request finished,
    so entries outlive their requests. Only the newest ``max_entries`` stay; a
    store that finds no entry announces no image blocks.
    """

    def __init__(self, max_entries: int = 4096) -> None:
        self._max_entries = max_entries
        self._lock = threading.Lock()
        self._spans: OrderedDict[str, tuple[ImageSpan, ...]] = OrderedDict()

    def add(self, request_id: str, spans: Sequence[ImageSpan]) -> None:
        with self._lock:
            self._spans[request_id] = tuple(spans)
            self._spans.move_to_end(request_id)
            while len(self._spans) > self._max_entries:
                self._spans.popitem(last=False)

    def get(self, request_id: str) -> tuple[ImageSpan, ...] | None:
        with self._lock:
            return self._spans.get(request_id)
