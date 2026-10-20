# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from dynamo_omlx.bridge.image_routing import (
    ImageRequests,
    ImageSpan,
    expand_placeholders,
    routing_tokens,
)
from dynamo_omlx.protocol import pad_value_for_mm_hash

pytestmark = [pytest.mark.post_merge, pytest.mark.gpu_0, pytest.mark.unit]

PAD, START, END = 9, 7, 8


def test_expand_placeholders_by_image_lengths():
    tokens = [1, START, PAD, END, 2, START, PAD, END]
    expanded, starts = expand_placeholders(tokens, PAD, [3, 1])
    assert expanded == [1, START, PAD, PAD, PAD, END, 2, START, PAD, END]
    assert starts == [2, 8]
    with pytest.raises(ValueError, match="2 image placeholders for 1 images"):
        expand_placeholders(tokens, PAD, [3])
    assert expand_placeholders([1, 2], PAD, []) == ([1, 2], [])


def test_routing_tokens_carry_each_images_pad_value():
    tokens = [1, START, PAD, PAD, END, START, PAD, END, 2]
    spans = [ImageSpan(2, 2, 0xA), ImageSpan(6, 1, 0xB)]
    routed, known = routing_tokens(tokens, spans)
    a, b = pad_value_for_mm_hash(0xA), pad_value_for_mm_hash(0xB)
    assert list(routed) == [1, START, a, a, END, START, b, END, 2]
    assert known == len(tokens)
    # A stored sequence can end inside an image.
    routed, known = routing_tokens(tokens[:3], spans)
    assert list(routed) == [1, START, a] and known == 3


def test_an_image_without_mm_hash_ends_what_the_router_can_know():
    tokens = [1, START, PAD, END, START, PAD, END]
    routed, known = routing_tokens(
        tokens, [ImageSpan(2, 1, 0xA), ImageSpan(5, 1, None)]
    )
    assert known == 5 and list(routed[:4]) == [
        1,
        START,
        pad_value_for_mm_hash(0xA),
        END,
    ]


def test_image_requests_keep_the_newest_entries():
    images = ImageRequests(max_entries=2)
    spans = (ImageSpan(0, 1, 1),)
    for request_id in ("a", "b", "c"):
        images.add(request_id, spans)
    assert images.get("a") is None and images.get("c") == spans
    assert images.get("missing") is None
