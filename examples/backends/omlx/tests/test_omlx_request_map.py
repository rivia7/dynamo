# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from dynamo_omlx.protocol import PROTOCOL_VERSION
from dynamo_omlx.proxy import request_map
from dynamo_omlx.proxy.request_map import (
    ImageRef,
    RequestRejected,
    build_generate_body,
    image_refs,
)

pytestmark = [pytest.mark.post_merge, pytest.mark.gpu_0, pytest.mark.unit]

PNG = "data:image/png;base64,iVBORw0KGgo="


def build_with_images(request, context_length=100, max_output_tokens=50, probe=False):
    return build_generate_body(
        request,
        request_id="ctx-1",
        context_length=context_length,
        max_output_tokens=max_output_tokens,
        probe=probe,
    )


def build(request, **kwargs):
    body, _ = build_with_images(request, **kwargs)
    return body


def test_maps_tokens_sampling_and_stop_ids():
    body = build(
        {
            "token_ids": [1, 2, 3],
            "stop_conditions": {
                "max_tokens": 10,
                "stop": ["</s>"],
                "stop_token_ids": [7],
                "stop_token_ids_hidden": [8, 7],
            },
            "sampling_options": {
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": -1,
                "seed": 42,
                "min_p": None,
            },
            "eos_token_ids": [9],
        }
    )
    assert body == {
        "v": PROTOCOL_VERSION,
        "request_id": "ctx-1",
        "token_ids": [1, 2, 3],
        "max_tokens": 10,
        "stop_token_ids": [7, 8, 9],
        "skip_cache_store": False,
        "temperature": 0.6,
        "top_p": 0.95,
        "seed": 42,
        "top_k": 0,
    }


def test_max_tokens_defaults_and_is_capped_by_the_context():
    assert build({"token_ids": [1] * 10})["max_tokens"] == 50
    assert build({"token_ids": [1] * 90})["max_tokens"] == 10
    assert (
        build({"token_ids": [1] * 90, "stop_conditions": {"max_tokens": 30}})[
            "max_tokens"
        ]
        == 10
    )
    assert build({"token_ids": [1]}, context_length=None)["max_tokens"] == 50
    with pytest.raises(RequestRejected, match="context"):
        build({"token_ids": [1] * 100})


@pytest.mark.parametrize(
    "extra, feature",
    [
        ({"stop_conditions": {"ignore_eos": True}}, "ignore_eos"),
        ({"stop_conditions": {"min_tokens": 4}}, "min_tokens"),
        ({"stop_conditions": {"max_thinking_tokens": 64}}, "max_thinking_tokens"),
        ({"sampling_options": {"n": 2}}, "n > 1"),
        ({"sampling_options": {"use_beam_search": True}}, "beam search"),
        ({"sampling_options": {"guided_decoding": {"json": {}}}}, "guided decoding"),
        ({"output_options": {"logprobs": 1}}, "logprobs"),
        ({"output_options": {"prompt_logprobs": 0}}, "logprobs"),
        ({"routing": {"lora_name": "adapter"}}, "LoRA"),
        ({"routing": {"cache_salt": "tenant"}}, "cache_salt"),
        ({"multi_modal_data": {"video_url": [{"Url": "v"}]}}, "video_url"),
        ({"multi_modal_data": {"image_url": [{"Decoded": {}}]}}, "not Decoded"),
        ({"multi_modal_data": {"image_url": [{"UuidOnly": "u"}]}}, "not UuidOnly"),
        ({"multi_modal_data": {"image_url": ["x"]}}, "not str"),
        ({"prefill_result": {"x": 1}}, "prefill_result"),
    ],
)
def test_rejects_features_omlx_cannot_honour(extra, feature):
    with pytest.raises(RequestRejected, match=feature):
        build({"token_ids": [1], **extra})


def test_allows_neutral_defaults():
    body = build(
        {
            "token_ids": [1],
            "stop_conditions": {"min_tokens": 0, "ignore_eos": False},
            "sampling_options": {"n": 1, "best_of": None},
            "output_options": {"skip_special_tokens": True},
            "routing": {"priority": 1},
        }
    )
    assert body["token_ids"] == [1]


@pytest.mark.parametrize("token_ids", [None, [], [-1], [2**32], "abc"])
def test_rejects_invalid_token_ids(token_ids):
    with pytest.raises(RequestRejected):
        build({"token_ids": token_ids})


def image_request(**extra):
    return {
        "token_ids": [1, 9, 2],
        "multi_modal_data": {"image_url": [{"Url": PNG}, {"Url": "https://x/y.png"}]},
        **extra,
    }


def test_images_take_the_frontends_mm_hashes():
    request = image_request(
        extra_args={"mm_hashes": ["00000000000000ff", "ABCDEF0123456789"]}
    )
    assert image_refs(request) == [
        ImageRef(url=PNG, mm_hash=0xFF),
        ImageRef(url="https://x/y.png", mm_hash=0xABCDEF0123456789),
    ]
    assert image_refs({"token_ids": [1]}) == []
    assert image_refs({"multi_modal_data": {"image_url": [], "video_url": []}}) == []


@pytest.mark.parametrize(
    "extra_args", [None, {"mm_hashes": ["00ff"]}, {"mm_hashes": ["xx" * 8, "0" * 16]}]
)
def test_images_hash_their_urls_without_usable_frontend_hashes(extra_args, monkeypatch):
    # Stands in for xxhash, which ai-dynamo does not require.
    fake = type("xxhash", (), {"xxh3_64_intdigest": staticmethod(len)})
    monkeypatch.setattr(request_map, "xxhash", fake)
    refs = image_refs(image_request(extra_args=extra_args))
    assert [ref.mm_hash for ref in refs] == [len(PNG), len("https://x/y.png")]
    monkeypatch.setattr(request_map, "xxhash", None)
    assert [ref.mm_hash for ref in image_refs(image_request())] == [None, None]


def test_url_hash_matches_the_frontends_xxh3():
    xxhash = pytest.importorskip("xxhash")
    assert request_map._url_mm_hash(PNG) == xxhash.xxh3_64_intdigest(PNG.encode())
    # The frontend hashes a URL with XXH3-64 and seed 0 (preprocessor.rs).
    assert xxhash.xxh3_64_intdigest(b"") == 0x2D06800538D394C2


def test_images_budget_with_the_frontends_expanded_length():
    body, images = build_with_images(
        image_request(mm_routing_info={"expanded_prompt_len": 80})
    )
    assert len(images) == 2
    assert body["expanded_prompt_len"] == 80 and body["max_tokens"] == 20
    with pytest.raises(RequestRejected, match="100 tokens"):
        build(image_request(mm_routing_info={"expanded_prompt_len": 100}))
    # Without the frontend's count the bridge checks the context after expanding.
    body, _ = build_with_images(image_request())
    assert "expanded_prompt_len" not in body and body["max_tokens"] == 50
    assert "expanded_prompt_len" not in build(
        {"token_ids": [1], "mm_routing_info": {"expanded_prompt_len": 9}}
    )


def test_health_probe_drops_ignore_eos_and_skips_the_cache():
    body = build(
        {
            "token_ids": [0],
            "stop_conditions": {"max_tokens": 1, "ignore_eos": True},
            "eos_token_ids": [2],
        },
        probe=True,
    )
    assert body["skip_cache_store"] is True and body["max_tokens"] == 1
    assert body["stop_token_ids"] == []
