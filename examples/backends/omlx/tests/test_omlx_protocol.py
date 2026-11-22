# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from dynamo_omlx.protocol import (
    ProtocolError,
    block_hash,
    decode_frame,
    decode_tokens,
    encode_frame,
    encode_tokens,
    format_mm_hash,
    pad_value_for_mm_hash,
    parse_blocks,
    parse_hashes,
    parse_mm_hash,
)

pytestmark = [pytest.mark.post_merge, pytest.mark.gpu_0, pytest.mark.unit]

MM_HASHES = [0, 1, 2**30 - 1, 2**30, 0xABCDEF0123456789, 2**64 - 1]


def test_pad_values_match_the_router():
    # dynamo_kv_router::protocols::pad_value_for_mm_hash keeps the low 30 bits.
    assert [pad_value_for_mm_hash(h) for h in MM_HASHES] == [
        1_000_000,
        1_000_001,
        1_000_000 + 2**30 - 1,
        1_000_000,
        1_000_000 + (0xABCDEF0123456789 & (2**30 - 1)),
        1_000_000 + 2**30 - 1,
    ]


def test_pad_values_match_dynamos_python_mirror():
    routing_utils = pytest.importorskip("dynamo.common.multimodal.routing_utils")
    for mm_hash in MM_HASHES:
        assert pad_value_for_mm_hash(mm_hash) == routing_utils.pad_value_for_mm_hash(
            mm_hash
        )


def test_mm_hash_round_trip():
    # The frontend writes extra_args.mm_hashes as format!("{:016x}").
    assert format_mm_hash(0xFF) == "00000000000000ff"
    for mm_hash in MM_HASHES:
        assert parse_mm_hash(format_mm_hash(mm_hash)) == mm_hash
    assert parse_mm_hash(None) is None
    for bad in ("ff", "g" * 16, " " * 16, 255, "0x" + "0" * 14):
        with pytest.raises(ProtocolError):
            parse_mm_hash(bad)


def test_frame_round_trip():
    frame = {"t": "tok", "ids": [1, 2, 3]}
    line = encode_frame(frame)
    assert line.endswith(b"\n") and line.count(b"\n") == 1
    assert decode_frame(line) == frame


@pytest.mark.parametrize(
    "line", [b"not json", b"[1, 2]", b'{"x": 1}', b'{"t": 5}', b"\xff\xfe"]
)
def test_decode_frame_rejects_malformed_lines(line):
    with pytest.raises(ProtocolError):
        decode_frame(line)


def test_token_encoding_golden_vector():
    # Little-endian uint32; changing this breaks mixed-version bridge/proxy pairs.
    assert encode_tokens([1, 2, 65536, 2**32 - 1]) == "AQAAAAIAAAAAAAEA/////w=="
    assert list(decode_tokens("AQAAAAIAAAAAAAEA/////w==")) == [1, 2, 65536, 2**32 - 1]


@pytest.mark.parametrize("tokens", [[-1], [2**32], [1.5]])
def test_encode_tokens_rejects_out_of_range_ids(tokens):
    with pytest.raises(ProtocolError):
        encode_tokens(tokens)


@pytest.mark.parametrize("payload", ["%%%", "AQA="])
def test_decode_tokens_rejects_bad_payloads(payload):
    with pytest.raises(ProtocolError):
        decode_tokens(payload)


def test_block_hash_golden_vectors():
    # Changing block_hash() requires bumping protocol.HASH_VERSION.
    root = block_hash(4, None, [1, 2, 3, 4])
    assert root == -4511206709735090996
    assert block_hash(4, root, [5, 6, 7, 8]) == -7944764126800262826
    assert block_hash(8, None, [1, 2, 3, 4, 5, 6, 7, 8]) == 4129036517190216141


def test_block_hash_depends_on_every_input():
    base = block_hash(4, None, [1, 2, 3, 4])
    assert block_hash(4, 7, [1, 2, 3, 4]) != base
    assert block_hash(4, None, [1, 2, 3, 5]) != base
    assert block_hash(2, None, [1, 2, 3, 4]) != base
    assert -(2**63) <= base < 2**63


def test_parse_blocks_validates_shape():
    wire = [
        {"h": 1, "p": None, "tok": encode_tokens([1, 2])},
        {"h": 2, "p": 1, "tok": encode_tokens([3, 4])},
    ]
    blocks = parse_blocks(wire, block_size=2)
    assert [(b.ext, b.parent, list(b.tokens)) for b in blocks] == [
        (1, None, [1, 2]),
        (2, 1, [3, 4]),
    ]
    with pytest.raises(ProtocolError, match="expected 3"):
        parse_blocks(wire, block_size=3)
    with pytest.raises(ProtocolError):
        parse_blocks(
            [{"h": True, "p": None, "tok": encode_tokens([1, 2])}], block_size=2
        )
    with pytest.raises(ProtocolError):
        parse_blocks(
            [{"h": 2**63, "p": None, "tok": encode_tokens([1, 2])}], block_size=2
        )


def test_parse_hashes():
    assert parse_hashes([1, -2]) == [1, -2]
    with pytest.raises(ProtocolError):
        parse_hashes("1,2")
    with pytest.raises(ProtocolError):
        parse_hashes([1.0])
