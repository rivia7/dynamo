# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import hashlib
import io

import pytest
from dynamo_omlx.bridge.images import (
    cumulative_keys,
    decode_pixels,
    image_key,
    read_images,
)
from dynamo_omlx.bridge.service import BridgeError
from dynamo_omlx.protocol import ErrorCode

Image = pytest.importorskip("PIL.Image")

pytestmark = [pytest.mark.post_merge, pytest.mark.gpu_0, pytest.mark.unit]


def encode(image, fmt="PNG", **save) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **save)
    return base64.b64encode(buffer.getvalue()).decode()


def test_reading_identifies_files_without_decoding_pixels():
    payload = encode(Image.new("RGBA", (30, 40), (0, 0, 255, 128)))
    [file] = read_images([payload], max_bytes=0)
    assert file.data == base64.b64decode(payload)
    assert file.digest == hashlib.sha256(file.data).hexdigest()


def test_pixels_are_upright_rgb_at_full_resolution():
    # oMLX's own API shrinks images past 2048 px; the router counts the original.
    exif = Image.Exif()
    exif[0x0112] = 6  # stored rotated 90 degrees
    files = read_images(
        [
            encode(Image.new("RGBA", (2400, 40), (0, 0, 255, 128))),
            encode(Image.new("RGB", (40, 20)), "JPEG", exif=exif),
        ],
        max_bytes=0,
    )
    wide, photo = decode_pixels(files)
    assert (wide.mode, wide.size) == ("RGB", (2400, 40))
    assert photo.size == (20, 40)


def test_keys_follow_the_file_and_the_pixel_range():
    key = image_key("a" * 64, 65536, 4194304)
    assert key == image_key("a" * 64, 65536, 4194304)
    assert key not in {
        image_key("b" * 64, 65536, 4194304),
        image_key("a" * 64, 65536, 16777216),
        image_key("a" * 64, None, 4194304),
    }
    first, both = cumulative_keys(["x", "y"])
    assert first == cumulative_keys(["x"])[0]
    assert both != cumulative_keys(["y", "x"])[1] and len(both) == 64


@pytest.mark.parametrize(
    "payload, max_bytes, message",
    [
        ("not base64!", 0, "images\\[0\\] is not valid base64"),
        (base64.b64encode(b"plain text").decode(), 0, "not an image"),
        (encode(Image.new("RGB", (8, 8))), 10, "more than the 10"),
    ],
)
def test_bad_images_are_invalid_requests(payload, max_bytes, message):
    with pytest.raises(BridgeError, match=message) as info:
        read_images([payload], max_bytes=max_bytes)
    assert info.value.code == ErrorCode.INVALID_REQUEST


def test_a_file_that_breaks_while_decoding_is_an_invalid_request():
    # A sound PNG header in front of a broken image stream.
    data = base64.b64decode(encode(Image.new("RGB", (64, 64))))
    [file] = read_images([base64.b64encode(data[:60]).decode()], max_bytes=0)
    with pytest.raises(BridgeError, match="images\\[0\\] is not an image") as info:
        decode_pixels([file])
    assert info.value.code == ErrorCode.INVALID_REQUEST
