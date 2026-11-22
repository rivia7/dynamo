# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reads the images of a generate request and names them for oMLX's caches.

An image is known by its file and the pixel limits it is processed within:
its vision features and its tokens follow from those alone. So a request
whose images are all in oMLX's vision feature cache needs no pixel decoding.

Images keep the resolution the pixel limits allow, which the proxy takes from
the same preprocessor_config.json that Dynamo's frontend counts image tokens
by. oMLX's own API would first shrink an image to ``max_image_side_length``,
so its token count would no longer match the router's.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from dynamo_omlx.bridge.service import BridgeError
from dynamo_omlx.protocol import ErrorCode

_KEY_DOMAIN = b"dynamo-omlx-image-v1\0"
_CUMULATIVE_DOMAIN = b"dynamo-omlx-images-v1\0"


@dataclass(frozen=True)
class ImageFile:
    data: bytes  # the encoded image file
    digest: str  # its SHA-256


def _invalid(message: str) -> BridgeError:
    return BridgeError(ErrorCode.INVALID_REQUEST, message)


def read_images(payloads: Sequence[str], max_bytes: int) -> list[ImageFile]:
    """Decode base64 image files and check their headers; ``max_bytes`` <= 0 means no limit."""
    from PIL import Image

    files = []
    for index, payload in enumerate(payloads):
        field = f"images[{index}]"
        try:
            data = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as e:
            raise _invalid(f"{field} is not valid base64: {e}") from e
        if max_bytes > 0 and len(data) > max_bytes:
            raise _invalid(
                f"{field} has {len(data)} bytes, more than the {max_bytes} oMLX accepts"
            )
        try:
            # Reads the header only; decoding waits for a vision feature cache miss.
            with Image.open(io.BytesIO(data)):
                pass
        except Image.DecompressionBombError as e:
            raise _invalid(f"{field} has more pixels than PIL decodes safely") from e
        # PIL raises many types for broken files.
        except Exception as e:  # noqa: BLE001
            raise _invalid(f"{field} is not an image PIL can read: {e}") from e
        files.append(ImageFile(data=data, digest=hashlib.sha256(data).hexdigest()))
    return files


def decode_pixels(files: Sequence[ImageFile]) -> list[Any]:
    """Upright RGB PIL images, as mlx-vlm loads them."""
    from PIL import Image, ImageOps

    images = []
    for index, file in enumerate(files):
        try:
            with Image.open(io.BytesIO(file.data)) as loaded:
                # Phone photos store their orientation in EXIF.
                images.append(ImageOps.exif_transpose(loaded).convert("RGB"))
        # PIL raises many types for broken files.
        except Exception as e:  # noqa: BLE001
            raise _invalid(
                f"images[{index}] is not an image PIL can decode: {e}"
            ) from e
    return images


def image_key(digest: str, min_pixels: int | None, max_pixels: int | None) -> str:
    """oMLX's cache identity of an image file processed within these pixel limits."""
    h = hashlib.sha256(_KEY_DOMAIN)
    h.update(f"{digest}\0{min_pixels}\0{max_pixels}".encode())
    return h.hexdigest()


def cumulative_keys(keys: Sequence[str]) -> list[str]:
    """For each image, the identity of the images up to and including it."""
    running = hashlib.sha256(_CUMULATIVE_DOMAIN)
    out = []
    for key in keys:
        running.update(key.encode())
        out.append(running.copy().hexdigest())
    return out
