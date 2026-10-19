# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo worker for a remote oMLX bridge.

Usage::

    python -m dynamo_omlx.proxy --bridge-url http://<mac>:8000 --model-path <config dir> [Dynamo runtime flags]
"""

from dynamo_omlx.proxy.engine import OmlxProxyEngine

from dynamo.common.backend.run import run


def main() -> None:
    run(OmlxProxyEngine)


if __name__ == "__main__":
    main()
