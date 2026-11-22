# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run ``omlx serve`` with the Dynamo bridge installed.

Usage::

    python -m dynamo_omlx.bridge --model-id <oMLX model id> -- serve <omlx serve flags>
    python -m dynamo_omlx.bridge --check
"""

import sys

from dynamo_omlx.bridge import omlx_adapter
from dynamo_omlx.bridge.config import parse_bridge_args
from omlx import cli as omlx_cli


def main() -> None:
    cfg, omlx_argv = parse_bridge_args(sys.argv[1:])
    if cfg.check:
        problems = omlx_adapter.probe()
        for problem in problems:
            print(f"incompatible: {problem}")
        # Image input is optional: the bridge turns it off and serves text.
        for problem in omlx_adapter.probe_vision():
            print(f"no image input: {problem}")
        print(
            "oMLX is compatible with the Dynamo bridge"
            if not problems
            else f"{len(problems)} problem(s) found"
        )
        sys.exit(1 if problems else 0)
    omlx_adapter.install(cfg)
    sys.argv = ["omlx", *omlx_argv]
    omlx_cli.main()


if __name__ == "__main__":
    main()
