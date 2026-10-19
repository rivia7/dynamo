# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from dynamo_omlx.bridge.config import parse_bridge_args

pytestmark = [pytest.mark.post_merge, pytest.mark.gpu_0, pytest.mark.unit]

SERVE = ["--", "serve", "--model-dir", "/models"]


@pytest.mark.parametrize(
    "flags, mtp", [([], None), (["--mtp", "on"], True), (["--mtp", "off"], False)]
)
def test_mtp_is_on_off_or_left_alone(flags, mtp):
    cfg, omlx_argv = parse_bridge_args(["--model-id", "m", *flags, *SERVE])
    assert cfg.mtp is mtp
    assert omlx_argv == SERVE[1:]


def test_mtp_depth_is_1_to_8():
    cfg, _ = parse_bridge_args(["--model-id", "m", "--mtp-depth", "4", *SERVE])
    assert cfg.mtp_depth == 4
    for bad in ("0", "9"):
        with pytest.raises(SystemExit):
            parse_bridge_args(["--model-id", "m", "--mtp-depth", bad, *SERVE])
