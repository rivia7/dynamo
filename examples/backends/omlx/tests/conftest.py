# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

# Make ``dynamo_omlx`` importable; the example is not an installed package.
_EXAMPLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_EXAMPLE_DIR))
