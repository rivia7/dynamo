# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve an oMLX engine on macOS behind a Dynamo control plane on Linux.

``dynamo_omlx.protocol`` is the wire contract shared by both sides,
``dynamo_omlx.bridge`` runs inside oMLX on the Mac, and ``dynamo_omlx.proxy``
is the Dynamo worker that runs on Linux.
"""
