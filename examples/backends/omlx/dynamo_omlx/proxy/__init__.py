# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo worker on Linux that serves a remote oMLX bridge.

Only ``engine``, ``args`` and the ``__main__`` entry point import ``dynamo``;
the remaining modules are testable without an ai-dynamo build.
"""
