# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo bridge that runs inside an unmodified ``omlx serve`` process.

oMLX is imported only by ``omlx_adapter`` and the ``__main__`` entry point,
FastAPI only by ``routes``; the rest is stdlib so it can be tested without
either.
"""
