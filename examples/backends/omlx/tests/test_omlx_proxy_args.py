# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

try:
    from dynamo_omlx.proxy.args import parse_args
except ImportError as e:
    pytest.skip(f"Skip (missing dependency): {e}", allow_module_level=True)

pytestmark = [pytest.mark.post_merge, pytest.mark.gpu_0, pytest.mark.unit]


@pytest.fixture
def model_path(tmp_path, monkeypatch):
    # RuntimeConfig.validate() exports these; record them so they are restored.
    monkeypatch.setenv("DYN_RESPONSE_PLANE", "tcp")
    monkeypatch.setenv("DYN_TCP_TLS_INSECURE", "0")
    (tmp_path / "config.json").write_text(json.dumps({"max_position_embeddings": 64}))
    return str(tmp_path)


def parse(model_path, *extra):
    return parse_args(
        ["--model-path", model_path, "--discovery-backend", "file", *extra]
    )[0]


def test_strips_newlines_from_secret_values(model_path, monkeypatch):
    monkeypatch.setenv("OMLX_BRIDGE_URL", "http://mac-1:8000\n")
    monkeypatch.setenv("OMLX_API_KEY", "key\r\n")
    cfg = parse(model_path)
    assert (cfg.bridge_url, cfg.bridge_api_key) == ("http://mac-1:8000", "key")


def test_blank_key_means_no_key(model_path, monkeypatch):
    monkeypatch.setenv("OMLX_API_KEY", " \n")
    assert parse(model_path, "--bridge-url", "http://mac-1:8000").bridge_api_key is None


def test_blank_bridge_url_is_an_error(model_path, monkeypatch):
    monkeypatch.delenv("OMLX_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        parse(model_path, "--bridge-url", "\n")
