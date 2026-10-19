# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""deploy/base/token_auth.py, the Ingress's API token check."""

import importlib.util
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "deploy" / "base" / "token_auth.py"
_SPEC = importlib.util.spec_from_file_location("omlx_token_auth", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
token_auth = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(token_auth)

pytestmark = [
    pytest.mark.post_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.timeout(30),
]

ALICE = "a" * 40
BOB = "b" * 40


@pytest.fixture
def tokens_dir(tmp_path):
    """A directory laid out like a mounted Secret with the keys alice and bob."""
    data = tmp_path / "..2026_10_09"
    data.mkdir()
    (data / "alice").write_text(ALICE + "\n")
    (data / "bob").write_text(BOB)
    (tmp_path / "..data").symlink_to(data.name)
    for name in ("alice", "bob"):
        (tmp_path / name).symlink_to(Path("..data") / name)
    return tmp_path


def test_loads_one_token_per_holder(tokens_dir):
    assert token_auth.load_tokens(tokens_dir) == {ALICE: "alice", BOB: "bob"}


def test_ignores_short_and_empty_tokens(tmp_path):
    (tmp_path / "short").write_text("x" * (token_auth.MIN_TOKEN_LENGTH - 1))
    (tmp_path / "empty").write_text("\n")
    assert token_auth.load_tokens(tmp_path) == {}
    assert token_auth.load_tokens(tmp_path / "missing") == {}


@pytest.mark.parametrize(
    "authorization, api_key, expected",
    [
        (f"Bearer {ALICE}", "", ALICE),
        (f"bearer  {ALICE} ", "", ALICE),
        ("", f" {ALICE}", ALICE),
        (f"Basic {ALICE}", "", ""),
        ("Bearer ", "", ""),
        ("", "", ""),
    ],
)
def test_reads_bearer_or_api_key(authorization, api_key, expected):
    assert token_auth.presented_token(authorization, api_key) == expected


def test_names_the_holder():
    tokens = {ALICE: "alice", BOB: "bob"}
    assert token_auth.holder_of(BOB, tokens) == "bob"
    assert token_auth.holder_of(BOB[:-1] + "c", tokens) is None
    assert token_auth.holder_of(BOB, {}) is None


@pytest.fixture
def server(tokens_dir):
    token_auth.Handler.tokens_dir = tokens_dir
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), token_auth.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def status(url: str, method: str = "GET", **headers: str) -> tuple[int, bytes]:
    request = urllib.request.Request(url, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_answers_traefik(server, tokens_dir):
    assert status(f"{server}/check", Authorization=f"Bearer {ALICE}")[0] == 200
    assert status(f"{server}/check", "POST", **{"x-api-key": BOB})[0] == 200
    code, body = status(f"{server}/check", Authorization="Bearer wrong")
    assert code == 401
    assert json.loads(body)["error"]["type"] == "authentication_error"
    assert status(f"{server}/check")[0] == 401
    # Probes need no token.
    assert status(f"{server}/healthz")[0] == 200

    # A Secret change applies without a restart.
    (tokens_dir / "bob").unlink()
    assert status(f"{server}/check", **{"x-api-key": BOB})[0] == 401
