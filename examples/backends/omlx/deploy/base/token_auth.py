# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API token check for the example's Ingress, behind Traefik's ForwardAuth.

Traefik sends the headers of every request for /v1 here first and passes the
request on only when this answers 200. A request passes with a token from
TOKENS_DIR, one file per token holder (the keys of a mounted Secret), in
``Authorization: Bearer <token>`` (OpenAI SDKs, Codex, Claude Code's
ANTHROPIC_AUTH_TOKEN) or ``x-api-key: <token>`` (Anthropic SDKs, Claude Code's
ANTHROPIC_API_KEY). The files are read on every check, so a changed Secret
applies once the kubelet has synced it, without a restart.

    python3 token_auth.py [--port 8080] [--tokens-dir /tokens]
"""

from __future__ import annotations

import argparse
import hmac
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Shorter tokens are ignored: they could be guessed.
MIN_TOKEN_LENGTH = 32
# The error shape of Anthropic's API; OpenAI SDKs read error.message as well.
DENIED = json.dumps(
    {
        "type": "error",
        "error": {
            "type": "authentication_error",
            "message": "this API needs a valid token: Authorization: Bearer <token> "
            "or x-api-key: <token>",
        },
    }
).encode()

log = logging.getLogger("token_auth")


def load_tokens(directory: Path) -> dict[str, str]:
    """Token -> holder, from the files in ``directory``."""
    tokens: dict[str, str] = {}
    try:
        entries = sorted(directory.iterdir())
    except FileNotFoundError:
        return tokens
    for entry in entries:
        # A mounted Secret also holds ..data and timestamped directories.
        if entry.name.startswith(".") or not entry.is_file():
            continue
        token = entry.read_text().strip()
        if len(token) < MIN_TOKEN_LENGTH:
            log.warning(
                "ignoring the token of %s: shorter than %d characters",
                entry.name,
                MIN_TOKEN_LENGTH,
            )
            continue
        tokens[token] = entry.name
    return tokens


def presented_token(authorization: str, api_key: str) -> str:
    """The token a request presents in either header, or ""."""
    scheme, _, value = authorization.strip().partition(" ")
    if scheme.lower() == "bearer" and value.strip():
        return value.strip()
    return api_key.strip()


def holder_of(token: str, tokens: dict[str, str]) -> str | None:
    """The holder of ``token``; every comparison takes constant time."""
    holder = None
    for known, name in tokens.items():
        if hmac.compare_digest(known.encode(), token.encode()):
            holder = name
    return holder


class Handler(BaseHTTPRequestHandler):
    tokens_dir = Path("/tokens")

    def check(self) -> None:
        if self.path == "/healthz":
            self.answer(200, b"ok\n", "text/plain")
            return
        token = presented_token(
            self.headers.get("Authorization", ""), self.headers.get("x-api-key", "")
        )
        holder = holder_of(token, load_tokens(self.tokens_dir)) if token else None
        client = self.headers.get("X-Forwarded-For") or self.client_address[0]
        request = (
            f"{self.headers.get('X-Forwarded-Method', self.command)} "
            f"{self.headers.get('X-Forwarded-Uri', self.path)}"
        )
        if holder is None:
            reason = "a wrong token" if token else "no token"
            log.info("denied %s from %s: %s", request, client, reason)
            self.answer(401, DENIED, "application/json")
            return
        log.info("allowed %s from %s for %s", request, client, holder)
        self.answer(200, b"", "text/plain")

    def answer(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = check

    def log_message(self, format: str, *args: object) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--tokens-dir", type=Path, default=Path("/tokens"))
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    Handler.tokens_dir = args.tokens_dir
    holders = sorted(load_tokens(args.tokens_dir).values())
    log.info("token holders: %s", ", ".join(holders) or "none, so every request fails")
    ThreadingHTTPServer(("", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
