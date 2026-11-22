# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test double for the ``lium`` CLI (run through a ``lium`` shell wrapper on PATH).

``$FAKE_LIUM_DIR/script.json`` maps a command key ("clusters up", "clusters show",
"clusters ps", "clusters list", "clusters rm", "ps") to a list of responses
``{"rc": int, "stdout": str | "stdout_file": path, "stderr": str |
"stderr_file": path}`` served in order; the last one repeats. Every call is
appended to ``$FAKE_LIUM_DIR/calls.jsonl`` with its argv and which variables
were set (never their values).
"""

import json
import os
import sys
from pathlib import Path


def _key(argv):
    if not argv:
        return ""
    if argv[0] == "clusters" and len(argv) > 1:
        return f"clusters {argv[1]}"
    return argv[0]


def _text(response, field):
    path = response.get(f"{field}_file")
    if path:
        text = Path(path).read_text(encoding="utf-8")
        # Fixtures carry the repo SPDX header; it is not part of lium's output.
        return "".join(
            line
            for line in text.splitlines(keepends=True)
            if not line.startswith("# SPDX-")
        )
    return response.get(field, "")


def main():
    state = Path(os.environ["FAKE_LIUM_DIR"])
    argv = sys.argv[1:]
    key = _key(argv)
    record = {
        "argv": argv,
        "key": key,
        "noninteractive": os.environ.get("LIUM_NONINTERACTIVE"),
        "output": os.environ.get("LIUM_OUTPUT"),
        "self_update_disable": os.environ.get("LIUM_SELF_UPDATE_DISABLE"),
        "api_key_set": bool(os.environ.get("LIUM_API_KEY")),
        "hf_token_set": "HF_TOKEN" in os.environ,
    }
    with open(state / "calls.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    script = json.loads((state / "script.json").read_text(encoding="utf-8"))
    responses = script.get(key)
    if not responses:
        sys.stderr.write(f"fake lium: no scripted response for {key!r}\n")
        return 99
    counts_path = state / "counts.json"
    counts = json.loads(counts_path.read_text()) if counts_path.exists() else {}
    index = counts.get(key, 0)
    counts[key] = index + 1
    counts_path.write_text(json.dumps(counts))
    response = responses[min(index, len(responses) - 1)]
    sys.stdout.write(_text(response, "stdout"))
    sys.stderr.write(_text(response, "stderr"))
    return int(response.get("rc", 0))


if __name__ == "__main__":
    sys.exit(main())
