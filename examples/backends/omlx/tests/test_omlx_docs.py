# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Dynamo issues that README.md and AGENTS.md track."""

import re
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.post_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.timeout(30),
]

EXAMPLE = Path(__file__).resolve().parents[1]


def section(name: str, heading: str) -> str:
    """The text of a level-2 section, up to the next level-2 heading."""
    text = (EXAMPLE / name).read_text()
    match = re.search(rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    assert match, f"{name} has no '## {heading}' section"
    return match.group(1)


def checked_against(text: str) -> str:
    match = re.search(
        r"`main` means commit `([0-9a-f]{7,40})` of (\d{4}-\d{2}-\d{2})",
        " ".join(text.split()),
    )
    assert match, "no '`main` means commit `<sha>` of <date>'"
    return match.group(0)


def test_readme_and_agents_track_the_same_dynamo_issues():
    readme = section("README.md", "Dynamo Issues")
    agents = section("AGENTS.md", "Dynamo Issues")
    issues = re.findall(r"^\| \*\*(.+?)\*\*", readme, re.M)
    assert issues, "the README's table names no issue"
    assert re.findall(r"^- \*\*(.+?)\*\*", agents, re.M) == issues
    assert checked_against(readme) == checked_against(agents)
