# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark datasets: what ``--dataset`` names and how AIPerf 0.14.0 loads it.

* an AgentX trace (default ``semianalysis_cc_traces_weka_062126``, the
  date-pinned full corpus; ``_256k`` is its variant without requests over 256k
  tokens): the SemiAnalysis Claude Code corpus replayed closed-loop by the
  ``agentx`` scenario (session affinity, cache bust, trace delays,
  ``--max-context-length`` trace filter);
* any other AIPerf public dataset (``sharegpt``, ``aimo``, ``spec_bench``, ...):
  plain closed-loop concurrency;
* a dataset file on node0: ``<custom-dataset-type>:<path>``, e.g.
  ``mooncake_trace:/root/cluster_bench/inputs/trace.jsonl`` (a ``weka_trace``
  file is replayed by the AgentX scenario, with ``--unsafe-override`` since a
  local file cannot pin the corpus); ``file:<path>`` means ``mooncake_trace``;
* synthetic prompts: ``synthetic:isl=8192,osl=1024[,isl_std=0,osl_std=0]``.
"""

from __future__ import annotations

import re
from typing import Literal

from benchmarks.cluster_bench.domain.base import Frozen

DatasetKind = Literal["agentx", "public", "file", "synthetic"]
DEFAULT_DATASET = "semianalysis_cc_traces_weka_062126"
# Loaders the agentx scenario accepts (require_loader in AIPerf 0.14.0's
# aiperf/common/scenario/inferencex_agentx_mvp.py). 0.14.0 also accepts
# mooncake_trace there; the harness replays mooncake files closed-loop instead
# (kind "file"), as before.
AGENTX_LOADERS = frozenset(
    {
        "semianalysis_cc_traces_weka_with_subagents",
        "semianalysis_cc_traces_weka_with_subagents_256k",
        "semianalysis_cc_traces_weka_with_subagents_060226",
        "semianalysis_cc_traces_weka_with_subagents_060226_256k",
        "semianalysis_cc_traces_weka_with_subagents_060526",
        "semianalysis_cc_traces_weka_with_subagents_060526_256k",
        "semianalysis_cc_traces_weka_with_subagents_060826",
        "semianalysis_cc_traces_weka_with_subagents_060826_256k",
        "semianalysis_cc_traces_weka_061326",
        "semianalysis_cc_traces_weka_061326_256k",
        "semianalysis_cc_traces_weka_061526",
        "semianalysis_cc_traces_weka_061526_256k",
        "semianalysis_cc_traces_weka_062126",
        "semianalysis_cc_traces_weka_062126_256k",
        "weka_trace",
        "weka_hf",
    }
)
DEFAULT_FILE_TYPE = "mooncake_trace"
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_SYNTH_KEYS = ("isl", "osl", "isl_std", "osl_std")


class DatasetError(ValueError):
    pass


class DatasetSpec(Frozen):
    kind: DatasetKind
    text: str  # what --dataset said (recorded in the summary and the KB)
    public: str | None = None  # --public-dataset alias
    path: str | None = None  # --input-file on node0
    file_type: str | None = None  # --custom-dataset-type
    isl: int | None = None
    isl_std: int = 0
    osl: int | None = None
    osl_std: int = 0

    @property
    def agentx(self) -> bool:
        return self.kind == "agentx"

    @property
    def label(self) -> str:
        """Short human label for reports."""
        if self.kind == "synthetic":
            return f"synthetic ISL {self.isl} / OSL {self.osl}"
        if self.kind == "file" or (self.agentx and self.path):
            return f"{self.file_type} file {self.path}"
        return self.public or self.text


def _synthetic(text: str, body: str) -> DatasetSpec:
    values: dict[str, int] = {}
    for item in filter(None, (p.strip() for p in body.split(","))):
        key, sep, raw = item.partition("=")
        key = key.strip()
        if not sep or key not in _SYNTH_KEYS:
            raise DatasetError(
                f"{text!r}: expected synthetic:isl=N,osl=M[,isl_std=S,osl_std=S]"
            )
        try:
            values[key] = int(raw)
        except ValueError as exc:
            raise DatasetError(f"{text!r}: {key} must be an integer") from exc
    if values.get("isl", 0) < 1 or values.get("osl", 0) < 1:
        raise DatasetError(f"{text!r}: synthetic datasets need isl >= 1 and osl >= 1")
    if values.get("isl_std", 0) < 0 or values.get("osl_std", 0) < 0:
        raise DatasetError(f"{text!r}: standard deviations must be >= 0")
    return DatasetSpec(
        kind="synthetic",
        text=text,
        isl=values["isl"],
        osl=values["osl"],
        isl_std=values.get("isl_std", 0),
        osl_std=values.get("osl_std", 0),
    )


def parse_dataset(text: str) -> DatasetSpec:
    """``--dataset`` -> DatasetSpec (see the module docstring)."""
    raw = text.strip()
    if not raw:
        raise DatasetError("empty --dataset")
    head, sep, rest = raw.partition(":")
    if sep and head == "synthetic":
        return _synthetic(raw, rest)
    if sep:
        file_type = DEFAULT_FILE_TYPE if head == "file" else head
        if not _NAME_RE.match(file_type):
            raise DatasetError(f"{raw!r}: invalid dataset type {file_type!r}")
        if not rest.startswith("/"):
            raise DatasetError(
                f"{raw!r}: the dataset file must be an absolute path on node0 "
                "(`controller sync --upload FILE` copies a local file to "
                "<workdir>/inputs/<name>)"
            )
        return DatasetSpec(
            kind="agentx" if file_type in AGENTX_LOADERS else "file",
            text=raw,
            path=rest,
            file_type=file_type,
        )
    if not _NAME_RE.match(raw):
        raise DatasetError(f"{raw!r} is not an AIPerf public dataset name")
    return DatasetSpec(
        kind="agentx" if raw in AGENTX_LOADERS else "public", text=raw, public=raw
    )
