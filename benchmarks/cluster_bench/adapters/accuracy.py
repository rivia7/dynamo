# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the few-shot GSM8K accuracy probe on a cluster node. Fail-soft.

Both steps run ``agents/accuracy_probe.py`` with the runner's Python through
an injected ``ports.Cluster`` - on node0 in practice, where the Dynamo
frontend serves the OpenAI API:

* ``ensure_dataset``  ``fetch``: download the GSM8K test split into a cache
                      directory under the workdir (curl or urllib, retried),
                      verify its sha256, reuse a verified copy (~1 s; nothing
                      to download when cached).
* ``run_accuracy``    ``gsm8k``: ask the questions through the frontend's
                      ``/v1/completions`` while the deployment is up
                      (typically 0.5-2 min for 200 questions at concurrency
                      32; never longer than ``params.deadline_s`` plus a
                      margin).

Every remote command has a timeout. A failed, timed-out or garbled run
becomes ``DatasetStatus(ok=False)`` / ``AccuracyResult(status="error")`` with
a note and a log line, never an exception. The agent writes its document to
``out`` (evidence for the knowledge base) and prints it; when stdout carries
no document the file is read back. Commands are rendered by pure
``*_command`` helpers so ``--dry-run`` can print them, and
``simulated_stdout`` answers them with synthetic documents.
"""

from __future__ import annotations

import json
import logging
import shlex
from typing import Sequence

from benchmarks.cluster_bench.domain.accuracy import (
    FETCH_KIND,
    GSM8K_FILENAME,
    GSM8K_SHA256,
    GSM8K_URL,
    PROBE_KIND,
    SYNTH_ACCURACY,
    AccuracyParams,
    AccuracyResult,
    DatasetStatus,
    ProbeNote,
    error_result,
    failed_dataset,
    find_document,
    parse_fetch_document,
    parse_probe_document,
    synthetic_fetch_document,
    synthetic_probe_document,
)
from benchmarks.cluster_bench.ports import Cluster, CommandError

logger = logging.getLogger(__name__)

AGENT_REL = "benchmarks/cluster_bench/agents/accuracy_probe.py"
DATASETS_REL = "datasets"
FETCH_ATTEMPTS = 3
FETCH_ATTEMPT_TIMEOUT_S = 60.0
FETCH_MARGIN_S = 60.0
# Interpreter start, dataset read and JSON write on top of the agent deadline
# (the agent abandons in-flight requests at its deadline and exits).
PROBE_MARGIN_S = 120.0
READ_TIMEOUT_S = 30.0
ERROR_TAIL_CHARS = 1000


# --------------------------------------------------------------------------
# Pure command builders
# --------------------------------------------------------------------------


def agent_path(src_dir: str) -> str:
    """The agent under the replicated source tree (``<workdir>/src``)."""
    return f"{src_dir.rstrip('/')}/{AGENT_REL}"


def dataset_cache_dir(workdir: str) -> str:
    return f"{workdir.rstrip('/')}/{DATASETS_REL}"


def dataset_path(cache_dir: str, filename: str = GSM8K_FILENAME) -> str:
    return f"{cache_dir.rstrip('/')}/{filename}"


def fetch_command(
    python: str,
    agent: str,
    *,
    dest: str,
    urls: Sequence[str] = (GSM8K_URL,),
    sha256: str = GSM8K_SHA256,
    attempts: int = FETCH_ATTEMPTS,
    attempt_timeout_s: float = FETCH_ATTEMPT_TIMEOUT_S,
) -> str:
    argv = [python, agent, "fetch"]
    for url in urls:
        argv += ["--url", url]
    argv += [
        "--sha256",
        sha256,
        "--dest",
        dest,
        "--attempts",
        str(int(attempts)),
        "--timeout-s",
        f"{attempt_timeout_s:g}",
    ]
    return shlex.join(argv)


def fetch_timeout_s(
    *, urls: int, attempts: int, attempt_timeout_s: float = FETCH_ATTEMPT_TIMEOUT_S
) -> float:
    """Worst case of ``fetch``: curl then urllib per URL and attempt, backoff."""
    downloads = attempts * max(1, urls) * 2 * (attempt_timeout_s + 10.0)
    backoff = 2.0 * attempts * (attempts - 1) / 2
    return downloads + backoff + FETCH_MARGIN_S


def probe_command(
    python: str,
    agent: str,
    *,
    url: str,
    model: str,
    dataset: str,
    out: str,
    params: AccuracyParams,
    expected_sha256: str | None = GSM8K_SHA256,
) -> str:
    """``gsm8k`` command; ``url`` is the frontend base URL."""
    argv = [
        python,
        agent,
        "gsm8k",
        "--url",
        url,
        "--model",
        model,
        "--dataset",
        dataset,
        "--shots",
        str(params.shots),
        "--questions",
        str(params.questions),
        "--concurrency",
        str(params.concurrency),
        "--max-tokens",
        str(params.max_tokens),
        "--request-timeout-s",
        f"{params.request_timeout_s:g}",
        "--deadline-s",
        f"{params.deadline_s:g}",
        "--retries",
        str(params.retries),
        "--out",
        out,
    ]
    if expected_sha256:
        argv += ["--expect-sha256", expected_sha256]
    return shlex.join(argv)


def probe_timeout_s(params: AccuracyParams) -> float:
    return params.deadline_s + PROBE_MARGIN_S


def _tail(text: str) -> str:
    return text.strip()[-ERROR_TAIL_CHARS:]


def _run(
    cluster: Cluster, node: str, command: str, timeout: float
) -> tuple[str | None, str]:
    """(stdout, error): stdout is None when the command could not run."""
    try:
        res = cluster.run(node, command, timeout=timeout, check=False)
    except (CommandError, OSError) as exc:
        return None, f"could not run: {exc}"
    if not res.ok:
        detail = _tail(res.stderr) or _tail(res.stdout)
        return res.stdout, f"exit {res.rc}: {detail}"
    return res.stdout, ""


def _with_note(result: AccuracyResult, source: str, error: str) -> AccuracyResult:
    note = ProbeNote(source=source, error=error[:300])
    return result.model_copy(update={"notes": (*result.notes, note)})


# --------------------------------------------------------------------------
# Runner-side steps
# --------------------------------------------------------------------------


def ensure_dataset(
    cluster: Cluster,
    node: str,
    *,
    python: str,
    agent: str,
    cache_dir: str,
    urls: Sequence[str] = (GSM8K_URL,),
    sha256: str = GSM8K_SHA256,
    filename: str = GSM8K_FILENAME,
    attempts: int = FETCH_ATTEMPTS,
    attempt_timeout_s: float = FETCH_ATTEMPT_TIMEOUT_S,
) -> DatasetStatus:
    """A sha256-verified dataset at ``<cache_dir>/<filename>`` on ``node``.

    Reuses a verified copy, else downloads one; ``ok=False`` with ``error``
    when no verified copy could be installed (e.g. no internet access).
    """
    dest = dataset_path(cache_dir, filename)
    command = fetch_command(
        python,
        agent,
        dest=dest,
        urls=urls,
        sha256=sha256,
        attempts=attempts,
        attempt_timeout_s=attempt_timeout_s,
    )
    timeout = fetch_timeout_s(
        urls=len(urls), attempts=attempts, attempt_timeout_s=attempt_timeout_s
    )
    stdout, error = _run(cluster, node, command, timeout)
    doc = find_document(stdout or "", FETCH_KIND)
    if doc is None:
        status = failed_dataset(
            dest, sha256, error or f"no {FETCH_KIND} JSON document in the output"
        )
    else:
        try:
            status = parse_fetch_document(doc, path=dest, expected_sha256=sha256)
        except ValueError as exc:
            status = failed_dataset(dest, sha256, f"unparsable fetch output: {exc}")
    if not status.ok:
        logger.warning("[%s] accuracy dataset unavailable: %s", node, status.error)
    return status


def run_accuracy(
    cluster: Cluster,
    node: str,
    *,
    python: str,
    agent: str,
    url: str,
    model: str,
    dataset: str,
    out: str,
    params: AccuracyParams | None = None,
    expected_sha256: str | None = GSM8K_SHA256,
) -> AccuracyResult:
    """Run the probe against the deployment behind ``url`` (blocking).

    ``url`` is the frontend base URL as seen from ``node`` (for node0:
    ``http://127.0.0.1:<frontend_port>``), ``model`` the served model name,
    ``dataset`` the path ``ensure_dataset`` returned, ``out`` where the agent
    keeps its JSON on ``node``. Takes up to ``params.deadline_s`` +
    ``PROBE_MARGIN_S``; returns ``status="error"`` instead of raising.
    """
    params = params or AccuracyParams()
    command = probe_command(
        python,
        agent,
        url=url,
        model=model,
        dataset=dataset,
        out=out,
        params=params,
        expected_sha256=expected_sha256,
    )
    stdout, error = _run(cluster, node, command, probe_timeout_s(params))
    doc = find_document(stdout or "", PROBE_KIND)
    if doc is None and stdout is not None:
        # The agent writes ``out`` before printing: read it back.
        text, read_error = _run(cluster, node, shlex.join(["cat", out]), READ_TIMEOUT_S)
        if not read_error:
            doc = find_document(text or "", PROBE_KIND)
    if doc is None:
        reason = error or f"no {PROBE_KIND} JSON document in the probe output"
        logger.warning("[%s] accuracy probe failed: %s", node, reason)
        return error_result(reason, params=params, model=model, endpoint=url)
    try:
        result = parse_probe_document(doc)
    except ValueError as exc:  # pydantic's ValidationError is a ValueError
        logger.warning("[%s] accuracy probe output unparsable: %s", node, exc)
        return error_result(
            f"unparsable probe output: {exc}", params=params, model=model, endpoint=url
        )
    result = result.model_copy(update={"params": params})
    if error:
        result = _with_note(result, "collector", error)
    if result.status == "error":
        logger.warning("[%s] accuracy probe error: %s", node, result.reason)
    return result


# --------------------------------------------------------------------------
# Dry-run wiring
# --------------------------------------------------------------------------


def _flag(argv: Sequence[str], name: str) -> str | None:
    if name in argv:
        position = list(argv).index(name)
        if position + 1 < len(argv):
            return argv[position + 1]
    return None


def _simulated_params(argv: Sequence[str]) -> AccuracyParams:
    fields = {
        "shots": ("--shots", int),
        "questions": ("--questions", int),
        "concurrency": ("--concurrency", int),
        "max_tokens": ("--max-tokens", int),
        "request_timeout_s": ("--request-timeout-s", float),
        "deadline_s": ("--deadline-s", float),
        "retries": ("--retries", int),
    }
    values = {}
    for key, (flag, kind) in fields.items():
        raw = _flag(argv, flag)
        if raw is not None:
            try:
                values[key] = kind(raw)
            except ValueError:
                continue
    try:
        return AccuracyParams(**values)
    except ValueError:  # pydantic's ValidationError is a ValueError
        return AccuracyParams()


def simulated_stdout(command: str, *, accuracy: float = SYNTH_ACCURACY) -> str | None:
    """Synthetic agent output for ``command`` (dry-run simulator), else None."""
    if AGENT_REL not in command:
        return None
    argv = shlex.split(command)
    position = next((i for i, a in enumerate(argv) if a.endswith(AGENT_REL)), None)
    if position is None or position + 1 >= len(argv):
        return None
    mode = argv[position + 1]
    if mode == "fetch":
        dest = _flag(argv, "--dest") or ""
        sha256 = _flag(argv, "--sha256") or GSM8K_SHA256
        return json.dumps(synthetic_fetch_document(dest, sha256=sha256)) + "\n"
    if mode == "gsm8k":
        url = (_flag(argv, "--url") or "http://127.0.0.1:8000").rstrip("/")
        doc = synthetic_probe_document(
            accuracy=accuracy,
            params=_simulated_params(argv),
            model=_flag(argv, "--model") or "model",
            endpoint=f"{url}/v1/completions",
            dataset=_flag(argv, "--dataset") or "",
        )
        return json.dumps(doc) + "\n"
    return None
