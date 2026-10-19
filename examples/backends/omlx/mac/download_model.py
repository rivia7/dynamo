# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download a Hugging Face model into oMLX's model directory.

On a Mac, download_model.sh runs this with oMLX's own interpreter, which ships
``huggingface_hub``:

    python download_model.py Jundot/Qwen3.8-Flash-Next-oQ4e-mtp --model-dir ~/.omlx/models

Files land in ``<model-dir>/.downloading/<name>`` first. oMLX's model discovery
skips hidden directories, and the finished model moves into place with one
rename, so oMLX never loads a partial model. Network failures are retried with
backoff, and every retry resumes the files already on disk. A status file
records the finished revision, so a rerun returns at once; ``--update`` fetches
a newer revision, downloading only the files that changed.

A Hugging Face token lifts the Hub's limits on anonymous downloads. It comes
from HF_TOKEN or from the token file that ``--login`` writes, which oMLX reads
as well, and it is checked with the Hub before a download starts. Without a
terminal, for example in the log of ``download_model.sh --background``, a
progress line every 30 seconds takes the place of progress bars.

On Linux, ``--metadata-only`` fetches what the Dynamo proxy registers the model
from: the config, tokenizer and chat template, without weights.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import shutil
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

STATUS_FILE = ".dynamo-omlx-download.json"
STAGING_DIR = ".downloading"
METADATA_PATTERNS = ["*.json", "*.jinja", "*.txt", "*.model", "*.tiktoken"]
GIB = 1024**3
MIB = 1024**2
MAX_RETRY_DELAY_S = 300.0
PROGRESS_INTERVAL_S = 30.0

T = TypeVar("T")


class DownloadError(Exception):
    """A download that cannot finish; the message says why."""


def _read_status(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _write_status(path: Path, **status: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(status, indent=1) + "\n")
    tmp.replace(path)


def _size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def _mismatched(directory: Path, files: dict[str, int | None]) -> list[str]:
    return sorted(
        file
        for file, size in files.items()
        if _size(directory / file) is None
        or (size is not None and _size(directory / file) != size)
    )


def _complete_bytes(directory: Path, files: dict[str, int | None]) -> int:
    """Bytes of the listed files that are already on disk in full."""
    return sum(
        size
        for file, size in files.items()
        if size is not None and _size(directory / file) == size
    )


def _amount(size: float) -> str:
    return f"{size / GIB:.1f} GiB" if size >= GIB else f"{size / MIB:.0f} MiB"


def _duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f} s"
    minutes = round(seconds / 60)
    return f"{minutes} min" if minutes < 60 else f"{minutes // 60} h {minutes % 60} min"


def progress_line(done: int, total: int, rate: float, left: float | None) -> str:
    """Bytes so far, the current rate and, once known, the time left."""
    done = min(done, total)
    line = (
        f"{done / GIB:.1f} / {total / GIB:.1f} GiB "
        f"({100 * done / max(total, 1):.0f}%), {max(rate, 0.0) / MIB:.1f} MiB/s"
    )
    if left is not None:
        line += f", about {_duration(left)} left"
    return line


class Progress:
    """Prints a progress line every ``interval`` seconds; ``read`` gives the bytes done."""

    # Below this much progress the time left would be a wild guess.
    ESTIMATE_AFTER_BYTES = 64 * MIB

    def __init__(self, read: Callable[[], int], total: int, interval: float):
        self._read = read
        self._total = total
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> Progress:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        start = last = self._read()
        start_time = last_time = time.monotonic()
        while not self._stop.wait(self._interval):
            done, now = self._read(), time.monotonic()
            left = None
            if done - start >= self.ESTIMATE_AFTER_BYTES:
                left = max(0, self._total - done) * (now - start_time) / (done - start)
            rate = (done - last) / (now - last_time)
            line = progress_line(done, self._total, rate, left)
            print(f"{time.strftime('%H:%M:%S')} {line}", flush=True)
            last, last_time = done, now


class _Discard:
    """A file for tqdm that keeps nothing."""

    def write(self, _: str) -> None:
        pass

    def flush(self) -> None:
        pass


def counting_tqdm(bars: list[Any]) -> type:
    """A tqdm class for snapshot_download that collects its bars instead of drawing them.

    Disk usage cannot stand in for these byte counts: Xet storage writes a file
    only once all of it has arrived. huggingface_hub 2 counts bytes as they
    arrive; older releases count Xet files only as they complete.
    """
    from tqdm.auto import tqdm

    class CountingTqdm(tqdm):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["file"] = _Discard()
            super().__init__(*args, **kwargs)
            bars.append(self)

    return CountingTqdm


def is_permanent(error: Exception) -> bool:
    """Whether retrying cannot help: a missing or gated repo, a bad revision, auth."""
    status = getattr(getattr(error, "response", None), "status_code", None)
    return isinstance(status, int) and 400 <= status < 500 and status not in (408, 429)


def describe_token(api: Any, token: str | None) -> str:
    """Whose token the downloads use, checked before a long download starts."""
    if not token:
        return "anonymous"
    try:
        return f"signed in as {api.whoami(token=token)['name']}"
    except Exception as e:  # noqa: BLE001
        if is_permanent(e):
            raise DownloadError(
                "the Hub rejected the Hugging Face token. Create a new one at "
                "https://huggingface.co/settings/tokens and save it with --login "
                "(or fix HF_TOKEN in the settings file)"
            ) from e
        return "token not checked: the Hub is unreachable"


def retry(
    what: str,
    attempts: int,
    call: Callable[[], T],
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    for attempt in range(1, attempts + 1):
        try:
            return call()
        # Anything without a 4xx status is treated as a network failure.
        except Exception as e:  # noqa: BLE001
            if is_permanent(e) or attempt == attempts:
                raise
            delay = min(MAX_RETRY_DELAY_S, 15.0 * attempt)
            print(
                f"{what} failed ({e!r}); retry {attempt}/{attempts - 1} in {delay:.0f}s",
                flush=True,
            )
            sleep(delay)
    raise AssertionError("unreachable")


def fetch(
    repo: str,
    model_dir: Path,
    *,
    name: str | None = None,
    revision: str = "main",
    metadata_only: bool = False,
    update: bool = False,
    max_workers: int = 8,
    attempts: int = 30,
    progress_interval: float = 0.0,
    api: Any = None,
    download: Callable[..., Any] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """Download ``repo`` into ``model_dir/<name>`` and return that directory.

    A positive ``progress_interval`` prints a progress line that often instead
    of letting huggingface_hub draw progress bars.
    """
    if api is None or download is None:
        from huggingface_hub import HfApi, snapshot_download

        api = api or HfApi()
        download = download or snapshot_download

    target = model_dir.expanduser() / (name or repo.rsplit("/", 1)[-1])
    staging = target.parent / STAGING_DIR / target.name
    previous = _read_status(target / STATUS_FILE)
    complete = previous.get("state") == "complete" and previous.get("repo") == repo
    try:
        info = retry(
            "Reading the model listing",
            attempts,
            lambda: api.model_info(repo, revision=revision, files_metadata=True),
            sleep,
        )
    # Offline or not, a finished download stays usable.
    except Exception as e:  # noqa: BLE001
        if complete:
            print(f"Cannot reach the Hub ({e!r}); keeping {target}")
            return target
        raise

    patterns = METADATA_PATTERNS if metadata_only else None
    files = {
        sibling.rfilename: sibling.size
        for sibling in info.siblings or []
        if patterns is None or any(Path(sibling.rfilename).match(p) for p in patterns)
    }
    total = sum(size or 0 for size in files.values())
    status = {"repo": repo, "sha": info.sha, "files": len(files), "total_bytes": total}

    if complete and previous.get("sha") == info.sha:
        print(f"{repo}@{info.sha[:12]} is already in {target}")
        return target
    if complete and not update:
        print(
            f"{target} holds {repo}@{str(previous.get('sha'))[:12]}; the Hub has "
            f"{info.sha[:12]}. Pass --update to fetch the changed files."
        )
        return target
    if not complete and target.is_dir() and not _mismatched(target, files):
        # Fetched some other way, for example by oMLX's own downloader.
        _write_status(target / STATUS_FILE, state="complete", **status)
        print(f"Found a complete copy of {repo}@{info.sha[:12]} in {target}")
        return target

    # Metadata is small and nothing loads it half-written: no staging.
    work = target if metadata_only else staging
    if work != target and target.exists():
        if not (complete or update):
            raise DownloadError(
                f"{target} exists but is not a complete download of {repo}; "
                "move it away, or pass --update to complete it"
            )
        if staging.exists():
            raise DownloadError(f"both {target} and {staging} exist; remove one")
        # The update happens out of oMLX's sight; unchanged files are kept.
        staging.parent.mkdir(parents=True, exist_ok=True)
        target.rename(staging)

    work.mkdir(parents=True, exist_ok=True)
    missing = total - sum(_size(work / file) or 0 for file in files)
    free = shutil.disk_usage(work).free
    if missing > free:
        raise DownloadError(
            f"{repo} needs {missing / GIB:.1f} GiB more but only "
            f"{free / GIB:.1f} GiB is free on {work}"
        )

    print(
        f"Downloading {repo}@{info.sha[:12]}: {len(files)} files, "
        f"{total / GIB:.1f} GiB, into {work}",
        flush=True,
    )
    _write_status(
        work / STATUS_FILE, state="downloading", started=time.time(), **status
    )
    had = base = _complete_bytes(work, files)
    bars: list[Any] = []
    options: dict[str, Any] = {}

    def attempt() -> None:
        nonlocal base
        # A retry skips the files finished so far, and its bars start at zero.
        bars.clear()
        base = _complete_bytes(work, files)
        download(
            repo,
            revision=info.sha,
            local_dir=work,
            allow_patterns=patterns,
            max_workers=max_workers,
            **options,
        )

    progress: contextlib.AbstractContextManager[Any] = contextlib.nullcontext()
    if progress_interval > 0:
        options["tqdm_class"] = counting_tqdm(bars)
        progress = Progress(
            lambda: max(
                base + max((bar.n for bar in bars if bar.unit == "B"), default=0),
                _complete_bytes(work, files),
            ),
            total,
            progress_interval,
        )
    started = time.monotonic()
    with progress:
        retry("Downloading", attempts, attempt, sleep)
    elapsed = time.monotonic() - started

    wrong = _mismatched(work, files)
    if wrong:
        _write_status(work / STATUS_FILE, state="failed", **status)
        raise DownloadError(f"missing or wrong size after download: {', '.join(wrong)}")
    _write_status(work / STATUS_FILE, state="complete", finished=time.time(), **status)
    if work != target:
        work.rename(target)
        try:
            staging.parent.rmdir()
        # Another model is still downloading there.
        except OSError:
            pass
    took = ""
    if total - had >= MIB and elapsed > 0:
        took = (
            f" ({_amount(total - had)} in {_duration(elapsed)}, "
            f"{(total - had) / elapsed / MIB:.1f} MiB/s)"
        )
    print(f"Downloaded {repo}@{info.sha[:12]} into {target}{took}")
    return target


def login() -> None:
    """Ask for a token, check it with the Hub, and save it for huggingface_hub."""
    from getpass import getpass

    from huggingface_hub import login as hub_login

    print(
        "Paste a Hugging Face access token with read access. Create one at "
        "https://huggingface.co/settings/tokens",
        flush=True,
    )
    token = getpass("Token (input stays hidden): ").strip()
    if not token:
        raise DownloadError("no token entered")
    # Saved to huggingface_hub's token file, where oMLX finds it as well.
    hub_login(token=token, add_to_git_credential=False)


def _xet_mode() -> str:
    def on(name: str) -> bool:
        return os.environ.get(name, "").lower() in ("1", "true", "yes", "on")

    if on("HF_HUB_DISABLE_XET") or importlib.util.find_spec("hf_xet") is None:
        return "Xet off"
    return "Xet high performance" if on("HF_XET_HIGH_PERFORMANCE") else "Xet"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "repo", nargs="?", help="Hugging Face repo id, for example Org/Name"
    )
    parser.add_argument("--model-dir", type=Path, help="oMLX --model-dir")
    parser.add_argument(
        "--name",
        help="model directory name, which is the oMLX model id (default: repo name)",
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="fetch only the config, tokenizer and chat template files",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="replace a finished download when the Hub has a newer revision",
    )
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument(
        "--attempts", type=int, default=30, help="tries per Hub call (default: 30)"
    )
    parser.add_argument(
        "--login",
        action="store_true",
        help="save a Hugging Face access token for downloads, then exit",
    )
    args = parser.parse_args()
    if not args.login and not (args.repo and args.model_dir):
        parser.error("give a repo and --model-dir, or --login")
    try:
        if args.login:
            login()
            return 0
        from huggingface_hub import HfApi, constants, get_token, snapshot_download

        api = HfApi()
        token = get_token()
        print(
            f"Hub: {constants.ENDPOINT}, {describe_token(api, token)}, {_xet_mode()}",
            flush=True,
        )
        if not token:
            print(
                "No Hugging Face token: the Hub limits anonymous downloads. Save "
                "one with --login, or set HF_TOKEN.",
                flush=True,
            )
        fetch(
            args.repo,
            args.model_dir,
            name=args.name,
            revision=args.revision,
            metadata_only=args.metadata_only,
            update=args.update,
            max_workers=args.max_workers,
            attempts=args.attempts,
            # Progress bars need a terminal; a log gets progress lines.
            progress_interval=0.0 if sys.stdout.isatty() else PROGRESS_INTERVAL_S,
            api=api,
            download=snapshot_download,
        )
    except DownloadError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted; rerun to resume.", file=sys.stderr)
        return 130
    # Hub and network errors that retrying did not fix.
    except Exception as e:  # noqa: BLE001
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
