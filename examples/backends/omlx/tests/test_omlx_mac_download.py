# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""mac/download_model.py against an in-memory Hugging Face repo."""

import importlib.util
import json
import re
import time
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import pytest

_PATH = Path(__file__).resolve().parents[1] / "mac" / "download_model.py"
_SPEC = importlib.util.spec_from_file_location("omlx_mac_download_model", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
download_model = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(download_model)

pytestmark = [pytest.mark.post_merge, pytest.mark.gpu_0, pytest.mark.unit]

REPO = "Org/Model-oQ4e"
NAME = "Model-oQ4e"
FILES = {
    "config.json": b'{"model_type": "qwen4_exp"}',
    "tokenizer.json": b"{}",
    "chat_template.jinja": b"{{ messages }}",
    "model-00001-of-00002.safetensors": b"w" * 64,
    "model-00002-of-00002.safetensors": b"x" * 32,
}
Sibling = namedtuple("Sibling", "rfilename size")


class HttpError(Exception):
    def __init__(self, status: int):
        super().__init__(f"HTTP {status}")
        self.response = SimpleNamespace(status_code=status)


class FakeHub:
    """``HfApi.model_info`` and ``snapshot_download`` over ``files``."""

    def __init__(self, files=FILES, sha="a" * 40):
        self.files = dict(files)
        self.sha = sha
        self.info_errors: list[Exception] = []
        self.download_errors: list[Exception] = []
        self.downloads: list[tuple[Path, list[str] | None]] = []

    def model_info(self, repo, revision, files_metadata):
        assert repo == REPO and files_metadata
        if self.info_errors:
            raise self.info_errors.pop(0)
        siblings = [Sibling(name, len(data)) for name, data in self.files.items()]
        return SimpleNamespace(sha=self.sha, siblings=siblings)

    def snapshot_download(self, repo, *, revision, local_dir, allow_patterns, **_):
        assert repo == REPO and revision == self.sha
        self.downloads.append((Path(local_dir), allow_patterns))
        for i, (name, data) in enumerate(self.files.items()):
            if allow_patterns and not any(Path(name).match(p) for p in allow_patterns):
                continue
            # A failure leaves the first file behind, like an interrupted download.
            if i == 1 and self.download_errors:
                raise self.download_errors.pop(0)
            (Path(local_dir) / name).write_bytes(data)


@pytest.fixture
def hub():
    return FakeHub()


def fetch(hub, model_dir, sleeps=None, **kwargs):
    return download_model.fetch(
        REPO,
        model_dir,
        api=hub,
        download=hub.snapshot_download,
        sleep=(sleeps if sleeps is not None else []).append,
        **kwargs,
    )


def status(directory: Path) -> dict:
    return json.loads((directory / download_model.STATUS_FILE).read_text())


def test_download_is_staged_out_of_sight_then_moved_into_place(hub, tmp_path):
    target = fetch(hub, tmp_path)
    assert target == tmp_path / NAME
    # oMLX's model discovery skips the hidden staging directory.
    assert hub.downloads == [(tmp_path / ".downloading" / NAME, None)]
    assert not (tmp_path / ".downloading").exists()
    assert {p.name for p in target.iterdir()} >= set(FILES)
    assert status(target)["state"] == "complete"
    assert status(target)["sha"] == hub.sha


def test_rerun_for_the_same_revision_downloads_nothing(hub, tmp_path):
    fetch(hub, tmp_path)
    fetch(hub, tmp_path)
    assert len(hub.downloads) == 1


def test_network_errors_are_retried_and_resume(hub, tmp_path):
    hub.info_errors = [ConnectionError("reset")]
    hub.download_errors = [TimeoutError("stalled")]
    sleeps: list[float] = []
    target = fetch(hub, tmp_path, sleeps)
    assert sleeps == [15.0, 15.0]
    assert len(hub.downloads) == 2
    assert status(target)["state"] == "complete"


@pytest.mark.parametrize("code", [401, 403, 404])
def test_client_errors_are_not_retried(hub, tmp_path, code):
    hub.info_errors = [HttpError(code)]
    sleeps: list[float] = []
    with pytest.raises(HttpError):
        fetch(hub, tmp_path, sleeps)
    assert sleeps == []


@pytest.mark.parametrize("code", [408, 429, 500, 503])
def test_server_errors_and_throttling_are_retried(hub, tmp_path, code):
    hub.info_errors = [HttpError(code)]
    fetch(hub, tmp_path)
    assert status(tmp_path / NAME)["state"] == "complete"


def test_retries_give_up_after_the_last_attempt(hub, tmp_path):
    hub.download_errors = [ConnectionError("down")] * 3
    with pytest.raises(ConnectionError):
        fetch(hub, tmp_path, attempts=3)
    assert len(hub.downloads) == 3
    assert not (tmp_path / NAME).exists()


def test_a_wrong_size_keeps_the_model_out_of_place(hub, tmp_path, monkeypatch):
    real = hub.snapshot_download

    def truncating(*args, **kwargs):
        real(*args, **kwargs)
        (kwargs["local_dir"] / "model-00002-of-00002.safetensors").write_bytes(b"x")

    monkeypatch.setattr(hub, "snapshot_download", truncating)
    with pytest.raises(download_model.DownloadError, match="00002-of-00002"):
        fetch(hub, tmp_path)
    assert not (tmp_path / NAME).exists()
    assert status(tmp_path / ".downloading" / NAME)["state"] == "failed"


def test_too_little_disk_space_fails_before_downloading(hub, tmp_path, monkeypatch):
    monkeypatch.setattr(
        download_model.shutil, "disk_usage", lambda _: SimpleNamespace(free=10)
    )
    with pytest.raises(download_model.DownloadError, match="GiB is free"):
        fetch(hub, tmp_path)
    assert hub.downloads == []


def test_a_newer_revision_is_fetched_only_with_update(hub, tmp_path):
    fetch(hub, tmp_path)
    hub.sha = "b" * 40
    hub.files["config.json"] = b'{"model_type": "qwen4_exp", "v": 2}'

    fetch(hub, tmp_path)
    assert len(hub.downloads) == 1
    assert status(tmp_path / NAME)["sha"] == "a" * 40

    fetch(hub, tmp_path, update=True)
    assert hub.downloads[-1] == (tmp_path / ".downloading" / NAME, None)
    assert status(tmp_path / NAME)["sha"] == "b" * 40
    assert b'"v": 2' in (tmp_path / NAME / "config.json").read_bytes()


def test_a_finished_download_survives_an_unreachable_hub(hub, tmp_path):
    fetch(hub, tmp_path)
    hub.info_errors = [ConnectionError("offline")] * 2
    assert fetch(hub, tmp_path, attempts=2) == tmp_path / NAME
    assert len(hub.downloads) == 1


def test_a_complete_copy_from_elsewhere_is_adopted(hub, tmp_path):
    target = tmp_path / NAME
    target.mkdir()
    for name, data in FILES.items():
        (target / name).write_bytes(data)
    fetch(hub, tmp_path)
    assert hub.downloads == []
    assert status(target)["state"] == "complete"


def test_an_incomplete_model_directory_is_completed_only_with_update(hub, tmp_path):
    target = tmp_path / NAME
    target.mkdir()
    (target / "config.json").write_bytes(FILES["config.json"])
    with pytest.raises(download_model.DownloadError, match="--update"):
        fetch(hub, tmp_path)
    assert hub.downloads == []

    fetch(hub, tmp_path, update=True)
    assert status(target)["state"] == "complete"
    assert not (tmp_path / ".downloading" / NAME).exists()


def test_metadata_only_skips_weights_and_staging(hub, tmp_path):
    target = fetch(hub, tmp_path, metadata_only=True)
    assert hub.downloads == [(target, download_model.METADATA_PATTERNS)]
    assert sorted(p.name for p in target.iterdir() if not p.name.startswith(".")) == [
        "chat_template.jinja",
        "config.json",
        "tokenizer.json",
    ]
    assert status(target)["files"] == 3


class FakeWhoami:
    def __init__(self, result):
        self.result = result

    def whoami(self, token):
        if isinstance(self.result, Exception):
            raise self.result
        return {"name": self.result}


def test_downloads_without_a_token_are_anonymous():
    assert download_model.describe_token(FakeWhoami("alice"), None) == "anonymous"


def test_a_token_is_checked_with_the_hub():
    api = FakeWhoami("alice")
    assert download_model.describe_token(api, "hf_x") == "signed in as alice"


def test_a_rejected_token_stops_before_downloading():
    with pytest.raises(download_model.DownloadError, match="--login"):
        download_model.describe_token(FakeWhoami(HttpError(401)), "hf_x")


def test_an_unreachable_hub_leaves_the_token_unchecked():
    api = FakeWhoami(ConnectionError("offline"))
    assert "not checked" in download_model.describe_token(api, "hf_x")


def test_progress_line_reports_rate_and_time_left():
    gib, mib = download_model.GIB, download_model.MIB
    line = download_model.progress_line(25 * gib, 100 * gib, 50 * mib, 5400.0)
    assert line == "25.0 / 100.0 GiB (25%), 50.0 MiB/s, about 1 h 30 min left"
    line = download_model.progress_line(0, 100 * gib, 0.0, None)
    assert line == "0.0 / 100.0 GiB (0%), 0.0 MiB/s"
    # Partial files and metadata never push it past 100%.
    line = download_model.progress_line(101 * gib, 100 * gib, 1.0, 0.0)
    assert line.startswith("100.0 / 100.0 GiB (100%)")


def test_progress_estimates_the_time_left_only_after_enough_bytes(capsys):
    mib = download_model.MIB
    reads = iter([0, 10 * mib, 100 * mib] + [200 * mib] * 1000)
    with download_model.Progress(lambda: next(reads), 400 * mib, 0.01):
        time.sleep(0.5)
    lines = capsys.readouterr().out.splitlines()
    assert re.match(r"\d\d:\d\d:\d\d 0\.0 / 0\.4 GiB \(2%\)", lines[0])
    assert "left" not in lines[0]
    assert "left" in lines[1]


def test_counting_bars_record_progress_without_drawing(capsys):
    pytest.importorskip("tqdm")
    bars: list = []
    counting = download_model.counting_tqdm(bars)
    with counting(total=100, unit="B") as bar:
        bar.update(40)
    # thread_map wraps its results this way.
    assert list(counting(iter([1, 2]), total=2)) == [1, 2]
    assert [(b.unit, b.n) for b in bars] == [("B", 40), ("it", 2)]
    assert capsys.readouterr() == ("", "")


def test_progress_lines_follow_the_byte_bars(hub, tmp_path, capsys, monkeypatch):
    pytest.importorskip("tqdm")
    real = hub.snapshot_download
    received = {}

    def slow(*args, tqdm_class, **kwargs):
        received["tqdm_class"] = tqdm_class
        with tqdm_class(total=96, unit="B") as bar:
            for _ in range(4):
                bar.update(24)
                time.sleep(0.05)
        real(*args, **kwargs)

    monkeypatch.setattr(hub, "snapshot_download", slow)
    fetch(hub, tmp_path, progress_interval=0.02)
    lines = capsys.readouterr().out.splitlines()
    assert "tqdm_class" in received
    assert any(re.match(r"\d\d:\d\d:\d\d .* MiB/s", line) for line in lines)


def test_the_last_line_reports_size_time_and_rate(tmp_path, capsys):
    hub = FakeHub({"model.safetensors": b"w" * (2 * download_model.MIB)})
    fetch(hub, tmp_path)
    assert re.search(
        r"Downloaded .* \(2 MiB in \d+ s, [\d.]+ MiB/s\)$", capsys.readouterr().out
    )
