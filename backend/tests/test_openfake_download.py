"""OpenFake downloads must respect the server's rate-limit retry window."""

from __future__ import annotations

import importlib.util
import io
from email.message import Message
from pathlib import Path
from urllib.error import HTTPError

import pytest


@pytest.fixture
def openfake():
    script = Path(__file__).resolve().parents[1] / "scripts" / "prepare_openfake.py"
    spec = importlib.util.spec_from_file_location("prepare_openfake", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "retry_after,server_date",
    [
        ("75", None),
        ("Sun, 01 Nov 2026 12:01:15 GMT", "Sun, 01 Nov 2026 12:00:00 GMT"),
    ],
    ids=["seconds", "http-date"],
)
def test_download_waits_for_retry_after_before_retrying(openfake, monkeypatch, retry_after, server_date):
    url = "https://datasets-server.huggingface.co/rows?dataset=example"
    headers = Message()
    headers["Retry-After"] = retry_after
    if server_date is not None:
        headers["Date"] = server_date
    sleeps = []
    calls = []

    def urlopen(request, timeout):
        calls.append(request.full_url)
        if len(calls) == 1:
            raise HTTPError(url, 429, "Too Many Requests", headers, None)
        assert sum(sleeps) >= 75, "Retry started before the server's Retry-After window elapsed"
        return io.BytesIO(b'{"rows": []}')

    monkeypatch.setattr(openfake, "urlopen", urlopen)
    monkeypatch.setattr(openfake.time, "sleep", sleeps.append)

    assert openfake.download(url) == b'{"rows": []}'
    assert calls == [url, url]


def test_successful_download_does_not_wait_for_retry(openfake, monkeypatch):
    url = "https://datasets-server.huggingface.co/rows?dataset=example"
    calls = []
    sleeps = []

    def urlopen(request, timeout):
        calls.append(request.full_url)
        return io.BytesIO(b"successful response")

    monkeypatch.setattr(openfake, "urlopen", urlopen)
    monkeypatch.setattr(openfake.time, "sleep", sleeps.append)

    assert openfake.download(url) == b"successful response"
    assert calls == [url]
    assert sleeps == []


def test_service_outage_gets_sustained_backoff_without_changing_selected_rows(openfake, monkeypatch):
    url = "https://datasets-server.huggingface.co/rows?dataset=example"
    calls, sleeps = [], []

    def urlopen(request, timeout):
        calls.append(request.full_url)
        if len(calls) <= 5:
            raise HTTPError(url, 503, "Service Unavailable", Message(), None)
        return io.BytesIO(b"recovered")

    monkeypatch.setattr(openfake, "urlopen", urlopen)
    monkeypatch.setattr(openfake.time, "sleep", sleeps.append)
    assert openfake.download(url) == b"recovered"
    assert len(calls) == 6
    assert sum(sleeps) >= 150


def link_row(module, index, expiry):
    return {
        "row_id": f"row-{index}",
        "row_idx": index,
        "upstream_split": "train",
        "model": "example",
        "prompt": f"prompt-{index}",
        "image_sha256": f"{index + 1:064x}",
        "image": {
            "src": f"https://datasets-server.huggingface.co/cached-assets/ComplexDataLab/OpenFake/--/{module.REVISION}/--/core/train/{index}/image/image.jpg?Expires={expiry}"
        },
    }


def test_one_slow_page_does_not_block_an_unrelated_page(openfake, monkeypatch):
    import json
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from urllib.parse import parse_qs, urlparse

    started, release = Event(), Event()
    monkeypatch.setattr(openfake, "pace_metadata", lambda: None)

    def download(url):
        index = int(parse_qs(urlparse(url).query)["offset"][0])
        if index == 0:
            started.set()
            assert release.wait(5)
        row = link_row(openfake, index, 9999999999)
        return json.dumps({"rows": [{"row": row}]}).encode()

    monkeypatch.setattr(openfake, "download", download)
    links = openfake.ViewerLinks()
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(links.source, link_row(openfake, 0, 0))
        try:
            assert started.wait(2)
            second = pool.submit(links.source, link_row(openfake, 100, 0))
            assert "train/100/" in second.result(timeout=2)
        finally:
            release.set()
        assert "train/0/" in first.result(timeout=2)


def test_expired_cached_links_are_refreshed_again(openfake, monkeypatch):
    import json

    now, calls = [1000], []
    monkeypatch.setattr(openfake.time, "time", lambda: now[0])
    monkeypatch.setattr(openfake, "pace_metadata", lambda: None)

    def download(url):
        calls.append(url)
        return json.dumps({"rows": [{"row": link_row(openfake, 0, now[0] + 3600)}]}).encode()

    monkeypatch.setattr(openfake, "download", download)
    links = openfake.ViewerLinks()
    row = link_row(openfake, 0, 0)
    original = links.source(row)
    assert links.source(row) == original and len(calls) == 1
    now[0] += 3601
    assert links.source(row) != original and len(calls) == 2
