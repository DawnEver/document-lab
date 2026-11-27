"""The corpus manifest's rules and its fetcher, offline: a local HTTP server stands in for the public URLs."""

from __future__ import annotations

import dataclasses
import hashlib
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, ClassVar

import pytest

from tests.corpus import sources
from tests.corpus.sources import CC_BY_4, FETCH_FAILED, SOURCES, US_PUBLIC_DOMAIN, FetchRefused, Source, fetch_into

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

BODY = b'%PDF-1.7 the pinned bytes'
SHA = hashlib.sha256(BODY).hexdigest()


class _Served(BaseHTTPRequestHandler):
    """Answers each path from :attr:`routes` (status, body); counts the requests."""

    routes: ClassVar[dict[str, tuple[int, bytes]]] = {}
    hits: ClassVar[list[str]] = []

    def do_GET(self) -> None:
        self.hits.append(self.path)
        status, body = self.routes.get(self.path, (404, b''))
        self.send_response(status)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        pass


@pytest.fixture
def server() -> Iterator[str]:
    _Served.routes, _Served.hits = {}, []
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), _Served)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{httpd.server_address[1]}'
    httpd.shutdown()
    httpd.server_close()


def _source(base: str, *, fallback: bool = True) -> Source:
    return Source(
        'served',
        f'{base}/original.pdf',
        'a test server',
        f'{base}/snapshot.pdf' if fallback else None,
        SHA,
        len(BODY),
        *CC_BY_4,
        'nobody',
        'nothing',
    )


def test_the_pinned_bytes_are_cached_by_sha256_and_reused(server: str, tmp_path: Path):
    _Served.routes['/original.pdf'] = (200, BODY)
    path = fetch_into(_source(server), tmp_path)
    assert path == tmp_path / f'{SHA}.pdf'
    assert path.read_bytes() == BODY
    assert fetch_into(_source(server), tmp_path) == path
    assert _Served.hits == ['/original.pdf']  # the second fetch asked nothing
    assert [p.name for p in tmp_path.iterdir()] == [path.name]  # no partial download left beside it


def test_a_cached_copy_with_other_bytes_is_fetched_again(server: str, tmp_path: Path):
    _Served.routes['/original.pdf'] = (200, BODY)
    (tmp_path / f'{SHA}.pdf').write_bytes(b'damaged')
    assert fetch_into(_source(server), tmp_path).read_bytes() == BODY


def test_the_fallback_is_taken_only_when_the_original_fails(server: str, tmp_path: Path):
    _Served.routes['/snapshot.pdf'] = (200, BODY)
    assert fetch_into(_source(server), tmp_path).read_bytes() == BODY
    assert _Served.hits == ['/original.pdf', '/snapshot.pdf']


def test_other_bytes_are_refused_naming_the_url(server: str, tmp_path: Path):
    _Served.routes['/original.pdf'] = (200, b'%PDF-1.7 other bytes')
    with pytest.raises(FetchRefused, match=r'not a reader regression.*original\.pdf served 20 bytes'):
        fetch_into(_source(server, fallback=False), tmp_path)
    assert not tmp_path.exists() or not any(tmp_path.iterdir())


def test_every_url_failing_is_refused_naming_each(server: str, tmp_path: Path):
    _Served.routes['/snapshot.pdf'] = (200, b'%PDF-1.7 other bytes')
    with pytest.raises(FetchRefused) as refused:
        fetch_into(_source(server), tmp_path)
    assert 'original.pdf failed: HTTP 404' in str(refused.value)
    assert 'snapshot.pdf served 20 bytes' in str(refused.value)


def test_a_transient_failure_is_retried(server: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _Served.routes['/original.pdf'] = (503, b'')

    def recover(_seconds: float) -> None:
        _Served.routes['/original.pdf'] = (200, BODY)

    monkeypatch.setattr(sources.time, 'sleep', recover)
    assert fetch_into(_source(server, fallback=False), tmp_path).read_bytes() == BODY
    assert _Served.hits == ['/original.pdf', '/original.pdf']


def test_the_fetch_command_exits_with_its_own_status(server: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sources, 'SOURCES', {'served': _source(server, fallback=False)})
    monkeypatch.setattr(sources, 'CACHE', tmp_path)
    assert sources.main() == FETCH_FAILED
    _Served.routes['/original.pdf'] = (200, BODY)
    assert sources.main() == 0


# --- the manifest's rules (D29, D30, and the coordinator's: immutable URLs, one verified fallback each) -------------


def zero_risk(source: Source) -> bool:
    """CC BY 4.0 (the license or its deed in a language) or a US Government work: nothing else (D30)."""
    cc_by = source.license.startswith(CC_BY_4[0]) and source.license_url.startswith(CC_BY_4[1])
    return cc_by or (source.license, source.license_url) == US_PUBLIC_DOMAIN


@pytest.mark.parametrize('source', SOURCES.values(), ids=lambda s: s.name)
def test_a_source_carries_no_licensing_risk(source: Source):
    assert zero_risk(source)
    assert source.attribution


IMMUTABLE = {
    'arxiv.org': re.compile(r'^https://arxiv\.org/pdf/\d{4}\.\d{4,5}v\d+$'),
    'www.irs.gov': re.compile(r'^https://www\.irs\.gov/pub/irs-prior/f[a-z0-9]+--\d{4}\.pdf$'),
    'pubs.usgs.gov': re.compile(r'^https://pubs\.usgs\.gov/fs/\d{4}/\d{4}/fs\d{8}\.pdf$'),
    'ntrs.nasa.gov': re.compile(r'^https://ntrs\.nasa\.gov/api/citations/(\d+)/downloads/\1\.pdf$'),
    'www.jstage.jst.go.jp': re.compile(r'^https://www\.jstage\.jst\.go\.jp/static/files/ja/pub_survey\d{4}_\w+\.pdf$'),
}


@pytest.mark.parametrize('source', SOURCES.values(), ids=lambda s: s.name)
def test_a_source_url_is_immutable_and_says_why(source: Source):
    host = source.url.split('/')[2]
    assert IMMUTABLE[host].match(source.url), f'{source.url} is not the immutable form of {host}'
    assert source.permanence


@pytest.mark.parametrize('source', SOURCES.values(), ids=lambda s: s.name)
def test_a_fallback_is_a_pinned_wayback_snapshot(source: Source):
    if source.fallback is None:
        assert source.name == 'cjk-zh-vertical-arxiv-2301.01621v3'  # the one source no snapshot holds
        return
    assert re.match(r'^https://web\.archive\.org/web/\d{14}id_/https://', source.fallback)


def test_the_immutable_url_rule_can_say_no():
    current_year = dataclasses.replace(SOURCES['form-irs-fw9-2024'], url='https://www.irs.gov/pub/irs-pdf/fw9.pdf')
    unversioned = dataclasses.replace(SOURCES['latex-type3-arxiv-2204.08094v1'], url='https://arxiv.org/pdf/2204.08094')
    for source in (current_year, unversioned):
        assert not IMMUTABLE[source.url.split('/')[2]].match(source.url)


def test_the_license_rule_can_say_no():
    share_alike = dataclasses.replace(
        SOURCES['latex-type3-arxiv-2204.08094v1'],
        license='CC BY-SA 4.0',
        license_url='https://creativecommons.org/licenses/by-sa/4.0/',
    )
    assert not zero_risk(share_alike)


def _racing(path: Path, held: bytes, monkeypatch: pytest.MonkeyPatch) -> None:
    """A sibling worker renames its copy into ``path`` first and holds it open: Windows refuses our rename over it."""
    replace = type(path).replace

    def refused(self: Path, target: Path) -> Path:
        if target == path:
            path.write_bytes(held)
            raise PermissionError(13, 'Access is denied', str(target))
        return replace(self, target)

    monkeypatch.setattr(type(path), 'replace', refused)


def test_a_sibling_that_renamed_the_pinned_bytes_first_wins(server: str, tmp_path: Path, monkeypatch):
    _Served.routes['/original.pdf'] = (200, BODY)
    path = tmp_path / f'{SHA}.pdf'
    _racing(path, BODY, monkeypatch)
    assert fetch_into(_source(server), tmp_path).read_bytes() == BODY
    assert [p.name for p in tmp_path.iterdir()] == [path.name]  # our partial download is gone


def test_a_refused_rename_over_other_bytes_is_still_raised(server: str, tmp_path: Path, monkeypatch):
    _Served.routes['/original.pdf'] = (200, BODY)
    _racing(tmp_path / f'{SHA}.pdf', b'damaged', monkeypatch)
    with pytest.raises(PermissionError):
        fetch_into(_source(server), tmp_path)
