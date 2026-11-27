"""The fonts fetched on first use (``document_lab.fonts``), offline: a local HTTP server stands in for their URLs."""

from __future__ import annotations

import dataclasses
import hashlib
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pytest

from document_lab import _pdf, fonts
from document_lab.fonts import FONTS, OFL, Font, Fonts, FontUnavailable, fetch
from document_lab.ir import BBox
from document_lab.pdf import Unsupported, read_raster
from tests.rendering import page_pdf

if TYPE_CHECKING:
    from collections.abc import Iterator

BODY = b'OTTO the pinned program'
SHA = hashlib.sha256(BODY).hexdigest()
#: Bytes no font reader takes for a program.
NO_FONT = b'%PDF-1.7 not a font program'
#: The reader's substitutes, served as fallbacks: Symbol's holds U+22C5 (Wingdings 2's code 0x95), the sans not.
SUBSTITUTES = Path(__file__).resolve().parents[2] / 'crates' / 'pdf' / 'src' / 'fonts'
SYMBOL = (SUBSTITUTES / 'FoxitSymbol.cff').read_bytes()
SANS = (SUBSTITUTES / 'FoxitSans.cff').read_bytes()


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
def server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """The server's base URL; the cache a fresh directory, downloads allowed, nothing read before -- and every font
    pinned to the server, which serves none until a test says (never the internet)."""
    _Served.routes, _Served.hits = {}, []
    kept = fonts.used()
    fonts.use(Fonts(tmp_path / 'cache'))
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), _Served)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{httpd.server_address[1]}'
    for name, font in list(FONTS.items()):
        monkeypatch.setitem(FONTS, name, dataclasses.replace(font, urls=(f'{base}/unserved/{name}',)))
    yield base
    httpd.shutdown()
    httpd.server_close()
    fonts.use(kept)


def _pin(monkeypatch: pytest.MonkeyPatch, name: str, urls: tuple, body: bytes = BODY) -> Font:
    font = Font(name, urls, hashlib.sha256(body).hexdigest(), len(body), OFL)
    monkeypatch.setitem(FONTS, name, font)
    return font


def test_the_pinned_bytes_are_cached_by_sha256_and_reused(server, monkeypatch, caplog):
    _Served.routes['/a.otf'] = (200, BODY)
    _pin(monkeypatch, 'served', (f'{server}/a.otf',))
    path = fetch('served')
    assert path == fonts.used().cache / f'{SHA}.otf'
    assert path.read_bytes() == BODY
    assert fetch('served') == path
    assert _Served.hits == ['/a.otf']  # the second fetch asked nothing
    assert [p.name for p in path.parent.iterdir()] == [path.name]  # no partial download left beside it
    assert f'downloading the font served (SIL Open Font License 1.1) from {server}/a.otf' in caplog.text


def test_a_failing_url_gives_way_to_the_next(server, monkeypatch):
    _Served.routes['/mirror.otf'] = (200, BODY)
    _pin(monkeypatch, 'served', (f'{server}/gone.otf', f'{server}/mirror.otf'))
    assert fetch('served').read_bytes() == BODY
    assert _Served.hits == ['/gone.otf', '/mirror.otf']


def test_other_bytes_are_refused_naming_each_url_and_nothing_is_kept(server, monkeypatch):
    _Served.routes['/a.otf'] = (200, b'OTTO other bytes')
    _pin(monkeypatch, 'served', (f'{server}/a.otf', f'{server}/b.otf'))
    with pytest.raises(FontUnavailable) as refused:
        fetch('served')
    said = str(refused.value)
    assert said.startswith(f'served cannot be had: it is pinned to {len(BODY)} bytes, sha256 {SHA}')
    assert f'{server}/a.otf served 16 bytes' in said
    assert f'{server}/b.otf: failed: HTTP 404' in said
    assert not fonts.used().cache.exists() or not any(fonts.used().cache.iterdir())


def test_a_cached_copy_with_other_bytes_is_fetched_again(server, monkeypatch):
    _Served.routes['/a.otf'] = (200, BODY)
    _pin(monkeypatch, 'served', (f'{server}/a.otf',))
    fonts.used().cache.mkdir(parents=True)
    (fonts.used().cache / f'{SHA}.otf').write_bytes(b'damaged')
    assert fetch('served').read_bytes() == BODY


def test_offline_nothing_is_downloaded(server, monkeypatch):
    _Served.routes['/a.otf'] = (200, BODY)
    _pin(monkeypatch, 'served', (f'{server}/a.otf',))
    cache = fonts.used().cache
    fonts.use(Fonts(cache, download=False))
    with pytest.raises(FontUnavailable, match=r'downloading is off \(DOCUMENT_LAB_OFFLINE=1\)'):
        fetch('served')
    assert _Served.hits == []
    fonts.use(Fonts(cache))
    fetch('served')
    fonts.use(Fonts(cache, download=False))
    assert fetch('served').read_bytes() == BODY  # a font the cache holds needs no network


def test_the_command_lines_environment_names_the_cache_and_forbids_the_network():
    default = Fonts.of({})
    assert ('document-lab' in default.cache.parts, default.cache.name, default.download) == (True, 'fonts', True)
    assert not default.cache.is_relative_to(Path(__file__).resolve().parents[2])  # the user's, not the checkout's
    assert Fonts.of({'DOCUMENT_LAB_CACHE': '/somewhere'}).cache == Path('/somewhere/fonts')
    assert not Fonts.of({'DOCUMENT_LAB_OFFLINE': '1'}).download
    assert Fonts.of({'DOCUMENT_LAB_OFFLINE': '0'}).download
    stated = Fonts.of({'LOCALAPPDATA': '/local', 'XDG_CACHE_HOME': '/xdg'}).cache
    if sys.platform == 'win32':
        assert stated == Path('/local/document-lab/cache/fonts')
    elif sys.platform != 'darwin':
        assert stated == Path('/xdg/document-lab/fonts')


def _wingdings(tmp_path: Path) -> Path:
    """A page showing Wingdings 2's code 0x95 (U+22C5) large, the font not embedded."""
    font = (
        '<< /Type /Font /Subtype /TrueType /BaseFont /Wingdings2 /FirstChar 149 /LastChar 149 /Widths [600] '
        '/FontDescriptor << /Type /FontDescriptor /FontName /Wingdings2 /Flags 4 >> >>'
    )
    path = tmp_path / 'wingdings.pdf'
    path.write_bytes(
        page_pdf('BT /F1 120 Tf 40 30 Td <95> Tj ET', {'F1': (font, None)}, resources='/Font << /F1 {F1} >>')
    )
    return path


def test_a_font_not_embedded_is_drawn_with_the_fallback_fetched_on_first_use(server, monkeypatch, tmp_path):
    """Through the production entry point: the first fallback reads but lacks the character, the second holds it,
    the third is never asked for."""
    _Served.routes['/symbols-2.ttf'] = (200, SANS)
    _Served.routes['/symbols.ttf'] = (200, SYMBOL)
    _pin(monkeypatch, 'noto-sans-symbols-2', (f'{server}/symbols-2.ttf',), SANS)
    _pin(monkeypatch, 'noto-sans-symbols', (f'{server}/symbols.ttf',), SYMBOL)
    _pin(monkeypatch, 'noto-sans-math', (f'{server}/math.ttf',))
    pixels = read_raster(_wingdings(tmp_path), 1, BBox(0, 0, 200, 150), dpi=72, renderer='own').pixels
    assert (pixels.min(axis=2) < 128).sum() > 20  # the dot operator, inked
    assert _Served.hits == ['/symbols-2.ttf', '/symbols.ttf']


def test_a_fallback_that_cannot_be_had_or_read_is_refused_by_name(server, monkeypatch, tmp_path):
    _pin(monkeypatch, 'noto-sans-symbols-2', (f'{server}/symbols-2.ttf',))
    page = _wingdings(tmp_path)
    with pytest.raises(Unsupported) as refused:
        read_raster(page, 1, BBox(0, 0, 200, 150), dpi=72, renderer='own')
    assert 'font Wingdings2 is not embedded and its fallback noto-sans-symbols-2 cannot be had' in str(refused.value)
    assert 'symbols-2.ttf: failed: HTTP 404' in str(refused.value)
    _Served.routes['/symbols-2.ttf'] = (200, NO_FONT)  # the pinned bytes, but no font program
    _pin(monkeypatch, 'noto-sans-symbols-2', (f'{server}/symbols-2.ttf',), NO_FONT)
    with pytest.raises(Unsupported, match='its fallback noto-sans-symbols-2 was fetched but does not read'):
        read_raster(page, 1, BBox(0, 0, 200, 150), dpi=72, renderer='own')


# --- the pins ------------------------------------------------------------------------------------------------------


def test_every_fallback_the_renderer_may_ask_for_is_pinned_under_a_free_license():
    assert set(_pdf.FALLBACK_FONTS) <= set(FONTS)
    assert all(FONTS[name].license == OFL for name in _pdf.FALLBACK_FONTS)


#: A URL that keeps serving the same bytes: a file of a repository at a commit.
IMMUTABLE = re.compile(
    r'^https://raw\.githubusercontent\.com/(notofonts/noto-cjk|google/fonts|ArtifexSoftware/urw-base35-fonts)/'
    r'[0-9a-f]{40}/'
)


@pytest.mark.parametrize('font', FONTS.values(), ids=lambda f: f.name)
def test_every_url_of_a_font_is_immutable(font):
    assert all(IMMUTABLE.match(url) for url in font.urls), font.urls


def test_the_immutable_url_rule_can_say_no():
    for url in (
        'https://raw.githubusercontent.com/google/fonts/main/ofl/notosansmath/NotoSansMath-Regular.ttf',
        'https://github.com/ArtifexSoftware/urw-base35-fonts/raw/master/fonts/NimbusSans-Regular.otf',
    ):
        assert not IMMUTABLE.match(url)


def test_every_pin_is_a_sha256_and_a_size():
    for font in FONTS.values():
        assert re.fullmatch(r'[0-9a-f]{64}', font.sha256), font.name
        assert font.size > 0
    assert np.unique([f.sha256 for f in FONTS.values()]).size == len(FONTS)
