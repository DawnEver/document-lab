r"""Font programs document-lab draws or reads with but never ships: fetched on first use into the user cache, pinned.

What they are (:data:`FONTS`): the own renderer's fallbacks (``_pdf.FALLBACK_FONTS``) -- Noto Sans CJK and Noto Serif
CJK for each of Adobe's CJK collections, Noto Sans Symbols 2, Noto Sans Symbols and Noto Sans Math, all under the SIL
Open Font License 1.1 -- which draw a font a PDF does not embed and no built-in substitute stands for (a CJK composite
font, Wingdings, Webdings); and URW's Nimbus Sans (the Helvetica MuPDF draws; GNU AGPL 3.0 with URW's document
exception), which :mod:`document_lab.glyphs` reads glyph shapes against. Each is pinned to the sha256 and size of its
program and to URLs that keep serving those bytes; nothing is committed or redistributed (user ruling D120 of task
document-lab-pdf-reader).

THE NETWORK is asked only the first time a font is needed -- a glyph of such a font drawn, a glyph shape read -- and
each download is logged (WARNING, logger ``document_lab.fonts``) with its URL and destination. Where fonts are kept
and whether they may be downloaded is a :class:`Fonts` record, set once a process by :func:`use` (the default: the
platform's user cache, downloads allowed). The command line sets it from its environment (:meth:`Fonts.of`):
``DOCUMENT_LAB_OFFLINE=1`` forbids every download, ``DOCUMENT_LAB_CACHE`` names the cache's root; ``document-lab
fonts`` lists the fonts and which the cache holds, ``document-lab fonts --fetch`` downloads them ahead (for a machine
that will be offline).

THE CACHE is the platform's user cache directory (``%LOCALAPPDATA%\document-lab\cache`` on Windows,
``~/Library/Caches/document-lab`` on macOS, ``$XDG_CACHE_HOME/document-lab`` or ``~/.cache/document-lab`` elsewhere),
as Hugging Face keeps weights in ``HF_HUB_CACHE``: each font is ``fonts/<sha256>.<suffix>`` there, used only while its
bytes are the ones pinned.

A font that cannot be had -- offline, every URL failing, other bytes served -- raises :class:`FontUnavailable`,
naming the font and what each URL did; nothing is drawn or read in its place (the renderer counts the glyphs it
would have drawn as undrawn, by that message).
"""

from __future__ import annotations

import functools
import hashlib
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Self

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ['FONTS', 'Font', 'FontUnavailable', 'Fonts', 'fetch', 'program', 'use', 'used']

_log = logging.getLogger(__name__)

#: The User-Agent of every request: the project, generically; nothing about who runs it.
USER_AGENT: Final = 'document-lab-fonts/1.0'
#: Attempts per URL before it is given up.
ATTEMPTS: Final = 3
#: Seconds a request may take.
TIMEOUT: Final = 120
#: HTTP statuses worth another attempt: the server's own failures and its rate limit.
_TRANSIENT: Final = frozenset({408, 425, 429, 500, 502, 503, 504})

OFL: Final = 'SIL Open Font License 1.1'
URW: Final = 'GNU AGPL 3.0 with a document exception (URW base35)'
_NOTO_CJK: Final = 'https://raw.githubusercontent.com/notofonts/noto-cjk/f8d157532fbfaeda587e826d4cd5b21a49186f7c'
_GOOGLE_FONTS: Final = 'https://raw.githubusercontent.com/google/fonts/5026800714d07231937d92eb4cceabf06e05d74e/ofl'
_URW: Final = (
    'https://raw.githubusercontent.com/ArtifexSoftware/urw-base35-fonts/3c0ba3b5687632dfc66526544a4e811fe0ec0cd9'
)


class FontUnavailable(RuntimeError):
    """A font cannot be had here: downloads are forbidden, every URL failed, or none served the pinned bytes."""


@dataclass(frozen=True)
class Font:
    """A font program: where it is fetched from and the bytes it is pinned to."""

    name: str
    #: URLs serving the program, tried in turn.
    urls: tuple[str, ...]
    sha256: str
    size: int
    license: str
    #: Its file suffix in the cache.
    suffix: str = '.otf'


def _cjk(name: str, style: str, region: str, sha256: str, size: int) -> Font:
    return Font(name, (f'{_NOTO_CJK}/{style}/SubsetOTF/{region}/Noto{style}{region}-Regular.otf',), sha256, size, OFL)


#: Every font, by name; the renderer's fallbacks are named by ``_pdf.FALLBACK_FONTS``.
FONTS: Final[dict[str, Font]] = {
    font.name: font
    for font in (
        _cjk('noto-sans-sc', 'Sans', 'SC', 'faa6c9df652116dde789d351359f3d7e5d2285a2b2a1f04a2d7244df706d5ea9', 8331336),
        _cjk('noto-sans-tc', 'Sans', 'TC', '5bab0cb3c1cf89dde07c4a95a4054b195afbcfe784d69d75c340780712237537', 5683368),
        _cjk('noto-sans-jp', 'Sans', 'JP', 'dff723ba59d57d136764a04b9b2d03205544f7cd785a711442d6d2d085ac5073', 4533028),
        _cjk('noto-sans-kr', 'Sans', 'KR', '69975a0ac8472717870aefeab0a4d52739308d90856b9955313b2ad5e0148d68', 4644748),
        _cjk(
            'noto-serif-sc', 'Serif', 'SC', 'e8f396decc1f0963a016a989c3d8852e863d1350996f573860a80767c83a1cd3', 11625800
        ),
        _cjk(
            'noto-serif-tc', 'Serif', 'TC', '63515eb0622d589a2f4062e8757c598b3da3bffd053f69abd181c210158ef10c', 7936104
        ),
        _cjk(
            'noto-serif-jp', 'Serif', 'JP', '2c9a12dbd4f2408c4610c7ee84a108b62d7236c3775baed618c64d9cb44b2f04', 6210492
        ),
        _cjk(
            'noto-serif-kr', 'Serif', 'KR', '5ea012e15cb7eacc1f680aee1703f3b164791b1443ea3e52b65080cca5d179cf', 7669020
        ),
        Font(
            'noto-sans-symbols-2',
            (f'{_GOOGLE_FONTS}/notosanssymbols2/NotoSansSymbols2-Regular.ttf',),
            '7d5fb73b7ca67a6798101741f5d280a3d016a56a197afcd4199dbb57b4b82a21',
            1233128,
            OFL,
            '.ttf',
        ),
        Font(
            'noto-sans-symbols',
            (f'{_GOOGLE_FONTS}/notosanssymbols/NotoSansSymbols%5Bwght%5D.ttf',),
            'f7e7e04b4a24b6c78893d50cbfd2b2f6cae49617ab047bfef668d252adb128f7',
            372684,
            OFL,
            '.ttf',
        ),
        Font(
            'noto-sans-math',
            (f'{_GOOGLE_FONTS}/notosansmath/NotoSansMath-Regular.ttf',),
            '3f495fe933c06786e4d5f6d86b8ee70b6753a68ee3b9d87528726de0f6e2c47d',
            1015396,
            OFL,
            '.ttf',
        ),
        Font(
            'nimbus-sans',
            (f'{_URW}/fonts/NimbusSans-Regular.otf',),
            '7c25be4d78155523080ab85b10277150657ff7dabbcad7037bdd536c9b6d0d08',
            82264,
            URW,
        ),
    )
}


@dataclass(frozen=True)
class Fonts:
    """Where fetched fonts are kept (``cache``, each ``<sha256>.<suffix>`` in it) and whether they may be downloaded."""

    cache: Path
    download: bool = True

    @classmethod
    def of(cls, variables: Mapping[str, str]) -> Self:
        """As the environment ``variables`` say.

        The cache is ``fonts`` under ``DOCUMENT_LAB_CACHE``, else the platform's user cache (under ``LOCALAPPDATA``
        or ``XDG_CACHE_HOME`` where they name one); nothing is downloaded when ``DOCUMENT_LAB_OFFLINE`` is ``1``.
        """
        root = variables.get('DOCUMENT_LAB_CACHE')
        cache = (
            Path(root) / 'fonts'
            if root
            else _user_cache(variables.get('LOCALAPPDATA'), variables.get('XDG_CACHE_HOME'))
        )
        return cls(cache, download=variables.get('DOCUMENT_LAB_OFFLINE') != '1')


def _user_cache(local: str | None = None, xdg: str | None = None) -> Path:
    """The platform's user cache directory for fonts; ``local`` (Windows) and ``xdg`` (elsewhere) where stated."""
    home = Path.home()
    if sys.platform == 'win32':  # a folder of its own beside the state capture keeps in %LOCALAPPDATA%\document-lab
        return Path(local or home / 'AppData' / 'Local') / 'document-lab' / 'cache' / 'fonts'
    base = home / 'Library' / 'Caches' if sys.platform == 'darwin' else Path(xdg or home / '.cache')
    return base / 'document-lab' / 'fonts'


_used = Fonts(_user_cache())


def use(fonts: Fonts) -> None:
    """Keep and fetch fonts as ``fonts`` says from now on, in this process."""
    global _used  # noqa: PLW0603 -- the one setting a process's entry makes
    _used = fonts
    program.cache_clear()


def used() -> Fonts:
    """How fonts are kept and fetched in this process (:func:`use`)."""
    return _used


def fetch(name: str) -> Path:
    """The cached copy of the font ``name`` (:func:`used`): kept when its bytes are the ones pinned, else downloaded.

    Raises:
        KeyError: no font of that name.
        FontUnavailable: it is not cached and downloading is off, or no URL gave the pinned bytes.

    """
    font = FONTS[name]
    path = _used.cache / f'{font.sha256}{font.suffix}'
    if path.is_file() and _digest(path.read_bytes()) == font.sha256:
        return path
    if not _used.download:
        msg = f'{name} cannot be had: it is not in {path.parent} and downloading is off (DOCUMENT_LAB_OFFLINE=1)'
        raise FontUnavailable(msg)
    said = []
    for url in font.urls:
        _log.warning('downloading the font %s (%s) from %s into %s', name, font.license, url, path.parent)
        try:
            data = _download(url)
        except _Unreachable as error:
            said.append(f'{url}: {error}')
            continue
        if _digest(data) != font.sha256 or len(data) != font.size:
            said.append(f'{url} served {len(data)} bytes, sha256 {_digest(data)}')
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(f'{path.name}.{os.getpid()}.part')  # this process's own; renamed whole, never read half
        part.write_bytes(data)
        try:
            part.replace(path)
        except PermissionError:  # Windows: another process renamed its copy in first and holds it open
            if not path.is_file() or _digest(path.read_bytes()) != font.sha256:
                raise
            part.unlink()
        return path
    msg = (
        f'{name} cannot be had: it is pinned to {font.size} bytes, sha256 {font.sha256}, and no URL gave them -- '
        f'{"; ".join(said)}'
    )
    raise FontUnavailable(msg)


@functools.cache
def program(name: str) -> bytes:
    """The bytes of the font ``name`` (:func:`fetch`), read once a process; what ``_pdf.open(fonts=)`` is handed."""
    return fetch(name).read_bytes()


class _Unreachable(Exception):
    """A URL that gave no response body after every attempt."""


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})  # noqa: S310 -- the pinned URLs above
    cause = ''
    for attempt in range(1, ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:  # noqa: S310 -- the pinned URLs above
                return response.read()
        except urllib.error.HTTPError as error:
            cause = f'HTTP {error.code} {error.reason}'
            if error.code not in _TRANSIENT:
                break
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            cause = str(getattr(error, 'reason', error))
        if attempt < ATTEMPTS:
            time.sleep(2**attempt)
    msg = f'failed: {cause}'
    raise _Unreachable(msg)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
