"""Write ``src/symbol_tables.rs``: the characters the symbol fonts' codes show, where no glyph name or Adobe table says.

Two tables, from three published sources, each pinned by its SHA-256 (``SOURCES``):

- ``CORPORATE_USE``: the code points Adobe's Symbol and ZapfDingbats tables (``ADOBE/symbol.txt``,
  ``ADOBE/zdingbat.txt``, Unicode, Inc., table 1.0 of 2011-07-12, data of 1999) and the Adobe Glyph List give in the
  Corporate Use Subarea (U+F6xx, U+F8xx) for glyphs Unicode had not encoded -- ``copyrightsans``, ``parenlefttp``,
  ``a89`` -- each with the character Unicode 3.2 and 4.0 since gave that glyph, as Apple's tables of the same encodings
  map the same code (``APPLE/SYMBOL.TXT``, ``APPLE/DINGBATS.TXT``, Apple Computer, version c02 of 2005-04-05; the Mac OS
  Symbol layout is the Adobe Symbol encoding's, its Dingbats ZapfDingbats'). Apple's transcoding hints (``+0xF87F``,
  its own corporate-use tags marking a sans-serif form) are dropped: the character is the sign. A code Apple maps into
  the private-use area still (``radicalex``) has no entry.
- ``WINGDINGS``, ``WINGDINGS_2``, ``WINGDINGS_3``, ``WEBDINGS``: each code's character in Microsoft's four symbol fonts,
  by the source reference of ISO/IEC JTC1/SC2/WG2 N4384 (UTC L2/12-368, Michel Suignard for Unicode, 2012-11-06),
  section 7 (pages 68-71): ``<UCS>;<index>[;<index>]``, an index ``F0CC`` being code ``CC`` (decimal) of font ``F``
  (0 Webdings, 1 Wingdings, 2 Wingdings 2, 3 Wingdings 3). Unicode 7.0 encoded these 866 characters (867 less
  w-1255, the Windows logo, not encoded: its code has no entry). Code 32 is each font's space glyph (the indexes start
  at 33). Every character is checked assigned in this Python's Unicode database.

Terms: the Unicode files grant free use of their information in products supporting the Unicode Standard; Apple's
state no restriction (the data is offered without warranty); N4384 is a Unicode standards document. Only facts --
which code shows which character -- are taken; no source text is copied. The downloads are kept under the repository's
``output/symbol_tables/``. Run from anywhere: ``python crates/pdf/data/make_symbol_tables.py`` (PyMuPDF reads N4384).
"""

import hashlib
import re
import unicodedata
import urllib.request
from pathlib import Path

import pymupdf

#: Each source: (cache name, URL, SHA-256).
SOURCES = {
    'adobe_symbol': (
        'adobe_symbol.txt',
        'https://unicode.org/Public/MAPPINGS/VENDORS/ADOBE/symbol.txt',
        'deb78ca840a429311939b9d165890873f71fb23ef223ceeb144a6c6d641a7e52',
    ),
    'adobe_zapf': (
        'adobe_zdingbat.txt',
        'https://unicode.org/Public/MAPPINGS/VENDORS/ADOBE/zdingbat.txt',
        '2d8128a7280cdd47d93272f13c580d7f31fb34586ff65eaf29c7457c224974c0',
    ),
    'apple_symbol': (
        'apple_symbol.txt',
        'https://unicode.org/Public/MAPPINGS/VENDORS/APPLE/SYMBOL.TXT',
        'b8c529daba09a45872ec3b493e944680528206b829193594dcdd774a02d49d12',
    ),
    'apple_dingbats': (
        'apple_dingbats.txt',
        'https://unicode.org/Public/MAPPINGS/VENDORS/APPLE/DINGBATS.TXT',
        'bed8a47f01f770b3175790a35ad6cf9ee118832f6a31dfa86d2947e1a56043a5',
    ),
    'n4384': (
        'n4384.pdf',
        'https://www.unicode.org/wg2/docs/n4384.pdf',
        '15839e133b3ab10b84d68f41f25138d9dedbc94a8a6c2dbb155439d597084a0b',
    ),
}
#: N4384's source reference: its pages 68 to 71 (0-based).
REFERENCE_PAGES = range(67, 71)
#: The Microsoft fonts by their N4384 index prefix: (table name, font).
MICROSOFT = {
    '1': ('WINGDINGS', 'Wingdings'),
    '2': ('WINGDINGS_2', 'Wingdings 2'),
    '3': ('WINGDINGS_3', 'Wingdings 3'),
    '0': ('WEBDINGS', 'Webdings'),
}
#: The space glyph's code in every symbol font.
SPACE = 0x20
#: The indexes N4384 maps: its 867 characters less w-1255, the Windows logo.
INDEXES = 866
#: The Basic Multilingual Plane's private-use area.
PRIVATE_USE = range(0xE000, 0xF900)

CRATE = Path(__file__).resolve().parent.parent
CACHE = CRATE.parent.parent / 'output' / 'symbol_tables'


def fetch(key: str) -> bytes:
    """The pinned source ``key``, downloaded once into the cache; refused when its SHA-256 is not the pinned one."""
    name, url, sha = SOURCES[key]
    path = CACHE / name
    if not path.exists():
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 -- each URL is a pinned https constant
            data = response.read()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    data = path.read_bytes()
    got = hashlib.sha256(data).hexdigest()
    check(got == sha, f'{url}: SHA-256 {got}, pinned {sha}; delete {path} to fetch it again, or re-pin')
    return data


def check(holds: bool, why: str) -> None:  # noqa: FBT001 -- a condition and its reason
    """Refuse to write the tables when a source is not what this generator reads."""
    if not holds:
        raise ValueError(why)


def private(c: int) -> bool:
    """Whether ``c`` is in the Basic Multilingual Plane's private-use area."""
    return c in PRIVATE_USE


def adobe(key: str) -> dict[int, int]:
    """An Adobe table's corporate-use code point by code (``<Unicode>``, tab, ``<code>``, tab, comment)."""
    out = {}
    for line in fetch(key).decode('latin-1').splitlines():
        if m := re.match(r'([0-9A-F]{4})\t([0-9A-F]{2})\t', line):
            unicode, code = int(m[1], 16), int(m[2], 16)
            if private(unicode):
                out[code] = unicode
    return out


def apple(key: str) -> dict[int, int]:
    """An Apple table's character by code (``0xCC``, tab, ``0xUUUU[+0xHINT]``), its transcoding hint dropped."""
    out = {}
    for line in fetch(key).decode('latin-1').splitlines():
        if m := re.match(r'0x([0-9A-F]{2})\t0x([0-9A-F]{4})', line):
            out[int(m[1], 16)] = int(m[2], 16)
    return out


def corporate_use() -> dict[int, int]:
    """Each Adobe corporate-use code point of Symbol and ZapfDingbats with the character Apple maps its code to."""
    out = {}
    for adobe_key, apple_key in (('adobe_symbol', 'apple_symbol'), ('adobe_zapf', 'apple_dingbats')):
        later = apple(apple_key)
        for code, unicode in adobe(adobe_key).items():
            character = later[code]
            if not private(character):
                check(unicode not in out, f'U+{unicode:04X} twice')
                out[unicode] = character
    return out


def microsoft() -> dict[str, dict[int, int]]:
    """Each Microsoft font's character by code, from N4384's source reference."""
    document = pymupdf.open(stream=fetch('n4384'), filetype='pdf')
    text = ''.join(document[i].get_text() for i in REFERENCE_PAGES)
    tables: dict[str, dict[int, int]] = {prefix: {SPACE: SPACE} for prefix in MICROSOFT}
    for ucs, *indexes in re.findall(r'^([0-9A-F]{5});(\d{4})(?:;(\d{4}))?\s*$', text, re.MULTILINE):
        for index in filter(None, indexes):
            table = tables[index[0]]
            code = int(index[1:])
            check(code not in table, f'w-{index} twice')
            table[code] = int(ucs, 16)
    for prefix, table in tables.items():
        for code, character in table.items():
            check(unicodedata.category(chr(character)) != 'Cn', f'w-{prefix}{code:03} U+{character:04X} unassigned')
    check(sum(len(t) - 1 for t in tables.values()) == INDEXES, f'N4384 maps {INDEXES} indexes')
    return tables


def scalar(c: int) -> str:
    """A Rust char literal."""
    return f"'\\u{{{c:04X}}}'"


def table(name: str, doc: str, rows: list[str], kind: str) -> str:
    """A static array of `kind` rows."""
    body = ''.join(f'    {row},\n' for row in rows)
    return f'/// {doc}\npub static {name}: [{kind}; {len(rows)}] = [\n{body}];\n'


if __name__ == '__main__':
    corporate = corporate_use()
    parts = [
        table(
            'CORPORATE_USE',
            "Adobe's corporate-use code points for glyphs Unicode has since encoded, each with that character; sorted.",
            [f'({scalar(u)}, {scalar(c)})' for u, c in sorted(corporate.items())],
            '(char, char)',
        )
    ]
    for prefix, codes in microsoft().items():
        name, font = MICROSOFT[prefix]
        parts.append(
            table(
                name,
                f'{font}: each code with the character it shows; sorted by code.',
                [f'(0x{code:02X}, {scalar(c)})' for code, c in sorted(codes.items())],
                '(u8, char)',
            )
        )
    head = (
        "//! The characters the symbol fonts' codes show where no glyph name or Adobe table says. GENERATED by\n"
        '//! `data/make_symbol_tables.py`; edit that, not this.\n//!\n'
        "//! From Unicode's `MAPPINGS/VENDORS/ADOBE` and `APPLE` Symbol and Dingbats tables and ISO/IEC JTC1/SC2/WG2\n"
        '//! N4384 section 7; the sources, their versions and terms are in the generator.\n\n'
        '// spelling-off: the code points below are data, not words.\n'
    )
    out = head + '\n'.join(parts) + '// spelling-on\n'
    CRATE.joinpath('src', 'symbol_tables.rs').write_text(out, encoding='utf-8', newline='\n')
