"""Write ``src/cid2unicode.rs`` and ``src/cid2unicode/``: Adobe's text of each CID of its CJK character collections.

The source is Adobe's CID-to-Unicode CMaps ``pdf2unicode/Adobe-<Ordering>-UCS2`` of ``mapping-resources-pdf``, the
tables ISO 32000-2 section 9.10.2 names for the text of a font on a registry collection, pinned at ``REVISION`` and
each file by its SHA-256 (``SHA256``): a file that is not the pinned bytes is refused, nothing written. They
are Adobe's own choice of each CID's text: one character, a character past U+FFFF, or a sequence (an ideographic
variation sequence, a decomposed form). ``cmap-resources``' ``cid2code.txt`` is the inverse (each encoding's code to its
CID), so read backward it leaves the CIDs no encoding reaches -- a third of Adobe-Japan1, its vertical, proportional and
variant glyphs -- without text, and offers several characters for a CID where these CMaps chose one.

What is written, per collection (``ORDERINGS``):
- ``src/cid2unicode/<Ordering>.bin``: each CID from 0 as a little-endian u16 -- its text when that is one UTF-16 code
  unit (never a surrogate), else 0;
- ``src/cid2unicode.rs``: the texts longer than one unit, sorted by CID, beside the ``.bin`` each collection includes;
- ``src/cid2unicode/LICENSE.txt``: Adobe's BSD-3-Clause notice of the source, which a redistribution must carry.

A CID mapped to U+0000 or U+FFFD (``.notdef``, a glyph Adobe gives no text) is left out: no text, never a guess. The
downloads are kept under the repository's ``output/cache/cid2unicode/``, each named by its SHA-256. Run from anywhere:
``python crates/pdf/data/make_cid2unicode.py``; the output is a function of the pinned files alone.
"""

import hashlib
import re
import urllib.request
from pathlib import Path

REPOSITORY = 'adobe-type-tools/mapping-resources-pdf'
#: The tag 20230118 (Adobe-Japan1-7, Adobe-GB1-5, Adobe-CNS1-7, Adobe-Korea1-2, Adobe-KR-9).
REVISION = '2dd5e53fb74a01718b9dfd448a0d1cce6fff2aa5'
#: The collections with a table, by ``/Ordering`` (``/Registry (Adobe)``).
ORDERINGS = ('Japan1', 'GB1', 'CNS1', 'Korea1', 'KR')
#: Each file read at ``REVISION``, by its SHA-256.
SHA256 = {
    'pdf2unicode/Adobe-Japan1-UCS2': '6a9693361647a37996312cc57071bb79f8c06411207be7c730a83fda1254cd82',
    'pdf2unicode/Adobe-GB1-UCS2': '368b40ec05568faf323cfbbddf23c60d566bbc88e366981eb6541b026381a3b8',
    'pdf2unicode/Adobe-CNS1-UCS2': '8375afd535e153a7e3bcf448be93e5012b26229547a20d52520e46e3c386351f',
    'pdf2unicode/Adobe-Korea1-UCS2': '45bcf869acdcec2507f75836919e589e7137b9f9dbaa8e4975747035024248ef',
    'pdf2unicode/Adobe-KR-UCS2': '6861a3208f331ecd73369c5d51eacf3b72c820fe3c87524f79c9ac1bb16830ec',
    'LICENSE.txt': 'feb8b068a417681821823ef2e8995072aa111121fa11f8664038f78717f4497c',
}

CRATE = Path(__file__).resolve().parent.parent
CACHE = CRATE.parent.parent / 'output' / 'cache' / 'cid2unicode'
HEX = r'<([0-9a-fA-F]+)>'
#: Texts that are none: U+0000 and U+FFFD, a glyph Adobe gives no text.
NO_TEXT = frozenset({'\0', '\N{REPLACEMENT CHARACTER}'})
#: The last code point one UTF-16 code unit holds.
LAST_UNIT = 0xFFFF
#: rustfmt's `fn_call_width`: a tuple's items up to it stay on one line.
CALL_WIDTH = 60


def fetch(name: str) -> str:
    """The pinned revision's file ``name``, downloaded once and kept by its SHA-256; refused unless it is the pinned."""
    sha = SHA256[name]
    path = CACHE / sha
    url = f'https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/{name}'
    if path.exists():
        data = path.read_bytes()
        door = f'delete {path} to fetch it again'
    else:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        door = 're-pin it only with a new REVISION'
    got = hashlib.sha256(data).hexdigest()
    if got != sha:
        msg = f'{url}: SHA-256 {got}, pinned {sha}; {door}'
        raise ValueError(msg)
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return data.decode('latin-1')


def units(hex_text: str) -> list[int]:
    """A CMap destination's UTF-16BE bytes as code units."""
    data = bytes.fromhex(hex_text)
    return [int.from_bytes(data[i : i + 2], 'big') for i in range(0, len(data), 2)]


def cid_text(cmap: str) -> dict[int, str]:
    """A ``ToUnicode``-style CMap's text by CID (``bfchar`` and ``bfrange``, both forms)."""
    out: dict[int, list[int]] = {}
    for block in re.findall(r'beginbfchar(.*?)endbfchar', cmap, re.DOTALL):
        for cid, dest in re.findall(rf'{HEX}\s*{HEX}', block):
            out[int(cid, 16)] = units(dest)
    for block in re.findall(r'beginbfrange(.*?)endbfrange', cmap, re.DOTALL):
        for lo, hi, dest in re.findall(rf'{HEX}\s*{HEX}\s*(<[0-9a-fA-F]+>|\[[^\]]*\])', block):
            first, last = int(lo, 16), int(hi, 16)
            if dest.startswith('['):
                for offset, each in enumerate(re.findall(HEX, dest)):
                    out[first + offset] = units(each)
                continue
            start = units(dest[1:-1])
            # The destination's last code unit increments (these CMaps carry past a byte: <73FF> runs on to <7400>, as
            # this crate's `CMap` reads a range too); a unit carried out of its range fails to decode below.
            for offset in range(last - first + 1):
                out[first + offset] = [*start[:-1], start[-1] + offset]
    texts = {cid: decode(u) for cid, u in out.items()}
    return {cid: text for cid, text in texts.items() if text not in NO_TEXT}


def decode(code_units: list[int]) -> str:
    """UTF-16 code units as text; a lone surrogate, or a unit past 0xFFFF, refuses."""
    return b''.join(u.to_bytes(2, 'big') for u in code_units).decode('utf-16-be')


def rust_string(text: str) -> str:
    """``text`` as a Rust string literal of escapes."""
    return '"' + ''.join(f'\\u{{{ord(c):X}}}' for c in text) + '"'


def collection(ordering: str) -> str:
    """Write ``ordering``'s ``.bin`` and return its ``Collection`` literal."""
    cmap = fetch(f'pdf2unicode/Adobe-{ordering}-UCS2')
    title = re.search(rf'%%Title: \(Adobe-{ordering}-UCS2 Adobe {ordering} (\d+)\)', cmap)
    if title is None:
        msg = f'Adobe-{ordering}-UCS2 states no supplement in its title'
        raise ValueError(msg)
    supplement = int(title.group(1))
    texts = cid_text(cmap)
    single = bytearray()
    long = []
    for cid in range(max(texts) + 1):
        text = texts.get(cid, '')
        one = len(text) == 1 and ord(text) <= LAST_UNIT
        single += (ord(text) if one else 0).to_bytes(2, 'little')
        if text and not one:
            items = f'{cid}, {rust_string(text)}'
            # As rustfmt lays a tuple: one line up to its call width, else an item a line.
            wide = f'            (\n                {cid},\n                {rust_string(text)},\n            ),\n'
            long.append(f'            ({items}),\n' if len(items) <= 60 else wide)  # noqa: PLR2004
    (CRATE / 'src' / 'cid2unicode' / f'{ordering}.bin').write_bytes(bytes(single))
    return (
        f'    // Adobe-{ordering}, supplements 0 to {supplement}.\n'
        f'    Collection {{\n        ordering: "{ordering}",\n'
        f'        single: include_bytes!("cid2unicode/{ordering}.bin"),\n'
        f'        long: &[\n{"".join(long)}        ],\n'
        '    },\n'
    )


HEADER = f"""\
//! Adobe's text of each CID of its CJK character collections. GENERATED by `data/make_cid2unicode.py`; edit that,
//! not this.
//!
//! From `pdf2unicode/Adobe-<Ordering>-UCS2` of <https://github.com/{REPOSITORY}> at
//! `{REVISION}`: Copyright Adobe, BSD-3-Clause, the notice in `cid2unicode/LICENSE.txt`.

/// One collection's text by CID (`/Registry (Adobe)`, its `/Ordering`); [`crate::cmap`] reads it.
#[derive(Debug)]
pub struct Collection {{
    /// The `/Ordering`.
    pub ordering: &'static str,
    /// Each CID from 0 as a little-endian u16: its text when that is one UTF-16 code unit, else 0.
    pub single: &'static [u8],
    /// The CIDs whose text is longer than one code unit, sorted by CID.
    pub long: &'static [(u16, &'static str)],
}}

"""

if __name__ == '__main__':
    # Every source is fetched and checked before anything is written.
    for name in SHA256:
        fetch(name)
    (CRATE / 'src' / 'cid2unicode').mkdir(exist_ok=True)
    body = ''.join(collection(o) for o in ORDERINGS)
    notice = fetch('LICENSE.txt')
    (CRATE / 'src' / 'cid2unicode' / 'LICENSE.txt').write_text(notice, encoding='utf-8', newline='\n')
    count = len(ORDERINGS)
    # The escapes' hex digits are code points, not words: the family spelling check's door for data skips them.
    table = (
        '// spelling-off: the code points below are data, not words.\n'
        f'/// Every collection with a table.\npub static COLLECTIONS: [Collection; {count}] = [\n{body}];\n'
        '// spelling-on\n'
    )
    (CRATE / 'src' / 'cid2unicode.rs').write_text(HEADER + table, encoding='utf-8', newline='\n')
