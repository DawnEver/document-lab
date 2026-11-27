"""Synthetic font programs whose every glyph is known, for the own renderer's text acceptance
(``tests/unit/test_render_text.py``).

Four glyphs on a 1000-unit em, each 700 wide and drawn so a wrong one cannot pass for another: ``A`` a square, ``B`` a
triangle, ``C`` a square ring (its hole wound the other way: drawn by the nonzero rule, it stays open), ``D`` an L.
Glyph 0 (``.notdef``) is empty, so a code that selects no glyph draws nothing.

:func:`truetype` writes them as a TrueType program (``glyf``) with the ``cmap`` subtables asked for; :func:`cff` as a
bare CFF program, plain (glyphs by name) or CID-keyed (glyphs by ``CID``, through a charset that is not the identity);
:func:`opentype` wraps a plain CFF program in an OpenType font; :func:`type1` writes them as a Type 1 program, its
private part eexec-encrypted, in the three parts a ``/FontFile`` stream holds.
"""

from __future__ import annotations

import struct

#: The glyphs: name, then contours of (x, y) points, each closed back to its first.
GLYPHS: tuple[tuple[str, tuple[tuple[tuple[int, int], ...], ...]], ...] = (
    ('A', (((100, 0), (100, 700), (600, 700), (600, 0)),)),
    ('B', (((100, 0), (350, 700), (600, 0)),)),
    ('C', (((100, 0), (100, 700), (600, 700), (600, 0)), ((250, 200), (450, 200), (450, 500), (250, 500)))),
    ('D', (((100, 0), (100, 700), (250, 700), (250, 150), (600, 150), (600, 0)),)),
)
#: Each glyph's advance, in its 1000-unit em.
ADVANCE = 700
#: The CIDs a CID-keyed program gives glyphs 1-4 (its charset): never their glyph ids.
CIDS = (10, 20, 30, 40)


# ---------------------------------------------------------------------------------------------------------------- sfnt


def _sfnt(version: bytes, tables: dict[bytes, bytes]) -> bytes:
    """An sfnt (TrueType or OpenType) font of ``tables``, its directory sorted and checksummed."""
    count = len(tables)
    power = 1 << (count.bit_length() - 1)
    head = version + struct.pack('>HHHH', count, power * 16, power.bit_length() - 1, count * 16 - power * 16)
    offset = len(head) + 16 * count
    directory, body = b'', b''
    for tag in sorted(tables):
        data = tables[tag]
        padded = data + b'\0' * (-len(data) % 4)
        checksum = sum(struct.unpack(f'>{len(padded) // 4}I', padded)) & 0xFFFFFFFF
        directory += tag + struct.pack('>III', checksum, offset + len(body), len(data))
        body += padded
    return head + directory + body


def _head(index_to_loc: int = 1) -> bytes:
    return struct.pack(
        '>IIIIHHqqhhhhHHhhh',
        0x00010000,
        0x00010000,
        0,
        0x5F0F3CF5,
        0,
        1000,
        0,
        0,
        0,
        0,
        700,
        700,
        0,
        8,
        2,
        index_to_loc,
        0,
    )


def _hhea(glyphs: int) -> bytes:
    return struct.pack(
        '>IhhhHhhhhhhhhhhhH', 0x00010000, 800, -200, 0, ADVANCE, 0, 0, 600, 1, 0, 0, 0, 0, 0, 0, 0, glyphs
    )


def _hmtx(glyphs: int) -> bytes:
    return b''.join(struct.pack('>Hh', ADVANCE, 100 if g else 0) for g in range(glyphs))


def _name() -> bytes:
    return struct.pack('>HHH', 0, 0, 6)


def _post(names: tuple[str, ...] | None) -> bytes:
    """A ``post`` table: format 3 (no names) when ``names`` is None, else format 2 naming glyph ``i`` ``names[i]``
    (``.notdef`` the Macintosh set's first; every other a name of its own)."""
    header = struct.pack('>IIhhIIIII', 0x00030000 if names is None else 0x00020000, 0, -100, 50, 0, 0, 0, 0, 0)
    if names is None:
        return header
    indices = [0] + [258 + i for i in range(len(names) - 1)]
    strings = b''.join(bytes([len(n)]) + n.encode() for n in names[1:])
    return header + struct.pack(f'>H{len(indices)}H', len(indices), *indices) + strings


def _cmap_format4(mapping: dict[int, int]) -> bytes:
    """A format 4 subtable mapping each code to its glyph: one segment a code, then the closing 0xFFFF."""
    codes = [*sorted(mapping), 0xFFFF]
    segments = len(codes)
    ends = struct.pack(f'>{segments}H', *codes)
    starts = struct.pack(f'>{segments}H', *codes)
    deltas = struct.pack(f'>{segments}h', *[((mapping.get(c, 0) - c + 0x8000) % 0x10000) - 0x8000 for c in codes])
    ranges = struct.pack(f'>{segments}H', *([0] * segments))
    power = 1 << (segments.bit_length() - 1)
    body = (
        struct.pack('>HHHH', segments * 2, power * 2, power.bit_length() - 1, segments * 2 - power * 2)
        + ends
        + b'\0\0'
        + starts
        + deltas
        + ranges
    )
    return struct.pack('>HHH', 4, 6 + len(body), 0) + body


def _cmap_format0(mapping: dict[int, int]) -> bytes:
    return struct.pack('>HHH', 0, 262, 0) + bytes(mapping.get(c, 0) for c in range(256))


def _cmap(subtables: dict[tuple[int, int], dict[int, int]]) -> bytes:
    """A ``cmap`` of ``subtables`` by (platform, encoding): (1, 0) in format 0, any other in format 4."""
    data = []
    for (platform, encoding), mapping in sorted(subtables.items()):
        data.append((platform, encoding, (_cmap_format0 if platform == 1 else _cmap_format4)(mapping)))
    offset = 4 + 8 * len(data)
    out, body = struct.pack('>HH', 0, len(data)), b''
    for platform, encoding, table in data:
        out += struct.pack('>HHI', platform, encoding, offset + len(body))
        body += table
    return out + body


def _glyf(contours: tuple[tuple[tuple[int, int], ...], ...]) -> bytes:
    if not contours:
        return b''
    points = [p for c in contours for p in c]
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    ends, n = [], 0
    for c in contours:
        n += len(c)
        ends.append(n - 1)
    out = struct.pack('>hhhhh', len(contours), min(xs), min(ys), max(xs), max(ys))
    out += struct.pack(f'>{len(ends)}H', *ends) + struct.pack('>H', 0)
    out += bytes([1] * len(points))  # each point on the curve, its coordinates 16-bit deltas
    previous = (0, 0)
    dxs, dys = [], []
    for x, y in points:
        dxs.append(x - previous[0])
        dys.append(y - previous[1])
        previous = (x, y)
    return out + struct.pack(f'>{len(dxs)}h', *dxs) + struct.pack(f'>{len(dys)}h', *dys)


def truetype(subtables: dict[tuple[int, int], dict[int, int]], *, post_names: bool = True) -> bytes:
    """The glyphs as a TrueType program with the ``cmap`` ``subtables`` (code to glyph id, by (platform, encoding)),
    its ``post`` table naming them unless ``post_names`` is false."""
    outlines = [(), *(contours for _, contours in GLYPHS)]
    glyphs = [_glyf(c) for c in outlines]
    glyphs = [g + b'\0' * (-len(g) % 4) for g in glyphs]
    offsets, at = [], 0
    for g in glyphs:
        offsets.append(at)
        at += len(g)
    offsets.append(at)
    count = len(glyphs)
    points = max(sum(len(c) for c in contours) for contours in outlines)
    maxp = struct.pack('>IHHHHHHHHHHHHHH', 0x00010000, count, points, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0)
    names = ('.notdef', *(name for name, _ in GLYPHS))
    return _sfnt(
        b'\0\1\0\0',
        {
            b'head': _head(),
            b'hhea': _hhea(count),
            b'maxp': maxp,
            b'hmtx': _hmtx(count),
            b'cmap': _cmap(subtables),
            b'lo' + b'ca': struct.pack(f'>{len(offsets)}I', *offsets),  # the glyph offsets' table
            b'glyf': b''.join(glyphs),
            b'post': _post(names if post_names else None),
            b'name': _name(),
        },
    )


# ----------------------------------------------------------------------------------------------------------------- CFF

#: The standard strings of the glyph names (CFF Appendix A): A to D.
_SID = {'A': 34, 'B': 35, 'C': 36, 'D': 37}


def _index(items: list[bytes]) -> bytes:
    if not items:
        return b'\0\0'
    offsets, at = [1], 1
    for item in items:
        at += len(item)
        offsets.append(at)
    return struct.pack('>HB', len(items), 4) + b''.join(struct.pack('>I', o) for o in offsets) + b''.join(items)


def _int(v: int) -> bytes:
    """A DICT integer in its 5-byte form, so a DICT's size does not depend on the offsets it holds."""
    return b'\x1d' + struct.pack('>i', v)


def _number(v: int) -> bytes:
    """A Type 2 charstring integer."""
    if -107 <= v <= 107:
        return bytes([v + 139])
    if 108 <= v <= 1131:
        v -= 108
        return bytes([(v >> 8) + 247, v & 0xFF])
    if -1131 <= v <= -108:
        v = -v - 108
        return bytes([(v >> 8) + 251, v & 0xFF])
    return b'\x1c' + struct.pack('>h', v)


def _charstring(contours: tuple[tuple[tuple[int, int], ...], ...]) -> bytes:
    out, at = _number(ADVANCE), (0, 0)  # the width, first
    for contour in contours:
        x, y = contour[0]
        out += _number(x - at[0]) + _number(y - at[1]) + b'\x15'  # rmoveto
        at = (x, y)
        for x, y in contour[1:]:
            out += _number(x - at[0]) + _number(y - at[1]) + b'\x05'  # rlineto
            at = (x, y)
    return out + b'\x0e'  # endchar


def cff(*, cid_keyed: bool = False) -> bytes:
    """The glyphs as a bare CFF program (``/FontFile3`` ``Type1C``; ``CIDFontType0C`` when ``cid_keyed``): plain,
    named A to D in the standard encoding; CID-keyed, glyph ``i`` the ``CID`` ``CIDS[i - 1]``."""
    charstrings = _index([_number(ADVANCE) + b'\x0e', *(_charstring(c) for _, c in GLYPHS)])
    private = _int(0) + b'\x14' + _int(0) + b'\x15'  # defaultWidthX 0, nominalWidthX 0
    charset = b'\0' + b''.join(
        struct.pack('>H', c if cid_keyed else _SID[n]) for (n, _), c in zip(GLYPHS, CIDS, strict=True)
    )
    strings = _index([b'Test', b'Glyphs'] if cid_keyed else [])
    name = _index([b'Synthetic'])
    header = b'\x01\x00\x04\x04'

    def top(offsets: dict[str, int]) -> bytes:
        dict_ = _int(-100) + _int(-100) + _int(800) + _int(800) + b'\x05'  # FontBBox
        if cid_keyed:
            dict_ = _int(391) + _int(392) + _int(0) + b'\x0c\x1e' + dict_  # ROS
            dict_ += _int(max(CIDS) + 1) + b'\x0c\x22'  # CIDCount
            dict_ += _int(offsets.get('fdarray', 0)) + b'\x0c\x24' + _int(offsets.get('fdselect', 0)) + b'\x0c\x25'
        else:
            dict_ += _int(len(private)) + _int(offsets.get('private', 0)) + b'\x12'
        return dict_ + _int(offsets.get('charset', 0)) + b'\x0f' + _int(offsets.get('charstrings', 0)) + b'\x11'

    def layout(offsets: dict[str, int]) -> tuple[bytes, dict[str, int]]:
        out = header + name + _index([top(offsets)]) + strings + _index([])
        found = {'charset': len(out)}
        out += charset
        found['charstrings'] = len(out)
        out += charstrings
        if cid_keyed:
            found['fdselect'] = len(out)
            out += struct.pack('>BHHBH', 3, 1, 0, 0, len(GLYPHS) + 1)  # format 3: every glyph in font DICT 0
            found['fdarray'] = len(out)
            fd = _int(len(private)) + _int(offsets.get('private', 0)) + b'\x12'
            out += _index([fd])
        found['private'] = len(out)
        out += private
        return out, found

    offsets: dict[str, int] = {}
    for _ in range(3):  # the offsets settle: every integer is 5 bytes
        data, offsets = layout(offsets)
    return data


def opentype(program: bytes, mapping: dict[int, int]) -> bytes:
    """``program`` (a plain CFF program of the glyphs) as an OpenType font, its (3, 1) ``cmap`` ``mapping``."""
    count = len(GLYPHS) + 1
    return _sfnt(
        b'OTTO',
        {
            b'CFF ': program,
            b'head': _head(),
            b'hhea': _hhea(count),
            b'maxp': struct.pack('>IH', 0x00005000, count),
            b'hmtx': _hmtx(count),
            b'cmap': _cmap({(3, 1): mapping}),
            b'post': _post(None),
            b'name': _name(),
        },
    )


# -------------------------------------------------------------------------------------------------------------- Type 1


def _encrypt(plain: bytes, key: int) -> bytes:
    """Type 1 encryption (Adobe Type 1 Font Format 7.2) of four leading zero bytes and ``plain`` from ``key``."""
    out, r = bytearray(), key
    for p in b'\0\0\0\0' + plain:
        c = p ^ (r >> 8)
        r = ((c + r) * 52845 + 22719) & 0xFFFF
        out.append(c)
    return bytes(out)


def _t1_number(v: int) -> bytes:
    if -107 <= v <= 1131 or -1131 <= v <= -108:
        return _number(v)
    return b'\xff' + struct.pack('>i', v)


def _t1_charstring(contours: tuple[tuple[tuple[int, int], ...], ...]) -> bytes:
    out, at = _t1_number(0) + _t1_number(ADVANCE) + b'\x0d', (0, 0)  # hsbw: no side bearing, the advance
    for contour in contours:
        x, y = contour[0]
        out += _t1_number(x - at[0]) + _t1_number(y - at[1]) + b'\x15'  # rmoveto
        at = (x, y)
        for x, y in contour[1:]:
            out += _t1_number(x - at[0]) + _t1_number(y - at[1]) + b'\x05'  # rlineto
            at = (x, y)
        out += b'	'  # closepath
    return out + b'\x0e'  # endchar


def type1() -> tuple[bytes, int, int, int]:
    """The glyphs as a Type 1 program, its built-in encoding the standard one's codes for A to D: as a ``/FontFile``
    stream holds it -- its clear text, its eexec-encrypted part (each charstring encrypted too) and its trailer,
    joined -- and their lengths (``/Length1``, ``/Length2``, ``/Length3``)."""
    clear = (
        b'%!PS-AdobeFont-1.0: Synthetic 001.000\n'
        b'11 dict begin\n'
        b'/FontInfo 1 dict dup begin /FullName (Synthetic) readonly def end readonly def\n'
        b'/FontName /Synthetic def\n'
        b'/Encoding 256 array 0 1 255 {1 index exch /.notdef put} for\n'
        b'dup 65 /A put dup 66 /B put dup 67 /C put dup 68 /D put readonly def\n'
        b'/PaintType 0 def /FontType 1 def /FontMatrix [0.001 0 0 0.001 0 0] readonly def\n'
        b'/FontBBox {0 -200 700 800} readonly def\n'
        b'currentdict end\n'
        b'currentfile eexec\n'
    )
    charstrings = [('.notdef', _t1_number(0) + _t1_number(ADVANCE) + b'\x0d\x0e')]
    charstrings += [(name, _t1_charstring(contours)) for name, contours in GLYPHS]
    private = (
        b'dup /Private 8 dict dup begin\n'
        b'/-| {string currentfile exch readstring pop} executeonly def\n'
        b'/|- {noaccess def} executeonly def\n'
        b'/| {noaccess put} executeonly def\n'
        b'/MinFeature {16 16} def\n'
        b'/password 5839 def\n'
        b'/Subrs 0 array |-\n' + f'2 index /CharStrings {len(charstrings)} dict dup begin\n'.encode()
    )
    for name, charstring in charstrings:
        encrypted = _encrypt(charstring, 4330)
        private += f'/{name} {len(encrypted)} -| '.encode() + encrypted + b' |-\n'
    private += (
        b'end\nend\nreadonly put\nnoaccess put\ndup /FontName get exch definefont pop\nmark currentfile closefile\n'
    )
    encrypted = _encrypt(private, 55665)
    trailer = (b'0' * 64 + b'\n') * 8 + b'cleartomark\n'
    return clear + encrypted + trailer, len(clear), len(encrypted), len(trailer)
