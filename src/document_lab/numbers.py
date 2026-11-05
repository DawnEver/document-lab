"""The one number grammar: how a document spells a number, read whole (a tick label) or first-in-text (a cell).

Recognised: a sign (hyphen or the typographic minus), thousands commas in groups of three, thin-space digit
groups, a decimal point, ``e`` exponents, ``x 10^n`` multipliers (``x``, ``*`` or the multiplication sign)
and a bare ``10ⁿ`` power. A decimal COMMA is not read as one: ``1,5`` is ambiguous with a thousands
separator, so its first number is ``1``.
"""

from __future__ import annotations

import re
from typing import Final

__all__ = ['first_number', 'parse_number']

_SUPERSCRIPT: Final = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺', '0123456789-+')
_MINUS: Final = chr(0x2212)
_THIN: Final = chr(0x2009)

_MANTISSA: Final = r'[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?(?:[eE][+-]?\d+)?|[+-]?\.\d+(?:[eE][+-]?\d+)?'
_TIMES_TEN: Final = rf'\s*[x{chr(0xD7)}*]\s*10\s*\^?\s*([+-]?\d+)'
_NUMBER: Final = re.compile(rf'(?<![\w.])({_MANTISSA})(?:{_TIMES_TEN})?')
_POWER: Final = re.compile(r'^10\^?([+-]?\d+)$')


def _normal(text: str) -> str:
    return text.replace(_MINUS, '-').replace(_THIN, '').translate(_SUPERSCRIPT).strip()


def _value(match: re.Match[str]) -> float:
    value = float(match.group(1).replace(',', ''))
    return value * 10.0 ** int(match.group(2)) if match.group(2) else value


def parse_number(text: str) -> float | None:
    """The value of ``text`` when it is a number and nothing else (a tick label), else ``None``."""
    clean = _normal(text)
    power = _POWER.match(clean)
    if power and any(c in _SUPERSCRIPT for c in map(ord, text)):  # 10 with a superscript power
        return 10.0 ** int(power.group(1))
    match = _NUMBER.fullmatch(clean)
    return _value(match) if match else None


def first_number(text: str) -> float | None:
    """The first number in ``text`` (a cell: ``≈2100``, ``7100±25%``, ``4.5e2 s``), else ``None``."""
    match = _NUMBER.search(_normal(text))
    return _value(match) if match else None
