"""Pixels as PNG (ISO/IEC 15948): 8-bit gray or RGB, each row as it is (filter 0), the rows deflated.

:func:`encode` writes an image held whole. A writer streaming rows as they come (`document_lab.capture.export`)
builds the same file from its parts: :data:`SIGNATURE`, :func:`header`, :func:`scanlines` deflated into ``IDAT``
:func:`chunk` s, and an empty ``IEND``.
"""

from __future__ import annotations

import struct
import zlib
from typing import Final

import numpy as np

__all__ = ['LEVEL', 'SIGNATURE', 'chunk', 'encode', 'header', 'scanlines']

SIGNATURE: Final = b'\x89PNG\r\n\x1a\n'
LEVEL: Final = 6  # zlib's own default: a PNG of a page is written once and read many times
_COLOR_TYPE: Final = {1: 0, 3: 2}  # channels -> PNG color type: gray, RGB


def chunk(kind: bytes, data: bytes) -> bytes:
    """A chunk: its length, ``kind``, ``data`` and their CRC."""
    return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))


def header(width: int, height: int, channels: int) -> bytes:
    """The ``IHDR`` chunk of an 8-bit image of ``channels`` (1 gray, 3 RGB), ``width`` x ``height``."""
    if channels not in _COLOR_TYPE:
        msg = f'a PNG here is gray or RGB: {channels} channels'
        raise ValueError(msg)
    return chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, _COLOR_TYPE[channels], 0, 0, 0))


def scanlines(rows: np.ndarray) -> bytes:
    """``rows`` -- ``(n, w)`` gray or ``(n, w, 3)`` RGB, ``uint8`` -- as PNG scanlines: each after its filter byte 0."""
    flat = np.ascontiguousarray(rows, dtype=np.uint8).reshape(len(rows), -1)
    lines = np.zeros((len(rows), 1 + flat.shape[1]), dtype=np.uint8)
    lines[:, 1:] = flat
    return lines.tobytes()


def encode(image: np.ndarray) -> bytes:
    """``image`` -- ``(h, w)`` gray or ``(h, w, 3)`` RGB, ``uint8`` -- as a PNG file's bytes."""
    channels = 1 if image.ndim == 2 else image.shape[2]  # noqa: PLR2004 -- rows x columns
    if not image.shape[0] or not image.shape[1]:  # ISO/IEC 15948 11.2.2: zero is an invalid width or height
        msg = f'a PNG holds at least one pixel: {image.shape[1]} x {image.shape[0]}'
        raise ValueError(msg)
    return b''.join(
        [
            SIGNATURE,
            header(image.shape[1], image.shape[0], channels),
            chunk(b'IDAT', zlib.compress(scanlines(image), LEVEL)),
            chunk(b'IEND', b''),
        ]
    )
