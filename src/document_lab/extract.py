"""Typed values read against a caller's schema, each carrying its provenance and its basis.

The schema is the caller's: document-lab knows no field names. A :class:`Field` names what to read, its kind,
and the labels that identify it in a table. A :class:`Value` keeps the source ``text`` it was read from, so a
qualifier the number drops (``≈``, ``≤``, ``±25%``) is never lost, and a ``basis``: ``rule`` when deterministic
code read it here, ``model`` when a model answered (:mod:`document_lab.ai`).

A field that no row or column names, or whose text does not read as its kind, is ABSENT from the result -- never a
``None`` value that would read as "the document says nothing". Two columns that both match one field refuse
(:class:`AmbiguousField`) rather than guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Final, Literal

from document_lab.numbers import first_number

if TYPE_CHECKING:
    from document_lab.ir import Provenance, Table

__all__ = ['AmbiguousField', 'Basis', 'Field', 'Kind', 'Schema', 'Value', 'coerce', 'from_columns', 'from_rows']

Kind = Literal['number', 'integer', 'text']
_KINDS: Final = ('number', 'integer', 'text')


class Basis(StrEnum):
    """Where a value came from."""

    RULE = 'rule'
    MODEL = 'model'


class AmbiguousField(ValueError):
    """More than one row or column matches one field."""


@dataclass(frozen=True, slots=True)
class Field:
    """One value to read: its name, kind, a description for a model, and the labels a table names it by."""

    name: str
    kind: Kind
    description: str = ''
    labels: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        """Refuse an unknown kind; a field with no labels is named by its name."""
        if self.kind not in _KINDS:
            msg = f'field {self.name!r}: kind {self.kind!r} is not one of {_KINDS}'
            raise ValueError(msg)
        if not self.labels:
            object.__setattr__(self, 'labels', (self.name,))

    def names(self, header: str) -> bool:
        """Whether a row key or column header names this field: it contains one label, case and spacing aside."""
        text = _norm(header)
        return any(_norm(label) in text for label in self.labels)


@dataclass(frozen=True, slots=True)
class Schema:
    """The fields a caller wants read."""

    fields: tuple[Field, ...]

    def json_schema(self) -> dict:
        """The schema a model answers in: per field its value (or null) and the verbatim text it read it from."""
        kind = {'number': 'number', 'integer': 'integer', 'text': 'string'}
        properties = {
            f.name: {
                'type': 'object',
                'description': f.description or f.name,
                'properties': {'value': {'type': [kind[f.kind], 'null']}, 'quote': {'type': ['string', 'null']}},
                'required': ['value', 'quote'],
                'additionalProperties': False,
            }
            for f in self.fields
        }
        names = [f.name for f in self.fields]
        return {'type': 'object', 'properties': properties, 'required': names, 'additionalProperties': False}


@dataclass(frozen=True, slots=True)
class Value:
    """A typed value, the text it was read from, where that text stands, and who read it."""

    name: str
    value: float | int | str
    text: str
    provenance: Provenance
    basis: Basis


def coerce(text: str, kind: Kind) -> float | int | str | None:
    """``text`` read as ``kind``: its first number, an integral first number, or the stripped text; else ``None``."""
    if kind == 'text':
        stripped = ' '.join(text.split())
        return stripped or None
    number = first_number(text)
    if number is None:
        return None
    if kind == 'integer':
        return int(number) if number == int(number) else None
    return number


def from_rows(table: Table, schema: Schema, *, key_column: int = 0, value_column: int = 1) -> dict[str, Value]:
    """A key-value table: a row whose ``key_column`` names a field gives its value from ``value_column``."""
    out: dict[str, Value] = {}
    for f in schema.fields:
        rows = [
            r
            for r, row in enumerate(table.rows)
            if len(row) > max(key_column, value_column) and f.names(row[key_column])
        ]
        if len(rows) > 1:
            msg = f'field {f.name!r}: rows {rows} all match {f.labels}'
            raise AmbiguousField(msg)
        if rows:
            value = _read(table, rows[0], value_column, f)
            if value is not None:
                out[f.name] = value
    return out


def from_columns(table: Table, schema: Schema, *, header_row: int = 0) -> list[dict[str, Value]]:
    """A column table: the header row names fields; each row below is one record."""
    header = table.rows[header_row]
    columns: dict[str, int] = {}
    for f in schema.fields:
        matches = [c for c, text in enumerate(header) if f.names(text)]
        if len(matches) > 1:
            msg = f'field {f.name!r}: columns {matches} all match {f.labels}'
            raise AmbiguousField(msg)
        if matches:
            columns[f.name] = matches[0]
    fields = {f.name: f for f in schema.fields}
    records = []
    for r in range(header_row + 1, len(table.rows)):
        record = {}
        for name, c in columns.items():
            value = _read(table, r, c, fields[name]) if c < len(table.rows[r]) else None
            if value is not None:
                record[name] = value
        records.append(record)
    return records


def _read(table: Table, row: int, col: int, f: Field) -> Value | None:
    cell = table.cell(row, col)
    value = coerce(cell.text, f.kind)
    return None if value is None else Value(f.name, value, cell.text, cell.provenance, Basis.RULE)


def _norm(text: str) -> str:
    return ' '.join(text.casefold().split())
