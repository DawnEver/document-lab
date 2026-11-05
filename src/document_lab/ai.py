"""Values a model reads out of a page or a table, through ai-lab's one ``respond`` interface (the ``ai`` extra).

The model answers the schema's JSON shape (:meth:`document_lab.extract.Schema.json_schema`): per field a value
or null, and the verbatim text it read it from. That quote is how an answer earns provenance: a quote found in a
span or cell cites that span's or cell's box; a quote not found cites the page or table as a whole -- never a
guessed box. A null, or a value not of the field's kind, is absent. Every value's basis is ``model``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

try:
    from ai_lab import Image, ResponseRequest, Text
except ImportError as missing:  # the base package never imports this module
    msg = "document_lab.ai needs the 'ai' extra: install document-lab[ai]"
    raise ImportError(msg) from missing

from document_lab.extract import Basis, Value, coerce
from document_lab.ir import Page, Provenance, Table

if TYPE_CHECKING:
    from ai_lab import Responder

    from document_lab.extract import Field, Schema

__all__ = ['INSTRUCTIONS', 'ask']

INSTRUCTIONS: Final = (
    'Read the requested values from the document text. For each field give its value, or null when the text '
    'does not state it, and quote the exact text you read it from. Do not infer or convert units.'
)


def ask(
    responder: Responder,
    schema: Schema,
    source: Page | Table,
    *,
    instructions: str = '',
    image: bytes | None = None,
) -> dict[str, Value]:
    """The schema's values as ``responder`` reads them from ``source`` (and an optional PNG of it)."""
    text, pieces, whole = _render(source)
    context = (Text(text), Image(image)) if image is not None else (Text(text),)
    request = ResponseRequest(
        instructions=f'{INSTRUCTIONS}\n{instructions}'.strip(), context=context, schema=schema.json_schema()
    )
    answer = responder.respond(request).value
    out = {}
    for f in schema.fields:
        item = answer.get(f.name) or {}
        value = _typed(item.get('value'), f)
        if value is None:
            continue
        quote = item.get('quote') or ''
        where = next((prov for piece, prov in pieces if quote and _norm(quote) in _norm(piece)), whole)
        out[f.name] = Value(f.name, value, quote or str(item['value']), where, Basis.MODEL)
    return out


def _typed(raw: object, f: Field) -> float | int | str | None:
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, int | float) and f.kind != 'text':
        if f.kind == 'integer':
            return int(raw) if raw == int(raw) else None
        return float(raw)
    return coerce(str(raw), f.kind)


def _render(source: Page | Table) -> tuple[str, list[tuple[str, Provenance]], Provenance]:
    """The text a model reads, the pieces a quote is looked up in, and the provenance of the whole."""
    if isinstance(source, Table):
        cells = [source.cell(r, c) for r, row in enumerate(source.rows) for c in range(len(row))]
        text = '\n'.join('\t'.join(row) for row in source.rows)
        return text, [(cell.text, cell.provenance) for cell in cells], source.provenance
    if not source.spans:
        msg = f'page {source.number} has no text to read'
        raise ValueError(msg)
    first = source.spans[0].provenance
    whole = Provenance(first.sha256, first.uri, page=source.number)
    return '\n'.join(s.text for s in source.spans), [(s.text, s.provenance) for s in source.spans], whole


def _norm(text: str) -> str:
    return ' '.join(text.casefold().split())
