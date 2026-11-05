"""A paper PDF as per-section markdown plus cropped figures, tables and equations (``document-lab paper``).

Absorbed from paper_pdf_ingest. Entry: :func:`document_lab.paper.ingest.ingest`; the layout it writes is
documented there. The steps are modules of their own: ``convert`` (PDF -> markdown), ``sections`` (split and
clean), ``output`` (section files, image routing, paper.md, INDEX.md), ``render``/``crop``/``pagemap`` (figure,
table and equation crops) and ``text`` (slugs, title).
"""
