"""What a document states beside its pages, read into the IR: outline, links, labels, metadata, fields, files."""

import shutil

import pytest

from document_lab.ir import BBox, Destination, Document, Metadata
from document_lab.pdf import read_attachment, read_pdf
from tests.synthetic import document_data_pdf, mixed_pdf


@pytest.fixture
def path(tmp_path):
    return document_data_pdf(tmp_path / 'data.pdf')


@pytest.fixture
def doc(path):
    return read_pdf(path)


def test_the_outline_is_a_tree_leading_to_each_heading_as_displayed(doc):
    assert [i.title for i in doc.outline] == ['Introduction', 'Method', 'Results']
    method = doc.outline[1]
    assert [c.title for c in method.children] == ['Setup']
    assert method.children[0].destination == Destination(2, 'XYZ', 72.0, 400.0)
    # Page 3 is turned a quarter: its heading's place is still the point the page shows it at.
    assert doc.outline[2].destination == Destination(3, 'XYZ', 72.0, 100.0)
    assert all(i.action is None for i in doc.outline)
    assert doc.warnings == ()


def test_links_carry_their_box_as_displayed_and_their_target(doc):
    uri, inner = doc.page(1).links
    assert uri.uri == 'https://example.org/'
    assert uri.provenance.bbox == BBox(72, 700, 200, 720)
    assert uri.provenance.page == 1
    assert inner.uri is None
    assert inner.destination == Destination(2, 'XYZ', 72.0, 400.0)
    assert doc.page(2).links == ()
    (turned,) = doc.page(3).links
    assert turned.uri == 'https://example.org/3'
    assert turned.provenance.bbox == BBox(672, 72, 692, 200)  # x 72..200 along the turned page's height


def test_links_are_read_whether_or_not_annotations_are_drawn(path):
    assert read_pdf(path, annotations=True).page(1).links == read_pdf(path).page(1).links


def test_each_page_has_its_label(doc):
    assert doc.page_labels == ('i', 'A-1', 'A-2')


def test_both_metadata_records_are_reported_and_the_newer_named(doc):
    info, xmp = doc.metadata.info, doc.metadata.xmp
    assert (info.title, info.author, info.modified) == ('Synthetic', 'Ada', "D:20240102030405+00'00'")
    assert (xmp.title, xmp.creators, xmp.modified) == ('Synthetic (XMP)', ('Ada', 'Grace'), '2024-01-03T03:04:05Z')
    assert '<x:xmpmeta' in xmp.packet
    assert doc.metadata.newer == 'xmp'


def test_form_fields_carry_their_values_and_widgets(doc):
    name, agree = doc.fields
    assert (name.name, name.kind, name.value) == ('name', 'Tx', 'Ada')
    (widget,) = name.widgets
    assert (widget.provenance.page, widget.provenance.bbox) == (1, BBox(72, 200, 272, 220))
    assert (agree.name, agree.kind, agree.value) == ('agree', 'Btn', 'Yes')
    (box,) = agree.widgets
    assert (box.provenance.page, box.state, box.export) == (2, 'Yes', 'Yes')


def test_embedded_files_are_listed_and_their_bytes_read(path, doc):
    data, note = doc.attachments
    assert (data.key, data.name, data.description, data.size) == ('data.csv', 'data.csv', 'numbers', 8)
    assert data.provenance.page is None
    assert read_attachment(path, data) == b'a,b\n1,2\n'
    assert (note.key, note.name, note.provenance.page) == (None, 'note.txt', 2)
    assert read_attachment(path, note) == b'hi'


def test_an_attachment_is_read_only_from_the_file_it_came_from(path, doc, tmp_path):
    other = mixed_pdf(tmp_path / 'other.pdf')
    with pytest.raises(ValueError, match='not the file the attachment was read from'):
        read_attachment(other, doc.attachments[0])
    copy = shutil.copy(path, tmp_path / 'copy.pdf')
    assert read_attachment(copy, doc.attachments[0]) == b'a,b\n1,2\n'  # the same bytes under another name


def test_what_a_document_does_not_state_is_absent(tmp_path):
    doc = read_pdf(mixed_pdf(tmp_path / 'plain.pdf'))
    assert (doc.outline, doc.page_labels, doc.fields, doc.attachments) == ((), (), (), ())
    assert doc.metadata.xmp is None
    assert doc.metadata.newer is None
    assert all(p.links == () for p in doc.pages)
    assert Document('0', 'x').metadata == Metadata()
