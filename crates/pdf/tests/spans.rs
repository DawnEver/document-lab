//! Glyphs join into spans along a baseline; a gap reads as a space; a new font, size or line starts a span.

mod common;

use pdf::File;
use pdf::spans::{Span, spans};

const FONTS: &str = "/Resources << /Font << /F1 5 0 R /F2 6 0 R >> >>";
const HELVETICA: &str = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>";
const BOLD: &str = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>";

fn page_spans(content: &str) -> Vec<Span> {
    let file = File::open(common::one_page(FONTS, content, &[HELVETICA, BOLD]), b"").unwrap();
    let page = file.pages().unwrap().remove(0);
    spans(&file.interpret(&page).unwrap().events)
}

fn texts(content: &str) -> Vec<String> {
    page_spans(content).into_iter().map(|s| s.text).collect()
}

#[test]
fn kerned_pieces_and_separate_shows_on_one_baseline_are_one_span() {
    assert_eq!(
        texts("BT /F1 10 Tf 10 50 Td [(He) 20 (llo)] TJ (!) Tj ET"),
        ["Hello!"]
    );
}

#[test]
fn a_gap_without_a_space_glyph_reads_as_a_space() {
    assert_eq!(
        texts("BT /F1 10 Tf 10 50 Td (one) Tj 20 0 Td (two) Tj ET"),
        ["one two"]
    );
    assert_eq!(texts("BT /F1 10 Tf 10 50 Td (one two) Tj ET"), ["one two"]);
}

#[test]
fn a_font_change_a_new_line_or_a_far_jump_starts_a_span() {
    assert_eq!(
        texts("BT /F1 10 Tf 10 50 Td (plain) Tj /F2 10 Tf (bold) Tj ET"),
        ["plain", "bold"]
    );
    assert_eq!(
        texts("BT /F1 10 Tf 10 50 Td (up) Tj 0 -12 Td (down) Tj ET"),
        ["up", "down"]
    );
    assert_eq!(
        texts("BT /F1 10 Tf 10 50 Td (left) Tj 100 0 Td (right) Tj ET"),
        ["left", "right"]
    );
}

#[test]
fn a_span_box_covers_its_glyphs_and_blank_text_is_no_span() {
    let found = page_spans("BT /F1 10 Tf 10 50 Td (AB) Tj (   ) Tj ET");
    assert_eq!(found.len(), 1);
    let bbox = found[0].bbox;
    assert!(
        (bbox.x0 - 10.0).abs() < 1e-9 && (bbox.x1 - 23.34).abs() < 1e-6,
        "{bbox:?}"
    );
    assert_eq!(found[0].text, "AB");
}

#[test]
fn a_span_carries_the_direction_its_baseline_runs_in_page_space() {
    let found = page_spans("BT /F1 10 Tf 10 50 Td (across) Tj 0 1 -1 0 100 20 Tm (up) Tj ET");
    let directions: Vec<(String, (f64, f64))> = found
        .into_iter()
        .map(|s| (s.text, (s.direction.0.round(), s.direction.1.round())))
        .collect();
    // y grows down in page space: text set bottom to top runs along -y.
    assert_eq!(
        directions,
        [
            ("across".to_string(), (1.0, 0.0)),
            ("up".to_string(), (0.0, -1.0))
        ]
    );
}
