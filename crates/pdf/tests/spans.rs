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
    spans(&file.interpret(&page, false).unwrap().events)
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

#[test]
fn a_vertical_run_is_its_own_span_read_top_to_bottom() {
    // F1 Helvetica, F3 an Identity-V font (CIDs 1, 2 read U+4E00, U+4E8C).
    let fonts = "/Resources << /Font << /F1 5 0 R /F3 6 0 R >> >>";
    let vertical = "<< /Type /Font /Subtype /Type0 /BaseFont /Song-Identity-V /Encoding /Identity-V
        /DescendantFonts [7 0 R] /ToUnicode 8 0 R >>";
    let cid = "<< /Type /Font /Subtype /CIDFontType0 /BaseFont /Song >>";
    let to_unicode = common::stream(
        "",
        "1 begincodespacerange <0000> <FFFF> endcodespacerange 2 beginbfchar <0001> <4E00> <0002> <4E8C> endbfchar",
    );
    let content = "BT /F1 10 Tf 10 50 Td (side) Tj /F3 10 Tf 1 0 0 1 100 90 Tm <0001> Tj <0002> Tj <0001> Tj ET";
    let file = File::open(
        common::one_page(fonts, content, &[HELVETICA, vertical, cid, &to_unicode]),
        b"",
    )
    .unwrap();
    let page = file.pages().unwrap().remove(0);
    let found = spans(&file.interpret(&page, false).unwrap().events);
    let read: Vec<_> = found
        .iter()
        .map(|s| (s.text.as_str(), s.vertical))
        .collect();
    assert_eq!(read, [("side", false), ("\u{4E00}\u{4E8C}\u{4E00}", true)]);
    let bbox = found[1].bbox;
    assert!(
        (bbox.x0 - 95.0).abs() < 1e-9
            && (bbox.y0 - 10.0).abs() < 1e-9
            && (bbox.x1 - 105.0).abs() < 1e-9
            && (bbox.y1 - 40.0).abs() < 1e-9,
        "{bbox:?}"
    );
}

/// Each span's text, and whether it shares its unspaced number with the one before.
fn joined(content: &str) -> Vec<(String, bool)> {
    let found = page_spans(content);
    let after: Vec<bool> = std::iter::once(false)
        .chain(
            found
                .windows(2)
                .map(|w| w[0].unspaced.is_some() && w[0].unspaced == w[1].unspaced),
        )
        .collect();
    found.into_iter().map(|s| s.text).zip(after).collect()
}

#[test]
fn spans_continuing_one_word_share_a_number() {
    let pair = |a: &str, b: &str, glued: bool| vec![(a.to_string(), false), (b.to_string(), glued)];
    // A change of font inside a word, and one at a word gap: a positioned gap, or a space glyph on either side.
    assert_eq!(
        joined("BT /F1 10 Tf 10 50 Td (Head) Tj /F2 10 Tf (line) Tj ET"),
        pair("Head", "line", true)
    );
    assert_eq!(
        joined("BT /F1 10 Tf 10 50 Td [(plain) -300] TJ /F2 10 Tf (bold) Tj ET"),
        pair("plain", "bold", false)
    );
    assert_eq!(
        joined("BT /F1 10 Tf 10 50 Td (plain ) Tj /F2 10 Tf (bold) Tj ET"),
        pair("plain", "bold", false)
    );
    assert_eq!(
        joined("BT /F1 10 Tf 10 50 Td (plain) Tj /F2 10 Tf ( bold) Tj ET"),
        pair("plain", " bold", false)
    );
    // A subscript set lower in a smaller size continues its word; the next line does not.
    assert_eq!(
        joined("BT /F1 10 Tf 10 50 Td (H) Tj /F1 7 Tf -3 Ts (2) Tj ET"),
        pair("H", "2", true)
    );
    assert_eq!(
        joined("BT /F1 10 Tf 10 50 Td (up) Tj /F2 10 Tf 0 -12 Td (down) Tj ET"),
        pair("up", "down", false)
    );
}

#[test]
fn a_fill_change_starts_a_span_and_each_carries_its_fill() {
    let found = page_spans("BT /F1 10 Tf 10 50 Td 1 0 0 rg (red) Tj 0 g (black) Tj ET");
    let fills: Vec<_> = found
        .iter()
        .map(|s| (s.text.as_str(), s.fill.as_ref().and_then(|c| c.rgb)))
        .collect();
    assert_eq!(
        fills,
        [
            ("red", Some([1.0, 0.0, 0.0])),
            ("black", Some([0.0, 0.0, 0.0]))
        ]
    );
}

#[test]
fn text_its_rendering_mode_does_not_fill_has_no_fill() {
    let found = page_spans("BT /F1 10 Tf 10 50 Td 3 Tr (hidden) Tj 1 Tr 0 -12 Td (outlined) Tj ET");
    assert_eq!(
        found
            .iter()
            .map(|s| (s.text.as_str(), s.fill.is_none()))
            .collect::<Vec<_>>(),
        [("hidden", true), ("outlined", true)]
    );
}

#[test]
fn an_accent_set_over_a_letter_composes_with_it() {
    assert_eq!(
        texts(r"BT /F1 10 Tf 10 50 Td [(f) -111.5 (\310) 444.5 (ur)] TJ ET"),
        ["f\u{FC}r"]
    );
    assert_eq!(
        texts(r"BT /F1 10 Tf 10 50 Td [(Universit) -111.5 (\310) 444.5 (at)] TJ ET"),
        ["Universit\u{E4}t"]
    );
    // Drawn after its letter, back over it; over a dotless i, the i it is drawn for.
    assert_eq!(
        texts(r"BT /F1 10 Tf 10 50 Td [(u) 444.5 (\310) -111.5 (ber)] TJ ET"),
        ["\u{FC}ber"]
    );
    assert_eq!(
        texts(r"BT /F1 10 Tf 10 50 Td [(na) 27.5 (\310) 305.5 (\365ve)] TJ ET"),
        ["na\u{EF}ve"]
    );
}

#[test]
fn an_accent_raised_over_a_capital_composes_and_the_letter_s_box_spans_both() {
    // `K` is 667 wide: `O` starts at 16.67 and the accent, over its middle, at 18.895, raised by a quarter size.
    let found = page_spans(
        r"BT /F1 10 Tf 10 50 Td (K) Tj ET BT /F1 10 Tf 18.895 52.5 Td (\310) Tj ET BT /F1 10 Tf 16.67 50 Td (Oln) Tj ET",
    );
    let texts: Vec<&str> = found.iter().map(|s| s.text.as_str()).collect();
    assert_eq!(texts, ["K\u{D6}ln"]);
    let plain = page_spans(r"BT /F1 10 Tf 10 50 Td (KOln) Tj ET")[0].bbox;
    let bbox = found[0].bbox;
    assert!((bbox.y0 - (plain.y0 - 2.5)).abs() < 1e-9, "{bbox:?}"); // the raised accent's top
    assert!((bbox.y1 - plain.y1).abs() < 1e-9 && (bbox.x0 - plain.x0).abs() < 1e-9);
}

#[test]
fn an_accent_beside_or_far_above_a_letter_stays_an_accent() {
    assert_eq!(
        texts(r"BT /F1 10 Tf 10 50 Td (the \310 sign, a\310) Tj ET"),
        ["the \u{A8} sign, a\u{A8}"]
    );
    // Over a letter's advance but a size above its baseline: another line's.
    assert_eq!(
        texts(r"BT /F1 10 Tf 10 60 Td (\302) Tj ET BT /F1 10 Tf 8.885 50 Td (u) Tj ET"),
        ["\u{B4}", "u"]
    );
}
