//! Text: each string's glyphs with their text, origin and box in page space, through every kind of font.

mod common;

use pdf::interpret::TextEvent;
use pdf::{Event, File};

const HELVETICA: &str = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>";

/// The text events of a one-page PDF (media box 200 x 100) whose font `/F1` is object 5.
fn texts(content: &str, font: &str, extra: &[&str]) -> (Vec<TextEvent>, Vec<String>) {
    let mut objects = vec![font];
    objects.extend_from_slice(extra);
    let file = File::open(
        common::one_page("/Resources << /Font << /F1 5 0 R >> >>", content, &objects),
        b"",
    )
    .unwrap();
    let page = file.pages().unwrap().remove(0);
    let content = file.interpret(&page).unwrap();
    let texts = content
        .events
        .into_iter()
        .filter_map(|e| {
            if let Event::Text(t) = e {
                Some(t)
            } else {
                None
            }
        })
        .collect();
    (texts, content.warnings)
}

fn text(event: &TextEvent) -> String {
    event
        .glyphs
        .iter()
        .map(|g| g.text.clone().unwrap_or_else(|| "\u{FFFD}".into()))
        .collect()
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-6
}

#[test]
fn a_standard_font_places_each_glyph_by_its_width() {
    let (events, _) = texts("BT /F1 10 Tf 1 0 0 1 10 50 Tm (AB) Tj ET", HELVETICA, &[]);
    let event = &events[0];
    assert_eq!(
        (text(event), event.font.name.as_str(), event.size),
        ("AB".to_string(), "Helvetica", 10.0)
    );
    let (a, b) = (&event.glyphs[0], &event.glyphs[1]);
    assert!(close(a.origin.x, 10.0) && close(a.origin.y, 50.0));
    assert!(close(b.origin.x, 16.67), "{}", b.origin.x);
    // Page y is down: the ascent (718) lies above the baseline, the descent (-207) below.
    assert!(
        close(a.bbox.y0, 50.0 - 7.18) && close(a.bbox.y1, 50.0 + 2.07),
        "{:?}",
        a.bbox
    );
    assert!(close(a.bbox.x1, 16.67));
}

#[test]
fn tj_adjustments_and_spacing_move_the_next_glyph() {
    let (events, _) = texts(
        "BT /F1 10 Tf 10 50 Td [(A) -1000 (B)] TJ ET",
        HELVETICA,
        &[],
    );
    assert!(close(events[0].glyphs[1].origin.x, 10.0 + 6.67 + 10.0));
    // Tc after every glyph, Tw after a space, Tz scaling it all horizontally.
    let (events, _) = texts(
        "BT /F1 10 Tf 1 Tc 2 Tw 50 Tz 10 50 Td (A A) Tj ET",
        HELVETICA,
        &[],
    );
    let x: Vec<f64> = events[0].glyphs.iter().map(|g| g.origin.x).collect();
    assert!(close(x[1], 10.0 + f64::midpoint(6.67, 1.0)), "{x:?}");
    assert!(close(x[2], x[1] + (2.78 + 1.0 + 2.0) * 0.5), "{x:?}");
}

#[test]
fn line_operators_move_by_the_leading() {
    let content = "BT /F1 10 Tf 12 TL 10 50 Td (A) Tj T* (B) Tj (C) ' 0 -12 TD (D) Tj ET";
    let (events, _) = texts(content, HELVETICA, &[]);
    let ys: Vec<f64> = events.iter().map(|e| e.glyphs[0].origin.y).collect();
    assert_eq!(ys, [50.0, 62.0, 74.0, 86.0]);
}

#[test]
fn differences_and_widths_override_the_base_encoding() {
    let font = "<< /Type /Font /Subtype /Type1 /BaseFont /CMR10 /FirstChar 65 /LastChar 66 /Widths [500 600]
        /Encoding << /Type /Encoding /BaseEncoding /WinAnsiEncoding /Differences [65 /alpha /uni2212] >> >>";
    let (events, _) = texts(r"BT /F1 10 Tf 0 50 Td (AB\351) Tj ET", font, &[]);
    assert_eq!(text(&events[0]), "\u{3B1}\u{2212}\u{E9}");
    assert!(close(events[0].glyphs[1].origin.x, 5.0));
}

#[test]
fn a_composite_font_reads_two_byte_codes_widths_and_to_unicode() {
    let font = "<< /Type /Font /Subtype /Type0 /BaseFont /ABCDEF+Cambria /Encoding /Identity-H
        /DescendantFonts [6 0 R] /ToUnicode 7 0 R >>";
    let cid = "<< /Type /Font /Subtype /CIDFontType2 /BaseFont /ABCDEF+Cambria /DW 1000 /W [3 [250] 10 11 500] >>";
    let cmap = common::stream(
        "",
        "1 begincodespacerange <0000> <FFFF> endcodespacerange 2 beginbfchar <0003> <0020> <000A> <0078> endbfchar 1 beginbfrange <000B> <000B> <0079> endbfrange",
    );
    let (events, _) = texts(
        "BT /F1 20 Tf 0 50 Td <000A0003000B0004> Tj ET",
        font,
        &[cid, &cmap],
    );
    let event = &events[0];
    assert_eq!(text(event), "x y\u{FFFD}");
    let x: Vec<f64> = event.glyphs.iter().map(|g| g.origin.x).collect();
    assert_eq!(x, [0.0, 10.0, 15.0, 25.0]);
}

#[test]
fn an_invisible_text_layer_is_kept_with_its_mode_and_size_follows_the_matrices() {
    let (events, _) = texts(
        "2 0 0 2 0 0 cm BT 3 Tr /F1 5 Tf 0 10 Td (A) Tj ET",
        HELVETICA,
        &[],
    );
    assert_eq!((events[0].render_mode, events[0].size), (3, 10.0));
}

#[test]
fn text_without_a_font_is_a_warning() {
    let (events, warnings) = texts("BT (A) Tj ET", HELVETICA, &[]);
    assert_eq!(events, Vec::new());
    assert_eq!(warnings.len(), 1);
}

#[test]
fn win_ansi_names_a_hyphen_and_spaces_not_their_latin_1_twins() {
    let font = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>";
    let (events, _) = texts(r"BT /F1 10 Tf 0 50 Td (a\255b c\240d) Tj ET", font, &[]);
    assert_eq!(text(&events[0]), "a-b c d"); // the glyphs `hyphen` and `space`: U+002D and U+0020
}

/// A non-embedded CID font on the collection `Registry-Ordering`, its codes by `encoding`: the text of `codes` (hex).
fn collection_text(collection: &str, encoding: &str, codes: &str, extra: &[&str]) -> String {
    let (registry, ordering) = collection.split_once('-').unwrap();
    let font = format!(
        "<< /Type /Font /Subtype /Type0 /BaseFont /Ryumin-Light-Identity-H /Encoding {encoding} /DescendantFonts [6 0 R] >>"
    );
    let cid = format!(
        "<< /Type /Font /Subtype /CIDFontType0 /BaseFont /Ryumin-Light
        /CIDSystemInfo << /Registry ({registry}) /Ordering ({ordering}) /Supplement 2 >> /DW 1000 >>"
    );
    let mut objects = vec![cid.as_str()];
    objects.extend_from_slice(extra);
    let (events, _) = texts(
        &format!("BT /F1 10 Tf 0 50 Td <{codes}> Tj ET"),
        &font,
        &objects,
    );
    text(&events[0])
}

#[test]
fn a_registry_collection_names_the_text_of_its_cids() {
    // Each CID's character as Adobe's cid2code.txt states it (cmap-resources 20231115, the UTF-32 column).
    for (collection, codes, expected) in [
        ("Adobe-Japan1", "034B0CD40E8A", "\u{3042}\u{65E5}\u{672C}"),
        ("Adobe-GB1", "11CF0ED3", "\u{4E2D}\u{6587}"),
        ("Adobe-CNS1", "02950A3736B2", "\u{4E2D}\u{570B}\u{2008A}"),
        ("Adobe-Korea1", "0CE004D6", "\u{D55C}\u{AE00}"),
        ("Adobe-KR", "0B130188", "\u{D55C}\u{AE00}"),
    ] {
        assert_eq!(
            collection_text(collection, "/Identity-H", codes, &[]),
            expected,
            "{collection}"
        );
    }
}

#[test]
fn an_embedded_cmap_selects_the_cids_the_collection_names() {
    let cmap = common::stream(
        "/Type /CMap",
        "1 begincodespacerange <00> <FF> endcodespacerange 1 begincidchar <41> 843 endcidchar",
    );
    assert_eq!(
        collection_text("Adobe-Japan1", "7 0 R", "41", &[&cmap]),
        "\u{3042}"
    );
}

#[test]
fn to_unicode_wins_over_the_collection() {
    let font =
        "<< /Type /Font /Subtype /Type0 /BaseFont /Ryumin-Light-Identity-H /Encoding /Identity-H
        /DescendantFonts [6 0 R] /ToUnicode 7 0 R >>";
    let cid = "<< /Type /Font /Subtype /CIDFontType0 /BaseFont /Ryumin-Light
        /CIDSystemInfo << /Registry (Adobe) /Ordering (Japan1) /Supplement 6 >> >>";
    let to_unicode = common::stream(
        "",
        "1 begincodespacerange <0000> <FFFF> endcodespacerange 1 beginbfchar <034B> <0041> endbfchar",
    );
    let (events, _) = texts(
        "BT /F1 10 Tf 0 50 Td <034B0CD4> Tj ET",
        font,
        &[cid, &to_unicode],
    );
    assert_eq!(text(&events[0]), "A\u{65E5}"); // the code ToUnicode lacks falls to the collection
}

#[test]
fn no_table_reads_cids_of_another_collection_or_unknown_cids() {
    // An ordering without Adobe's table, a registry that is not Adobe's, and a predefined CMap this reader does not
    // carry (its codes are not CIDs): no text, never a guess.
    for (collection, encoding) in [
        ("Adobe-Identity", "/Identity-H"),
        ("Acme-Japan1", "/Identity-H"),
        ("Adobe-Japan1", "/H"),
    ] {
        assert_eq!(
            collection_text(collection, encoding, "034B", &[]),
            "\u{FFFD}",
            "{collection} {encoding}"
        );
    }
}
