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
    let content = file.interpret(&page, false).unwrap();
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
    // An ordering without Adobe's table, a registry that is not Adobe's, and a CMap name outside ISO 32000-2's
    // predefined ones (its codes are not known CIDs): no text, never a guess.

    for (collection, encoding) in [
        ("Adobe-Identity", "/Identity-H"),
        ("Acme-Japan1", "/Identity-H"),
        ("Adobe-Japan1", "/UniJIS-UTF8-H"),
    ] {
        assert_eq!(
            collection_text(collection, encoding, "034B", &[]),
            "\u{FFFD}",
            "{collection} {encoding}"
        );
    }
}

/// The text and glyph origins of `codes` (hex) in a non-embedded CID font whose codes are by the predefined `CMap`
/// `encoding`, its `CIDFont` stating `cid_font_extra` (widths, `/CIDSystemInfo`).
fn predefined_glyphs(encoding: &str, codes: &str, cid_font_extra: &str) -> (String, Vec<f64>) {
    let font = format!(
        "<< /Type /Font /Subtype /Type0 /BaseFont /STSong-Light-{encoding} /Encoding /{encoding} /DescendantFonts [6 0 R] >>"
    );
    let cid = format!(
        "<< /Type /Font /Subtype /CIDFontType0 /BaseFont /STSong-Light {cid_font_extra} >>"
    );
    let (events, _) = texts(
        &format!("BT /F1 10 Tf 0 50 Td <{codes}> Tj ET"),
        &font,
        &[&cid],
    );
    let x = events[0].glyphs.iter().map(|g| g.origin.x).collect();
    (text(&events[0]), x)
}

#[test]
fn a_predefined_cmap_selects_the_cids_its_widths_and_text_are_by() {
    // GBK-EUC-H maps <20> to CID 7716 and <21>-<7E> from CID 814 (so <3D> is 842), not to 1-95: the widths are
    // those CIDs' and the text the collection's (Adobe-GB1-UCS2 gives CID 7716, the half-width space, as U+2002).
    let (text, x) = predefined_glyphs(
        "GBK-EUC-H",
        "203D20C4E3",
        "/CIDSystemInfo << /Registry (Adobe) /Ordering (GB1) /Supplement 2 >> /DW 1000 /W [842 [563] 7716 [250]]",
    );
    assert_eq!(text, "\u{2002}=\u{2002}\u{4F60}");
    assert_eq!(x, [0.0, 2.5, 8.13, 10.63]);
}

#[test]
fn a_predefined_cmap_names_its_own_collection() {
    // `/H` (JIS X 0208 row and cell): <2422> is CID 843, and the CMap's collection (Adobe-Japan1) its text, though
    // the CIDFont states none; the vertical `/V` gives the ideographic comma <2122> its vertical CID, same text.
    assert_eq!(predefined_glyphs("H", "24222122", "").0, "\u{3042}\u{3001}");
    assert_eq!(predefined_glyphs("V", "2122", "").0, "\u{3001}");
    // A Unicode CMap's codes are the text, a surrogate pair four bytes; its CIDs give the widths.
    let (text, x) = predefined_glyphs("UniJIS-UTF16-H", "3042D840DC0B3042", "/W [843 [500]]");
    assert_eq!(text, "\u{3042}\u{2000B}\u{3042}");
    assert_eq!(x, [0.0, 5.0, 15.0]);
}

// --- vertical writing (ISO 32000-2 9.7.4.3, 9.4.4) ------------------------------------------------------------------

/// An `Identity-V` composite font whose descendant is object 6 and `ToUnicode` object 7: CIDs 1, 2, 3 read U+4E00, U+4E8C,
/// 1.
const VERTICAL: &str =
    "<< /Type /Font /Subtype /Type0 /BaseFont /Song-Identity-V /Encoding /Identity-V
    /DescendantFonts [6 0 R] /ToUnicode 7 0 R >>";
/// Its descendant: CID 3 is half an em wide and half an em tall (`/W2`'s w1y -500); CIDs 1 and 2 are the default.
const VERTICAL_CID: &str =
    "<< /Type /Font /Subtype /CIDFontType0 /BaseFont /Song /W [1 [1000 1000 500]]
    /W2 [3 [-500 250 880]] >>";

fn vertical_to_unicode() -> String {
    common::stream(
        "",
        "1 begincodespacerange <0000> <FFFF> endcodespacerange 3 beginbfchar <0001> <4E00> <0002> <4E8C> <0003> <0031> endbfchar",
    )
}

fn vertical(content: &str) -> TextEvent {
    let (mut events, warnings) = texts(content, VERTICAL, &[VERTICAL_CID, &vertical_to_unicode()]);
    assert_eq!(warnings, Vec::<String>::new());
    assert_eq!(events.len(), 1);
    events.remove(0)
}

/// Each glyph's (origin x, origin y, end y) in page space (y down).
fn stops(event: &TextEvent) -> Vec<(f64, f64, f64)> {
    event
        .glyphs
        .iter()
        .map(|g| (g.origin.x, g.origin.y, g.end.y))
        .collect()
}

fn all_close(found: &[(f64, f64, f64)], want: &[(f64, f64, f64)]) -> bool {
    found.len() == want.len()
        && found
            .iter()
            .zip(want)
            .all(|(a, b)| close(a.0, b.0) && close(a.1, b.1) && close(a.2, b.2))
}

#[test]
fn identity_v_glyphs_advance_down_by_their_vertical_displacement() {
    // From the vertical origin (100, 10): CIDs 1 and 2 by /DW2's default -1000, CID 3 by its /W2 -500.
    let event = vertical("BT /F1 10 Tf 1 0 0 1 100 90 Tm <000100020003> Tj ET");
    assert!(event.vertical);
    assert_eq!(text(&event), "\u{4E00}\u{4E8C}1");
    let found = stops(&event);
    let want = [
        (100.0, 10.0, 20.0),
        (100.0, 20.0, 30.0),
        (100.0, 30.0, 35.0),
    ];
    assert!(all_close(&found, &want), "{found:?}");
}

#[test]
fn a_vertical_glyph_box_is_an_em_across_its_column_and_its_advance_along_it() {
    let event = vertical("BT /F1 10 Tf 1 0 0 1 100 90 Tm <00010003> Tj ET");
    let boxes: Vec<_> = event
        .glyphs
        .iter()
        .map(|g| (g.bbox.x0, g.bbox.y0, g.bbox.x1, g.bbox.y1))
        .collect();
    let want = [(95.0, 10.0, 105.0, 20.0), (95.0, 20.0, 105.0, 25.0)];
    assert!(
        boxes
            .iter()
            .zip(&want)
            .all(|(a, b)| close(a.0, b.0) && close(a.1, b.1) && close(a.2, b.2) && close(a.3, b.3)),
        "{boxes:?}"
    );
}

#[test]
fn dw2_states_the_default_vertical_displacement() {
    let cid = "<< /Type /Font /Subtype /CIDFontType0 /BaseFont /Song /DW2 [880 -800] /W2 [3 3 -500 250 880] >>";
    let (events, _) = texts(
        "BT /F1 10 Tf 1 0 0 1 100 90 Tm <000100030002> Tj ET",
        VERTICAL,
        &[cid, &vertical_to_unicode()],
    );
    // CID 1 by /DW2's -800, CID 3 by /W2's range form, CID 2 by /DW2 again.
    let want = [
        (100.0, 10.0, 18.0),
        (100.0, 18.0, 23.0),
        (100.0, 23.0, 31.0),
    ];
    assert!(
        all_close(&stops(&events[0]), &want),
        "{:?}",
        stops(&events[0])
    );
}

#[test]
fn character_spacing_and_tj_numbers_move_vertically_and_horizontal_scaling_does_not() {
    // ty = (w1 - Tj/1000) * Tfs + Tc: Tc 1 shortens each step down by 1; Tz scales across only.
    let event = vertical("BT /F1 10 Tf 1 Tc 50 Tz 1 0 0 1 100 90 Tm <00010002> Tj ET");
    let want = [(100.0, 10.0, 20.0), (100.0, 19.0, 29.0)];
    assert!(all_close(&stops(&event), &want), "{:?}", stops(&event));
    let across = event.glyphs[0].bbox;
    assert!(
        close(across.x0, 97.5) && close(across.x1, 102.5),
        "{across:?}"
    );
    // A positive TJ number moves the next glyph down.
    let event = vertical("BT /F1 10 Tf 1 0 0 1 100 90 Tm [<0001> 500 <0002>] TJ ET");
    let want = [(100.0, 10.0, 20.0), (100.0, 25.0, 35.0)];
    assert!(all_close(&stops(&event), &want), "{:?}", stops(&event));
}

/// A one-byte embedded `CMap` (codes 0x20-0x7E are CIDs 1-95) whose dictionary has `dict` and whose program has
/// `program` before its mappings.
fn one_byte_cmap(dict: &str, program: &str) -> String {
    common::stream(
        &format!("/Type /CMap {dict}"),
        &format!(
            "{program} 1 begincodespacerange <00> <FF> endcodespacerange 1 begincidrange <20> <7E> 1 endcidrange"
        ),
    )
}

fn embedded_cmap_event(cmap: &str, content: &str) -> TextEvent {
    let font =
        "<< /Type /Font /Subtype /Type0 /BaseFont /Song /Encoding 6 0 R /DescendantFonts [7 0 R]
        /ToUnicode 8 0 R >>";
    let to_unicode = common::stream(
        "",
        "1 begincodespacerange <00> <FF> endcodespacerange 1 beginbfrange <20> <7E> <0020> endbfrange",
    );
    let (mut events, _) = texts(content, font, &[cmap, VERTICAL_CID, &to_unicode]);
    events.remove(0)
}

#[test]
fn an_embedded_cmap_is_vertical_by_its_wmode_and_word_spacing_moves_down_too() {
    let content = "BT /F1 10 Tf 2 Tw 1 0 0 1 100 90 Tm (a b) Tj ET";
    for cmap in [
        one_byte_cmap("/WMode 1", ""),
        one_byte_cmap("", "/WMode 1 def"),
    ] {
        let event = embedded_cmap_event(&cmap, content);
        assert!(event.vertical, "{cmap}");
        // Every CID by the default -1000; Tw shortens the step after the one-byte code 32 only.
        let want = [
            (100.0, 10.0, 20.0),
            (100.0, 20.0, 30.0),
            (100.0, 28.0, 38.0),
        ];
        assert!(all_close(&stops(&event), &want), "{:?}", stops(&event));
    }
    let event = embedded_cmap_event(&one_byte_cmap("", ""), content);
    assert!(!event.vertical);
}

#[test]
fn a_predefined_v_cmap_writes_vertically() {
    let font = "<< /Type /Font /Subtype /Type0 /BaseFont /STSong-Light /Encoding /UniGB-UCS2-V
        /DescendantFonts [6 0 R] >>";
    let cid = "<< /Type /Font /Subtype /CIDFontType0 /BaseFont /STSong-Light >>";
    let (events, _) = texts(
        "BT /F1 10 Tf 1 0 0 1 100 90 Tm <4E004E8C> Tj ET",
        font,
        &[cid],
    );
    assert!(events[0].vertical);
    assert_eq!(text(&events[0]), "\u{4E00}\u{4E8C}");
    let want = [(100.0, 10.0, 20.0), (100.0, 20.0, 30.0)];
    assert!(
        all_close(&stops(&events[0]), &want),
        "{:?}",
        stops(&events[0])
    );
}
