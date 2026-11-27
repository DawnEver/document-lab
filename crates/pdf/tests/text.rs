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
        (text(event), event.font.info.name.as_str(), event.size),
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
fn a_font_names_and_measures_itself_through_references() {
    let (events, _) = texts(
        "BT /F1 10 Tf 1 0 0 1 10 50 Tm (AB) Tj ET",
        "<< /Type /Font /Subtype 6 0 R /BaseFont 7 0 R /FontDescriptor 8 0 R >>",
        &[
            "/Type1",
            "/Helvetica",
            "<< /Type /FontDescriptor /FontName /Helvetica /Ascent 9 0 R /Descent 10 0 R /Flags 11 0 R /FontWeight 12 0 R >>",
            "900",
            "-100",
            "262176",
            "700",
        ],
    );
    let event = &events[0];
    let info = &event.font.info;
    assert_eq!(
        (text(event), info.name.as_str(), info.subtype.as_str()),
        ("AB".to_string(), "Helvetica", "Type1")
    );
    assert_eq!((info.flags, info.descriptor_weight), (262_176, Some(700.0)));
    // The standard font's widths, found by its name, and the descriptor's ascent and descent.
    let a = &event.glyphs[0];
    assert!(close(event.glyphs[1].origin.x, 16.67));
    assert!(
        close(a.bbox.y0, 50.0 - 9.0) && close(a.bbox.y1, 50.0 + 1.0),
        "{:?}",
        a.bbox
    );
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

/// The text of the one-byte `codes` (hex) in the simple font `/BaseFont /<name>` of `subtype`, whose `ToUnicode`
/// states `to_unicode` (`bfrange`/`bfchar` blocks; none when empty), with the descriptor flags `flags`.
fn simple_text(name: &str, subtype: &str, flags: u32, to_unicode: &str, codes: &str) -> String {
    let cmap = common::stream(
        "",
        &format!("1 begincodespacerange <00> <FF> endcodespacerange {to_unicode}"),
    );
    let reference = if to_unicode.is_empty() {
        ""
    } else {
        "/ToUnicode 7 0 R"
    };
    let font = format!(
        "<< /Type /Font /Subtype /{subtype} /BaseFont /{name} /FontDescriptor 6 0 R {reference} >>"
    );
    let descriptor = format!("<< /Type /FontDescriptor /FontName /{name} /Flags {flags} >>");
    let (events, _) = texts(
        &format!("BT /F1 10 Tf 0 50 Td <{codes}> Tj ET"),
        &font,
        &[&descriptor, &cmap],
    );
    text(&events[0])
}

/// A `ToUnicode` stating each code `c` as U+F000 + `c`: the (3,0) symbol `cmap` convention (ISO 32000-2 9.6.5.4).
const PRIVATE_USE: &str = "1 beginbfrange <00> <FF> <F000> endbfrange";

#[test]
fn a_symbol_fonts_private_use_text_reads_through_the_symbol_encoding() {
    // Omega, greaterequal, lessequal, copyrightsans, copyrightserif, parenlefttp, radicalex: the private-use code
    // less U+F000 is the Symbol code (Annex D.5); a corporate-use code Adobe's table gives for a glyph Unicode has
    // since encoded is that character (Apple's Symbol table); the radical extender, which Unicode does not encode,
    // stays as Adobe states it.
    for name in ["SymbolMT", "ABCDEF+SymbolMT", "Symbol", "Symbol,Bold"] {
        assert_eq!(
            simple_text(name, "TrueType", 4, PRIVATE_USE, "57B3A3E3D3E660"),
            "\u{2126}\u{2265}\u{2264}\u{A9}\u{A9}\u{239B}\u{F8E5}",
            "{name}"
        );
    }
}

#[test]
fn a_dingbat_fonts_private_use_text_reads_through_its_table() {
    for (name, expected) in [
        // Wingdings: black small square, check mark, black medium square, a space; w-1255 (code FF, not encoded in
        // Unicode) stays as stated, as does a code a font has no glyph for (Wingdings 2 and 3 end before FC).
        ("Wingdings", "\u{25AA}\u{2713}\u{25FC} \u{F0FF}"),
        (
            "ABCDEF+Wingdings-Regular",
            "\u{25AA}\u{2713}\u{25FC} \u{F0FF}",
        ),
        ("Wingdings2", "\u{1F794}\u{F0FC}\u{2464} \u{F0FF}"),
        ("Wingdings#202", "\u{1F794}\u{F0FC}\u{2464} \u{F0FF}"),
        ("Wingdings3", "\u{1F824}\u{F0FC}\u{1F858} \u{F0FF}"),
        ("Webdings", "\u{1F56E}\u{1F30D}\u{2B24} \u{1F54A}"),
    ] {
        assert_eq!(
            simple_text(name, "TrueType", 4, PRIVATE_USE, "A7FC6E20FF"),
            expected,
            "{name}"
        );
    }
    // ZapfDingbats: a20 (heavy check mark), a89 (Adobe's corporate use; U+2768 since Unicode 3.2).
    assert_eq!(
        simple_text("ZapfDingbats", "Type1", 4, PRIVATE_USE, "3480"),
        "\u{2714}\u{2768}"
    );
}

#[test]
fn a_stated_character_wins_and_other_fonts_private_use_stays() {
    // A ToUnicode stating a real character is read as stated, in a symbol font too.
    let stated = "2 beginbfchar <57> <03A9> <61> <0061> endbfchar";
    assert_eq!(
        simple_text("SymbolMT", "TrueType", 4, stated, "5761"),
        "\u{3A9}a"
    );
    // A font other than the symbol fonts keeps its private-use text: nothing says what it shows.
    assert_eq!(
        simple_text("ArialMT", "TrueType", 4, PRIVATE_USE, "41FC"),
        "\u{F041}\u{F0FC}"
    );
    assert_eq!(
        simple_text("WingdingsLookalike", "TrueType", 4, PRIVATE_USE, "FC"),
        "\u{F0FC}"
    );
}

#[test]
fn a_symbol_font_without_text_reads_its_codes_through_its_own_encoding() {
    // A symbolic TrueType font, not embedded, no ToUnicode: its codes select glyphs through its (3,0) cmap, which
    // for these fonts is their encoding.
    assert_eq!(
        simple_text("Wingdings", "TrueType", 4, "", "A7FC"),
        "\u{25AA}\u{2713}"
    );
    assert_eq!(
        simple_text("SymbolMT", "TrueType", 4, "", "57E3"),
        "\u{2126}\u{A9}"
    );
    // A Type 1 font named for Symbol, without an encoding: Symbol's built-in one, its corporate-use codes resolved.
    assert_eq!(
        simple_text("SymbolMT", "Type1", 4, "", "61E3E6"),
        "\u{3B1}\u{A9}\u{239B}"
    );
    assert_eq!(
        simple_text("Symbol", "Type1", 4, "", "61E3E6"),
        "\u{3B1}\u{A9}\u{239B}"
    );
}

#[test]
fn a_glyph_name_adobe_gave_a_corporate_use_code_reads_as_the_character_unicode_since_encoded() {
    // A TeX extension font names its bracket pieces as Symbol does; the Adobe Glyph List gives them U+F8xx, Unicode
    // 3.2 their own characters. A `uniF8EB` name states the private-use code itself, which stays.
    let font = "<< /Type /Font /Subtype /Type1 /BaseFont /CMEX10
        /Encoding << /Type /Encoding /Differences [65 /parenlefttp /bracketleftex /integralex /uniF8EB] >> >>";
    let (events, _) = texts("BT /F1 10 Tf 0 50 Td (ABCD) Tj ET", font, &[]);
    assert_eq!(text(&events[0]), "\u{239B}\u{23A2}\u{23AE}\u{F8EB}");
}

/// A TrueType program, as ASCII hex, holding only a `name` table whose PostScript name (name ID 6, Windows Unicode)
/// is `name`.
fn sfnt_named(name: &str) -> String {
    let utf16: Vec<u8> = name.encode_utf16().flat_map(u16::to_be_bytes).collect();
    let mut table = Vec::new();
    for v in [
        0u16,
        1,
        18,
        3,
        1,
        0x409,
        6,
        u16::try_from(utf16.len()).unwrap(),
        0,
    ] {
        table.extend(v.to_be_bytes());
    }
    table.extend(&utf16);
    let mut data = Vec::new();
    data.extend(0x0001_0000u32.to_be_bytes());
    for v in [1u16, 16, 0, 0] {
        data.extend(v.to_be_bytes());
    }
    data.extend(b"name");
    for v in [0u32, 28, u32::try_from(table.len()).unwrap()] {
        data.extend(v.to_be_bytes());
    }
    data.extend(table);
    let mut hex: String = data
        .iter()
        .flat_map(|b| [b >> 4, b & 15].map(|n| char::from_digit(u32::from(n), 16).unwrap_or('0')))
        .collect();
    hex.push('>');
    hex
}

/// The text of `<codes>` in a composite font `/BaseFont /<base>` whose embedded TrueType program names itself
/// `program`, its `ToUnicode` stating each code as U+F000 + code.
fn embedded_text(base: &str, program: &str, codes: &str) -> String {
    let font = format!(
        "<< /Type /Font /Subtype /Type0 /BaseFont /{base} /Encoding /Identity-H /DescendantFonts [6 0 R]
        /ToUnicode 7 0 R >>"
    );
    let cid =
        format!("<< /Type /Font /Subtype /CIDFontType2 /BaseFont /{base} /FontDescriptor 8 0 R >>");
    let cmap = common::stream(
        "",
        "1 begincodespacerange <0000> <FFFF> endcodespacerange 1 beginbfrange <0000> <00FF> <F000> endbfrange",
    );
    let descriptor =
        format!("<< /Type /FontDescriptor /FontName /{base} /Flags 4 /FontFile2 9 0 R >>");
    let program = common::stream("/Filter /ASCIIHexDecode", &sfnt_named(program));
    let (events, _) = texts(
        &format!("BT /F1 10 Tf 0 50 Td <{codes}> Tj ET"),
        &font,
        &[&cid, &cmap, &descriptor, &program],
    );
    text(&events[0])
}

/// The text of `(aW)` in a simple font of `subtype` named `name`, under `/Encoding /WinAnsiEncoding`, embedding a
/// TrueType program that names itself `name` when `embedded`.
fn win_ansi_text(name: &str, subtype: &str, embedded: bool) -> String {
    let font = format!(
        "<< /Type /Font /Subtype /{subtype} /BaseFont /{name} /Encoding /WinAnsiEncoding /FontDescriptor 6 0 R >>"
    );
    let file = if embedded { "/FontFile2 7 0 R" } else { "" };
    let descriptor = format!("<< /Type /FontDescriptor /FontName /{name} /Flags 4 {file} >>");
    let program = common::stream("/Filter /ASCIIHexDecode", &sfnt_named(name));
    let (events, _) = texts(
        "BT /F1 10 Tf 0 50 Td (aW) Tj ET",
        &font,
        &[&descriptor, &program],
    );
    text(&events[0])
}

#[test]
fn a_symbol_font_not_embedded_reads_as_the_glyphs_it_draws_whatever_base_encoding_is_named() {
    // D121: a Symbol embedding no program is drawn by its built-in encoding, as viewers draw it, and its text
    // follows: alpha and Omega, not the Latin `aW` that WinAnsiEncoding names. So for Zapf Dingbats.
    for (name, subtype) in [("Symbol", "Type1"), ("SymbolMT", "TrueType")] {
        assert_eq!(
            win_ansi_text(name, subtype, false),
            "\u{3B1}\u{2126}",
            "{name}"
        );
    }
    assert_eq!(
        win_ansi_text("ZapfDingbats", "Type1", false),
        "\u{2741}\u{2737}"
    );
    // The planted control: one that embeds its program is drawn by the encoding named, and reads as it.
    assert_eq!(win_ansi_text("SymbolMT", "TrueType", true), "aW");
    assert_eq!(win_ansi_text("Symbol", "Type1", true), "aW");
}

#[test]
fn a_symbol_font_under_a_generic_name_is_known_by_its_programs_own_name() {
    // A producer's `CIDFont+F7` whose program names itself SymbolMT: Omega and greaterequal.
    assert_eq!(
        embedded_text("CIDFont+F7", "SymbolMT", "005700B3"),
        "\u{2126}\u{2265}"
    );
    // A program naming itself anything else says nothing of what its private-use codes show.
    assert_eq!(
        embedded_text("CIDFont+F5", "ArialMT", "005700B3"),
        "\u{F057}\u{F0B3}"
    );
}

#[test]
fn variation_selectors_a_to_unicode_states_are_kept() {
    let font = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /ToUnicode 6 0 R >>";
    let cmap = common::stream(
        "",
        "1 begincodespacerange <00> <FF> endcodespacerange 2 beginbfchar <31> <0031FE0E> <32> <8FBADB40DD00> endbfchar",
    );
    let (events, _) = texts("BT /F1 10 Tf 0 50 Td (12) Tj ET", font, &[&cmap]);
    assert_eq!(text(&events[0]), "1\u{FE0E}\u{8FBA}\u{E0100}"); // as stated, not stripped: numbers strips them
}

/// A Type 1 font named `/ABCDEF+F1` (no weight word) embedding the bare CFF program whose bytes are `hex`.
fn cff_font(hex: &str) -> [String; 3] {
    [
        "<< /Type /Font /Subtype /Type1 /BaseFont /ABCDEF+F1 /FontDescriptor 6 0 R >>".to_string(),
        "<< /Type /FontDescriptor /FontName /ABCDEF+F1 /Flags 32 /FontFile3 7 0 R >>".to_string(),
        common::stream(
            "/Subtype /Type1C /Filter /ASCIIHexDecode",
            &format!("{hex}>"),
        ),
    ]
}

#[test]
fn a_cff_program_s_weight_string_makes_its_font_bold() {
    // Header; Name INDEX (`F1`); Top DICT INDEX (Weight: standard string 384, `Bold`); an empty String INDEX.
    let [font, descriptor, program] =
        cff_font("01000401 0001 01 01 03 4631 0001 01 01 05 1C018004 0000");
    let (events, warnings) = texts(
        "BT /F1 10 Tf 0 50 Td (A) Tj ET",
        &font,
        &[&descriptor, &program],
    );
    assert_eq!(
        events[0].font.info.program_weight,
        Some(700.0),
        "{warnings:?}"
    );
    assert!(events[0].font.info.bold());
}

#[test]
fn a_malformed_cff_program_is_a_warning_and_its_text_still_reads() {
    let [font, descriptor, program] = cff_font("02000401 0001 01 01 03 4631");
    let (events, warnings) = texts(
        "BT /F1 10 Tf 0 50 Td (A) Tj ET",
        &font,
        &[&descriptor, &program],
    );
    assert_eq!(text(&events[0]), "A");
    assert!(!events[0].font.info.bold());
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(
        warnings[0].contains("/F1 (ABCDEF+F1)") && warnings[0].contains("CFF"),
        "{warnings:?}"
    );
}

/// The width across a glyph's outline, in ems.
fn outline_width(event: &TextEvent) -> f64 {
    let pdf::shape::Shape::Outline(outline) = event.font.shape(event.glyphs[0].code) else {
        panic!("no outline");
    };
    let xs: Vec<f64> = outline
        .iter()
        .filter_map(|s| match s {
            pdf::geometry::Segment::MoveTo(p)
            | pdf::geometry::Segment::LineTo(p)
            | pdf::geometry::Segment::CurveTo(_, _, p) => Some(p.x),
            pdf::geometry::Segment::Close => None,
        })
        .collect();
    xs.iter().copied().fold(f64::MIN, f64::max) - xs.iter().copied().fold(f64::MAX, f64::min)
}

#[test]
fn a_font_not_embedded_is_drawn_by_a_substitute_stretched_to_its_widths_unless_standard() {
    let standard = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /FirstChar 72 /LastChar 72 /Widths [1444] >>";
    let other = "<< /Type /Font /Subtype /TrueType /BaseFont /Calibri /FirstChar 72 /LastChar 72 /Widths [1444] \
                 /FontDescriptor << /Type /FontDescriptor /FontName /Calibri /Flags 32 >> >>";
    let (helvetica, _) = texts("BT /F1 10 Tf (H) Tj ET", standard, &[]);
    let (calibri, _) = texts("BT /F1 10 Tf (H) Tj ET", other, &[]);
    // Helvetica's H is 722 units wide: a standard font's substitute is as it is; another's is stretched to the
    // width the file states for it, twice the substitute's own.
    let ratio = outline_width(&calibri[0]) / outline_width(&helvetica[0]);
    assert!((ratio - 1444.0 / 722.0).abs() < 1e-6, "{ratio}");
}

/// A Foxit program, here a stand-in for a fallback a test serves.
fn foxit(name: &str) -> Vec<u8> {
    std::fs::read(format!(
        "{}/src/fonts/{name}.cff",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap()
}

/// The shape of each glyph `(codes)` shows in the font `/F1` (with `extra` objects 6, 7, ...), its fallbacks
/// fetched through `served` (a fallback's name to the Foxit program served for it, else a failure), and each name
/// asked for, in order. No fetch at all when `served` is None.
fn fallback_shapes(
    font: &str,
    extra: &[&str],
    codes: &str,
    served: Option<&'static [(&'static str, &'static str)]>,
) -> (Vec<pdf::shape::Shape>, Vec<String>) {
    let mut objects = vec![font];
    objects.extend_from_slice(extra);
    let file = File::open(
        common::one_page(
            "/Resources << /Font << /F1 5 0 R >> >>",
            &format!("BT /F1 10 Tf 0 50 Td <{codes}> Tj ET"),
            &objects,
        ),
        b"",
    )
    .unwrap();
    let asked = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    if let Some(served) = served {
        let log = std::rc::Rc::clone(&asked);
        file.fetch_fallbacks(std::rc::Rc::new(move |name: &str| {
            log.borrow_mut().push(name.to_string());
            served
                .iter()
                .find(|(n, _)| *n == name)
                .map(|(_, program)| foxit(program))
                .ok_or_else(|| format!("{name} cannot be had: offline"))
        }));
    }
    let page = file.pages().unwrap().remove(0);
    let content = file.interpret(&page, false).unwrap();
    let shapes = content
        .events
        .iter()
        .filter_map(|e| match e {
            Event::Text(t) => Some(t),
            _ => None,
        })
        .flat_map(|t| t.glyphs.iter().map(|g| t.font.shape(g.code)))
        .collect();
    let asked = asked.borrow().clone();
    (shapes, asked)
}

/// A composite font `/BaseFont /<name>` embedding no program, `Identity-H` over `CID`s of Adobe's `ordering`, its
/// descriptor's flags `flags`: the font and its descendant and descriptor (objects 6 and 7).
fn cjk_font(name: &str, ordering: &str, flags: u32) -> [String; 3] {
    [
        format!(
            "<< /Type /Font /Subtype /Type0 /BaseFont /{name} /Encoding /Identity-H /DescendantFonts [6 0 R] >>"
        ),
        format!(
            "<< /Type /Font /Subtype /CIDFontType0 /BaseFont /{name} /FontDescriptor 7 0 R \
             /CIDSystemInfo << /Registry (Adobe) /Ordering ({ordering}) /Supplement 0 >> >>"
        ),
        format!("<< /Type /FontDescriptor /FontName /{name} /Flags {flags} >>"),
    ]
}

fn outlined(shape: &pdf::shape::Shape) -> bool {
    matches!(shape, pdf::shape::Shape::Outline(o) if !o.is_empty())
}

fn missing(shape: &pdf::shape::Shape) -> String {
    match shape {
        pdf::shape::Shape::Missing(why) => why.to_string(),
        other => panic!("drawn: {other:?}"),
    }
}

#[test]
fn a_cjk_font_not_embedded_is_drawn_from_its_collections_fallback_fetched_once() {
    // GB1's CIDs 34 and 35 are `A` and `B` (served by FoxitSans for the test): the fallback is fetched once, when
    // the first glyph needs it; sans or serif by the descriptor's flags, as MuPDF chooses.
    for (ordering, flags, wanted) in [
        ("GB1", 4, "noto-sans-sc"),
        ("GB1", 6, "noto-serif-sc"),
        ("CNS1", 4, "noto-sans-tc"),
        ("Japan1", 4, "noto-sans-jp"),
        ("Korea1", 6, "noto-serif-kr"),
    ] {
        let [font, cid, descriptor] = cjk_font("FZHTJW--GB1-0", ordering, flags);
        let served: &'static [(&str, &str)] = Box::leak(Box::new([(wanted, "FoxitSans")]));
        let (shapes, asked) =
            fallback_shapes(&font, &[&cid, &descriptor], "00220023", Some(served));
        assert_eq!(asked, [wanted], "{ordering} {flags}");
        if ordering == "GB1" {
            assert!(shapes.iter().all(outlined), "{shapes:?}");
        }
    }
}

#[test]
fn a_fallback_that_cannot_be_had_is_missing_by_name_and_never_guessed() {
    let [font, cid, descriptor] = cjk_font("FZS3JW--GB1-0", "GB1", 6);
    let (shapes, asked) = fallback_shapes(&font, &[&cid, &descriptor], "00220023", Some(&[]));
    assert_eq!(asked, ["noto-serif-sc"]); // asked once: the failure is kept while the file is open
    assert_eq!(
        missing(&shapes[1]),
        "font FZS3JW--GB1-0 is not embedded and its fallback noto-serif-sc cannot be had: offline"
    );
    // Nothing fetches: missing too, saying so.
    let (shapes, _) = fallback_shapes(&font, &[&cid, &descriptor], "0022", None);
    assert!(missing(&shapes[0]).contains("nothing fetches fallback fonts here"));
}

#[test]
fn a_microsoft_symbol_font_not_embedded_is_drawn_from_the_first_fallback_holding_its_character() {
    // Wingdings 2's 0x95 is U+22C5 (dot operator): not in the first fallback (FoxitSans served), in the second
    // (FoxitSymbol's dotmath, and its space); the third is never fetched. A character none holds is missing, naming
    // the fallbacks.
    let font = "<< /Type /Font /Subtype /TrueType /BaseFont /Wingdings2 /FirstChar 32 /LastChar 149 \
                /FontDescriptor << /Type /FontDescriptor /FontName /Wingdings2 /Flags 4 >> >>";
    let served = &[
        ("noto-sans-symbols-2", "FoxitSans"),
        ("noto-sans-symbols", "FoxitSymbol"),
        ("noto-sans-math", "FoxitSans"),
    ];
    let (shapes, asked) = fallback_shapes(font, &[], "9520", Some(served));
    assert!(outlined(&shapes[0]), "{shapes:?}");
    assert_eq!(shapes[1], pdf::shape::Shape::Outline(std::rc::Rc::from([])));
    assert_eq!(asked, ["noto-sans-symbols-2", "noto-sans-symbols"]);
    let (shapes, asked) = fallback_shapes(font, &[], "A7", Some(served)); // U+1F794, in none
    assert_eq!(asked.len(), 3);
    assert_eq!(
        missing(&shapes[0]),
        "font Wingdings2 is not embedded and a character it shows is in none of its fallbacks \
         (noto-sans-symbols-2, noto-sans-symbols, noto-sans-math)"
    );
}

#[test]
fn a_font_with_its_program_or_a_substitute_or_no_collection_fetches_nothing() {
    // The planted controls: an embedded program, a standard font's substitute, and a composite font whose CIDs are
    // of no collection known (Adobe-Identity) never ask for a fallback; the last stays missing as before.
    let served = &[("noto-sans-sc", "FoxitSans")];
    let embedded = "<< /Type /Font /Subtype /Type1 /BaseFont /F /FontDescriptor << /Type /FontDescriptor \
                    /FontName /F /Flags 32 /FontFile3 6 0 R >> >>";
    let program = common::hex_stream("/Subtype /Type1C", &foxit("FoxitSans"));
    let (shapes, asked) = fallback_shapes(embedded, &[&program], "41", Some(served));
    assert!(outlined(&shapes[0]) && asked == Vec::<String>::new());
    let (shapes, asked) = fallback_shapes(HELVETICA, &[], "41", Some(served));
    assert!(outlined(&shapes[0]) && asked == Vec::<String>::new());
    let [font, cid, descriptor] = cjk_font("F", "Identity", 4);
    let (shapes, asked) = fallback_shapes(&font, &[&cid, &descriptor], "0022", Some(served));
    assert_eq!(
        missing(&shapes[0]),
        "font F is not embedded and has no substitute"
    );
    assert_eq!(asked, Vec::<String>::new());
}
