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

/// Each span's text and whether it continues the run of the span before it.
fn runs(content: &str) -> Vec<(String, bool)> {
    page_spans(content)
        .into_iter()
        .map(|s| (s.text, s.continues_run))
        .collect()
}

#[test]
fn a_piece_a_fill_change_splits_off_continues_the_run_before_it() {
    let pieces = |a: &str, b: &str, c: &str| {
        vec![
            (a.to_string(), false),
            (b.to_string(), true),
            (c.to_string(), true),
        ]
    };
    assert_eq!(
        runs("BT /F1 10 Tf 10 50 Td (one) Tj 1 0 0 rg (two) Tj 0 g (three) Tj ET"),
        pieces("one", "two", "three")
    );
    // A blank piece in a fill of its own is left out; the run goes on through it.
    assert_eq!(
        runs("BT /F1 10 Tf 10 50 Td (one) Tj 1 0 0 rg ( ) Tj 0 0 1 rg (two) Tj ET"),
        [("one".to_string(), false), ("two".to_string(), true)]
    );
}

#[test]
fn a_fill_change_with_another_break_starts_a_run() {
    // The planted wrong truths: a fill change together with a font change, a new line, a far jump, or a blank
    // piece set apart before the next.
    for content in [
        "BT /F1 10 Tf 10 50 Td (one) Tj /F2 10 Tf 1 0 0 rg (two) Tj ET",
        "BT /F1 10 Tf 10 50 Td (one) Tj 1 0 0 rg 0 -12 Td (two) Tj ET",
        "BT /F1 10 Tf 10 50 Td (one) Tj 1 0 0 rg 100 0 Td (two) Tj ET",
        "BT /F1 10 Tf 10 50 Td (one) Tj 1 0 0 rg 100 0 Td ( ) Tj 0 0 1 rg (two) Tj ET",
    ] {
        assert_eq!(
            runs(content),
            [("one".to_string(), false), ("two".to_string(), false)],
            "{content}"
        );
    }
}

/// Each span's text and whether it is filled in the color painted under it.
fn unseen(content: &str) -> Vec<(String, bool)> {
    page_spans(content)
        .into_iter()
        .map(|s| (s.text, s.in_background_color))
        .collect()
}

#[test]
fn text_filled_in_the_color_under_it_shows_nothing() {
    let one = |text: &str, unseen: bool| vec![(text.to_string(), unseen)];
    // White on the white page.
    assert_eq!(
        unseen("BT 1 g /F1 10 Tf 10 50 Td (white) Tj ET"),
        one("white", true)
    );
    // White on a black box: the visible control.
    assert_eq!(
        unseen("0 g 0 0 200 100 re f BT 1 g /F1 10 Tf 10 50 Td (white) Tj ET"),
        one("white", false)
    );
    // Black on a black box closed twice (`re h`), its color stated in another space: compared as sRGB.
    assert_eq!(
        unseen("0 0 0 rg 0 0 200 100 re h f BT 0 g /F1 10 Tf 10 50 Td (black) Tj ET"),
        one("black", true)
    );
    // Black on the white page: text as it is mostly set.
    assert_eq!(
        unseen("BT /F1 10 Tf 10 50 Td (black) Tj ET"),
        one("black", false)
    );
}

#[test]
fn text_shows_where_anything_under_it_is_another_color() {
    let shown = |resources: &str, content: &str| {
        let file = File::open(
            common::one_page(resources, content, &[HELVETICA, BOLD]),
            b"",
        )
        .unwrap();
        let page = file.pages().unwrap().remove(0);
        let found = spans(&file.interpret(&page, false).unwrap().events);
        assert!(
            !found.is_empty() && found.iter().all(|s| !s.in_background_color),
            "{content}"
        );
    };
    // Black text half on a black box: the rest is on the white page.
    shown(
        FONTS,
        "0 g 0 0 20 100 re f BT /F1 10 Tf 10 50 Td (black) Tj ET",
    );
    // White text on a white box drawn over half a black one.
    shown(
        FONTS,
        "0 g 0 0 20 100 re f 1 g 15 0 185 100 re f BT 1 g /F1 10 Tf 10 50 Td (white) Tj ET",
    );
    // Black text on a white glyph on a black box (a caution sign's `!` on its triangle).
    shown(
        FONTS,
        "0 g 0 0 200 100 re f BT 1 g /F1 60 Tf 5 20 Td (M) Tj ET BT 0 g /F1 10 Tf 20 40 Td (!) Tj ET",
    );
    // White text stroked too (render mode 2): its outline shows.
    shown(FONTS, "BT 1 g 2 Tr /F1 10 Tf 10 50 Td (white) Tj ET");
    // Black text on a black box painted half transparent: grey under it.
    shown(
        "/Resources << /Font << /F1 5 0 R /F2 6 0 R >> /ExtGState << /G1 << /ca 0.5 >> >> >>",
        "q /G1 gs 0 g 0 0 200 100 re f Q BT 0 g /F1 10 Tf 10 50 Td (black) Tj ET",
    );
}

/// Black text set where the backdrops below are drawn around it.
const BLACK: &str = "BT 0 g /F1 10 Tf 50 45 Td (black) Tj ET";

/// Whether [`BLACK`] drawn over `backdrop` (painted in black) shows nothing, the backdrop made from the box the
/// text is set in, in user space: left, bottom, right, top.
fn hidden_by(backdrop: impl Fn(f64, f64, f64, f64) -> String) -> bool {
    let set = page_spans(BLACK).remove(0).bbox;
    let (x0, b, x1, t) = (set.x0, 100.0 - set.y1, set.x1, 100.0 - set.y0);
    let content = format!("0 g {} {BLACK}", backdrop(x0, b, x1, t));
    let found = page_spans(&content);
    assert_eq!(found.len(), 1, "{content}");
    found[0].in_background_color
}

/// A box `margin` around the text's, its corners rounded by `radius`: four curves.
fn rounded(text: [f64; 4], margin: f64, radius: f64) -> String {
    let [x0, y0, x1, y1] = text;
    let (west, east, south, north) = (x0 - margin, x1 + margin, y0 - margin, y1 + margin);
    let (wr, er, sr, nr) = (west + radius, east - radius, south + radius, north - radius);
    let arm = radius * 0.552; // a quarter circle's control arm
    let (wa, ea, sa, na) = (wr - arm, er + arm, sr - arm, nr + arm);
    format!(
        "{wr} {south} m {er} {south} l {ea} {south} {east} {sa} {east} {sr} c {east} {nr} l {east} {na} {ea} {north} \
         {er} {north} c {wr} {north} l {wa} {north} {west} {na} {west} {nr} c {west} {sr} l {west} {sa} {wa} {south} \
         {wr} {south} c h f"
    )
}

/// The page with a notch cut at the text's lower left corner, as one outline: the diagonal through the corner.
fn notched(x0: f64, b: f64) -> String {
    let (right, down, left, up) = (x0 + 2.0, b - 1.0, x0 - 1.0, b + 2.0);
    format!("0 0 m 0 100 l 200 100 l 200 0 l {right} 0 l {right} {down} l {left} {up} l 0 {up} l h")
}

#[test]
fn a_filled_region_or_clip_holding_the_text_whole_covers_it() {
    // A triangle around the box: no rectangle, still the text's whole backdrop.
    assert!(hidden_by(|_, _, _, _| "0 0 m 400 0 l 0 200 l h f".into()));
    // A box with rounded corners, its curves clear of the text.
    assert!(hidden_by(|x0, b, x1, t| rounded([x0, b, x1, t], 5.0, 4.0)));
    // A frame cut out by the even-odd rule, the text on its band, not in its hole.
    assert!(hidden_by(|_, _, _, _| {
        "0 0 200 100 re 150 10 20 20 re f*".into()
    }));
    // A box inside another, wound alike: nonzero fills the inner one too.
    assert!(hidden_by(|x0, b, x1, t| {
        let (w, h) = (x1 - x0 + 2.0, t - b + 2.0);
        format!("0 0 200 100 re {} {} {w} {h} re f", x0 - 1.0, b - 1.0)
    }));
    // The page filled inside a triangle clipping it, and inside a frame clipped by the even-odd rule.
    assert!(hidden_by(|_, _, _, _| {
        "q 0 0 m 400 0 l 0 200 l h W n 0 0 200 100 re f Q".into()
    }));
    assert!(hidden_by(|_, _, _, _| {
        "q 0 0 200 100 re 150 10 20 20 re W* n 0 0 200 100 re f Q".into()
    }));
}

#[test]
fn a_region_or_clip_leaving_any_of_the_text_out_does_not_cover_it() {
    // The planted wrong truths: each a backdrop of the test above, but for a part of the text's box.
    // The page with a notch cut at the text's corner.
    assert!(!hidden_by(|x0, b, _, _| format!("{} f", notched(x0, b))));
    // The same corner cut out by the even-odd rule: a triangle hole over it.
    assert!(!hidden_by(|x0, b, _, _| {
        let (right, down, left, up) = (x0 + 2.0, b - 1.0, x0 - 1.0, b + 2.0);
        format!("0 0 200 100 re {left} {down} m {right} {down} l {left} {up} l h f*")
    }));
    // A triangle whose long side runs through the text's middle.
    assert!(!hidden_by(|x0, b, x1, t| format!(
        "0 0 m {} 0 l 0 {} l h f",
        x0 + x1,
        b + t
    )));
    // A box inside another, by the even-odd rule: a hole the text sits in.
    assert!(!hidden_by(|x0, b, x1, t| {
        let (w, h) = (x1 - x0 + 2.0, t - b + 2.0);
        format!("0 0 200 100 re {} {} {w} {h} re f*", x0 - 1.0, b - 1.0)
    }));
    // A curve bowing into the text's corner where its chord passes well clear of it: the curve, not its chord,
    // bounds the region.
    assert!(!hidden_by(|x0, b, _, _| {
        let (start, bow, end) = (b + 5.0, (x0 + 8.0, b + 8.0), x0 + 5.0);
        format!(
            "0 {start} m {} {} {} {} {end} 0 c 200 0 l 200 100 l 0 100 l h f",
            bow.0, bow.1, bow.0, bow.1
        )
    }));
    // Rounded corners reaching into the text: their curves' boxes meet it.
    assert!(!hidden_by(|x0, b, x1, t| rounded([x0, b, x1, t], 1.0, 4.0)));
    // The page filled inside a clip notched at the text's corner, and inside one with a hole on the text.
    assert!(!hidden_by(|x0, b, _, _| format!(
        "q {} W n 0 0 200 100 re f Q",
        notched(x0, b)
    )));
    assert!(!hidden_by(|x0, b, _, _| format!(
        "q 0 0 200 100 re {} {} 2 2 re W* n 0 0 200 100 re f Q",
        x0 + 1.0,
        b + 1.0
    )));
}
