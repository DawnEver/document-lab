//! Marked content: what optional content a default view hides draws is reported in a hidden layer; `/ActualText`
//! replaces the text of the glyphs it spans.

mod common;

use pdf::spans::spans;
use pdf::{Content, Event, File};

const HELVETICA: &str = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>";

/// What the one page of a PDF draws: its catalog carries `ocproperties`, its resources `resources` (the font
/// `/F1` is object 5, the groups `On` and `Off` objects 6 and 7), and `extra` objects follow as 8, 9, ...
fn page(ocproperties: &str, resources: &str, content: &str, extra: &[&str]) -> Content {
    let catalog = format!("<< /Type /Catalog /Pages 2 0 R {ocproperties} >>");
    let page = format!(
        "<< /Type /Page /Parent 2 0 R /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> {resources} >> >>"
    );
    let content = common::stream("", content);
    let mut objects = vec![
        catalog.as_str(),
        "<< /Type /Pages /Kids [3 0 R] /Count 1 /MediaBox [0 0 200 100] >>",
        &page,
        &content,
        HELVETICA,
        "<< /Type /OCG /Name (On) >>",
        "<< /Type /OCG /Name (Off) >>",
    ];
    objects.extend_from_slice(extra);
    let file = File::open(common::pdf(&objects), b"").unwrap();
    let page = file.pages().unwrap().remove(0);
    file.interpret(&page, false).unwrap()
}

/// The default configuration: `On` on, `Off` off.
const OC: &str = "/OCProperties << /OCGs [6 0 R 7 0 R] /D << /OFF [7 0 R] >> >>";
/// The two groups as marked-content properties.
const PROPERTIES: &str = "/Properties << /On 6 0 R /Off 7 0 R >>";

/// The text of the glyphs in the layer `hidden` says, glyph by glyph.
fn text_in(content: &Content, hidden: bool) -> String {
    content
        .events
        .iter()
        .filter_map(|e| match e {
            Event::Text(t) if t.layer.hidden == hidden => Some(
                t.glyphs
                    .iter()
                    .map(|g| g.text.clone().unwrap_or_else(|| "\u{FFFD}".into()))
                    .collect::<String>(),
            ),
            _ => None,
        })
        .collect()
}

/// The text shown, glyph by glyph.
fn shown(content: &Content) -> String {
    text_in(content, false)
}

/// Each event's kind, `hidden` before it when optional content hides it.
fn kinds(content: &Content) -> Vec<String> {
    content
        .events
        .iter()
        .map(|e| {
            let (kind, layer) = match e {
                Event::Path(p) => ("path", &p.layer),
                Event::Image(i) => ("image", &i.layer),
                Event::Text(t) => ("text", &t.layer),
                Event::Shading(s) => ("shading", &s.layer),
            };
            if layer.hidden {
                format!("hidden {kind}")
            } else {
                kind.to_string()
            }
        })
        .collect()
}

#[test]
fn what_a_group_that_is_off_draws_is_reported_hidden() {
    let content = page(
        OC,
        PROPERTIES,
        "/OC /Off BDC 0 0 10 10 re f BT /F1 10 Tf 10 50 Td (hidden) Tj ET EMC \
         /OC /On BDC BT /F1 10 Tf 10 20 Td (seen) Tj ET EMC 20 20 5 5 re f",
        &[],
    );
    assert_eq!(
        (shown(&content), text_in(&content, true)),
        ("seen".into(), "hidden".into())
    );
    assert_eq!(
        kinds(&content),
        ["hidden path", "hidden text", "text", "path"]
    );
    assert_eq!(content.warnings, Vec::<String>::new());
}

#[test]
fn a_base_state_off_hides_every_group_not_turned_on() {
    let oc = "/OCProperties << /OCGs [6 0 R 7 0 R] /D << /BaseState /OFF /ON [6 0 R] >> >>";
    let content = page(
        oc,
        PROPERTIES,
        "BT /F1 10 Tf 10 50 Td /OC /On BDC (a) Tj EMC /OC /Off BDC (b) Tj EMC ET",
        &[],
    );
    assert_eq!(
        (shown(&content), text_in(&content, true)),
        ("a".into(), "b".into())
    );
}

#[test]
fn a_membership_dictionary_applies_its_policy_or_its_expression() {
    let cases = [
        ("/P /AllOn", false),
        ("/P /AnyOn", true),
        ("", true), // AnyOn is the default policy
        ("/P /AnyOff", true),
        ("/P /AllOff", false),
        ("/P /AllOn /VE [/Not 7 0 R]", true), // an expression overrides the policy
        ("/VE [/And 6 0 R 7 0 R]", false),
        ("/VE [/Or 7 0 R [/Not 7 0 R]]", true),
    ];
    for (entries, shows) in cases {
        let ocmd = format!("<< /Type /OCMD /OCGs [6 0 R 7 0 R] {entries} >>");
        let content = page(
            OC,
            "/Properties << /M 8 0 R >>",
            "BT /F1 10 Tf 10 50 Td /OC /M BDC (x) Tj EMC ET",
            &[&ocmd],
        );
        assert_eq!(text_in(&content, !shows), "x", "{entries}");
    }
}

#[test]
fn hidden_content_ends_at_its_own_emc_across_nesting_and_q() {
    let content = page(
        OC,
        PROPERTIES,
        "BT /F1 10 Tf 10 50 Td \
         /OC /Off BDC (a) Tj /Span BMC (b) Tj EMC (c) Tj q Q /P << /MCID 0 >> BDC (d) Tj EMC (e) Tj EMC \
         (f) Tj ET \
         q /OC /Off BDC Q 0 0 5 5 re f EMC 0 0 5 5 re f",
        &[],
    );
    assert_eq!(
        (shown(&content), text_in(&content, true)),
        ("f".into(), "abcde".into())
    );
    assert_eq!(&kinds(&content)[5..], ["text", "hidden path", "path"]);
    // A span never joins hidden glyphs to shown ones, though they stand on one baseline.
    let read: Vec<(String, bool)> = spans(&content.events)
        .into_iter()
        .map(|s| (s.text, s.layer.hidden))
        .collect();
    assert_eq!(read, [("abcde".into(), true), ("f".into(), false)]);
}

#[test]
fn hidden_content_still_moves_the_text_and_sets_the_clip() {
    let content = page(
        OC,
        PROPERTIES,
        "BT /F1 10 Tf 10 50 Td /OC /Off BDC (ab) Tj EMC (c) Tj ET \
         /OC /Off BDC 0 0 50 50 re W n EMC 100 80 5 5 re f 10 10 5 5 re f",
        &[],
    );
    let Event::Text(c) = &content.events[1] else {
        panic!("{:?}", content.events)
    };
    // `c` stands after the hidden `ab`: 10 + (556 + 556) / 100.
    assert!((c.glyphs[0].origin.x - 21.12).abs() < 1e-6, "{c:?}");
    assert_eq!(kinds(&content), ["hidden text", "text", "path"]); // the hidden clip still cuts the first square
}

#[test]
fn what_a_form_or_an_image_whose_group_is_off_draws_is_hidden() {
    let form_off = common::stream(
        "/Type /XObject /Subtype /Form /BBox [0 0 50 50] /OC 7 0 R",
        "0 0 9 9 re f",
    );
    let form_on = common::stream(
        "/Type /XObject /Subtype /Form /BBox [0 0 50 50] /OC 6 0 R",
        "0 0 9 9 re f",
    );
    let image = common::stream(
        "/Type /XObject /Subtype /Image /Width 1 /Height 1 /ColorSpace /DeviceGray /BitsPerComponent 8 /OC 7 0 R",
        "\u{0}",
    );
    let content = page(
        OC,
        "/XObject << /A 8 0 R /B 9 0 R /I 10 0 R >>",
        "/A Do /B Do 10 0 0 10 50 50 cm /I Do 0 0 1 1 re f",
        &[&form_off, &form_on, &image],
    );
    assert_eq!(
        kinds(&content),
        ["hidden path", "path", "hidden image", "path"]
    );
}

#[test]
fn content_inside_a_form_inside_hidden_content_is_hidden() {
    let form = common::stream(
        "/Type /XObject /Subtype /Form /BBox [0 0 50 50]",
        "0 0 9 9 re f",
    );
    let content = page(
        OC,
        &format!("{PROPERTIES} /XObject << /A 8 0 R >>"),
        "/OC /Off BDC /A Do EMC /A Do",
        &[&form],
    );
    assert_eq!(kinds(&content), ["hidden path", "path"]);
}

#[test]
fn a_form_leaving_its_marked_content_open_is_closed_with_a_warning() {
    let form = common::stream(
        "/Type /XObject /Subtype /Form /BBox [0 0 50 50] /Resources << /Properties << /Off 7 0 R >> >>",
        "/OC /Off BDC 0 0 9 9 re f",
    );
    let content = page(
        OC,
        "/XObject << /A 8 0 R >>",
        "/A Do 0 0 9 9 re f",
        &[&form],
    );
    assert_eq!(kinds(&content), ["hidden path", "path"]);
    assert_eq!(content.warnings.len(), 1, "{:?}", content.warnings);
    assert!(
        content.warnings[0].contains("marked content"),
        "{:?}",
        content.warnings
    );
}

#[test]
fn an_unmatched_emc_or_an_unknown_group_is_a_warning() {
    let content = page(OC, PROPERTIES, "EMC /OC /Missing BDC 0 0 5 5 re f EMC", &[]);
    assert_eq!(kinds(&content), ["path"]); // a group that cannot be read hides nothing
    assert_eq!(content.warnings.len(), 2, "{:?}", content.warnings);
}

#[test]
fn without_optional_content_properties_every_group_shows() {
    // A catalog with no /OCProperties: no group is off, and a membership showing only when all are off shows.
    let file = File::open(
        common::one_page(
            "/Resources << /Properties << /Off 5 0 R /M 6 0 R >> >>",
            "/OC /Off BDC 0 0 5 5 re f EMC /OC /M BDC 0 0 5 5 re f EMC",
            &[
                "<< /Type /OCG /Name (Off) >>",
                "<< /Type /OCMD /OCGs [5 0 R] /P /AllOff >>",
            ],
        ),
        b"",
    )
    .unwrap();
    let content = file.interpret(&file.pages().unwrap()[0], false).unwrap();
    assert_eq!(kinds(&content), ["path", "path"]);
}

#[test]
fn actual_text_replaces_the_text_of_the_glyphs_it_spans() {
    let content = page(
        "",
        "",
        "BT /F1 10 Tf 10 50 Td (Ex) Tj /Span << /ActualText (fi) >> BDC (XY) Tj EMC (nal) Tj ET",
        &[],
    );
    let glyphs: Vec<Option<String>> = content
        .events
        .iter()
        .filter_map(|e| match e {
            Event::Text(t) => Some(t.glyphs.iter().map(|g| g.text.clone())),
            _ => None,
        })
        .flatten()
        .collect();
    let some = |s: &str| Some(s.to_string());
    assert_eq!(
        glyphs,
        [
            some("E"),
            some("x"),
            some("fi"),
            some(""),
            some("n"),
            some("a"),
            some("l")
        ]
    );
    let read = spans(&content.events);
    assert_eq!(read.len(), 1);
    assert_eq!(read[0].text, "Exfinal");
    // Every glyph's box is kept: the span ends where `l` does, the replaced `Y` inside it.
    let Event::Text(last) = content.events.last().unwrap() else {
        panic!()
    };
    assert!((read[0].bbox.x1 - last.glyphs[2].bbox.x1).abs() < 1e-9);
}

#[test]
fn actual_text_reads_utf16_and_pdfdocencoding_and_may_be_named() {
    let content = page(
        "",
        "/Properties << /Acute << /ActualText <FEFF00E9> >> >>",
        "BT /F1 10 Tf 10 50 Td /Span /Acute BDC (e) Tj EMC \
         /Span << /ActualText (\\223) >> BDC (fi) Tj EMC ET", // spelling-ignore-line
        &[],
    );
    assert_eq!(shown(&content), "\u{e9}\u{fb01}");
}

#[test]
fn an_empty_actual_text_takes_a_hyphen_out_of_the_text_but_not_its_box() {
    let content = page(
        "",
        "",
        "BT /F1 10 Tf 10 50 Td (exam) Tj /Span << /ActualText () >> BDC (-) Tj EMC (ple) Tj ET",
        &[],
    );
    let read = spans(&content.events);
    assert_eq!(read.len(), 1);
    assert_eq!(read[0].text, "example");
}

#[test]
fn the_outermost_actual_text_replaces_everything_inside_it_across_shows() {
    let content = page(
        "",
        "",
        "BT /F1 10 Tf 10 50 Td /Span << /ActualText (all) >> BDC (a) Tj \
         /Span << /ActualText (inner) >> BDC (b) Tj EMC (c) Tj EMC (d) Tj ET",
        &[],
    );
    assert_eq!(shown(&content), "alld");
}

#[test]
fn actual_text_over_glyphs_cut_by_the_clip_goes_on_the_first_seen() {
    let content = page(
        "",
        "",
        "0 0 200 100 re W n BT /F1 10 Tf -20 50 Td /Span << /ActualText (word) >> BDC (xxxyz) Tj EMC ET",
        &[],
    );
    assert_eq!(shown(&content), "word");
}
