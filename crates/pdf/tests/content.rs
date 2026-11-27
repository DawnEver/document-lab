//! What a page draws: paths, images and Form `XObject`s as events in page space (points, y down).

mod common;

use pdf::color::Pattern;
use pdf::geometry::{Point, Rect};
use pdf::interpret::{ImageEvent, PathEvent, Segment};
use pdf::{Event, File};

fn events(page_extra: &str, content: &str, extra: &[&str]) -> (pdf::Content, pdf::Page) {
    let file = File::open(common::one_page(page_extra, content, extra), b"").unwrap();
    let page = file.pages().unwrap().remove(0);
    (file.interpret(&page, false).unwrap(), page)
}

fn paths(content: &pdf::Content) -> Vec<&PathEvent> {
    content
        .events
        .iter()
        .filter_map(|e| {
            if let Event::Path(p) = e {
                Some(p)
            } else {
                None
            }
        })
        .collect()
}

fn images(content: &pdf::Content) -> Vec<&ImageEvent> {
    content
        .events
        .iter()
        .filter_map(|e| {
            if let Event::Image(i) = e {
                Some(i)
            } else {
                None
            }
        })
        .collect()
}

fn p(x: f64, y: f64) -> Point {
    Point { x, y }
}

#[test]
fn a_filled_rectangle_lands_in_page_space_y_down() {
    let (content, page) = events("", "1 0 0 rg 10 20 30 40 re f", &[]);
    assert_eq!(page.size(), (200.0, 100.0));
    let path = paths(&content)[0];
    assert_eq!(
        path.segments,
        [
            Segment::MoveTo(p(10.0, 80.0)),
            Segment::LineTo(p(40.0, 80.0)),
            Segment::LineTo(p(40.0, 40.0)),
            Segment::LineTo(p(10.0, 40.0)),
            Segment::Close
        ]
    );
    assert!(path.stroke.is_none());
    let fill = path.fill.as_ref().unwrap();
    assert_eq!(
        (fill.color.rgb, fill.even_odd),
        (Some([1.0, 0.0, 0.0]), false)
    );
}

#[test]
fn a_stroke_scales_its_width_and_dashes_with_the_matrix() {
    let (content, _) = events(
        "",
        "2 0 0 2 0 0 cm 0.5 w [3 1] 2 d 0 0 1 RG 0 0 m 10 0 l S",
        &[],
    );
    let stroke = paths(&content)[0].stroke.clone().unwrap();
    assert_eq!(
        (stroke.width, stroke.dashes, stroke.dash_phase),
        (1.0, vec![6.0, 2.0], 4.0)
    );
    assert_eq!(stroke.color.rgb, Some([0.0, 0.0, 1.0]));
    assert_eq!(
        paths(&content)[0].segments[1],
        Segment::LineTo(p(20.0, 100.0))
    );
}

#[test]
fn q_and_big_q_save_and_restore_the_state() {
    let (content, _) = events("", "q 1 0 0 rg 5 0 0 5 0 0 cm Q 0 0 1 1 re f", &[]);
    let path = paths(&content)[0];
    assert_eq!(path.fill.as_ref().unwrap().color.rgb, Some([0.0; 3]));
    assert_eq!(path.segments[2], Segment::LineTo(p(1.0, 99.0)));
}

#[test]
fn curves_keep_their_control_points() {
    let (content, _) = events("", "0 0 m 0 10 10 10 10 0 c 20 20 30 0 v S", &[]);
    let segments = &paths(&content)[0].segments;
    assert_eq!(
        segments[1],
        Segment::CurveTo(p(0.0, 90.0), p(10.0, 90.0), p(10.0, 100.0))
    );
    assert_eq!(
        segments[2],
        Segment::CurveTo(p(10.0, 100.0), p(20.0, 80.0), p(30.0, 100.0))
    );
}

#[test]
fn a_clip_or_an_unpainted_path_draws_nothing() {
    let (content, _) = events("", "0 0 10 10 re W n 0 0 5 5 re n", &[]);
    assert_eq!(content.events, Vec::new());
}

#[test]
fn an_image_xobject_covers_its_transformed_unit_square() {
    let image = common::stream(
        "/Type /XObject /Subtype /Image /Width 4 /Height 2 /BitsPerComponent 8 /ColorSpace /DeviceGray",
        "abcdefgh",
    );
    let (content, _) = events(
        "/Resources << /XObject << /Im1 5 0 R >> >>",
        "q 40 0 0 20 10 10 cm /Im1 Do Q",
        &[&image],
    );
    let found = images(&content)[0];
    assert_eq!(
        found.bbox,
        Rect {
            x0: 10.0,
            y0: 70.0,
            x1: 50.0,
            y1: 90.0
        }
    );
    assert_eq!((found.width, found.height, found.object), (4, 2, Some(5)));
}

#[test]
fn a_form_xobject_draws_through_its_matrix_and_resources() {
    let form = common::stream(
        "/Type /XObject /Subtype /Form /BBox [0 0 10 10] /Matrix [1 0 0 1 100 0] /Resources << >>",
        "0 0 10 10 re f",
    );
    let (content, _) = events(
        "/Resources << /XObject << /Fm1 5 0 R >> >>",
        "/Fm1 Do 0 0 1 1 re f",
        &[&form],
    );
    let found = paths(&content);
    assert_eq!(found.len(), 2);
    assert_eq!(found[0].segments[0], Segment::MoveTo(p(100.0, 100.0)));
    assert_eq!(found[1].segments[0], Segment::MoveTo(p(0.0, 100.0)));
}

#[test]
fn an_inline_image_is_an_event_with_its_data() {
    let (content, _) = events(
        "",
        "q 10 0 0 10 0 0 cm BI /W 1 /H 1 /BPC 8 /CS /G ID \x7f EI Q",
        &[],
    );
    let found = images(&content)[0];
    assert_eq!(
        (found.object, found.inline.as_ref().map(|(_, d)| d.clone())),
        (None, Some(vec![0x7f]))
    );
}

#[test]
fn a_rotated_page_turns_its_page_space() {
    let (content, page) = events("/Rotate 90", "0 0 10 10 re f", &[]);
    assert_eq!(page.size(), (100.0, 200.0));
    // User (0, 0), the bottom-left, is the top-left once the page is turned clockwise.
    assert_eq!(paths(&content)[0].segments[0], Segment::MoveTo(p(0.0, 0.0)));
    assert_eq!(
        paths(&content)[0].segments[1],
        Segment::LineTo(p(0.0, 10.0))
    );
}

#[test]
fn an_unreadable_operation_is_a_named_warning() {
    let (content, _) = events("", "/Missing gs 0 0 1 1 re f", &[]);
    assert_eq!(paths(&content).len(), 1);
    assert_eq!(content.warnings.len(), 1);
    assert!(content.warnings[0].contains("gs"), "{:?}", content.warnings);
}

#[test]
fn colors_of_named_spaces_resolve_through_the_resources() {
    let resources =
        "/Resources << /ColorSpace << /CS0 [/Indexed /DeviceRGB 1 <000000 00FF00>] >> >>";
    let (content, _) = events(resources, "/CS0 cs 1 sc 0 0 1 1 re f", &[]);
    assert_eq!(
        paths(&content)[0].fill.as_ref().unwrap().color.rgb,
        Some([0.0, 1.0, 0.0])
    );
}

/// The text a page shows, glyph by glyph.
fn shown(content: &pdf::Content) -> String {
    content
        .events
        .iter()
        .filter_map(|e| match e {
            Event::Text(t) => Some(
                t.glyphs
                    .iter()
                    .filter_map(|g| g.text.clone())
                    .collect::<String>(),
            ),
            _ => None,
        })
        .collect()
}

const FONT: &str = "/Resources << /Font << /F1 5 0 R >> /XObject << /X 6 0 R >> >>";
const HELVETICA: &str = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>";

#[test]
fn a_mark_outside_a_clipping_path_is_not_seen() {
    let content = "q 0 0 100 100 re W n \
                   BT /F1 10 Tf 10 10 Td (in) Tj 150 0 Td (out) Tj ET \
                   0 0 1 rg 150 10 20 20 re f 10 10 20 20 re f Q \
                   BT /F1 10 Tf 150 50 Td (back) Tj ET";
    let form = common::stream("/Type /XObject /Subtype /Form /BBox [0 0 1 1]", "");
    let (content, _) = events(FONT, content, &[HELVETICA, &form]);
    assert_eq!(shown(&content), "inback"); // the clip ends with its `Q`
    assert_eq!(paths(&content).len(), 1);
}

#[test]
fn a_form_shows_only_what_lies_in_its_bbox() {
    let form = common::stream(
        "/Type /XObject /Subtype /Form /BBox [0 0 50 50] /Resources << /Font << /F1 5 0 R >> >>",
        "BT /F1 10 Tf 10 10 Td (seen) Tj 100 0 Td (cut) Tj ET 0 0 1 rg 120 10 5 5 re f",
    );
    let (content, _) = events(FONT, "1 0 0 1 20 20 cm /X Do", &[HELVETICA, &form]);
    assert_eq!(shown(&content), "seen");
    assert_eq!(paths(&content).len(), 0);
}

#[test]
fn a_mark_off_the_page_is_not_seen() {
    let (content, _) = events(
        FONT,
        "BT /F1 10 Tf 10 10 Td (on) Tj 300 0 Td (off) Tj ET",
        &[HELVETICA],
    );
    assert_eq!(shown(&content), "on");
}

const AXIAL: &str = "<< /ShadingType 2 /ColorSpace /DeviceRGB /Coords [0 0 1 0] \
                     /Function << /FunctionType 2 /Domain [0 1] /C0 [1 0 0] /C1 [0 0 1] /N 1 >> >>";

fn shadings(content: &pdf::Content) -> Vec<Rect> {
    content
        .events
        .iter()
        .filter_map(|e| {
            if let Event::Shading(s) = e {
                Some(s.bbox)
            } else {
                None
            }
        })
        .collect()
}

#[test]
fn a_shading_painted_with_sh_covers_the_clip_within_its_bbox() {
    let boxed = AXIAL.replace(">> >>", ">> /BBox [0 0 20 20] >>");
    let (content, _) = events(
        "/Resources << /Shading << /S 5 0 R /B 6 0 R >> >>",
        "/S sh q 10 10 50 50 re W n /S sh /B sh Q q 100 10 50 50 re W n /B sh Q",
        &[AXIAL, &boxed],
    );
    let r = |x0, y0, x1, y1| Rect { x0, y0, x1, y1 };
    assert_eq!(
        shadings(&content),
        [
            r(0.0, 0.0, 200.0, 100.0),
            r(10.0, 40.0, 60.0, 90.0),
            r(10.0, 80.0, 20.0, 90.0)
        ]
    );
    assert_eq!(content.warnings, Vec::<String>::new());
}

#[test]
fn a_fill_with_a_pattern_says_which_pattern_it_is() {
    let tiling = common::stream(
        "/PatternType 1 /PaintType 2 /TilingType 1 /BBox [0 0 4 4] /XStep 4 /YStep 4 /Resources << >>",
        "0 0 2 2 re f",
    );
    let (content, _) = events(
        "/Resources << /Pattern << /G 7 0 R /T 6 0 R >> >>",
        "/Pattern cs /G scn 10 10 30 30 re f [/Pattern /DeviceRGB] cs 1 0 0 /T scn 50 10 5 5 re f \
         /Missing scn 0 0 1 1 re f",
        &[AXIAL, &tiling, "<< /PatternType 2 /Shading 5 0 R >>"],
    );
    let fills: Vec<_> = paths(&content)
        .iter()
        .map(|p| {
            let color = &p.fill.as_ref().unwrap().color;
            (color.pattern, color.components.clone(), color.rgb)
        })
        .collect();
    assert_eq!(
        fills,
        [
            (Some(Pattern::Shading), vec![], None),
            (Some(Pattern::Tiling), vec![1.0, 0.0, 0.0], None),
            (Some(Pattern::Tiling), vec![1.0, 0.0, 0.0], None), // the unreadable `scn` is skipped
        ]
    );
    assert_eq!(content.warnings.len(), 1, "{:?}", content.warnings);
    assert!(content.warnings[0].contains("Missing"));
}

#[test]
fn text_in_a_clipping_mode_narrows_the_clip_to_its_glyphs_at_et() {
    let (content, _) = events(
        FONT,
        "q BT 7 Tr /F1 10 Tf 10 50 Td (AB) Tj 0 Tr 100 0 Td (far) Tj ET \
         150 10 5 5 re f 12 50 2 2 re f Q 150 10 5 5 re f \
         BT 0 Tr /F1 10 Tf 10 50 Td (AB) Tj ET 150 10 5 5 re f",
        &[HELVETICA],
    );
    assert_eq!(shown(&content), "ABfarAB"); // a clip mode's glyphs are reported, its mode with them
    let modes: Vec<i64> = content
        .events
        .iter()
        .filter_map(|e| match e {
            Event::Text(t) => Some(t.render_mode),
            _ => None,
        })
        .collect();
    assert_eq!(modes, [7, 0, 0]);
    assert_eq!(paths(&content).len(), 3); // only the square off the glyphs is cut, and only until `Q`
}
