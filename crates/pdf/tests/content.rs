//! What a page draws: paths, images and Form `XObject`s as events in page space (points, y down).

mod common;

use pdf::geometry::{Point, Rect};
use pdf::interpret::{ImageEvent, PathEvent, Segment};
use pdf::{Event, File};

fn events(page_extra: &str, content: &str, extra: &[&str]) -> (pdf::Content, pdf::Page) {
    let file = File::open(common::one_page(page_extra, content, extra), b"").unwrap();
    let page = file.pages().unwrap().remove(0);
    (file.interpret(&page).unwrap(), page)
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
