//! What a page draws: paths, images and Form `XObject`s as events in page space (points, y down).

mod common;

use pdf::color::PatternKind;
use pdf::geometry::Matrix;
use pdf::geometry::Segment;
use pdf::geometry::{Point, Rect};
use pdf::interpret::{ImageEvent, PathEvent, Reach};
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
        (found.object, found.inline.as_ref().map(|i| i.data.clone())),
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
    assert!(
        content.warnings[0].why.contains("gs"),
        "{:?}",
        content.warnings
    );
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
    assert!(content.warnings.is_empty(), "{:?}", content.warnings);
}

#[test]
fn a_fill_with_a_pattern_says_which_pattern_it_is_and_an_uncolored_tilings_tint() {
    let tiling = |paint: u8| {
        common::stream(
            &format!(
                "/PatternType 1 /PaintType {paint} /TilingType 1 /BBox [0 0 4 4] /XStep 4 /YStep 4 /Resources << >>"
            ),
            "0 0 2 2 re f",
        )
    };
    let (content, _) = events(
        "/Resources << /Pattern << /G 7 0 R /T 6 0 R /C 8 0 R >> >>",
        "/Pattern cs /G scn 10 10 30 30 re f [/Pattern /DeviceCMYK] cs 0 1 1 0 /T scn 50 10 5 5 re f          /Missing scn 0 0 1 1 re f /Pattern cs /C scn 0 0 2 2 re f",
        &[
            AXIAL,
            &tiling(2),
            "<< /PatternType 2 /Shading 5 0 R >>",
            &tiling(1),
        ],
    );
    let fills: Vec<_> = paths(&content)
        .iter()
        .map(|p| {
            let color = &p.fill.as_ref().unwrap().color;
            (
                color.pattern.as_ref().map(|p| p.kind),
                color.components.clone(),
                color.rgb,
            )
        })
        .collect();
    let uncolored = Some(PatternKind::Tiling { uncolored: true });
    assert_eq!(
        fills,
        [
            (Some(PatternKind::Shading), vec![], None),
            // The tint, in the pattern space's base: CMYK (0, 1, 1, 0) is red.
            (uncolored, vec![0.0, 1.0, 1.0, 0.0], Some([1.0, 0.0, 0.0])),
            (uncolored, vec![0.0, 1.0, 1.0, 0.0], Some([1.0, 0.0, 0.0])), // the unreadable `scn` is skipped
            (Some(PatternKind::Tiling { uncolored: false }), vec![], None), // a colored pattern paints its own colors
        ]
    );
    // Pattern space is the page's default space (y up, 100 high), whatever the CTM.
    let first = paths(&content)[0]
        .fill
        .as_ref()
        .unwrap()
        .color
        .pattern
        .clone()
        .unwrap();
    assert_eq!(first.matrix, Matrix([1.0, 0.0, 0.0, -1.0, 0.0, 100.0]));
    assert_eq!(content.warnings.len(), 1, "{:?}", content.warnings);
    assert!(content.warnings[0].why.contains("Missing"));
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

#[test]
fn each_event_carries_the_clipping_paths_in_force_as_they_nest() {
    use pdf::interpret::Outline;
    let (content, _) = events(
        "",
        "q 0 0 100 100 re W n q 10 10 50 50 re W* n 0 0 20 20 re f Q 0 0 6 6 re f Q 0 0 7 7 re f",
        &[],
    );
    let found = paths(&content);
    let inner = found[0].clip.as_ref().expect("two clips");
    assert!(matches!(
        inner.outline,
        Outline::Path { even_odd: true, .. }
    ));
    let outer = inner.within.as_ref().expect("the outer clip");
    assert!(matches!(
        outer.outline,
        Outline::Path {
            even_odd: false,
            ..
        }
    ));
    assert!(outer.within.is_none());
    assert_eq!(
        found[1].clip.as_deref(),
        Some(&**outer),
        "Q restores the outer clip"
    );
    assert!(found[2].clip.is_none(), "the page's clip alone");
}

#[test]
fn a_form_bbox_is_a_clipping_path_through_its_matrix() {
    use pdf::interpret::Outline;
    let form = common::stream(
        "/Type /XObject /Subtype /Form /BBox [0 0 10 10] /Matrix [2 0 0 2 0 0]",
        "0 0 5 5 re f",
    );
    let (content, _) = events(
        "/Resources << /XObject << /Fm1 5 0 R >> >>",
        "/Fm1 Do",
        &[&form],
    );
    let clip = paths(&content)[0].clip.as_ref().expect("the form's box");
    let Outline::Path { segments, even_odd } = &clip.outline else {
        panic!("{clip:?}");
    };
    assert!(!even_odd);
    // The box's far corner (10, 10) through the matrix is (20, 20) in user space, (20, 80) on the page.
    assert_eq!(segments[2], Segment::LineTo(p(20.0, 80.0)));
}

#[test]
fn an_empty_string_in_a_clipping_mode_adds_nothing_to_the_clip() {
    let (content, _) = events(
        "/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >>",
        "BT /F1 10 Tf 7 Tr () Tj ET 0 0 50 50 re f",
        &[],
    );
    assert_eq!(paths(&content)[0].clip, None);
}

#[test]
fn a_text_object_clips_to_its_clipping_glyphs_outlines() {
    use pdf::interpret::Outline;
    let (content, page) = events(
        "/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >>",
        "BT /F1 10 Tf 7 Tr 10 10 Td (A) Tj ET 0 0 50 50 re f",
        &[],
    );
    let clip = paths(&content)[0].clip.as_ref().expect("the glyphs' clip");
    let Outline::Glyphs { segments, unshaped } = &clip.outline else {
        panic!("{clip:?}");
    };
    assert_eq!(unshaped.as_slice(), []);
    // The A of Helvetica's substitute, 10 points tall at most and 6.67 wide, on the baseline 10 points up.
    let points: Vec<_> = segments
        .iter()
        .filter_map(|s| match s {
            Segment::MoveTo(q) | Segment::LineTo(q) | Segment::CurveTo(_, _, q) => Some(*q),
            Segment::Close => None,
        })
        .collect();
    let around = pdf::geometry::Rect::around(&points).expect("an outline");
    let baseline = page.size().1 - 10.0;
    assert!(
        around.x0 >= 9.9 && around.x1 <= 16.8 && around.width() > 6.0,
        "{around:?}"
    );
    assert!(
        (around.y1 - baseline).abs() < 0.1 && around.height() > 6.5,
        "{around:?}"
    );
}

#[test]
fn a_graphics_state_sets_alphas_blend_mode_soft_mask_and_miter_limit() {
    use pdf::interpret::Blend;
    let group = common::stream(
        "/Type /XObject /Subtype /Form /BBox [0 0 10 10] /Group << /S /Transparency /CS /DeviceGray >>",
        "0.5 g 0 0 5 5 re f",
    );
    let (content, _) = events(
        "/Resources << /ExtGState << /G1 << /ca 0.5 /CA 0.25 /BM [/Bogus /Multiply] /SMask << /S /Luminosity /G 5 0 R /BC [0.25] >> /ML 3 >> /G2 << /SMask /None >> /G3 << /SMask << /S /Luminosity >> >> >> >>",
        "2 0 0 2 0 0 cm /G1 gs 0 0 5 5 re B /G2 gs 2 M 0 0 5 5 re S /G3 gs",
        &[&group],
    );
    let found = paths(&content);
    let c = &found[0].compositing;
    assert_eq!(
        (c.fill_alpha, c.stroke_alpha, c.blend),
        (0.5, 0.25, Blend::Multiply)
    );
    let mask = c.soft_mask.as_ref().expect("a soft mask in force");
    assert!(mask.luminosity);
    assert_eq!(mask.object, Some(5));
    assert_eq!(mask.backdrop, [0.25]);
    // The group's space is the user space where the mask was set: the CTM then.
    assert_eq!(mask.matrix, Matrix([2.0, 0.0, 0.0, -2.0, 0.0, 100.0]));
    assert_eq!(found[0].stroke.as_ref().unwrap().miter_limit, 3.0);
    assert!(
        found[1].compositing.soft_mask.is_none(),
        "/SMask /None ends the soft mask"
    );
    assert_eq!(found[1].stroke.as_ref().unwrap().miter_limit, 2.0);
    assert_eq!(content.warnings.len(), 1, "{:?}", content.warnings);
    assert!(
        content.warnings[0].why.contains("/G is not a Form"),
        "a mask with no group is named: {:?}",
        content.warnings
    );
    // The state it would have set is not in force: the marks drawn after it (none here) may differ.
    assert_eq!(content.warnings[0].reach, Reach::From(2));
}

#[test]
fn a_warning_says_what_of_the_drawing_it_may_change() {
    let (content, _) = events(
        "",
        "0 0 1 1 re f /Missing gs 0 0 2 2 re f /Nothing Do /Tag BMC",
        &[],
    );
    let reaches: Vec<Reach> = content.warnings.iter().map(|w| w.reach).collect();
    assert_eq!(
        reaches,
        [Reach::From(1), Reach::Anywhere, Reach::Unseen],
        "{:?}",
        content.warnings
    );
}

#[test]
fn a_stroke_keeps_the_matrix_it_was_painted_under() {
    let (content, _) = events("", "2 0 0 1 0 0 cm 0 0 m 10 0 l S", &[]);
    let stroke = paths(&content)[0].stroke.clone().unwrap();
    assert_eq!(stroke.ctm.0, [2.0, 0.0, 0.0, -1.0, 0.0, 100.0]);
}

#[test]
fn an_image_carries_the_fill_color_a_stencil_paints_in() {
    let (content, _) = events(
        "",
        "1 0 0 rg q 10 0 0 10 0 0 cm BI /W 1 /H 1 /IM true ID \x00 EI Q",
        &[],
    );
    assert_eq!(images(&content)[0].fill.rgb, Some([1.0, 0.0, 0.0]));
}

#[test]
fn a_transparency_group_entered_with_an_alpha_starts_its_marks_afresh_and_flags_them() {
    let group = common::stream(
        "/Type /XObject /Subtype /Form /BBox [0 0 10 10] /Group << /S /Transparency >>",
        "0 0 5 5 re f",
    );
    let (content, _) = events(
        "/Resources << /XObject << /Fm1 5 0 R >> /ExtGState << /G1 << /ca 0.35 >> >> >>",
        "/Fm1 Do /G1 gs /Fm1 Do 0 0 5 5 re f",
        &[&group],
    );
    let found = paths(&content);
    let plain = found[0].compositing.group.as_ref().expect("a group");
    assert!(!plain.isolated && !plain.knockout);
    assert!(
        !found[0].compositing.in_composited_group(),
        "a group entered opaque composites as its marks do"
    );
    let inside = &found[1].compositing;
    assert_eq!(inside.fill_alpha, 1.0, "a group's marks start afresh");
    assert!(inside.in_composited_group());
    let group = inside.group.as_ref().unwrap();
    assert_eq!(group.compositing.fill_alpha, 0.35);
    assert_eq!(
        group.bbox,
        Rect {
            x0: 0.0,
            y0: 90.0,
            x1: 10.0,
            y1: 100.0
        }
    );
    assert_ne!(
        found[0].compositing.group, found[1].compositing.group,
        "each painting is a group of its own"
    );
    assert_eq!(found[2].compositing.fill_alpha, 0.35);
    assert!(found[2].compositing.group.is_none());
}
