//! Annotations, when asked: each one's normal appearance drawn onto its rectangle, its events in its layer.

mod common;

use pdf::geometry::Rect;
use pdf::{Content, Event, File};

const HELVETICA: &str = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>";

/// What the one page (200 x 100) of a PDF draws, its annotations when `annotations`: the catalog carries
/// `catalog_extra`, the page's `/Annots` are `annots`, its content `content`, its font `/F1` object 5, and `extra`
/// objects follow as 6, 7, ...
fn drawn(
    catalog_extra: &str,
    annots: &str,
    content: &str,
    extra: &[&str],
    annotations: bool,
) -> Content {
    let catalog = format!("<< /Type /Catalog /Pages 2 0 R {catalog_extra} >>");
    let page = format!(
        "<< /Type /Page /Parent 2 0 R /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> /Annots [{annots}] >>"
    );
    let content = common::stream("", content);
    let mut objects = vec![
        catalog.as_str(),
        "<< /Type /Pages /Kids [3 0 R] /Count 1 /MediaBox [0 0 200 100] >>",
        &page,
        &content,
        HELVETICA,
    ];
    objects.extend_from_slice(extra);
    let file = File::open(common::pdf(&objects), b"").unwrap();
    let page = file.pages().unwrap().remove(0);
    file.interpret(&page, annotations).unwrap()
}

/// An appearance stream of the box `bbox` drawing `content`, with the font `/F1`.
fn appearance(bbox: &str, content: &str) -> String {
    common::stream(
        &format!(
            "/Type /XObject /Subtype /Form /BBox [{bbox}] /Resources << /Font << /F1 5 0 R >> >>"
        ),
        content,
    )
}

/// Each path's box and the annotation it is drawn for.
fn paths(content: &Content) -> Vec<(Rect, Option<String>)> {
    content
        .events
        .iter()
        .filter_map(|e| match e {
            Event::Path(p) => {
                let (xs, ys): (Vec<f64>, Vec<f64>) =
                    p.segments
                        .iter()
                        .filter_map(|s| match s {
                            pdf::interpret::Segment::MoveTo(q)
                            | pdf::interpret::Segment::LineTo(q) => Some((q.x, q.y)),
                            _ => None,
                        })
                        .unzip();
                let fold = |v: &[f64], f: fn(f64, f64) -> f64, start: f64| {
                    v.iter().copied().fold(start, f)
                };
                Some((
                    Rect {
                        x0: fold(&xs, f64::min, f64::INFINITY),
                        y0: fold(&ys, f64::min, f64::INFINITY),
                        x1: fold(&xs, f64::max, f64::NEG_INFINITY),
                        y1: fold(&ys, f64::max, f64::NEG_INFINITY),
                    },
                    p.layer.annotation.clone(),
                ))
            }
            _ => None,
        })
        .collect()
}

fn r(x0: f64, y0: f64, x1: f64, y1: f64) -> Rect {
    Rect { x0, y0, x1, y1 }
}

#[test]
fn an_appearance_is_drawn_onto_its_rectangle_only_when_asked() {
    let note = "<< /Type /Annot /Subtype /FreeText /Rect [50 50 150 70] /AP << /N 7 0 R >> >>";
    let ap = appearance("0 0 100 20", "BT /F1 10 Tf 0 0 Td (note) Tj ET");
    let page_only = drawn("", "6 0 R", "", &[note, &ap], false);
    assert_eq!(page_only.events, Vec::new());
    let content = drawn("", "6 0 R", "", &[note, &ap], true);
    let Event::Text(text) = &content.events[0] else {
        panic!("{:?}", content.events)
    };
    let origin = text.glyphs[0].origin;
    assert!(
        (origin.x - 50.0).abs() < 1e-9 && (origin.y - 50.0).abs() < 1e-9,
        "{origin:?}"
    );
    assert_eq!(text.layer.annotation.as_deref(), Some("FreeText"));
    assert!(!text.layer.hidden);
}

#[test]
fn an_appearance_box_is_mapped_through_its_matrix_and_scaled_onto_the_rectangle() {
    let scaled = "<< /Type /Annot /Subtype /Square /Rect [20 20 40 40] /AP << /N 8 0 R >> >>";
    let turned = "<< /Type /Annot /Subtype /Square /Rect [100 10 120 20] /AP << /N 9 0 R >> >>";
    let turned_ap = common::stream(
        "/Type /XObject /Subtype /Form /BBox [0 0 10 20] /Matrix [0 1 -1 0 0 0]",
        "0 0 10 20 re f",
    );
    let content = drawn(
        "",
        "6 0 R 7 0 R",
        "",
        &[
            scaled,
            turned,
            &appearance("0 0 10 10", "0 0 10 10 re f"),
            &turned_ap,
        ],
        true,
    );
    let square = Some("Square".to_string());
    assert_eq!(
        paths(&content),
        [
            (r(20.0, 60.0, 40.0, 80.0), square.clone()),
            (r(100.0, 80.0, 120.0, 90.0), square)
        ]
    );
}

#[test]
fn a_state_appearance_is_the_one_its_state_names() {
    let on = appearance("0 0 10 10", "0 0 1 1 re f");
    let off = appearance("0 0 10 10", "0 0 3 3 re f");
    let checked = "<< /Type /Annot /Subtype /Widget /Rect [0 0 10 10] /AP << /N << /On 8 0 R /Off 9 0 R >> >> /AS /Off >>";
    let stateless = "<< /Type /Annot /Subtype /Widget /Rect [0 0 10 10] /AP << /N << /On 8 0 R /Off 9 0 R >> >> >>";
    let content = drawn(
        "",
        "6 0 R 7 0 R",
        "",
        &[checked, stateless, &on, &off],
        true,
    );
    let widget = Some("Widget".to_string());
    assert_eq!(paths(&content), [(r(0.0, 97.0, 3.0, 100.0), widget)]);
}

#[test]
fn a_popup_or_an_annotation_flagged_hidden_or_not_to_view_is_not_drawn() {
    let ap = appearance("0 0 10 10", "0 0 10 10 re f");
    let annotation = |extra: &str| {
        format!("<< /Type /Annot /Subtype /Square /Rect [0 0 10 10] /AP << /N 10 0 R >> {extra} >>")
    };
    let (popup, hidden, no_view, print) = (
        "<< /Type /Annot /Subtype /Popup /Rect [0 0 10 10] /AP << /N 10 0 R >> >>".to_string(),
        annotation("/F 2"),
        annotation("/F 32"),
        annotation("/F 4"),
    );
    let content = drawn(
        "",
        "6 0 R 7 0 R 8 0 R 9 0 R",
        "",
        &[&popup, &hidden, &no_view, &print, &ap],
        true,
    );
    assert_eq!(paths(&content).len(), 1); // only the one flagged to print
}

#[test]
fn an_appearance_is_clipped_to_its_box_and_starts_from_the_page_s_initial_state() {
    let annotation = "<< /Type /Annot /Subtype /Square /Rect [0 0 10 10] /AP << /N 7 0 R >> >>";
    let ap = appearance("0 0 10 10", "0 0 5 5 re f 20 20 5 5 re f");
    let content = drawn(
        "",
        "6 0 R",
        "2 0 0 2 0 0 cm 1 0 0 rg",
        &[annotation, &ap],
        true,
    );
    let Event::Path(path) = &content.events[0] else {
        panic!("{:?}", content.events)
    };
    assert_eq!(path.fill.as_ref().unwrap().color.rgb, Some([0.0; 3])); // black, not the page's red
    assert_eq!(paths(&content).len(), 1); // the square off the box is cut
    assert_eq!(paths(&content)[0].0, r(0.0, 95.0, 5.0, 100.0)); // not scaled by the page's `cm`
}

#[test]
fn an_annotation_whose_group_is_off_is_drawn_hidden() {
    let oc = "/OCProperties << /OCGs [8 0 R] /D << /OFF [8 0 R] >> >>";
    let annotation =
        "<< /Type /Annot /Subtype /Square /Rect [0 0 10 10] /AP << /N 7 0 R >> /OC 8 0 R >>";
    let ap = appearance("0 0 10 10", "0 0 5 5 re f");
    let content = drawn(
        oc,
        "6 0 R",
        "",
        &[annotation, &ap, "<< /Type /OCG /Name (Off) >>"],
        true,
    );
    let Event::Path(path) = &content.events[0] else {
        panic!("{:?}", content.events)
    };
    assert!(path.layer.hidden);
}

#[test]
fn an_annotation_that_cannot_be_drawn_is_a_warning() {
    let annotation = "<< /Type /Annot /Subtype /Square /Rect [0 0 10] /AP << /N 7 0 R >> >>";
    let ap = appearance("0 0 10 10", "0 0 5 5 re f");
    let content = drawn("", "6 0 R", "", &[annotation, &ap], true);
    assert_eq!(content.events, Vec::new());
    assert_eq!(content.warnings.len(), 1, "{:?}", content.warnings);
    assert!(content.warnings[0].contains("annotation 0"));
}

#[test]
fn an_annotation_without_an_appearance_draws_nothing() {
    let file = File::open(
        common::one_page(
            "/Annots [5 0 R]",
            "",
            &["<< /Type /Annot /Subtype /Text /Rect [0 0 10 10] /Contents (no appearance) >>"],
        ),
        b"",
    )
    .unwrap();
    let content = file.interpret(&file.pages().unwrap()[0], true).unwrap();
    assert_eq!(
        (content.events, content.warnings),
        (Vec::new(), Vec::<String>::new())
    );
}
