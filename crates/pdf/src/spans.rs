//! Glyphs to spans: runs of text in one font and size along one baseline -- or down one column, when the font writes
//! vertically -- the unit a page's text is read in.
//!
//! Consecutive glyphs join while their font, size, fill, writing mode and direction hold and each starts where the
//! last ended: on its baseline (within [`BASELINE`] sizes) and no further on than [`BREAK`] sizes. A gap wider than
//! [`SPACE`] sizes with no space glyph in it reads as a space, as the eye reads it. Text events are taken in
//! drawing order, so a span never joins glyphs drawn apart, nor glyphs of two layers (hidden, an annotation's), and
//! a vertical span's text reads top to bottom.
//!
//! Where a word gap is, is this module's alone. A span continues the word of another that ends at most [`SPACE`]
//! sizes before its start, raised or lowered by no more than [`RISE`] sizes, no space glyph between -- a change
//! of font, size or baseline inside a word -- whenever either was drawn (a subscript is often drawn after the
//! formula it sits in). Spans continuing one another share a number ([`Span::unspaced`]), so a reader setting
//! spans into lines spaces them by that and measures no gap of its own.
//!
//! An accent drawn as a glyph of its own over (or under) a letter, as TeX without T1 encoding draws `ü` (`u` and
//! `¨`), is one glyph with it before spans are made: the two composed as [`crate::accents`] says, the letter's
//! place in the drawing order kept, its box spanning both. One set on no letter stays as it is (a lone `¨`).
//!
//! A fill change splits a run (a scan's glyphs each a grey of their own, a colored digit): the piece after it says it
//! [continues the run](Span::continues_run) of the span before it, so a reader setting spans into lines reads the
//! run whole without judging again where it goes on.
//!
//! Text filled in the color painted under it shows nothing ([`Span::in_background_color`]): a white word on the
//! white page, a black one on a black box. What is under a span is what was drawn before it, top down: every
//! mark overlapping its box must be in its fill, down to one covering its box whole -- a path filled in one opaque
//! color whose region, by its fill rule, holds the box, and so does every clip in force: a rectangle, a polygon, a
//! rounded box, a clip intersected in (`covers`: it never covers wrongly, a curve covering only where the box
//! of its control points stays clear of the span's) -- or, when none covers it, the page, white. The colors are
//! compared exactly, as sRGB. Any other mark overlapping its box (an image, a shading, a pattern, a translucent
//! paint, a glyph or a path in a color of its own: a black `!` on a white triangle glyph on a black box) shows the
//! text; a glyph covers no box for sure, its outline not read. What hidden or inkless layers draw is under nothing.

use std::borrow::Cow;
use std::cmp::Ordering;
use std::rc::Rc;

use crate::accents;
use crate::color::Color;
use crate::font::FontInfo;
use crate::geometry::{Point, Rect};
use crate::interpret::{Blend, Clip, Compositing, Event, Layer, Outline, Segment, TextEvent};

/// A gap wider than this many font sizes between two glyphs is a space.
pub const SPACE: f64 = 0.15;
/// A glyph further on than this many font sizes (or back by more than half) starts another span.
pub const BREAK: f64 = 1.0;
/// A glyph whose baseline is off the span's by more than this many font sizes starts another span.
pub const BASELINE: f64 = 0.1;
/// A glyph raised or lowered by up to this many font sizes (a superscript, a subscript) still continues the word
/// before it.
pub const RISE: f64 = 0.5;
/// The page's own color, under every mark: white, in sRGB.
pub const PAGE: [f64; 3] = [1.0, 1.0, 1.0];

/// A run of text.
#[derive(Debug, Clone, PartialEq)]
pub struct Span {
    /// Its text; a glyph no source names is U+FFFD.
    pub text: String,
    /// Its font.
    pub font: Rc<FontInfo>,
    /// The font size in page space.
    pub size: f64,
    /// The union of its glyphs' boxes, in page space.
    pub bbox: Rect,
    /// The text rendering mode it was shown in (3: invisible, as an OCR text layer is).
    pub render_mode: i64,
    /// The layer its glyphs are drawn in: a span never joins glyphs of two.
    pub layer: Layer,
    /// Whether it is written vertically (its font's writing mode): its glyphs stand upright and step down its
    /// column, its text reading top to bottom.
    pub vertical: bool,
    /// The unit vector its baseline runs along, in page space (y down): `(1, 0)` for text set left to right,
    /// `(0, -1)` for text set bottom to top.
    pub direction: (f64, f64),
    /// The number it shares, on its page, with the spans it continues or that continue it, no word gap between:
    /// a change of font, size or baseline inside a word (`Head` set bold in `Headline`; the `2` of `H2O`). None
    /// when it continues no span and none continues it.
    pub unspaced: Option<usize>,
    /// The color its glyphs are filled with; None when its rendering mode does not fill (stroke, invisible, clip).
    pub fill: Option<Color>,
    /// Whether it continues the run of the span before it: its first glyph would have joined that span but for a
    /// change of fill (see the module).
    pub continues_run: bool,
    /// Whether its glyphs are filled (and only filled: render mode 0 or 4) in the color of what is painted under its
    /// box, so it shows nothing (see the module).
    pub in_background_color: bool,
}

/// The fill of glyphs shown in `text`: its fill color when its rendering mode fills (0, 2, 4, 6).
fn fill_of(text: &TextEvent) -> Option<Color> {
    matches!(text.render_mode, 0 | 2 | 4 | 6).then(|| text.fill.clone())
}

struct Open {
    span: Span,
    /// The index in the page's events of the text event showing its first glyph: what was drawn before it is under
    /// it.
    first: usize,
    /// How its first glyph is composited.
    compositing: Compositing,
    /// Where its first glyph starts.
    origin: Point,
    end: Point,
    /// Whether a glyph with ink (not white-space) is in it yet: only those make its box.
    inked: bool,
}

/// A glyph as spans read it: its text event, its text (U+FFFD when no source names it), where it starts and ends,
/// and its box.
struct Placed<'a> {
    event: &'a TextEvent,
    /// The index of `event` in the page's events.
    index: usize,
    text: Cow<'a, str>,
    origin: Point,
    end: Point,
    bbox: Rect,
}

/// The spans of a page's events, in drawing order.
#[must_use]
pub fn spans(events: &[Event]) -> Vec<Span> {
    let mut glyphs: Vec<Placed> = events
        .iter()
        .enumerate()
        .filter_map(|(index, event)| match event {
            Event::Text(text) => Some((index, text)),
            _ => None,
        })
        .flat_map(|(index, event)| {
            event.glyphs.iter().map(move |glyph| Placed {
                event,
                index,
                text: Cow::Borrowed(glyph.text.as_deref().unwrap_or("\u{FFFD}")),
                origin: glyph.origin,
                end: glyph.end,
                bbox: glyph.bbox,
            })
        })
        .collect();
    compose_accents(&mut glyphs);
    let mut out: Vec<Open> = Vec::new();
    let mut open: Option<Open> = None;
    for glyph in &glyphs {
        let text = glyph.event;
        let length = (glyph.end.x - glyph.origin.x).hypot(glyph.end.y - glyph.origin.y);
        let direction = if length > 0.0 {
            (
                (glyph.end.x - glyph.origin.x) / length,
                (glyph.end.y - glyph.origin.y) / length,
            )
        } else {
            let along = if text.vertical {
                (0.0, 1.0)
            } else {
                (1.0, 0.0)
            };
            open.as_ref().map_or(along, |o| o.span.direction)
        };
        let piece = glyph.text.as_ref();
        let step = open
            .as_ref()
            .and_then(|o| join(o, text, glyph.origin, direction));
        let joined = open
            .as_mut()
            .zip(step)
            .filter(|(o, _)| o.span.fill == fill_of(text));
        if let Some((o, gap)) = joined {
            if gap && !o.span.text.ends_with(' ') && !piece.starts_with(' ') {
                o.span.text.push(' ');
            }
            o.span.text.push_str(piece);
            let ink = inks(piece);
            if ink {
                o.span.bbox = if o.inked {
                    o.span.bbox.union(glyph.bbox)
                } else {
                    glyph.bbox
                };
                o.inked = true;
            }
            o.end = glyph.end;
            continue;
        }
        out.extend(open.take());
        open = Some(Open {
            span: Span {
                text: piece.to_string(),
                font: Rc::clone(&text.font),
                size: text.size,
                bbox: glyph.bbox,
                render_mode: text.render_mode,
                layer: text.layer.clone(),
                vertical: text.vertical,
                direction,
                unspaced: None,
                fill: fill_of(text),
                continues_run: step.is_some(),
                in_background_color: false,
            },
            first: glyph.index,
            compositing: text.compositing,
            origin: glyph.origin,
            end: glyph.end,
            inked: inks(piece),
        });
    }
    out.extend(open);
    finished(out, events)
}

/// The spans `out` opened on the page of `events`, finished: each its [`Span::unspaced`] number, a blank one left
/// out -- the next continuing the run of the span kept before it only when every span between went on with that
/// run --, trailing white-space trimmed, and whether it is [in its background's color](in_background_color).
fn finished(out: Vec<Open>, events: &[Event]) -> Vec<Span> {
    let unspaced = unspaced(&out);
    let marks: Vec<Option<Rect>> = events.iter().map(mark).collect();
    let mut runs_on = false;
    out.into_iter()
        .zip(unspaced)
        .filter_map(|(o, unspaced)| {
            if o.span.text.trim().is_empty() {
                runs_on &= o.span.continues_run;
                return None;
            }
            let continues_run = o.span.continues_run && runs_on;
            runs_on = true;
            let in_background_color =
                in_background_color(&o, &events[..o.first], &marks[..o.first]);
            let mut span = o.span;
            let trimmed = span.text.trim_end().len();
            span.text.truncate(trimmed);
            Some(Span {
                unspaced,
                continues_run,
                in_background_color,
                ..span
            })
        })
        .collect()
}

/// The box `event` may paint something seen in; None when it paints nothing seen (in a hidden or inkless layer,
/// invisible text, a path of no points). A stroke reaches half its width past its path.
fn mark(event: &Event) -> Option<Rect> {
    match event {
        Event::Text(text)
            if !text.layer.hidden && !text.layer.inkless && !matches!(text.render_mode, 3 | 7) =>
        {
            text.glyphs
                .iter()
                .filter(|g| g.text.as_deref().is_none_or(inks))
                .map(|g| g.bbox)
                .reduce(Rect::union)
        }
        Event::Path(path) if !path.layer.hidden && !path.layer.inkless => {
            let bounds = Rect::around(&points(&path.segments))?;
            let half = path.stroke.as_ref().map_or(0.0, |s| s.width / 2.0);
            Some(Rect {
                x0: bounds.x0 - half,
                y0: bounds.y0 - half,
                x1: bounds.x1 + half,
                y1: bounds.y1 + half,
            })
        }
        Event::Image(image) if !image.layer.hidden && !image.layer.inkless => Some(image.bbox),
        Event::Shading(shading) if !shading.layer.hidden && !shading.layer.inkless => {
            Some(shading.bbox)
        }
        _ => None,
    }
}

/// Whether `open`'s glyphs, only filled and composited as they are, are in the color painted under its box by the
/// events `below` it, whose [`mark`]s are `marks` (see the module).
fn in_background_color(open: &Open, below: &[Event], marks: &[Option<Rect>]) -> bool {
    let span = &open.span;
    let Some(fill) = span
        .fill
        .as_ref()
        .filter(|_| matches!(span.render_mode, 0 | 4))
    else {
        return false;
    };
    if !opaque(open.compositing, open.compositing.fill_alpha) {
        return false;
    }
    let bbox = span.bbox;
    let paints = |color: &Color, compositing: Compositing, alpha: f64| {
        opaque(compositing, alpha) && same_color(color, fill)
    };
    for (event, _) in below
        .iter()
        .zip(marks)
        .rev()
        .filter(|(_, m)| m.is_some_and(|m| overlaps(m, bbox)))
    {
        match event {
            Event::Text(text) => {
                if !text_in(text, fill, bbox) {
                    return false;
                }
            }
            Event::Path(path) => {
                let stroke = path.stroke.as_ref().is_none_or(|s| {
                    paints(&s.color, path.compositing, path.compositing.stroke_alpha)
                });
                let filled = path.fill.as_ref().is_none_or(|f| {
                    paints(&f.color, path.compositing, path.compositing.fill_alpha)
                });
                if !(stroke && filled) {
                    return false;
                }
                if path
                    .fill
                    .as_ref()
                    .is_some_and(|f| covers(&path.segments, f.even_odd, bbox))
                    && path.clip.as_deref().is_none_or(|c| clip_covers(c, bbox))
                {
                    return true;
                }
            }
            Event::Image(_) | Event::Shading(_) => return false,
        }
    }
    fill.pattern.is_none() && fill.rgb == Some(PAGE)
}

/// Whether every glyph of `text` with ink in `bbox` paints only `fill`, opaque: glyphs cover no box for sure (their
/// outlines are not read), but one in another color shows what is drawn over it (a black `!` on a white triangle).
fn text_in(text: &TextEvent, fill: &Color, bbox: Rect) -> bool {
    let paints =
        |color: &Color, alpha: f64| opaque(text.compositing, alpha) && same_color(color, fill);
    let fills = !matches!(text.render_mode, 0 | 2 | 4 | 6)
        || paints(&text.fill, text.compositing.fill_alpha);
    let strokes = !matches!(text.render_mode, 1 | 2 | 5 | 6)
        || paints(&text.stroke, text.compositing.stroke_alpha);
    (fills && strokes)
        || !text
            .glyphs
            .iter()
            .any(|g| g.text.as_deref().is_none_or(inks) && overlaps(g.bbox, bbox))
}

/// Whether two boxes share some area (touching edges do not).
fn overlaps(a: Rect, b: Rect) -> bool {
    a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1
}

/// Whether a mark composited so, at constant alpha `alpha`, paints its own color: alpha 1, `Normal`, no soft mask,
/// in no transparency group composited otherwise.
fn opaque(compositing: Compositing, alpha: f64) -> bool {
    alpha >= 1.0
        && compositing.blend == Blend::Normal
        && !compositing.soft_mask
        && !compositing.grouped
}

/// Whether two colors are one sRGB color exactly; a pattern, or a color with no sRGB, is no one color.
fn same_color(a: &Color, b: &Color) -> bool {
    a.pattern.is_none() && b.pattern.is_none() && a.rgb.is_some() && a.rgb == b.rgb
}

/// Every point `segments` name, a curve's control points too: their box holds the path.
fn points(segments: &[Segment]) -> Vec<Point> {
    segments
        .iter()
        .flat_map(|segment| match *segment {
            Segment::MoveTo(p) | Segment::LineTo(p) => vec![p],
            Segment::CurveTo(a, b, c) => vec![a, b, c],
            Segment::Close => vec![],
        })
        .collect()
}

/// Whether the region `segments` fill -- by the even-odd rule or the nonzero winding one (8.5.3.3) -- holds `bbox`
/// whole, never wrongly. A box no edge reaches inside lies in one piece of the plane the edges cut, filled or not as
/// its center winds round them: exact there. A region an edge reaches inside the box (a seam where two subpaths
/// overlap), a side no exact sign settles, or a curve whose control points' box reaches inside the box, covers
/// nothing.
fn covers(segments: &[Segment], even_odd: bool, bbox: Rect) -> bool {
    let Some(edges) = edges(segments, bbox) else {
        return false;
    };
    if edges.iter().any(|&(p, q)| crosses(p, q, bbox)) {
        return false;
    }
    let center = Point {
        x: f64::midpoint(bbox.x0, bbox.x1),
        y: f64::midpoint(bbox.y0, bbox.y1),
    };
    // The edges crossing the ray from the center along x, each counted by its direction (Sunday's winding number).
    let mut winding = 0_i64;
    for &(p, q) in &edges {
        let up = p.y <= center.y && center.y < q.y;
        let down = q.y <= center.y && center.y < p.y;
        if up || down {
            match orientation(p, q, center) {
                Some(Ordering::Greater) if up => winding += 1,
                Some(Ordering::Less) if down => winding -= 1,
                Some(Ordering::Greater | Ordering::Less) => {}
                None | Some(Ordering::Equal) => return false,
            }
        }
    }
    if even_odd {
        winding % 2 != 0
    } else {
        winding != 0
    }
}

/// The straight edges bounding the region `segments` fill, every subpath closed back to its start as a fill closes
/// it; a curve by its chord, when its control points' box -- which holds the curve -- stays clear of `bbox`'s inside,
/// so the curve and its chord wind alike round any point in the box. None when a curve's box reaches inside it.
fn edges(segments: &[Segment], bbox: Rect) -> Option<Vec<(Point, Point)>> {
    let mut out = Vec::new();
    let (mut start, mut at): (Option<Point>, Option<Point>) = (None, None);
    for segment in segments {
        match *segment {
            Segment::MoveTo(p) => {
                out.extend(at.zip(start));
                (start, at) = (Some(p), Some(p));
            }
            Segment::LineTo(p) => {
                out.extend(at.map(|a| (a, p)));
                start = start.or(Some(p));
                at = Some(p);
            }
            Segment::CurveTo(b, c, d) => {
                let a = at.unwrap_or(b);
                if overlaps(Rect::around(&[a, b, c, d])?, bbox) {
                    return None;
                }
                out.push((a, d));
                start = start.or(Some(a));
                at = Some(d);
            }
            Segment::Close => {
                out.extend(at.zip(start));
                at = start;
            }
        }
    }
    out.extend(at.zip(start));
    Some(out)
}

/// Whether the segment from `p` to `q` reaches inside `bbox` (along an edge of it does not): unless an axis or the
/// segment's normal parts them -- every corner of the box on one side of its line, or on it. A corner no exact sign
/// places counts as on both sides.
fn crosses(p: Point, q: Point, bbox: Rect) -> bool {
    if !overlaps(Rect::spanning(p, q), bbox) {
        return false;
    }
    let corners = [
        Point {
            x: bbox.x0,
            y: bbox.y0,
        },
        Point {
            x: bbox.x1,
            y: bbox.y0,
        },
        Point {
            x: bbox.x1,
            y: bbox.y1,
        },
        Point {
            x: bbox.x0,
            y: bbox.y1,
        },
    ]
    .map(|c| orientation(p, q, c));
    let none_on = |side: Ordering| corners.iter().all(|c| c.is_some_and(|c| c != side));
    !(none_on(Ordering::Less) || none_on(Ordering::Greater))
}

/// The side of the line from `a` to `b` that `c` lies on -- the sign of their cross product -- when the floating
/// point product settles it: within Shewchuk's first error bound of zero (Adaptive Precision Floating-Point
/// Arithmetic and Fast Robust Geometric Predicates, 1997: `ccwerrboundA`) rounding could flip it, and None says so.
fn orientation(a: Point, b: Point, c: Point) -> Option<Ordering> {
    const HALF_ULP: f64 = f64::EPSILON / 2.0;
    const BOUND: f64 = (3.0 + 16.0 * HALF_ULP) * HALF_ULP;
    let left = (b.x - a.x) * (c.y - a.y);
    let right = (b.y - a.y) * (c.x - a.x);
    let det = left - right;
    let bound = BOUND * (left.abs() + right.abs());
    (det.abs() > bound || bound <= 0.0)
        .then(|| det.partial_cmp(&0.0))
        .flatten()
}

/// Whether `clip`, and every clip it narrows, [covers](covers) `bbox` whole: a clip of glyphs covers nothing, their
/// outlines not read.
fn clip_covers(clip: &Clip, bbox: Rect) -> bool {
    let here = match &clip.outline {
        Outline::Path { segments, even_odd } => covers(segments, *even_odd, bbox),
        Outline::Glyphs(_) => false,
    };
    here && clip.within.as_deref().is_none_or(|c| clip_covers(c, bbox))
}

/// Each span's [`Span::unspaced`] number: the lowest index among the spans linked to it by continuing one
/// another's words; None when it is linked to none.
fn unspaced(spans: &[Open]) -> Vec<Option<usize>> {
    fn root(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    let mut parent: Vec<usize> = (0..spans.len()).collect();
    let mut linked = vec![false; spans.len()];
    for (i, next) in spans.iter().enumerate() {
        for (j, before) in spans.iter().enumerate() {
            if i != j && continues_word(before, next) {
                let (a, b) = (root(&mut parent, i), root(&mut parent, j));
                parent[a.max(b)] = a.min(b);
                linked[i] = true;
                linked[j] = true;
            }
        }
    }
    (0..spans.len())
        .map(|i| linked[i].then(|| root(&mut parent, i)))
        .collect()
}

/// Each accent glyph set on a letter merged into it ([`sits_on`]): the letter's text becomes the two composed
/// ([`accents::compose`]), its box the union of theirs; the accent is gone. An accent tries the nearest letter
/// drawn after it (TeX draws the accent first), then the nearest drawn before it, other accents passed over; the
/// last accents are taken first, so of two stacked on one letter the inner composes first (`\'{\"u}` is `ǘ`).
fn compose_accents(glyphs: &mut Vec<Placed>) {
    let marks: Vec<Option<char>> = glyphs.iter().map(|g| accents::mark(&g.text)).collect();
    if marks.iter().all(Option::is_none) {
        return;
    }
    let mut gone = vec![false; glyphs.len()];
    for i in (0..glyphs.len()).rev() {
        let Some(mark) = marks[i] else { continue };
        let letter = |j: &usize| !gone[*j] && marks[*j].is_none();
        let after = (i + 1..glyphs.len()).find(letter);
        let before = (0..i).rev().find(letter);
        for j in [after, before].into_iter().flatten() {
            if !sits_on(&glyphs[i], &glyphs[j]) {
                continue;
            }
            if let Some(text) = accents::compose(&glyphs[j].text, mark) {
                glyphs[j].text = Cow::Owned(text);
                glyphs[j].bbox = glyphs[j].bbox.union(glyphs[i].bbox);
                gone[i] = true;
                break;
            }
        }
    }
    let mut gone = gone.into_iter();
    glyphs.retain(|_| !gone.next().unwrap_or(false));
}

/// Whether `accent` is set on `letter`: in its layer, the middle of its advance within the letter's, its baseline
/// the letter's -- raised by no more than [`RISE`] sizes, as TeX raises an accent over a capital, or lowered by no
/// more than [`BASELINE`] -- along the letter's direction. (An accent glyph is drawn above or below the baseline it
/// stands on: the glyph boxes, from the font's descent to its ascent, do not tell where its ink is.)
fn sits_on(accent: &Placed, letter: &Placed) -> bool {
    let (dx, dy) = (
        letter.end.x - letter.origin.x,
        letter.end.y - letter.origin.y,
    );
    let advance = dx.hypot(dy);
    if advance <= 0.0 || accent.event.layer != letter.event.layer {
        return false;
    }
    let (ux, uy) = (dx / advance, dy / advance);
    let middle = Point {
        x: f64::midpoint(accent.origin.x, accent.end.x) - letter.origin.x,
        y: f64::midpoint(accent.origin.y, accent.end.y) - letter.origin.y,
    };
    let (ox, oy) = (
        accent.origin.x - letter.origin.x,
        accent.origin.y - letter.origin.y,
    );
    let size = letter.event.size;
    let rise = ox * uy - oy * ux; // up from the letter's baseline (page space grows down)
    (0.0..=advance).contains(&(middle.x * ux + middle.y * uy))
        && (-BASELINE * size..=RISE * size).contains(&rise)
}

/// Whether the glyph at `origin` continues the run `open` ends, whatever its fill: None when it starts another run,
/// else whether a space lies before it. It joins `open` when its fill is `open`'s too.
fn join(open: &Open, text: &TextEvent, origin: Point, direction: (f64, f64)) -> Option<bool> {
    let size = open.span.size.max(f64::MIN_POSITIVE);
    if open.span.font != text.font
        || (open.span.size - text.size).abs() > 0.01 * size
        || open.span.render_mode != text.render_mode
        || open.span.layer != text.layer
        || open.span.vertical != text.vertical
        || (open.span.direction.0 - direction.0).abs() + (open.span.direction.1 - direction.1).abs()
            > 0.01
    {
        return None;
    }
    let (dx, dy) = (origin.x - open.end.x, origin.y - open.end.y);
    let along = dx * direction.0 + dy * direction.1;
    let across = -dx * direction.1 + dy * direction.0;
    if across.abs() > BASELINE * size || along > BREAK * size || along < -0.5 * size {
        return None;
    }
    Some(along > SPACE * size)
}

/// Whether a glyph whose text is `piece` makes its span's box: it shows ink -- its text is not white-space, or is
/// empty because an `/ActualText` before it in its sequence carries its text.
fn inks(piece: &str) -> bool {
    piece.is_empty() || !piece.trim().is_empty()
}

/// Whether `next` continues the word `before` ends in: no space glyph between them, both written the same way (one
/// writing mode, one direction) in one layer, `next` starting on from `before`'s end by no more than a word gap
/// (back by no more than half a size), raised or lowered by no more than [`RISE`] sizes.
fn continues_word(before: &Open, next: &Open) -> bool {
    if before.span.vertical != next.span.vertical
        || before.span.text.ends_with(char::is_whitespace)
        || next.span.text.starts_with(char::is_whitespace)
        || before.span.layer != next.span.layer
    {
        return false;
    }
    let size = before.span.size.max(next.span.size).max(f64::MIN_POSITIVE);
    let (dx, dy) = before.span.direction;
    let along = |from: Point| (next.origin.x - from.x) * dx + (next.origin.y - from.y) * dy;
    let across = -(next.origin.x - before.end.x) * dy + (next.origin.y - before.end.y) * dx;
    (dx - next.span.direction.0).abs() + (dy - next.span.direction.1).abs() <= 0.01
        && (-0.5 * size..=SPACE * size).contains(&along(before.end))
        && across.abs() <= RISE * size
}
